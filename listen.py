#!/usr/bin/env python3
# This file is part of voiceClaude.
# Copyright (C) 2026 Gijs Vliegen
#
# voiceClaude is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# voiceClaude is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with voiceClaude.
# If not, see <https://www.gnu.org/licenses/>.
"""Keyword listener: microphone -> openWakeWord models -> one stdout line per detection.

  listen.py --model models/start.onnx --threshold 0.8 [--model ... --threshold ...] [options]

Prints "ready" once the models are loaded, then "<name> <score>" for every detection, where <name>
is the model's file name without .onnx. Diagnostics (why a hearing was dropped) go to stderr.

Per-model options (the i-th value belongs to the i-th --model):
  --threshold    score needed (default: the first threshold given, or 0.5)
  --pre-silence  seconds of quiet required right before the word (0 = off)
  --repeat       how many separate hearings within 6 s are needed before it fires (1 = off)
Global options:
  --min-rms      the loudest 80 ms frame of the word must reach this level (0 = off)
  --quiet-rms    frames below this level count as quiet for --pre-silence
  --reset-rms    a sound this loud between two hearings restarts a --repeat count (0 = off)
"""
import argparse
import ctypes
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

RATE = 16000
FRAME = 1280              # 80 ms: the chunk size openWakeWord works on
FRAME_S = FRAME / RATE
HISTORY = 75              # frames of loudness kept (6 s)
WORD_FRAMES = 19          # ~1.5 s: the frames that hold the word that was just detected
REFRACTORY_S = 2.0        # one utterance = one detection
REPEAT_WINDOW_S = 6.0
HEARING_FLOOR = 0.5       # for --repeat: a hearing lasts while the score stays at or above this
SPEECH_TAIL_S = 1.0       # repeat hearings are ignored while Claude speaks (aplay runs) and this long after


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class SpeechWatcher(threading.Thread):
    """Notes when aplay (the speak hook's player) was last running."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.last = 0.0

    def run(self) -> None:
        while True:
            if subprocess.run(["pgrep", "-x", "aplay"], capture_output=True).returncode == 0:
                self.last = time.monotonic()
            time.sleep(0.2)

    def speaking(self) -> bool:
        return time.monotonic() - self.last < SPEECH_TAIL_S


def quiet_before(levels: list[float], seconds: float, quiet: float) -> bool:
    """True when the word that just ended was preceded by >= `seconds` of quiet frames.
    Walks back from the newest frame: the quiet tail (detection lag), then the word itself (short dips
    of <= 3 frames belong to it; longer than 1.2 s = continuous speech), then the quiet run before it."""
    need = max(1, round(seconds / FRAME_S))
    i = len(levels) - 1
    while i >= 0 and levels[i] < quiet and len(levels) - 1 - i < 8:
        i -= 1
    word = run = 0
    while i >= 0:
        if levels[i] < quiet:
            run += 1
            if run >= need:
                return True
        else:
            if run > 3:
                return False  # a short pause, then more speech before the word
            word += run + 1
            run = 0
            if word > 15:
                return False
        i -= 1
    return False


def word_start(levels: list[float], loud: float) -> int:
    """Index of the first frame of the word that just ended. Walks back from the newest frame: the quiet
    tail (detection lag), then the word itself (dips of <= 3 quiet frames belong to it)."""
    i = len(levels) - 1
    while i >= 0 and levels[i] < loud and len(levels) - 1 - i < 8:
        i -= 1
    run = 0
    while i >= 0:
        if levels[i] < loud:
            run += 1
            if run > 3:
                return i + run
        else:
            run = 0
        i -= 1
    return 0


class Word:
    """The detection state of one keyword model."""

    def __init__(self, name: str, threshold: float, pre_silence: float, repeat: int) -> None:
        self.name, self.threshold, self.pre_silence, self.repeat = name, threshold, pre_silence, repeat
        self.last_fired = -REFRACTORY_S
        # --repeat bookkeeping
        self.in_hearing = False     # score currently >= HEARING_FLOOR
        self.reached = False        # ... and this hearing reached the threshold
        self.frames_below = 0
        self.hearings: deque[float] = deque()  # times at which counted hearings ended
        self.last_hearing_frame = 0

    def hearing_ended(self, score: float) -> bool:
        """Feed one score; True on the frame where a hearing that reached the threshold has just ended."""
        if score >= HEARING_FLOOR:
            self.in_hearing, self.frames_below = True, 0
            self.reached |= score >= self.threshold
            return False
        self.frames_below += 1
        if self.frames_below < 2:
            return False  # a one-frame dip is still the same word
        ended = self.in_hearing and self.reached
        self.in_hearing = self.reached = False
        return ended


class Listener:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        n = len(args.model)
        thresholds = args.threshold or [0.5]
        thresholds += [thresholds[0]] * (n - len(thresholds))
        silences = (args.pre_silence or []) + [0.0] * n
        repeats = (args.repeat or []) + [1] * n
        self.model = Model(wakeword_models=args.model, inference_framework="onnx")
        names = [Path(m).stem for m in args.model]
        self.words = [Word(names[i], thresholds[i], silences[i], repeats[i]) for i in range(n)]
        self.levels: deque[float] = deque(maxlen=HISTORY)
        self.frame_no = 0
        self.speech = SpeechWatcher()

    def word_level(self) -> float:
        return max(list(self.levels)[-WORD_FRAMES:])

    def fire(self, word: Word, score: float, now: float) -> None:
        if now - word.last_fired < REFRACTORY_S:
            return
        level = self.word_level()
        if level < self.args.min_rms:
            log(f"too quiet: {word.name} {score:.3f} level {level:.4f} < {self.args.min_rms}")
            return
        word.last_fired = now
        if word.pre_silence and not quiet_before(list(self.levels), word.pre_silence, self.args.quiet_rms):
            log(f"no silence before: {word.name} {score:.3f}")
            return
        if self.args.min_rms:
            log(f"level {level:.4f}")
        print(f"{word.name} {score:.3f}", flush=True)

    def count_hearing(self, word: Word, now: float) -> bool:
        """A --repeat word was heard once more; True when that completes the count."""
        level = self.word_level()
        if level < self.args.min_rms:
            log(f"too quiet: {word.name} hearing, level {level:.4f} < {self.args.min_rms}")
            return False
        if self.speech.speaking():
            log(f"ignored: {word.name} hearing while Claude is speaking")
            return False
        if self.args.reset_rms and word.hearings:
            levels = list(self.levels)
            since_last = len(levels) - (self.frame_no - word.last_hearing_frame) + 2
            gap = levels[max(0, since_last):word_start(levels, self.args.reset_rms)]
            if gap and max(gap) >= self.args.reset_rms:
                log(f"reset: {word.name} count, sound in between (level {max(gap):.4f})")
                word.hearings.clear()
        word.last_hearing_frame = self.frame_no
        word.hearings.append(now)
        while now - word.hearings[0] > REPEAT_WINDOW_S:
            word.hearings.popleft()
        log(f"heard {word.name} {len(word.hearings)}/{word.repeat}")
        if len(word.hearings) < word.repeat:
            return False
        word.hearings.clear()
        return True

    def step(self, pcm: np.ndarray) -> None:
        x = pcm.astype(np.float32) / 32768
        self.levels.append(float(np.std(x)))  # RMS with the DC offset removed
        self.frame_no += 1
        scores = list(self.model.predict(pcm).values())
        now = time.monotonic()
        for word, score in zip(self.words, scores):
            score = float(score)
            if word.repeat > 1:
                if word.hearing_ended(score) and self.count_hearing(word, now):
                    self.fire(word, word.threshold, now)
            elif score >= word.threshold:
                self.fire(word, score, now)

    def run(self) -> None:
        self.speech.start()
        print("ready", flush=True)
        with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME) as mic:
            while True:
                frames, _ = mic.read(FRAME)
                self.step(frames[:, 0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True, help="path to a keyword .onnx (repeatable)")
    ap.add_argument("--threshold", action="append", type=float)
    ap.add_argument("--pre-silence", action="append", type=float)
    ap.add_argument("--repeat", action="append", type=int)
    ap.add_argument("--min-rms", type=float, default=0.0)
    ap.add_argument("--quiet-rms", type=float, default=0.02)
    ap.add_argument("--reset-rms", type=float, default=0.0)
    args = ap.parse_args()
    listener = Listener(args)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if sys.platform.startswith("linux"):  # die with the parent, so the mic is never left open
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    listener.run()


if __name__ == "__main__":
    main()
