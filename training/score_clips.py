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
"""Score the wake-word models on your own recordings (made with record_clips.py).

  score_clips.py                  # threshold 0.8, models from the talk plugin
  score_clips.py --threshold 0.7

Word clips: per word, how often its model reaches the threshold (hit rate) and the score spread;
also whether the OTHER models fire on it (confusion).
Talking: false triggers per model (score >= threshold, 2 s refractory like listen.py), per hour,
with the time in the file so you can listen back.
Model scores only: the loudness, pre-silence and repeat rules of voice_keys.py come on top.
"""
import argparse
import wave
from pathlib import Path

import numpy as np
from openwakeword.model import Model

RATE = 16000
CHUNK = 1280  # 80 ms
ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "recordings" / "test"  # never used for training
MODELS = ROOT / "models"
WORDS = ["start", "stop", "refresh"]


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def stream(model: Model, x: np.ndarray):
    """Scores per 80 ms frame, fed exactly like the mic."""
    model.reset()
    for i in range(0, len(x) - CHUNK + 1, CHUNK):
        yield model.predict(x[i:i + CHUNK])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()
    th = args.threshold
    model = Model(wakeword_models=[str(MODELS / f"{w}.onnx") for w in WORDS], inference_framework="onnx")

    print(f"threshold {th}\n\nWORD CLIPS")
    for word in WORDS:
        clips = sorted((REC / word).glob("*.wav"))
        if not clips:
            print(f"  {word}: no recordings")
            continue
        best = {w: [] for w in WORDS}
        for clip in clips:
            x = load(clip)
            x = np.concatenate([x, x[:RATE]])  # 1 s room tail so the score can rise after the word
            top = dict.fromkeys(WORDS, 0.0)
            for s in stream(model, x):
                for w in WORDS:
                    top[w] = max(top[w], float(s[w]))
            for w in WORDS:
                best[w].append(top[w])
        own = np.array(best[word])
        missed = [c.name for c, s in zip(clips, own) if s < th]
        print(f"  {word}: hit {np.sum(own >= th)}/{len(own)}  "
              f"score min {own.min():.2f}  median {np.median(own):.2f}")
        if missed:
            print(f"      missed: {', '.join(missed)}")
        for w in WORDS:
            if w != word and (n := sum(s >= th for s in best[w])):
                print(f"      also fired '{w}' on {n} clip(s)")

    print("\nTALKING (should trigger nothing)")
    talks = sorted((REC / "talk").glob("*.wav"))
    if not talks:
        print("  no recordings")
    for talk in talks:
        x = load(talk)
        minutes = len(x) / RATE / 60
        hits = {w: [] for w in WORDS}
        last = dict.fromkeys(WORDS, -99.0)
        top = dict.fromkeys(WORDS, 0.0)
        for i, s in enumerate(stream(model, x)):
            t = i * CHUNK / RATE
            for w in WORDS:
                top[w] = max(top[w], float(s[w]))
                if s[w] >= th and t - last[w] >= 2.0:
                    hits[w].append(t)
                    last[w] = t
        print(f"  {talk.name} ({minutes:.1f} min)")
        for w in WORDS:
            times = ", ".join(f"{int(t // 60)}:{int(t % 60):02d}" for t in hits[w])
            print(f"      {w}: {len(hits[w])} false trigger(s), {len(hits[w]) / minutes * 60:.0f}/hour, "
                  f"max score {top[w]:.2f}" + (f"  at {times}" if times else ""))


if __name__ == "__main__":
    main()
