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
"""Record your own voice for testing (and later training) the wake-word models.

  record_clips.py                      # 20 clips of each keyword (start, stop, refresh), then 5 min of talking
  record_clips.py stop --count 30      # only the stop word, 30 clips
  record_clips.py talk --minutes 3     # only normal talking

Word clips: 3.5 s each. The first second is room sound ("ready…"), then say the word once when
">>> SAY" appears. Saved as recordings/<word>/NNN.wav (16 kHz mono); numbering continues
across runs, so you can stop with Ctrl+C and carry on later.
Goes to recordings/train/ (for training) unless --set test (kept out of training, for score_clips.py).
The word to say is the first spelling in keywords.json.
Talking: speak normally WITHOUT the keywords (read something aloud, explain your work, …).
Saved as recordings/talk/<timestamp>.wav.

Stop voice_keys.py first, or it will press keys when it hears you.
"""
import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

RATE = 16000
OUT = Path(__file__).resolve().parent.parent / "recordings"
KEYWORDS = Path(__file__).resolve().parent.parent / "keywords.json"
WORDS = ["start", "stop", "refresh"]


def spoken(word: str) -> str:
    """What to say for a keyword: its first spelling in keywords.json."""
    try:
        return json.loads(KEYWORDS.read_text())[word]["spellings"][0]
    except (OSError, ValueError, KeyError, IndexError):
        return f"your {word} word"


def save(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(x.astype(np.int16).tobytes())


def record_word(word: str, count: int) -> None:
    folder = OUT / word
    start = len(list(folder.glob("*.wav"))) if folder.exists() else 0
    print(f"\n=== {word.upper()} ({spoken(word)}): {count} clips (Ctrl+C to stop) ===")
    input("Press Enter to start…")
    for i in range(start, start + count):
        with sd.InputStream(samplerate=RATE, channels=1, dtype="int16") as mic:
            print(f"  {i - start + 1}/{count}  ready…", end="", flush=True)
            lead, _ = mic.read(RATE)  # 1 s room sound
            print(f"   >>> SAY {spoken(word).upper()}", flush=True)
            rest, _ = mic.read(int(2.5 * RATE))
        x = np.concatenate([lead[:, 0], rest[:, 0]])
        peak = np.abs(x).max() / 32768
        save(folder / f"{i:03d}.wav", x)
        if peak < 0.05:
            print(f"      (very quiet: peak {peak:.3f})")
        time.sleep(0.6)


def record_talk(minutes: float) -> None:
    print(f"\n=== TALKING: {minutes:g} min. Speak normally, do NOT say the keywords ===")
    input("Press Enter to start…")
    chunks = []
    end = time.monotonic() + minutes * 60
    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype="int16") as mic:
            while time.monotonic() < end:
                chunks.append(mic.read(RATE)[0][:, 0])
                if len(chunks) % 30 == 0:
                    print(f"  {len(chunks) // 60}:{len(chunks) % 60:02d} recorded", flush=True)
    except KeyboardInterrupt:
        print("  stopped early")
    if chunks:
        path = OUT / "talk" / f"{time.strftime('%Y%m%d-%H%M%S')}.wav"
        save(path, np.concatenate(chunks))
        print(f"  saved {path} ({len(chunks)} s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=WORDS + ["talk"], help="start, stop, refresh and/or talk")
    ap.add_argument("--count", type=int, default=20, help="clips per word")
    ap.add_argument("--minutes", type=float, default=5, help="minutes of talking")
    ap.add_argument("--set", choices=["train", "test"], default="train",
                    help="train = used by train_word.py --own; test = only for score_clips.py")
    args = ap.parse_args()
    global OUT
    OUT = OUT / args.set
    try:
        for w in args.what:
            record_talk(args.minutes) if w == "talk" else record_word(w, args.count)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
