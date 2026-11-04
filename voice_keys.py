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
"""Hands-free Enter for Claude Code on Wayland.

On start, opens a terminal running `claude`. Then listens to the microphone:
  - start word           -> holds Space down (Claude's built-in voice input)
  - stop word            -> releases Space, then presses Enter (sends what was said)
                            (only after ~1 s of silence right before it)
  - refresh word ×3      -> releases Space if held, clears the input line, types /clear, presses Enter

Detection runs listen.py (openwakeword). Each keyword is only active when its
models/<start|stop|refresh>.onnx exists (train it with training/train_word.py <start|stop|refresh>).
Words spoken more quietly than VOICE_KEYS_MIN_RMS are ignored
(listen.py prints "too quiet: …" / "level …" on stderr for tuning).
Claude runs inside tmux session "voiceclaud"; keys are sent into that session with
tmux send-keys, so they reach Claude even when its window is not focused, and never
reach any other window. Holding Space = a stream of spaces (what a held key sends).

Run:  ./voiceclaud (or .venv/bin/python voice_keys.py)
Stop: Ctrl+C
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.environ.get("VOICECLAUDE_PYTHON", os.path.join(ROOT, ".venv", "bin", "python"))
WAKE_LISTEN = os.path.join(ROOT, "listen.py")
WAKE_MODEL = os.path.join(ROOT, "models", "start.onnx")
WAKE_KEY = "start"  # listen.py prints the model's basename
ENTER_MODEL = os.path.join(ROOT, "models", "stop.onnx")
CLEAR_MODEL = os.path.join(ROOT, "models", "refresh.onnx")

THRESHOLD = float(os.environ.get("VOICE_KEYS_THRESHOLD", "0.8"))
# Loudest 80 ms of the word must reach this (mic floor here is ~0.001-0.003, peaks ~0.009)
MIN_RMS = float(os.environ.get("VOICE_KEYS_MIN_RMS", "0.04"))
# The stop word only counts after this much silence (frame RMS below QUIET_RMS) right before it
ENTER_PRE_SILENCE_S = float(os.environ.get("VOICE_KEYS_STOP_SILENCE_S", "1.0"))
QUIET_RMS = float(os.environ.get("VOICE_KEYS_QUIET_RMS", "0.02"))
# The refresh word must be heard this many separate times within 6 s (leave a tiny pause between them)
CLEAR_REPEAT = int(os.environ.get("VOICE_KEYS_REFRESH_REPEAT", "3"))
# Any sound at least this loud between two refresh words restarts the count
CLEAR_RESET_RMS = float(os.environ.get("VOICE_KEYS_REFRESH_RESET_RMS", "0.03"))
# Written by the SessionStart hook ~/.claude/hooks/voiceclaud_session.py
LAST_SESSION = os.path.join(ROOT, "last_session.json")
TMUX_SESSION = "voiceclaud"  # keys go only into this tmux session, focused or not
SPACE_REPEAT_S = 0.03  # "holding" Space = a stream of spaces, like a held key's auto-repeat
RELEASE_PAUSE_S = 0.3  # gap between releasing Space and pressing Enter
holding = threading.Event()  # set while Space is "held"
# Words are ignored while Claude's reply is playing (aplay, from the speak hook) and this long after
SPEECH_TAIL_S = float(os.environ.get("VOICE_KEYS_SPEECH_TAIL_S", "1.0"))
last_speech = 0.0  # monotonic time aplay was last seen running


def watch_speech() -> None:
    global last_speech
    while True:
        if subprocess.run(["pgrep", "-x", "aplay"], capture_output=True).returncode == 0:
            last_speech = time.monotonic()
        time.sleep(0.2)


def claude_speaking() -> bool:
    return time.monotonic() - last_speech < SPEECH_TAIL_S


def send_keys(*keys: str, literal: bool = False) -> None:
    """Send keys straight into the Claude tmux session (not to the focused window)."""
    result = subprocess.run(
        ["tmux", "send-keys", "-t", TMUX_SESSION, *(["-l"] if literal else []), *keys],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"tmux send-keys failed: {result.stderr.strip()}", flush=True)


def press_enter() -> None:
    send_keys("Enter")


def hold_space() -> None:
    """While `holding` is set, keep sending spaces: Claude sees a held Space key."""
    while True:
        holding.wait()
        send_keys("Space")
        time.sleep(SPACE_REPEAT_S)


def terminal_cmd() -> list[str]:
    """Resume the last voiceclaud session if there is one, else start a new one, inside tmux.
    -A: if the tmux session still runs (terminal closed earlier), just show it again."""
    claude = ["env", "VOICECLAUD=1", "claude"]
    tmux = ["tmux", "new-session", "-A", "-s", TMUX_SESSION]
    try:
        with open(LAST_SESSION) as f:
            last = json.load(f)
    except (OSError, ValueError):
        return ["gnome-terminal", "--", *tmux, *claude]
    transcript = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{last['session_id']}.jsonl"))
    if not transcript or not os.path.isdir(last["cwd"]):
        print(f"session {last['session_id']} no longer exists: starting a new one", flush=True)
        return ["gnome-terminal", "--", *tmux, *claude]
    print(f"resuming session {last['session_id']}", flush=True)
    return ["gnome-terminal", "--", *tmux, "-c", last["cwd"],
            *claude, "--resume", last["session_id"]]


def main() -> None:
    subprocess.Popen(terminal_cmd())

    models = [WAKE_MODEL]
    if os.path.exists(ENTER_MODEL):
        models.append(ENTER_MODEL)
    else:
        print("no models/stop.onnx yet: only the start word is active", flush=True)
    if os.path.exists(CLEAR_MODEL):
        models.append(CLEAR_MODEL)
    else:
        print("no models/refresh.onnx yet: refresh word inactive", flush=True)

    cmd = [VENV_PY, WAKE_LISTEN, "--min-rms", str(MIN_RMS), "--quiet-rms", str(QUIET_RMS),
           "--reset-rms", str(CLEAR_RESET_RMS)]
    for model in models:
        cmd += ["--model", model, "--threshold", str(THRESHOLD),
                "--pre-silence", str(ENTER_PRE_SILENCE_S if model == ENTER_MODEL else 0),
                "--repeat", str(CLEAR_REPEAT if model == CLEAR_MODEL else 1)]

    listener = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    threading.Thread(target=watch_speech, daemon=True).start()
    threading.Thread(target=hold_space, daemon=True).start()
    talking = False
    try:
        for line in listener.stdout:
            parts = line.split()
            if len(parts) != 2:
                if parts == ["ready"]:
                    print("listening", flush=True)
                continue
            word, score = parts[0], parts[1]
            if claude_speaking():
                print(f"ignored '{word}' ({score}): Claude is speaking", flush=True)
                continue
            if word == WAKE_KEY and not talking:
                print(f"heard start ({score}) -> hold Space", flush=True)
                holding.set()
                talking = True
            elif word == "stop":
                print(f"heard stop ({score}) -> release Space, Enter", flush=True)
                if talking:
                    holding.clear()
                    talking = False
                    time.sleep(RELEASE_PAUSE_S)
                press_enter()
            elif word == "refresh" and not talking:
                print(f"heard refresh ({score}) -> /clear", flush=True)
                if talking:
                    holding.clear()
                    talking = False
                    time.sleep(RELEASE_PAUSE_S)
                send_keys("C-u")  # clear the input line
                send_keys("/clear", literal=True)
                time.sleep(RELEASE_PAUSE_S)
                press_enter()
    except KeyboardInterrupt:
        pass
    finally:
        if talking:
            holding.clear()
        listener.terminate()
        listener.wait()


if __name__ == "__main__":
    if not os.path.exists(VENV_PY):
        sys.exit(f"missing venv python: {VENV_PY}")
    main()
