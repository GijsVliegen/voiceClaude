# voiceClaude

Talk to [Claude Code](https://claude.com/claude-code) hands-free, like a voice assistant.

- **Speaking to Claude:** you say a keyword, and Claude Code's built-in voice input (hold Space) starts. Another keyword sends the message.
- **Claude speaking back:** a hook reads the `<speak>` part of each reply aloud with [Piper](https://github.com/rhasspy/piper). A skill tells Claude to put only the conclusion there.

The keywords are small [openWakeWord](https://github.com/dscripka/openWakeWord) models that you train yourself, on synthetic voices and your own recordings. Everything except Claude's voice input runs locally, and the microphone is only open while `voiceclaud` runs.

Built and tested on Ubuntu (Wayland, GNOME Terminal). Other setups will need changes.

## Keywords

| Keyword | Action |
|---|---|
| `start` | start speaking (holds Space down for Claude's voice input) |
| `stop` | send: releases Space and presses Enter (only after ~1 s of silence before it) |
| `refresh` ×3 | new session: types `/clear` and presses Enter |

You pick the actual word for each keyword yourself: copy `keywords.example.json` to `keywords.json` and replace the placeholders. Invented words work best (nobody says them by accident), and so do near-misses that sound like your word.

Guardrails against false triggers: a minimum loudness, nothing is detected while Claude is speaking, the stop word needs silence before it, and the refresh word must be said three times with nothing loud in between.

## Layout

```
voiceclaud                launcher: starts Claude in tmux and the keyword listener
voice_keys.py             maps keywords to keys sent into the tmux session
listen.py                 microphone -> keyword models -> detections
keywords.example.json     template for keywords.json: your words, near-misses and hard cases
training/
  record_clips.py         record your own voice (keywords + normal talking)
  train_word.py           train one keyword model -> models/<word>.onnx
  train_all.sh            train every keyword in keywords.json
  score_clips.py          test the models on recordings that were not used for training
hooks/                    Claude Code hooks: speak replies, voice reminder, remember session
skills/voice-output/      the skill that tells Claude how to write the spoken part
settings.example.json     the hooks block for ~/.claude/settings.json
```

## Setup

1. **System tools:** `tmux`, `ydotool`, `aplay` (alsa-utils), `gnome-terminal`.
2. **Python environment** in the repo folder:
   ```sh
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   .venv/bin/pip install --no-deps openwakeword
   ```
3. **Piper voices** for training: put a few voices (`.onnx` + `.onnx.json`) from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) in `voices/`. Multi-speaker voices (`en_US-libritts_r-medium`, `en_GB-vctk-medium`, `en_US-l2arctic-medium`) give the most variety.
4. **Piper for speaking replies:** install the `piper` command and one voice; set `VOICECLAUDE_PIPER` and `VOICECLAUDE_VOICE` if they are not at the default paths in `hooks/speak_reply.py`.
5. **Hooks and skill:** copy the hooks block from `settings.example.json` into `~/.claude/settings.json` (with your real path), and copy `skills/voice-output` to `~/.claude/skills/`. The reminder hook only acts when `"voice": {"enabled": true}` is set in your settings.

## Train your keywords

```sh
cp keywords.example.json keywords.json                      # then fill in your three words
.venv/bin/python training/record_clips.py                  # 20 clips per keyword + 5 min of normal talking
.venv/bin/python training/record_clips.py --set test       # a separate set, only for testing
training/train_all.sh                                       # ~20 min per keyword on a laptop CPU
.venv/bin/python training/score_clips.py                    # hit rate and false triggers on the test set
```

Training without your own recordings also works (`train_word.py start`), but recordings of your own voice make a big difference.

## Run

```sh
./voiceclaud
```

It opens Claude in a terminal (resuming the last voiceclaud session) and listens until Ctrl+C.

## Credits

The keyword training approach (`training/train_word.py`) is based on `bin/wake-train` from [claude-talk](https://github.com/jjacke13/claude-talk) by jjacke13.

## License

Copyright (C) 2026 Gijs Vliegen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).
