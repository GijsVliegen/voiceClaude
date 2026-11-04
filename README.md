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
voiceclaud                launcher: runs voice_keys.py
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

Plan about an hour and a half: most of it is training, which runs on its own.

### 1. Requirements

- Linux with `tmux`, `aplay` (alsa-utils) and `gnome-terminal`, and Python 3.10 or newer.
- Claude Code, logged in, with voice dictation working: run `/voice` once in Claude and check that holding Space records you.

### 2. Get the code and install

```sh
git clone https://github.com/GijsVliegen/voiceClaude.git
cd voiceClaude
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps openwakeword        # --no-deps: skips tflite-runtime, which is not needed
.venv/bin/python -c "import openwakeword.utils as u; u.download_models()"   # its feature models
```

### 3. Download voices

Piper voices go in `voices/`. They are used to train the keywords (many synthetic speakers) and to speak Claude's replies (`en_US-lessac-medium`):

```sh
.venv/bin/python -m piper.download_voices --data-dir voices \
    en_US-lessac-medium en_US-libritts_r-medium en_GB-vctk-medium en_US-l2arctic-medium en_US-arctic-medium
```

### 4. Pick your keywords

```sh
cp keywords.example.json keywords.json
```

Replace `YOUR_START_WORD`, `YOUR_STOP_WORD` and `YOUR_REFRESH_WORD` with your own words, in a few spellings (with `!` and `?`, and alternative spellings that sound the same). Invented words that sound like nothing else work best. Under `near_misses`, add real words that sound similar to yours: training uses them as "this is not it". Words that keep falsely triggering can get extra weight under `hard`.

### 5. Record your voice

```sh
.venv/bin/python training/record_clips.py                  # 20 clips of each keyword + 5 min of normal talking
.venv/bin/python training/record_clips.py --set test       # a second set, only for testing
```

Say the word when `>>> SAY` appears. For the talking part, read something aloud or explain your work, without the keywords.

### 6. Train

```sh
training/train_all.sh                                       # ~20 min per keyword on a laptop CPU
.venv/bin/python training/score_clips.py                    # hit rate and false triggers on the test set
```

Training first records 15 s of room noise: stay quiet when it says so. Logs are in `.cache/train-<keyword>.log`. If a keyword misses too often or triggers during normal talking, add near-misses, record more clips, and train again.

### 7. Connect Claude Code

- Copy the hooks block from `settings.example.json` into `~/.claude/settings.json`, with `/path/to/voiceClaude` replaced by the real path.
- Copy the skill: `cp -r skills/voice-output ~/.claude/skills/`.
- The reminder hook only tells Claude to use the skill while Claude Code's voice dictation is on (`/voice`, step 1).

To silence the spoken replies for a while, create `~/.claude/speak_off`; remove it to turn them back on.

## Run

```sh
./voiceclaud
```

It opens Claude in a new terminal (inside tmux, resuming the last voiceclaud session) and listens until Ctrl+C. Say your start word, speak, and say your stop word to send.

To start it from anywhere: `ln -s "$PWD/voiceclaud" ~/.local/bin/voiceclaud`.

Detection can be tuned with environment variables (see the top of `voice_keys.py`), for example `VOICE_KEYS_THRESHOLD` (default 0.8) and `VOICE_KEYS_MIN_RMS` (default 0.04) when your microphone is quieter or louder.

## Credits

The keyword training approach (`training/train_word.py`) is based on `bin/wake-train` from [claude-talk](https://github.com/jjacke13/claude-talk) by jjacke13.

## License

Copyright (C) 2026 Gijs Vliegen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).
