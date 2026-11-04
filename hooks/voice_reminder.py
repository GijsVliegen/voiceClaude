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
"""UserPromptSubmit hook: remind Claude about the voice-output skill once per session.

Only acts when voice mode is enabled in ~/.claude/settings.json. Keeps a marker
file per session so the reminder is added once; voice_reset.py clears it after
compaction or /clear.
"""
import json
import os
import sys

SETTINGS = os.path.expanduser("~/.claude/settings.json")
MARKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_reminded")
REMINDER = (
    "Voice output is on. Load and follow the voice-output skill: end every reply that "
    "answers the user or reports a result with one speak block containing only the "
    "spoken conclusion. Never write the literal tag text anywhere else in the reply."
)


def voice_enabled() -> bool:
    try:
        with open(SETTINGS, encoding="utf-8") as handle:
            return bool(json.load(handle).get("voice", {}).get("enabled"))
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    data = json.load(sys.stdin)
    session_id = data.get("session_id", "")
    if not session_id or not voice_enabled():
        return 0
    marker = os.path.join(MARKER_DIR, session_id)
    if os.path.exists(marker):
        return 0
    os.makedirs(MARKER_DIR, exist_ok=True)
    open(marker, "w", encoding="utf-8").close()
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": REMINDER,
        }
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
