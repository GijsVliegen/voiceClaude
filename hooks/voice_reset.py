#!/usr/bin/env python3
"""SessionStart hook (matchers compact and clear): forget that the voice reminder was sent,
so voice_reminder.py adds it again on the next prompt.
"""
import json
import os
import sys

MARKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_reminded")


def main() -> int:
    data = json.load(sys.stdin)
    session_id = data.get("session_id", "")
    marker = os.path.join(MARKER_DIR, session_id)
    if session_id and os.path.exists(marker):
        os.remove(marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
