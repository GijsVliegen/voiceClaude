#!/usr/bin/env python3
"""SessionStart hook: remember the session of the Claude that voiceclaud started.

voice_keys.py launches claude with VOICECLAUD=1; other sessions are ignored.
It fires on startup, resume and /clear, so the file always holds the current one.
"""
import json
import os
import sys
from pathlib import Path

if os.environ.get("VOICECLAUD") == "1":
    data = json.load(sys.stdin)
    out = Path(__file__).resolve().parent.parent / "last_session.json"
    out.write_text(json.dumps({"session_id": data["session_id"], "cwd": data["cwd"]}))
