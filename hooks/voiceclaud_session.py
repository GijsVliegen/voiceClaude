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
