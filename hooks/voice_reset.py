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
