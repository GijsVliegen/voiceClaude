#!/usr/bin/env bash
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
# Trains every keyword in keywords.json, with your own recordings when recordings/train exists.
# One log per word: .cache/train-<word>.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VOICECLAUDE_PYTHON:-$ROOT/.venv/bin/python}"
OWN=()
[ -d "$ROOT/recordings/train" ] && OWN=(--own "$ROOT/recordings/train")
mkdir -p "$ROOT/.cache"
for word in $("$PY" -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))))' "$ROOT/keywords.json"); do
    echo "training $word (log: .cache/train-$word.log)"
    "$PY" "$ROOT/training/train_word.py" "$word" "${OWN[@]}" "$@" > "$ROOT/.cache/train-$word.log" 2>&1
    echo "exit $?" >> "$ROOT/.cache/train-$word.log"
done
