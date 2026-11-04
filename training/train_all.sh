#!/usr/bin/env bash
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
