#!/usr/bin/env python3
"""Stop hook: reads Claude's reply aloud with spd-say.

Speaks only the <speak>...</speak> blocks in the assistant text written after the
latest user prompt (see the voice-output skill). Code blocks and markdown markup are skipped. To turn speech off, create ~/.claude/speak_off
(remove it to turn speech back on).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime

OFF_FLAG = os.path.expanduser("~/.claude/speak_off")
LOG = os.path.splitext(os.path.abspath(__file__))[0] + ".log"
MAX_CHARS = 1500
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPER = os.environ.get("VOICECLAUDE_PIPER", os.path.join(ROOT, ".venv", "bin", "piper"))
# en_US-lessac-medium or en_US-amy-medium
VOICE = os.environ.get("VOICECLAUDE_VOICE", os.path.join(ROOT, "voices", "en_US-lessac-medium.onnx"))


def last_reply_text(transcript_path: str) -> str:
    """Text of the assistant messages written after the most recent real user prompt."""
    parts = []
    with open(transcript_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = entry.get("message") or {}
            content = message.get("content")
            if entry.get("type") == "user" and isinstance(content, str):
                # a plain string is a real prompt; tool results are lists
                parts = []
            elif entry.get("type") == "assistant" and isinstance(content, list):
                parts.extend(
                    block.get("text", "") for block in content if block.get("type") == "text"
                )
    return "\n".join(p for p in parts if p.strip())


def spoken_only(text: str) -> str:
    """Text of the last complete <speak>...</speak> block; empty if there is none.

    An opening tag that is followed by another opening tag before it closes (for
    example a mention of the tag in prose) is skipped, so it cannot swallow the text around it.
    """
    blocks = re.findall(r"<speak>((?:(?!<speak>).)*?)</speak>", text, flags=re.DOTALL)
    return blocks[-1].strip() if blocks else ""


def speakable(text: str) -> str:
    """Strip code and markdown so the speech reads naturally."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)        # fenced code blocks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)            # [label](url) -> label
    text = re.sub(r"https?://\S+", " ", text)                        # bare links
    text = re.sub(r"`([^`]*)`", r"\1", text)                         # inline code
    text = re.sub(r"[*_#>|~]+", " ", text)                           # emphasis, headings, tables
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS]


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat(timespec='milliseconds')} {msg}\n")


def main() -> int:
    data = json.load(sys.stdin)
    if data.get("stop_hook_active") or os.path.exists(OFF_FLAG):
        log(f"skip: stop_hook_active={data.get('stop_hook_active')} off_flag={os.path.exists(OFF_FLAG)}")
        return 0
    path = data["transcript_path"]
    time.sleep(1.0)   # the final reply is written to the transcript just after the Stop event
    with open(path, encoding="utf-8") as handle:
        lines = sum(1 for _ in handle)
    text = speakable(spoken_only(last_reply_text(path)))
    log(f"path={path} lines={lines} text chars={len(text)} text={text[:120]!r}")
    if not text:
        return 0
    subprocess.run(["pkill", "-x", "aplay"], check=False)   # drop any speech still playing
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    piper = subprocess.run([PIPER, "-m", VOICE, "-f", wav], input=text.encode("utf-8"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    log(f"piper rc={piper.returncode} wav_bytes={os.path.getsize(wav)} err={piper.stderr[-200:]!r}")
    # play in the background and delete the clip afterwards
    subprocess.Popen(["sh", "-c", 'aplay -q "$1"; rm -f "$1"', "sh", wav],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
