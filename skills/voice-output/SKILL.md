---
name: voice-output
description: Use for every reply while the user has spoken voice output on (the Stop hook at ~/.claude/hooks/speak_reply.py is active), and whenever the user talks about voice, speech, or what gets read aloud. Defines the <speak> block that is the only part of a reply that is read aloud.
---

# Voice output

When voice output is on, a Stop hook reads aloud **only the text inside a `<speak>…</speak>` block** in your reply. Everything else is written for reading. If there is no `<speak>` block, nothing is spoken.

The user wants speech to be sparse. Spoken text should be the part they must hear, not a narration of your work.

## Writing the block

- Put one `<speak>` block at the **end** of the reply, after the written answer.
- Never write the literal tag text anywhere else in a reply, including in backticks or when discussing the voice feature. Say "the speak block" in prose instead.
- Include a block in **every** reply that answers the user or reports a result. Leave it out only for pure acknowledgements such as "Done" with nothing else to report. Do not skip it because the reply is short.
- Keep it to 1–3 short sentences, ideally under 50 words.
- Say the conclusion, the outcome, or the decision the user needs to make. Also say any warning they must act on.
- Write it as natural speech: no markdown, no code, no file paths, no symbols, no bullet lists.
- If the outcome is clear from the reply, the block can be a single sentence.

## Never speak

- Openers and narration: "I will start…", "Let me…", "Sure, here's…", "Now I'm going to…".
- What you did step by step, which files you read, which commands you ran, or how you found something.
- Process details, self-corrections, or tool errors that were resolved.
- Lists of changes, file names, line numbers, or identifiers.

## Examples

Written answer covers the details. The spoken part is only the conclusion:

<speak>The voice hook now speaks only what you mark as spoken. Nothing was tested end to end yet, so check whether the next reply is read aloud.</speak>

Routine work with nothing to hear: no block at all.
