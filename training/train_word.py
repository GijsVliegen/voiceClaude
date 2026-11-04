#!/usr/bin/env python3
"""Train a keyword model for openWakeWord: models/<word>.onnx.

  train_word.py start                          # synthetic voices only
  train_word.py start --own recordings/train   # plus your own recordings (see record_clips.py)

The keyword's spellings, near-misses and hard cases come from keywords.json (copy
keywords.example.json and fill in your own words). CPU only, no torch.
  1. Synthesis: piper renders every spelling (three speeds), every near-miss, the other keywords and
     a set of everyday sentences with each voice in voices/ (multi-speaker voices give
     --speakers-per-voice speakers each). Renders are cached in .cache/train/.
  2. Examples: 2 s windows. Positives end at most 0.3 s before the window end; the same word ending
     earlier, or cut off halfway, is a negative, because the live detector scores every 80 ms.
     Every window gets a quiet noise bed (live audio is never digital silence), then copies get
     random speed, reverb, noise and volume. The room's own noise is recorded from the mic first.
  3. Features: openWakeWord's own melspectrogram + embedding models turn each window into (16, 96).
  4. Classifier: a small scikit-learn MLP, exported as ONNX with the [1, 16, 96] -> [1, 1]
     input/output that openWakeWord's Model loads.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly

ROOT = Path(__file__).resolve().parent.parent
RATE = 16000
WINDOW = 2 * RATE  # 2 s gives exactly 16 embedding frames
SPEEDS = (0.85, 1.0, 1.2)

SENTENCES = [
    "can you open the settings file", "the train was twenty minutes late this morning", "what time is it in Tokyo",
    "please run the tests again", "I forgot my umbrella at the office", "the soup needs a little more salt",
    "we are going to the beach on Saturday", "send the report before lunch", "the printer is out of paper again",
    "my bike has a flat tyre", "this function returns the wrong value", "let us take a short break",
    "the garden looks beautiful in spring", "how far is it to the station", "turn the music down a bit",
    "the cat is sleeping on the sofa", "I will call you back in five minutes", "the database migration finished",
    "she bought fresh bread at the market", "the meeting moved to Thursday afternoon", "put the plates in the dishwasher",
    "where is the charger for my phone", "the river flooded the old bridge", "write a summary of this document",
    "the children built a sandcastle", "the server needs a restart tonight", "a cup of tea would be lovely",
    "check the logs for any errors", "the museum is closed on Mondays", "he plays the guitar in a small band",
    "the forecast says it will snow", "save the file and close the editor", "the bus stop is around the corner",
    "our neighbours have a new puppy", "can you explain that one more time", "the sun sets very early in winter",
]


# --- keywords --------------------------------------------------------------------------------

def load_profile(path: Path, word: str) -> dict:
    """The word's entry in keywords.json, plus 'others': two spellings of every other keyword
    (negatives, so the keywords never trigger each other)."""
    if not path.exists():
        sys.exit(f"{path} not found: copy keywords.example.json to keywords.json and fill in your words")
    words = json.loads(path.read_text())
    if word not in words:
        sys.exit(f"'{word}' is not in {path} (known: {', '.join(words)})")
    p = dict(words[word])
    p.setdefault("hard", {})
    p["others"] = [s for w, e in words.items() if w != word for s in e["spellings"][:2]]
    return p


# --- 1. synthesis ------------------------------------------------------------------------------

def speakers(voice: Path, per_voice: int, rng: random.Random) -> list:
    count = json.loads(Path(f"{voice}.json").read_text()).get("num_speakers", 1)
    return [None] if count <= 1 else sorted(rng.sample(range(count), min(count, per_voice)))


def synthesize(task: tuple) -> dict:
    """One voice x a few speakers -> {text: [int16 clip, ...]}; read from / written to the cache file."""
    voice, spks, texts, cache, seed = task
    cache = Path(cache)
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            return {k: list(v) for k, v in z.items()}
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
    # The pool already runs one process per core; let each piper session use a single thread.
    make_options = onnxruntime.SessionOptions

    def one_thread():
        options = make_options()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        return options
    onnxruntime.SessionOptions = one_thread
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    v = PiperVoice.load(voice)
    rng = random.Random(seed)
    out: dict[str, list] = {}
    for spk in spks:
        for text, speeds in texts:
            for speed in speeds or [rng.choice(SPEEDS)]:
                cfg = SynthesisConfig(speaker_id=spk, length_scale=speed, noise_scale=rng.uniform(0.5, 0.9))
                audio = np.concatenate([c.audio_float_array for c in v.synthesize(text, cfg)])
                if v.config.sample_rate != RATE:
                    audio = resample_poly(audio, RATE, v.config.sample_rate)
                out.setdefault(text, []).append((np.clip(audio, -1, 1) * 32767).astype(np.int16))
    np.savez(cache, **{k: np.array(c, dtype=object) for k, c in out.items()})
    return out


def synthesize_all(word: str, prof: dict, voices: list[Path], cache: Path, per_voice: int, jobs: int, seed: int) -> dict:
    texts = [(t, SPEEDS) for t in prof["spellings"]] + [(t, None) for t in prof["near_misses"] + prof["others"] + SENTENCES]
    key = hashlib.sha1(json.dumps([texts, per_voice, seed]).encode()).hexdigest()[:8]
    rng = random.Random(seed)
    tasks = []
    for voice in voices:
        spks = speakers(voice, per_voice, rng)
        for i in range(0, len(spks), 4):  # small tasks keep the pool busy until the end
            tasks.append((str(voice), spks[i:i + 4], texts, str(cache / f"{word}-{key}-{voice.stem}-{i // 4:03d}.npz"),
                          seed + len(tasks)))
    clips: dict[str, list] = {}
    start = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for n, out in enumerate(pool.map(synthesize, tasks), 1):
            for k, c in out.items():
                clips.setdefault(k, []).extend(c)
            print(f"\r  {n}/{len(tasks)} voice batches ({time.time() - start:.0f} s)", end="", flush=True)
    print()
    return clips


def record_room(cache: Path, seconds: float) -> np.ndarray:
    """Background noise of the room, from the default mic (cached)."""
    path = cache / "room.npy"
    if path.exists():
        return np.load(path)
    if seconds <= 0:
        return np.zeros(0, np.float32)
    import sounddevice as sd
    print(f"  recording {seconds:g} s of room noise: stay quiet...", flush=True)
    room = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    room = room[:, 0].copy()
    np.save(path, room)
    return room


# --- your own recordings -------------------------------------------------------------------

def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)[::ch].astype(np.float32) / 32768
    return resample_poly(x, RATE, rate).astype(np.float32) if rate != RATE else x


def word_bounds(x: np.ndarray) -> tuple[int, int] | None:
    """Start and end sample of the word in a record_clips.py clip (1 s room sound, then the word):
    the 20 ms frames after the first 0.5 s that are louder than 15 % of the loudest one.
    None when the clip is unusable: the word runs into the clip's end (cut off) or is shorter than 0.2 s."""
    hop = RATE // 50
    rms = np.array([np.std(x[i:i + hop]) for i in range(0, len(x) - hop + 1, hop)])
    skip = RATE // 2 // hop
    peak = rms[skip:].max()
    loud = np.flatnonzero(rms[skip:] >= min(peak, max(0.15 * peak, 3 * np.median(rms)))) + skip
    s, e = int(loud[0] * hop), int((loud[-1] + 1) * hop)
    return None if e > len(x) - 2 * hop or e - s < RATE // 5 else (s, e)


def hard_ends(talk: np.ndarray, model_path: Path, floor: float = 0.3) -> list[int]:
    """Sample positions in a talk recording where the CURRENT model scores >= floor (streamed like the
    mic): the moments it gets wrong, which get extra copies as negatives."""
    if not model_path.exists():
        return []
    from openwakeword.model import Model
    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    key = next(iter(m.models))
    pcm = (talk * 32767).astype(np.int16)
    return [i + 1280 for i in range(0, len(pcm) - 1280 + 1, 1280) if m.predict(pcm[i:i + 1280])[key] >= floor]


def load_own(folder: Path, word: str, model_path: Path) -> dict:
    own = dict(word=[], other=[], talk=[], hard=[])
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        wavs = [load_wav(p) for p in sorted(sub.glob("*.wav"))]
        if sub.name == "talk":
            for t in wavs:
                own["talk"].append(t)
                own["hard"].append(hard_ends(t, model_path))
        else:
            own["word" if sub.name == word else "other"] += wavs
    return own


# --- 2. examples -----------------------------------------------------------------------------

def place(x: np.ndarray, gap_s: float) -> np.ndarray:
    """A 2 s window in which x ends gap_s before the end (the front of x is cut if it does not fit)."""
    end = WINDOW - int(gap_s * RATE)
    x = x[max(0, len(x) - end):]
    out = np.zeros(WINDOW, np.float32)
    out[end - len(x):end] = x
    return out


def pink(n: int, nrng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(nrng.standard_normal(n)) / np.sqrt(np.arange(1, n // 2 + 2))
    x = np.fft.irfft(spectrum, n)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def mix(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    gain = np.sqrt((np.mean(x ** 2) + 1e-9) / ((np.mean(noise ** 2) + 1e-9) * 10 ** (snr_db / 10)))
    return x + gain * noise


class Augmenter:
    def __init__(self, room: np.ndarray, seed: int) -> None:
        self.room, self.rng, self.nrng = room, random.Random(seed), np.random.default_rng(seed)

    def room_piece(self) -> np.ndarray | None:
        if len(self.room) < WINDOW:
            return None
        s = self.rng.randrange(len(self.room) - WINDOW + 1)
        return self.room[s:s + WINDOW]

    def bed(self) -> np.ndarray:
        """Quiet background: a piece of the room, or soft pink noise."""
        piece = self.room_piece()
        if piece is None or self.rng.random() < 0.3:
            piece = 0.03 * pink(WINDOW, self.nrng)
        return piece * 10 ** (self.rng.uniform(-30, 0) / 20)

    def distort(self, x: np.ndarray) -> np.ndarray:
        rng = self.rng
        if rng.random() < 0.6:  # faster or slower (pitch moves with it)
            y = resample_poly(x, 100, round(100 * rng.uniform(0.88, 1.12))).astype(np.float32)
            x = place(y, 0) if len(y) < WINDOW else y[-WINDOW:]
        if rng.random() < 0.4:  # room echo: a decaying noise tail
            length = rng.uniform(0.15, 0.6)
            t = np.arange(int(length * RATE)) / RATE
            tail = self.nrng.standard_normal(len(t)).astype(np.float32) * np.exp(-7 * t / length)
            tail[0] = 3
            x = fftconvolve(x, tail / np.abs(tail).sum())[:WINDOW].astype(np.float32)
        piece = self.room_piece()
        if piece is not None and rng.random() < 0.6:
            x = mix(x, piece, rng.uniform(0, 25))
        if rng.random() < 0.5:
            noise = pink(WINDOW, self.nrng) if rng.random() < 0.5 else self.nrng.standard_normal(WINDOW).astype(np.float32)
            x = mix(x, noise, rng.uniform(5, 30))
        return np.clip(x * 10 ** (rng.uniform(-30, 0) / 20), -1, 1).astype(np.float32)

    def copies(self, x: np.ndarray, label: int, tag: str, extra: int):
        """The window on a noise bed, plus `extra` distorted copies of it."""
        x = np.clip(x * 10 ** (self.rng.uniform(-12, 0) / 20) + self.bed(), -1, 1).astype(np.float32)
        yield x, label, tag
        for _ in range(extra):
            yield self.distort(x), label, tag


def examples(clips: dict, prof: dict, aug: Augmenter, own: dict | None, own_reps: int, talk_hop: float):
    """Yields (2 s window, label, tag); a generator, so the audio is never all in memory."""
    rng = aug.rng
    for text, cs in clips.items():
        for c in cs:
            c = c.astype(np.float32) / 32768
            if text in prof["spellings"]:
                yield from aug.copies(place(c, rng.uniform(0, 0.3)), 1, text, 2)
                yield from aug.copies(place(c, rng.uniform(0.7, 1.3)), 0, "late " + text, 0)   # already over
                half = c[:int(len(c) * rng.uniform(0.35, 0.6))]                                # only the start
                yield from aug.copies(place(half, rng.uniform(0, 0.3)), 0, "partial " + text, 0)
            elif text in SENTENCES:
                s = rng.randrange(len(c) - WINDOW) if len(c) > WINDOW else 0
                yield from aug.copies(place(c[s:s + WINDOW], rng.uniform(0, 0.5)), 0, "sentence", 1)
            else:  # near-misses and the other keywords
                yield from aug.copies(place(c, rng.uniform(0, 0.4)), 0, text, prof["hard"].get(text, 1))
                yield from aug.copies(place(c, rng.uniform(0.4, 1.2)), 0, text, 0)
    for _ in range(200):  # the room alone, and plain noise, are negatives too
        piece = aug.room_piece()
        if piece is not None:
            yield from aug.copies(piece * 10 ** (rng.uniform(-6, 12) / 20), 0, "room", 0)
        yield from aug.copies(pink(WINDOW, aug.nrng) * 10 ** (rng.uniform(-30, -6) / 20), 0, "noise", 0)
    if not own:
        return

    def ending(c: np.ndarray, e: int, gap: float) -> np.ndarray:
        """The clip up to `gap` s after the word's end (real audio where the clip has it), framed."""
        g = int(gap * RATE)
        seg = c[:e + g]
        return place(seg, (e + g - len(seg)) / RATE)

    for c in own["word"]:  # your voice: many copies, each with its own end gap
        if not (b := word_bounds(c)):
            continue
        s, e = b
        for _ in range(max(1, own_reps // 5)):
            yield from aug.copies(ending(c, e, rng.uniform(0, 0.3)), 1, "own word", 4)
        yield from aug.copies(ending(c, e, rng.uniform(0.7, 1.3)), 0, "own late", 2)
        cut = c[:s + int((e - s) * rng.uniform(0.35, 0.6))]
        yield from aug.copies(place(cut, rng.uniform(0, 0.3)), 0, "own partial", 2)
    for c in own["other"]:  # your other keywords
        if not (b := word_bounds(c)):
            continue
        s, e = b
        yield from aug.copies(ending(c, e, rng.uniform(0, 0.4)), 0, "own other", 4)
        yield from aug.copies(ending(c, e, rng.uniform(0.4, 1.2)), 0, "own other", 0)
    hop = int(talk_hop * RATE)
    for t, hard in zip(own["talk"], own["hard"]):  # your normal speech: every window, worst ones extra
        for i in range(0, len(t) - WINDOW + 1, hop):
            yield from aug.copies(t[i:i + WINDOW], 0, "own talk", 1)
        for e in hard:
            if e >= WINDOW:
                yield from aug.copies(t[e - WINDOW:e], 0, "own hard", 6)


# --- 3. features, 4. classifier --------------------------------------------------------------

def features(gen, batch: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from openwakeword.utils import AudioFeatures
    extractor = AudioFeatures(inference_framework="onnx", ncpu=max(1, (os.cpu_count() or 2) - 1))
    feats, labels, tags, buf = [], [], [], []

    def flush() -> None:
        pcm = (np.stack(buf) * 32767).astype(np.int16)
        feats.append(extractor.embed_clips(pcm, batch_size=len(pcm)).astype(np.float32))
        buf.clear()
        print(f"\r  {sum(map(len, feats))} windows", end="", flush=True)

    for x, label, tag in gen:
        buf.append(x)
        labels.append(label)
        tags.append(tag)
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    print()
    out = np.concatenate(feats)
    if out.shape[1:] != (16, 96):
        sys.exit(f"unexpected feature shape {out.shape[1:]}")
    return out, np.array(labels, np.int8), np.array(tags)


def train(x: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(96, 32), alpha=1e-3, batch_size=256, max_iter=60,
                        early_stopping=True, n_iter_no_change=6, random_state=seed)
    return clf.fit(x.reshape(len(x), -1), y)


def to_onnx(clf, path: Path, name: str) -> None:
    """[1, 16, 96] -> flatten -> dense + ReLU layers -> dense + sigmoid -> [1, 1]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    weights = [numpy_helper.from_array(np.array([1, -1], np.int64), "flat_shape")]
    nodes = [helper.make_node("Reshape", ["input", "flat_shape"], ["h0"])]
    n = len(clf.coefs_)
    for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        weights += [numpy_helper.from_array(w.astype(np.float32), f"w{i}"),
                    numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
        nodes.append(helper.make_node("Gemm", [f"h{i}", f"w{i}", f"b{i}"], [f"z{i}"]))
        nodes.append(helper.make_node("Sigmoid" if i == n - 1 else "Relu", [f"z{i}"], ["output" if i == n - 1 else f"h{i + 1}"]))
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 96])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="voiceClaude")
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def check_onnx(path: Path, clf, x: np.ndarray) -> float:
    """Largest difference between the ONNX file and scikit-learn on 50 windows."""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    idx = np.linspace(0, len(x) - 1, 50).astype(int)
    a = np.array([session.run(None, {"input": x[i][None]})[0][0, 0] for i in idx])
    b = clf.predict_proba(x[idx].reshape(len(idx), -1))[:, 1]
    return float(np.abs(a - b).max())


def report(clf, x: np.ndarray, y: np.ndarray, tags: np.ndarray, held_out: np.ndarray) -> None:
    p = clf.predict_proba(x.reshape(len(x), -1))[:, 1]
    yt, pt = y[held_out], p[held_out]
    print(f"  held out: hit rate {np.mean(pt[yt == 1] >= 0.5):.3f}, false alarms {np.mean(pt[yt == 0] >= 0.5):.4f} "
          f"(at score 0.5; {int((yt == 1).sum())} positive, {int((yt == 0).sum())} negative)")
    for tag in dict.fromkeys(tags):
        m = tags == tag
        label = "pos" if y[m][0] else "neg"
        stat = p[m].min() if y[m][0] else p[m].max()
        print(f"    {label} {str(tag)!r}: {'min' if y[m][0] else 'max'} {stat:.3f}, mean {p[m].mean():.3f}, n={int(m.sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("word", help="a keyword from keywords.json")
    ap.add_argument("--keywords", default=str(ROOT / "keywords.json"))
    ap.add_argument("--voices", default=os.environ.get("VOICECLAUDE_VOICES", str(ROOT / "voices")),
                    help="folder with piper voices (.onnx + .onnx.json)")
    ap.add_argument("--own", default="", help="your recordings: <word>/*.wav, other word folders, talk/*.wav")
    ap.add_argument("--out", default="", help="default: models/<word>.onnx")
    ap.add_argument("--cache", default=str(ROOT / ".cache" / "train"))
    ap.add_argument("--speakers-per-voice", type=int, default=40)
    ap.add_argument("--room-seconds", type=float, default=15, help="room noise to record from the mic (0 = skip)")
    ap.add_argument("--own-reps", type=int, default=30, help="augmented copies per own recording of the word")
    ap.add_argument("--talk-hop", type=float, default=0.25, help="seconds between windows cut from talk recordings")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    prof = load_profile(Path(a.keywords), a.word)
    out = Path(a.out or ROOT / "models" / f"{a.word}.onnx")
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    voices = sorted(Path(a.voices).glob("*.onnx"))
    if not voices:
        sys.exit(f"no piper voices in {a.voices} (see README)")
    print(f"word: {a.word}; voices: {', '.join(v.stem for v in voices)}")
    room = record_room(cache, a.room_seconds)
    print("synthesizing...")
    clips = synthesize_all(a.word, prof, voices, cache, a.speakers_per_voice, a.jobs, a.seed)
    own = None
    if a.own:
        own = load_own(Path(a.own).expanduser(), a.word, out)  # finds hard spots with the old model first
        unusable = sum(word_bounds(c) is None for c in own["word"] + own["other"])
        print(f"own recordings: {len(own['word'])} of {a.word}, {len(own['other'])} other words ({unusable} unusable), "
              f"{sum(map(len, own['talk'])) / RATE / 60:.1f} min of talk ({sum(map(len, own['hard']))} hard spots)")
    print("features...")
    x, y, tags = features(examples(clips, prof, Augmenter(room, a.seed), own, a.own_reps, a.talk_hop))
    del clips
    print(f"windows: {len(y)} ({int(y.sum())} positive)")
    held_out = np.random.default_rng(a.seed).random(len(y)) < 0.1
    clf = train(x[~held_out], y[~held_out], a.seed)
    print(f"trained: {clf.n_iter_} epochs, best validation score {clf.best_validation_score_:.4f}")
    report(clf, x, y, tags, held_out)
    to_onnx(clf, out, a.word)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB); ONNX vs scikit-learn max difference {check_onnx(out, clf, x):.1e}")


if __name__ == "__main__":
    main()
