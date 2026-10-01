# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Toy "LLM internals" used by the demo.

Nothing here is a real language model. Every function produces numbers that
LOOK like what a real model does (tokens, ids, vectors, attention weights,
probabilities), deterministically, so every student sees the same thing.
"""
import math
import random
import re
import zlib

TOOL_NAMES = ["calculator", "run_terminal", "plot_chart", "generate_image"]
SPECIAL_TOKENS = ["<tool_call>", "</tool_call>"]

# Words treated as a single token even though they contain "_" etc.
_WHOLE = SPECIAL_TOKENS + TOOL_NAMES
_WHOLE_RE = re.compile("(" + "|".join(re.escape(w) for w in _WHOLE) + ")")
_PIECE_RE = re.compile(r"\s?[A-Za-z]+|\s?\d{1,3}|\s?[^\sA-Za-z\d]|\s+")


def _split_word(w):
    """Split long words into sub-word pieces, like BPE does ("temperature" -> "temper" + "ature")."""
    lead = " " if w.startswith(" ") else ""
    core = w.strip()
    if not core.isalpha() or len(core) <= 7:
        return [w]
    if len(core) <= 12:
        cut = (len(core) + 1) // 2
        return [lead + core[:cut], core[cut:]]
    a = len(core) // 3
    return [lead + core[:a], core[a:2 * a], core[2 * a:]]


def tokenize(text):
    """Return a list of token strings. A leading space stays part of the token."""
    tokens = []
    for part in _WHOLE_RE.split(text):
        if not part:
            continue
        if part in _WHOLE:
            tokens.append(part)
            continue
        for m in _PIECE_RE.finditer(part):
            piece = m.group()
            if piece.isspace():
                continue  # (spaces glue to the next token)
            tokens.extend(_split_word(piece))
    return tokens


def show(tok):
    """How a token is printed on screen. '_' marks a leading space."""
    return "_" + tok[1:] if tok.startswith(" ") else tok


def token_id(tok):
    if tok == "<tool_call>":
        return 100257
    if tok == "</tool_call>":
        return 100258
    return 100 + zlib.crc32(tok.encode("utf-8")) % 50000


def embedding(tok, dims=8):
    """A fake but stable vector for a token (real models: 4096+ numbers)."""
    rng = random.Random(zlib.crc32(("emb" + tok).encode("utf-8")))
    return [round(rng.uniform(-1, 1), 2) for _ in range(dims)]


def softmax(xs, temp=1.0):
    m = max(xs)
    es = [math.exp((x - m) / temp) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def attention(labels, focus, seed=""):
    """
    Attention weights from the LAST position to every position.
    Tokens listed in `focus` (the words that matter for this question) get a boost.
    """
    rng = random.Random(zlib.crc32(("att" + seed + "|".join(labels)).encode("utf-8")))
    focus_l = [f.lower() for f in focus if f and f.strip()]
    scores = []
    for lab in labels:
        s = rng.uniform(0.0, 1.0)
        t = lab.strip().lstrip("_").lower()
        if t and any(t == f or (len(t) > 2 and (f.startswith(t) or t.startswith(f))) for f in focus_l):
            s += 2.6
        scores.append(s)
    return softmax(scores, temp=0.8)


_SENTENCE_START = ["The", "Here", "I", "Sure", "It", "Done", "Yes", "This", "Okay"]
_POOL = [" the", " a", ",", " is", " I", " to", " it", ".", " and", " of", " you",
         " chart", " file", " in", " for", " with", '"', ":", " that", " on"]


def candidates(chosen, step_seed, prev=None, override=None):
    """
    Top-5 next-token candidates with probabilities; `chosen` is always the winner.
    `override` = list of (token, prob) to show instead (for key teaching moments).
    """
    if override:
        return override
    rng = random.Random(zlib.crc32(("cand" + step_seed + chosen).encode("utf-8")))
    pool = _SENTENCE_START if (prev is None or prev.strip() in ("", ".", "!", "?")) else _POOL
    others = [t for t in pool if t != chosen]
    rng.shuffle(others)
    others = others[:4]
    p = rng.uniform(0.55, 0.96)
    rest = 1.0 - p
    raw = sorted([rng.uniform(0.2, 1.0) for _ in others], reverse=True)
    raw_sum = sum(raw)
    out = [(chosen, p)] + [(t, rest * r / raw_sum * 0.9) for t, r in zip(others, raw)]
    return out


def tool_name_candidates(chosen, seed):
    """When the model writes the tool name, the 'competition' is between tool names."""
    rng = random.Random(zlib.crc32(("tool" + seed + chosen).encode("utf-8")))
    p = rng.uniform(0.82, 0.94)
    others = [t for t in TOOL_NAMES if t != chosen]
    rng.shuffle(others)
    rest = 1.0 - p
    w = [0.55, 0.3, 0.15]
    out = [(chosen, p)] + [(t, rest * wi * 0.95) for t, wi in zip(others, w)]
    out.append(("I", rest * 0.05))
    return out
