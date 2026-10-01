# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
The tiny GPT-style Transformer, written in plain numpy so every number can be shown on screen.

Architecture (the classic one, nothing left out):

    token ids -> embedding E + positional encoding P              x : [n x d]
    repeat for each layer:
        for each head h:   Q = x Wq_h,  K = x Wk_h,  V = x Wv_h        [n x dh]
                           S = Q K^T / sqrt(dh)                        [n x n]
                           S[masked] = -inf   (no looking at later tokens)
                           A = softmax(S)  (row by row)
                           head_h = A V                                [n x dh]
        delta = concat(head_1, head_2) Wo                              [n x d]
        x = LayerNorm(x + delta)          ("Add & Norm")
        f = ReLU(x W1 + b1) W2 + b2       (feed-forward, d -> 4d -> d)
        x = LayerNorm(x + f)              ("Add & Norm")
    logits = x Wout                                                    [n x vocab]
    probabilities = softmax(logits of the LAST token)

forward(..., trace=True) records every intermediate matrix, which the animation shows.
"""
import json
import os
import sys

import numpy as np

if getattr(sys, "frozen", False):                  # the packaged .exe: tiny/ lives next to the exe
    HERE = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "tiny")
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-5


def positional_encoding(n, d):
    """The original sine/cosine position code: each position gets its own pattern of waves."""
    pos = np.arange(n)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * g + b


class TinyGPT:
    def __init__(self, path=None):
        path = path or os.path.join(HERE, "weights.npz")
        w = np.load(path)
        self.w = {k: w[k].astype(np.float64) for k in w.files}
        with open(os.path.join(HERE, "vocab.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.vocab = meta["vocab"]
        self.cfg = meta["config"]
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.d, self.h, self.L = self.cfg["d"], self.cfg["heads"], self.cfg["layers"]
        self.dh = self.d // self.h

    def ids(self, toks):
        return [self.index.get(t, 0) for t in toks]          # unknown token (e.g. odd tool output) -> <pad>

    def forward(self, toks, trace=False):
        w = self.w
        ids = self.ids(toks)
        n = len(ids)
        T = {"tokens": list(toks), "ids": ids}
        emb = w["E"][ids]
        pe = positional_encoding(n, self.d)
        x = emb + pe
        if trace:
            T.update(emb=emb, pe=pe, x0=x.copy(), layers=[])
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        for l in range(self.L):
            p = "l{}.".format(l)
            heads, LT = [], {"x_in": x.copy(), "heads": []}
            for hh in range(self.h):
                q = x @ w[p + "Wq"][hh]
                k = x @ w[p + "Wk"][hh]
                v = x @ w[p + "Wv"][hh]
                raw = q @ k.T
                s = raw / np.sqrt(self.dh)
                s_masked = np.where(mask, -np.inf, s)
                a = softmax(s_masked, -1)
                out = a @ v
                heads.append(out)
                if trace:
                    LT["heads"].append(dict(Wq=w[p + "Wq"][hh], Wk=w[p + "Wk"][hh], Wv=w[p + "Wv"][hh],
                                            q=q, k=k, v=v, raw=raw, scaled=s, masked=s_masked, att=a, out=out))
            cat = np.concatenate(heads, -1)
            delta = cat @ w[p + "Wo"]
            res1 = x + delta
            x1 = layer_norm(res1, w[p + "ln1_g"], w[p + "ln1_b"])
            hid_pre = x1 @ w[p + "W1"] + w[p + "b1"]
            hid = np.maximum(hid_pre, 0)
            ff = hid @ w[p + "W2"] + w[p + "b2"]
            res2 = x1 + ff
            x = layer_norm(res2, w[p + "ln2_g"], w[p + "ln2_b"])
            if trace:
                LT.update(cat=cat, Wo=w[p + "Wo"], delta=delta, res1=res1, x1=x1, hid_pre=hid_pre, hid=hid,
                          ff=ff, res2=res2, x_out=x.copy())
                T["layers"].append(LT)
        logits = x @ w["Wout"]
        probs = softmax(logits[-1])
        if trace:
            T.update(x_final=x, logits=logits[-1], probs=probs)
        return probs, T

    def top(self, probs, k=5):
        idx = np.argsort(-probs)[:k]
        return [(self.vocab[i], float(probs[i])) for i in idx]
