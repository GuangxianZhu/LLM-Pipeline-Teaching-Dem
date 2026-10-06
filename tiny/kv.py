# -*- coding: utf-8 -*-
# Claude Opus 写的（第 7 题：带 KV 缓存的一步计算）
"""
The same tiny Transformer as model.py, but computing ONE new token with a KV cache.

model.forward(tokens) recomputes every row from scratch. step() below only computes the row of the
new token: its Q, K, V; its K and V are appended to the cache; its Q is compared with ALL keys in the cache.
Question 7 runs both and shows that the next-token probabilities are exactly the same.

    cache = KVCache.from_trace(model, trace)     # the K and V of every token already processed
    probs = step(model, "<tool_call>", cache)    # one new row; cache grows by one row per layer and head
"""
import numpy as np

from tiny.model import layer_norm, positional_encoding, softmax


class KVCache:
    def __init__(self, layers, heads):
        self.K = [[None] * heads for _ in range(layers)]
        self.V = [[None] * heads for _ in range(layers)]
        self.n = 0                                   # number of tokens stored

    @classmethod
    def from_trace(cls, m, T):
        c = cls(m.L, m.h)
        for l, LT in enumerate(T["layers"]):
            for h, H in enumerate(LT["heads"]):
                c.K[l][h] = H["k"].copy()
                c.V[l][h] = H["v"].copy()
        c.n = len(T["tokens"])
        return c


def step(m, tok, cache):
    """Process one new token with the cache. Returns (probs, info): info holds that row's q, k, v and scores."""
    w = m.w
    p = cache.n                                      # position of the new token
    x = w["E"][m.ids([tok])[0]] + positional_encoding(p + 1, m.d)[p]
    info = []
    for l in range(m.L):
        pre = "l{}.".format(l)
        heads, li = [], []
        for hh in range(m.h):
            q = x @ w[pre + "Wq"][hh]
            k = x @ w[pre + "Wk"][hh]
            v = x @ w[pre + "Wv"][hh]
            cache.K[l][hh] = np.vstack([cache.K[l][hh], k[None, :]])
            cache.V[l][hh] = np.vstack([cache.V[l][hh], v[None, :]])
            s = cache.K[l][hh] @ q / np.sqrt(m.dh)    # one row of scores: the new token against every key
            a = softmax(s)
            heads.append(a @ cache.V[l][hh])
            li.append(dict(q=q, k=k, v=v, scores=s, att=a))
        delta = np.concatenate(heads) @ w[pre + "Wo"]
        x1 = layer_norm(x + delta, w[pre + "ln1_g"], w[pre + "ln1_b"])
        ff = np.maximum(x1 @ w[pre + "W1"] + w[pre + "b1"], 0) @ w[pre + "W2"] + w[pre + "b2"]
        x = layer_norm(x1 + ff, w[pre + "ln2_g"], w[pre + "ln2_b"])
        info.append(li)
    cache.n += 1
    return softmax(x @ w["Wout"]), info


def mults_per_row(m, n_ctx):
    """Multiplications to push ONE token row through the whole model, when it can see n_ctx tokens."""
    per_layer = (m.h * 3 * m.d * m.dh            # x W_Q, x W_K, x W_V for every head
                 + m.h * 2 * n_ctx * m.dh        # q against every key, then the weighted sum of the values
                 + m.d * m.d                     # W_O
                 + 2 * m.d * m.cfg["ff"])        # feed forward: W1 and W2
    return m.L * per_layer


def mults_output(m):
    return m.d * len(m.vocab)                    # the last row x W_out


def mults_no_cache(m, n):
    """All n rows from scratch (every row is computed against all n tokens, like model.forward)."""
    return n * mults_per_row(m, n) + mults_output(m)


def mults_with_cache(m, n):
    """Only the newest row (it sees n tokens: the n-1 in the cache and itself)."""
    return mults_per_row(m, n) + mults_output(m)
