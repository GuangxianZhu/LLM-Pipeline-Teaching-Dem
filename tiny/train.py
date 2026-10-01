# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Train the tiny model (only needed once, by the teacher; students just use weights.npz).

    pip install torch
    python tiny/train.py

The maths is exactly the same as tiny/model.py (numpy), so the numpy version reproduces it.
"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from tiny import data                       # noqa: E402
from tiny.model import positional_encoding  # noqa: E402

CFG = dict(d=32, heads=2, layers=2, ff=128)
torch.manual_seed(0)


class Net(torch.nn.Module):
    def __init__(self, V, d, H, L, F):
        super().__init__()
        dh = d // H
        r = lambda *s: torch.nn.Parameter(torch.randn(*s) * (1 / math.sqrt(s[-2] if len(s) > 1 else 1)))  # noqa
        self.E = torch.nn.Parameter(torch.randn(V, d) * 0.5)
        self.layers = torch.nn.ModuleList()
        for _ in range(L):
            m = torch.nn.Module()
            m.Wq, m.Wk, m.Wv = r(H, d, dh), r(H, d, dh), r(H, d, dh)
            m.Wo = r(d, d)
            m.ln1_g, m.ln1_b = torch.nn.Parameter(torch.ones(d)), torch.nn.Parameter(torch.zeros(d))
            m.W1, m.b1 = r(d, F), torch.nn.Parameter(torch.zeros(F))
            m.W2, m.b2 = r(F, d), torch.nn.Parameter(torch.zeros(d))
            m.ln2_g, m.ln2_b = torch.nn.Parameter(torch.ones(d)), torch.nn.Parameter(torch.zeros(d))
            self.layers.append(m)
        self.Wout = r(d, V)
        self.d, self.H, self.dh = d, H, dh

    def forward(self, ids):                     # ids: [B, n]
        B, n = ids.shape
        x = self.E[ids] + torch.tensor(positional_encoding(n, self.d), dtype=torch.float32)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        for m in self.layers:
            heads = []
            for h in range(self.H):
                q, k, v = x @ m.Wq[h], x @ m.Wk[h], x @ m.Wv[h]
                s = (q @ k.transpose(1, 2)) / math.sqrt(self.dh)
                s = s.masked_fill(mask, float("-inf"))
                heads.append(torch.softmax(s, -1) @ v)
            x = torch.nn.functional.layer_norm(x + torch.cat(heads, -1) @ m.Wo, (self.d,), m.ln1_g, m.ln1_b, 1e-5)
            f = torch.relu(x @ m.W1 + m.b1) @ m.W2 + m.b2
            x = torch.nn.functional.layer_norm(x + f, (self.d,), m.ln2_g, m.ln2_b, 1e-5)
        return x @ self.Wout


def main():
    seqs = data.all_training_sequences()
    vocab = data.SPECIAL + sorted({t for s, _ in seqs for t in s} - set(data.SPECIAL))
    idx = {t: i for i, t in enumerate(vocab)}
    n = max(len(s) for s, _ in seqs)
    X = torch.zeros(len(seqs), n, dtype=torch.long)
    M = torch.zeros(len(seqs), n, dtype=torch.bool)
    for b, (s, m) in enumerate(seqs):
        X[b, :len(s)] = torch.tensor([idx[t] for t in s])
        M[b, :len(s)] = torch.tensor(m)
    net = Net(len(vocab), CFG["d"], CFG["heads"], CFG["layers"], CFG["ff"])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.0)
    steps = 4000
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps, pct_start=0.1)
    for step in range(steps):
        logits = net(X[:, :-1])
        tgt = X[:, 1:]
        m = M[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits[m], tgt[m])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1)[m] == tgt[m]).float().mean().item()
            print("step {:5d}  loss {:.4f}  acc {:.4f}".format(step, loss.item(), acc))
    out = {"E": net.E, "Wout": net.Wout}
    for l, m in enumerate(net.layers):
        for k in ("Wq", "Wk", "Wv", "Wo", "ln1_g", "ln1_b", "W1", "b1", "W2", "b2", "ln2_g", "ln2_b"):
            out["l{}.{}".format(l, k)] = getattr(m, k)
    np.savez(os.path.join(HERE, "weights.npz"), **{k: v.detach().numpy().astype(np.float32) for k, v in out.items()})
    with open(os.path.join(HERE, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "config": CFG}, f, ensure_ascii=False, indent=0)
    print("saved", len(vocab), "tokens")


if __name__ == "__main__":
    main()
