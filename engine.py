# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Runs a whole conversation with the REAL tiny model and the REAL tools.

    context = <sys> <user> ...prompt... <ai>
    loop:
        the model predicts one token at a time (greedy; the very first token of question 1 is sampled)
        if it wrote <tool_call>...</tool_call>: the program parses the JSON, runs the tool,
            appends <tool> + result + <ai> to the context, and the model continues
        if it wrote <end>: done
"""
import json
import os
import random

import numpy as np

import sim
import tools
from tiny import data
from tiny.model import TinyGPT

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = TinyGPT()
    return _MODEL


class Engine:
    MAX_TOKENS = 120

    def __init__(self, sc, seed=None):
        self.m = model()
        self.sc = sc
        self.prompt_toks = sim.tokenize(sc["prompt"])
        self.context = [data.sys_token(), "<user>"] + self.prompt_toks + ["<ai>"]
        self.first_context = list(self.context)
        self.rounds = []        # one per model reply: dict(kind, toks, steps, text, name, args, start)
        self.results = []
        rng = random.Random(seed if seed is not None else random.randrange(10 ** 6))
        self._run(rng)

    def _generate(self, rng, sample_first=False):
        start = len(self.context)
        toks, steps = [], []
        for j in range(self.MAX_TOKENS):
            probs, _ = self.m.forward(self.context)
            top = self.m.top(probs, 5)
            if sample_first and j == 0:
                r, acc, pick = rng.random(), 0.0, top[0][0]
                for t, p in top:
                    acc += p
                    if r <= acc:
                        pick = t
                        break
            else:
                pick = top[0][0]
            steps.append(dict(top=top, pick=pick, p=float(probs[self.m.index[pick]])))
            self.context.append(pick)
            toks.append(pick)
            if pick in ("</tool_call>", "<end>"):
                break
        return toks, steps, start

    def _run(self, rng):
        for _ in range(4):
            toks, steps, start = self._generate(rng, sample_first=(not self.rounds and self.sc["key"] == "hello"))
            text = "".join(t for t in toks if t != "<end>")
            r = dict(toks=toks, steps=steps, text=text, start=start)
            if toks and toks[0] == "<tool_call>" and toks[-1] == "</tool_call>":
                call = json.loads(text[len("<tool_call>"):-len("</tool_call>")])   # the program reads the JSON
                r.update(kind="tool", name=call["name"], args=call["arguments"])
                res = tools.run(call["name"], call["arguments"])
                self.results.append(res)
                r["result"] = res
                r["result_text"] = data.tool_text(res)
                r["result_toks"] = sim.tokenize(r["result_text"])
                self.rounds.append(r)
                self.context += ["<tool>"] + r["result_toks"] + ["<ai>"]
            else:
                r.update(kind="answer")
                self.rounds.append(r)
                break

    def trace_first(self):
        """Full trace (every intermediate matrix) for predicting the very first reply token."""
        _, T = self.m.forward(self.first_context, trace=True)
        return T


if __name__ == "__main__":
    import scenarios
    for sc in scenarios.SCENARIOS[:6]:
        e = Engine(sc, seed=1)
        print("====", sc["prompt"])
        for r in e.rounds:
            print("  ", r["kind"], r["text"][:110])
            if r["kind"] == "tool":
                print("     ->", r["result_text"][:90])
    print(os.path.basename(__file__), "ok", np.__version__)
