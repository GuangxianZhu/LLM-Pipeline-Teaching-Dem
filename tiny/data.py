# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Training / runtime data for the tiny model.

A conversation becomes ONE token sequence:

    <sys> <user> ...your words... <ai> <tool_call>{...}</tool_call> <tool> ...result... <ai> ...answer... <end>

<sys> stands for the whole system prompt + tool list (squeezed into one token in this tiny model).
There are two versions, because a real system prompt tells the model which computer it runs on:
<sys:unix> and <sys:win>  (the terminal command is different on Windows).

The model is only trained to predict what comes after <ai> (its own replies), never the user's
words or the tool results - exactly like real chat models.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sim          # noqa: E402
import tools        # noqa: E402

SPECIAL = ["<pad>", "<sys:unix>", "<sys:win>", "<user>", "<ai>", "<tool>", "<end>"]
HELLO_VARIANTS = [
    "Hi! I am a language model. I read your words as tokens and write my reply one token at a time.",
    "Hello! I am a language model. I read your words as tokens and write my reply one token at a time.",
]


def sys_token(os_name=None):
    return "<sys:win>" if (os_name or os.name) == "nt" else "<sys:unix>"


def terminal_command_for(os_name):
    return "dir /b /a-d" if os_name == "nt" else "ls -p"


def conversation(sc, os_name=None):
    """
    Run a scenario the 'scripted' way and return a list of segments:
        [(role, text_or_tokens), ...]  role in sys/user/ai/tool/end
    Used to build TRAINING data. At runtime the model writes the ai parts itself.
    """
    import scenarios
    os_name = os_name or os.name
    segs = [("sys", sys_token(os_name)), ("user", sc["prompt"])]
    results = []
    for f in sc["calls"]:
        name, args = f(results)
        if name == "run_terminal":
            args = {"command": terminal_command_for(os_name)}
        segs.append(("ai", scenarios.tool_call_text(name, args)))
        res = tools.run(name, args) if name != "run_terminal" else _fake_terminal(args["command"])
        results.append(res)
        segs.append(("tool", tool_text(res)))
    segs.append(("ai", sc["answer"](results)))
    return segs


def _fake_terminal(command):
    """Terminal result with the same text on every OS (the real tool lists demo_files/)."""
    names = sorted(f for f in os.listdir(tools.FILES_DIR) if os.path.isfile(os.path.join(tools.FILES_DIR, f)))
    out = "\n".join(names)
    return {"kind": "terminal", "output": out, "text": out}


def tool_text(res):
    """What the program puts into the context for a tool result."""
    return " ".join(res["text"].split())


def to_tokens(segs):
    """Segments -> (tokens, train_mask). train_mask[i] = True if token i must be PREDICTED."""
    toks, mask = [], []

    def add(ts, train):
        toks.extend(ts)
        mask.extend([train] * len(ts))
    for role, txt in segs:
        if role == "sys":
            add([txt], False)
        elif role == "user":
            add(["<user>"], False)
            add(sim.tokenize(txt), False)
        elif role == "tool":
            add(["<tool>"], False)
            add(sim.tokenize(txt), False)
        elif role == "ai":
            add(["<ai>"], False)
            add(sim.tokenize(txt), True)
    # the final answer ends with <end>
    add(["<end>"], True)
    return toks, mask


def all_training_sequences():
    import scenarios
    seqs = []
    for os_name in ("posix", "nt"):
        for sc in scenarios.SCENARIOS:
            if sc.get("kind") == "cache":
                continue
            if sc["key"] == "hello":
                for v in HELLO_VARIANTS:
                    segs = [("sys", sys_token(os_name)), ("user", sc["prompt"]), ("ai", v)]
                    seqs.append(to_tokens(segs))
                continue
            seqs.append(to_tokens(conversation(sc, os_name)))
    return seqs


if __name__ == "__main__":
    for toks, mask in all_training_sequences():
        print(len(toks), " ".join(sim.show(t) for t in toks)[:200])
