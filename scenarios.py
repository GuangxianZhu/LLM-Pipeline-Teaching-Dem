# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
The preset questions students can pick.

Each scenario lists:
  prompt  - what the student "says"
  focus   - words the attention beams should light up
  calls   - list of functions  results_so_far -> (tool_name, arguments)
            (a later call may depend on an earlier tool result)
  answer  - function results -> final reply text
  first   - optional custom top-5 for the very first token of the reply
"""
import json
import os
from collections import Counter

import tools

SYSTEM_PROMPT = "You are a helpful assistant. You can call tools."


def _files(results):
    return tools.file_names(results[0]["output"])


def _by_type(results):
    c = Counter((os.path.splitext(n)[1] or "(none)") for n in _files(results))
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items], [v for _, v in items]


def _files_answer(results):
    names = _files(results)
    ex = ", ".join(names[:3])
    return "There are {} files in the demo folder, for example {}.".format(len(names), ex)


def _files_chart_answer(results):
    labels, values = _by_type(results)
    parts = ", ".join("{} {}".format(v, k) for k, v in zip(labels, values))
    return "Done. The folder has {} files ({}). The bar chart shows them by type.".format(
        sum(values), parts)


SCENARIOS = [
    dict(key="hello", prompt="Hi! Who are you?",
         groups={"hi": "greet", "who": "ask", "are": "ask", "you": "ask"},
         focus=" you", focus_group="ask", focus_label='you + context = "the assistant being asked"',
         keywords=["hi", "who", "you"],
         calls=[],
         answer=lambda r: "Hi! I am a language model. I read your words as tokens "
                          "and write my reply one token at a time.",
         first=[("Hi", 0.46), ("Hello", 0.31), ("I", 0.11), ("Hey", 0.07), ("Good", 0.05)]),

    dict(key="calc", prompt="What is 17% of 2350?",
         groups={"17": "num", "%": "num", "235": "num", "0": "num", "what": "ask"},
         focus="0", focus_group="num", focus_label='0 + context = part of the number "2350"',
         keywords=["17", "%", "2350"],
         calls=[lambda r: ("calculator", {"expression": "2350 * 17 / 100"})],
         answer=lambda r: "17% of 2350 is {}.".format(r[0]["value"])),

    dict(key="chart", prompt="Plot the tank temperature for the last hour.",
         groups={"plot": "action", "tank": "equip", "temper": "equip", "ature": "equip",
                 "last": "time", "hour": "time"},
         focus="ature", focus_group="equip", focus_label='ature + context = "tank temperature"',
         keywords=["plot", "temperature", "hour", "tank"],
         calls=[lambda r: ("plot_chart", {"type": "line", "title": "Tank temperature, last 60 min",
                                          "minutes": 60})],
         answer=lambda r: "Here is the chart. The tank stayed between {:.1f} and {:.1f} C "
                          "over the last hour.".format(r[0]["stats"]["min"], r[0]["stats"]["max"])),

    dict(key="image", prompt="Draw a cat wearing a cleanroom suit.",
         groups={"draw": "action", "cat": "animal", "clean": "suit", "room": "suit",
                 "suit": "suit"},
         focus="room", focus_group="suit", focus_label='room + context = "cleanroom suit" (not a room!)',
         keywords=["draw", "cat", "cleanroom", "suit"],
         calls=[lambda r: ("generate_image", {"prompt": "a cute cat in a white cleanroom suit, "
                                                        "goggles and mask, holding a silicon wafer"})],
         answer=lambda r: "Done! Here is a cat in a cleanroom suit, holding a wafer."),

    dict(key="terminal", prompt="How many files are in the demo folder?",
         groups={"how": "count", "many": "count", "files": "fs", "demo": "fs", "folder": "fs"},
         focus=" folder", focus_group="fs", focus_label='folder + context = "the demo folder"',
         keywords=["how", "many", "files", "folder"],
         calls=[lambda r: ("run_terminal", {"command": tools.terminal_command()})],
         answer=_files_answer),

    dict(key="agent", prompt="List the files here and make a bar chart by file type.",
         groups={"list": "fs", "files": "fs", "here": "fs", "file": "fs", "type": "fs",
                 "make": "chart", "bar": "chart", "chart": "chart"},
         focus=" chart", focus_group="chart", focus_label='chart + context = "bar chart"',
         keywords=["list", "files", "bar", "chart", "type"],
         calls=[lambda r: ("run_terminal", {"command": tools.terminal_command()}),
                lambda r: ("plot_chart", {"type": "bar", "title": "Files by type",
                                          "labels": _by_type(r)[0], "values": _by_type(r)[1]})],
         answer=_files_chart_answer),

    dict(key="cache", kind="cache", prompt="Why is it fast? (KV cache and cache hits)"),
]


def tool_call_text(name, args):
    """Exactly what the model writes when it wants a tool."""
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"
