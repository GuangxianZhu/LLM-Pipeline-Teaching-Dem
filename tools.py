# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
The four REAL tools the "model" can call.

The model only writes JSON text. These are ordinary Python functions that the
program (the "harness") runs when it sees that JSON. Their output is real:
the terminal command really runs, the chart is really drawn by matplotlib.
"""
import ast
import math
import operator
import os
import random
import subprocess
import sys

if getattr(sys, "frozen", False):                  # the packaged .exe: data lives next to the exe
    HERE = os.path.dirname(os.path.abspath(sys.executable))
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
FILES_DIR = os.path.join(HERE, "demo_files")      # the folder the terminal tool looks at

def _load_frozen_matplotlib():
    """In the packaged .exe matplotlib looks for its data folder next to its own __file__, which points to
    a folder that does not exist. Import it by hand with __file__ next to the .exe, where mpl-data is."""
    import importlib.util
    spec = importlib.util.find_spec("matplotlib")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = os.path.join(HERE, "matplotlib.py")      # -> HERE/mpl-data
    sys.modules["matplotlib"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules["matplotlib"]
        raise


try:
    if getattr(sys, "frozen", False) and "matplotlib" not in sys.modules:
        _load_frozen_matplotlib()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    import numpy as np
    HAVE_MPL = True
except Exception:  # pragma: no cover - shown to the user instead
    import traceback
    traceback.print_exc()                          # ends up in the log file of the packaged .exe
    HAVE_MPL = False

# chart colours (dark screen)
SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
GRID = "#3a3a37"
SERIES = "#3987e5"

_counter = {"chart": 0, "image": 0}


# ---------------------------------------------------------------- calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos, ast.Mod: operator.mod}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only + - * / ** % and numbers are allowed")


def calculator(expression):
    value = _eval(ast.parse(expression, mode="eval"))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"kind": "number", "value": value,
            "text": str(value),
            "screen": "{}\n\n= {}".format(expression, value),
            "python": "calculator(expression={!r})".format(expression)}


# ---------------------------------------------------------------- terminal
# Only these exact, read-only commands may run. Anything else is refused.
_WHITELIST = {
    "dir /b /a-d": ["cmd", "/c", "dir", "/b", "/a-d"],   # Windows: files only
    "ls -p": ["ls", "-p"],                                # macOS / Linux
}


def terminal_command():
    """The listing command that fits this computer."""
    return "dir /b /a-d" if os.name == "nt" else "ls -p"


def file_names(output):
    """Turn the listing into file names (folders end with '/' in `ls -p`)."""
    return [ln.strip() for ln in output.splitlines()
            if ln.strip() and not ln.strip().endswith("/")]


def run_terminal(command):
    if command not in _WHITELIST:
        out = "REFUSED: '{}' is not on the whitelist".format(command)
    else:
        try:
            flags = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW: no console pops up
            r = subprocess.run(_WHITELIST[command], cwd=FILES_DIR, capture_output=True,
                               text=True, errors="replace", timeout=5, creationflags=flags)
            out = (r.stdout or r.stderr).strip()
        except Exception as e:  # e.g. command missing
            out = "ERROR: {}".format(e)
    prompt = "demo_files> " if os.name == "nt" else "demo_files $ "
    return {"kind": "terminal", "output": out,
            "text": out,
            "screen": prompt + command + "\n" + out,
            "python": "run_terminal(command={!r})".format(command)}


# ---------------------------------------------------------------- charts
def _fig_to_rgba(fig):
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return arr


def _slug(s):
    out = "".join(c if c.isalnum() else "_" for c in s.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:30] or "out"


def _save(arr, kind, name=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    _counter[kind] += 1
    name = "{}.png".format(_slug(name)) if name else "{}_{}.png".format(kind, _counter[kind])
    path = os.path.join(OUT_DIR, name)
    plt.imsave(path, arr)
    return "outputs/" + name


def tank_temperature(minutes=60):
    """Simulated SC1 bath temperature log (setpoint 65 C)."""
    rng = random.Random(7)
    t, v = [], []
    for i in range(minutes):
        drift = 0.35 * math.sin(i / 9.0) + 0.15 * math.sin(i / 3.1)
        dip = -0.6 * math.exp(-((i - 38) ** 2) / 10.0)   # a chemical refill at ~38 min
        t.append(i - minutes)
        v.append(round(65.0 + drift + dip + rng.uniform(-0.08, 0.08), 2))
    return t, v


def _style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_chart(type, title, labels=None, values=None, source=None, minutes=60):
    if not HAVE_MPL:
        return {"kind": "text", "text": "ERROR: matplotlib is not installed",
                "screen": "pip install matplotlib", "python": "plot_chart(...)"}
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=100)
    _style(ax, fig)
    summary = ""
    if type == "line":
        x, y = tank_temperature(minutes)
        ax.plot(x, y, color=SERIES, linewidth=2)
        ax.axhline(65.0, color=INK2, linewidth=0.8, linestyle="--")
        ax.text(x[0], 65.03, "setpoint 65 C", color=INK2, fontsize=8, va="bottom")
        ax.set_xlabel("minutes ago", color=INK2, fontsize=9)
        ax.set_ylabel("temperature (C)", color=INK2, fontsize=9)
        summary = "{} points, min {:.1f} C, max {:.1f} C".format(len(y), min(y), max(y))
        stats = {"min": min(y), "max": max(y), "n": len(y)}
    else:  # bar
        bars = ax.bar(labels, values, color=SERIES, width=0.6)
        for b, val in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(val),
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_ylabel("files", color=INK2, fontsize=9)
        ax.yaxis.get_major_locator().set_params(integer=True)
        summary = "{} bars".format(len(values))
        stats = {}
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    arr = _fig_to_rgba(fig)
    path = _save(arr, "chart", title)
    args = "type={!r}, title={!r}".format(type, title)
    return dict(kind="image", image=arr, path=path, stats=stats,
                text="Chart saved to {} ({})".format(path, summary),
                screen="saved " + path, python="plot_chart({}, ...)".format(args))


# ---------------------------------------------------------------- "AI" image
def _draw_cat():
    """Hand-drawn (by code) picture standing in for a text-to-image model."""
    fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=(0, 10, 0, 10), cmap="Blues", vmin=-0.6, vmax=1.8,
              origin="lower", aspect="auto")
    ax.add_patch(patches.Rectangle((0, 0), 10, 1.2, color="#cfd8dc"))      # floor
    tail = patches.FancyArrowPatch((6.8, 1.6), (8.8, 4.6), connectionstyle="arc3,rad=-0.5",
                                   arrowstyle="-", linewidth=9, color="#f2994a")
    ax.add_patch(tail)
    ax.add_patch(patches.FancyBboxPatch((3.0, 0.9), 4.0, 4.4, boxstyle="round,pad=0.3",
                                        fc="white", ec="#b0bec5", lw=1.5))   # suit body
    ax.add_patch(patches.Rectangle((4.85, 1.0), 0.3, 4.0, color="#e3eaee"))  # zipper
    for sx in (1, -1):                                                        # arms + gloves
        ax.add_patch(patches.FancyBboxPatch((5 + sx * 2.1 - 0.5, 2.2), 1.0, 2.4,
                                            boxstyle="round,pad=0.2", fc="white", ec="#b0bec5"))
        ax.add_patch(patches.Circle((5 + sx * 1.0, 2.4), 0.55, fc="#7e57c2", ec="#5e35b1"))
    ax.add_patch(patches.Circle((5, 2.6), 1.05, fc="#cfd8dc", ec="#90a4ae", lw=1.5))  # wafer
    for k in np.linspace(4.2, 5.8, 5):
        ax.plot([k, k], [1.8, 3.4], color="#90a4ae", lw=0.5)
        ax.plot([4.2, 5.8], [k - 2.4, k - 2.4], color="#90a4ae", lw=0.5)
    for sx in (1, -1):                                                        # hood ears
        ax.add_patch(patches.Polygon([[5 + sx * 0.9, 7.9], [5 + sx * 1.7, 9.5], [5 + sx * 2.0, 7.4]],
                                     fc="white", ec="#b0bec5"))
        ax.add_patch(patches.Polygon([[5 + sx * 1.2, 8.0], [5 + sx * 1.65, 9.0], [5 + sx * 1.8, 7.7]],
                                     fc="#f8bbd0"))
    ax.add_patch(patches.Circle((5, 6.6), 2.2, fc="white", ec="#b0bec5", lw=1.5))   # hood
    ax.add_patch(patches.Circle((5, 6.5), 1.55, fc="#f2994a"))                      # face
    ax.add_patch(patches.FancyBboxPatch((3.55, 6.55), 2.9, 0.75, boxstyle="round,pad=0.15",
                                        fc="#1e3a5f", ec="#0d2137"))                # goggles
    ax.add_patch(patches.Ellipse((4.3, 7.05), 0.5, 0.2, fc="#90caf9"))
    ax.add_patch(patches.Ellipse((5.9, 7.05), 0.5, 0.2, fc="#90caf9"))
    ax.add_patch(patches.FancyBboxPatch((4.15, 5.25), 1.7, 0.8, boxstyle="round,pad=0.15",
                                        fc="#bbdefb", ec="#90caf9"))                # mask
    for sy in (5.75, 5.5):                                                          # whiskers
        ax.plot([3.1, 3.9], [sy + 0.15, sy], color="#5d4037", lw=1)
        ax.plot([6.1, 6.9], [sy, sy + 0.15], color="#5d4037", lw=1)
    return _fig_to_rgba(fig)


def generate_image(prompt, size="256x256"):
    if not HAVE_MPL:
        return {"kind": "text", "text": "ERROR: matplotlib is not installed",
                "screen": "pip install matplotlib", "python": "generate_image(...)"}
    arr = _draw_cat()
    path = _save(arr, "image", "cat_image")
    return dict(kind="image", image=arr, path=path, diffusion=True,
                text="Image saved to {}".format(path),
                screen="saved " + path,
                python="generate_image(prompt={!r})".format(prompt[:40] + "..."))


TOOLS = {"calculator": calculator, "run_terminal": run_terminal,
         "plot_chart": plot_chart, "generate_image": generate_image}

# What the model is told about the tools (part of the hidden system prompt).
TOOL_SPECS = [
    "calculator(expression): exact arithmetic",
    "run_terminal(command): run a whitelisted shell command",
    "plot_chart(type, title, ...): draw a line or bar chart",
    "generate_image(prompt): create a picture from text",
]


def run(name, args):
    return TOOLS[name](**args)
