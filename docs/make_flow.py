# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
The big data-flow diagram of the whole demo (one straight line, left to right), drawn twice from ONE layout:
    docs/flow.drawio   - editable in draw.io / diagrams.net
    docs/flow.png      - preview
Convention: every token is a ROW (a horizontal bar). The last row is always <ai> (yellow) - in every matrix.
Run:  python docs/make_flow.py
"""
import os
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOKS = ["<sys>", "<user>", "Hi", "!", "Who", "are", "you", "?", "<ai>"]
IDS = [1, 3, 124, 96, 26, 32, 93, 117, 4]
N = len(TOKS)
RH = 14                      # one row (bar 10 + gap 4)
SH = N * RH                  # stack height
HERO_FILL, HERO_LINE = "#ffd966", "#b8860b"
W_FILL = "#dae8fc"           # weight matrices (learned)
OP_FILL = "#ffffff"

shapes = []                  # (kind, dict)


def rect(x, y, w, h, label="", fill="#ffffff", stroke="#000000", fs=12, dashed=False, bold=False, align="center"):
    shapes.append(("rect", dict(x=x, y=y, w=w, h=h, label=label, fill=fill, stroke=stroke, fs=fs, dashed=dashed,
                                bold=bold, align=align)))


def label(x, y, s, fs=12, color="#000000", w=160, align="center", bold=False):
    shapes.append(("text", dict(x=x, y=y, s=s, fs=fs, color=color, w=w, align=align, bold=bold)))


def circle(cx, cy, s, r=15):
    shapes.append(("circle", dict(cx=cx, cy=cy, r=r, s=s)))


def arrow(pts, fat=False, color="#000000", dashed=False, fixed=0):
    """fixed = how many of the first points are not moved by _shift()"""
    shapes.append(("arrow", dict(pts=pts, fat=fat, color=color, dashed=dashed, fixed=fixed)))


def stack(x, y, w, name, shape, rows=N, hero=True, half=None, dim_half=False):
    """n bars on top of each other: one per token. Returns (x, y, w, h)."""
    for i in range(rows):
        last = hero and i == rows - 1
        if half:            # two-tone bar (concat): left = head 1, right = head 2
            rect(x, y + i * RH, w / 2, RH - 4, fill=HERO_FILL if last else half[0], stroke=HERO_LINE if last else "#000000")
            rect(x + w / 2, y + i * RH, w / 2, RH - 4, fill=HERO_FILL if last else half[1],
                 stroke=HERO_LINE if last else "#000000")
        else:
            rect(x, y + i * RH, w, RH - 4, fill=HERO_FILL if last else "#ffffff", stroke=HERO_LINE if last else "#000000")
            if dim_half:     # ReLU: some neurons switched off
                rect(x + w * 0.55, y + i * RH, w * 0.45, RH - 4, fill="#e6e6e6" if not last else "#e8c95a",
                     stroke=HERO_LINE if last else "#000000")
    label(x + w / 2, y + rows * RH + 10, "{}  ({})".format(name, shape) if shape else name, 12, bold=True,
          w=max(w, 30 if not shape else 120))
    return x, y, w, rows * RH


def grid(x, y, name, sub, mask=False, cell=RH):
    """n x n table: rows = the query token (who looks), columns = the key token (who is looked at)."""
    for r in range(N):
        for c in range(N):
            hidden = mask and c > r
            last = r == N - 1
            rect(x + c * cell, y + r * cell, cell, cell,
                 fill="#d9d9d9" if hidden else (HERO_FILL if last else "#ffffff"),
                 stroke="#999999" if hidden else (HERO_LINE if last else "#666666"))
    label(x + N * cell / 2, y + N * cell + 10, name, 12, bold=True, w=150)
    label(x + N * cell / 2, y + N * cell + 26, sub, 11, color="#555555", w=170)
    return x, y, N * cell, N * cell


def weight(cx, cy, name, shape, w=84, h=54):
    rect(cx - w / 2, cy - h / 2, w, h, "{}\n{}".format(name, shape), fill=W_FILL, stroke="#6c8ebf", fs=12)
    return cx - w / 2, cy - h / 2, w, h


def mid(s):
    x, y, w, h = s
    return x + w / 2, y + h / 2


# ====================================================================== layout
Y = 300                         # top of the main row (the residual stream)
YC = Y + SH / 2                 # its centre line
rowy = lambda i, y0=Y: y0 + i * RH + (RH - 4) / 2      # noqa: E731

# ---- 1  text -> tokens -> IDs -> embedding table -> E
label(60, Y - 40, "1  tokens", 13, bold=True)
for i, t in enumerate(TOKS):
    label(55, rowy(i), t, 12, color=HERO_LINE if i == N - 1 else "#000000", w=60, bold=i == N - 1)
    label(135, rowy(i), str(IDS[i]), 12, color=HERO_LINE if i == N - 1 else "#000000", w=40)
label(135, Y - 40, "IDs", 13, bold=True, w=40)
arrow([(90, YC), (113, YC)])
tab = (190, Y - 70, 70, SH + 140)
rect(*tab, "", fill="#f5f5f5")
for tid in IDS:                 # the rows that are looked up
    yy = tab[1] + 8 + tid / 181 * (tab[3] - 16)
    rect(tab[0], yy, tab[2], 3, fill=HERO_FILL if tid == IDS[-1] else "#7f7f7f", stroke="none")
label(225, tab[1] + tab[3] + 12, "embedding table", 12, bold=True)
label(225, tab[1] + tab[3] + 28, "181 x 32  (learned)", 11, color="#555555")
arrow([(158, YC), (186, YC)])
E = stack(300, Y, 120, "E", "9 x 32")
arrow([(262, YC), (296, YC)])
label(225, Y - 105, "look up row = ID", 11, color="#555555")

# ---- 2  + positional encoding -> X
circle(470, YC, "+")
arrow([(422, YC), (453, YC)])
P = stack(410, Y + SH + 70, 120, "P  position", "9 x 32", hero=True)
arrow([(470, P[1] - 4), (470, YC + 17)])
X = stack(520, Y, 120, "X", "9 x 32")
arrow([(487, YC), (516, YC)])
label(530, Y - 40, "2  + position", 13, bold=True)

# ---- 3  attention, head 1: Q K V
H1 = 60                          # top of head 1 lane
qy, ky, vy = H1, Y, Y + 260
WQ = weight(720, qy + SH / 2, "x W_Q", "32 x 16")
WK = weight(720, ky + SH / 2, "x W_K", "32 x 16")
WV = weight(720, vy + SH / 2, "x W_V", "32 x 16")
arrow([(642, YC), (676, YC)], fat=True)
arrow([(660, YC), (660, qy + SH / 2), (676, qy + SH / 2)], fat=False)
arrow([(660, YC), (660, vy + SH / 2), (676, vy + SH / 2)], fat=False)
Q = stack(800, qy, 64, "Q", "9 x 16")
K = stack(800, ky, 64, "K", "9 x 16")
V = stack(800, vy, 64, "V", "9 x 16")
for w, s in ((WQ, Q), (WK, K), (WV, V)):
    arrow([(w[0] + w[2] + 2, mid(s)[1]), (s[0] - 4, mid(s)[1])])
label(830, H1 - 40, "3  attention (head 1)", 13, bold=True, w=200)

# ---- 4  scores = Q . K^T  -> scale, mask, softmax -> A
S = grid(950, H1 + 40, "scores = Q K^T", "9 x 9,  then / sqrt(16), mask")
arrow([(866, mid(Q)[1]), (946, mid(Q)[1])])
arrow([(866, mid(K)[1]), (900, mid(K)[1]), (900, S[1] + S[3] - 20), (946, S[1] + S[3] - 20)])
A = grid(1140, H1 + 40, "A = softmax", "weights, each row adds up to 1", mask=True)
arrow([(1078, mid(S)[1]), (1136, mid(S)[1])])

# ---- 5  A x V -> head output
circle(1203, mid(V)[1], "x")
arrow([(1203, A[1] + A[3] + 40), (1203, mid(V)[1] - 17)])
arrow([(866, mid(V)[1]), (1186, mid(V)[1])])
O1 = stack(1250, vy, 64, "head 1 out", "9 x 16")
arrow([(1220, mid(V)[1]), (1246, mid(V)[1])])

# ---- head 2: the same, compact lane below
H2 = Y + 470
label(830, H2 - 30, "head 2: the same steps with its own W_Q, W_K, W_V", 12, bold=True, w=420)
rect(675, H2 + 36, 90, 54, "x W_Q2, W_K2,\nW_V2", fill=W_FILL, stroke="#6c8ebf", fs=11)
arrow([(660, mid(V)[1]), (660, H2 + 63), (671, H2 + 63)])
for k, nm in enumerate(("Q2", "K2", "V2")):
    stack(790 + k * 34, H2, 26, nm, "", hero=True)
arrow([(767, H2 + 63), (786, H2 + 63)])
S2 = grid(950, H2, "scores 2", "")
A2 = grid(1140, H2, "A2", "", mask=True)
arrow([(894, H2 + 40), (946, H2 + 40)])
arrow([(1078, H2 + 63), (1136, H2 + 63)])
circle(1295, H2 + 63, "x")
arrow([(1268, H2 + 63), (1278, H2 + 63)])
arrow([(874, H2 + SH + 22), (874, H2 + SH + 40), (1295, H2 + SH + 40), (1295, H2 + 80)])
label(1100, H2 + SH + 52, "V2", 11, color="#555555", w=40)
O2 = stack(1328, H2, 64, "head 2 out", "9 x 16")
arrow([(1311, H2 + 63), (1325, H2 + 63)])
SHIFT_FROM, SHIFT = 1375, 110      # everything right of head 2 moves right to make room

# ---- 6  concat x W_O = delta X
C = stack(1380, Y, 120, "concat", "9 x 32", half=("#ffffff", "#f2f2f2"))
arrow([(1316, mid(O1)[1]), (1345, mid(O1)[1]), (1345, YC + 15), (1376, YC + 15)])
arrow([(1394, mid(O2)[1]), (1440, mid(O2)[1]), (1440, YC + 35), (1376, YC + 35)], fixed=3)
WO = weight(1560, YC, "x W_O", "32 x 32")
arrow([(1502, YC), (1516, YC)], fat=True)
D = stack(1640, Y, 120, "dX  (change)", "9 x 32")
arrow([(1604, YC), (1636, YC)])

# ---- 7  add & norm  (residual: X skips the whole attention block)
circle(1810, YC, "+")
arrow([(1762, YC), (1793, YC)])
arrow([(580, Y - 4), (580, -2), (1810, -2), (1810, YC - 17)], color="#b85450")
label(1195, -14, "residual: X skips attention and is ADDED back", 11, color="#b85450", w=360)
rect(1850, YC - 18, 70, 36, "norm", fill=OP_FILL)
arrow([(1827, YC), (1846, YC)])
X1 = stack(1960, Y, 120, "X1", "9 x 32")
arrow([(1922, YC), (1956, YC)])
label(1880, Y - 40, "4  add & norm", 13, bold=True)

# ---- 8  feed forward (every row on its own)
W1 = weight(2150, YC, "x W1", "32 x 128")
arrow([(2082, YC), (2106, YC)], fat=True)
Hh = stack(2230, Y, 240, "hidden + ReLU", "9 x 128", dim_half=True)
arrow([(2194, YC), (2226, YC)])
W2 = weight(2550, YC, "x W2", "128 x 32")
arrow([(2472, YC), (2506, YC)], fat=True)
F = stack(2630, Y, 120, "F", "9 x 32")
arrow([(2594, YC), (2626, YC)])
label(2400, Y - 40, "5  feed forward (each row separately)", 13, bold=True, w=320)

# ---- 9  add & norm -> layer output
circle(2800, YC, "+")
arrow([(2752, YC), (2783, YC)])
arrow([(2020, Y - 4), (2020, Y - 62), (2800, Y - 62), (2800, YC - 17)], color="#b85450")
rect(2840, YC - 18, 70, 36, "norm", fill=OP_FILL)
arrow([(2817, YC), (2836, YC)])
X2 = stack(2950, Y, 120, "X2 = layer output", "9 x 32")
arrow([(2912, YC), (2946, YC)])
rect(650, -40, 2440, 990, "", fill="none", stroke="#82b366", dashed=True)
label(2990, Y - 90, "repeat: layer 2 (same steps,\nits own weights)", 12, color="#3b7a28", w=220, bold=True)

# ---- 10  output: only the <ai> row
hy = rowy(N - 1)
xl = (3150, hy - 5, 120, RH - 4)
rect(*xl, fill=HERO_FILL, stroke=HERO_LINE)
label(3210, hy + 22, "only the <ai> row (1 x 32)", 12, bold=True, w=200)
arrow([(3072, hy), (3146, hy)], color=HERO_LINE)
WOUT = weight(3360, hy, "x W_out", "32 x 181", w=90, h=60)
arrow([(3272, hy), (3313, hy)], fat=True)
lg = (3450, hy - 5, 300, RH - 4)
rect(*lg, fill="#ffffff")
label(3600, hy + 22, "181 scores (one per token in the table)", 12, bold=True, w=300)
arrow([(3407, hy), (3446, hy)])
rect(3790, hy - 18, 80, 36, "softmax", fill=OP_FILL)
arrow([(3752, hy), (3786, hy)])
probs = [("Hi", 0.50), ("Hello", 0.50), ("mask", 0.00)]
for i, (t, p) in enumerate(probs):
    yy = hy - 30 + i * 26
    label(3935, yy + 6, t, 12, w=50, align="right")
    rect(3965, yy, max(2, 180 * p), 14, fill=HERO_FILL if i == 0 else "#cfe2f3", stroke="#000000")
    label(3965 + max(2, 180 * p) + 8, yy + 6, "{:.0%}".format(p), 11, w=40, align="left")
arrow([(3872, hy), (3905, hy)])
label(4040, hy - 60, "6  next token", 13, bold=True, w=160)
rect(4230, hy - 20, 70, 40, "Hi", fill=HERO_FILL, stroke=HERO_LINE, fs=14, bold=True)
arrow([(4192, hy - 24), (4226, hy)])

# ---- 11  loop: append and run again  /  tool call
arrow([(4265, hy + 22), (4265, 1010), (55, 1010), (55, Y + SH + 4)], color="#6c8ebf", dashed=True)
label(2150, 995, "append the new token to the tokens and run everything again (until <end>)", 12, color="#6c8ebf",
      w=600, bold=True)
rect(3700, 1040, 560, 60, "if the model writes <tool_call>{...}</tool_call>: the PROGRAM runs the tool,\n"
     "its result is turned into tokens, appended, and the model continues", fill="#fff2cc", stroke="#d6b656", fs=12)


def _shift():
    def sx(x):
        return x + SHIFT if x >= SHIFT_FROM else x
    for kind, d in shapes:
        if kind == "rect":
            if d["dashed"] and d["x"] < SHIFT_FROM <= d["x"] + d["w"]:       # the layer frame: grow
                d["w"] += SHIFT
            d["x"] = sx(d["x"])
        elif kind == "text":
            d["x"] = sx(d["x"])
        elif kind == "circle":
            d["cx"] = sx(d["cx"])
        elif kind == "arrow":
            d["pts"] = [(x, y) if k < d["fixed"] else (sx(x), y) for k, (x, y) in enumerate(d["pts"])]


_shift()


# ====================================================================== draw.io
def drawio(path):
    cells, k = [], [2]

    def nid():
        k[0] += 1
        return "c{}".format(k[0])
    for kind, d in shapes:
        if kind == "rect":
            st = "rounded=0;whiteSpace=wrap;html=1;fillColor={};strokeColor={};fontSize={};{}{}".format(
                "none" if d["fill"] == "none" else d["fill"], "none" if d["stroke"] == "none" else d["stroke"], d["fs"],
                "dashed=1;" if d["dashed"] else "", "fontStyle=1;" if d["bold"] else "")
            cells.append('<mxCell id="{}" value="{}" style="{}" vertex="1" parent="1"><mxGeometry x="{}" y="{}" '
                         'width="{}" height="{}" as="geometry"/></mxCell>'.format(
                             nid(), escape(d["label"]).replace("\n", "&lt;br&gt;"), st, d["x"], d["y"], d["w"], d["h"]))
        elif kind == "text":
            w = d["w"]
            x0 = d["x"] - w / 2 if d["align"] == "center" else (d["x"] - w if d["align"] == "right" else d["x"])
            st = "text;html=1;align={};verticalAlign=middle;whiteSpace=wrap;fontSize={};fontColor={};{}".format(
                d["align"], d["fs"], d["color"], "fontStyle=1;" if d["bold"] else "")
            lines_ = d["s"].count("\n") + 1
            cells.append('<mxCell id="{}" value="{}" style="{}" vertex="1" parent="1"><mxGeometry x="{}" y="{}" '
                         'width="{}" height="{}" as="geometry"/></mxCell>'.format(
                             nid(), escape(d["s"]).replace("\n", "&lt;br&gt;"), st, x0, d["y"] - 9 * lines_, w,
                             18 * lines_))
        elif kind == "circle":
            r = d["r"]
            cells.append('<mxCell id="{}" value="{}" style="ellipse;whiteSpace=wrap;html=1;fontSize=16;fontStyle=1;" '
                         'vertex="1" parent="1"><mxGeometry x="{}" y="{}" width="{}" height="{}" as="geometry"/>'
                         '</mxCell>'.format(nid(), escape(d["s"]), d["cx"] - r, d["cy"] - r, 2 * r, 2 * r))
        elif kind == "arrow":
            p = d["pts"]
            st = ("shape=flexArrow;endArrow=classic;html=1;width=10;endSize=6;fillColor=#ffffff;" if d["fat"] else
                  "endArrow=classic;html=1;rounded=0;strokeColor={};{}".format(d["color"],
                                                                               "dashed=1;" if d["dashed"] else ""))
            pts = "".join('<mxPoint x="{}" y="{}"/>'.format(x, y) for x, y in p[1:-1])
            cells.append('<mxCell id="{}" style="{}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                         '<mxPoint x="{}" y="{}" as="sourcePoint"/><mxPoint x="{}" y="{}" as="targetPoint"/>'
                         '{}</mxGeometry></mxCell>'.format(nid(), st, p[0][0], p[0][1], p[-1][0], p[-1][1],
                                                         '<Array as="points">{}</Array>'.format(pts) if pts else ""))
    xml = ('<mxfile host="app.diagrams.net"><diagram name="LLM flow"><mxGraphModel grid="1" gridSize="10" '
           'page="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{}</root></mxGraphModel></diagram></mxfile>'
           .format("".join(cells)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


# ====================================================================== png
def png(path):
    W, H = 4460, 1150
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H - 60, -60)
    ax.axis("off")
    for kind, d in shapes:
        if kind == "rect":
            ax.add_patch(Rectangle((d["x"], d["y"]), d["w"], d["h"],
                                   facecolor="none" if d["fill"] == "none" else d["fill"],
                                   edgecolor="none" if d["stroke"] == "none" else d["stroke"], linewidth=0.8,
                                   linestyle="--" if d["dashed"] else "-"))
            if d["label"]:
                ax.text(d["x"] + d["w"] / 2, d["y"] + d["h"] / 2, d["label"], ha="center", va="center",
                        fontsize=d["fs"] * 0.75, fontweight="bold" if d["bold"] else "normal")
        elif kind == "text":
            ax.text(d["x"], d["y"], d["s"], ha=d["align"] if d["align"] != "center" else "center", va="center",
                    fontsize=d["fs"] * 0.75, color=d["color"], fontweight="bold" if d["bold"] else "normal")
        elif kind == "circle":
            ax.add_patch(Circle((d["cx"], d["cy"]), d["r"], facecolor="white", edgecolor="black", linewidth=0.8))
            ax.text(d["cx"], d["cy"], d["s"], ha="center", va="center", fontsize=12, fontweight="bold")
        elif kind == "arrow":
            p = d["pts"]
            ls = "--" if d["dashed"] else "-"
            if d["fat"]:
                ax.add_patch(FancyArrowPatch(p[0], p[-1], arrowstyle="simple,head_width=16,head_length=12,tail_width=8",
                                             facecolor="white", edgecolor="black", linewidth=0.8))
                continue
            for a, b in zip(p[:-2], p[1:-1]):
                ax.plot([a[0], b[0]], [a[1], b[1]], color=d["color"], linewidth=1.0, linestyle=ls)
            ax.add_patch(FancyArrowPatch(p[-2], p[-1], arrowstyle="-|>,head_width=3,head_length=6",
                                         color=d["color"], linewidth=1.0, linestyle=ls))
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    drawio(os.path.join(HERE, "flow.drawio"))
    png(os.path.join(HERE, "flow.png"))
    print("ok", len(shapes), "shapes")
