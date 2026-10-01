# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Inside the Transformer as ONE FIXED DATA FLOW, left to right (the plan is docs/flow.drawio).

* Every token is a ROW of numbers (red = positive, blue = negative, dark = near 0).
  The rows keep the same order everywhere, so the LAST row is always <ai> (yellow frame) - the row
  that finally predicts the next token.
* Every intermediate result stays where it was made. Nothing moves away; the camera travels.
* The whole strip is drawn first as empty frames (the map). Each step fills in its part.
* Every "x W" is an ordinary matrix multiplication, shown the way students learned it.

All numbers are the real numbers of the tiny trained model (tiny/).
"""
import numpy as np
from direct.interval.IntervalGlobal import (Func, LerpColorScaleInterval, LerpFunc, LerpHprInterval,
                                            LerpPosInterval, Parallel, Sequence, Wait)
from panda3d.core import Point3, TextNode

from kit import (BLUE, GREY, ORANGE, TOKEN_COLORS, WHITE, YELLOW, Fonts, arrow2d, disc, fade_in, fade_out, fill,
                 heatmap, lines, rect, text)

FX = 40.0              # world x where the strip starts
RH = 0.62              # row pitch (one token)
GAPZ = 0.30            # gap between rows (fraction)
CD = 0.15              # width of one number in a 32/16-wide row
CW = 0.075             # width of one number in a 128-wide row
G = 2.0                # gap between stations
SQ = "√"
FRAME = (0.30, 0.30, 0.34, 1)
LINE = (0.62, 0.62, 0.68, 1)
RESID = (0.85, 0.45, 0.42, 1)
MAPTXT = (0.6, 0.6, 0.66, 1)
MASKED = (0.30, 0.30, 0.34, 1)
TS = 0.5               # name of a matrix
TSH = 0.36             # its shape
TN = 0.38              # notes


def disp(t):
    if t.startswith("<sys"):
        return "<sys>"
    return t.strip() or t


def sc_of(M):
    M = np.asarray(M, dtype=float)
    M = M[np.isfinite(M)]
    return max(1e-6, 1.6 * float(np.std(M)) if M.size else 1.0)


def show_fade(node, dur=0.4):
    return Sequence(Func(node.show), fade_in(node, dur))


def gray_color(v, scale=1.0):
    """Attention weights: 0 = dark, 1 = white."""
    b = 0.10 + 0.9 * min(1.0, max(0.0, v / scale))
    return (b, b, b, 1)


class Box:
    """Where one matrix lives on the board (top-left corner, cell sizes)."""

    def __init__(self, x0, z_top, rows, cols, cw, rh):
        self.x0, self.z_top, self.rows, self.cols, self.cw, self.rh = x0, z_top, rows, cols, cw, rh
        self.w, self.h = cols * cw, rows * rh
        self.x1, self.z_bot = x0 + self.w, z_top - self.h

    def rz(self, i):
        """centre z of row i"""
        return self.z_top - (i + 0.5) * self.rh

    def cx(self, j):
        return self.x0 + (j + 0.5) * self.cw

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2

    @property
    def zc(self):
        return (self.z_top + self.z_bot) / 2


class TransformerSteps:
    """Mixin for Story. Needs: self.board, self.view(), self.engine, self.app."""

    # ================================================================== setup + layout
    def tf_setup(self):
        self.T = self.engine.trace_first()
        self.ttoks = self.T["tokens"]
        self.n = n = len(self.ttoks)
        self.tcolor = [GREY if t.startswith("<") else TOKEN_COLORS[i % len(TOKEN_COLORS)]
                       for i, t in enumerate(self.ttoks)]
        self.hi = n - 1                             # the followed token: the last one, <ai>
        self.d = self.T["x0"].shape[1]
        self.dh = self.T["layers"][0]["heads"][0]["q"].shape[1]
        self.ff = self.T["layers"][0]["hid"].shape[1]
        self.V = len(self.engine.m.vocab)
        stream = [self.T["emb"], self.T["x0"]] + [LT[k] for LT in self.T["layers"] for k in ("res1", "x1", "x_out")]
        self.rs = sc_of(np.vstack(stream))          # ONE colour scale for all 32-wide token rows
        self.tf_root = None
        self.persist = []
        self._layout()

    def _layout(self):
        n, d, dh, ff = self.n, self.d, self.dh, self.ff
        SH = n * RH
        self.SH = SH
        L = {}
        zm = 0.0                                    # main lane (the residual stream)
        zq = SH + 4.2                               # Q lane (above)
        zv = -(SH + 3.6)                            # V lane (below)
        zh2 = zv - SH - 4.4                         # head 2 lane
        self.zm, self.zq, self.zv, self.zh2 = zm, zq, zv, zh2
        x = FX
        L["words_x"] = x + 1.6                      # right edge of the token words
        L["ids_x"] = x + 2.4
        L["table"] = Box(x + 3.4, zm + 2.2, self.V, d, CW, (SH + 4.4) / self.V)
        x = L["table"].x1 + G
        L["E"] = Box(x, zm, n, d, CD, RH)
        L["plus0"] = (L["E"].x1 + 1.4, L["E"].zc)
        L["P"] = Box(L["plus0"][0] - d * CD / 2, zm - SH - 2.4, n, d, CD, RH)
        L["X"] = Box(L["E"].x1 + 2.8, zm, n, d, CD, RH)
        fan = L["X"].x1 + 1.2
        L["fan"] = fan
        wx = fan + 1.2
        for nm, zt in (("Q", zq), ("K", zm), ("V", zv)):
            zc = zt - SH / 2
            L["W" + nm] = Box(wx, zc + d * CD / 2, d, dh, CD, CD)
        sx = L["WQ"].x1 + 1.8
        for nm, zt in (("Q", zq), ("K", zm), ("V", zv)):
            L[nm] = Box(sx, zt, n, dh, CD, RH)
        gx = L["Q"].x1 + 0.9
        L["S"] = Box(gx, zq, n, n, RH, RH)                          # rows = queries, columns = keys
        L["KT"] = Box(gx, zq + 0.6 + dh * CD, dh, n, RH, CD)        # K transposed, above the table
        L["A"] = Box(L["S"].x1 + 7.8, zq, n, n, RH, RH)            # notes about the scores go in between
        L["mix"] = L["A"].x0 + 0.9                                    # x of the weights column (V lane)
        L["Vc"] = Box(L["A"].x0 + 1.7, zv, n, dh, CD, RH)            # V copies being mixed
        L["O1"] = Box(max(L["A"].x1, L["Vc"].x1) + 2.2, zv, n, dh, CD, RH)
        # head 2 (compact lane)
        c2 = 0.06
        L["W2"] = Box(wx, zh2 - SH / 2 + 1.2, 1, 1, 2.4, 2.4)        # just a labelled box
        L["Q2"] = Box(sx - 0.5, zh2, n, dh, c2, RH)
        L["K2"] = Box(L["Q2"].x1 + 0.25, zh2, n, dh, c2, RH)
        L["V2"] = Box(L["K2"].x1 + 0.25, zh2, n, dh, c2, RH)
        L["S2"] = Box(gx, zh2, n, n, RH, RH)
        L["A2"] = Box(L["A"].x0, zh2, n, n, RH, RH)
        L["O2"] = Box(L["O1"].x0, zh2, n, dh, CD, RH)
        # concat, W_O, delta
        L["C"] = Box(L["O1"].x1 + 2.6, zm, n, d, CD, RH)
        L["WO"] = Box(L["C"].x1 + 1.6, L["C"].zc + d * CD / 2, d, d, CD, CD)
        L["D"] = Box(L["WO"].x1 + 1.6, zm, n, d, CD, RH)
        L["plus1"] = (L["D"].x1 + 1.3, L["D"].zc)
        L["norm1"] = (L["plus1"][0] + 1.7, L["D"].zc)
        L["X1"] = Box(L["norm1"][0] + 1.6, zm, n, d, CD, RH)
        # feed forward
        L["W1"] = Box(L["X1"].x1 + 1.6, L["X1"].zc + d * CD / 2, d, ff, CW, CD)
        L["H"] = Box(L["W1"].x1 + 1.6, zm, n, ff, CW, RH)
        L["W2f"] = Box(L["H"].x1 + 1.6, L["H"].zc + ff * CW / 2, ff, d, CD, CW)
        L["F"] = Box(L["W2f"].x1 + 1.6, zm, n, d, CD, RH)
        L["plus2"] = (L["F"].x1 + 1.3, L["F"].zc)
        L["norm2"] = (L["plus2"][0] + 1.7, L["F"].zc)
        L["X2"] = Box(L["norm2"][0] + 1.6, zm, n, d, CD, RH)
        # layer 2 (one box) and its output
        L["L2"] = (L["X2"].x1 + 1.6, L["X2"].x1 + 8.6)
        L["X3"] = Box(L["L2"][1] + 1.6, zm, n, d, CD, RH)
        # output: only the <ai> row
        hz = L["X3"].z_top - (n - 1) * RH
        L["row"] = Box(L["X3"].x1 + 2.0, hz, 1, d, CD, RH)
        L["WOUT"] = Box(L["row"].x1 + 1.6, L["row"].zc + d * CD / 2, d, self.V, 0.06, CD)
        L["logit"] = Box(L["WOUT"].x1 + 1.6, hz, 1, self.V, 0.06, RH)
        L["probs_x"] = L["logit"].x1 + 4.0
        L["end_x"] = L["probs_x"] + 9.0
        self.L = L
        self.top_z = zq + 0.6 + dh * CD + 2.2
        self.bot_z = zh2 - SH - 1.4

    # ================================================================== drawing helpers
    def root(self):
        if self.tf_root is None:
            self.tf_root = self.board.attachNewNode("transformer")
        return self.tf_root

    def keep(self, node):
        self.persist.append(node)
        return node

    def mapkey(self, *keys):
        return Func(self.app.ui.arch.highlight, *keys)

    def track(self, vec, where):
        return Func(self.app.ui.track, disp(self.ttoks[self.hi]), vec, self.rs, where)

    def go(self, x0, x1, z0, z1, h=0.0, p=0.0):
        """Camera so that the board rectangle x0..x1, z0..z1 is visible above the caption."""
        w, hgt = x1 - x0, z1 - z0
        d = max(w / 0.95, hgt / 0.66, 12.0)
        return self.view(Point3((x0 + x1) / 2, 0, (z0 + z1) / 2 - 0.07 * d), h, p, d)

    def mat(self, M, box, scale=None, colors=None, hero=True, gapx=0.08, parent=None, nan=(0.04, 0.04, 0.05, 1)):
        """Heatmap of M in box (rows = tokens). Hidden. Hero row framed."""
        g = (parent or self.root()).attachNewNode("mat")
        heatmap(g, M, box.x0, box.z_top, box.cw, box.rh, scale or sc_of(M), gap=GAPZ if box.rh == RH else 0.06,
                gapx=gapx, colors=colors, nan_color=nan)
        if hero and box.rows == self.n:
            z = box.rz(self.hi)
            rect(g, box.x0 - 0.08, z - RH / 2 + 0.02, box.x1 + 0.08, z + RH / 2 - 0.02, YELLOW, 2.4, y=-0.02)
        g.hide()
        return g

    def frame(self, parent, box, label=None, shape=None):
        rect(parent, box.x0 - 0.1, box.z_bot - 0.06, box.x1 + 0.1, box.z_top + 0.06, FRAME, 1.2)
        if label:
            z = box.z_bot - 0.65
            text(parent, label, Point3(box.xc, 0, z), TS, MAPTXT, Fonts.symbol)
            if shape:
                text(parent, shape, Point3(box.xc, 0, z - 0.5), TSH, (0.5, 0.5, 0.56, 1), Fonts.symbol)

    def row_node(self, vec, cw, scale=None, colors=None, ai=True):
        """One token row (1 x k) with its top-left corner at the node origin. Hidden."""
        nd = self.board.attachNewNode("row")
        heatmap(nd, np.asarray(vec)[None, :], 0, 0, cw, RH, scale or self.rs, gap=GAPZ, gapx=0.08, colors=colors)
        if ai:
            rect(nd, -0.08, -RH + 0.02, len(vec) * cw + 0.08, -0.02, YELLOW, 2.4, y=-0.02)
        nd.hide()
        return nd

    def fly(self, nd, frm, to, dur=0.9, delay=0.0, hide_end=True):
        nd.setPos(*frm)
        s = Sequence(Wait(delay), Func(nd.setPos, *frm), Func(nd.show),
                     LerpPosInterval(nd, dur, Point3(*to), blendType="easeInOut"))
        if hide_end:
            s.append(Func(nd.hide))
        return s

    def sweep(self, nodes, total=0.8):
        gap = total / max(1, len(nodes))
        return Parallel(*[Sequence(Wait(i * gap), show_fade(nd, 0.25)) for i, nd in enumerate(nodes)])

    def rows_of(self, M, box, scale=None):
        """Matrix drawn as separate row nodes (so they can appear one by one). All hidden, all kept."""
        out = []
        for i in range(self.n):
            g = self.root().attachNewNode("r")
            heatmap(g, M[i:i + 1], box.x0, box.z_top - i * box.rh, box.cw, box.rh, scale or sc_of(M), gap=GAPZ,
                    gapx=0.08)
            if i == self.hi:
                z = box.rz(i)
                rect(g, box.x0 - 0.08, z - RH / 2 + 0.02, box.x1 + 0.08, z + RH / 2 - 0.02, YELLOW, 2.4, y=-0.02)
            g.hide()
            out.append(self.keep(g))
        return out

    def ai_tag(self, box):
        """small '<ai>' label left of the hero row (hidden, kept)"""
        t = text(self.root(), "<ai>", Point3(box.x0 - 0.2, 0, box.rz(self.hi) - 0.11), 0.32, YELLOW,
                 align=TextNode.ARight)
        t.hide()
        return self.keep(t)

    def note(self, s, x, z, scale=TN, color=GREY, align=TextNode.ALeft):
        g = self.root().attachNewNode("note")
        text(g, s, Point3(x, 0, z), scale, color, Fonts.symbol, align=align)
        g.hide()
        return self.keep(g)

    def top_attn(self, att, k=3):
        idx = np.argsort(-att[self.hi])[:k]
        return ", ".join("'{}' {:.2f}".format(disp(self.ttoks[j]), att[self.hi, j]) for j in idx)

    # ================================================================== 0 the map (empty frames)
    def build_map(self):
        """Draw the whole strip as empty frames + names + connectors: the map of the journey."""
        L, n = self.L, self.n
        g = self.root().attachNewNode("map")
        d, dh, ff = self.d, self.dh, self.ff
        sh = lambda a, b: "{} × {}".format(a, b)  # noqa: E731
        for key, lab, shp in (("E", "E", sh(n, d)), ("P", "P  (position)", sh(n, d)), ("X", "X", sh(n, d)),
                              ("Q", "Q", sh(n, dh)), ("K", "K", sh(n, dh)), ("V", "V", sh(n, dh)),
                              ("S", "scores", sh(n, n)), ("A", "weights A", sh(n, n)), ("O1", "head 1 out", sh(n, dh)),
                              ("S2", "scores (head 2)", None), ("A2", "weights (head 2)", None),
                              ("O2", "head 2 out", sh(n, dh)), ("C", "concat", sh(n, d)), ("D", "ΔX  (change)", sh(n, d)),
                              ("X1", "X1", sh(n, d)), ("H", "hidden", sh(n, ff)), ("F", "F", sh(n, d)),
                              ("X2", "X2  (layer 1 out)", sh(n, d)), ("X3", "X3  (final)", sh(n, d))):
            self.frame(g, L[key], label=lab, shape=shp)
        for key, lab, shp in (("WQ", "W_Q", sh(d, dh)), ("WK", "W_K", sh(d, dh)), ("WV", "W_V", sh(d, dh)),
                              ("WO", "W_O", sh(d, d)), ("W1", "W1", sh(d, ff)), ("W2f", "W2", sh(ff, d)),
                              ("WOUT", "W_out", sh(d, self.V))):
            b = L[key]
            fill(g, b.x0, b.z_bot, b.x1, b.z_top, (0.35, 0.5, 0.85, 1), 0.10)
            rect(g, b.x0, b.z_bot, b.x1, b.z_top, (0.45, 0.6, 0.95, 1), 1.2)
            text(g, lab, Point3(b.xc, 0, b.z_bot - 0.65), TS, (0.55, 0.68, 1.0, 1), Fonts.symbol)
            text(g, shp, Point3(b.xc, 0, b.z_bot - 1.15), TSH, (0.5, 0.56, 0.72, 1), Fonts.symbol)
        for key in ("Q2", "K2", "V2"):
            self.frame(g, L[key])
        text(g, "Q2  K2  V2", Point3(L["K2"].xc, 0, L["Q2"].z_bot - 0.55), 0.32, MAPTXT)
        b = L["W2"]
        rect(g, b.x0, b.z_bot, b.x1, b.z_top, (0.45, 0.6, 0.95, 1), 1.2)
        text(g, "× its own\nW_Q, W_K, W_V", Point3(b.xc, 0, b.zc + 0.2), 0.28, (0.55, 0.68, 1.0, 1), Fonts.symbol)
        text(g, "head 2", Point3(b.x0, 0, L["S2"].z_top + 0.5), 0.4, MAPTXT, align=TextNode.ALeft)
        text(g, "head 1", Point3(L["WQ"].x0, 0, L["Q"].z_top + 0.5), 0.4, MAPTXT, align=TextNode.ALeft)
        self.frame(g, L["table"])
        text(g, "embedding table", Point3(L["table"].xc, 0, L["table"].z_bot - 0.5), 0.36, MAPTXT)
        text(g, sh(self.V, d), Point3(L["table"].xc, 0, L["table"].z_bot - 0.9), 0.26, (0.45, 0.45, 0.5, 1),
             Fonts.symbol)
        for key in ("row", "logit"):
            self.frame(g, L[key])
        for key in ("plus0", "plus1", "plus2"):
            cx, cz = L[key]
            disc(g, 0.42, (0.18, 0.18, 0.2, 1), 24).setPos(cx, 0.01, cz)
            text(g, "+", Point3(cx, 0, cz - 0.17), 0.55, WHITE)
        for key in ("norm1", "norm2"):
            cx, cz = L[key]
            rect(g, cx - 0.7, cz - 0.38, cx + 0.7, cz + 0.38, LINE, 1.2)
            text(g, "norm", Point3(cx, 0, cz - 0.12), 0.32, WHITE)
        x0, x1 = L["L2"]
        zc = L["X2"].zc
        rect(g, x0, zc - 3.0, x1, zc + 3.0, (0.45, 0.75, 0.45, 1), 1.6)
        text(g, "layer 2", Point3((x0 + x1) / 2, 0, zc + 1.6), 0.5, (0.6, 0.9, 0.6, 1))
        text(g, "the same steps again\n(attention, add & norm,\nfeed forward, add & norm)\nwith its own weights",
             Point3((x0 + x1) / 2, 0, zc + 0.7), 0.27, GREY)
        lx0, lx1 = L["fan"] - 0.4, L["X2"].x1 + 0.6
        rect(g, lx0, self.bot_z + 0.3, lx1, self.top_z - 0.3, (0.3, 0.5, 0.3, 1), 1.0)
        text(g, "layer 1", Point3(lx0 + 0.3, 0, self.top_z - 0.9), 0.45, (0.5, 0.8, 0.5, 1), align=TextNode.ALeft)
        text(g, "ATTENTION", Point3(L["S"].x0, 0, self.top_z - 0.9), 0.45, MAPTXT, align=TextNode.ALeft)
        text(g, "FEED FORWARD", Point3(L["W1"].x0, 0, L["W1"].z_top + 1.6), 0.45, MAPTXT, align=TextNode.ALeft)
        text(g, "OUTPUT", Point3(L["row"].x0, 0, L["WOUT"].z_top + 1.6), 0.45, MAPTXT, align=TextNode.ALeft)
        self._connectors(g)
        g.hide()
        return g

    def _connectors(self, g):
        """Thin arrows of the flow (grey) and the two residual lines (red)."""
        L, zc = self.L, self.L["X"].zc
        segs = []
        a = lambda p0, p1, c=LINE: arrow2d(g, (p0[0], 0, p0[1]), (p1[0], 0, p1[1]), c, 1.4, 0.16)  # noqa: E731
        a((L["table"].x1 + 0.2, zc), (L["E"].x0 - 0.2, zc))
        a((L["E"].x1 + 0.2, zc), (L["plus0"][0] - 0.5, zc))
        a((L["plus0"][0], L["P"].z_top + 0.15), (L["plus0"][0], zc - 0.5))
        a((L["plus0"][0] + 0.5, zc), (L["X"].x0 - 0.2, zc))
        f = L["fan"]
        segs += [[(L["X"].x1 + 0.15, 0, zc), (f, 0, zc)], [(f, 0, L["Q"].zc), (f, 0, L["W2"].zc)]]
        for k in ("Q", "K", "V"):
            a((f, L[k].zc), (L["W" + k].x0 - 0.15, L[k].zc))
            a((L["W" + k].x1 + 0.15, L[k].zc), (L[k].x0 - 0.2, L[k].zc))
        a((f, L["W2"].zc), (L["W2"].x0 - 0.15, L["W2"].zc))
        a((L["W2"].x1 + 0.15, L["W2"].zc), (L["Q2"].x0 - 0.2, L["W2"].zc))
        a((L["V2"].x1 + 0.15, L["S2"].zc), (L["S2"].x0 - 0.2, L["S2"].zc))
        a((L["S"].x1 + 0.2, L["S"].rz(self.hi)), (L["A"].x0 - 0.2, L["S"].rz(self.hi)))
        a((L["S2"].x1 + 0.2, L["S2"].zc), (L["A2"].x0 - 0.2, L["S2"].zc))
        kx = L["KT"].x0 - 0.5
        segs += [[(L["K"].x1 + 0.2, 0, L["K"].zc), (kx, 0, L["K"].zc)], [(kx, 0, L["K"].zc), (kx, 0, L["KT"].zc)]]
        a((kx, L["KT"].zc), (L["KT"].x0 - 0.15, L["KT"].zc))
        a((L["Vc"].x1 + 0.2, L["O1"].zc), (L["O1"].x0 - 0.2, L["O1"].zc))
        a((L["A2"].x1 + 0.2, L["O2"].zc), (L["O2"].x0 - 0.2, L["O2"].zc))
        a((L["A"].xc, L["A"].z_bot - 1.6), (L["A"].xc, L["Vc"].z_top + 0.3))
        a((L["V"].x1 + 0.2, L["V"].zc), (L["mix"] - 0.6, L["V"].zc))
        cx = L["C"].x0 - 1.0
        segs += [[(L["O1"].x1 + 0.2, 0, L["O1"].zc), (cx, 0, L["O1"].zc)],
                 [(L["O2"].x1 + 0.2, 0, L["O2"].zc), (cx, 0, L["O2"].zc)], [(cx, 0, L["O2"].zc), (cx, 0, zc)]]
        a((cx, zc), (L["C"].x0 - 0.2, zc))
        for p, q in (("C", "WO"), ("WO", "D"), ("X1", "W1"), ("W1", "H"), ("H", "W2f"), ("W2f", "F")):
            a((L[p].x1 + 0.2, zc), (L[q].x0 - 0.2, zc))
        a((L["D"].x1 + 0.2, zc), (L["plus1"][0] - 0.5, zc))
        a((L["plus1"][0] + 0.5, zc), (L["norm1"][0] - 0.75, zc))
        a((L["norm1"][0] + 0.75, zc), (L["X1"].x0 - 0.2, zc))
        a((L["F"].x1 + 0.2, zc), (L["plus2"][0] - 0.5, zc))
        a((L["plus2"][0] + 0.5, zc), (L["norm2"][0] - 0.75, zc))
        a((L["norm2"][0] + 0.75, zc), (L["X2"].x0 - 0.2, zc))
        a((L["X2"].x1 + 0.2, zc), (L["L2"][0] - 0.1, zc))
        a((L["L2"][1] + 0.1, zc), (L["X3"].x0 - 0.2, zc))
        a((L["X3"].x1 + 0.2, L["row"].zc), (L["row"].x0 - 0.2, L["row"].zc), YELLOW)
        a((L["row"].x1 + 0.2, L["row"].zc), (L["WOUT"].x0 - 0.2, L["row"].zc))
        a((L["WOUT"].x1 + 0.2, L["row"].zc), (L["logit"].x0 - 0.2, L["row"].zc))
        a((L["logit"].x1 + 0.2, L["row"].zc), (L["probs_x"] - 2.6, L["row"].zc))
        lines(g, segs, LINE, 1.4)
        # residual lines: X jumps over attention, X1 jumps over feed forward
        r1 = self.top_z - 0.2
        rx = L["X"].xc
        lines(g, [[(rx, 0, L["X"].z_top + 0.1), (rx, 0, r1), (L["plus1"][0], 0, r1)]], RESID, 1.8)
        arrow2d(g, (L["plus1"][0], 0, r1), (L["plus1"][0], 0, L["plus1"][1] + 0.48), RESID, 1.8, 0.18)
        text(g, "residual: X skips attention and is ADDED back", Point3((rx + L["plus1"][0]) / 2, 0, r1 + 0.25),
             0.34, RESID)
        r2 = L["W2f"].z_top + 0.8
        x1c = L["X1"].xc
        lines(g, [[(x1c, 0, L["X1"].z_top + 0.1), (x1c, 0, r2), (L["plus2"][0], 0, r2)]], RESID, 1.8)
        arrow2d(g, (L["plus2"][0], 0, r2), (L["plus2"][0], 0, L["plus2"][1] + 0.48), RESID, 1.8, 0.18)
        text(g, "residual: X1 skips feed forward", Point3((x1c + L["plus2"][0]) / 2, 0, r2 + 0.25), 0.34, RESID)
        self.res_z = (r1, r2)

    def overview(self):
        return self.go(FX - 1.0, self.L["end_x"], self.bot_z, self.top_z)

    # ================================================================== 1 embedding
    def s_embed(self):
        L, n, T = self.L, self.n, self.T
        r = self.root()
        g_map = self.keep(self.build_map())
        E = self.engine.m.w["E"]
        ids = [int(t) for t in T["ids"]]
        tb = L["table"]
        words = r.attachNewNode("words")
        for i, t in enumerate(self.ttoks):
            c = YELLOW if i == self.hi else WHITE
            text(words, disp(t), Point3(L["words_x"], 0, L["E"].rz(i) - 0.11), 0.3, c, align=TextNode.ARight)
            text(words, str(ids[i]), Point3(L["ids_x"], 0, L["E"].rz(i) - 0.1), 0.26, YELLOW if i == self.hi else GREY)
        text(words, "token   ID", Point3(L["words_x"] + 0.1, 0, L["E"].z_top + 0.4), 0.28, GREY)
        words.hide()
        self.keep(words)
        table = self.keep(self.mat(E, tb, scale=self.rs, hero=False, gapx=0.0))
        marks = r.attachNewNode("marks")
        for i, tid in enumerate(ids):
            zr = tb.z_top - (tid + 0.5) * tb.rh
            c = YELLOW if i == self.hi else (0.8, 0.8, 0.85, 1)
            rect(marks, tb.x0 - 0.06, zr - 0.05, tb.x1 + 0.06, zr + 0.05, c, 1.6 if i == self.hi else 1.0, y=-0.02)
        marks.hide()
        self.keep(marks)
        rows = self.rows_of(T["emb"], L["E"], scale=self.rs)
        flies = Parallel()
        for i, tid in enumerate(ids):
            zr = tb.z_top - tid * tb.rh + RH / 2 - tb.rh / 2
            nd = self.row_node(E[tid], CW, ai=(i == self.hi))
            flies.append(Sequence(Wait(0.16 * i), Func(nd.setPos, tb.x0, 0, zr), Func(nd.setSx, 1.0), Func(nd.show),
                                  Parallel(LerpPosInterval(nd, 0.7, Point3(L["E"].x0, 0, L["E"].z_top - i * RH),
                                                           blendType="easeInOut"),
                                           LerpFunc(nd.setSx, 0.7, 1.0, CD / CW)),
                                  Func(nd.hide), show_fade(rows[i], 0.15)))
        tag = self.ai_tag(L["E"])
        seq = Sequence(self.mapkey("embed"), self.overview(), show_fade(g_map, 1.2), Wait(2.4),
                       self.go(FX - 1.0, L["X"].x1 + 1.0, L["P"].z_bot - 1.2, tb.z_top + 0.6),
                       show_fade(words, 0.5), show_fade(table, 0.5), show_fade(marks, 0.4), flies,
                       show_fade(tag, 0.3), self.track(T["emb"][self.hi], "token embedding"))
        return ("THE MAP: everything the Transformer does, from left to right. Every token is one ROW of numbers and "
                "keeps its row the whole way; the last row (yellow) is <ai>, whose row will predict the next word.  "
                "STEP 1, EMBEDDING: each token ID picks its row of {} numbers from a learned table ({} x {}). "
                "Stacked up, the rows form matrix E ({} x {}). Red = positive, blue = negative."
                .format(self.d, self.V, self.d, n, self.d), seq)

    # ================================================================== 2 positional encoding
    def s_position(self):
        L, T = self.L, self.T
        P = self.keep(self.mat(T["pe"], L["P"], scale=self.rs))
        X = self.rows_of(T["x0"], L["X"], scale=self.rs)
        tag = self.ai_tag(L["X"])
        seq = Sequence(self.mapkey("pos"),
                       self.go(L["E"].x0 - 1.0, L["X"].x1 + 1.0, L["P"].z_bot - 1.2, L["E"].z_top + 1.0),
                       show_fade(P, 0.6), Wait(0.6), self.mapkey("pos", "embed"),
                       Parallel(*[Sequence(Wait(0.07 * i), show_fade(nd, 0.3)) for i, nd in enumerate(X)]),
                       show_fade(tag, 0.3), self.track(T["x0"][self.hi], "added position"))
        return ("STEP 2, POSITION: so far the model would not know the ORDER of the rows. Each position gets a fixed "
                "pattern of numbers (P, made of sine waves) and it is simply ADDED, number by number: X = E + P. "
                "X ({} x {}) goes into the attention block.".format(self.n, self.d), seq)

    # ================================================================== 3 Q, K, V  (one multiplication worked out)
    def s_qkv(self):
        L, T, n = self.L, self.T, self.n
        H = T["layers"][0]["heads"][0]
        mats, rowsets, tags = [], [], []
        for nm in ("Q", "K", "V"):
            mats.append(self.keep(self.mat(H["W" + nm.lower()], L["W" + nm], hero=False, gapx=0.06)))
            rowsets.append(self.rows_of(H[nm.lower()], L[nm], scale=sc_of(H[nm.lower()])))
            tags.append(self.ai_tag(L[nm]))
        xr = T["x0"][self.hi]
        wq = H["Wq"][:, 0]
        q0 = float(xr @ wq)
        bx, bq, wb = L["X"], L["Q"], L["WQ"]
        hz = bx.rz(self.hi)
        # the worked multiplication: <ai>'s row of X stands up next to W_Q, so its 32 numbers sit
        # beside the 32 numbers of W_Q's first column; multiply pair by pair and add up = first number of Q
        hl_x = self.root().attachNewNode("hlx")
        rect(hl_x, bx.x0 - 0.16, hz - RH / 2 - 0.06, bx.x1 + 0.16, hz + RH / 2 + 0.06, WHITE, 2.6, y=-0.03)
        hl_x.hide()
        cp = self.row_node(xr, CD, ai=False)
        cp_to = (wb.x0 - 0.35, 0, wb.z_top + (RH - CD) * 0)       # after turning, the row hangs down-left
        demo = self.root().attachNewNode("demo")
        rect(demo, wb.x0 - 0.06, wb.z_bot - 0.06, wb.x0 + CD + 0.06, wb.z_top + 0.06, WHITE, 2.6, y=-0.03)
        for k in range(0, self.d, 4):
            lines(demo, [[(wb.x0 - 0.33, -0.03, wb.z_top - (k + 0.5) * CD), (wb.x0 + 0.02, -0.03,
                                                                                 wb.z_top - (k + 0.5) * CD)]],
                  (1, 1, 1, 0.5), 1.0)
        terms = " + ".join("({:.1f})·({:.1f})".format(xr[k], wq[k]) for k in range(3))
        tx = wb.x0 - 1.4
        fill(demo, tx - 0.3, self.zq + 1.0, tx + 17.5, self.zq + 2.9, (0.02, 0.02, 0.03, 1), 0.92, y=-0.05)
        text(demo, "<ai>'s row of X  ·  first column of W_Q  =  first number of <ai>'s row of Q",
             Point3(tx, -0.06, self.zq + 2.2), 0.42, WHITE, Fonts.symbol, align=TextNode.ALeft)
        text(demo, "{} + ...   ({} products added up)  =  {:.2f}".format(terms, self.d, q0),
             Point3(tx, -0.06, self.zq + 1.4), 0.4, YELLOW, Fonts.symbol, align=TextNode.ALeft)
        rect(demo, bq.x0 - 0.06, bq.rz(self.hi) - RH / 2, bq.x0 + CD + 0.06, bq.rz(self.hi) + RH / 2, WHITE, 2.6,
             y=-0.03)
        demo.hide()
        eqs = [self.note("X · W_{0} = {0}\n({1} × {2}) · ({2} × {3}) = ({1} × {3})".format(nm, n, self.d, self.dh),
                         L[nm].xc, L[nm].z_bot - 1.95, 0.34, WHITE, TextNode.ACenter) for nm in ("Q", "K", "V")]
        lane_q = (wb.x0 - 2.6, bq.x1 + 7.0, bq.z_bot - 2.8, self.zq + 3.4)
        seq = Sequence(self.mapkey("attn"),
                       self.go(bx.x0 - 1.0, bq.x1 + 2.0, bx.z_bot - 1.0, self.zq + 1.0),
                       show_fade(mats[0], 0.4), show_fade(hl_x, 0.3), Wait(0.5),
                       Func(cp.setPos, bx.x0, 0, bx.z_top - self.hi * RH), Func(cp.setHpr, 0, 0, 0), Func(cp.show),
                       Parallel(LerpPosInterval(cp, 1.6, Point3(*cp_to), blendType="easeInOut"),
                                LerpHprInterval(cp, 1.6, (0, 0, 90))),
                       fade_out(hl_x, 0.3), Func(hl_x.hide),
                       self.go(*lane_q), show_fade(demo, 0.5), Wait(3.0), show_fade(rowsets[0][self.hi], 0.4),
                       Wait(1.0), fade_out(demo, 0.4), Func(demo.hide), fade_out(cp, 0.3), Func(cp.hide),
                       Func(cp.setColorScale, 1, 1, 1, 1),
                       show_fade(eqs[0], 0.3),
                       self.sweep([nd for i, nd in enumerate(rowsets[0]) if i != self.hi], 0.8),
                       show_fade(tags[0], 0.2), Wait(0.6),
                       self.go(bx.x0 - 1.0, bq.x1 + 6.0, L["V"].z_bot - 2.6, self.zq + 1.0),
                       show_fade(mats[1], 0.3), self.sweep(rowsets[1], 0.7), show_fade(tags[1], 0.2),
                       show_fade(eqs[1], 0.2),
                       show_fade(mats[2], 0.3), self.sweep(rowsets[2], 0.7), show_fade(tags[2], 0.2),
                       show_fade(eqs[2], 0.2))
        return ("STEP 3, Q K V: three ordinary matrix multiplications with learned matrices: X · W_Q = Q, X · W_K = "
                "K, X · W_V = V (each {} x {}). One number is worked out on top: <ai>'s row of X times the first "
                "column of W_Q. The idea: Q = what each token is LOOKING FOR, K = what it OFFERS, V = what it will "
                "PASS ON.".format(n, self.dh), seq)

    # ================================================================== 4 scores = Q K^T
    def s_scores(self):
        L, T, n = self.L, self.T, self.n
        H = T["layers"][0]["heads"][0]
        S, KT = L["S"], L["KT"]
        ks = sc_of(H["k"])
        kt = self.keep(self.mat(H["k"].T, KT, scale=ks, hero=False, gapx=0.28))
        flies = Parallel()
        for j in range(n):
            nd = self.row_node(H["k"][j], CD, scale=ks, ai=False)
            x_to = KT.cx(j) + RH / 2                     # after turning 90 degrees the row hangs down-left
            flies.append(Sequence(Wait(0.05 * j), Func(nd.setPos, L["K"].x0, 0, L["K"].z_top - j * RH),
                                  Func(nd.setHpr, 0, 0, 0), Func(nd.show),
                                  Parallel(LerpPosInterval(nd, 1.0, Point3(x_to, 0, KT.z_top), blendType="easeInOut"),
                                           LerpHprInterval(nd, 1.0, (0, 0, 90))),
                                  Func(nd.hide)))
        heads = self.root().attachNewNode("kt_heads")
        for j, t in enumerate(self.ttoks):
            text(heads, disp(t)[:5], Point3(KT.cx(j), 0, KT.z_top + 0.25), 0.24, YELLOW if j == self.hi else GREY)
        text(heads, "Kᵀ", Point3(KT.x0 - 1.1, 0, KT.zc - 0.15), TS, WHITE, Fonts.symbol)
        heads.hide()
        self.keep(heads)
        raw = H["scaled"]
        sc = sc_of(raw[np.isfinite(raw)])
        masked = np.where(np.triu(np.ones((n, n)), 1) > 0, np.nan, raw)
        grid_raw = self.mat(raw, S, scale=sc, gapx=0.08)
        grid_m = self.keep(self.mat(masked, S, scale=sc, gapx=0.08, nan=MASKED))
        nums = self.root().attachNewNode("nums")
        for j in range(n):
            text(nums, "{:.1f}".format(raw[self.hi, j]), Point3(S.cx(j), 0, S.rz(self.hi) - 0.08), 0.2, WHITE)
        nums.hide()
        self.keep(nums)
        j0 = int(np.argmax(H["att"][self.hi]))
        demo = self.root().attachNewNode("demo")
        rect(demo, L["Q"].x0 - 0.12, S.rz(self.hi) - RH / 2, L["Q"].x1 + 0.12, S.rz(self.hi) + RH / 2, WHITE, 2.6,
             y=-0.03)
        rect(demo, KT.x0 + j0 * RH, KT.z_bot - 0.05, KT.x0 + (j0 + 1) * RH, KT.z_top + 0.05, WHITE, 2.6, y=-0.03)
        rect(demo, S.x0 + j0 * RH, S.rz(self.hi) - RH / 2, S.x0 + (j0 + 1) * RH, S.rz(self.hi) + RH / 2, WHITE, 2.8,
             y=-0.04)
        demo.hide()
        nx = S.x1 + 0.6
        info = self.note("Q · Kᵀ = scores\n({} × {}) · ({} × {}) = ({} × {})".format(n, self.dh, self.dh, n, n, n),
                         nx, S.z_top - 0.35, 0.34, WHITE)
        info2 = self.note("cell (i, j) = how well token i's\nquestion (Q) matches token j (K).\n"
                          "Then all ÷ {}{} = {:.0f}".format(SQ, self.dh, np.sqrt(self.dh)), nx, S.z_top - 1.6, 0.32)
        mask_t = self.note("MASK: no looking at LATER\ntokens -> grey (−∞)", nx, S.z_top - 3.3, 0.32, ORANGE)
        seq = Sequence(self.mapkey("attn"), self.go(L["K"].x0 - 0.8, S.x1 + 6.6, L["K"].zc, KT.z_top + 1.0),
                       flies, show_fade(kt, 0.3), show_fade(heads, 0.3), show_fade(info, 0.3), show_fade(demo, 0.3),
                       Wait(1.2), show_fade(grid_raw, 0.8), show_fade(nums, 0.3), show_fade(info2, 0.3), Wait(1.0),
                       fade_out(demo, 0.3), Func(demo.hide),
                       show_fade(grid_m, 0.5), Func(grid_raw.hide), show_fade(mask_t, 0.4))
        return ("STEP 4, SCORES: Q · Kᵀ is again a matrix multiplication. The K rows turn into columns (Kᵀ, on "
                "top), so cell (i, j) = row i of Q · column j of Kᵀ = how well token i's question matches token j. "
                "The yellow row is <ai> compared with every token. Then the MASK: no token may look at tokens that "
                "come AFTER it (grey). <ai> is last, so it sees everything.", seq)

    # ================================================================== 5 softmax -> A
    def s_softmax(self):
        L, T, n = self.L, self.T, self.n
        H = T["layers"][0]["heads"][0]
        A = L["A"]
        att = np.where(np.triu(np.ones((n, n)), 1) > 0, np.nan, H["att"])
        grid = self.keep(self.mat(att, A, colors=gray_color, gapx=0.08, nan=MASKED))
        nums = self.root().attachNewNode("anums")
        for j in range(n):
            v = H["att"][self.hi, j]
            text(nums, "{:.2f}".format(v), Point3(A.cx(j), 0, A.rz(self.hi) - 0.08), 0.18,
                 (0, 0, 0, 1) if v > 0.45 else WHITE)
        for j, t in enumerate(self.ttoks):
            text(nums, disp(t)[:5], Point3(A.cx(j), 0, A.z_top + 0.25), 0.24, YELLOW if j == self.hi else GREY)
        nums.hide()
        self.keep(nums)
        info = self.note("softmax, row by row:\nall weights ≥ 0,\nevery row adds up to 1\n\nbright = big weight",
                         A.x1 + 0.6, A.z_top - 0.4)
        seq = Sequence(self.mapkey("attn"), self.go(L["S"].x0 - 0.6, A.x1 + 6.0, A.z_bot - 1.6, A.z_top + 1.4),
                       show_fade(grid, 0.8), show_fade(nums, 0.4), show_fade(info, 0.3))
        return ("STEP 5, WEIGHTS: softmax turns every row of scores into WEIGHTS (e^score, then divide by the row's "
                "sum): all positive, every row adds up to 1, bright = big. The yellow row says how much <ai> pays "
                "attention to each token - most to {}.".format(self.top_attn(H["att"], 2)), seq)

    # ================================================================== 6 A . V  (weighted sum)
    def s_weighted(self):
        L, T, n = self.L, self.T, self.n
        H = T["layers"][0]["heads"][0]
        att, Vm = H["att"], H["v"]
        vs = sc_of(Vm)
        A, Vc, O1 = L["A"], L["Vc"], L["O1"]
        w = att[self.hi]
        wcol = self.root().attachNewNode("wcol")
        for j in range(n):
            text(wcol, "× {:.2f}".format(w[j]), Point3(L["mix"], 0, Vc.rz(j) - 0.1), 0.28,
                 YELLOW if w[j] > 0.1 else GREY)
        wcol.hide()
        copies, moves = [], Parallel()
        for j in range(n):
            nd = self.row_node(Vm[j], CD, scale=vs, ai=False)
            copies.append(nd)
            moves.append(self.fly(nd, (L["V"].x0, 0, L["V"].z_top - j * RH), (Vc.x0, 0, Vc.z_top - j * RH), 1.0,
                                  0.03 * j, hide_end=False))
        bright = [0.12 + 0.88 * min(1.0, x * 2.5) for x in w]
        dims = Parallel(*[LerpColorScaleInterval(copies[j], 0.6, (b, b, b, 1)) for j, b in enumerate(bright)])
        merge = Parallel(*[Sequence(LerpPosInterval(copies[j], 0.9, Point3(O1.x0, 0, O1.z_top - self.hi * RH),
                                                    blendType="easeIn"), Func(copies[j].hide),
                                    Func(copies[j].setColorScale, 1, 1, 1, 1)) for j in range(n)])
        rows = self.rows_of(H["out"], O1, scale=sc_of(H["out"]))
        tag = self.ai_tag(O1)
        info = self.note("A · V = head output\n({} × {}) · ({} × {}) = ({} × {})".format(n, n, n, self.dh, n, self.dh),
                         O1.x1 + 0.5, O1.z_top - 0.4, 0.34, WHITE)
        hl = self.root().attachNewNode("ahl")
        rect(hl, A.x0 - 0.12, A.rz(self.hi) - RH / 2 - 0.05, A.x1 + 0.12, A.rz(self.hi) + RH / 2 + 0.05, WHITE, 2.8,
             y=-0.03)
        hl.hide()
        top2 = np.argsort(-w)[:2]
        seq = Sequence(self.mapkey("attn"),
                       self.go(L["V"].x0 - 0.6, O1.x1 + 6.0, Vc.z_bot - 1.6, A.z_top + 0.8),
                       show_fade(hl, 0.3), Wait(0.8), show_fade(wcol, 0.5),
                       self.go(L["V"].x0 - 0.6, O1.x1 + 6.0, Vc.z_bot - 1.6, Vc.z_top + 2.4), moves, Wait(0.3), dims, Wait(0.8), merge,
                       show_fade(rows[self.hi], 0.3), Wait(0.5), fade_out(hl, 0.3), Func(hl.hide),
                       fade_out(wcol, 0.3), Func(wcol.hide),
                       self.sweep([nd for i, nd in enumerate(rows) if i != self.hi], 0.8), show_fade(tag, 0.2),
                       show_fade(info, 0.3))
        return ("STEP 6, MIX: A · V is a matrix multiplication again, and what it does is MIX the V rows: <ai>'s new "
                "row = (its weight for token 1) × V row 1 + (weight 2) × V row 2 + ... Strong weights stay bright, "
                "weak ones fade, then all are added up. So <ai> now carries information from '{}' and '{}'. Every "
                "row does the same with its own weights.".format(disp(self.ttoks[int(top2[0])]),
                                                                 disp(self.ttoks[int(top2[1])])), seq)

    # ================================================================== 7 head 2
    def s_head2(self):
        L, T, n = self.L, self.T, self.n
        H = T["layers"][0]["heads"][1]
        H1 = T["layers"][0]["heads"][0]
        parts = [self.keep(self.mat(H[nm[0].lower()], L[nm], scale=sc_of(H[nm[0].lower()]), gapx=0.0))
                 for nm in ("Q2", "K2", "V2")]
        mask = np.triu(np.ones((n, n)), 1) > 0
        s2 = self.keep(self.mat(np.where(mask, np.nan, H["scaled"]), L["S2"], scale=sc_of(H["scaled"]), gapx=0.08,
                                nan=MASKED))
        a2 = self.keep(self.mat(np.where(mask, np.nan, H["att"]), L["A2"], colors=gray_color, gapx=0.08,
                                nan=MASKED))
        o2 = self.keep(self.mat(H["out"], L["O2"], scale=sc_of(H["out"])))
        tag = self.ai_tag(L["O2"])
        seq = Sequence(self.mapkey("attn"),
                       self.go(L["fan"] - 1.0, L["O2"].x1 + 1.5, L["O2"].z_bot - 1.6, L["V"].z_bot + 0.5),
                       self.sweep(parts, 0.6), Wait(0.3), show_fade(s2, 0.5), Wait(0.3), show_fade(a2, 0.5), Wait(0.3),
                       show_fade(o2, 0.5), show_fade(tag, 0.2))
        return ("STEP 7, HEAD 2: attention is done twice side by side, with a second set of W_Q, W_K, W_V - the same "
                "steps 3 to 6. Each head can look for a different kind of relation. <ai> - head 1: {};  head 2: {}."
                .format(self.top_attn(H1["att"], 2), self.top_attn(H["att"], 2)), seq)

    # ================================================================== 8 concat x W_O = delta X
    def s_concat(self):
        L, T, n = self.L, self.T, self.n
        LT = T["layers"][0]
        C, D = L["C"], L["D"]
        o1, o2 = LT["heads"][0]["out"], LT["heads"][1]["out"]
        cat = np.hstack([o1, o2])
        cs = sc_of(cat)
        cmat = self.keep(self.mat(cat, C, scale=cs))
        flies = Parallel()
        for i in range(n):
            a = self.row_node(o1[i], CD, scale=cs, ai=False)
            b = self.row_node(o2[i], CD, scale=cs, ai=False)
            flies.append(self.fly(a, (L["O1"].x0, 0, L["O1"].z_top - i * RH), (C.x0, 0, C.z_top - i * RH), 1.2,
                                  0.03 * i))
            flies.append(self.fly(b, (L["O2"].x0, 0, L["O2"].z_top - i * RH),
                                  (C.x0 + self.dh * CD, 0, C.z_top - i * RH), 1.4, 0.3 + 0.03 * i))
        wo = self.keep(self.mat(LT["Wo"], L["WO"], hero=False, gapx=0.06))
        rows = self.rows_of(LT["delta"], D, scale=self.rs)
        tags = [self.ai_tag(b) for b in (C, D)]
        info = self.note("head 1 | head 2", C.xc, C.z_top + 0.4, 0.34, GREY, TextNode.ACenter)
        info2 = self.note("concat · W_O = ΔX    ({} × {}) · ({} × {}) = ({} × {})".format(n, self.d, self.d, self.d, n,
                                                                                         self.d),
                          L["WO"].xc, C.z_bot - 2.0, 0.36, WHITE, TextNode.ACenter)
        seq = Sequence(self.mapkey("attn"),
                       self.go(L["O1"].x0 - 1.0, D.x1 + 1.0, L["O2"].z_bot - 1.4, C.z_top + 1.4),
                       flies, self.go(C.x0 - 1.5, D.x1 + 1.5, C.z_bot - 2.8, C.z_top + 1.4), show_fade(cmat, 0.3), show_fade(tags[0], 0.2), show_fade(info, 0.3), Wait(0.3),
                       show_fade(wo, 0.4), self.sweep(rows, 0.8), show_fade(tags[1], 0.2), show_fade(info2, 0.3))
        return ("STEP 8: the two head outputs are put side by side (16 + 16 = {} numbers per row) and multiplied by "
                "one more learned matrix W_O. The result ΔX is the CHANGE that attention wants to make to every "
                "token's row.".format(self.d), seq)

    # ================================================================== residual travel helper
    def _residual(self, src_box, M, rz, plus):
        """Copies of all rows of M climb up, travel along the red line, and drop into the + circle."""
        n = self.n
        px, pz = plus
        trav = [self.row_node(M[i], CD, ai=(i == self.hi)) for i in range(n)]
        lift = lambda i: rz + 0.9 + (n - 1 - i) * 0.07  # noqa: E731
        up = Parallel(*[self.fly(nd, (src_box.x0, 0, src_box.z_top - i * RH), (src_box.x0, 0, lift(i)), 0.8,
                                 0.02 * i, hide_end=False) for i, nd in enumerate(trav)])
        over = Parallel(*[LerpPosInterval(nd, 2.4, Point3(px - self.d * CD / 2, 0, lift(i)), blendType="easeInOut")
                          for i, nd in enumerate(trav)])
        down = Parallel(*[Sequence(LerpPosInterval(nd, 0.7, Point3(px - self.d * CD / 2, 0, pz + 0.3),
                                                   blendType="easeIn"), Func(nd.hide)) for nd in trav])
        return up, over, down

    # ================================================================== 9 add & norm
    def s_add1(self):
        L, T, n = self.L, self.T, self.n
        LT = T["layers"][0]
        X1 = L["X1"]
        r1 = self.res_z[0]
        up, over, down = self._residual(L["X"], T["x0"], r1, L["plus1"])
        rows = self.rows_of(LT["x1"], X1, scale=self.rs)
        tag = self.ai_tag(X1)
        info = self.note("X + ΔX, then NORM each row:\nminus its mean, ÷ its spread,\n× gain + bias (learned)",
                         L["norm1"][0], L["D"].z_bot - 2.0, 0.34, GREY, TextNode.ACenter)
        seq = Sequence(self.mapkey("add1"),
                       self.go(L["X"].x0 - 1.0, X1.x1 + 1.0, L["D"].z_bot - 3.0, r1 + 2.4),
                       up, over, self.go(L["D"].x0 - 1.5, X1.x1 + 1.5, L["D"].z_bot - 3.4, r1 + 2.4),
                       down, self.go(L["D"].x0 - 1.5, X1.x1 + 1.5, L["D"].z_bot - 3.6, L["D"].z_top + 1.6),
                       show_fade(info, 0.3), self.sweep(rows, 0.8), show_fade(tag, 0.2),
                       self.track(LT["x1"][self.hi], "layer 1: add & norm"))
        return ("STEP 9, ADD & NORM: X itself jumps over the whole attention block (red line, the 'residual') and "
                "is ADDED to ΔX: X + ΔX. So attention only adds a correction and nothing is lost. Then every row is "
                "normalized (mean 0, spread 1) to keep the numbers in a stable range. Result: X1 ({} x {})."
                .format(n, self.d), seq)

    # ================================================================== 10 feed forward
    def s_ffn(self):
        L, T, n = self.L, self.T, self.n
        LT = T["layers"][0]
        w = self.engine.m.w
        W1 = self.keep(self.mat(w["l0.W1"], L["W1"], hero=False, gapx=0.0))
        W2 = self.keep(self.mat(w["l0.W2"], L["W2f"], hero=False, gapx=0.06))
        pre, hid = LT["hid_pre"], LT["hid"]
        hs = sc_of(pre)
        Hpre = self.mat(pre, L["H"], scale=hs, gapx=0.0)
        Hrelu = self.keep(self.mat(hid, L["H"], scale=hs, gapx=0.0))
        rows = self.rows_of(LT["ff"], L["F"], scale=self.rs)
        tags = [self.ai_tag(b) for b in (L["H"], L["F"])]
        off = int((pre[self.hi] <= 0).sum())
        relu = self.note("ReLU: every negative number -> 0\n(<ai>: {} of {} switched off)".format(off, self.ff),
                         L["H"].xc, L["H"].z_bot - 2.0, 0.36, ORANGE, TextNode.ACenter)
        info = self.note("X1 · W1\n({} × {}) · ({} × {}) = ({} × {})".format(n, self.d, self.d, self.ff, n, self.ff),
                         L["W1"].xc, L["W1"].z_bot - 1.9, 0.34, WHITE, TextNode.ACenter)
        info2 = self.note("hidden · W2\n({} × {}) · ({} × {}) = ({} × {})".format(n, self.ff, self.ff, self.d, n,
                                                                                 self.d),
                          L["F"].xc, L["F"].z_bot - 2.0, 0.34, WHITE, TextNode.ACenter)
        seq = Sequence(self.mapkey("ffn"),
                       self.go(L["X1"].x0 - 1.0, L["H"].x1 + 1.0, L["H"].z_bot - 3.4, L["X1"].z_top + 2.0),
                       show_fade(W1, 0.4), show_fade(Hpre, 0.6), show_fade(tags[0], 0.2), show_fade(info, 0.3),
                       Wait(0.8), show_fade(relu, 0.3), Wait(0.3), show_fade(Hrelu, 0.6), Func(Hpre.hide), Wait(0.8),
                       self.go(L["H"].x0 - 1.0, L["F"].x1 + 1.5, L["W2f"].z_bot - 1.0, L["W2f"].z_top + 0.8),
                       show_fade(W2, 0.4), self.sweep(rows, 0.8), show_fade(tags[1], 0.2), show_fade(info2, 0.3))
        return ("STEP 10, FEED FORWARD: two more matrix multiplications: X1 · W1 gives {} numbers per token, ReLU sets "
                "every negative one to 0 (dark), then · W2 brings it back to {}. Each row is processed ON ITS OWN - "
                "no mixing between rows. Attention = tokens talk to each other; feed forward = each token thinks "
                "for itself.".format(self.ff, self.d), seq)

    # ================================================================== 11 add & norm -> layer 1 output
    def s_add2(self):
        L, T, n = self.L, self.T, self.n
        LT = T["layers"][0]
        X2 = L["X2"]
        r2 = self.res_z[1]
        up, over, down = self._residual(L["X1"], LT["x1"], r2, L["plus2"])
        rows = self.rows_of(LT["x_out"], X2, scale=self.rs)
        tag = self.ai_tag(X2)
        seq = Sequence(self.mapkey("add2"), self.go(L["X1"].x0 - 1.0, X2.x1 + 1.0, X2.z_bot - 2.0, r2 + 2.4),
                       up, over, down, self.go(L["F"].x0 - 1.5, X2.x1 + 1.5, X2.z_bot - 2.0, X2.z_top + 1.6),
                       self.sweep(rows, 0.8), show_fade(tag, 0.2),
                       self.track(LT["x_out"][self.hi], "end of layer 1"))
        return ("STEP 11, ADD & NORM again: X1 jumps over the feed forward block and is added to F, then every row "
                "is normalized. X2 is the output of LAYER 1 - the same shape as X ({} x {}), but now every row knows "
                "something about the other tokens.".format(n, self.d), seq)

    # ================================================================== 12 layer 2
    def s_layer2(self):
        L, T = self.L, self.T
        LT = T["layers"][1]
        rows = self.rows_of(LT["x_out"], L["X3"], scale=self.rs)
        tag = self.ai_tag(L["X3"])
        x0, x1 = L["L2"]
        box = self.root().attachNewNode("l2flash")
        fill(box, x0, L["X2"].zc - 3.0, x1, L["X2"].zc + 3.0, (0.45, 0.75, 0.45, 1), 0.18)
        box.hide()
        seq = Sequence(Func(self.app.ui.arch.set_layer, "layer 2 of 2"), self.mapkey("attn"),
                       self.go(L["X2"].x0 - 1.0, L["X3"].x1 + 1.0, L["X2"].z_bot - 2.0, L["X2"].z_top + 3.0),
                       show_fade(box, 0.4), self.mapkey("add1"), Wait(0.4), self.mapkey("ffn"), Wait(0.4),
                       self.mapkey("add2"), Wait(0.3), fade_out(box, 0.4), Func(box.hide), self.sweep(rows, 0.8),
                       show_fade(tag, 0.2), self.track(LT["x_out"][self.hi], "end of layer 2"),
                       Func(self.app.ui.arch.set_layer, "x 2 layers"))
        return ("STEP 12, LAYER 2: steps 3-11 once more, with its own learned matrices. In layer 2, <ai> looks most "
                "at: {}. Real models stack 30-100 such layers.".format(self.top_attn(LT["heads"][0]["att"], 2)), seq)

    # ================================================================== 13 output
    def s_output(self):
        L, T = self.L, self.T
        m = self.engine.m
        xl = T["x_final"][-1]
        logits, probs = T["logits"], T["probs"]
        row, WO, lg, X3 = L["row"], L["WOUT"], L["logit"], L["X3"]
        nd = self.row_node(xl, CD)
        r = self.keep(self.mat(xl[None, :], row, scale=self.rs, hero=False))
        rect(r, row.x0 - 0.08, row.z_bot + 0.02, row.x1 + 0.08, row.z_top - 0.02, YELLOW, 2.4, y=-0.02)
        wout = self.keep(self.mat(m.w["Wout"], WO, hero=False, gapx=0.0))
        lmat = self.keep(self.mat(logits[None, :], lg, scale=sc_of(logits), hero=False, gapx=0.0))
        best = int(np.argmax(logits))
        mark = self.root().attachNewNode("best")
        rect(mark, lg.x0 + best * lg.cw - 0.06, lg.z_bot - 0.12, lg.x0 + (best + 1) * lg.cw + 0.06, lg.z_top + 0.12,
             YELLOW, 2.0, y=-0.03)
        text(mark, "'{}'".format(disp(m.vocab[best])), Point3(lg.x0 + (best + 0.5) * lg.cw, 0, lg.z_top + 0.35), 0.3,
             YELLOW)
        mark.hide()
        self.keep(mark)
        top = np.argsort(-probs)[:5]
        pg = self.root().attachNewNode("probs")
        px = L["probs_x"]
        text(pg, "softmax -> probabilities", Point3(px - 2.2, 0, row.z_top + 1.4), 0.42, GREY, align=TextNode.ALeft)
        for k, j in enumerate(top):
            z = row.zc + 0.5 - k * 0.9
            text(pg, disp(m.vocab[j]), Point3(px - 0.2, 0, z - 0.15), 0.44, WHITE, align=TextNode.ARight)
            wdt = max(0.03, 5.0 * probs[j])
            fill(pg, px, z - 0.3, px + wdt, z + 0.3, YELLOW if k == 0 else BLUE, 0.9, y=0)
            text(pg, "{:.1%}".format(probs[j]) if probs[j] >= 0.001 else "<0.1%", Point3(px + wdt + 0.2, 0, z - 0.13),
                 0.34, GREY, align=TextNode.ALeft)
        pg.hide()
        self.keep(pg)
        info = self.note("only <ai>'s row:  (1 × {}) · ({} × {}) = (1 × {})\none score for every token the model knows"
                         .format(self.d, self.d, self.V, self.V), WO.xc, WO.z_bot - 1.9, 0.36, WHITE, TextNode.ACenter)
        hz = X3.z_top - self.hi * RH
        seq = Sequence(Func(self.app.ui.arch.set_layer, "x 2 layers"), self.mapkey("linear"),
                       self.go(X3.x0 - 1.0, lg.x1 + 1.0, WO.z_bot - 3.0, WO.z_top + 1.6),
                       self.fly(nd, (X3.x0, 0, hz), (row.x0, 0, row.z_top), 1.2), show_fade(r, 0.2),
                       show_fade(wout, 0.5), Wait(0.4), show_fade(lmat, 0.6), show_fade(info, 0.3), Wait(1.2),
                       self.go(lg.x0 - 1.0, px + 8.0, WO.z_bot - 2.0, WO.z_top + 1.6),
                       show_fade(mark, 0.3), self.mapkey("softmax", "output"), Wait(0.3), show_fade(pg, 0.5),
                       self.track(xl, "used for the prediction"))
        extra = (" (Almost 100%: this tiny model learned these few conversations by heart - big models are much less "
                 "certain.)" if probs[top[0]] > 0.99 else "")
        return ("STEP 13, OUTPUT: only <ai>'s row - the yellow row we followed all the way - is used now. One last "
                "matrix multiplication with W_out gives one score for each of the {} tokens the model knows; softmax "
                "turns the scores into probabilities. Winner: '{}' ({:.1%}).".format(
                    self.V, disp(m.vocab[int(top[0])]), probs[top[0]]) + extra, seq)

    # ================================================================== deep dive OFF
    def s_quick(self):
        """Build steps 3-12 at once and fly the camera along the finished strip."""
        start = len(self.persist)
        for f in (self.s_qkv, self.s_scores, self.s_softmax, self.s_weighted, self.s_head2, self.s_concat,
                  self.s_add1, self.s_ffn, self.s_add2, self.s_layer2):
            f()
        nodes = self.persist[start:]
        L = self.L
        cams = [(L["X"].x0 - 1, L["O1"].x1 + 2), (L["O1"].x0 - 1, L["X1"].x1 + 1), (L["X1"].x0 - 1, L["X2"].x1 + 1),
                (L["X2"].x0 - 1, L["X3"].x1 + 1)]

        def show_all():
            for nd in nodes:
                nd.show()
                nd.setColorScale(1, 1, 1, 1)
        seq = Sequence(self.mapkey("attn"), Func(show_all),
                       *[Sequence(self.go(a, b, self.bot_z, self.top_z), Wait(1.8)) for a, b in cams])
        seq.append(self.track(self.T["layers"][1]["x_out"][self.hi], "end of layer 2"))
        return ("Inside the Transformer (deep dive OFF - press D to see every step): attention lets the token rows "
                "exchange information, feed forward processes each row, and that twice (2 layers). Follow the "
                "yellow <ai> row.", seq)
