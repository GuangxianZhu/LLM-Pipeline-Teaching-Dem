# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Inside the Transformer, step by step, with the REAL numbers of the tiny model.

Every step returns (caption, interval) and names the block of the architecture map it belongs to.
Matrices are drawn as heatmaps: red = positive, blue = negative, dark = near zero.
A token's vector is drawn as a COLUMN of coloured cells (all 32 numbers, nothing hidden).
"""
import numpy as np
from direct.interval.IntervalGlobal import (Func, LerpColorScaleInterval, LerpFunc, LerpPosInterval, Parallel,
                                            Sequence, Wait)
from panda3d.core import Point3, TextNode, Vec3

from kit import (BLUE, DIM, GREEN, GREY, ORANGE, RED, TOKEN_COLORS, WHITE, YELLOW, Arrow, Fonts, disc,
                 fade_in, fade_out, fill, heatmap, lines, rect, shape_label, text, token_box, vlabel)

ROW_Z = -40.0          # all transformer stages live on this row of the board
STAGE_DX = 40.0
CS = 0.95              # spacing between token columns
CH = 0.17              # height of one cell in a 32-number column
SQ = "√"


def disp(t):
    if t.startswith("<sys"):
        return "<sys>"
    return t.strip() or t


def region(i):
    return Point3(i * STAGE_DX, 0, ROW_Z)


def sc_of(M):
    M = np.asarray(M, dtype=float)
    M = M[np.isfinite(M)]
    return max(1e-6, 1.6 * float(np.std(M)) if M.size else 1.0)


def show_and_fade(node, dur=0.5):
    return Sequence(Func(node.show), fade_in(node, dur))


class TransformerSteps:
    """Mixin for Story. Needs: self.board, self.view(), self.orbit(), self.engine, self.focus_index."""

    # ------------------------------------------------------------ setup
    def tf_setup(self):
        self.T = self.engine.trace_first()
        self.ttoks = self.T["tokens"]
        self.n = len(self.ttoks)
        self.tcolor = [GREY if t.startswith("<") else TOKEN_COLORS[i % len(TOKEN_COLORS)]
                       for i, t in enumerate(self.ttoks)]
        f = self.sc.get("focus")
        self.fi = self.ttoks.index(f) if f in self.ttoks else self.n - 1
        self.d = self.T["x0"].shape[1]
        self.dh = self.T["layers"][0]["heads"][0]["q"].shape[1]
        self.V = len(self.engine.m.vocab)

    def xs(self, x0):
        return [x0 + i * CS for i in range(self.n)]

    def header(self, parent, x0, z, scale=0.25):
        g = parent.attachNewNode("hdr")
        for i, t in enumerate(self.ttoks):
            token_box(g, disp(t), self.tcolor[i], (x0 + i * CS, 0, z), scale)
        g.flattenStrong()
        return g

    def columns(self, parent, M, x0, z_top, ch=CH, scale=None):
        """One heatmap column per token (M is n x dim). Returns list of nodes."""
        scale = scale or sc_of(M)
        out = []
        for i in range(M.shape[0]):
            c = heatmap(parent, M[i:i + 1].T, x0 + i * CS - CS * 0.4, z_top, CS * 0.8, ch, scale, gap=0.1)
            out.append(c)
        return out

    def matrix(self, parent, M, x0, z_top, cw, ch, label=None, shape=None, scale=None, color=WHITE):
        g = parent.attachNewNode("mat")
        heatmap(g, M, x0, z_top, cw, ch, scale or sc_of(M), gap=0.08)
        R, C = M.shape
        rect(g, x0 - 0.05, z_top - R * ch - 0.05, x0 + C * cw + 0.05, z_top + 0.05, DIM, 1.2)
        if label:
            text(g, label, Point3(x0 + C * cw / 2, 0, z_top + 0.35), 0.4, color, Fonts.symbol)
        if shape:
            shape_label(g, shape, Point3(x0 + C * cw / 2, 0, z_top - R * ch - 0.5), 0.3)
        return g

    def sweep(self, nodes, total=1.6):
        gap = total / max(1, len(nodes))
        return Parallel(*[Sequence(Wait(i * gap), Func(nd.show), fade_in(nd, 0.3)) for i, nd in enumerate(nodes)])

    def hide_all(self, nodes):
        for nd in nodes:
            nd.hide()

    def mapkey(self, *keys):
        return Func(self.app.ui.arch.highlight, *keys)

    def top_attn(self, att, q, k=3):
        idx = np.argsort(-att[q])[:k]
        return ", ".join("'{}' {:.2f}".format(disp(self.ttoks[j]), att[q, j]) for j in idx)

    # ------------------------------------------------------------ 1 embedding
    def s_embed(self):
        R = region(0)
        g = self.board.attachNewNode("embed")
        self.g_embed = g
        E = self.engine.m.w["E"]
        ids = self.T["ids"]
        # the embedding table (all 181 rows)
        ex0, ez0, ecw, ech = R.x - 13.0, R.z + 6.6, 0.1, 0.07
        tab = self.matrix(g, E, ex0, ez0, ecw, ech, "embedding table", "{} tokens x {} numbers".format(self.V, self.d))
        text(tab, "one row per token in the vocabulary", Point3(ex0 + self.d * ecw / 2, 0, ez0 + 0.85), 0.26, GREY)
        hl = g.attachNewNode("rows")
        for i, tid in enumerate(ids):
            rect(hl, ex0 - 0.08, ez0 - (tid + 1) * ech, ex0 + self.d * ecw + 0.08, ez0 - tid * ech, YELLOW, 1.4, y=-0.02)
        x0 = R.x - 2.0
        hdr = self.header(g, x0, R.z + 6.0)
        cols = self.columns(g, self.T["emb"], x0, R.z + 5.2)
        self.hide_all(cols)
        lab = shape_label(g, "X = {} x {}   (one column per token)".format(self.n, self.d),
                          Point3(x0 + (self.n - 1) * CS / 2, 0, R.z + 5.2 - self.d * CH - 0.45), 0.34)
        legend = g.attachNewNode("legend")
        lx = x0 - 2.6
        heatmap(legend, np.array([[2], [1], [0], [-1], [-2]]), lx, R.z + 4.0, 0.36, 0.36, 1.0)
        for k, t in enumerate(("+", "", "0", "", "-")):
            text(legend, t, Point3(lx - 0.25, 0, R.z + 4.0 - (k + 0.7) * 0.36), 0.3, GREY)
        flies = Parallel()
        for i, tid in enumerate(ids):
            row = heatmap(g, E[tid:tid + 1], ex0, ez0 - tid * ech, ecw, ech, sc_of(E), gap=0.08)
            row.hide()
            dst = Point3(x0 + i * CS, 0, R.z + 5.2) - Point3(ex0, 0, ez0 - tid * ech)
            flies.append(Sequence(Wait(0.12 * i), Func(row.show),
                                  LerpPosInterval(row, 0.9, dst + Point3(0, 0, 0), blendType="easeInOut"),
                                  Func(row.hide), Func(cols[i].show)))
        for nd in (hl, lab, legend):
            nd.hide()
        seq = Sequence(self.mapkey("embed"), self.view(R + Vec3(-1.8, 0, -0.6), 0, 0, 25),
                       show_and_fade(hdr, 0.5), show_and_fade(tab, 0.6), Wait(0.3), show_and_fade(hl, 0.4),
                       flies, show_and_fade(lab, 0.4), show_and_fade(legend, 0.4))
        return ("TOKEN EMBEDDING. The model has a table with one row of {} numbers for each of its {} tokens "
                "(learned during training). Each token ID picks its row, and the row becomes that token's column. "
                "Red = positive, blue = negative. This tiny model uses {} numbers; big models use thousands."
                .format(self.d, self.V, self.d), seq)

    # ------------------------------------------------------------ 2 positional encoding
    def s_position(self):
        R = region(0)
        g = self.g_embed
        x0 = R.x - 2.0
        pe_top = R.z - 2.2
        pe = self.columns(g, self.T["pe"], x0, pe_top, scale=1.0)
        self.hide_all(pe)
        lab = text(g, "positional encoding  (position 0, 1, 2, ...)", Point3(x0 + (self.n - 1) * CS / 2, 0,
                                                                             pe_top + 0.4), 0.34, YELLOW)
        lab.hide()
        x0cols = self.columns(g, self.T["x0"], x0, R.z + 5.2)
        self.hide_all(x0cols)
        plus = text(g, "+", Point3(x0 - 1.2, 0, R.z + 0.2), 0.9, WHITE)
        plus.hide()
        merge = Parallel()
        for i, c in enumerate(pe):
            merge.append(Sequence(Wait(0.05 * i), LerpPosInterval(c, 1.0, Point3(0, 0, 5.2 - pe_top + R.z),
                                                                   blendType="easeInOut"),
                                  Func(c.hide), Func(x0cols[i].show)))
        seq = Sequence(self.mapkey("pos"), self.view(R + Vec3(3.6, 0, 0.6), 0, 0, 23),
                       show_and_fade(lab, 0.4), self.sweep(pe, 1.2), Wait(0.6), show_and_fade(plus, 0.3),
                       self.mapkey("pos", "embed"), merge, Wait(0.3))
        self.x0cols = x0cols
        return ("POSITIONAL ENCODING. Attention by itself does not know word order ('dog bites man' = 'man bites "
                "dog'). So each position gets its own fixed wave pattern (sine and cosine waves of different "
                "speeds - see the stripes) and it is ADDED to the token's column. Now every column also says "
                "where the token stands.", seq)

    # ------------------------------------------------------------ 3 Q, K, V (one head)
    def _qkv_stage(self, layer, head, idx, fast=False):
        R = region(idx)
        LT = self.T["layers"][layer]
        H = LT["heads"][head]
        g = self.board.attachNewNode("qkv{}{}".format(layer, head))
        x0 = R.x - 10.0
        hdr = self.header(g, x0, R.z + 6.4)
        X = LT["x_in"]
        xc = self.columns(g, X, x0, R.z + 5.6)
        xl = shape_label(g, "X  {} x {}".format(self.n, self.d), Point3(x0 + (self.n - 1) * CS / 2, 0,
                                                                         R.z + 5.6 - self.d * CH - 0.4), 0.32)
        out = {}
        parts = []
        for j, (nm, col) in enumerate((("Q", YELLOW), ("K", GREEN), ("V", RED))):
            W = H["W" + nm.lower()]
            wx = R.x + 4.6 + j * 2.6
            mat = self.matrix(g, W, wx, R.z + 5.6, 0.12, CH * 0.5, "W_" + nm, "{} x {}".format(self.d, self.dh),
                              color=col)
            M = H[nm.lower()]
            zt = R.z - 0.7 - j * 2.75
            cols = self.columns(g, M, x0, zt, ch=0.14)
            lab = g.attachNewNode("lab")
            vlabel(lab, nm, "", (x0 - 1.4, 0, zt - 1.3), 0.55, col)
            shape_label(lab, "X W_{} = {} x {}".format(nm, self.n, self.dh),
                        Point3(x0 + self.n * CS + 1.8, 0, zt - 1.2), 0.32)
            parts.append((mat, cols, lab))
            out[nm] = cols
        for nd in [hdr, xl] + [p[0] for p in parts] + [p[2] for p in parts]:
            nd.hide()
        for c in xc + [c for p in parts for c in p[1]]:
            c.hide()
        seq = Sequence(show_and_fade(hdr, 0.3), self.sweep(xc, 0.6 if fast else 1.0), show_and_fade(xl, 0.3))
        for mat, cols, lab in parts:
            seq.append(Sequence(show_and_fade(mat, 0.4), self.sweep(cols, 0.6 if fast else 1.2), show_and_fade(lab, 0.3)))
        return g, seq, R

    def s_qkv(self):
        g, seq, R = self._qkv_stage(0, 0, 1)
        self.g_qkv = g
        return ("LAYER 1, ATTENTION, HEAD 1.  Every token column is multiplied by three learned matrices: W_Q gives "
                "its QUERY ('what am I looking for?'), W_K its KEY ('what do I offer?'), W_V its VALUE ('what I will "
                "pass on'). 32 numbers in, 16 out, for all {} tokens at once.".format(self.n),
                Sequence(self.mapkey("attn"), self.view(R + Vec3(-1.0, 0, -1.6), 0, 0, 28), seq))

    # ------------------------------------------------------------ 4 scores, scale, mask, softmax
    def _table(self, parent, H, x0, z0, cw=0.9, ch=0.5, small=False):
        """Q/K table: rows = keys (tokens), columns = queries (tokens)."""
        n = self.n
        g = parent.attachNewNode("table")
        hdr = g.attachNewNode("th")
        for i, t in enumerate(self.ttoks):
            token_box(hdr, disp(t), self.tcolor[i], (x0 + (i + 0.5) * cw, 0, z0 + 0.35), 0.19)
            text(hdr, disp(t), Point3(x0 - 0.15, 0, z0 - (i + 0.62) * ch), 0.2, self.tcolor[i], align=TextNode.ARight)
        vlabel(hdr, "Q", "", (x0 + n * cw / 2, 0, z0 + 0.95), 0.42, YELLOW)
        vlabel(hdr, "K", "", (x0 - 1.7, 0, z0 - n * ch / 2), 0.42, GREEN)
        lines(hdr, [[(x0, 0, z0), (x0 + n * cw, 0, z0)], [(x0, 0, z0), (x0, 0, z0 - n * ch)]], GREY, 1.2)
        hdr.flattenStrong()
        states = {}
        raw, scaled, att = H["raw"], H["scaled"], H["att"]
        for key, M, fmt in (("raw", raw, "{:.1f}"), ("scaled", scaled, "{:.1f}"), ("att", att, "{:.2f}")):
            s = g.attachNewNode(key)
            up = g.attachNewNode(key + "_hi")          # cells above the diagonal (later tokens)
            for q in range(n):
                for k in range(n):
                    if key == "att" and k > q:
                        continue
                    v = M[q, k]
                    col = WHITE
                    if key == "att":
                        col = WHITE if v > 0.15 else GREY
                    text(up if k > q else s, fmt.format(v), Point3(x0 + (q + 0.5) * cw, 0, z0 - (k + 0.64) * ch),
                         0.2, col)
            for nd in (s, up):
                nd.flattenStrong()
                nd.hide()
            states[key] = s
            states[key + "_hi"] = up
        # masked cells
        m = g.attachNewNode("mask")
        for q in range(n):
            for k in range(q + 1, n):
                fill(m, x0 + q * cw + 0.03, z0 - (k + 1) * ch + 0.03, x0 + (q + 1) * cw - 0.03, z0 - k * ch - 0.03,
                     (0.35, 0.35, 0.4, 1), 0.35)
                text(m, "-∞", Point3(x0 + (q + 0.5) * cw, -0.01, z0 - (k + 0.62) * ch), 0.17, GREY, Fonts.symbol)
        m.flattenStrong()
        m.hide()
        states["mask"] = m
        # weight discs behind the numbers
        dsc = g.attachNewNode("discs")
        for q in range(n):
            for k in range(q + 1):
                w = att[q, k]
                b = 0.12 + 0.7 * min(1.0, w * 1.5)
                d = disc(dsc, min(cw, ch) * 0.42, (b * 0.8, b * 0.8, b * 0.8, 1), 20)
                d.setPos(x0 + (q + 0.5) * cw, 0.02, z0 - (k + 0.5) * ch)
        dsc.flattenStrong()
        dsc.hide()
        states["discs"] = dsc
        hdr.hide()
        return g, hdr, states

    def s_scores(self):
        R = region(2)
        H = self.T["layers"][0]["heads"][0]
        x0, z0 = R.x - 6.0, R.z + 4.5
        g, hdr, st = self._table(self.board, H, x0, z0)
        self.tab1 = (g, st, x0, z0)
        n = self.n
        cw, ch = 0.9, 0.5
        hl = rect(g, x0 + self.fi * cw + 0.02, z0 - n * ch, x0 + (self.fi + 1) * cw - 0.02, z0 + 0.6, YELLOW, 2.6,
                  y=-0.03)
        hl.hide()
        info = text(g, "", Point3(x0 + n * cw + 0.6, 0, z0 - 0.3), 0.32, WHITE, Fonts.symbol, TextNode.ALeft, wrap=22)
        sh = shape_label(g, "Q Kᵀ = {} x {}".format(n, n), Point3(x0 + n * cw / 2, 0, z0 - n * ch - 0.5), 0.32)
        sh.hide()

        def say(s):
            info.node().setText(s)
        seq = Sequence(
            self.mapkey("attn"), self.view(R + Vec3(-0.2, 0, 1.2), 0, 0, 18.0),
            show_and_fade(hdr, 0.5), show_and_fade(sh, 0.3),
            Func(say, "1. score = Q . K\n(how well a key\nmatches a query)"),
            Parallel(show_and_fade(st["raw"], 0.8), show_and_fade(st["raw_hi"], 0.8)), Wait(1.2),
            Func(say, "2. divide by {}{} = {:.0f}\n(keeps the numbers\nin a good range)".format(SQ, self.dh,
                                                                                                np.sqrt(self.dh))),
            Parallel(fade_out(st["raw"], 0.4), fade_out(st["raw_hi"], 0.4)), Func(st["raw"].hide),
            Func(st["raw_hi"].hide), Parallel(show_and_fade(st["scaled"], 0.5), show_and_fade(st["scaled_hi"], 0.5)),
            Wait(1.0),
            Func(say, "3. MASK: a token may\nnot look at tokens\nAFTER it -> score = -∞"),
            fade_out(st["scaled_hi"], 0.4), Func(st["scaled_hi"].hide), show_and_fade(st["mask"], 0.6), Wait(1.0),
            Func(say, "4. softmax on each\ncolumn: weights between\n0 and 1 that add up to 1"),
            fade_out(st["scaled"], 0.4), Func(st["scaled"].hide), show_and_fade(st["discs"], 0.5),
            show_and_fade(st["att"], 0.5), Wait(0.6), show_and_fade(hl, 0.4),
            Func(say, "column '{}':\n{}".format(disp(self.ttoks[self.fi]),
                                                self.top_attn(H["att"], self.fi).replace(", ", "\n"))))
        return ("ATTENTION SCORES. Each query (column) is compared with every key (row): a dot product, scaled down, "
                "masked so no token can peek at later tokens, then turned into weights with softmax. These are the "
                "REAL numbers of the tiny model. Token '{}' attends most to: {}."
                .format(disp(self.ttoks[self.fi]), self.top_attn(H["att"], self.fi)), seq)

    # ------------------------------------------------------------ 5 weighted sum of values
    def s_weighted(self):
        R = region(3)
        H = self.T["layers"][0]["heads"][0]
        att, V, out = H["att"], H["v"], H["out"]
        q = self.fi
        g = self.board.attachNewNode("wsum")
        ks = [k for k in np.argsort(-att[q])[:4] if att[q, k] > 0.02]
        ks = sorted(ks)
        x = R.x - 9.0
        parts = []
        for j, k in enumerate(ks):
            grp = g.attachNewNode("term")
            token_box(grp, disp(self.ttoks[k]), self.tcolor[k], (x + 1.0, 0, R.z + 4.4), 0.26)
            text(grp, "{:.2f}".format(att[q, k]), Point3(x, 0, R.z + 1.6), 0.5, YELLOW)
            text(grp, "x", Point3(x + 0.55, 0, R.z + 1.6), 0.35, GREY)
            heatmap(grp, V[k:k + 1].T, x + 0.75, R.z + 3.8, 0.5, 0.25, sc_of(V), gap=0.1)
            vlabel(grp, "V", "", (x + 1.0, 0, R.z - 0.5), 0.35, RED)
            if j < len(ks) - 1:
                text(grp, "+", Point3(x + 1.85, 0, R.z + 1.6), 0.55, WHITE)
            grp.hide()
            parts.append(grp)
            x += 2.6
        res = g.attachNewNode("res")
        text(res, "=", Point3(x - 0.3, 0, R.z + 1.6), 0.6, WHITE)
        heatmap(res, out[q:q + 1].T, x + 0.25, R.z + 3.8, 0.5, 0.25, sc_of(out), gap=0.1)
        text(res, "head output\nfor '{}'".format(disp(self.ttoks[q])), Point3(x + 0.5, 0, R.z - 0.4), 0.3, YELLOW)
        res.hide()
        others = sum(1 for k in range(q + 1) if k not in ks)
        note = text(g, "(+ {} more tokens with tiny weights)".format(others) if others else "",
                    Point3(R.x - 3.0, 0, R.z - 1.4), 0.3, GREY)
        note.hide()
        # all tokens at once
        allg = g.attachNewNode("all")
        x0 = R.x - 6.0
        z0 = R.z - 2.8
        cols = self.columns(allg, out, x0, z0, ch=0.17)
        self.hide_all(cols)
        alab = text(allg, "the same for EVERY token at once:  weights x V  =  {} x {}".format(self.n, self.dh),
                    Point3(x0 + (self.n - 1) * CS / 2, 0, z0 + 0.4), 0.3, WHITE, Fonts.symbol)
        alab.hide()
        seq = Sequence(self.mapkey("attn"), self.view(R + Vec3(-1.5, 0, -0.4), 0, 0, 19.5),
                       Parallel(*[Sequence(Wait(0.35 * j), show_and_fade(p, 0.5)) for j, p in enumerate(parts)]),
                       show_and_fade(note, 0.3), Wait(0.3), show_and_fade(res, 0.6), Wait(0.6),
                       show_and_fade(alab, 0.3), self.sweep(cols, 1.0))
        return ("Now each token collects information: its weights mix the VALUE vectors of the tokens it attends to "
                "(weighted sum). For '{}' that is mostly {}. The result is this head's output for '{}' - and the "
                "same happens for all {} tokens in parallel."
                .format(disp(self.ttoks[q]), " and ".join("'{}'".format(disp(self.ttoks[k])) for k in ks[:2]),
                        disp(self.ttoks[q]), self.n), seq)

    # ------------------------------------------------------------ 6 head 2
    def s_head2(self):
        g, seq_qkv, R = self._qkv_stage(0, 1, 4, fast=True)
        H1 = self.T["layers"][0]["heads"][0]
        H2 = self.T["layers"][0]["heads"][1]
        x0, z0 = R.x + 13.0, R.z + 4.5
        tg, hdr, st = self._table(self.board, H2, x0, z0)
        cw, ch = 0.9, 0.5
        hl = rect(tg, x0 + self.fi * cw + 0.02, z0 - self.n * ch, x0 + (self.fi + 1) * cw - 0.02, z0 + 0.6, YELLOW,
                  2.6, y=-0.03)
        hl.hide()
        oc = self.columns(tg, H2["out"], x0, z0 - self.n * ch - 1.2, ch=0.12)
        self.hide_all(oc)
        olab = text(tg, "head 2 output  {} x {}".format(self.n, self.dh), Point3(x0 + (self.n - 1) * CS / 2, 0,
                                                                                 z0 - self.n * ch - 0.8), 0.3, WHITE)
        olab.hide()
        title = text(self.board, "HEAD 2  (its own W_Q, W_K, W_V)", Point3(R.x - 4.0, 0, R.z + 7.6), 0.6, WHITE)
        title.hide()
        seq = Sequence(self.mapkey("attn"), self.view(R + Vec3(-1.0, 0, -1.6), 0, 0, 28), show_and_fade(title, 0.4),
                       seq_qkv, Wait(0.4), self.view(Point3(x0 + 5.9, 0, z0 - 4.3), 0, 0, 20),
                       show_and_fade(hdr, 0.4), show_and_fade(st["mask"], 0.3),
                       show_and_fade(st["discs"], 0.4), show_and_fade(st["att"], 0.4), show_and_fade(hl, 0.3),
                       show_and_fade(olab, 0.3), self.sweep(oc, 0.8))
        f = disp(self.ttoks[self.fi])
        return ("MULTI-HEAD: the same calculation runs again with a second, independent set of matrices. Each head "
                "learns to look for different relations. For '{}': head 1 -> {};  head 2 -> {}."
                .format(f, self.top_attn(H1["att"], self.fi, 2), self.top_attn(H2["att"], self.fi, 2)), seq)

    # ------------------------------------------------------------ 7 concat x W_O
    def s_concat(self):
        R = region(5)
        LT = self.T["layers"][0]
        g = self.board.attachNewNode("concat")
        x0 = R.x - 9.0
        hdr = self.header(g, x0, R.z + 6.4)
        h1 = self.columns(g, LT["heads"][0]["out"], x0, R.z + 5.6, ch=CH)
        h2 = self.columns(g, LT["heads"][1]["out"], x0, R.z + 5.6 - self.dh * CH, ch=CH)
        lab1 = text(g, "head 1", Point3(x0 - 1.2, 0, R.z + 4.3), 0.3, YELLOW)
        lab2 = text(g, "head 2", Point3(x0 - 1.2, 0, R.z + 1.6), 0.3, YELLOW)
        cl = shape_label(g, "concat = {} x {}".format(self.n, 2 * self.dh), Point3(x0 + (self.n - 1) * CS / 2, 0,
                                                                                   R.z + 5.6 - 2 * self.dh * CH - 0.4),
                         0.32)
        wo = self.matrix(g, LT["Wo"], R.x + 5.5, R.z + 5.6, CH, CH, "W_O", "{} x {}".format(self.d, self.d))
        dx0 = x0
        dz = R.z - 1.6
        de = self.columns(g, LT["delta"], dx0, dz)
        dl = g.attachNewNode("dl")
        vlabel(dl, "ΔE", "", (dx0 - 1.5, 0, dz - 2.8), 0.5, YELLOW)
        shape_label(dl, "concat W_O = {} x {}".format(self.n, self.d), Point3(dx0 + (self.n - 1) * CS / 2, 0,
                                                                              dz - self.d * CH - 0.4), 0.32)
        for nd in [hdr, lab1, lab2, cl, wo, dl] + h1 + h2 + de:
            nd.hide()
        seq = Sequence(self.mapkey("attn"), self.view(R + Vec3(-1.0, 0, -0.2), 0, 0, 30),
                       show_and_fade(hdr, 0.3), show_and_fade(lab1, 0.2), self.sweep(h1, 0.6), show_and_fade(lab2, 0.2),
                       self.sweep(h2, 0.6), show_and_fade(cl, 0.3), Wait(0.3), show_and_fade(wo, 0.5),
                       self.sweep(de, 1.2), show_and_fade(dl, 0.4))
        return ("The two head outputs (16 + 16 numbers) are stacked into one column of 32 per token, then mixed by "
                "one more learned matrix W_O. The result, ΔE, is the CHANGE each token wants to make to itself "
                "after looking at the others.", seq)

    # ------------------------------------------------------------ 8 residual: every token is updated
    def s_residual(self):
        R = region(6)
        LT = self.T["layers"][0]
        g = self.board.attachNewNode("resid")
        x0 = R.x - 6.5
        z = R.z + 5.4
        hdr = self.header(g, x0, z + 0.8)
        X = self.columns(g, LT["x_in"], x0, z)
        D = self.columns(g, LT["delta"], x0, z - 6.4)
        Y = self.columns(g, LT["res1"], x0, z)
        tx = text(g, "X  (before)", Point3(x0 - 2.2, 0, z - 2.7), 0.34, WHITE)
        td = text(g, "+ ΔE", Point3(x0 - 2.2, 0, z - 9.1), 0.4, YELLOW, Fonts.symbol)
        for nd in [hdr, tx, td] + X + D + Y:
            nd.hide()
        move = Parallel()
        for i in range(self.n):
            move.append(Sequence(LerpPosInterval(D[i], 1.2, Point3(0, 0, 6.4), blendType="easeInOut"),
                                 Parallel(fade_out(D[i], 0.4), fade_out(X[i], 0.4)),
                                 Func(D[i].hide), Func(X[i].hide), Func(Y[i].show), fade_in(Y[i], 0.4)))
        tafter = text(g, "X + ΔE  (after)", Point3(x0 - 2.2, 0, z - 2.7), 0.34, YELLOW, Fonts.symbol)
        tafter.hide()
        # arrow space: real vectors projected to 3D (PCA), before and after
        sp, arrows = self._space(R + Vec3(17, 6, -2.5), LT["x_in"], LT["res1"])
        sp.hide()
        c3 = R + Vec3(17, 6, -1.6)
        seq = Sequence(self.mapkey("add1"), self.view(R + Vec3(-1.0, 0, 0.4), 0, 0, 25),
                       show_and_fade(hdr, 0.3), show_and_fade(tx, 0.2), self.sweep(X, 0.5), show_and_fade(td, 0.2),
                       self.sweep(D, 0.5), Wait(0.4), move, Func(tx.hide), show_and_fade(tafter, 0.3), Wait(0.8),
                       self.view(c3, -30, 24, 15), Func(sp.show), fade_in(sp, 0.6),
                       Parallel(*[LerpFunc(a.grow, fromData=0.001, toData=1.0, duration=0.8) for a in arrows["before"]]),
                       Wait(0.6),
                       Parallel(*[self._move_arrow(a, b) for a, b in zip(arrows["before"], arrows["after"])],
                                self.orbit(c3, -30, 25, 24, 15, 5.0)))
        return ("ADD (the residual connection): ΔE is added to EVERY token's column at the same time - this is "
                "how what each token learned from the others gets written into it. In the 3D view (the real "
                "vectors, squeezed to 3 dimensions) every arrow moves at once: each word's meaning now includes its "
                "context.", seq)

    def _space(self, center, A, B):
        sp = self.board.attachNewNode("space")
        sp.setPos(center)
        both = np.vstack([A, B])
        mu = both.mean(0)
        U, S, Vt = np.linalg.svd(both - mu, full_matrices=False)
        P = (both - mu) @ Vt[:3].T
        P = P / (np.abs(P).max() + 1e-9) * 3.6
        pa, pb = P[:len(A)], P[len(A):]
        L = 4.0
        grid = []
        for k in range(-4, 5):
            grid.append([(k, -4, 0), (k, 4, 0)])
            grid.append([(-4, k, 0), (4, k, 0)])
        lines(sp, grid, (0.16, 0.18, 0.2, 1), 1.0)
        lines(sp, [[(-L, 0, 0), (L, 0, 0)], [(0, -L, 0), (0, L, 0)], [(0, 0, -L), (0, 0, L)]], GREY, 1.6)
        arrows = {"before": [], "after": []}
        for i in range(len(A)):
            a = Arrow(sp, Point3(0, 0, 0), Point3(*pa[i]), self.tcolor[i], 2.6)
            lab = text(a.root, disp(self.ttoks[i]), Point3(*(pa[i] * 1.12)), 0.3, self.tcolor[i])
            lab.setBillboardPointEye()
            a.label = lab
            a.after = Point3(*pb[i])
            arrows["before"].append(a)
            arrows["after"].append(Point3(*pb[i]))
        return sp, arrows

    def _move_arrow(self, a, end):
        start = Point3(a.end)
        ghost = disc(a.root.getParent(), 0.07, a.color[:3] + (0.5,), 12)
        ghost.setPos(start)
        ghost.setBillboardPointEye()
        trail = lines(a.root.getParent(), [[start, end]], a.color[:3] + (0.45,), 1.2)
        trail.hide()

        def f(t):
            a.end = start * (1 - t) + end * t
            a.grow(1.0)
            a.label.setPos(a.end * 1.12)
        return Sequence(LerpFunc(f, fromData=0.0, toData=1.0, duration=2.0, blendType="easeInOut"), Func(trail.show))

    # ------------------------------------------------------------ 9 layer norm
    def s_norm(self):
        R = region(7)
        LT = self.T["layers"][0]
        g = self.board.attachNewNode("norm")
        i = self.fi
        v = LT["res1"][i]
        normed = (v - v.mean()) / np.sqrt(v.var() + 1e-5)
        out = LT["x1"][i]
        bars = []
        for k, (vals, col, title) in enumerate(((v, WHITE, "before: '{}'".format(disp(self.ttoks[i]))),
                                               (normed, YELLOW, "minus mean, divide by spread"),
                                               (out, GREEN, "x gain + bias (learned)"))):
            bg = g.attachNewNode("bars")
            zb = R.z + 4.0 - k * 3.6
            sc = 1.2 / max(1e-6, max(np.abs(v).max(), np.abs(normed).max(), np.abs(out).max()))
            for j, val in enumerate(vals):
                h = val * sc
                fill(bg, R.x - 8 + j * 0.42, zb + min(0, h), R.x - 8 + j * 0.42 + 0.3, zb + max(0, h), col, 0.9, y=0)
            lines(bg, [[(R.x - 8.2, 0, zb), (R.x - 8 + self.d * 0.42, 0, zb)]], GREY, 1.0)
            text(bg, title, Point3(R.x - 8.2, 0, zb + 1.5), 0.32, col, align=TextNode.ALeft)
            text(bg, "mean {:+.2f}   spread {:.2f}".format(vals.mean(), vals.std()), Point3(R.x + 6.0, 0, zb + 0.3),
                 0.3, GREY, align=TextNode.ALeft)
            bg.hide()
            bars.append(bg)
        seq = Sequence(self.mapkey("add1"), self.view(R + Vec3(-0.5, 0, 0.6), 0, 0, 20),
                       Parallel(*[Sequence(Wait(0.9 * k), show_and_fade(b, 0.5)) for k, b in enumerate(bars)]))
        return ("NORM (layer normalization): each token's 32 numbers are shifted to mean 0 and scaled to spread 1, "
                "then multiplied by a learned gain and bias. This keeps the numbers in a stable range layer after "
                "layer. (Shown for '{}'; every token gets the same treatment.)".format(disp(self.ttoks[i])), seq)

    # ------------------------------------------------------------ 10 feed-forward
    def s_ffn(self):
        R = region(8)
        LT = self.T["layers"][0]
        g = self.board.attachNewNode("ffn")
        i = self.fi
        x_in, pre, hid, ff = LT["x1"][i], LT["hid_pre"][i], LT["hid"][i], LT["ff"][i]
        F = len(pre)
        ztop = R.z + 6.4
        inc = heatmap(g, x_in[None, :].T, R.x - 9.0, R.z + 4.6, 0.45, 0.2, sc_of(x_in), gap=0.1)
        text(g, "'{}'\n{} numbers".format(disp(self.ttoks[i]), self.d), Point3(R.x - 8.8, 0, R.z + 5.2), 0.3, WHITE)
        neurons = g.attachNewNode("neurons")
        dots = []
        sc = sc_of(pre)
        from kit import value_color
        per = F // 4
        for j in range(F):
            d = disc(neurons, 0.11, value_color(pre[j], sc), 14)
            d.setPos(R.x - 1.6 + (j // per) * 0.42, -0.01, ztop - (j % per) * 0.36)
            dots.append(d)
        text(neurons, "{} neurons".format(F), Point3(R.x - 0.97, 0, ztop + 0.5), 0.34, WHITE)
        shp = shape_label(g, "x W1 + b1 :  {} -> {}".format(self.d, F), Point3(R.x - 4.8, 0, ztop + 0.2), 0.3)
        relu = g.attachNewNode("relu")
        off = int((pre <= 0).sum())
        text(relu, "ReLU: negative -> 0\n{} of {} neurons\nswitched off (dark)".format(off, F),
             Point3(R.x + 0.4, 0, ztop - per * 0.36 - 0.3), 0.34, ORANGE)
        outc = heatmap(g, ff[None, :].T, R.x + 7.5, R.z + 4.6, 0.45, 0.2, sc_of(ff), gap=0.1)
        olab = g.attachNewNode("ol")
        text(olab, "output\n{} numbers".format(self.d), Point3(R.x + 7.72, 0, R.z + 5.2), 0.3, GREEN)
        shape_label(olab, "x W2 + b2 :  {} -> {}".format(F, self.d), Point3(R.x + 3.9, 0, ztop + 0.2), 0.3)
        # a FEW representative connections (the strongest ones), not all of them
        act = np.argsort(-hid)[:6]
        cons = g.attachNewNode("cons")
        for j in act:
            p = dots[j].getPos()
            for k in np.argsort(-np.abs(self.engine.m.w["l0.W1"][:, j]))[:2]:
                lines(cons, [[(R.x - 8.55, 0, R.z + 4.6 - (k + 0.5) * 0.2), (p.x, 0, p.z)]], BLUE[:3] + (0.6,), 1.2)
            for k in np.argsort(-np.abs(self.engine.m.w["l0.W2"][j]))[:2]:
                lines(cons, [[(p.x, 0, p.z), (R.x + 7.5, 0, R.z + 4.6 - (k + 0.5) * 0.2)]], RED[:3] + (0.6,), 1.2)
        allg = g.attachNewNode("all")
        text(allg, "every token goes through the SAME network on its own:\n{} x {}  ->  {} x {}".format(
            self.n, self.d, self.n, self.d), Point3(R.x + 7.7, 0, R.z - 2.4), 0.3, WHITE, Fonts.symbol)
        for nd in (inc, neurons, shp, relu, outc, olab, cons, allg):
            nd.hide()
        dim = Parallel(*[LerpColorScaleInterval(dots[j], 0.6, (0.15, 0.15, 0.15, 1)) for j in range(F) if pre[j] <= 0])
        seq = Sequence(self.mapkey("ffn"), self.view(R + Vec3(-0.6, 0, -0.4), 0, 0, 24),
                       show_and_fade(inc, 0.4), show_and_fade(shp, 0.3), show_and_fade(neurons, 0.8),
                       show_and_fade(cons, 0.5), Wait(0.5), show_and_fade(relu, 0.4), dim, Wait(0.5),
                       show_and_fade(olab, 0.3), show_and_fade(outc, 0.5), Wait(0.3), show_and_fade(allg, 0.4))
        return ("FEED FORWARD (the MLP): each token on its own goes through a small 2-layer network - expand 32 "
                "numbers to {} neurons, ReLU switches off the negative ones ({} here), then compress back to 32. "
                "Only the strongest connections are drawn. This is where much of the model's stored knowledge "
                "lives.".format(F, off), seq)

    # ------------------------------------------------------------ 11 add & norm 2 (layer output)
    def s_add2(self):
        R = region(9)
        LT = self.T["layers"][0]
        g = self.board.attachNewNode("add2")
        x0 = R.x - 6.5
        z = R.z + 5.4
        hdr = self.header(g, x0, z + 0.8)
        A = self.columns(g, LT["x1"], x0, z)
        B = self.columns(g, LT["ff"], x0, z - 6.4)
        C = self.columns(g, LT["x_out"], x0, z)
        t1 = text(g, "x", Point3(x0 - 2.0, 0, z - 2.7), 0.36, WHITE)
        t2 = text(g, "+ feed-forward", Point3(x0 - 2.6, 0, z - 9.1), 0.32, GREEN)
        t3 = text(g, "layer 1 output", Point3(x0 - 2.6, 0, z - 2.7), 0.32, YELLOW)
        for nd in [hdr, t1, t2, t3] + A + B + C:
            nd.hide()
        move = Parallel(*[Sequence(LerpPosInterval(B[i], 1.0, Point3(0, 0, 6.4), blendType="easeInOut"),
                                   Func(B[i].hide), Func(A[i].hide), Func(C[i].show), fade_in(C[i], 0.4))
                          for i in range(self.n)])
        seq = Sequence(self.mapkey("add2"), self.view(R + Vec3(-1.0, 0, 0.4), 0, 0, 25), show_and_fade(hdr, 0.3),
                       show_and_fade(t1, 0.2), self.sweep(A, 0.4), show_and_fade(t2, 0.2), self.sweep(B, 0.4),
                       move, Func(t1.hide), show_and_fade(t3, 0.3))
        return ("ADD & NORM again: the feed-forward result is added to every token's column and normalized. That "
                "completes LAYER 1. Its output - same shape, {} x {} - is the input of layer 2.".format(self.n, self.d),
                seq)

    # ------------------------------------------------------------ 12 layer 2 (same structure, faster)
    def s_layer2(self):
        R = region(10)
        LT = self.T["layers"][1]
        g = self.board.attachNewNode("layer2")
        tabs = []
        for h in range(2):
            x0, z0 = R.x - 13.0 + h * 15.0, R.z + 5.0
            tg, hdr, st = self._table(g, LT["heads"][h], x0, z0)
            title = text(g, "layer 2, head {}".format(h + 1), Point3(x0 + 6, 0, z0 + 1.7), 0.45, WHITE)
            title.hide()
            tabs.append((title, hdr, st))
        X = self.columns(g, LT["x_out"], R.x - 6.0, R.z - 3.4, ch=0.13)
        self.hide_all(X)
        lab = text(g, "layer 2 output  {} x {}".format(self.n, self.d), Point3(R.x, 0, R.z - 2.9), 0.36, YELLOW)
        lab.hide()
        cycle = Sequence(self.mapkey("attn"), Wait(0.5), self.mapkey("add1"), Wait(0.5), self.mapkey("ffn"),
                         Wait(0.5), self.mapkey("add2"), Wait(0.5))
        anim = Sequence()
        for title, hdr, st in tabs:
            anim.append(Sequence(show_and_fade(title, 0.3), show_and_fade(hdr, 0.3), show_and_fade(st["mask"], 0.2),
                                 show_and_fade(st["discs"], 0.4), show_and_fade(st["att"], 0.4)))
        seq = Sequence(Func(self.app.ui.arch.set_layer, "layer 2 of 2"), self.view(R + Vec3(0.5, 0, 0.6), 0, 0, 36),
                       Parallel(cycle, anim), show_and_fade(lab, 0.3), self.sweep(X, 0.8),
                       Func(self.app.ui.arch.set_layer, "x 2 layers"))
        H1, H2 = LT["heads"]
        last = self.n - 1
        return ("LAYER 2: exactly the same steps (2 heads, add & norm, feed-forward, add & norm) with its own "
                "learned matrices. The last token '{}' now looks at: {}  /  {}. Real models stack 30-100 such "
                "layers.".format(disp(self.ttoks[last]), self.top_attn(H1["att"], last, 2),
                                 self.top_attn(H2["att"], last, 2)), seq)

    # ------------------------------------------------------------ 13 output: linear + softmax
    def s_output(self):
        R = region(11)
        g = self.board.attachNewNode("out")
        m = self.engine.m
        xl = self.T["x_final"][-1]
        logits = self.T["logits"]
        probs = self.T["probs"]
        inc = heatmap(g, xl[None, :].T, R.x - 13.0, R.z + 7.6, 0.4, 0.2, sc_of(xl), gap=0.1)
        text(g, "last token\n'{}'".format(disp(self.ttoks[-1])), Point3(R.x - 12.8, 0, R.z + 8.3), 0.3, WHITE)
        wout = self.matrix(g, m.w["Wout"], R.x - 11.4, R.z + 7.6, 0.06, 0.2, "W_out",
                           "{} x {}".format(self.d, self.V))
        # logits as bars for ALL vocabulary tokens
        lg = g.attachNewNode("logits")
        zb = R.z - 1.6
        sc = 2.0 / max(1e-6, np.abs(logits).max())
        for j, v in enumerate(logits):
            h = v * sc
            fill(lg, R.x - 11.4 + j * 0.115, zb + min(0, h), R.x - 11.4 + j * 0.115 + 0.08, zb + max(0, h),
                 YELLOW if j == int(np.argmax(logits)) else BLUE, 0.9, y=0)
        text(lg, "x W_out  ->  {} scores (logits), one for every token in the vocabulary".format(self.V),
             Point3(R.x - 11.4, 0, zb - 2.6), 0.3, GREY, align=TextNode.ALeft)
        top = np.argsort(-probs)[:5]
        pg = g.attachNewNode("probs")
        for r, j in enumerate(top):
            z = R.z + 6.6 - r * 0.85
            text(pg, disp(m.vocab[j]), Point3(R.x + 3.4, 0, z), 0.42, WHITE, align=TextNode.ARight)
            w = max(0.02, 7.0 * probs[j])
            fill(pg, R.x + 3.7, z - 0.1, R.x + 3.7 + w, z + 0.45, YELLOW if r == 0 else BLUE, 0.9, y=0)
            text(pg, "{:.2%}".format(probs[j]) if probs[j] >= 0.0001 else "<0.01%", Point3(R.x + 3.9 + w, 0, z), 0.36,
                 GREY, align=TextNode.ALeft)
        text(pg, "softmax -> probabilities (top 5)", Point3(R.x + 1.0, 0, R.z + 7.7), 0.38, GREY, align=TextNode.ALeft)
        for nd in (inc, wout, lg, pg):
            nd.hide()
        seq = Sequence(Func(self.app.ui.arch.set_layer, "x 2 layers"), self.mapkey("linear"),
                       self.view(R + Vec3(-1.0, 0, 2.6), 0, 0, 27), show_and_fade(inc, 0.4), show_and_fade(wout, 0.5),
                       Wait(0.3), show_and_fade(lg, 0.8), Wait(0.6), self.mapkey("softmax", "output"),
                       show_and_fade(pg, 0.6))
        best = m.vocab[int(top[0])]
        extra = ""
        if probs[top[0]] > 0.99:
            extra = (" (Almost 100%: this tiny model has only learned these few conversations by heart - "
                     "big models are much less certain.)")
        return ("OUTPUT: only the LAST token's column is used. It is multiplied by W_out, giving one score for each "
                "of the {} tokens the model knows; softmax turns the scores into probabilities. Winner: '{}' ({:.1%})."
                .format(self.V, disp(best), probs[top[0]]) + extra, seq)

    # ------------------------------------------------------------ deep dive OFF: one quick pass
    def s_quick(self):
        R = region(1)
        g = self.board.attachNewNode("quick")
        x0 = R.x - 6.0
        hdr = self.header(g, x0, R.z + 6.4)
        stages = [("input  (embedding + position)", self.T["x0"])]
        for l, LT in enumerate(self.T["layers"]):
            stages += [("layer {}: after attention + add & norm".format(l + 1), LT["x1"]),
                       ("layer {}: after feed-forward + add & norm".format(l + 1), LT["x_out"])]
        cols = [self.columns(g, M, x0, R.z + 5.6) for _, M in stages]
        for cs in cols:
            self.hide_all(cs)
        lab = text(g, "", Point3(x0 + (self.n - 1) * CS / 2, 0, R.z - 0.6), 0.4, YELLOW)
        keys = ["embed", "add1", "add2", "add1", "add2"]
        seq = Sequence(self.view(R + Vec3(0, 0, 2.5), 0, 0, 20), show_and_fade(hdr, 0.3))
        prev = None
        for k, ((title, _), cs) in enumerate(zip(stages, cols)):
            step = [Func(lab.node().setText, title),
                    self.mapkey(keys[k]) if k == 0 else Sequence(self.mapkey("attn"), Wait(0.5), self.mapkey(keys[k]))]
            if prev:
                step.append(Parallel(*[Sequence(fade_out(a, 0.3), Func(a.hide)) for a in prev]))
            step.append(self.sweep(cs, 0.5))
            step.append(Wait(0.6))
            seq.append(Sequence(*step))
            prev = cs
        return ("Inside the Transformer (deep dive OFF - press D for every detail): the {} token columns pass "
                "through 2 layers; in each layer attention lets tokens exchange information and the feed-forward "
                "network processes each token. Watch the numbers change.".format(self.n), seq)
