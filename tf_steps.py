# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Inside the Transformer, as ONE CONTINUOUS FLOW of data, with the REAL numbers of the tiny model.

The "residual stream" (one column of 32 numbers per token) travels to the right along a highway at the
top of the board. At every station a computation unfolds BELOW the highway, and its result rises back up
into the stream. The last token <ai> - the one whose column will predict the next token - is framed in
yellow the whole way, and the tracker (bottom right) always shows its current 32 numbers.

All highway vectors share ONE colour scale, so the same numbers always have the same colour.
Red = positive, blue = negative, dark = near zero.
"""
import numpy as np
from direct.interval.IntervalGlobal import (Func, LerpColorScaleInterval, LerpFunc, LerpPosInterval,
                                            LerpScaleInterval, Parallel, Sequence, Wait)
from panda3d.core import Point3, TextNode, Vec3

from kit import (BLUE, DIM, GREEN, GREY, ORANGE, RED, TOKEN_COLORS, WHITE, YELLOW, Arrow, Fonts, disc,
                 fade_in, fade_out, fill, heatmap, lines, rect, shape_label, text, token_box, value_color, vlabel)

CS = 0.95              # spacing between token columns
CH = 0.17              # cell height in a highway column (32 numbers)
HDR_Z = 9.1            # token boxes above the highway
HW_TOP = 8.3           # top of the highway columns
SQ = "√"

# stations along the highway (x of the highway centre)
S_EMB, S_ATT, S_CAT, S_NORM, S_FFN, S_ADD2, S_L2, S_OUT = 44.0, 74.0, 132.0, 154.0, 180.0, 204.0, 230.0, 258.0


def disp(t):
    if t.startswith("<sys"):
        return "<sys>"
    return t.strip() or t


def sc_of(M):
    M = np.asarray(M, dtype=float)
    M = M[np.isfinite(M)]
    return max(1e-6, 1.6 * float(np.std(M)) if M.size else 1.0)


def show_and_fade(node, dur=0.5):
    return Sequence(Func(node.show), fade_in(node, dur))


class Highway:
    """The residual stream: a row of token columns that travels to the right."""

    def __init__(self, st, M, label, xc):
        self.st = st
        self.n = st.n
        self.w = (self.n - 1) * CS
        self.g = st.board.attachNewNode("highway")
        self.g.setPos(xc - self.w / 2, 0, 0)
        self.bot = HW_TOP - st.d * CH
        self.hdr = st.header(self.g, 0, HDR_Z)
        h = self.n - 1
        self.hero = rect(self.g, h * CS - CS * 0.5, self.bot - 0.12, h * CS + CS * 0.5, HDR_Z + 0.42, YELLOW, 2.6,
                         y=-0.03)
        self.label = text(self.g, label, Point3(self.w / 2, 0, self.bot - 0.6), 0.36, GREY, Fonts.symbol)
        self.cols = st.columns(self.g, M, 0, HW_TOP, scale=st.rs)
        self.M = M

    def xc(self):
        return self.g.getX() + self.w / 2

    def col_x(self, i):
        """World x of column i's centre (at the highway's CURRENT position)."""
        return self.g.getX() + i * CS

    def morph(self, M, label, dur=0.6, stagger=0.04):
        """Cross-fade every column to new values (same place). State switches at build time."""
        new = self.st.columns(self.g, M, 0, HW_TOP, scale=self.st.rs)
        old = self.cols
        for c in new:
            c.hide()
        self.cols, self.M = new, M
        par = Parallel(Func(self.label.node().setText, label))
        for i, (a, b) in enumerate(zip(old, new)):
            par.append(Sequence(Wait(i * stagger), Parallel(fade_out(a, dur), show_and_fade(b, dur)), Func(a.hide)))
        return par

    def move(self, xc, dur=1.6):
        return LerpPosInterval(self.g, dur, Point3(xc - self.w / 2, 0, 0), blendType="easeInOut")


class TransformerSteps:
    """Mixin for Story. Needs: self.board, self.view(), self.orbit(), self.engine, self.app."""

    # ------------------------------------------------------------ setup
    def tf_setup(self):
        self.T = self.engine.trace_first()
        self.ttoks = self.T["tokens"]
        self.n = len(self.ttoks)
        self.tcolor = [GREY if t.startswith("<") else TOKEN_COLORS[i % len(TOKEN_COLORS)]
                       for i, t in enumerate(self.ttoks)]
        self.hi = self.n - 1                     # the hero: the last token, <ai>
        self.d = self.T["x0"].shape[1]
        self.dh = self.T["layers"][0]["heads"][0]["q"].shape[1]
        self.V = len(self.engine.m.vocab)
        stream = [self.T["emb"], self.T["x0"]] + [LT[k] for LT in self.T["layers"] for k in ("res1", "x1", "x_out")]
        self.rs = sc_of(np.vstack(stream))       # ONE colour scale for everything on the highway
        self.hw = None
        self.hx = None                            # where the highway centre will be after the current step
        self.g_embed = self.g_pos = self.g_att1 = self.g_head2 = self.g_concat = None
        self.g_norm = self.g_ffn = self.g_add2 = self.g_l2 = None

    # ------------------------------------------------------------ helpers
    def header(self, parent, x0, z, scale=0.25):
        g = parent.attachNewNode("hdr")
        for i, t in enumerate(self.ttoks):
            token_box(g, disp(t), self.tcolor[i], (x0 + i * CS, 0, z), scale)
        g.flattenStrong()
        return g

    def columns(self, parent, M, x0, z_top, ch=CH, scale=None, cs=CS):
        """One heatmap column per token (M is n x dim). Returns list of nodes."""
        scale = scale or sc_of(M)
        return [heatmap(parent, M[i:i + 1].T, x0 + i * cs - cs * 0.4, z_top, cs * 0.8, ch, scale, gap=0.1)
                for i in range(M.shape[0])]

    def frame(self, parent, x0, z_top, ncols, nrows, cs, ch, color=DIM):
        return rect(parent, x0 - cs * 0.5, z_top - nrows * ch - 0.08, x0 + (ncols - 0.5) * cs, z_top + 0.08, color, 1.2)

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

    def sweep(self, nodes, total=1.2):
        gap = total / max(1, len(nodes))
        return Parallel(*[Sequence(Wait(i * gap), show_and_fade(nd, 0.3)) for i, nd in enumerate(nodes)])

    def hide_all(self, nodes):
        for nd in nodes:
            nd.hide()

    def dim(self, *groups):
        """An earlier station stays visible as a dark trail, so it does not compete with the current one."""
        return Parallel(*[LerpColorScaleInterval(g, 0.6, (0.22, 0.22, 0.25, 1)) for g in groups if g])

    def retire(self, *groups):
        """Fade an earlier station out completely (when it would overlap the current one)."""
        return Parallel(*[Sequence(fade_out(g, 0.5), Func(g.hide)) for g in groups if g])

    def mapkey(self, *keys):
        return Func(self.app.ui.arch.highlight, *keys)

    def track(self, vec, where):
        return Func(self.app.ui.track, disp(self.ttoks[self.hi]), vec, self.rs, where)

    def top_attn(self, att, q, k=3):
        idx = np.argsort(-att[q])[:k]
        return ", ".join("'{}' {:.2f}".format(disp(self.ttoks[j]), att[q, j]) for j in idx)

    def fly(self, vec, x_from, z_from, ch_from, x_to, z_to, ch_to, dur=1.0, scale=None, cw=CS * 0.8, delay=0.0):
        """A copy of one column (centre x, top z) flies from one place to another, changing its cell height."""
        nd = heatmap(self.board, np.asarray(vec)[None, :].T, -cw / 2, 0, cw, ch_to, scale or self.rs, gap=0.1)
        nd.setPos(x_from, 0, z_from)
        nd.setSz(ch_from / ch_to)
        nd.hide()
        return nd, Sequence(Wait(delay), Func(nd.show),
                            Parallel(LerpPosInterval(nd, dur, Point3(x_to, 0, z_to), blendType="easeInOut"),
                                     LerpScaleInterval(nd, dur, Vec3(1, 1, 1), blendType="easeInOut")))

    def hero_x(self, station):
        """World x of the <ai> column when the highway stands at `station`."""
        return station - (self.n - 1) * CS / 2 + self.hi * CS

    def go(self, x, z=3.0, d=26.0, h=0.0, p=0.0):
        return self.view(Point3(x, 0, z), h, p, d)

    def move_hw(self, xc, dur=1.6):
        """Move the highway to a new station and let the camera follow it."""
        self.hx = xc
        return Parallel(self.hw.move(xc, dur), self.go(xc, 3.0, 26))

    # ------------------------------------------------------------ 1 embedding
    def s_embed(self):
        x = S_EMB
        E = self.engine.m.w["E"]
        ids = self.T["ids"]
        g = self.board.attachNewNode("embed")
        self.g_embed = g
        self.hw = Highway(self, self.T["emb"], "token embeddings  ({} x {})".format(self.n, self.d), x)
        self.hx = x
        hw = self.hw
        for nd in [hw.hdr, hw.hero, hw.label] + hw.cols:
            nd.hide()
        ex0, ez0, ecw, ech = x - 13.6, HW_TOP + 0.4, 0.09, 0.05
        tab = self.matrix(g, E, ex0, ez0, ecw, ech, "embedding table",
                          "{} tokens x {} numbers".format(self.V, self.d), scale=self.rs)
        text(tab, "one learned row per token", Point3(ex0 + self.d * ecw / 2, 0, ez0 + 0.9), 0.26, GREY)
        hl = g.attachNewNode("rows")
        for tid in ids:
            rect(hl, ex0 - 0.08, ez0 - (tid + 1) * ech, ex0 + self.d * ecw + 0.08, ez0 - tid * ech, YELLOW, 1.4,
                 y=-0.02)
        tab.hide()
        hl.hide()
        # the token boxes from the tokenizer step fly over to the highway
        moves = Parallel()
        for i, nd in enumerate(getattr(self, "a_tokens", [])):
            moves.append(Sequence(Wait(0.04 * i), LerpPosInterval(nd, 1.6, Point3(hw.col_x(i) - self.a_tok_x[i], 0,
                                                                                   HDR_Z - 2.0), blendType="easeInOut"),
                                  Func(nd.hide)))
        flies = Parallel()
        for i, tid in enumerate(ids):
            row = heatmap(self.board, E[tid:tid + 1], ex0, ez0 - tid * ech, ecw, ech, self.rs, gap=0.08)
            row.hide()
            dx = hw.col_x(i) - (ex0 + self.d * ecw / 2)
            flies.append(Sequence(Wait(0.1 * i), Func(row.show),
                                  LerpPosInterval(row, 0.8, Point3(dx, 0, HW_TOP - 2.0 - (ez0 - tid * ech)),
                                                  blendType="easeInOut"),
                                  fade_out(row, 0.2), Func(row.hide), show_and_fade(hw.cols[i], 0.25)))
        seq = Sequence(self.mapkey("embed"), self.go(x - 3.5, 3.5, 27),
                       Parallel(moves, Sequence(Wait(1.2), show_and_fade(hw.hdr, 0.4), show_and_fade(hw.hero, 0.4))),
                       show_and_fade(tab, 0.5), show_and_fade(hl, 0.4), flies, show_and_fade(hw.label, 0.3),
                       self.track(self.T["emb"][self.hi], "token embedding"))
        return ("TOKEN EMBEDDING. Each token ID picks its row in a learned table ({} x {}); the row becomes that "
                "token's COLUMN of {} numbers. These columns are the 'residual stream' that now travels through the "
                "model. Follow the YELLOW column, the last token <ai>: its column will predict the next token. "
                "(Tracker: bottom right.)".format(self.V, self.d, self.d), seq)

    # ------------------------------------------------------------ 2 positional encoding
    def s_position(self):
        hw = self.hw
        x = self.hx
        g = self.board.attachNewNode("pos")
        self.g_pos = g
        top = hw.bot - 1.6
        pe = self.columns(g, self.T["pe"], hw.g.getX(), top, scale=self.rs)
        self.hide_all(pe)
        lab = text(g, "positional encoding: a fixed wave pattern for position 0, 1, 2 ... {}".format(self.n - 1),
                   Point3(x, 0, top - self.d * CH - 0.6), 0.34, YELLOW)
        lab.hide()
        rise = Parallel()
        for i, c in enumerate(pe):
            rise.append(Sequence(Wait(0.05 * i), LerpPosInterval(c, 0.9, Point3(0, 0, HW_TOP - top), blendType="easeIn"),
                                 Func(c.hide)))
        seq = Sequence(self.mapkey("pos"), Parallel(self.go(x, 0.0, 30), self.dim(self.g_embed)), show_and_fade(lab, 0.3), self.sweep(pe, 1.0),
                       Wait(0.8), self.mapkey("pos", "embed"), rise, hw.morph(self.T["x0"], "embedding + position"),
                       self.track(self.T["x0"][self.hi], "added position"))
        return ("POSITIONAL ENCODING. Attention alone cannot tell word order. So every position gets its own fixed "
                "pattern of sine/cosine waves (the stripes below), and it is ADDED to the column above it. Watch the "
                "highway columns (and the tracker) change: now each column also knows where its token stands.", seq)

    # ------------------------------------------------------------ 3 Q, K, V of head 1
    def _qkv(self, layer, head, xc, g, fast=False):
        """Q, K, V matrices below the highway at station xc. Returns (nodes dict, interval)."""
        H = self.T["layers"][layer]["heads"][head]
        x0 = xc - (self.n - 1) * CS / 2
        out, seq = {}, Parallel()
        for j, (nm, col) in enumerate((("Q", YELLOW), ("K", GREEN), ("V", RED))):
            W = H["W" + nm.lower()]
            M = H[nm.lower()]
            zt = 1.4 - j * 3.1
            mat = self.matrix(g, W, xc + 7.3 + j * 2.5, HW_TOP - 0.6, 0.12, 0.075, "W_" + nm,
                              "{} x {}".format(self.d, self.dh), color=col)
            mat.hide()
            cols = self.columns(g, M, x0, zt, ch=0.15)
            self.hide_all(cols)
            lab = g.attachNewNode("lab")
            vlabel(lab, nm, "", (x0 - 1.5, 0, zt - 1.4), 0.55, col)
            shape_label(lab, "{} x {}".format(self.n, self.dh), Point3(x0 - 1.5, 0, zt - 2.0), 0.28)
            lab.hide()
            drops = Parallel()
            for i in range(self.n):
                # a copy of the highway column drops down and becomes the Q / K / V column
                nd, iv = self.fly(self.hw.M[i], x0 + i * CS, HW_TOP, CH, x0 + i * CS, zt, 0.15 * self.dh / self.d,
                                  dur=0.7, delay=0.03 * i)
                drops.append(Sequence(iv, Func(nd.hide), show_and_fade(cols[i], 0.2)))
            seq.append(Sequence(Wait(j * (0.9 if fast else 1.5)), show_and_fade(mat, 0.3), drops, show_and_fade(lab, 0.3)))
            out[nm] = (cols, x0, zt)
        return out, seq

    def s_qkv(self):
        x = self.hx
        g = self.board.attachNewNode("qkv1")
        self.g_att1 = g
        nodes, seq = self._qkv(0, 0, x, g)
        self.qkv1 = nodes
        return ("LAYER 1, ATTENTION, HEAD 1. Copies of every column drop down and are multiplied by three learned "
                "matrices: W_Q gives a QUERY ('what am I looking for?'), W_K a KEY ('what do I offer?'), W_V a VALUE "
                "('what I pass on'). 32 numbers in, 16 out, for all {} tokens at once.".format(self.n),
                Sequence(self.mapkey("attn"), Parallel(self.go(x + 3.0, -0.5, 30), self.retire(self.g_pos)), seq))

    # ------------------------------------------------------------ 4 scores table
    def _table(self, parent, H, x0, z0, cw=0.9, ch=0.5, num=0.2):
        """Q/K table: rows = keys, columns = queries. Returns (hdr, states)."""
        n = self.n
        hdr = parent.attachNewNode("th")
        for i, t in enumerate(self.ttoks):
            token_box(hdr, disp(t), self.tcolor[i], (x0 + (i + 0.5) * cw, 0, z0 + 0.35), 0.19 * cw / 0.9)
            text(hdr, disp(t), Point3(x0 - 0.15, 0, z0 - (i + 0.64) * ch), 0.2 * ch / 0.5, self.tcolor[i],
                 align=TextNode.ARight)
        vlabel(hdr, "Q", "", (x0 + n * cw / 2, 0, z0 + 0.95), 0.42, YELLOW)
        vlabel(hdr, "K", "", (x0 - 1.8, 0, z0 - n * ch / 2), 0.42, GREEN)
        lines(hdr, [[(x0, 0, z0), (x0 + n * cw, 0, z0)], [(x0, 0, z0), (x0, 0, z0 - n * ch)]], GREY, 1.2)
        hdr.flattenStrong()
        hdr.hide()
        st = {}
        for key, M, fmt in (("raw", H["raw"], "{:.1f}"), ("scaled", H["scaled"], "{:.1f}"), ("att", H["att"], "{:.2f}")):
            lo, up = parent.attachNewNode(key), parent.attachNewNode(key + "_hi")
            for q in range(n):
                for k in range(n):
                    if key == "att" and k > q:
                        continue
                    v = M[q, k]
                    col = (WHITE if v > 0.15 else GREY) if key == "att" else WHITE
                    text(up if k > q else lo, fmt.format(v), Point3(x0 + (q + 0.5) * cw, 0, z0 - (k + 0.64) * ch),
                         num, col)
            for nd in (lo, up):
                nd.flattenStrong()
                nd.hide()
            st[key], st[key + "_hi"] = lo, up
        m = parent.attachNewNode("mask")
        for q in range(n):
            for k in range(q + 1, n):
                fill(m, x0 + q * cw + 0.03, z0 - (k + 1) * ch + 0.03, x0 + (q + 1) * cw - 0.03, z0 - k * ch - 0.03,
                     (0.35, 0.35, 0.4, 1), 0.35)
                text(m, "-∞", Point3(x0 + (q + 0.5) * cw, -0.01, z0 - (k + 0.64) * ch), num, GREY, Fonts.symbol)
        m.flattenStrong()
        m.hide()
        st["mask"] = m
        dsc = parent.attachNewNode("discs")
        for q in range(n):
            for k in range(q + 1):
                b = 0.12 + 0.7 * min(1.0, H["att"][q, k] * 1.5)
                d = disc(dsc, min(cw, ch) * 0.42, (b * 0.8, b * 0.8, b * 0.8, 1), 20)
                d.setPos(x0 + (q + 0.5) * cw, 0.02, z0 - (k + 0.5) * ch)
        dsc.flattenStrong()
        dsc.hide()
        st["discs"] = dsc
        return hdr, st

    def s_scores(self):
        H = self.T["layers"][0]["heads"][0]
        g = self.g_att1
        x0, z0, cw, ch = self.hx + 16.0, HW_TOP - 1.4, 0.9, 0.5
        hdr, st = self._table(g, H, x0, z0)
        self.tab1 = (x0, z0, cw, ch)
        n = self.n
        hl = rect(g, x0 + self.hi * cw + 0.02, z0 - n * ch, x0 + (self.hi + 1) * cw - 0.02, z0 + 0.62, YELLOW, 2.6,
                  y=-0.03)
        hl.hide()
        info = text(g, "", Point3(x0 + n * cw + 0.5, 0, z0 - 0.3), 0.32, WHITE, Fonts.symbol, TextNode.ALeft, wrap=14)
        sh = shape_label(g, "Q Kᵀ = {} x {}".format(n, n), Point3(x0 + n * cw / 2, 0, z0 - n * ch - 0.5), 0.32)
        sh.hide()
        # Q columns fly up to become the column heads, K columns fly to become the row heads
        qcols, qx0, qz = self.qkv1["Q"]
        kcols, kx0, kz = self.qkv1["K"]
        flies = Parallel()
        for i in range(n):
            nd, iv = self.fly(H["q"][i], qx0 + i * CS, qz, 0.15, x0 + (i + 0.5) * cw, z0 + 1.45, 0.06, dur=1.0,
                              scale=sc_of(H["q"]), cw=cw * 0.6, delay=0.04 * i)
            flies.append(Sequence(iv, fade_out(nd, 0.3), Func(nd.hide)))
            nd2, iv2 = self.fly(H["k"][i], kx0 + i * CS, kz, 0.15, x0 - 2.8, z0 - (i + 0.15) * ch, 0.02, dur=1.0,
                                scale=sc_of(H["k"]), cw=0.9, delay=0.6 + 0.04 * i)
            flies.append(Sequence(iv2, fade_out(nd2, 0.3), Func(nd2.hide)))

        def say(s):
            info.node().setText(s)
        seq = Sequence(
            self.mapkey("attn"), self.go(x0 + 4.0, 1.6, 22), flies, show_and_fade(hdr, 0.4), show_and_fade(sh, 0.3),
            Func(say, "1. score = Q . K\nfor every pair"),
            Parallel(show_and_fade(st["raw"], 0.7), show_and_fade(st["raw_hi"], 0.7)), Wait(1.0),
            Func(say, "2. divide by {}{} = {:.0f}".format(SQ, self.dh, np.sqrt(self.dh))),
            Parallel(fade_out(st["raw"], 0.3), fade_out(st["raw_hi"], 0.3)), Func(st["raw"].hide),
            Func(st["raw_hi"].hide), Parallel(show_and_fade(st["scaled"], 0.4), show_and_fade(st["scaled_hi"], 0.4)),
            Wait(0.9), Func(say, "3. MASK: no looking\nat LATER tokens\n-> -∞"),
            fade_out(st["scaled_hi"], 0.3), Func(st["scaled_hi"].hide), show_and_fade(st["mask"], 0.5), Wait(0.9),
            Func(say, "4. softmax per\ncolumn -> weights\n(add up to 1)"),
            fade_out(st["scaled"], 0.3), Func(st["scaled"].hide), show_and_fade(st["discs"], 0.4),
            show_and_fade(st["att"], 0.4), Wait(0.4), show_and_fade(hl, 0.4),
            Func(say, "our <ai> column:\n" + self.top_attn(H["att"], self.hi).replace(", ", "\n")))
        return ("ATTENTION SCORES. The Q columns fly up to label the table's columns, the K columns its rows. Every "
                "query is compared with every key (dot product), scaled, MASKED so no token sees later tokens, then "
                "softmax turns each column into weights. The yellow column is our <ai>: it attends most to {}."
                .format(self.top_attn(H["att"], self.hi)), seq)

    # ------------------------------------------------------------ 5 weighted sum of V
    def _wsum(self, H, g, x0, z_top, vsrc, label):
        """weights x V for the hero, then the head output for all tokens. Returns (out_cols, out_x0, out_z, iv)."""
        n, cw = self.n, 0.9
        att, V, out = H["att"], H["v"], H["out"]
        vcols, vx0, vz = vsrc
        vs = sc_of(V)
        copies = []
        moves = Parallel()
        for k in range(n):
            nd, iv = self.fly(V[k], vx0 + k * CS, vz, 0.15, x0 + (k + 0.5) * cw, z_top, 0.15, dur=1.1, scale=vs,
                              cw=cw * 0.7, delay=0.03 * k)
            copies.append(nd)
            moves.append(iv)
        wtxt = g.attachNewNode("w")
        for k in range(n):
            w = att[self.hi, k]
            text(wtxt, "x {:.2f}".format(w), Point3(x0 + (k + 0.5) * cw, 0, z_top + 0.25), 0.2,
                 YELLOW if w > 0.1 else GREY)
        wtxt.flattenStrong()
        wtxt.hide()
        bright = [0.12 + 0.88 * min(1.0, att[self.hi, k] * 2.5) for k in range(n)]
        dims = Parallel(*[LerpColorScaleInterval(copies[k], 0.6, (bright[k], bright[k], bright[k], 1))
                          for k in range(n)])
        ox = x0 + n * cw + 1.6
        res = heatmap(g, out[self.hi:self.hi + 1].T, ox - 0.35, z_top, 0.7, 0.15, sc_of(out), gap=0.1)
        rect(res, ox - 0.45, z_top - self.dh * 0.15 - 0.08, ox + 0.45, z_top + 0.08, YELLOW, 2.0)
        res.hide()
        conv = lines(g, [[(x0 + (k + 0.5) * cw, 0, z_top - self.dh * 0.15 - 0.15),
                          (ox, 0, z_top - self.dh * 0.15 - 0.6)] for k in range(n) if att[self.hi, k] > 0.05],
                     YELLOW[:3] + (0.6,), 1.4)
        conv.hide()
        rl = text(g, "= {} output\nfor <ai>".format(label), Point3(ox, 0, z_top - self.dh * 0.15 - 1.2), 0.28, YELLOW)
        rl.hide()
        oz = z_top - self.dh * 0.15 - 2.4
        ocols = self.columns(g, out, x0 + cw * 0.5, oz, ch=0.15, cs=cw)
        self.hide_all(ocols)
        ol = text(g, "{} output for ALL tokens  ({} x {})".format(label, n, self.dh),
                  Point3(x0 + n * cw / 2, 0, oz - self.dh * 0.15 - 0.45), 0.3, WHITE)
        ol.hide()
        iv = Sequence(moves, show_and_fade(wtxt, 0.4), dims, Wait(0.3), show_and_fade(conv, 0.4),
                      show_and_fade(res, 0.4), show_and_fade(rl, 0.3), Wait(0.5), self.sweep(ocols, 0.8),
                      show_and_fade(ol, 0.3))
        return ocols, x0 + cw * 0.5, oz, iv

    def s_weighted(self):
        H = self.T["layers"][0]["heads"][0]
        x0, z0, cw, ch = self.tab1
        ztop = z0 - self.n * ch - 1.6
        ocols, ox0, oz, iv = self._wsum(H, self.g_att1, x0, ztop, self.qkv1["V"], "head 1")
        self.h1out = (ocols, ox0, oz)
        top = np.argsort(-H["att"][self.hi])[:2]
        return ("WEIGHTED SUM. The V columns fly over to the table. Each is multiplied by <ai>'s weight from the "
                "yellow column (strong weights stay bright, weak ones fade) and they are added up: that is what "
                "<ai> collects from the others - mostly from '{}' and '{}'. Every token does the same at once."
                .format(disp(self.ttoks[top[0]]), disp(self.ttoks[top[1]])),
                Sequence(self.mapkey("attn"), self.go(x0 + 6.0, -0.8, 31), iv))

    # ------------------------------------------------------------ 6 head 2 (complete, faster)
    def s_head2(self):
        H = self.T["layers"][0]["heads"][1]
        H1 = self.T["layers"][0]["heads"][0]
        g = self.board.attachNewNode("head2")
        bx = self.hx + 34.0
        # its own Q, K, V (compact): copies of the highway drop into three small matrices
        mats = []
        for j, (nm, col) in enumerate((("Q", YELLOW), ("K", GREEN), ("V", RED))):
            M = H[nm.lower()]
            zt = 1.4 - j * 2.6
            cols = self.columns(g, M, bx, zt, ch=0.13, cs=0.36)
            self.hide_all(cols)
            lab = text(g, nm + "  (W_{} of head 2)".format(nm), Point3(bx - 0.3, 0, zt + 0.25), 0.26, col,
                       align=TextNode.ALeft)
            lab.hide()
            mats.append((cols, lab, zt))
        tx0, tz0 = bx + 11.0, HW_TOP - 1.4
        hdr, st = self._table(g, H, tx0, tz0)
        hl = rect(g, tx0 + self.hi * 0.9 + 0.02, tz0 - self.n * 0.5, tx0 + (self.hi + 1) * 0.9 - 0.02, tz0 + 0.62,
                  YELLOW, 2.6, y=-0.03)
        hl.hide()
        title = text(g, "HEAD 2: same steps, its own matrices", Point3(tx0 + self.n * 0.45 + 1.0, 0, HDR_Z + 0.9), 0.5,
                     WHITE)
        hc = bx + 2.2                                   # the highway stands above head 2's Q, K, V
        hx0 = hc - (self.n - 1) * CS / 2
        title.hide()
        drops = Parallel()
        for j, (cols, lab, zt) in enumerate(mats):
            seqj = Sequence(Wait(0.6 * j), show_and_fade(lab, 0.2))
            for i in range(self.n):
                nd, iv = self.fly(self.hw.M[i], hx0 + i * CS, HW_TOP, CH, bx + i * 0.36, zt, 0.13 * self.dh / self.d,
                                  dur=0.9, cw=0.3, delay=0.02 * i)
                seqj.append(Sequence(iv, Func(nd.hide), Func(cols[i].show)))
            drops.append(seqj)
        ocols, ox0, oz, iv = self._wsum_simple(H, g, tx0, tz0 - self.n * 0.5 - 1.6, "head 2")
        self.h2out = (ocols, ox0, oz)
        self.g_head2 = g
        self.hx = hc
        seq = Sequence(self.mapkey("attn"), self.dim(self.g_att1),
                       Parallel(self.hw.move(hc, 1.6), self.go(bx + 9.6, 1.5, 33)),
                       show_and_fade(title, 0.4), drops,
                       show_and_fade(hdr, 0.4), show_and_fade(st["mask"], 0.3), show_and_fade(st["discs"], 0.4),
                       show_and_fade(st["att"], 0.4), show_and_fade(hl, 0.3), iv)
        return ("MULTI-HEAD: a second head repeats ALL the steps with its own W_Q, W_K, W_V, so it can look for a "
                "different kind of relation. For <ai>: head 1 -> {};  head 2 -> {}."
                .format(self.top_attn(H1["att"], self.hi, 2), self.top_attn(H["att"], self.hi, 2)), seq)

    def _wsum_simple(self, H, g, x0, z_top, label):
        """Head-2 version of the weighted sum (V comes from the compact matrix)."""
        n, cw = self.n, 0.9
        out = H["out"]
        ocols = self.columns(g, out, x0 + cw * 0.5, z_top, ch=0.15, cs=cw)
        self.hide_all(ocols)
        ol = text(g, "weights x V  =  {} output  ({} x {})".format(label, n, self.dh),
                  Point3(x0 + n * cw / 2, 0, z_top - self.dh * 0.15 - 0.45), 0.3, WHITE)
        ol.hide()
        hr = rect(g, x0 + (self.hi + 0.5) * cw - 0.42, z_top - self.dh * 0.15 - 0.08, x0 + (self.hi + 0.5) * cw + 0.42,
                  z_top + 0.08, YELLOW, 2.0)
        hr.hide()
        return ocols, x0 + cw * 0.5, z_top, Sequence(self.sweep(ocols, 0.8), show_and_fade(hr, 0.2),
                                                     show_and_fade(ol, 0.3))

    # ------------------------------------------------------------ 7 concat x W_O = delta E
    def s_concat(self):
        LT = self.T["layers"][0]
        g = self.board.attachNewNode("concat")
        x = S_CAT
        x0 = x - (self.n - 1) * CS / 2
        z1, z2 = 1.4, 1.4 - self.dh * 0.15
        flies = Parallel()
        for (ocols, ox0, oz), zt, H in ((self.h1out, z1, LT["heads"][0]), (self.h2out, z2, LT["heads"][1])):
            for i in range(self.n):
                nd, iv = self.fly(H["out"][i], ox0 + i * 0.9, oz, 0.15, x0 + i * CS, zt, 0.15, dur=1.6,
                                  scale=sc_of(H["out"]), delay=0.03 * i)
                flies.append(Sequence(iv, Func(nd.wrtReparentTo, g)))    # belongs to this station from now on
        lab = g.attachNewNode("lab")
        text(lab, "head 1", Point3(x0 - 1.6, 0, z1 - 1.2), 0.3, YELLOW)
        text(lab, "head 2", Point3(x0 - 1.6, 0, z2 - 1.2), 0.3, YELLOW)
        shape_label(lab, "concat = {} x {}".format(self.n, 2 * self.dh), Point3(x, 0, z2 - self.dh * 0.15 - 0.45), 0.3)
        lab.hide()
        wo = self.matrix(g, LT["Wo"], x + 7.2, 1.4, 0.13, 0.13, "W_O", "{} x {}".format(self.d, self.d))
        wo.hide()
        dz = z2 - self.dh * 0.15 - 1.4
        de = self.columns(g, LT["delta"], x0, dz, ch=0.12, scale=self.rs)
        self.hide_all(de)
        dl = g.attachNewNode("dl")
        vlabel(dl, "ΔE", "", (x0 - 1.6, 0, dz - 2.2), 0.5, YELLOW)
        shape_label(dl, "concat x W_O = {} x {}".format(self.n, self.d), Point3(x, 0, dz - self.d * 0.12 - 0.45), 0.3)
        hr = rect(dl, x0 + self.hi * CS - CS * 0.45, dz - self.d * 0.12 - 0.08, x0 + self.hi * CS + CS * 0.45, dz + 0.08,
                  YELLOW, 2.2)
        dl.hide()
        hr.hide()
        self.delta = (de, x0, dz)
        self.g_concat = g
        self.hx = x
        seq = Sequence(self.mapkey("attn"), self.dim(self.g_head2),
                       Parallel(self.hw.move(x, 2.2), self.go(x + 3.0, -0.3, 33), flies),
                       show_and_fade(lab, 0.3), Wait(0.3),
                       show_and_fade(wo, 0.4), self.sweep(de, 1.0), show_and_fade(dl, 0.4), Func(hr.show))
        return ("The highway moves on, and both head outputs fly along below it and are stacked (16 + 16 = 32 numbers per token), then mixed by one "
                "more learned matrix W_O. The result ΔE is the CHANGE each token wants to make to its column "
                "after looking at the others. <ai>'s change is framed.", seq)

    # ------------------------------------------------------------ 8 residual: add delta E into the stream
    def s_residual(self):
        LT = self.T["layers"][0]
        hw = self.hw
        de, dx0, dz = self.delta
        rise = Parallel()
        for i in range(self.n):
            nd, iv = self.fly(LT["delta"][i], dx0 + i * CS, dz, 0.12, dx0 + i * CS, HW_TOP, CH, dur=1.0, delay=0.04 * i)
            rise.append(Sequence(iv, fade_out(nd, 0.3), Func(nd.hide)))
        plus = text(self.board, "+", Point3(S_CAT - 7.6, 0, 5.0), 1.0, YELLOW)
        plus.hide()
        sp, arrows = self._space(Point3(S_CAT, 12, 17.5), LT["x_in"], LT["res1"])
        sp.hide()
        c3 = Point3(S_CAT, 12, 18.2)
        seq = Sequence(self.mapkey("add1"), self.go(S_CAT, 1.0, 30), show_and_fade(plus, 0.3), rise,
                       hw.morph(LT["res1"], "after attention: x + ΔE"), fade_out(plus, 0.3),
                       self.track(LT["res1"][self.hi], "layer 1: added attention"), Wait(0.8),
                       self.view(c3, -30, 24, 15), Func(sp.show), fade_in(sp, 0.6),
                       Parallel(*[LerpFunc(a.grow, fromData=0.001, toData=1.0, duration=0.8) for a in arrows["before"]]),
                       Wait(0.5),
                       Parallel(*[self._move_arrow(a, b) for a, b in zip(arrows["before"], arrows["after"])],
                                self.orbit(c3, -30, 25, 24, 15, 5.0)),
                       Parallel(self.go(S_CAT, 3.0, 26), self.retire(sp), self.dim(self.g_concat)))
        return ("ADD (residual connection): every token's change ΔE rises up into the highway and is "
                "ADDED to its column - all {} tokens at the same time. This is how what each token learned from the "
                "others gets written into it. Above: the same vectors squeezed into 3D - every arrow moves at once."
                .format(self.n), seq)

    def _space(self, center, A, B):
        sp = self.board.attachNewNode("space")
        sp.setPos(center)
        both = np.vstack([A, B])
        mu = both.mean(0)
        _, _, Vt = np.linalg.svd(both - mu, full_matrices=False)
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
            col = YELLOW if i == self.hi else self.tcolor[i]
            a = Arrow(sp, Point3(0, 0, 0), Point3(*pa[i]), col, 3.6 if i == self.hi else 2.4)
            lab = text(a.root, disp(self.ttoks[i]), Point3(*(pa[i] * 1.12)), 0.36 if i == self.hi else 0.28, col)
            lab.setBillboardPointEye()
            a.label = lab
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
        LT = self.T["layers"][0]
        hw = self.hw
        x = S_NORM
        g = self.board.attachNewNode("norm")
        self.g_norm = g
        v = LT["res1"][self.hi]
        normed = (v - v.mean()) / np.sqrt(v.var() + 1e-5)
        out = LT["x1"][self.hi]
        sc = 1.15 / max(np.abs(v).max(), np.abs(normed).max(), np.abs(out).max())
        bars = []
        for k, (vals, col, title) in enumerate(((v, WHITE, "<ai> before"), (normed, YELLOW, "minus mean, / spread"),
                                               (out, GREEN, "x gain + bias (learned)"))):
            bg = g.attachNewNode("bars")
            zb = 0.4 - k * 2.9
            for j, val in enumerate(vals):
                h = val * sc
                fill(bg, x - 7.0 + j * 0.4, zb + min(0, h), x - 7.0 + j * 0.4 + 0.28, zb + max(0, h), col, 0.9, y=0)
            lines(bg, [[(x - 7.2, 0, zb), (x - 7.0 + self.d * 0.4, 0, zb)]], GREY, 1.0)
            text(bg, title, Point3(x - 7.2, 0, zb + 1.3), 0.3, col, align=TextNode.ALeft)
            text(bg, "mean {:+.2f}  spread {:.2f}".format(vals.mean(), vals.std()), Point3(x + 6.0, 0, zb + 1.3), 0.28,
                 GREY, align=TextNode.ARight)
            bg.hide()
            bars.append(bg)
        nd, drop = self.fly(v, self.hero_x(x), HW_TOP, CH, x - 8.2, 0.4 + 2.7, 0.08, dur=1.0)
        seq = Sequence(self.mapkey("add1"), self.dim(self.g_concat), self.move_hw(x, 1.6), self.go(x, 1.0, 27), drop, Func(nd.hide),
                       Parallel(*[Sequence(Wait(0.7 * k), show_and_fade(b, 0.4)) for k, b in enumerate(bars)]),
                       Wait(0.4), hw.morph(LT["x1"], "after Add & Norm"),
                       self.track(LT["x1"][self.hi], "layer 1: Add & Norm"))
        return ("NORM (layer normalization): each column is shifted to mean 0 and scaled to spread 1, then multiplied "
                "by a learned gain and bias (shown for <ai> as bars). It keeps the numbers in a stable range layer "
                "after layer. Then the whole highway is updated.", seq)

    # ------------------------------------------------------------ 10 feed-forward
    def s_ffn(self):
        LT = self.T["layers"][0]
        x = S_FFN
        g = self.board.attachNewNode("ffn")
        self.g_ffn = g
        i = self.hi
        x_in, pre, hid, ff = LT["x1"][i], LT["hid_pre"][i], LT["hid"][i], LT["ff"][i]
        F = len(pre)
        ztop = 1.0
        ix, ox = x - 7.5, x + 7.5
        dz = 0.28
        nd, drop = self.fly(x_in, self.hero_x(x), HW_TOP, CH, ix, ztop, 0.2, dur=1.0)
        text(g, "<ai>  ({} numbers)".format(self.d), Point3(ix, 0, ztop + 0.45), 0.3, YELLOW)
        neurons = g.attachNewNode("neurons")
        dots = []
        sc = sc_of(pre)
        per = F // 4
        for j in range(F):
            d = disc(neurons, 0.11, value_color(pre[j], sc), 14)
            d.setPos(x - 0.6 + (j // per) * 0.42, -0.01, ztop - (j % per) * dz)
            dots.append(d)
        text(neurons, "{} neurons".format(F), Point3(x, 0, ztop + 0.45), 0.32, WHITE)
        shape_label(neurons, "x W1 + b1:  {} -> {}".format(self.d, F), Point3(x - 3.8, 0, ztop + 0.45), 0.28)
        relu = text(g, "ReLU: negative -> 0\n{} of {} switched off".format(int((pre <= 0).sum()), F),
                    Point3(x, 0, ztop - per * dz - 0.3), 0.32, ORANGE)
        outc = heatmap(g, ff[None, :].T, ox - 0.4, ztop, 0.8, 0.2, self.rs, gap=0.1)
        self.ffn_out = (outc, ztop, 0.2, ox)
        olab = g.attachNewNode("ol")
        text(olab, "<ai> output", Point3(ox, 0, ztop + 0.45), 0.3, GREEN)
        shape_label(olab, "x W2 + b2:  {} -> {}".format(F, self.d), Point3(x + 3.8, 0, ztop + 0.45), 0.28)
        cons = g.attachNewNode("cons")
        W1, W2 = self.engine.m.w["l0.W1"], self.engine.m.w["l0.W2"]
        for j in np.argsort(-hid)[:6]:
            p = dots[j].getPos()
            for k in np.argsort(-np.abs(W1[:, j]))[:2]:
                lines(cons, [[(ix + 0.4, 0, ztop - (k + 0.5) * 0.2), (p.x, 0, p.z)]], BLUE[:3] + (0.6,), 1.2)
            for k in np.argsort(-np.abs(W2[j]))[:2]:
                lines(cons, [[(p.x, 0, p.z), (ox - 0.4, 0, ztop - (k + 0.5) * 0.2)]], RED[:3] + (0.6,), 1.2)
        for nd2 in (neurons, relu, outc, olab, cons):
            nd2.hide()
        off = [d for j, d in enumerate(dots) if pre[j] <= 0]
        seq = Sequence(self.mapkey("ffn"), self.dim(self.g_norm), self.move_hw(x, 1.4), self.go(x, -0.3, 28), drop,
                       show_and_fade(neurons, 0.6), show_and_fade(cons, 0.4), Wait(0.4), show_and_fade(relu, 0.3),
                       Parallel(*[LerpColorScaleInterval(d, 0.5, (0.15, 0.15, 0.15, 1)) for d in off]), Wait(0.4),
                       show_and_fade(olab, 0.3), show_and_fade(outc, 0.4))
        return ("FEED FORWARD (the MLP). <ai>'s column drops into a small 2-layer network: expand to {} neurons, ReLU "
                "switches off the negative ones (dark), compress back to {}. Only the strongest connections are drawn. "
                "This is where each token 'thinks on its own', after attention let it look at the others."
                .format(F, self.d), seq)

    # ------------------------------------------------------------ 11 add & norm (end of layer 1)
    def s_add2(self):
        LT = self.T["layers"][0]
        hw = self.hw
        g = self.board.attachNewNode("add2")
        self.g_add2 = g
        fz = 1.4
        fx0 = S_ADD2 - (self.n - 1) * CS / 2
        fcols = self.columns(g, LT["ff"], fx0, fz, scale=self.rs)
        self.hide_all(fcols)
        fl = text(g, "the SAME network ran on EVERY token:  feed-forward output  {} x {}".format(self.n, self.d),
                  Point3(S_ADD2, 0, fz - self.d * CH - 0.5), 0.3, WHITE)
        fl.hide()
        hr = rect(g, fx0 + self.hi * CS - CS * 0.45, fz - self.d * CH - 0.08, fx0 + self.hi * CS + CS * 0.45, fz + 0.08,
                  YELLOW, 2.2)
        hr.hide()
        # <ai>'s own result (from the network on the left) slides over into its place
        _, _, _, ox = self.ffn_out
        nd, slide = self.fly(LT["ff"][self.hi], ox, fz, 0.2, fx0 + self.hi * CS, fz, CH, dur=1.2)
        rise = Parallel()
        for i in range(self.n):
            rise.append(Sequence(Wait(0.04 * i), LerpPosInterval(fcols[i], 0.9, Point3(0, 0, HW_TOP - fz)),
                                 fade_out(fcols[i], 0.2), Func(fcols[i].hide)))
        seq = Sequence(self.mapkey("add2"), Parallel(self.move_hw(S_ADD2, 1.6), slide), Func(nd.hide),
                       Func(fcols[self.hi].show), self.sweep([c for i, c in enumerate(fcols) if i != self.hi], 0.8),
                       show_and_fade(fl, 0.3), show_and_fade(hr, 0.2), Wait(0.8), fade_out(hr, 0.2), fade_out(fl, 0.2),
                       rise, hw.morph(LT["x_out"], "layer 1 output"), self.track(LT["x_out"][self.hi], "end of layer 1"))
        return ("ADD & NORM again: every token went through the SAME feed-forward network (their outputs appear next "
                "to <ai>'s). They rise into the highway, are ADDED to every column and normalized. That completes "
                "LAYER 1 - same shape ({} x {}), new numbers. Watch the tracker.".format(self.n, self.d), seq)

    # ------------------------------------------------------------ 12 layer 2
    def s_layer2(self):
        LT = self.T["layers"][1]
        hw = self.hw
        x = S_L2
        g = self.board.attachNewNode("layer2")
        self.g_l2 = g
        tabs = []
        cw, ch = 0.6, 0.34
        for h in range(2):
            x0, z0 = x - 9.4 + h * 10.2, 0.5
            hdr, st = self._table(g, LT["heads"][h], x0, z0, cw, ch, num=0.13)
            hl = rect(g, x0 + self.hi * cw + 0.02, z0 - self.n * ch, x0 + (self.hi + 1) * cw - 0.02, z0 + 0.45, YELLOW,
                      2.0, y=-0.03)
            hl.hide()
            t = text(g, "layer 2, head {}".format(h + 1), Point3(x0 + self.n * cw / 2, 0, z0 + 1.5), 0.36, WHITE)
            t.hide()
            tabs.append((t, hdr, st, hl))
        anim = Sequence()
        for t, hdr, st, hl in tabs:
            anim.append(Sequence(show_and_fade(t, 0.2), show_and_fade(hdr, 0.2), show_and_fade(st["mask"], 0.2),
                                 show_and_fade(st["discs"], 0.3), show_and_fade(st["att"], 0.3), show_and_fade(hl, 0.2)))
        seq = Sequence(Func(self.app.ui.arch.set_layer, "layer 2 of 2"), self.mapkey("attn"),
                       self.dim(self.g_ffn), self.move_hw(x, 1.4), self.go(x, 1.0, 28), anim,
                       self.mapkey("add1"), hw.morph(LT["x1"], "layer 2: after attention + Add & Norm"),
                       self.track(LT["x1"][self.hi], "layer 2: attention"), Wait(0.6),
                       self.mapkey("ffn"), Wait(0.5), self.mapkey("add2"),
                       hw.morph(LT["x_out"], "layer 2 output = final"), self.track(LT["x_out"][self.hi], "end of layer 2"),
                       Func(self.app.ui.arch.set_layer, "x 2 layers"))
        H1, H2 = LT["heads"]
        return ("LAYER 2: exactly the same steps with its own matrices (shown faster): 2 attention heads, Add & Norm, "
                "feed-forward, Add & Norm. Now <ai> looks at: {}  /  {}. Real models stack 30-100 such layers."
                .format(self.top_attn(H1["att"], self.hi, 2), self.top_attn(H2["att"], self.hi, 2)), seq)

    # ------------------------------------------------------------ 13 output
    def s_output(self):
        x = S_OUT
        g = self.board.attachNewNode("out")
        m = self.engine.m
        xl = self.T["x_final"][-1]
        logits, probs = self.T["logits"], self.T["probs"]
        wx0, wz = x - 6.5, 1.6
        nd, drop = self.fly(xl, self.hero_x(x), HW_TOP, CH, x - 8.3, wz, 0.2, dur=1.1)
        lab = text(g, "<ai>", Point3(x - 8.3, 0, wz + 0.45), 0.32, YELLOW)
        lab.hide()
        wout = self.matrix(g, m.w["Wout"], wx0, wz, 0.06, 0.2, "W_out", "{} x {}".format(self.d, self.V))
        wout.hide()
        lg = g.attachNewNode("logits")
        zb = wz - self.d * 0.2 - 2.4
        sc = 1.6 / max(1e-6, np.abs(logits).max())
        for j, v in enumerate(logits):
            h = v * sc
            fill(lg, wx0 + j * 0.06, zb + min(0, h), wx0 + j * 0.06 + 0.045, zb + max(0, h),
                 YELLOW if j == int(np.argmax(logits)) else BLUE, 0.9, y=0)
        text(lg, "{} scores (logits), one per vocabulary token".format(self.V), Point3(wx0, 0, zb - 2.0), 0.28, GREY,
             align=TextNode.ALeft)
        lg.hide()
        top = np.argsort(-probs)[:5]
        pg = g.attachNewNode("probs")
        for r, j in enumerate(top):
            z = wz - 0.3 - r * 0.8
            text(pg, disp(m.vocab[j]), Point3(x + 7.6, 0, z), 0.4, WHITE, align=TextNode.ARight)
            w = max(0.02, 5.5 * probs[j])
            fill(pg, x + 7.9, z - 0.1, x + 7.9 + w, z + 0.42, YELLOW if r == 0 else BLUE, 0.9, y=0)
            text(pg, "{:.2%}".format(probs[j]) if probs[j] >= 0.0001 else "<0.01%", Point3(x + 8.1 + w, 0, z), 0.32,
                 GREY, align=TextNode.ALeft)
        text(pg, "softmax -> probabilities", Point3(x + 5.0, 0, wz + 0.45), 0.34, GREY, align=TextNode.ALeft)
        pg.hide()
        seq = Sequence(Func(self.app.ui.arch.set_layer, "x 2 layers"), self.mapkey("linear"), self.dim(self.g_l2), self.move_hw(x, 1.4),
                       fade_out(self.hw.label, 0.3),
                       self.go(x + 1.5, -0.4, 31), drop, show_and_fade(lab, 0.2), show_and_fade(wout, 0.5), Wait(0.3),
                       show_and_fade(lg, 0.7), Wait(0.5), self.mapkey("softmax", "output"), show_and_fade(pg, 0.5),
                       self.track(xl, "used for the prediction"))
        best = m.vocab[int(top[0])]
        extra = (" (Almost 100%: this tiny model learned these few conversations by heart - big models are much less "
                 "certain.)" if probs[top[0]] > 0.99 else "")
        return ("OUTPUT: only <ai>'s column - the one we followed all the way - is used now. It drops into W_out, "
                "giving one score for each of the {} tokens; softmax turns scores into probabilities. Winner: '{}' "
                "({:.1%}).".format(self.V, disp(best), probs[top[0]]) + extra, seq)

    # ------------------------------------------------------------ deep dive OFF: one quick pass
    def s_quick(self):
        hw = self.hw
        steps = Sequence()
        for l, LT in enumerate(self.T["layers"]):
            for key, mk, label, where, xs in (("x1", "attn", "layer {}: after attention + Add & Norm", "attention",
                                               S_CAT if l == 0 else S_L2 - 8),
                                              ("x_out", "ffn", "layer {} output", "feed-forward",
                                               S_ADD2 if l == 0 else S_L2)):
                steps.append(Sequence(self.mapkey(mk), self.move_hw(xs, 1.0), self.mapkey("add1" if key == "x1" else "add2"),
                                      hw.morph(LT[key], label.format(l + 1)),
                                      self.track(LT[key][self.hi], "layer {}: {}".format(l + 1, where)), Wait(0.5)))
        return ("Inside the Transformer (deep dive OFF - press D for every detail): the highway passes 2 layers; in "
                "each, attention lets tokens exchange information and the feed-forward network processes each token. "
                "Follow the yellow <ai> column and the tracker.", steps)
