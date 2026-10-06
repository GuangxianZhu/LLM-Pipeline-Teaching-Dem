# -*- coding: utf-8 -*-
# Claude Opus 写的
# 第 7 题重做（Claude）：全部改用 tiny 模型的真实矩阵，逐步解说，加上乘法次数、费用、耗电的具体例子
"""
Question 7: what does "compute again" mean, and what does a cache save?

ONE thing is followed from start to end: the K matrix of layer 1, head 1 - the green K the students already
met in question 3 ("Plot the tank temperature for the last hour."). Every number is real (tiny/):

  A  compute    X (13 rows) . W_K = K, row by row. "Computing" = these multiplications.
     no cache   the model wrote one token; X has 14 rows; without a cache all 14 rows are multiplied AGAIN.
                Row by row, the new K equals the old K (difference 0.000) - only the last row is new.
  B  why same   the mask: a row only reads the rows above it. The same holds in layer 2.
  C  KV cache   K and V go on the shelf. The next token needs ONE row (tiny/kv.py) - same probabilities.
  D  work       multiplications per new token, without / with cache.
  E  cost       the same idea at real size: tokens, money (API price list), electricity (estimate).
  F  requests   the next request reuses the shelf; change one early word -> every row below it changes;
                another model -> nothing can be reused.
"""
import numpy as np
from direct.interval.IntervalGlobal import Func, LerpFunc, LerpPosInterval, Parallel, Sequence, Wait
from panda3d.core import Point3, TextNode

from engine import Engine
from i18n import t
import i18n
from kit import (GREEN, GREY, ORANGE, PURPLE, RED, TOKEN_COLORS, WHITE, YELLOW, Fonts, arrow2d, fade_in, fade_out,
                 fill, fx, heatmap, lines, rect, text, text_width)
from scenarios import SCENARIOS
from story import Step
from tf_steps import disp, sc_of
from tiny.kv import KVCache, mults_no_cache, mults_with_cache, step

STAGES = ["Compute", "No cache", "Why same", "KV cache", "Work", "Cost", "Next request", "Cache miss", "Summary"]

RH = 0.62                   # one token row
CD = 0.15                   # one number in a row
GAP = 0.30
FRAME = (0.30, 0.30, 0.34, 1)
MUTED = (0.6, 0.6, 0.66, 1)
NOTE = 0.38                 # size of scene notes
TOO = (0.98, 0.42, 0.36, 1)

O = Point3(240, 0, 0)       # A: compute / no cache
B = O + Point3(0, 0, -24)   # B: why the old rows stay the same
C = O + Point3(38, 0, 0)    # C: the KV cache shelf
D = O + Point3(0, 0, -54)   # D: work bars
E = O + Point3(34, 0, -54)  # E: money and electricity
F = O + Point3(0, 0, -82)   # F: the next request, cache misses

# ---------------------------------------------------------------- the real world (numbers on screen)
# API price list, Claude Sonnet 5.5, US$ per million tokens (platform.claude.com/docs/en/about-claude/pricing,
# checked 2026-10-06): input 2.00, writing to the 5-minute cache 2.50 (1.25 x), reading the cache 0.20 (0.1 x).
PRICE_IN, PRICE_WRITE, PRICE_READ = 2.00, 2.50, 0.20
YEN = 150                   # example exchange rate, US$ 1 = 150 yen
# Electricity: an ESTIMATE with a made-up example model (Claude's size is not public):
#   70 billion parameters -> reading one token = about 70 billion multiplications (every weight once)
#   one H100 GPU: about 400 trillion multiply-adds/s at 40 % of its peak, 700 W  (cooling not counted)
PARAMS = 70e9
GPU_MAC_S = 4.0e14 / 2      # 400 trillion operations/s = 200 trillion multiply-adds/s
GPU_W = 700.0
J_PER_TOKEN = PARAMS / GPU_MAC_S * GPU_W          # about 0.25 J
S_PER_TOKEN = PARAMS / GPU_MAC_S                  # about 0.35 ms
PHONE_WH = 15.0                                   # a phone battery, about 15 Wh
KWH_YEN = 31                                      # Japan, about 31 yen per kWh (household, example)
# the two examples
HIST, NEW = 20000, 50                             # a long chat: 20,000 tokens so far, a new message of 50
SYS0, PER_TURN, USER, TURNS, CLASS = 5000, 500, 100, 30, 40


def fmt_n(n):
    """1234567 -> 1,234,567 (en) / 123.5 万 (zh)"""
    n = int(round(n))
    if i18n.LANG == "zh" and n >= 10000:
        v = n / 1e4
        return ("{:.0f} 万" if v >= 100 else "{:.1f} 万").format(v)
    return "{:,}".format(n)


def fmt_usd(x):
    return "${:.4f}".format(x) if x < 0.1 else "${:.2f}".format(x)


def fmt_yen(x):
    y = x * YEN
    return "{:.1f}".format(y) if y < 10 else "{:,.0f}".format(y)


def fmt_wh(j):
    wh = j / 3600.0
    if wh < 0.1:
        return "{:.0f} J".format(j)
    return "{:.1f} Wh".format(wh) if wh < 100 else "{:,.0f} Wh".format(wh)


def fmt_s(s):
    return "{:.2f} s".format(s) if s < 1 else "{:.1f} s".format(s)


def one_message():
    """A chat with HIST tokens so far, plus a new message of NEW tokens."""
    n_all = HIST + NEW
    no = dict(computed=n_all, usd=n_all * PRICE_IN / 1e6)
    hit = dict(computed=NEW, usd=HIST * PRICE_READ / 1e6 + NEW * PRICE_WRITE / 1e6)
    for r in (no, hit):
        r["J"] = r["computed"] * J_PER_TOKEN
        r["s"] = r["computed"] * S_PER_TOKEN
    switch = n_all * PRICE_WRITE / 1e6                  # another model: nothing cached, all written again
    return no, hit, switch


def conversation():
    """TURNS rounds: a system prompt of SYS0 tokens, every round adds a user message + a reply (PER_TURN).
    Returns per-round input cost without / with cache, and the computed tokens."""
    no_c, hit_c, no_t, hit_t = [], [], 0, 0
    prev = 0
    for k in range(TURNS):
        n = SYS0 + PER_TURN * k + USER                  # input of round k
        no_c.append(n * PRICE_IN / 1e6)
        no_t += n
        if k == 0:
            hit_c.append(n * PRICE_WRITE / 1e6)
            hit_t += n
        else:
            new = n - prev
            hit_c.append(prev * PRICE_READ / 1e6 + new * PRICE_WRITE / 1e6)
            hit_t += new
        prev = n
    return no_c, hit_c, no_t, hit_t


def show(nd, dur=0.4):
    return Sequence(Func(nd.show), fade_in(nd, dur))


def hide(nd, dur=0.3):
    return Sequence(fade_out(nd, dur), Func(nd.hide))


class Box:
    def __init__(self, x0, z_top, rows, cols, cw=CD, rh=RH):
        self.x0, self.z_top, self.rows, self.cols, self.cw, self.rh = x0, z_top, rows, cols, cw, rh
        self.x1, self.z_bot = x0 + cols * cw, z_top - rows * rh

    def rz(self, i):
        return self.z_top - (i + 0.5) * self.rh

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2


class CacheStory:
    stages = STAGES

    def __init__(self, app, sc, deep=True):
        self.app = app
        self.sc = sc
        self.board = app.board
        chart = next(s for s in SCENARIOS if s["key"] == "chart")
        self.eng = Engine(chart, 1)
        self.m = m = self.eng.m
        self.prompt = chart["prompt"]
        self.ctx = list(self.eng.first_context)                 # 13 tokens: <sys> <user> ... <ai>
        self.gen = list(self.eng.rounds[0]["toks"])             # the tool call the model writes (real)
        self.n0 = len(self.ctx)
        _, self.T0 = m.forward(self.ctx, trace=True)
        self.p1, self.T1 = m.forward(self.ctx + self.gen[:1], trace=True)
        self.cache = KVCache.from_trace(m, self.T0)
        self.p1c, self.info = step(m, self.gen[0], self.cache)  # the same thing WITH the cache
        self.H0 = self.T0["layers"][0]["heads"][0]
        self.H1 = self.T1["layers"][0]["heads"][0]
        self.ks = sc_of(self.H1["k"])
        self.vs = sc_of(self.H1["v"])
        self.xs = sc_of(self.T1["x0"])
        self.d, self.dh = m.d, m.dh
        self.root = self.board.attachNewNode("cache7")
        self.steps = [Step("Compute", self.s_x), Step("Compute", self.s_k), Step("Compute", self.s_first),
                      Step("No cache", self.s_redo), Step("No cache", self.s_compare),
                      Step("Why same", self.s_mask), Step("Why same", self.s_layer2),
                      Step("KV cache", self.s_store), Step("KV cache", self.s_onerow),
                      Step("Work", self.s_bars),
                      Step("Cost", self.s_money1), Step("Cost", self.s_money2),
                      Step("Next request", self.s_req2),
                      Step("Cache miss", self.s_change), Step("Cache miss", self.s_model),
                      Step("Summary", self.s_summary)]
        self.intro = t("cache.intro", prompt=self.prompt)
        self._layout()
        self.app.cam_ctl.go_to(O + Point3(10, 0, 0), 0, 0, 34, force=True)

    # ================================================================ helpers
    def view(self, target, d):
        return Func(self.app.cam_ctl.go_to, Point3(target), 0.0, 0.0, d)

    def go(self, x0, x1, z0, z1):
        """Camera so that the rectangle x0..x1, z0..z1 is visible above the caption."""
        d = max((x1 - x0) / 0.95, (z1 - z0) / 0.66, 12.0)
        return self.view(Point3((x0 + x1) / 2, 0, (z0 + z1) / 2 - 0.07 * d), d)

    def chat(self, who, s):
        return Func(self.app.ui.chat, who, s)

    def node(self, name="n"):
        nd = self.root.attachNewNode(name)
        nd.hide()
        return nd

    def note(self, s, x, z, scale=NOTE, color=GREY, align=TextNode.ALeft, parent=None):
        g = (parent or self.root).attachNewNode("note")
        text(g, s, Point3(x, -0.02, z), scale, color, Fonts.symbol, align=align)
        if parent is None:
            g.hide()
        return g

    def live(self, x, z, scale=0.42, color=YELLOW, align=TextNode.ALeft):
        """A text that changes while the animation plays (set with self.say)."""
        nd = text(self.root, "", Point3(x, -0.02, z), scale, color, Fonts.serif, align=align)
        return nd

    def say(self, nd, s):
        return Func(nd.node().setText, fx(s, Fonts.serif))

    def heat(self, M, box, scale, parent=None, colors=None):
        g = (parent or self.root).attachNewNode("mat")
        heatmap(g, M, box.x0, box.z_top, box.cw, box.rh, scale, gap=GAP if box.rh == RH else 0.06, gapx=0.08,
                colors=colors, nan_color=(0.04, 0.04, 0.05, 1))
        if parent is None:
            g.hide()
        return g

    def rows(self, M, box, scale, hero=None):
        """Matrix as separate rows (hidden), so they can appear one by one."""
        out = []
        for i in range(M.shape[0]):
            g = self.root.attachNewNode("r")
            heatmap(g, M[i:i + 1], box.x0, box.z_top - i * box.rh, box.cw, box.rh, scale, gap=GAP, gapx=0.08)
            if i == hero:
                z = box.rz(i)
                rect(g, box.x0 - 0.08, z - RH / 2 + 0.02, box.x1 + 0.08, z + RH / 2 - 0.02, YELLOW, 2.4, y=-0.02)
            g.hide()
            out.append(g)
        return out

    def frame(self, parent, box, label=None, shape=None, color=FRAME, lab_color=MUTED):
        rect(parent, box.x0 - 0.1, box.z_bot - 0.06, box.x1 + 0.1, box.z_top + 0.06, color, 1.2)
        if label:
            text(parent, label, Point3(box.xc, 0, box.z_bot - 0.62), 0.42, lab_color, Fonts.symbol)
        if shape:
            text(parent, shape, Point3(box.xc, 0, box.z_bot - 1.12), 0.32, (0.5, 0.5, 0.56, 1), Fonts.symbol)

    def weight_block(self, parent, box, name):
        fill(parent, box.x0, box.z_bot, box.x1, box.z_top, (0.35, 0.5, 0.85, 1), 0.10)
        rect(parent, box.x0, box.z_bot, box.x1, box.z_top, (0.45, 0.6, 0.95, 1), 1.2)
        text(parent, name, Point3(box.xc, 0, box.z_bot - 0.62), 0.42, (0.55, 0.68, 1.0, 1), Fonts.symbol)
        text(parent, "{} × {}".format(box.rows, box.cols), Point3(box.xc, 0, box.z_bot - 1.12), 0.32,
             (0.5, 0.56, 0.72, 1), Fonts.symbol)

    def word_col(self, toks, x, box, colors=None, scale=0.3):
        g = self.node("words")
        for i, tk in enumerate(toks):
            c = colors[i] if colors else (YELLOW if tk == "<ai>" else WHITE)
            text(g, disp(tk), Point3(x, 0, box.rz(i) - 0.11), scale, c, align=TextNode.ARight)
        return g

    def flash_row(self, box, i, color=WHITE, dur=0.3, pad=0.14):
        g = self.root.attachNewNode("hl")
        z = box.rz(i)
        rect(g, box.x0 - pad, z - RH / 2 - 0.04, box.x1 + pad, z + RH / 2 + 0.04, color, 2.6, y=-0.03)
        g.hide()
        return g

    # ================================================================ layout (A)
    def _layout(self):
        n1 = self.n0 + 1
        zt = 4.4
        self.zt = zt
        self.bX = Box(O.x, zt, n1, self.d)                                   # X grows from 13 to 14 rows
        wz = zt - self.n0 * RH / 2 + self.d * CD / 2
        self.bW = Box(O.x + 6.6, wz, self.d, self.dh, CD, CD)                # W_K (32 x 16)
        self.bK0 = Box(O.x + 10.6, zt, n1, self.dh)                          # K, first time
        self.bV0 = Box(O.x + 14.4, zt, n1, self.dh)                          # V, first time
        self.bK1 = Box(O.x + 24.0, zt, n1, self.dh)                          # K, computed again (no cache)
        self.cmp_x = self.bK1.x1 + 0.7
        self.A_x1 = self.cmp_x + 7.6

    # ================================================================ 1  X
    def s_x(self):
        n0 = self.n0
        X0 = self.T1["x0"]                    # 14 rows; the first 13 are the X of question 3, the 14th comes later
        g = self.node("frames")
        self.frames = g
        bx13 = Box(self.bX.x0, self.zt, n0, self.d)
        self.frame(g, bx13, "X", "{} × {}".format(n0, self.d))
        words = self.word_col(self.ctx, O.x - 0.35, self.bX)
        self.words = words
        rowsX = self.rows(X0, self.bX, self.xs, hero=n0 - 1)
        self.rowsX = rowsX
        info = self.note(t("cache.x.note", n=n0, d=self.d), O.x - 4.2, self.zt + 1.4, NOTE, WHITE)
        seq = Sequence(self.go(O.x - 6, O.x + 9, self.bX.z_bot - 2.2, self.zt + 2.2),
                       self.chat("You", self.prompt), show(g, 0.5), show(words, 0.5),
                       Parallel(*[Sequence(Wait(0.08 * i), show(r, 0.25)) for i, r in enumerate(rowsX[:n0])]),
                       Wait(0.3), show(info, 0.5))
        self.x_info = info
        return t("cache.x.caption", n=n0, d=self.d), seq

    # ================================================================ 2  K = X . W_K, row by row
    def s_k(self):
        n0, d, dh = self.n0, self.d, self.dh
        H = self.H0
        g = self.node("wk")
        self.weight_block(g, self.bW, "W_K")
        heatmap(g, H["Wk"], self.bW.x0, self.bW.z_top, CD, CD, sc_of(H["Wk"]), gap=0.06, gapx=0.06)
        fr = self.node("kframe")
        self.frame(fr, Box(self.bK0.x0, self.zt, n0, dh), "K", "{} × {}".format(n0, dh), GREEN, GREEN)
        self.frame(fr, Box(self.bV0.x0, self.zt, n0, dh), "V", "{} × {}".format(n0, dh), RED, RED)
        eq = self.note("X · W_K = K\n({0} × {1}) · ({1} × {2}) = ({0} × {2})".format(n0, d, dh),
                       self.bW.xc + 0.6, self.zt + 1.0, 0.32, WHITE, TextNode.ACenter)
        rowsK = self.rows(H["k"], self.bK0, self.ks, hero=n0 - 1)
        rowsV = self.rows(H["v"], self.bV0, self.vs, hero=n0 - 1)
        self.rowsK0, self.rowsV0 = rowsK, rowsV
        # one number worked out: the first number of <ai>'s row of K
        xr, wk = self.T0["x0"][n0 - 1], H["Wk"][:, 0]
        terms = " + ".join("({:.2f})·({:.2f})".format(xr[k], wk[k]) for k in range(3))
        demo = self.note(t("cache.k.demo", terms=terms, d=d, v=float(xr @ wk)), O.x - 4.2, self.zt + 3.4, 0.38,
                         YELLOW)
        cnt = self.live(O.x - 4.2, self.zt + 2.4, 0.42, YELLOW)
        per = d * dh
        self.count_A = cnt
        seq = Sequence(hide(self.x_info, 0.3),
                       self.go(O.x - 6, self.bV0.x1 + 1.5, self.bX.z_bot - 2.4, self.zt + 4.0),
                       show(g, 0.5), show(fr, 0.4), show(eq, 0.4), show(demo, 0.5), Wait(1.2))
        for i in range(n0):
            hl = self.flash_row(self.bX, i)
            seq.append(Sequence(Func(hl.show), show(rowsK[i], 0.18 if i else 0.4),
                                self.say(cnt, t("cache.k.count", rows=i + 1, per=per, n=fmt_n((i + 1) * per))),
                                Wait(0.12 if i else 0.5), Func(hl.removeNode)))
        seq.append(Sequence(Wait(0.3), Parallel(*[Sequence(Wait(0.04 * i), show(r, 0.2)) for i, r in
                                                   enumerate(rowsV)]),
                            self.say(cnt, t("cache.k.count2", n=fmt_n(2 * n0 * per)))))
        self.k_demo = demo
        return t("cache.k.caption", n=n0, d=d, dh=dh, per=per), seq

    # ================================================================ 3  the first new token
    def s_first(self):
        n0 = self.n0
        tok = self.gen[0]
        p0 = float(self.T0["probs"][self.m.index[tok]])
        out = self.note(t("cache.first.out", tok=tok, p=p0 * 100), O.x - 4.2, self.zt + 3.0, 0.42, WHITE)
        new_w = self.node("neww")
        text(new_w, tok, Point3(O.x - 0.35, 0, self.bX.rz(n0) - 0.11), 0.3, YELLOW, align=TextNode.ARight)
        hl = self.flash_row(self.bX, n0, YELLOW, pad=0.1)
        hl.reparentTo(new_w)
        hl.show()
        g = self.node("xframe14")
        self.frame(g, self.bX, None)
        tip = self.note(t("cache.first.note", n=n0 + 1), self.bX.x0, self.bX.z_bot - 1.6, NOTE, WHITE)
        seq = Sequence(hide(self.k_demo, 0.3), Func(self.frames.hide),
                       self.go(O.x - 6, self.bV0.x1 + 1.5, self.bX.z_bot - 2.6, self.zt + 3.4),
                       show(out, 0.5), self.chat("Model", tok), Wait(0.6),
                       show(g, 0.3), show(new_w, 0.4), show(self.rowsX[n0], 0.5), Wait(0.3), show(tip, 0.5),
                       self.say(self.count_A, ""))
        self.first_out, self.first_tip = out, tip
        return t("cache.first.caption", tok=tok, n=n0 + 1), seq

    # ================================================================ 4  no cache: everything again
    def s_redo(self):
        n1, d, dh = self.n0 + 1, self.d, self.dh
        H = self.H1
        fr = self.node("k1frame")
        self.frame(fr, self.bK1, t("cache.redo.label"), "{} × {}".format(n1, dh), ORANGE, ORANGE)
        rowsK1 = self.rows(H["k"], self.bK1, self.ks, hero=n1 - 1)
        self.rowsK1 = rowsK1
        cnt = self.live(self.bK1.x0, self.zt + 2.0, 0.42, ORANGE)
        self.count_B = cnt
        per = d * dh
        arrow = self.node("arr")
        arrow2d(arrow, (self.bV0.x1 + 0.6, 0, self.zt - n1 * RH / 2), (self.bK1.x0 - 0.6, 0, self.zt - n1 * RH / 2),
                ORANGE, 2.0, 0.22)
        text(arrow, t("cache.redo.arrow"), Point3((self.bV0.x1 + self.bK1.x0) / 2, 0, self.zt - n1 * RH / 2 + 0.45),
             0.32, ORANGE, Fonts.symbol)
        seq = Sequence(hide(self.first_out, 0.3), hide(self.first_tip, 0.3),
                       self.go(O.x - 5, self.A_x1, self.bX.z_bot - 2.6, self.zt + 2.6),
                       show(arrow, 0.4), show(fr, 0.4))
        for i in range(n1):
            hl = self.flash_row(self.bX, i, ORANGE)
            seq.append(Sequence(Func(hl.show), show(rowsK1[i], 0.15),
                                self.say(cnt, t("cache.k.count", rows=i + 1, per=per, n=fmt_n((i + 1) * per))),
                                Wait(0.1), Func(hl.removeNode)))
        return t("cache.redo.caption", n=n1, n0=self.n0), seq

    # ================================================================ 5  compare row by row
    def s_compare(self):
        n0, n1 = self.n0, self.n0 + 1
        K0, K1 = self.H0["k"], self.H1["k"]
        per = self.d * self.dh
        marks = []
        for i in range(n1):
            g = self.node("cmp")
            z = self.bK1.rz(i) - 0.12
            if i < n0:
                diff = float(np.abs(K1[i] - K0[i]).max())
                text(g, t("cache.cmp.same", d=diff), Point3(self.cmp_x, 0, z), 0.32, GREEN, Fonts.symbol,
                     TextNode.ALeft)
                lines(g, [[(self.bK0.x1 + 0.1, -0.01, self.bK0.rz(i)), (self.bK0.x1 + 0.35, -0.01, self.bK0.rz(i))]],
                      (0.5, 0.75, 0.4, 0.6), 1.0)
            else:
                text(g, t("cache.cmp.new"), Point3(self.cmp_x, 0, z), 0.32, YELLOW, Fonts.symbol, TextNode.ALeft)
            marks.append(g)
        brace = self.node("brace")
        x = self.cmp_x + 2.9
        lines(brace, [[(x, 0, self.bK1.z_top - 0.05), (x + 0.25, 0, self.bK1.z_top - 0.05),
                       (x + 0.25, 0, self.bK1.rz(n0 - 1) - RH / 2), (x, 0, self.bK1.rz(n0 - 1) - RH / 2)]],
              GREEN, 1.6)
        text(brace, t("cache.cmp.waste", n0=n0, k=fmt_n(n0 * per), all=fmt_n(mults_no_cache(self.m, n1)),
                      new=fmt_n(mults_with_cache(self.m, n1))),
             Point3(x + 0.6, 0, self.bK1.z_top - 1.0), 0.36, WHITE, Fonts.symbol, TextNode.ALeft)
        hl_old = self.node("hlold")
        rect(hl_old, self.bK0.x0 - 0.2, self.bK0.rz(n0 - 1) - RH / 2 - 0.08, self.bK0.x1 + 0.2, self.zt + 0.1,
             GREEN, 2.0, y=-0.03)
        rect(hl_old, self.bK1.x0 - 0.2, self.bK1.rz(n0 - 1) - RH / 2 - 0.08, self.bK1.x1 + 0.2, self.zt + 0.1,
             GREEN, 2.0, y=-0.03)
        seq = Sequence(self.go(O.x + 8, self.A_x1 + 7.5, self.bX.z_bot - 2.6, self.zt + 2.0),
                       self.say(self.count_B, ""),
                       Parallel(*[Sequence(Wait(0.14 * i), show(mk, 0.25)) for i, mk in enumerate(marks)]),
                       Wait(0.4), show(hl_old, 0.4), show(brace, 0.6))
        self.cmp_nodes = marks + [brace, hl_old]
        return t("cache.cmp.caption", n0=n0), seq

    # ================================================================ 6  the mask
    def s_mask(self):
        n1 = self.n0 + 1
        att = self.H1["att"]
        toks = self.ctx + self.gen[:1]
        bS = Box(B.x + 2.0, B.z + 4.2, n1, n1, RH, RH)
        g = self.node("mask")
        self.frame(g, bS, t("cache.mask.label"), "{} × {}".format(n1, n1))
        heatmap(g, att, bS.x0, bS.z_top, RH, RH, 1.0, gap=0.1, gapx=0.1,
                colors=lambda v: (0.1 + 0.9 * min(1, v * 2.2),) * 3 + (1,))
        for i in range(n1):
            for j in range(i + 1, n1):
                fill(g, bS.x0 + j * RH + 0.03, bS.z_top - (i + 1) * RH + 0.03, bS.x0 + (j + 1) * RH - 0.03,
                     bS.z_top - i * RH - 0.03, (0.04, 0.04, 0.05, 1), 1.0, y=-0.01)
        for i, tk in enumerate(toks):
            c = YELLOW if i == n1 - 1 else WHITE
            text(g, disp(tk), Point3(bS.x0 - 0.25, 0, bS.rz(i) - 0.1), 0.28, c, align=TextNode.ARight)
            text(g, str(i + 1), Point3(bS.x0 + (i + 0.5) * RH, 0, bS.z_top + 0.2), 0.24, c)
        text(g, t("cache.mask.rows"), Point3(bS.x0 - 0.25, 0, bS.z_top + 0.6), 0.3, GREY, align=TextNode.ARight)
        text(g, t("cache.mask.cols"), Point3(bS.xc, 0, bS.z_top + 0.75), 0.3, GREY)
        text(g, t("cache.mask.legend"), Point3(bS.xc, 0, bS.z_bot - 1.75), 0.32, GREY, Fonts.symbol)
        col = self.node("col")
        cx = bS.x0 + (n1 - 1) * RH
        rect(col, cx - 0.05, bS.z_bot - 0.05, cx + RH + 0.05, bS.z_top + 0.05, YELLOW, 2.6, y=-0.03)
        row = self.node("row")
        rect(row, bS.x0 - 0.05, bS.z_bot - 0.05, bS.x1 + 0.05, bS.z_bot + RH + 0.05, YELLOW, 2.6, y=-0.03)
        n_col = self.note(t("cache.mask.col_note", n0=self.n0), bS.x1 + 1.2, bS.z_top - 0.6, NOTE, WHITE)
        n_row = self.note(t("cache.mask.row_note"), bS.x1 + 1.2, bS.z_bot + 1.4, NOTE, YELLOW)
        self.bS = bS
        seq = Sequence(self.go(B.x - 3.5, bS.x1 + 14, bS.z_bot - 1.6, bS.z_top + 1.6),
                       show(g, 0.6), Wait(0.8), show(col, 0.4), show(n_col, 0.5), Wait(1.5),
                       show(row, 0.4), show(n_row, 0.5))
        return t("cache.mask.caption"), seq

    # ================================================================ 7  layer 2 is the same
    def s_layer2(self):
        n0, n1, dh = self.n0, self.n0 + 1, self.dh
        K0 = self.T0["layers"][1]["heads"][0]["k"]
        K1 = self.T1["layers"][1]["heads"][0]["k"]
        sc = sc_of(K1)
        x0 = self.bS.x1 + 15.5
        b0 = Box(x0, B.z + 4.2, n0, dh)
        b1 = Box(x0 + 4.2, B.z + 4.2, n1, dh)
        g = self.node("l2")
        self.frame(g, b0, t("cache.l2.old"), None, GREEN, GREEN)
        self.frame(g, b1, t("cache.l2.new"), None, ORANGE, ORANGE)
        self.heat(K0, b0, sc, parent=g)
        self.heat(K1, b1, sc, parent=g)
        for i in range(n1):
            z = b1.rz(i) - 0.11
            if i < n0:
                text(g, "{:.3f}".format(float(np.abs(K1[i] - K0[i]).max())), Point3(b1.x1 + 0.4, 0, z), 0.3, GREEN,
                     align=TextNode.ALeft)
            else:
                text(g, t("cache.cmp.new"), Point3(b1.x1 + 0.4, 0, z), 0.3, YELLOW, Fonts.symbol, TextNode.ALeft)
        text(g, t("cache.l2.diff"), Point3(b1.x1 + 0.4, 0, b1.z_top + 0.35), 0.3, GREY, align=TextNode.ALeft)
        why = self.note(t("cache.l2.why"), x0, b1.z_bot - 2.0, NOTE, WHITE)
        seq = Sequence(self.go(x0 - 2, b1.x1 + 9, b1.z_bot - 5.0, b1.z_top + 1.4), show(g, 0.6), Wait(0.8),
                       show(why, 0.6))
        return t("cache.l2.caption"), seq

    # ================================================================ 8  onto the shelf
    def s_store(self):
        n0, dh = self.n0, self.dh
        zt = self.zt
        self.sV = Box(C.x + 1.4, zt, n0 + 1, dh)
        self.sK = Box(C.x + 4.8, zt, n0 + 1, dh)
        shelf = self.node("shelf")
        x0s, x1s = C.x - 4.4, C.x + 7.8
        z0s = self.sK.z_bot - 1.6
        fill(shelf, x0s, z0s, x1s, zt + 1.0, GREEN, 0.06, y=0.03)
        rect(shelf, x0s, z0s, x1s, zt + 1.0, GREEN, 1.8)
        text(shelf, t("cache.store.shelf"), Point3((x0s + x1s) / 2, 0, zt + 1.35), 0.46, GREEN)
        text(shelf, "K", Point3(self.sK.xc, 0, self.sK.z_bot - 0.6), 0.42, GREEN)
        text(shelf, "V", Point3(self.sV.xc, 0, self.sV.z_bot - 0.6), 0.42, RED)
        # the other heads and layer 2 are stored too (small blocks)
        others = [(t("cache.store.l1h2"), 0), (t("cache.store.l2h1"), 1), (t("cache.store.l2h2"), 2)]
        for lab, k in others:
            bz = zt - 0.3 - k * 2.6
            for j, col in enumerate((RED, GREEN)):
                bx = C.x - 3.6 + j * 0.9
                fill(shelf, bx, bz - 1.6, bx + 0.7, bz, col, 0.35)
                rect(shelf, bx, bz - 1.6, bx + 0.7, bz, col, 1.0)
            text(shelf, lab, Point3(C.x - 2.85, 0, bz - 2.1), 0.32, GREY)
        mem = self.note(t("cache.store.mem", per=2 * self.m.L * self.d, n=n0, tot=fmt_n(2 * self.m.L * self.d * n0)),
                        x0s, z0s - 0.8, 0.42, WHITE)
        # K and V of layer 1, head 1 fly from A to the shelf
        flies = Parallel()
        self.shelfK, self.shelfV = [], []
        for i in range(n0):
            for src, dst_box, lst, M, sc in ((self.bK0, self.sK, self.shelfK, self.H0["k"], self.ks),
                                             (self.bV0, self.sV, self.shelfV, self.H0["v"], self.vs)):
                r = self.root.attachNewNode("fly")
                heatmap(r, M[i:i + 1], 0, 0, CD, RH, sc, gap=GAP, gapx=0.08)
                r.hide()
                lst.append(r)
                frm = Point3(src.x0, 0, src.z_top - i * RH)
                to = Point3(dst_box.x0, 0, dst_box.z_top - i * RH)
                flies.append(Sequence(Wait(0.05 * i), Func(r.setPos, frm), Func(r.show),
                                      LerpPosInterval(r, 1.2, to, blendType="easeInOut")))
        seq = Sequence(self.go(self.bK0.x0 - 1.0, x1s + 1.0, z0s - 1.8, zt + 2.2),
                       Func(self._clear_cmp), show(shelf, 0.6), flies, Wait(0.3), show(mem, 0.5))
        self.shelf_mem = mem
        self.shelf_x1 = x1s
        return t("cache.store.caption", n=n0), seq

    def _clear_cmp(self):
        for nd in self.cmp_nodes:
            nd.hide()

    # ================================================================ 9  one row with the cache
    def s_onerow(self):
        n0, n1, d, dh = self.n0, self.n0 + 1, self.d, self.dh
        inf = self.info[0][0]
        zr = self.sK.rz(n0)                                       # the new row's height
        xr0 = self.shelf_x1 + 2.0
        g = self.node("one")
        xrow = Box(xr0, zr + RH / 2, 1, d)
        heatmap(g, self.T1["x0"][n0:n0 + 1], xrow.x0, xrow.z_top, CD, RH, self.xs, gap=GAP, gapx=0.08)
        rect(g, xrow.x0 - 0.08, xrow.z_bot + 0.02, xrow.x1 + 0.08, xrow.z_top - 0.02, YELLOW, 2.4, y=-0.02)
        text(g, t("cache.one.xrow", tok=self.gen[0]), Point3(xrow.x0, 0, xrow.z_bot - 0.55), 0.34, YELLOW,
             Fonts.symbol, TextNode.ALeft)
        newk, newv = self.node("newk"), self.node("newv")
        heatmap(newk, inf["k"][None, :], 0, 0, CD, RH, self.ks, gap=GAP, gapx=0.08)
        heatmap(newv, inf["v"][None, :], 0, 0, CD, RH, self.vs, gap=GAP, gapx=0.08)
        for nd, box in ((newk, self.sK), (newv, self.sV)):
            rect(nd, -0.08, -RH + 0.02, box.cols * CD + 0.08, -0.02, YELLOW, 2.4, y=-0.02)
        arr = self.node("arrs")
        arrow2d(arr, (xrow.x0 - 0.3, 0, zr), (self.sK.x1 + 0.4, 0, zr), YELLOW, 2.0, 0.2)
        text(arr, "· W_K ,  · W_V", Point3((xrow.x0 + self.shelf_x1) / 2, 0, zr + 0.35), 0.32, YELLOW,
             Fonts.symbol)
        # q against every key on the shelf -> ONE row of scores
        qz = self.zt + 0.4
        qrow = self.node("q")
        heatmap(qrow, inf["q"][None, :], xr0, qz, CD, RH, sc_of(inf["q"]), gap=GAP, gapx=0.08)
        text(qrow, t("cache.one.q"), Point3(xr0, 0, qz + 0.35), 0.34, WHITE, Fonts.symbol, TextNode.ALeft)
        reads = self.node("reads")
        segs = [[(self.sK.x1 + 0.1, -0.01, self.sK.rz(i)), (xr0 - 0.2, -0.01, qz - RH / 2)] for i in range(n1)]
        lines(reads, segs, (0.51, 0.76, 0.40, 0.45), 1.0)
        srow = self.node("scores")
        sx = xr0
        sz = qz - 2.0
        heatmap(srow, inf["att"][None, :], sx, sz, RH, RH, 1.0, gap=0.1, gapx=0.1,
                colors=lambda v: (0.1 + 0.9 * min(1, v * 2.2),) * 3 + (1,))
        text(srow, t("cache.one.scores", n=n1), Point3(sx, 0, sz - RH - 0.45), 0.34, WHITE, Fonts.symbol,
             TextNode.ALeft)
        tok1 = self.gen[1]
        pc = float(self.p1c[self.m.index[tok1]])
        pf = float(self.p1[self.m.index[tok1]])
        res = self.note(t("cache.one.result", tok=tok1, pc=pc * 100, pf=pf * 100,
                          diff=float(np.abs(self.p1c - self.p1).max())), xr0, sz - 2.3, 0.38, WHITE)
        per = d * dh
        cnt = self.note(t("cache.one.count", per=per, n0=n0, all=fmt_n(mults_no_cache(self.m, n1)),
                          new=fmt_n(mults_with_cache(self.m, n1))), xr0, sz - 4.3, 0.38, YELLOW)
        seq = Sequence(hide(self.shelf_mem, 0.3),
                       self.go(C.x - 4.8, xr0 + 14.0, sz - 6.0, self.zt + 2.0),
                       show(g, 0.5), Wait(0.4), show(arr, 0.4),
                       Func(newk.setPos, xrow.x0, 0, xrow.z_top), Func(newv.setPos, xrow.x0, 0, xrow.z_top),
                       Func(newk.show), Func(newv.show),
                       Parallel(LerpPosInterval(newk, 1.0, Point3(self.sK.x0, 0, self.sK.z_top - n0 * RH),
                                                blendType="easeInOut"),
                                LerpPosInterval(newv, 1.0, Point3(self.sV.x0, 0, self.sV.z_top - n0 * RH),
                                                blendType="easeInOut")),
                       Wait(0.3), show(qrow, 0.4), show(reads, 0.5), show(srow, 0.5), Wait(0.5),
                       show(res, 0.5), self.chat("Model", self.gen[0] + tok1), Wait(0.4), show(cnt, 0.5))
        return t("cache.one.caption", n0=n0, n1=n1), seq

    # ================================================================ 10  work, token by token
    def s_bars(self):
        m = self.m
        G = 8
        ns = [self.n0 + 1 + k for k in range(G)]          # rows in the context when writing new token k+2
        no = [mults_no_cache(m, n) for n in ns]
        ca = [mults_with_cache(m, n) for n in ns]
        top = max(no)
        H = 6.0
        g = self.node("bars")
        x0 = D.x
        base = D.z
        lines(g, [[(x0 - 0.5, 0, base), (x0 + G * 2.7, 0, base)]], GREY, 1.4)
        text(g, t("cache.bars.title"), Point3(x0 - 0.5, 0, base + H + 1.4), 0.42, WHITE, align=TextNode.ALeft)
        bars = []
        for k in range(G):
            bx = x0 + k * 2.7
            for j, (v, col) in enumerate(((no[k], (0.55, 0.55, 0.58, 1)), (ca[k], YELLOW))):
                bb = self.root.attachNewNode("bar")
                fill(bb, 0, 0, 0.9, 1, col, 0.9, y=0)
                bb.setPos(bx + j * 1.0, 0, base)
                bb.setSz(0.001)
                bb.hide()
                lab = text(self.root, fmt_n(v), Point3(bx + j * 1.0 + 0.45, -0.02, base + v / top * H + 0.25),
                           0.32, WHITE if j == 0 else YELLOW)
                lab.hide()
                bars.append((bb, v / top * H, lab))
            text(g, t("cache.bars.tok", k=k + 2, n=ns[k]), Point3(bx + 0.95, 0, base - 0.6), 0.32, GREY)
        leg = self.note(t("cache.bars.legend"), x0 + G * 2.7 + 0.6, base + H, 0.42, WHITE)
        nreply = len(self.gen)
        tot_no = sum(mults_no_cache(m, self.n0 + k) for k in range(nreply))
        tot_ca = mults_no_cache(m, self.n0) + sum(mults_with_cache(m, self.n0 + k) for k in range(1, nreply))
        tot = self.note(t("cache.bars.total", k=nreply, no=fmt_n(tot_no), ca=fmt_n(tot_ca), x=tot_no / tot_ca),
                        x0 + G * 2.7 + 0.6, base + H - 2.4, 0.46, YELLOW)

        def grow(bb, h):
            return LerpFunc(lambda v: bb.setSz(max(0.001, v)), fromData=0.001, toData=h, duration=0.45,
                            blendType="easeOut")
        anim = Sequence()
        for bb, h, lab in bars:
            anim.append(Sequence(Func(bb.show), grow(bb, h), Func(lab.show)))
        seq = Sequence(self.go(x0 - 1.5, x0 + G * 2.7 + 13.5, base - 1.8, base + H + 2.2), show(g, 0.5),
                       show(leg, 0.4), anim, Wait(0.4), show(tot, 0.6))
        return t("cache.bars.caption", n=self.n0 + 1), seq

    # ================================================================ 11  money, electricity, time: one message
    def s_money1(self):
        no, hit, _ = one_message()
        g = self.node("money1")
        x0, z0 = E.x, E.z + 6.5
        text(g, t("cache.m1.title", hist=fmt_n(HIST), new=NEW), Point3(x0, 0, z0 + 1.2), 0.46, WHITE,
             align=TextNode.ALeft)
        cols = (x0 + 8.0, x0 + 14.5)
        text(g, t("cache.m1.no"), Point3(cols[0], 0, z0), 0.4, (0.75, 0.75, 0.78, 1), Fonts.symbol, TextNode.ALeft)
        text(g, t("cache.m1.hit"), Point3(cols[1], 0, z0), 0.4, YELLOW, Fonts.symbol, TextNode.ALeft)
        rows = [("cache.m1.r_tok", fmt_n(no["computed"]), fmt_n(hit["computed"])),
                ("cache.m1.r_usd", fmt_usd(no["usd"]), fmt_usd(hit["usd"])),
                ("cache.m1.r_yen", fmt_yen(no["usd"]), fmt_yen(hit["usd"])),
                ("cache.m1.r_time", fmt_s(no["s"]), fmt_s(hit["s"])),
                ("cache.m1.r_energy", fmt_wh(no["J"]), fmt_wh(hit["J"]))]
        line_nodes = []
        for i, (key, a, b) in enumerate(rows):
            r = self.node("mrow")
            z = z0 - 1.0 - i * 0.95
            text(r, t(key), Point3(x0, 0, z), 0.38, GREY, Fonts.symbol, TextNode.ALeft)
            text(r, a, Point3(cols[0], 0, z), 0.42, WHITE, Fonts.symbol, TextNode.ALeft)
            text(r, b, Point3(cols[1], 0, z), 0.42, YELLOW, Fonts.symbol, TextNode.ALeft)
            line_nodes.append(r)
        # bars: tokens that must be computed
        bz = z0 - 6.4
        bar = self.node("mbar")
        W = 18.0
        fill(bar, x0, bz, x0 + W, bz + 0.5, (0.55, 0.55, 0.58, 1), 0.9, y=0)
        fill(bar, x0, bz - 0.8, x0 + max(0.06, W * hit["computed"] / no["computed"]), bz - 0.3, YELLOW, 0.9, y=0)
        text(bar, t("cache.m1.bar_no", n=fmt_n(no["computed"])), Point3(x0 + W + 0.3, 0, bz + 0.1), 0.32,
             WHITE, Fonts.symbol, TextNode.ALeft)
        text(bar, t("cache.m1.bar_hit", n=NEW), Point3(x0 + 0.5, 0, bz - 0.7), 0.32, YELLOW, Fonts.symbol,
             TextNode.ALeft)
        phone = no["J"] / 3600.0 / PHONE_WH * 100
        how = self.note(t("cache.m1.how", phone=phone, J=J_PER_TOKEN, ms=S_PER_TOKEN * 1000), x0, bz - 1.7, 0.33,
                        MUTED)
        seq = Sequence(self.go(x0 - 1.0, x0 + 23.5, bz - 5.6, z0 + 2.0), show(g, 0.5),
                       Sequence(*[Sequence(show(r, 0.35), Wait(0.5)) for r in line_nodes]), show(bar, 0.6),
                       Wait(0.4), show(how, 0.5))
        return t("cache.m1.caption", x=no["usd"] / hit["usd"]), seq

    # ================================================================ 12  a whole conversation, a whole class
    def s_money2(self):
        no_c, hit_c, no_t, hit_t = conversation()
        g = self.node("money2")
        x0, base = E.x + 28.0, E.z
        H = 6.0
        top = max(no_c)
        text(g, t("cache.m2.title", turns=TURNS, sys=fmt_n(SYS0), per=PER_TURN), Point3(x0 - 0.5, 0, base + H + 1.7),
             0.5, WHITE, align=TextNode.ALeft)
        lines(g, [[(x0 - 0.5, 0, base), (x0 + TURNS * 0.62, 0, base)]], GREY, 1.4)
        bars = []
        for k in range(TURNS):
            bx = x0 + k * 0.62
            for j, (v, col) in enumerate(((no_c[k], (0.55, 0.55, 0.58, 1)), (hit_c[k], YELLOW))):
                bb = self.root.attachNewNode("bar")
                fill(bb, 0, 0, 0.26, 1, col, 0.9, y=0)
                bb.setPos(bx + j * 0.28, 0, base)
                bb.setSz(0.001)
                bb.hide()
                bars.append((bb, max(0.02, v / top * H)))
            if k in (0, 9, 19, 29):
                text(g, str(k + 1), Point3(bx + 0.27, 0, base - 0.55), 0.34, GREY)
        text(g, t("cache.m2.axis"), Point3(x0 + TURNS * 0.31, 0, base - 1.25), 0.36, GREY)
        text(g, fmt_usd(top), Point3(x0 - 0.7, 0, base + H - 0.1), 0.34, GREY, align=TextNode.ARight)
        sn, sh = sum(no_c), sum(hit_c)
        tx = x0 + TURNS * 0.62 + 1.0
        sumn = self.note(t("cache.m2.sum", no=fmt_usd(sn), noy=fmt_yen(sn), hit=fmt_usd(sh), hity=fmt_yen(sh),
                           x=sn / sh, tn=fmt_n(no_t), th=fmt_n(hit_t), wn=fmt_wh(no_t * J_PER_TOKEN),
                           wh=fmt_wh(hit_t * J_PER_TOKEN)), tx, base + H + 0.4, 0.44, WHITE)
        kwh_no = no_t * J_PER_TOKEN * CLASS / 3.6e6
        kwh_hit = hit_t * J_PER_TOKEN * CLASS / 3.6e6
        cls = self.note(t("cache.m2.class", c=CLASS, no=fmt_usd(sn * CLASS), noy=fmt_yen(sn * CLASS),
                          hit=fmt_usd(sh * CLASS), hity=fmt_yen(sh * CLASS), kn=kwh_no, kh=kwh_hit,
                          en=kwh_no * KWH_YEN), tx, base + H - 4.8, 0.44, YELLOW)

        def grow(bb, h):
            return LerpFunc(lambda v: bb.setSz(max(0.001, v)), fromData=0.001, toData=h, duration=0.12)
        seq = Sequence(self.go(x0 - 2.5, tx + 21.0, base - 2.4, base + H + 2.6), show(g, 0.5),
                       Sequence(*[Sequence(Func(bb.show), grow(bb, h)) for bb, h in bars]),
                       Wait(0.4), show(sumn, 0.6), Wait(0.8), show(cls, 0.6))
        return t("cache.m2.caption"), seq

    # ================================================================ 13  the next request: prefix hit
    def s_req2(self):
        g = self.node("req")
        x0, z1 = F.x, F.z + 6.0
        z2 = z1 - 2.6
        text(g, t("cache.req.r1"), Point3(x0 - 0.4, 0, z1 - 0.05), 0.38, WHITE, Fonts.symbol, TextNode.ARight)
        text(g, t("cache.req.r2"), Point3(x0 - 0.4, 0, z2 - 0.05), 0.38, WHITE, Fonts.symbol, TextNode.ARight)
        nres = len(self.eng.rounds[0].get("result_toks", []))
        segs1 = [(disp(tk), TOKEN_COLORS[i % 8] if not tk.startswith("<") else GREY) for i, tk in
                 enumerate(self.ctx)] + [(t("cache.req.call", n=len(self.gen)), ORANGE)]
        segs2 = segs1 + [("<tool> " + t("cache.req.result", n=nres), GREEN), ("<ai>", GREY)]
        chips1, x_end1 = self._chips(g, segs1, x0, z1)
        chips2, x_end2 = self._chips(self.root, segs2, x0, z2)
        n_hit = len(segs1)
        for c in chips2:
            c[0].hide()
        marks = []
        for c1 in chips1:
            mk = self.node("tick")
            text(mk, "=", Point3(c1[2], 0, (z1 + z2) / 2 - 0.15), 0.4, GREEN)
            marks.append(mk)
        hit = self.node("hit")
        fill(hit, x0 - 0.1, z2 - 0.5, chips2[n_hit - 1][3] + 0.05, z2 + 0.65, GREEN, 0.18, y=0.03)
        text(hit, t("cache.req.hit", n=self.n0 + len(self.gen)), Point3(x0, 0, z2 - 1.15), 0.38, GREEN,
             Fonts.symbol, TextNode.ALeft)
        miss = self.node("miss")
        fill(miss, chips2[n_hit][1] - 0.05, z2 - 0.5, x_end2 + 0.05, z2 + 0.65, ORANGE, 0.22, y=0.03)
        text(miss, t("cache.req.miss", n=nres + 2), Point3(x_end2, 0, z2 - 1.15), 0.38, ORANGE, Fonts.symbol,
             TextNode.ARight)
        rule = self.note(t("cache.req.why"), x0, z2 - 2.3, 0.36, WHITE)
        seq = Sequence(self.go(x0 - 4.5, x_end2 + 1.0, z2 - 4.0, z1 + 1.2), show(g, 0.5), Wait(0.3),
                       Parallel(*[Sequence(Wait(0.05 * i), show(c, 0.25)) for i, c in
                                  enumerate([c[0] for c in chips2])]),
                       Wait(0.3),
                       Sequence(*[Sequence(show(mk, 0.12), Wait(0.05)) for mk in marks]),
                       show(hit, 0.4), Wait(0.3), show(miss, 0.4), Wait(0.3), show(rule, 0.5))
        self.req_nodes = [g, hit, miss, rule] + marks + [c[0] for c in chips2]
        return t("cache.req.caption", n=self.n0 + len(self.gen), m=nres + 2), seq

    def _chips(self, parent, segs, x0, z):
        out = []
        x = x0
        for lab, col in segs:
            w = max(0.5, text_width(lab) * 0.3 + 0.3)
            c = parent.attachNewNode("chip")
            fill(c, x, z - 0.18, x + w, z + 0.42, col, 0.2)
            rect(c, x, z - 0.18, x + w, z + 0.42, col, 1.3)
            text(c, lab, Point3(x + w / 2, 0, z), 0.3, WHITE)
            out.append((c, x, x + w / 2, x + w))
            x += w + 0.1
        return out, x

    # ================================================================ 14  change one early word
    def s_change(self):
        n0, dh = self.n0, self.dh
        ctx2 = list(self.ctx)
        ci = ctx2.index(" tank")
        ctx2[ci] = " wafer"
        _, T2 = self.m.forward(ctx2, trace=True)
        zt = F.z - 3.0
        g = self.node("change")
        cols = [RED if i == ci else WHITE for i in range(n0)]
        bx = F.x + 1.6
        for i, tk in enumerate(ctx2):
            text(g, disp(tk), Point3(bx - 0.3, 0, zt - (i + 0.5) * RH - 0.11), 0.3, cols[i], align=TextNode.ARight)
        text(g, t("cache.chg.edit"), Point3(bx, 0, zt + 1.1), 0.4, RED, Fonts.symbol, TextNode.ALeft)
        diffs = []
        for l in range(2):
            K0 = self.T0["layers"][l]["heads"][0]["k"]
            K2 = T2["layers"][l]["heads"][0]["k"]
            b = Box(bx + l * 7.2, zt, n0, dh)
            self.frame(g, b, t("cache.chg.layer", l=l + 1), None)
            self.heat(K2, b, sc_of(K0), parent=g)
            dg = self.node("diff")
            for i in range(n0):
                dv = float(np.abs(K2[i] - K0[i]).max())
                col = GREEN if dv < 1e-9 else RED
                fill(dg, b.x1 + 0.3, b.rz(i) - RH / 2 + 0.06, b.x1 + 2.4, b.rz(i) + RH / 2 - 0.06, col, 0.25, y=0.03)
                text(dg, "{:.3f}".format(dv), Point3(b.x1 + 0.45, 0, b.rz(i) - 0.11), 0.28,
                     GREEN if dv < 1e-9 else TOO, align=TextNode.ALeft)
            text(dg, t("cache.l2.diff"), Point3(b.x1 + 0.3, 0, b.z_top + 0.35), 0.28, GREY, align=TextNode.ALeft)
            diffs.append(dg)
        n1 = self.note(t("cache.chg.n1"), bx, zt - n0 * RH - 1.6, 0.38, WHITE)
        n2 = self.note(t("cache.chg.n2"), bx, zt - n0 * RH - 2.4, 0.38, TOO)
        rule = self.note(t("cache.chg.rule"), bx, zt - n0 * RH - 4.2, 0.38, YELLOW)
        seq = Sequence(self.go(F.x - 2.5, bx + 19.0, zt - n0 * RH - 5.2, zt + 1.8), show(g, 0.6), Wait(0.6),
                       show(diffs[0], 0.5), show(n1, 0.4), Wait(1.4), show(diffs[1], 0.5), show(n2, 0.4),
                       Wait(1.0), show(rule, 0.5))
        self.chg_x1 = bx + 19.0
        return t("cache.chg.caption"), seq

    # ================================================================ 15  another model
    def s_model(self):
        n0, dh = self.n0, self.dh
        X0 = self.T0["x0"]
        Wk = self.H0["Wk"]
        rng = np.random.default_rng(7)
        Wk_b = rng.normal(0, float(np.std(Wk)), Wk.shape)          # "model B": other weights, same shape
        K_a, K_b = self.H0["k"], X0 @ Wk_b
        zt = F.z - 3.0
        x0 = self.chg_x1 + 3.5
        g = self.node("model")
        ba = Box(x0, zt, n0, dh)
        bb = Box(x0 + 6.0, zt, n0, dh)
        sc = sc_of(K_a)
        self.frame(g, ba, t("cache.mod.a"), None, GREEN, GREEN)
        self.frame(g, bb, t("cache.mod.b"), None, PURPLE, PURPLE)
        self.heat(K_a, ba, sc, parent=g)
        dg = self.node("mdiff")
        self.heat(K_b, bb, sc, parent=dg)
        for i in range(n0):
            dv = float(np.abs(K_b[i] - K_a[i]).max())
            fill(dg, bb.x1 + 0.3, bb.rz(i) - RH / 2 + 0.06, bb.x1 + 2.4, bb.rz(i) + RH / 2 - 0.06, RED, 0.25, y=0.03)
            text(dg, "{:.3f}".format(dv), Point3(bb.x1 + 0.45, 0, bb.rz(i) - 0.11), 0.28, TOO, align=TextNode.ALeft)
        text(dg, t("cache.mod.diff"), Point3(bb.x1 + 0.3, 0, bb.z_top + 0.35), 0.28, GREY, align=TextNode.ALeft)
        _, _, switch = one_message()
        _, hit, _ = one_message()
        note = self.note(t("cache.mod.note", hist=fmt_n(HIST), hit=fmt_usd(hit["usd"]), hity=fmt_yen(hit["usd"]),
                           sw=fmt_usd(switch), swy=fmt_yen(switch)), x0, zt - n0 * RH - 1.6, 0.36, WHITE)
        seq = Sequence(self.go(x0 - 1.5, bb.x1 + 12.0, zt - n0 * RH - 5.2, zt + 1.4), show(g, 0.5), Wait(0.8),
                       show(dg, 0.6), Wait(0.8), show(note, 0.6))
        return t("cache.mod.caption"), seq

    # ================================================================ 16  summary
    def s_summary(self):
        mem = self.note(t("cache.sum.mem"), O.x - 4.2, O.z + 8.2, 0.9, YELLOW)
        seq = Sequence(self.go(O.x - 6, E.x + 48, F.z - 14, O.z + 10.5), show(mem, 0.6))
        return t("cache.sum.caption"), seq

