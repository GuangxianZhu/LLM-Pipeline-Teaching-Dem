# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Question 7: why is the model fast? -- the KV cache and prompt-cache hits.

Board regions (far to the right of the other questions):
  K   top: context row -> Transformer -> KV cache shelf, and a "work per token" bar chart
  G   below: the attention table growing by one column (why old K and V can be reused)
  R   further below: request 1 vs request 2, prefix matching (cache hit / miss)
"""
from direct.interval.IntervalGlobal import (Func, LerpColorScaleInterval, LerpFunc, LerpPosInterval,
                                            Parallel, Sequence, Wait)
import random

from panda3d.core import Point3, TextNode, Vec3

import sim
from kit import (BLUE, DIM, GREEN, GREY, ORANGE, RED, TOKEN_COLORS, WHITE, YELLOW, arrow2d, disc, fade_in,
                 fade_out, fill, lines, rect, text, token_box, vlabel)
from story import Chip, Step, word

STAGES = ["No cache", "Why reuse", "KV cache", "Memory", "Request 2", "Prefix rule", "Summary"]

K = Point3(240, 0, 0)            # main area
G = K + Vec3(0, 0, -17)          # growing attention table
R = K + Vec3(0, 0, -34)          # request 1 / request 2

PROMPT = "Plot the tank temperature for the last hour."
GEN = ["Here", " is", " the", " chart"]          # the reply being generated
ROW_Z = 6.6
BOX = (K.x - 13, K.x - 1, 1.0, 3.2)              # transformer box: x0, x1, z0, z1
SHELF = (K.x + 1.5, K.x + 13.5, -0.6, 4.4)       # KV cache shelf: x0, x1, z0, z1
CHART_Z = -6.2                                   # baseline of the bar chart
GROUP_X = [-9.0, -4.0, 1.0, 6.0]                 # bar groups (one per generated token)
BAR_K = 0.26                                     # bar height per processed token
SEG = [("system + tools", 180, (0.55, 0.55, 0.58, 1)), ("user message", 10, BLUE),
       ("tool call", 25, ORANGE), ("tool result", 30, GREEN)]
UNIT = 0.088                                     # board units per token in the request bars


def hfill(parent, x0, z0, z1, color, alpha=0.35):
    """A horizontal bar that grows to the right from x0: call .setSx(width)."""
    np = fill(parent, 0, z0, 1, z1, color, alpha, y=-0.02)
    np.setX(x0)
    np.setSx(0.01)
    return np


class CacheStory:
    stages = STAGES

    def __init__(self, app, sc, deep=True):
        self.app = app
        self.sc = sc
        self.board = app.board
        self.steps = [Step("No cache", self.s_nocache), Step("Why reuse", self.s_why),
                      Step("KV cache", self.s_cache), Step("Memory", self.s_memory),
                      Step("Request 2", self.s_request2), Step("Prefix rule", self.s_prefix),
                      Step("Summary", self.s_summary)]
        self.intro = ('Question 7 continues question 3 ("{}"). Why can a model with billions of numbers write '
                      'its answer so fast - and why is the second request after a tool call even faster? '
                      'The answer is CACHING.'.format(PROMPT))
        self.app.cam_ctl.go_to(K + Vec3(0, 0, 0.6), 0, 0, 31, force=True)

    # ================================================================ helpers
    def view(self, target, h=0.0, p=0.0, d=24.0):
        return Func(self.app.cam_ctl.go_to, Point3(target), h, p, d)

    def chat(self, who, s):
        return Func(self.app.ui.chat, who, s)

    def row_positions(self, chips):
        x = K.x - 13.0
        out = []
        for c in chips:
            out.append(Point3(x + c.w / 2, 0, ROW_Z))
            x += c.w + 0.12
        return out

    def next_pos(self, chips, w):
        """Where a chip of width w goes after the chips already in the row (ignores animation state)."""
        x = K.x - 13.0 + sum(c.w + 0.12 for c in chips)
        return Point3(x + w / 2, 0, ROW_Z)

    def make_row(self, parent):
        chips = []
        for i, t in enumerate(sim.tokenize(PROMPT)):
            c = Chip(parent, word(t), TOKEN_COLORS[i % len(TOKEN_COLORS)], ("p",))
            chips.append(c)
        for c, p in zip(chips, self.row_positions(chips)):
            c.np.setPos(p)
        return chips

    def beams(self, chips, color=WHITE):
        """Lines from chips down into the transformer box (the tokens being processed)."""
        g = self.area.attachNewNode("beams")
        segs = []
        for c in chips:
            x = c.np.getX()
            bx = min(max(x, BOX[0] + 0.4), BOX[1] - 0.4)
            segs.append([(x, -0.01, ROW_Z - 0.2), (bx, -0.01, BOX[3])])
        lines(g, segs, color[:3] + (0.75,), 1.6)
        g.hide()
        return g

    def flash(self, nodes, color=(1.8, 1.7, 0.9, 1), dur=0.25):
        return Parallel(*[Sequence(LerpColorScaleInterval(n, dur, color, startColorScale=(1, 1, 1, 1)),
                                   LerpColorScaleInterval(n, dur, (1, 1, 1, 1))) for n in nodes])

    def counter(self, s):
        return Func(self.count_text.node().setText, s)

    def bar(self, slot, value, color, offset):
        """One bar of the 'work per token' chart. Returns an interval that grows it."""
        x = K.x + GROUP_X[slot] + offset
        b = fill(self.area, -0.35, 0, 0.35, 1, color, 0.9, y=0)
        b.setPos(x, 0, CHART_Z)
        b.setSz(0.001)
        lab = text(self.area, str(value), Point3(x, 0, CHART_Z + value * BAR_K + 0.2), 0.42, WHITE)
        lab.hide()
        return Sequence(LerpFunc(lambda v: b.setSz(max(0.001, v)), fromData=0.001, toData=value * BAR_K,
                                 duration=0.5, blendType="easeOut"),
                        Func(lab.show), fade_in(lab, 0.2))

    # ================================================================ build the main area once
    def _build_area(self):
        a = self.board.attachNewNode("cache-area")
        self.area = a
        text(a, "context", Point3(K.x - 13.0, 0, ROW_Z + 0.9), 0.4, GREY, align=TextNode.ALeft)
        fill(a, BOX[0], BOX[2], BOX[1], BOX[3], (0.75, 0.75, 0.78, 1), 0.12)
        self.box = rect(a, BOX[0], BOX[2], BOX[1], BOX[3], WHITE, 1.6)
        text(a, "Transformer", Point3((BOX[0] + BOX[1]) / 2, 0, 2.25), 0.6, WHITE)
        text(a, "computes K and V for every token it processes",
             Point3((BOX[0] + BOX[1]) / 2, 0, 1.45), 0.34, GREY)
        self.count_text = text(a, "", Point3((BOX[0] + BOX[1]) / 2, 0, 0.2), 0.42, YELLOW)
        # bar chart
        lines(a, [[(K.x - 11.5, 0, CHART_Z), (K.x + 8.5, 0, CHART_Z)]], GREY, 1.4)
        text(a, "work for each new token  (tokens pushed through the Transformer)",
             Point3(K.x - 11.5, 0, CHART_Z + 4.6), 0.4, GREY, align=TextNode.ALeft)
        for i, gx in enumerate(GROUP_X):
            text(a, "new token {}".format(i + 1), Point3(K.x + gx, 0, CHART_Z - 0.6), 0.36, GREY)
        fill(a, K.x + 9.4, CHART_Z + 3.0, K.x + 9.8, CHART_Z + 3.4, (0.5, 0.5, 0.52, 1), 0.9, y=0)
        text(a, "without cache", Point3(K.x + 10.0, 0, CHART_Z + 3.05), 0.38, WHITE, align=TextNode.ALeft)
        self.leg2 = a.attachNewNode("leg2")
        fill(self.leg2, K.x + 9.4, CHART_Z + 2.3, K.x + 9.8, CHART_Z + 2.7, YELLOW, 0.9, y=0)
        text(self.leg2, "with KV cache", Point3(K.x + 10.0, 0, CHART_Z + 2.35), 0.38, WHITE, align=TextNode.ALeft)
        self.leg2.hide()
        self.total1 = text(a, "", Point3(K.x + 9.4, 0, CHART_Z + 1.3), 0.38, WHITE, align=TextNode.ALeft)
        self.total2 = text(a, "", Point3(K.x + 9.4, 0, CHART_Z + 0.6), 0.38, YELLOW, align=TextNode.ALeft)

    # ================================================================ 1 no cache
    def s_nocache(self):
        self._build_area()
        self.row_node = self.area.attachNewNode("row1")
        chips = self.make_row(self.row_node)
        for c in chips:
            c.np.hide()
        seq = Sequence(self.view(K + Vec3(-0.5, 0, 0.4), 0, 0, 31), self.chat("You", PROMPT),
                       Parallel(*[Sequence(Wait(0.04 * i), Func(c.np.show), fade_in(c.np, 0.3))
                                  for i, c in enumerate(chips)]), Wait(0.4))
        total = 0
        for g, tok in enumerate(GEN):
            n = len(chips)
            total += n
            bm = self.beams(list(chips))
            new = Chip(self.row_node, word(tok), YELLOW, ("g",))
            dest = self.next_pos(chips, new.w)
            new.np.setPos((BOX[0] + BOX[1]) / 2, 0, BOX[2] + 1.1)
            new.np.hide()
            seq.append(Sequence(
                self.counter("processing {} tokens ...".format(n)),
                Func(bm.show), Parallel(fade_in(bm, 0.3), self.flash([c.np for c in chips])),
                self.flash([self.box], dur=0.3), Wait(0.2),
                Func(new.np.show), LerpPosInterval(new.np, 0.6, dest, blendType="easeInOut"),
                fade_out(bm, 0.3), Func(bm.removeNode),
                self.bar(g, n, (0.5, 0.5, 0.52, 1), -0.42),
                self.counter("generated '{}'  -  that took {} tokens of work".format(word(tok), n)),
                Wait(0.3)))
            chips.append(new)
        seq.append(Func(self.total1.node().setText, "total work: {}".format(total)))
        self.row1 = chips
        return ("Without any cache: to write each new token (yellow), the model pushes the WHOLE context through "
                "the Transformer again - all its layers, for every token. The work grows with every token "
                "(bars), and for a long answer that is a huge waste.", seq)

    # ================================================================ 2 why can we reuse?
    def s_why(self):
        toks = [word(t) for t in sim.tokenize(PROMPT)][:6]
        n = len(toks)
        cw, ch = 1.25, 0.8
        g = self.board.attachNewNode("grow")
        g.setPos(G)
        x0, z0 = -3.0, 2.2
        heads = g.attachNewNode("heads")
        nodes = {}

        def col_head(q, color=YELLOW):
            h = g.attachNewNode("ch")
            cx = x0 + (q + 0.5) * cw
            token_box(h, toks[q], TOKEN_COLORS[q % 8], (cx, 0, z0 + 1.5), 0.26)
            vlabel(h, "Q", q + 1, (cx, 0, z0 + 0.45), 0.36, color)
            return h

        def row_head(k, color=GREEN):
            h = g.attachNewNode("rh")
            cz = z0 - (k + 0.5) * ch - 0.1
            token_box(h, toks[k], TOKEN_COLORS[k % 8], (x0 - 2.2, 0, cz), 0.26)
            vlabel(h, "K", k + 1, (x0 - 0.7, 0, cz), 0.34, color)
            return h

        def cell(k, q, b):
            d = disc(g, 0.28, (b, b, b, 1), 24)
            d.setPos(x0 + (q + 0.5) * cw, 0.02, z0 - (k + 0.5) * ch)
            return d

        rng = random.Random(4)
        for i in range(n - 1):
            col_head(i).reparentTo(heads)
            row_head(i).reparentTo(heads)
            for k in range(i + 1):
                nodes[(k, i)] = cell(k, i, rng.uniform(0.25, 0.85))
        old = heads.attachNewNode("grid")
        lines(old, [[(x0, 0, z0), (x0 + (n - 1) * cw, 0, z0)], [(x0, 0, z0 + 2.0), (x0, 0, z0 - (n - 1) * ch)]],
              GREY, 1.4)
        q = n - 1
        new = g.attachNewNode("new")
        nh = col_head(q)
        nh.reparentTo(new)
        rh = row_head(q)
        rh.reparentTo(new)
        new_cells = [cell(k, q, rng.uniform(0.35, 0.95)) for k in range(n)]
        for d in new_cells:
            d.reparentTo(new)
        cx = x0 + (q + 0.5) * cw
        rect(new, cx - cw / 2, z0 - n * ch, cx + cw / 2, z0 + 0.95, YELLOW, 3.0, y=-0.03)
        note_old = text(g, "all of this was computed\nin the previous step\n= same numbers again",
                        Point3(x0 + (n - 1) * cw / 2 - 0.6, -0.05, z0 - n * ch - 0.9), 0.32, GREY)
        note_new = text(g, "only this column is new:\nthe new token's Q\ncompares with all K",
                        Point3(cx + 1.0, -0.05, z0 - n * ch / 2), 0.32, YELLOW, align=TextNode.ALeft)
        note_mask = text(g, "old tokens never look at\nnew ones (no arrows back)",
                         Point3(x0 - 2.6, -0.05, z0 - n * ch - 0.9), 0.3, GREY)
        for nd in (heads, new, note_old, note_new, note_mask):
            nd.hide()
        for d in nodes.values():
            d.hide()
        dim = Parallel(*[LerpColorScaleInterval(d, 0.6, (0.35, 0.35, 0.38, 1)) for d in nodes.values()] +
                       [LerpColorScaleInterval(heads, 0.6, (0.55, 0.55, 0.58, 1))])
        seq = Sequence(self.view(G + Vec3(0.6, 0, -0.6), 0, 0, 19),
                       Func(heads.show), fade_in(heads, 0.6),
                       Parallel(*[Sequence(Wait(0.05 * (k + qq)), Func(d.show), fade_in(d, 0.3))
                                  for (k, qq), d in nodes.items()]),
                       Wait(0.6), Func(new.show), fade_in(new, 0.6), Wait(0.3), dim,
                       Func(note_new.show), fade_in(note_new, 0.4), Wait(0.3),
                       Func(note_old.show), fade_in(note_old, 0.4),
                       Func(note_mask.show), fade_in(note_mask, 0.4))
        return ("Why is recomputing a waste? Remember the attention table. When a new token arrives, only ONE new "
                "column appears: its query compares with all keys. Old tokens may only look backwards, so their "
                "K and V can never change. Same input, same numbers - so just keep them!", seq)

    # ================================================================ 3 KV cache
    def s_cache(self):
        a = self.area
        rect(a, SHELF[0], SHELF[2], SHELF[1], SHELF[3], GREEN, 1.6)
        text(a, "KV cache  (stored K and V)", Point3((SHELF[0] + SHELF[1]) / 2, 0, SHELF[3] + 0.3), 0.4, GREEN)
        self.pairs = []
        old_row = self.row_node
        self.row_node = a.attachNewNode("row2")
        chips = self.make_row(self.row_node)
        for c in chips:
            c.np.hide()
        seq = Sequence(self.view(K + Vec3(-0.5, 0, 0.4), 0, 0, 31),
                       fade_out(old_row, 0.4), Func(old_row.hide), self.counter(""),
                       Parallel(*[Sequence(Wait(0.03 * i), Func(c.np.show), fade_in(c.np, 0.3))
                                  for i, c in enumerate(chips)]), Wait(0.3),
                       Func(self.leg2.show), fade_in(self.leg2, 0.4))
        # prefill: all prompt tokens once, their K/V go to the shelf
        bm = self.beams(chips)
        fly = Parallel()
        for i in range(len(chips)):
            fly.append(Sequence(Wait(0.06 * i), self.store_pair(chips[i])))
        seq.append(Sequence(self.counter("PREFILL: process the prompt once ({} tokens)".format(len(chips))),
                            Func(bm.show), fade_in(bm, 0.3), self.flash([self.box], dur=0.3), fly,
                            fade_out(bm, 0.3), Func(bm.removeNode),
                            self.bar(0, len(chips), YELLOW, 0.42)))
        total = len(chips)
        # decode: one token at a time
        for g, tok in enumerate(GEN):
            new = Chip(self.row_node, word(tok), YELLOW, ("g",))
            dest = self.next_pos(chips, new.w)
            new.np.setPos((BOX[0] + BOX[1]) / 2, 0, BOX[2] + 1.1)
            new.np.hide()
            chips.append(new)
            reads = self.read_beams()
            steps = [Func(new.np.show), LerpPosInterval(new.np, 0.6, dest, blendType="easeInOut")]
            if g < len(GEN) - 1:
                nb = self.beams([new], YELLOW)
                steps += [self.counter("only the NEW token is processed; old K, V are read from the cache"),
                          Func(nb.show), fade_in(nb, 0.2),
                          Func(reads.show), fade_in(reads, 0.25), self.flash([self.box], dur=0.25),
                          self.store_pair(new), fade_out(nb, 0.2), Func(nb.removeNode),
                          fade_out(reads, 0.25), Func(reads.removeNode),
                          self.bar(g + 1, 1, YELLOW, 0.42)]
                total += 1
            else:
                reads.removeNode()
            seq.append(Sequence(*steps))
        seq.append(Func(self.total2.node().setText, "total work: {}".format(total)))
        seq.append(self.counter("same answer, a fraction of the work"))
        return ("With a KV CACHE: the prompt is processed once (PREFILL) and every token's K and V are stored on "
                "the shelf. After that each new token needs only ONE pass; old K and V are simply read from "
                "memory (DECODE). Compare the bars.", seq)

    def pair_pos(self, i):
        col, row = i % 12, i // 12
        return Point3(SHELF[0] + 0.55 + col * 0.98, 0, SHELF[3] - 0.6 - row * 1.15)

    def store_pair(self, chip, dur=0.7):
        """K and V tiles fly from the transformer box to the next free place on the shelf."""
        i = len(self.pairs)
        p = self.pair_pos(i)
        g = self.area.attachNewNode("pair")
        for dz, col, letter in ((0.0, GREEN, "K"), (-0.47, RED, "V")):
            fill(g, -0.2, dz - 0.2, 0.2, dz + 0.2, col, 0.85, y=0)
            text(g, letter, Point3(0, -0.02, dz - 0.1), 0.24, (0.05, 0.05, 0.05, 1))
        text(g, chip.label[:5], Point3(0, -0.02, -0.92), 0.2, GREY)
        g.setPos((BOX[0] + BOX[1]) / 2 + 2, 0, 2.1)
        g.hide()
        self.pairs.append(g)
        return Sequence(Func(g.show), LerpPosInterval(g, dur, p, blendType="easeInOut"))

    def read_beams(self):
        g = self.area.attachNewNode("reads")
        segs = [[(pr.getX(), -0.01, pr.getZ()), (BOX[1], -0.01, 2.1)] for pr in self.pairs]
        lines(g, segs, GREEN[:3] + (0.55,), 1.2)
        g.hide()
        return g

    # ================================================================ 4 memory
    def s_memory(self):
        a = self.area
        fill_more = Parallel()
        extra = 24
        for k in range(extra):
            fake = Chip(a, "...", DIM, ("f",))
            fake.np.hide()
            fill_more.append(Sequence(Wait(0.05 * k), self.store_pair(fake, 0.4)))
        info = a.attachNewNode("mem")
        lines_ = ["for EVERY token the cache keeps:",
                  "K and V  x  32 layers  x  4096 numbers",
                  "= 262,144 numbers  =  about 0.5 MB",
                  "",
                  "10,000 tokens  ->  about 5 GB of GPU memory"]
        for i, s in enumerate(lines_):
            text(info, s, Point3(SHELF[0], 0, SHELF[2] - 1.0 - i * 0.55), 0.34,
                 YELLOW if i in (2, 4) else WHITE, align=TextNode.ALeft)
        info.hide()
        seq = Sequence(self.view(Point3((SHELF[0] + SHELF[1]) / 2, 0, SHELF[2] + 0.4), 0, 0, 19),
                       self.counter(""), fill_more, Wait(0.3), Func(info.show), fade_in(info, 0.6))
        return ("The price: memory. The cache must hold K and V for every token, in every layer (numbers for a "
                "typical 7-billion-parameter model, 16-bit). That is why long conversations are expensive, why "
                "models have a context limit, and why newer models use tricks to shrink the cache.", seq)

    # ================================================================ 5 request 2: prefix cache hit
    def req_bar(self, parent, segs, z, label):
        g = parent.attachNewNode("req")
        text(g, label, Point3(R.x - 10.9, 0, z - 0.02), 0.44, WHITE, align=TextNode.ARight)
        x = R.x - 10.5
        spans = []
        for name, n, col in segs:
            w = n * UNIT
            fill(g, x, z - 0.3, x + w - 0.06, z + 0.4, col, 0.25)
            rect(g, x, z - 0.3, x + w - 0.06, z + 0.4, col, 1.4)
            if w > 3.2:
                text(g, "{}  {}".format(name, n), Point3(x + w / 2, 0, z - 0.05), 0.38, WHITE)
            elif w > 1.5:
                text(g, "{}\n{}".format(name, n), Point3(x + w / 2, 0, z + 0.12), 0.27, WHITE)
            else:
                text(g, str(n), Point3(x + w / 2, 0, z - 0.05), 0.34, WHITE)
            spans.append((x, x + w, name, n))
            x += w
        return g, spans

    def s_request2(self):
        r = self.board.attachNewNode("requests")
        r.setPos(0, 0, 0)
        self.req = r
        z1, zc, z2 = R.z + 4.2, R.z + 2.4, R.z + 0.0
        b1, s1 = self.req_bar(r, SEG[:2], z1, "request 1")
        out1 = r.attachNewNode("out1")
        x_end = s1[-1][1]
        text(out1, "-> tool call -> tool runs ...", Point3(x_end + 0.4, 0, z1 - 0.05),
             0.38, GREY, align=TextNode.ALeft)
        cache, sc = self.req_bar(r, SEG[:2], zc, "server cache")
        for x0, x1, _, _ in sc:
            for xx in [x0 + 0.25 + k * 0.5 for k in range(int((x1 - x0 - 0.2) / 0.5))]:
                fill(cache, xx - 0.07, zc - 0.25, xx + 0.07, zc - 0.05, GREEN, 0.9, y=-0.01)
                fill(cache, xx - 0.07, zc + 0.12, xx + 0.07, zc + 0.32, RED, 0.9, y=-0.01)
        b2, s2 = self.req_bar(r, SEG, z2, "request 2")
        cursor = lines(r, [[(0, -0.03, z2 - 0.7), (0, -0.03, zc + 0.7)]], YELLOW, 3.0)
        cursor.setX(R.x - 10.5)
        hit_end = s2[1][1]
        full_end = s2[-1][1]
        hit_box = hfill(r, R.x - 10.5, z2 - 0.45, z2 + 0.55, GREEN)
        miss_box = hfill(r, hit_end, z2 - 0.45, z2 + 0.55, ORANGE)
        hit_lab = text(r, "HIT: 190 tokens reused", Point3((R.x - 10.5 + hit_end) / 2, 0, z2 - 1.2), 0.46, GREEN)
        miss_lab = text(r, "MISS: 55 computed", Point3((hit_end + full_end) / 2, 0, z2 - 1.2), 0.46, ORANGE)
        # time to first token
        tz = R.z - 3.4
        ttft = r.attachNewNode("ttft")
        text(ttft, "work before the first new token", Point3(R.x - 10.5, 0, tz + 0.75), 0.4, GREY,
             align=TextNode.ALeft)
        text(ttft, "no cache", Point3(R.x - 10.9, 0, tz - 0.05), 0.38, WHITE, align=TextNode.ARight)
        text(ttft, "cache hit", Point3(R.x - 10.9, 0, tz - 0.85), 0.38, WHITE, align=TextNode.ARight)
        fill(ttft, R.x - 10.5, tz - 0.15, R.x - 10.5 + 245 * UNIT, tz + 0.3, (0.5, 0.5, 0.52, 1), 0.9, y=0)
        fill(ttft, R.x - 10.5, tz - 0.95, R.x - 10.5 + 55 * UNIT, tz - 0.5, YELLOW, 0.9, y=0)
        text(ttft, "245 tokens", Point3(R.x - 10.5 + 245 * UNIT + 0.3, 0, tz - 0.02), 0.38, WHITE,
             align=TextNode.ALeft)
        text(ttft, "55 tokens  -> faster, and cached tokens are usually billed much cheaper",
             Point3(R.x - 10.5 + 55 * UNIT + 0.3, 0, tz - 0.82), 0.38, YELLOW, align=TextNode.ALeft)
        for nd in (b1, out1, cache, b2, cursor, hit_box, miss_box, hit_lab, miss_lab, ttft):
            nd.hide()

        def grow(box, x0, v):
            box.setSx(max(0.01, v - x0))

        hb0 = R.x - 10.5
        seq = Sequence(
            self.view(R + Vec3(-0.6, 0, 0.8), 0, 0, 28),
            Func(b1.show), fade_in(b1, 0.5), Func(out1.show), fade_in(out1, 0.5), Wait(0.4),
            Func(cache.show), fade_in(cache, 0.6), Wait(0.4),
            Func(b2.show), fade_in(b2, 0.6), Wait(0.4),
            Func(cursor.show), Func(hit_box.show),
            LerpFunc(lambda v: (cursor.setX(v), grow(hit_box, hb0, v)), fromData=hb0, toData=hit_end,
                     duration=2.4, blendType="easeInOut"),
            Func(hit_lab.show), fade_in(hit_lab, 0.3),
            Func(cursor.setColor, *ORANGE), Func(miss_box.show),
            LerpFunc(lambda v: (cursor.setX(v), grow(miss_box, hit_end, v)), fromData=hit_end, toData=full_end,
                     duration=1.2, blendType="easeInOut"),
            Func(miss_lab.show), fade_in(miss_lab, 0.3), Wait(0.4),
            Func(ttft.show), fade_in(ttft, 0.6))
        self.z2, self.s2 = z2, s2
        return ("Across requests: after the tool runs, the program sends a SECOND request. Its beginning is exactly "
                "the same as request 1. The server kept the K and V of that prefix (prompt cache), so it compares "
                "token by token from the start: everything identical is a CACHE HIT; only the new part is computed.",
                seq)

    # ================================================================ 6 the prefix rule
    def s_prefix(self):
        r = self.req
        z3 = R.z - 7.0
        segs = [("system + tools", 180, (0.55, 0.55, 0.58, 1))] + SEG[1:]
        b3, s3 = self.req_bar(r, segs, z3, "request 2'")
        cut_x = R.x - 10.5 + 6 * UNIT
        mark = r.attachNewNode("mark")
        lines(mark, [[(cut_x, -0.04, z3 - 0.5), (cut_x, -0.04, z3 + 0.6)]], RED, 3.5)
        text(mark, 'someone added "Time: 14:47" here', Point3(cut_x - 0.1, 0, z3 + 1.35), 0.4, RED,
             align=TextNode.ALeft)
        full_end = s3[-1][1]
        hit = fill(r, R.x - 10.5, z3 - 0.45, cut_x, z3 + 0.55, GREEN, 0.35, y=-0.02)
        miss = hfill(r, cut_x, z3 - 0.45, z3 + 0.55, RED)
        cursor = lines(r, [[(0, -0.03, z3 - 0.7), (0, -0.03, z3 + 0.7)]], RED, 3.0)
        cursor.setX(cut_x)
        lab = text(r, "MISS from the change to the end: about 240 tokens computed again",
                   Point3((cut_x + full_end) / 2, 0, z3 - 1.2), 0.44, RED)
        why = r.attachNewNode("why")
        for k in range(7):
            x = cut_x + 1.5 + k * 3.0
            if x < full_end - 0.3:
                arrow2d(why, (cut_x + 0.2, -0.05, z3 + 0.45 + 0.08 * k), (x, -0.05, z3 + 0.45), RED, 1.2, 0.18)
        tip = text(r, "rule: put things that never change FIRST, things that change at the END",
                   Point3(R.x - 10.5, 0, z3 - 2.3), 0.44, YELLOW, align=TextNode.ALeft)
        for nd in (b3, mark, hit, miss, cursor, lab, why, tip):
            nd.hide()
        seq = Sequence(
            self.view(R + Vec3(-0.6, 0, -2.4), 0, 0, 31),
            Func(b3.show), fade_in(b3, 0.5), Func(mark.show), fade_in(mark, 0.4), Wait(0.4),
            Func(hit.show), fade_in(hit, 0.4), Func(cursor.show), Func(miss.show),
            LerpFunc(lambda v: (cursor.setX(v), miss.setSx(max(0.01, v - cut_x))), fromData=cut_x,
                     toData=full_end, duration=1.8, blendType="easeInOut"),
            Func(lab.show), fade_in(lab, 0.3), Wait(0.3), Func(why.show), fade_in(why, 0.5), Wait(0.4),
            Func(tip.show), fade_in(tip, 0.5))
        return ("Why only the PREFIX? Every token's K and V depend on ALL tokens before it (attention mixed them "
                "in). Change one early token and every K and V after it is different - the cache is useless from "
                "there on. One small change at the start = a full miss.", seq)

    # ================================================================ 7 summary
    def s_summary(self):
        seq = Sequence(self.view(K + Vec3(0, 0, -14), 0, 0, 62), Wait(0.5))
        return ("Summary:  KV cache = inside one answer, store each token's K and V so every new token needs only "
                "one pass.  Prompt cache = between requests, reuse the stored K and V of an identical prefix "
                "(hit), compute only the new part (miss).  Cost: GPU memory.", seq)
