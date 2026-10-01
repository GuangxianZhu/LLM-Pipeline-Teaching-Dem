# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
Story = one question turned into a list of animated steps on one big black board.

Board regions (the camera flies between them):
  A  context, tokens, vectors, and the big-picture model behind them (+y)
  B  3D meaning space (arrows)
  C  Q/K attention table, with the weighted sum (delta E) below it
  P  prediction loop: context strip -> Transformer x32 -> next-token probabilities
  T  the program (harness) and the four real tools
"""
import random
import zlib

from direct.interval.IntervalGlobal import (Func, LerpColorScaleInterval, LerpFunc,
                                            LerpPosInterval, LerpScaleInterval, Parallel,
                                            Sequence, Wait)
from panda3d.core import AmbientLight, DirectionalLight, Point3, TextNode, TransparencyAttrib, Vec3, Vec4

import sim
import tools
from kit import (BLUE, DIM, EDGE_BLUE, EDGE_RED, GREEN, GREY, ORANGE, RED, TOKEN_COLORS, WHITE,
                 YELLOW, Arrow, Fonts, arrow2d, array_to_texture, column, curve_arrow2d, disc,
                 fade_in, fade_out, fill, image_card, lines, num, rect, slab, sphere, text,
                 text_width, token_box, vlabel)
from scenarios import SYSTEM_PROMPT, tool_call_text

STAGES = ["Context", "Tokens", "Vectors", "Attention", "MLP", "Predict", "Tool", "Return", "Answer"]

SEC_A = Point3(0, 0, 0)
SEC_B = Point3(30, 6, -0.5)
SEC_C = Point3(60, 0, 0)
SEC_P = Point3(104, 0, 0)
SEC_T = Point3(152, 0, 0)

CENTERS = [Point3(3.3, 0.9, 0.9), Point3(-1.6, 2.6, 2.3), Point3(0.9, -3.2, 1.0), Point3(2.6, 3.2, -0.4)]
FUNC_C = Point3(-3.0, -1.0, -0.3)
VAGUE = Point3(-0.4, -0.9, 3.3)

TOOL_PANELS = {"calculator": (-1.5, 3.6), "run_terminal": (9.0, 3.6),
               "plot_chart": (-1.5, -2.6), "generate_image": (9.0, -2.6)}
PANEL_W, PANEL_H = 9.6, 5.6
CTX_W = 24.0                     # width of the context strip in the prediction area
PRED_VIEW = SEC_P + Vec3(0.6, 0, 2.6)
PRED_D = 27.0


class Step:
    def __init__(self, stage, build):
        self.stage, self.build = stage, build


class Chip:
    """A small token chip in the context strip."""

    def __init__(self, parent, label, color, seg, scale=0.36):
        self.np = parent.attachNewNode("chip")
        self.label, self.color, self.seg = label, color, seg
        w = max(0.4, text_width(label) * scale + 0.22)
        self.w = w
        fill(self.np, -w / 2, -0.15, w / 2, 0.43, color, 0.2)
        rect(self.np, -w / 2, -0.15, w / 2, 0.43, color, 1.3)
        text(self.np, label, Point3(0, 0, 0), scale, WHITE)


def word(t):
    return t.strip() or t


class Story:
    def __init__(self, app, sc, deep=True):
        self.app = app
        self.sc = sc
        self.deep = deep
        self.board = app.board
        self.toks = sim.tokenize(sc["prompt"])
        self.fq = self.toks.index(sc["focus"])
        self.results = []
        self.rounds = {}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": sc["prompt"]}]
        order = [sc["focus_group"]]
        for t in self.toks:
            g = self.group(t)
            if g != "func" and g not in order:
                order.append(g)
        self.group_order = order
        self.W, self.S = self._weights()
        self.steps = self._build_steps()

    # ================================================================ helpers
    def group(self, t):
        return self.sc["groups"].get(t.strip().lower(), "func")

    def space_pos(self, t):
        if t == self.sc["focus"]:
            return Point3(VAGUE)
        g = self.group(t)
        c = FUNC_C if g == "func" else CENTERS[self.group_order.index(g) % len(CENTERS)]
        rng = random.Random(zlib.crc32(("pos" + t).encode("utf-8")))
        return c + Vec3(rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), rng.uniform(-0.5, 0.5))

    def view(self, target, h=0.0, p=0.0, d=24.0):
        return Func(self.app.cam_ctl.go_to, Point3(target), h, p, d)

    def orbit(self, target, h0, h1, p, d, dur):
        return LerpFunc(lambda h: self.app.cam_ctl.go_to(Point3(target), h, p, d), fromData=h0, toData=h1,
                        duration=dur, blendType="easeInOut")

    def chat(self, who, s):
        return Func(self.app.ui.chat, who, s)

    def _weights(self):
        n = len(self.toks)
        rng = random.Random(11 + len(self.sc["prompt"]))
        W = [[0.0] * n for _ in range(n)]          # W[key][query]
        S = [[None] * n for _ in range(n)]         # raw scores
        for q in range(n):
            gq = self.group(self.toks[q])
            scores = []
            for k in range(q + 1):
                s = rng.uniform(0, 0.8)
                if gq != "func" and self.group(self.toks[k]) == gq:
                    s += 2.2
                if q == n - 1 and self.group(self.toks[k]) != "func":
                    s += 1.2
                scores.append(s)
            p = sim.softmax(scores, 0.7)
            for k in range(q + 1):
                W[k][q] = p[k]
                S[k][q] = scores[k] / 0.7 - 1.5
        return W, S

    # ================================================================ step list
    def _build_steps(self):
        st = [Step("Context", self.s_context), Step("Tokens", self.s_tokens),
              Step("Vectors", self.s_vectors), Step("Vectors", self.s_space),
              Step("Attention", self.s_big_picture)]
        if self.deep:
            st += [Step("Attention", self.s_qk_table), Step("Attention", self.s_weights),
                   Step("Attention", self.s_delta), Step("Attention", self.s_update),
                   Step("MLP", self.s_mlp)]
        n = len(self.sc["calls"])
        for i in range(n + 1):
            if i < n:
                st += [Step("Predict", lambda i=i: self.s_predict(i, "first")),
                       Step("Predict", lambda i=i: self.s_fast(i, "to_name")),
                       Step("Predict", lambda i=i: self.s_predict(i, "name")),
                       Step("Predict", lambda i=i: self.s_fast(i, "rest")),
                       Step("Tool", lambda i=i: self.s_program(i)),
                       Step("Tool", lambda i=i: self.s_run(i)),
                       Step("Return", lambda i=i: self.s_return(i))]
            else:
                st += [Step("Predict", lambda i=i: self.s_predict(i, "first")),
                       Step("Predict", lambda i=i: self.s_fast(i, "rest")),
                       Step("Answer", self.s_answer)]
        return st

    # ================================================================ A: context
    def s_context(self):
        b = self.board
        a = b.attachNewNode("context")
        self.ctx_node = a
        pr = self.sc["prompt"]
        sc_big = min(0.95, 21.0 / max(1.0, text_width(pr)))
        lines_ = [("system:", SYSTEM_PROMPT), ("tools:", "calculator    run_terminal    plot_chart    generate_image")]
        hidden = a.attachNewNode("hidden")
        for i, (k, v) in enumerate(lines_):
            text(hidden, k, Point3(-10.5, 0, 7.4 - i * 0.75), 0.4, GREY, align=TextNode.ARight)
            text(hidden, v, Point3(-10.1, 0, 7.4 - i * 0.75), 0.4, GREY, align=TextNode.ALeft)
        text(hidden, "user:", Point3(-10.5, 0, 5.9), 0.4, GREY, align=TextNode.ARight)
        self.hidden = hidden
        self.sentence = text(a, pr, Point3(0, 0, 4.2), sc_big, WHITE)
        self.sentence.hide()
        seq = Sequence(self.view(SEC_A + Vec3(0, 0, 3.0), 0, 0, 24), self.chat("You", pr),
                       fade_in(hidden, 0.8), Wait(0.3), Func(self.sentence.show), fade_in(self.sentence, 1.0))
        return ("Your message is never sent alone. A hidden SYSTEM PROMPT and the LIST OF TOOLS the model may "
                "use come first. Together with your words they form the CONTEXT - the only thing the model sees.",
                seq)

    # ================================================================ A: tokens
    def s_tokens(self):
        sc = 0.62
        widths = [max(1.15, text_width(word(t)) * sc + 0.45) for t in self.toks]
        gap = 0.18
        total = sum(widths) + gap * (len(widths) - 1)
        f = min(1.0, 21.0 / total)
        self.fit = f
        widths = [w * f for w in widths]
        gap *= f
        total = sum(widths) + gap * (len(widths) - 1)
        x = SEC_A.x - total / 2
        self.tok_x = []
        for w in widths:
            self.tok_x.append(x + w / 2)
            x += w + gap
        self.tok_w = widths
        self.tok_color = [TOKEN_COLORS[i % len(TOKEN_COLORS)] for i in range(len(self.toks))]
        self.tok_z = 2.0
        seq = Parallel(Sequence(fade_out(self.sentence, 0.5), Func(self.sentence.hide)),
                       LerpColorScaleInterval(self.hidden, 0.6, (1, 1, 1, 0.45)))
        z = self.tok_z
        for i, (t, w) in enumerate(zip(self.toks, widths)):
            g = self.board.attachNewNode("tok")
            c, xx = self.tok_color[i], self.tok_x[i]
            fill(g, xx - w / 2, z - 0.35 * f, xx + w / 2, z + 0.62 * f, c)
            rect(g, xx - w / 2, z - 0.35 * f, xx + w / 2, z + 0.62 * f, c, 2.2)
            text(g, word(t), Point3(xx, 0, z), sc * f, WHITE)
            text(g, str(sim.token_id(t)), Point3(xx, 0, z - 0.85 * f), 0.32 * f, GREY)
            g.hide()
            seq.append(Sequence(Wait(0.3 + 0.1 * i), Func(g.show), fade_in(g, 0.4)))
        splits = [word(t) for t in self.toks if not t.startswith(" ") and t.isalpha()][1:]
        note = (" Look: '{}' is only part of a word.".format(splits[0]) if splits else "")
        return ("The tokenizer cuts your text into TOKENS (words or pieces of words). Each token gets an ID "
                "number." + note, Sequence(self.view(SEC_A + Vec3(0, 0, 2.6), 0, 0, 24), seq))

    # ================================================================ A: vectors
    def s_vectors(self):
        par = Parallel()
        sc = 0.34 * self.fit
        for i, t in enumerate(self.toks):
            x = self.tok_x[i]
            top = self.tok_z - 1.5
            col = column(self.board, x, top, sim.embedding(t, 6), WHITE, sc, dz=0.48 * self.fit)
            lines(col, [[(x, 0, self.tok_z - 0.95 * self.fit), (x, 0, top + 0.5)]], DIM, 1.2)
            col.hide()
            par.append(Sequence(Wait(0.1 * i), Func(col.show), fade_in(col, 0.5),
                                LerpPosInterval(col, 0.5, Point3(0, 0, 0), startPos=Point3(0, 0, 0.6),
                                                blendType="easeOut")))
        return ("Each token ID is looked up in a giant table and turned into a long list of numbers - a VECTOR "
                "(6 shown here, real models use thousands). From now on the model only computes with numbers.",
                Sequence(self.view(SEC_A + Vec3(0, 0, -0.3), 0, 0, 22), par))

    # ================================================================ B: meaning space
    def s_space(self):
        sp = self.board.attachNewNode("space")
        sp.setPos(SEC_B)
        self.space = sp
        L = 4.2
        grid = []
        for k in range(-4, 5):
            grid.append([(k, -4, 0), (k, 4, 0)])
            grid.append([(-4, k, 0), (4, k, 0)])
        g = lines(sp, grid, (0.16, 0.18, 0.2, 1), 1.0)
        axes = lines(sp, [[(-L, 0, 0), (L, 0, 0)], [(0, -L, 0), (0, L, 0)], [(0, 0, -2.5), (0, 0, L)]], GREY, 2.0)
        ticks = []
        for k in range(-4, 5):
            if k:
                ticks += [[(k, 0, -0.08), (k, 0, 0.08)], [(0, k, -0.08), (0, k, 0.08)]]
        for k in range(-2, 5):
            if k:
                ticks.append([(-0.08, 0, k), (0.08, 0, k)])
        tk = lines(sp, ticks, GREY, 1.5)
        par = Parallel(fade_in(g, 1.0), fade_in(axes, 1.0), fade_in(tk, 1.0))
        seen = set()
        flights = Parallel()
        self.arrows = {}
        self.space_labels = {}
        for i, t in enumerate(self.toks):
            end = self.space_pos(t)
            a = Arrow(sp, Point3(0, 0, 0), end, self.tok_color[i], 3.0)
            self.arrows.setdefault(t, a)
            steps = [Wait(1.2 + 0.22 * i), LerpFunc(a.grow, fromData=0.001, toData=1.0, duration=0.9,
                                                    blendType="easeOut")]
            if t not in seen:
                seen.add(t)
                nb = sum(1 for o in self.space_labels if (self.space_pos(o) - end).length() < 1.1)
                lab = text(sp, word(t), end * 1.13 + Vec3(0, 0, 0.12 + 0.42 * nb), 0.34, self.tok_color[i])
                lab.setBillboardPointEye()
                lab.hide()
                self.space_labels[t] = lab
                steps += [Func(lab.show), fade_in(lab, 0.3)]
            flights.append(Sequence(*steps))
        grp = [word(t) for t in self.toks if self.group(t) == self.sc["focus_group"] and t != self.sc["focus"]]
        return ("Think of each vector as an ARROW in space (squeezed into 3D here). Words with related meaning "
                "point in similar directions ({}). '{}' on its own is vague."
                .format(" / ".join(dict.fromkeys(grp)), word(self.sc["focus"])),
                Sequence(self.view(SEC_B + Vec3(0, 0, 0.8), -35, 24, 16),
                         Parallel(par, flights, self.orbit(SEC_B + Vec3(0, 0, 0.8), -35, 30, 24, 16, 7.0))))

    # ================================================================ A: big picture
    def s_big_picture(self):
        xs = self.tok_x
        f = self.fit
        x0, x1 = min(xs) - 0.9, max(xs) + 0.9
        top = self.tok_z - 1.5
        zt, zb = top + 0.9, top - 4.0
        xc = (x0 + x1) / 2
        bp = self.board.attachNewNode("bigpicture")
        att = slab(bp, x0, x1, 1.6, 6.0, zb, zt)
        att_lab = text(bp, "Attention", Point3(xc, 3.8, zt + 0.6), 1.1, WHITE)
        att_lab.setBillboardAxis()
        moves = Parallel()
        for i, t in enumerate(self.toks):
            vals = [v + 0.6 * d for v, d in zip(sim.embedding(t, 6), sim.embedding("ctx" + t, 6))]
            c = column(bp, xs[i], top, vals, WHITE, 0.34 * f, dz=0.48 * f)
            c.hide()
            moves.append(Sequence(Wait(0.06 * i), Func(c.show),
                                  LerpPosInterval(c, 2.0, Point3(0, 8.0, 0), startPos=Point3(0, 0, 0),
                                                  blendType="easeInOut")))
        mlp = bp.attachNewNode("mlp")
        al = AmbientLight("a")
        al.setColor(Vec4(0.35, 0.35, 0.38, 1))
        dl = DirectionalLight("d")
        dl.setColor(Vec4(0.9, 0.9, 0.9, 1))
        dnp = mlp.attachNewNode(dl)
        dnp.setHpr(-35, -40, 0)
        mlp.setLight(mlp.attachNewNode(al))
        mlp.setLight(dnp)
        rng = random.Random(5)
        layers = []
        for ly, n in ((10.5, 24), (13.0, 32), (15.5, 24)):
            layers.append([Point3(rng.uniform(x0 + 0.4, x1 - 0.4), ly + rng.uniform(-0.5, 0.5),
                                  rng.uniform(zb + 0.4, zt - 0.4)) for _ in range(n)])
        self.mlp_edges = []
        for a, b2 in zip(layers, layers[1:]):
            reds, blues = [], []
            for p in a:
                for q in rng.sample(b2, 4):
                    (reds if rng.random() < 0.5 else blues).append([p, q])
            grp = mlp.attachNewNode("edges")
            grp.setLightOff(1)
            lines(grp, reds, EDGE_RED, 2.0)
            lines(grp, blues, EDGE_BLUE, 2.0)
            self.mlp_edges.append(grp)
        self.mlp_nodes = []
        for layer in layers:
            g = mlp.attachNewNode("layer")
            for p in layer:
                sphere(g, 0.16, (0.88, 0.88, 0.9, 1)).setPos(p)
            g.flattenStrong()
            self.mlp_nodes.append(g)
        mlp_lab = text(bp, "Multilayer\nPerceptron", Point3(xc, 13.0, zt + 1.6), 0.95, WHITE)
        mlp_lab.setBillboardAxis()
        self.mlp_center = Point3(xc, 13.0, (zt + zb) / 2)
        holder = bp.attachNewNode("cols3")
        holder.setY(8.0)
        out = Parallel()
        for i, t in enumerate(self.toks):
            vals = [v + 0.9 * d for v, d in zip(sim.embedding(t, 6), sim.embedding("mlp" + t, 6))]
            c = column(holder, xs[i], top, vals, WHITE, 0.34 * f, dz=0.48 * f)
            c.hide()
            out.append(Sequence(Wait(0.05 * i), Func(c.show),
                                LerpPosInterval(c, 2.4, Point3(0, 10.5, 0), startPos=Point3(0, 0, 0),
                                                blendType="easeInOut")))
        dots = text(bp, ". . .", Point3(xc, 22.0, (zt + zb) / 2), 1.2, GREY)
        dots.setBillboardAxis()
        rep = text(bp, "x 32 layers", Point3(xc, 22.0, zt + 0.6), 0.8, YELLOW)
        rep.setBillboardAxis()
        for nd in (att, att_lab, mlp_lab, dots, rep, mlp):
            nd.hide()
        center = Point3(xc, 10.5, (zt + zb) / 2 + 0.5)
        d = 33 * max(1.0, (x1 - x0) / 18.0)
        seq = Sequence(
            self.view(center, 62, 16, d),
            Parallel(
                self.orbit(center, 62, 44, 16, d, 11.0),
                Sequence(Func(att.show), fade_in(att, 0.8), Func(att_lab.show), fade_in(att_lab, 0.5),
                         moves,
                         Func(mlp.show), Parallel(*[Sequence(Wait(0.25 * k), fade_in(nd, 0.5))
                                                    for k, nd in enumerate(self.mlp_nodes + self.mlp_edges)]),
                         Func(mlp_lab.show), fade_in(mlp_lab, 0.5),
                         Parallel(out, self.pulse(0.3)),
                         Func(dots.show), Func(rep.show), Parallel(fade_in(dots, 0.5), fade_in(rep, 0.5)))))
        tail = ("" if self.deep else "  (Deep dive is OFF - press D and restart to see inside attention.)")
        return ("The big picture: all vectors flow through an ATTENTION block (tokens exchange information), "
                "then a MULTILAYER PERCEPTRON (each vector is processed on its own). This pair is repeated "
                "many times." + tail, seq)

    def pulse(self, each=0.35):
        seq = Sequence()
        for k, nd in enumerate(self.mlp_nodes):
            seq.append(LerpColorScaleInterval(nd, each, (1.8, 1.8, 1.6, 1), startColorScale=(1, 1, 1, 1)))
            seq.append(LerpColorScaleInterval(nd, each, (1, 1, 1, 1)))
            if k < len(self.mlp_edges):
                e = self.mlp_edges[k]
                seq.append(LerpColorScaleInterval(e, each, (1.7, 1.7, 1.7, 1), startColorScale=(1, 1, 1, 1)))
                seq.append(LerpColorScaleInterval(e, each, (1, 1, 1, 1)))
        return seq

    # ================================================================ C: Q / K table
    def s_qk_table(self):
        n = len(self.toks)
        cw, ch = 1.3, 0.8
        g = self.board.attachNewNode("qk")
        g.setPos(SEC_C)
        self.qk = g
        gx0 = -n * cw / 2 + 2.6
        gz0 = 2.6
        self.cw, self.ch, self.gx0, self.gz0 = cw, ch, gx0, gz0
        sep = lines(g, [[(gx0 - 6.3, 0, gz0), (gx0 + n * cw, 0, gz0)],
                        [(gx0, 0, gz0 + 3.6), (gx0, 0, gz0 - n * ch)]], GREY, 1.6)
        cols, rows = [], []
        for q, t in enumerate(self.toks):
            cx = gx0 + (q + 0.5) * cw
            h = g.attachNewNode("colhead")
            token_box(h, word(t), self.tok_color[q], (cx, 0, gz0 + 3.15), 0.26)
            arrow2d(h, (cx, 0, gz0 + 2.95), (cx, 0, gz0 + 2.55), WHITE, 1.4, 0.12)
            vlabel(h, "E", q + 1, (cx, 0, gz0 + 2.1), 0.36, WHITE)
            arrow2d(h, (cx, 0, gz0 + 1.9), (cx, 0, gz0 + 1.2), WHITE, 1.4, 0.12)
            vlabel(h, "W", "Q", (cx + 0.32, 0, gz0 + 1.5), 0.22, YELLOW, arrow=False)
            vlabel(h, "Q", q + 1, (cx, 0, gz0 + 0.65), 0.38, YELLOW)
            h.flattenStrong()
            cols.append(h)
        for k, t in enumerate(self.toks):
            cz = gz0 - (k + 0.5) * ch - 0.1
            h = g.attachNewNode("rowhead")
            token_box(h, word(t), self.tok_color[k], (gx0 - 5.4, 0, cz), 0.26)
            arrow2d(h, (gx0 - 4.75, 0, cz + 0.1), (gx0 - 4.25, 0, cz + 0.1), WHITE, 1.4, 0.12)
            vlabel(h, "E", k + 1, (gx0 - 3.85, 0, cz), 0.32, WHITE)
            arrow2d(h, (gx0 - 3.35, 0, cz + 0.1), (gx0 - 1.75, 0, cz + 0.1), WHITE, 1.4, 0.12)
            vlabel(h, "W", "K", (gx0 - 2.55, 0, cz + 0.22), 0.2, GREEN, arrow=False)
            vlabel(h, "K", k + 1, (gx0 - 1.15, 0, cz), 0.34, GREEN)
            h.flattenStrong()
            rows.append(h)
        self.cells = {}
        cells_anim = Parallel()
        for q in range(n):
            for k in range(q + 1):
                cx = gx0 + (q + 0.5) * cw
                cz = gz0 - (k + 0.5) * ch - 0.08
                c = g.attachNewNode("cell")
                vlabel(c, "K", k + 1, (cx - 0.3, 0, cz), 0.24, GREEN)
                text(c, "·", Point3(cx, 0, cz), 0.3, WHITE)
                vlabel(c, "Q", q + 1, (cx + 0.3, 0, cz), 0.24, YELLOW)
                c.flattenStrong()
                c.hide()
                self.cells[(k, q)] = c
                cells_anim.append(Sequence(Wait(2.2 + 0.1 * q + 0.03 * k), Func(c.show), fade_in(c, 0.35)))
        for h in cols + rows:
            h.hide()
        width = n * cw + 6.3
        self.qk_center = SEC_C + Vec3(gx0 - 6.3 + width / 2, 0, gz0 - n * ch / 2 + 0.9)
        self.qk_dist = max(20.0, width * 1.12)
        seq = Sequence(
            self.view(self.qk_center, 0, 0, self.qk_dist),
            Parallel(fade_in(sep, 0.6),
                     Parallel(*[Sequence(Wait(0.07 * i), Func(h.show), fade_in(h, 0.5)) for i, h in enumerate(cols)]),
                     Parallel(*[Sequence(Wait(0.8 + 0.07 * i), Func(h.show), fade_in(h, 0.5))
                                for i, h in enumerate(rows)]),
                     cells_anim))
        return ("Inside the attention block. Each vector E is multiplied by a matrix W_Q to get a QUERY Q "
                "('what am I looking for?') and by W_K to get a KEY K ('what do I contain?'). "
                "Every cell compares a key with a query: K . Q", seq)

    # ================================================================ C: weights
    def s_weights(self):
        n = len(self.toks)
        g, cw, ch, gx0, gz0 = self.qk, self.cw, self.ch, self.gx0, self.gz0
        to_scores, to_weights = Parallel(), Parallel()
        self.wtext = {}
        for (k, q), c in self.cells.items():
            cx = gx0 + (q + 0.5) * cw
            cz = gz0 - (k + 0.5) * ch - 0.08
            s = text(g, num(self.S[k][q]), Point3(cx, 0, cz), 0.3, GREY)
            s.hide()
            w = self.W[k][q]
            b = 0.25 + 0.75 * min(1.0, w * 1.6)
            bg = disc(g, 0.3, (b * 0.55, b * 0.55, b * 0.55, 1), 24)
            bg.setPos(cx, 0.03, cz + 0.08)
            bg.setScale(0.001)
            wt = text(g, "{:.2f}".format(w), Point3(cx, -0.02, cz), 0.3, WHITE if w > 0.15 else GREY)
            wt.hide()
            self.wtext[(k, q)] = wt
            to_scores.append(Sequence(fade_out(c, 0.3), Func(c.hide), Func(s.show), fade_in(s, 0.3)))
            to_weights.append(Sequence(fade_out(s, 0.3), Func(s.hide), Func(wt.show),
                                       Parallel(fade_in(wt, 0.4), LerpScaleInterval(bg, 0.5, 1.0, startScale=0.001))))
        zeros = g.attachNewNode("zeros")
        for q in range(n):
            for k in range(q + 1, n):
                text(zeros, "0.00", Point3(gx0 + (q + 0.5) * cw, 0, gz0 - (k + 0.5) * ch - 0.08), 0.26,
                     (0.25, 0.25, 0.27, 1))
        zeros.flattenStrong()
        zeros.hide()
        to_weights.append(Sequence(Func(zeros.show), fade_in(zeros, 0.4)))
        cx = gx0 + (self.fq + 0.5) * cw
        hl = rect(g, cx - cw / 2 + 0.04, gz0 - n * ch, cx + cw / 2 - 0.04, gz0 + 0.95, YELLOW, 3.0, y=-0.04)
        total = text(g, "sum = 1.00", Point3(cx, 0, gz0 - n * ch - 0.55), 0.3, YELLOW)
        soft = text(g, "softmax: each column -> weights that add up to 1",
                    Point3(gx0 + n * cw / 2, 0, gz0 - n * ch - 1.25), 0.34, GREY)
        for nd in (hl, total, soft):
            nd.hide()
        q = self.fq
        top = sorted(range(q + 1), key=lambda k: -self.W[k][q])[:3]
        names = ", ".join("'{}'".format(word(self.toks[k])) for k in top)
        seq = Sequence(to_scores, Wait(0.6), Func(soft.show), fade_in(soft, 0.4), to_weights, Wait(0.3),
                       self.view(self.qk_center + Vec3(2.0, 0, -0.6), -24, 6, self.qk_dist * 1.05),
                       Func(hl.show), fade_in(hl, 0.5), Func(total.show), fade_in(total, 0.4))
        return ("Each K . Q gives a score (bigger = better match). Every column is then normalized (softmax) into "
                "weights between 0 and 1. A token may only look at tokens BEFORE it, so the lower-left is 0. "
                "Column '{}' looks mostly at {}.".format(word(self.sc["focus"]), names), seq)

    # ================================================================ C: weighted sum -> delta E
    def s_delta(self):
        q = self.fq
        ks = sorted(sorted(range(q + 1), key=lambda k: -self.W[k][q])[:3])
        e = self.board.attachNewNode("delta")
        pos_e = self.qk_center + Vec3(0, 0, -12.5)
        e.setPos(pos_e)
        n = len(ks)
        gap = 3.2
        x_start = -(n * gap) / 2 - 1.0
        vals_sum = [0.0] * 6
        flights, parts = Parallel(), Parallel()
        for j, k in enumerate(ks):
            t = self.toks[k]
            w = self.W[k][q]
            V = sim.embedding("V" + t, 6)
            vals_sum = [a + w * b for a, b in zip(vals_sum, V)]
            x = x_start + j * gap
            grp = e.attachNewNode("term")
            token_box(grp, word(t), self.tok_color[k], (x + 0.5, 0, 3.0), 0.26)
            arrow2d(grp, (x + 0.5, 0, 2.8), (x + 0.5, 0, 2.35), WHITE, 1.4, 0.12)
            vlabel(grp, "W", "V", (x + 0.85, 0, 2.5), 0.2, RED, arrow=False)
            vlabel(grp, "V", k + 1, (x + 0.5, 0, 1.85), 0.36, RED)
            column(grp, x + 0.5, 1.15, V, RED, 0.28, dz=0.38)
            if j < n - 1:
                text(grp, "+", Point3(x + 1.85, 0, -0.2), 0.6, WHITE)
            grp.hide()
            parts.append(Sequence(Wait(0.3 * j), Func(grp.show), fade_in(grp, 0.5)))
            src = self.wtext[(k, q)]
            fly = text(self.board, "{:.2f}".format(w), self.board.getRelativePoint(self.qk, src.getPos()),
                       0.3, YELLOW)
            fly.hide()
            dst = self.board.getRelativePoint(e, Point3(x - 0.55, 0, -0.2))
            flights.append(Sequence(Wait(0.25 * j), Func(fly.show),
                                    Parallel(LerpPosInterval(fly, 1.4, dst, blendType="easeInOut"),
                                             LerpScaleInterval(fly, 1.4, 0.42))))
        xr = x_start + n * gap + 0.4
        res = e.attachNewNode("result")
        text(res, "=", Point3(xr - 0.9, 0, -0.2), 0.6, WHITE)
        vlabel(res, "ΔE", q + 1, (xr + 0.5, 0, 1.85), 0.38, YELLOW)
        column(res, xr + 0.5, 1.15, vals_sum, YELLOW, 0.28, dz=0.38)
        rect(res, xr - 0.2, -2.0, xr + 1.2, 2.5, YELLOW, 2.0)
        res.hide()
        seq = Sequence(self.view(pos_e + Vec3(0, 0, 0.6), 0, 0, 16.5),
                       Wait(0.4), parts, flights, Wait(0.2), Func(res.show), fade_in(res, 0.6))
        mix = " + ".join("{:.2f} x V({})".format(self.W[k][q], word(self.toks[k])) for k in ks)
        return ("Each token also offers a VALUE vector V (made with W_V). The weights of the '{}' column mix "
                "these values:  {}  =  ΔE, a change for '{}'."
                .format(word(self.sc["focus"]), mix, word(self.sc["focus"])), seq)

    # ================================================================ B: the arrow moves
    def s_update(self):
        sp = self.space
        f = self.sc["focus"]
        base = self.space_pos(f)
        target = CENTERS[0] + Vec3(0.2, -0.3, 1.6)
        delta = Arrow(sp, base, target, YELLOW, 3.0)
        new = Arrow(sp, Point3(0, 0, 0), target, WHITE, 3.5)
        lab = text(sp, self.sc["focus_label"].replace(" = ", " =\n"), target + Vec3(-0.2, 0, 0.75), 0.3, WHITE,
                   align=TextNode.ARight)
        lab.setBillboardPointEye()
        lab.hide()
        dl = text(sp, "ΔE", (base + target) * 0.5 + Vec3(0, 0, -0.6), 0.4, YELLOW, Fonts.symbol)
        dl.setBillboardPointEye()
        dl.hide()
        old = self.arrows[f].root
        old.setTransparency(TransparencyAttrib.MAlpha)
        flab = self.space_labels[f]
        flab.setTransparency(TransparencyAttrib.MAlpha)
        c = SEC_B + Vec3(1.2, 0, 1.4)
        seq = Sequence(self.view(c, 15, 22, 14), Wait(1.0),
                       LerpFunc(delta.grow, fromData=0.001, toData=1.0, duration=1.2, blendType="easeOut"),
                       Func(dl.show), fade_in(dl, 0.3),
                       LerpFunc(new.grow, fromData=0.001, toData=1.0, duration=1.2, blendType="easeOut"),
                       Parallel(LerpColorScaleInterval(old, 0.8, (1, 1, 1, 0.25)),
                                LerpColorScaleInterval(flab, 0.8, (1, 1, 1, 0.3))),
                       Func(lab.show), fade_in(lab, 0.5), self.orbit(c, 15, -20, 22, 14, 4.0))
        return ("Add ΔE to the old arrow: '{}' alone was vague, but now its vector lands next to the words "
                "it belongs with. This is what attention DOES - it moves meanings around using context."
                .format(word(f)), seq)

    # ================================================================ A: MLP
    def s_mlp(self):
        c = self.mlp_center
        seq = Sequence(self.view(c + Vec3(0, 0, 0.5), 60, 14, 16), Wait(1.0), self.pulse(0.3), self.pulse(0.25),
                       self.view(c + Vec3(0, -2.5, 0), 48, 18, 34), Wait(1.0))
        return ("Then every vector goes through the MULTILAYER PERCEPTRON: millions of weights (blue = positive, "
                "red = negative) that store facts. Attention + MLP = one layer; real models repeat it 30-100 times.",
                seq)

    # ================================================================ rounds (what the model writes)
    def round(self, i):
        if i not in self.rounds:
            if i < len(self.sc["calls"]):
                name, args = self.sc["calls"][i](self.results)
                txt = tool_call_text(name, args)
                toks = sim.tokenize(txt)
                self.rounds[i] = dict(kind="tool", name=name, args=args, text=txt, toks=toks, k=toks.index(name))
            else:
                txt = self.sc["answer"](self.results)
                self.rounds[i] = dict(kind="answer", text=txt, toks=sim.tokenize(txt), k=None)
        return self.rounds[i]

    def _range(self, i, which):
        r = self.round(i)
        if which == "first":
            return [0]
        if which == "name":
            return [r["k"]]
        if which == "to_name":
            return list(range(1, r["k"]))
        start = r["k"] + 1 if r["kind"] == "tool" else 1
        return list(range(start, len(r["toks"])))

    def _cands(self, i, j):
        r = self.round(i)
        tok = r["toks"][j]
        seed = self.sc["key"] + str(i)
        if r["kind"] == "tool" and j == 0:
            return [("<tool_call>", 0.87), ("Sure", 0.05), ("I", 0.03), ("Let", 0.03), ("To", 0.02)]
        if r["kind"] == "tool" and j == r["k"]:
            return sim.tool_name_candidates(tok, seed)
        if r["kind"] == "answer" and j == 0 and i == 0 and self.sc.get("first"):
            return self.sc["first"]
        prev = r["toks"][j - 1] if j > 0 else None
        return sim.candidates(tok, seed + str(j), prev)

    # ================================================================ P: prediction area
    def _build_predict_area(self):
        P = SEC_P
        a = self.board.attachNewNode("predict")
        self.pa = a
        rect(a, P.x - CTX_W / 2 - 0.3, P.z + 4.6, P.x + CTX_W / 2 + 0.3, P.z + 10.6, DIM, 1.4)
        text(a, "context  (everything the model reads)", Point3(P.x - CTX_W / 2 - 0.3, 0, P.z + 10.9), 0.4,
             GREY, align=TextNode.ALeft)
        arrow2d(a, (P.x, 0, P.z + 4.55), (P.x, 0, P.z + 3.95), WHITE, 1.6, 0.18)
        fill(a, P.x - 6, P.z + 1.6, P.x + 6, P.z + 3.8, (0.75, 0.75, 0.78, 1), 0.12)
        self.tf_box = rect(a, P.x - 6, P.z + 1.6, P.x + 6, P.z + 3.8, WHITE, 1.6)
        text(a, "Transformer", Point3(P.x, 0, P.z + 2.75), 0.62, WHITE)
        text(a, "attention + MLP,  x 32 layers", Point3(P.x, 0, P.z + 1.95), 0.36, GREY)
        arrow2d(a, (P.x, 0, P.z + 1.55), (P.x, 0, P.z + 0.65), WHITE, 1.6, 0.18)
        text(a, "next-token probabilities", Point3(P.x - 3.0, 0, P.z + 0.15), 0.36, GREY, align=TextNode.ALeft)
        rx = P.x + CTX_W / 2 + 1.3
        curve_arrow2d(a, [(P.x + 8.5, 0, P.z - 1.6), (rx, 0, P.z - 1.6), (rx, 0, P.z + 7.6),
                          (rx - 0.9, 0, P.z + 7.6)], YELLOW, 2.0, 0.25)
        lab = text(a, "append the new token, run again", Point3(rx + 0.55, 0, P.z + 3.0), 0.36, YELLOW)
        lab.setR(-90)
        self.bars = []
        for i in range(5):
            z = P.z - 0.55 - i * 0.8
            lt = text(a, "", Point3(P.x - 3.3, 0, z), 0.4, WHITE, align=TextNode.ARight)
            bar = fill(a, 0, -0.08, 1, 0.42, BLUE, 0.85, y=0)
            bar.setPos(P.x - 3.0, 0, z)
            bar.setSx(0.001)
            pt = text(a, "", Point3(P.x - 2.7, 0, z), 0.36, GREY, align=TextNode.ALeft)
            self.bars.append((lt, bar, pt, z))
        self.answer_text = text(a, "", Point3(P.x - CTX_W / 2, 0, P.z - 5.2), 0.5, WHITE, align=TextNode.ALeft,
                                wrap=CTX_W / 0.5)
        self.chips = []
        self.chip_root = a.attachNewNode("chips")
        a.hide()

    def set_bars(self, cands, chosen=None, grow=1.0):
        top = max([p for _, p in cands] or [1.0])
        for i, (lt, bar, pt, z) in enumerate(self.bars):
            if i < len(cands):
                t, p = cands[i]
                lt.node().setText(word(t))
                w = max(0.001, 8.0 * p / top * min(1.0, top / 0.9) * grow)
                bar.setSx(w)
                bar.setColor(*(YELLOW[:3] + (0.9,) if i == chosen else BLUE[:3] + (0.75,)))
                pt.node().setText("{:.0f}%".format(p * 100) if p >= 0.01 else "<1%")
                pt.setX(SEC_P.x - 3.0 + w + 0.25)
            else:
                lt.node().setText("")
                bar.setSx(0.001)
                pt.node().setText("")

    def chip_layout(self):
        P = SEC_P
        x0, x1 = P.x - CTX_W / 2, P.x + CTX_W / 2
        x, z = x0, P.z + 9.9
        out = []
        for c in self.chips:
            if x + c.w > x1:
                x, z = x0, z - 0.72
            out.append((c, Point3(x + c.w / 2, 0, z)))
            x += c.w + 0.1
        return out

    def next_chip_pos(self, w):
        lay = self.chip_layout()
        if not lay:
            return Point3(SEC_P.x - CTX_W / 2 + w / 2, 0, SEC_P.z + 9.9)
        c, p = lay[-1]
        x = p.x + c.w / 2 + 0.1
        if x + w > SEC_P.x + CTX_W / 2:
            return Point3(SEC_P.x - CTX_W / 2 + w / 2, 0, p.z - 0.72)
        return Point3(x + w / 2, 0, p.z)

    def add_chip(self, label, color, seg):
        c = Chip(self.chip_root, label, color, seg)
        c.np.setPos(self.next_chip_pos(c.w))
        self.chips.append(c)
        return c

    def relayout(self, dur=0.5):
        return Parallel(*[LerpPosInterval(c.np, dur, p, blendType="easeInOut") for c, p in self.chip_layout()])

    def collapse(self, seg, label, color):
        group = [c for c in self.chips if c.seg == seg]
        if len(group) < 2:
            return Parallel()
        k = self.chips.index(group[0])
        for c in group:
            self.chips.remove(c)
        block = Chip(self.chip_root, label, color, ("blk",) + seg)
        block.np.setPos(group[0].np.getPos())
        block.np.setScale(0.01)
        self.chips.insert(k, block)
        return Sequence(Parallel(*[Sequence(LerpScaleInterval(c.np, 0.3, 0.01), Func(c.np.hide)) for c in group]),
                        Parallel(LerpScaleInterval(block.np, 0.3, 1.0), self.relayout(0.5)))

    def tf_flash(self):
        return Sequence(LerpColorScaleInterval(self.tf_box, 0.25, (1, 0.95, 0.4, 1), startColorScale=(1, 1, 1, 1)),
                        LerpColorScaleInterval(self.tf_box, 0.35, (1, 1, 1, 1)))

    def emit(self, i, j, fly=0.7):
        """Generated token j flies from its probability bar to the end of the context."""
        r = self.round(i)
        tok = r["toks"][j]
        cands = self._cands(i, j)
        ci = [c for c, _ in cands].index(tok) if tok in dict(cands) else 0
        lt, bar, pt, z = self.bars[ci]
        c = Chip(self.chip_root, word(tok), ORANGE, ("out", i))
        dest = self.next_chip_pos(c.w)
        self.chips.append(c)
        c.np.setPos(SEC_P.x - 3.6, 0, z)
        c.np.hide()
        return Sequence(Func(self.set_bars, cands, ci), Func(c.np.show),
                        LerpPosInterval(c.np, fly, dest, blendType="easeInOut"))

    # ================================================================ P: one predicted token
    def s_predict(self, i, which):
        r = self.round(i)
        seq = Sequence()
        if not hasattr(self, "pa"):
            self._build_predict_area()
            sys_chip = self.add_chip("system + tools (~180 tokens)", GREY, ("sys",))
            sys_chip.np.hide()
            user = []
            for k, t in enumerate(self.toks):
                c = self.add_chip(word(t), self.tok_color[k], ("user",))
                c.np.hide()
                user.append(c)
            seq.append(Func(self.pa.show))
            seq.append(fade_in(self.pa, 0.6))
            seq.append(Parallel(*[Sequence(Wait(0.05 * k), Func(c.np.show), fade_in(c.np, 0.3))
                                  for k, c in enumerate([sys_chip] + user)]))
        if which == "first" and i > 1:
            seq.append(self.collapse(("res", i - 2), "tool result", GREEN))
        j = self._range(i, which)[0]
        cands = self._cands(i, j)
        tok = r["toks"][j]
        ci = [c for c, _ in cands].index(tok)
        grow = LerpFunc(lambda g: self.set_bars(cands, None, g), fromData=0.01, toData=1.0, duration=0.8,
                        blendType="easeOut")
        seq = Sequence(self.view(PRED_VIEW, 0, 0, PRED_D), seq, Wait(0.3),
                       self.tf_flash(), self.tf_flash(), grow, Wait(0.4), Func(self.set_bars, cands, ci),
                       Wait(0.5), self.emit(i, j, 1.0), Wait(0.4))
        p = dict(cands)[tok]
        if r["kind"] == "tool" and which == "first":
            cap = ("Now the model writes its answer, ONE token at a time: it gives every possible next token a "
                   "probability. The winner is a special token <tool_call> ({:.0f}%) - 'deciding to use a tool' "
                   "is just predicting this token!".format(p * 100))
            if i > 0:
                cap = ("The tool result is now in the context, and the model runs again on EVERYTHING. It decides "
                       "it needs another tool: <tool_call> again ({:.0f}%).".format(p * 100))
        elif which == "name":
            cap = ("Which tool? The model writes the tool's NAME. The tool names from the hidden tool list compete "
                   "with each other: '{}' wins with {:.0f}%. Writing the name does not run anything yet."
                   .format(r["name"], p * 100))
        elif self.results:
            cap = ("The tool result is now part of the context, so this time the most likely first token is normal "
                   "text: '{}' ({:.0f}%). The model starts its reply.".format(word(tok), p * 100))
        else:
            cap = ("A greeting needs no tool, so the most likely first token is ordinary text. '{}' wins with only "
                   "{:.0f}% - the model SAMPLES from these probabilities, so next time it might start with "
                   "'Hello'. That is why answers vary.".format(word(tok), p * 100))
        return cap, seq

    def s_fast(self, i, which):
        r = self.round(i)
        js = self._range(i, which)
        par = Parallel()
        gap = 0.13 if len(js) < 35 else 0.09
        for n, j in enumerate(js):
            par.append(Sequence(Wait(n * gap), self.emit(i, j, 0.35)))
        seq = Sequence(self.view(PRED_VIEW, 0, 0, PRED_D), par, Wait(0.8))
        n = len(js)
        if r["kind"] == "tool" and which == "to_name":
            cap = ("Again and again: each new token is appended to the context and the whole model runs once more to "
                   "predict the next one (orange chips). This loop is called AUTOREGRESSIVE generation.")
        elif r["kind"] == "tool":
            cap = ("The model finishes the tool call: the arguments ({} more tokens), then </tool_call>. It is still "
                   "only text - but text in a strict format that a program can read.".format(n))
        else:
            cap = ("The reply is generated token by token ({} more), until the model predicts a special "
                   "'end of text' token.".format(n))
        return cap, seq

    # ================================================================ T: tool area
    def _build_tool_area(self):
        T = SEC_T
        a = self.board.attachNewNode("tools")
        self.ta = a
        rect(a, T.x - 21, T.z - 4.2, T.x - 8.6, T.z + 7.2, WHITE, 1.6)
        text(a, "program  (the harness)", Point3(T.x - 14.8, 0, T.z + 6.3), 0.48, WHITE)
        text(a, "ordinary code around the model", Point3(T.x - 14.8, 0, T.z + 5.7), 0.28, GREY)
        self.prog_text = text(a, "", Point3(T.x - 20.5, 0, T.z + 4.8), 0.3, (0.85, 1, 0.85, 1), Fonts.mono,
                              TextNode.ALeft, wrap=39)
        self.panels = {}
        for name, (px, pz) in TOOL_PANELS.items():
            cx, cz = T.x + px, T.z + pz
            g = a.attachNewNode("panel")
            border = rect(g, cx - PANEL_W / 2, cz - PANEL_H / 2, cx + PANEL_W / 2, cz + PANEL_H / 2, DIM, 1.6)
            hot = rect(g, cx - PANEL_W / 2, cz - PANEL_H / 2, cx + PANEL_W / 2, cz + PANEL_H / 2, YELLOW, 3.0, y=-0.02)
            hot.hide()
            text(g, name + "()", Point3(cx, 0, cz + PANEL_H / 2 - 0.55), 0.38, GREY, Fonts.mono)
            body = text(g, "", Point3(cx - PANEL_W / 2 + 0.3, 0, cz + PANEL_H / 2 - 1.3), 0.3, (0.85, 1, 0.85, 1),
                        Fonts.mono, TextNode.ALeft, wrap=30)
            self.panels[name] = dict(node=g, border=border, hot=hot, body=body, cx=cx, cz=cz)
        a.hide()

    def s_program(self, i):
        r = self.round(i)
        if not hasattr(self, "ta"):
            self._build_tool_area()
        T = SEC_T
        pnl = self.panels[r["name"]]
        args = ", ".join("{}={!r}".format(k, v) for k, v in r["args"].items())
        call = "{}({})".format(r["name"], args)
        if len(call) > 150:
            call = call[:147] + "..."
        lines_ = ["model output:", r["text"] if len(r["text"]) < 170 else r["text"][:167] + "...", "",
                  "found <tool_call> -> stop the model", "read the tool name and arguments", "",
                  "call:", call]
        shown = []

        def add(line):
            shown.append(line)
            self.prog_text.node().setText("\n".join(shown))
        arrow = arrow2d(self.ta, (T.x - 8.5, 0, T.z + 1.5), (pnl["cx"] - PANEL_W / 2 - 0.1, 0, pnl["cz"]),
                        YELLOW, 2.2, 0.28)
        arrow.hide()
        self.call_arrow = arrow
        seq = Sequence(Func(self.prog_text.node().setText, ""), self.chat("Model", r["text"]),
                       self.view(T + Vec3(-5.5, 0, 1.6), 0, 0, 36),
                       Func(self.ta.show), fade_in(self.ta, 0.6), Wait(0.4))
        for ln in lines_:
            seq.append(Func(add, ln))
            seq.append(Wait(0.35))
        for other in self.panels.values():
            seq.append(Func(other["hot"].hide))
        seq.append(Func(pnl["hot"].show))
        seq.append(Func(arrow.show))
        seq.append(fade_in(arrow, 0.5))
        return ("The model itself cannot run code, open files or draw pictures - it only produced text. The program "
                "around it (the 'harness') watches the output. When it sees <tool_call>, it stops the model, reads "
                "the tool name and arguments, and calls the matching Python function.", seq)

    def s_run(self, i):
        r = self.round(i)
        res = tools.run(r["name"], r["args"])
        self.results.append(res)
        pnl = self.panels[r["name"]]
        body = pnl["body"]
        cx, cz = pnl["cx"], pnl["cz"]
        seq = Sequence(self.view(Point3(cx, 0, cz + 0.3), 0, 0, 12.5), Wait(0.6))
        if res["kind"] in ("number", "terminal", "text"):
            full = res["screen"] if res["kind"] != "number" else "{}\n\n= {}".format(r["args"]["expression"],
                                                                                   res["value"])
            if res["kind"] == "number":
                body.setScale(0.55)
                body.node().setWordwrap(16)
            if full.count("\n") > 11:
                full = "\n".join(full.splitlines()[:11] + ["..."])
            seq.append(LerpFunc(lambda v: body.node().setText(full[:int(v)]), fromData=0, toData=len(full),
                                duration=min(2.5, 0.4 + len(full) * 0.02)))
        else:
            img = res["image"]
            h, w = img.shape[:2]
            bh = PANEL_H - 1.4
            bw = bh * w / h
            if bw > PANEL_W - 0.6:
                bw = PANEL_W - 0.6
                bh = bw * h / w
            tex = array_to_texture(img)
            card = image_card(pnl["node"], tex, cx - bw / 2, cz - PANEL_H / 2 + 0.3, cx + bw / 2,
                              cz - PANEL_H / 2 + 0.3 + bh)
            card.hide()
            if res.get("diffusion"):
                seq.append(Func(card.show))
                seq.append(self._diffusion(tex, img, body))
            else:
                seq.append(Func(card.show))
                seq.append(fade_in(card, 1.0))
        seq.append(self.chat("Tool", res["text"]))
        seq.append(Wait(0.8))
        cap = {
            "calculator": "Ordinary Python computes the exact result. LLMs see numbers as tokens (remember 235 + 0?) "
                          "and can make arithmetic mistakes - a calculator tool is far more reliable.",
            "run_terminal": "The program REALLY runs this command on your computer, right now. Only whitelisted, "
                            "read-only commands are allowed - giving an AI a terminal is powerful, so real systems "
                            "ask the user first.",
            "plot_chart": "matplotlib (a normal Python library) draws the chart. The language model never drew a "
                          "single pixel - it only chose the chart type, title and data.",
            "generate_image": "A DIFFERENT AI (an image / diffusion model) starts from pure noise and removes noise "
                              "step by step, guided by the prompt (simulated here). Note: the LLM rewrote your "
                              "request into a more detailed prompt!",
        }[r["name"]]
        return cap, seq

    def _diffusion(self, tex, img, body):
        import numpy as np
        rng = np.random.default_rng(3)
        h, w = img.shape[:2]
        noise = rng.integers(0, 256, size=(h, w, 4)).astype(np.float32)
        noise[..., 3] = 255
        target = img.astype(np.float32)
        steps = 25
        state = {"k": -1}

        def upd(t):
            k = int(t * steps)
            if k == state["k"]:
                return
            state["k"] = k
            a = (k / steps) ** 1.6
            fresh = rng.normal(0, 60 * (1 - a), size=(h, w, 1)).astype(np.float32)
            mix = np.clip(target * a + noise * (1 - a) + fresh, 0, 255).astype(np.uint8)
            tex.setRamImageAs(mix[::-1].tobytes(), "RGBA")
            body.node().setText("denoising step {}/{}".format(k, steps))
        return Sequence(LerpFunc(upd, fromData=0, toData=1, duration=3.0), Func(upd, 1.0),
                        Func(body.node().setText, ""))

    # ================================================================ return
    def s_return(self, i):
        r = self.round(i)
        res = self.results[i]
        self.messages.append({"role": "assistant", "content": r["text"]})
        self.messages.append({"role": "tool", "name": r["name"], "content": res["text"]})
        pnl = self.panels[r["name"]]
        crate = Chip(self.board, "tool result", GREEN, ("tmp",), 0.42)
        start = Point3(pnl["cx"], 0, pnl["cz"] - PANEL_H / 2 - 0.4)
        crate.np.setPos(start)
        crate.np.hide()
        squash = self.collapse(("out", i), "tool call ({} tokens)".format(len(r["toks"])), ORANGE)
        toks = sim.tokenize(res["text"].replace("\n", " "))
        shown = toks[:12]
        new = []
        for t in shown:
            c = self.add_chip(word(t), GREEN, ("res", i))
            c.np.hide()
            new.append(c)
        if len(toks) > 12:
            c = self.add_chip("+{} more".format(len(toks) - 12), GREEN, ("res", i))
            c.np.hide()
            new.append(c)
        squash.start()                 # instant layout change; finish so positions are final
        squash.finish()
        for c, p in self.chip_layout():
            c.np.setPos(p)
        dest = self.board.getRelativePoint(self.chip_root, new[0].np.getPos()) if new else SEC_P
        mid = (start + dest) * 0.5
        ctrl = mid + Vec3(0, 0, 12)

        def fly(t):
            crate.np.setPos(start * (1 - t) * (1 - t) + ctrl * 2 * t * (1 - t) + dest * t * t)
        seq = Sequence(Func(crate.np.show), fade_in(crate.np, 0.3),
                       self.view(mid + Vec3(0, 0, 4), 0, 0, 62),
                       LerpFunc(fly, fromData=0, toData=1, duration=2.6, blendType="easeInOut"),
                       self.view(PRED_VIEW, 0, 0, PRED_D),
                       fade_out(crate.np, 0.3), Func(crate.np.hide),
                       Parallel(*[Sequence(Wait(0.06 * k), Func(c.np.show), fade_in(c.np, 0.3))
                                  for k, c in enumerate(new)]), Wait(0.8))
        return ("The program turns the tool's result into text and appends it to the context as a new message "
                "(green chips). The model gets back a short text - numbers, file names, a file path - not the "
                "picture itself.", seq)

    # ================================================================ answer
    def s_answer(self):
        n = len(self.sc["calls"])
        r = self.round(n)
        self.messages.append({"role": "assistant", "content": r["text"]})
        self.answer_text.node().setText("Answer:  " + r["text"])
        self.answer_text.hide()
        seq = Sequence(self.view(PRED_VIEW + Vec3(0, 0, -0.8), 0, 0, PRED_D + 1), Func(self.set_bars, []),
                       Func(self.answer_text.show), fade_in(self.answer_text, 0.8), self.chat("Model", r["text"]),
                       Wait(0.5))
        loop = ("" if n == 0 else
                "  -> <tool_call> -> the program runs real code -> result back into the context{}\n"
                .format(" (twice here)" if n > 1 else ""))
        return ("Summary:  your words -> tokens -> vectors -> attention + MLP (x32) -> next-token probabilities\n"
                + loop + "  -> final answer.   The LLM itself only ever predicts the next token.", seq)
