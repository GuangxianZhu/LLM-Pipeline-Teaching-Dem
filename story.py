# -*- coding: utf-8 -*-
# Claude Opus 写的
# 中英文切换：Claude 改了这个文件（界面文字改用 i18n.t，Engine 可传 seed）
"""
Story = one question turned into a list of animated steps on one big black board.

Everything the model does comes from the REAL tiny Transformer (tiny/), run by engine.Engine:
the tokens, every matrix inside the Transformer, the next-token probabilities, the tool call it writes,
and the final answer. The tools are real Python too.

Board regions (the camera flies between them):
  A        context and tokens
  x 40-230 inside the Transformer: one fixed left-to-right data flow (see tf_steps.py)
  P        prediction loop: context strip -> Transformer -> next-token probabilities
  T        the program (harness) and the four real tools
"""
from direct.interval.IntervalGlobal import (Func, LerpColorScaleInterval, LerpFunc,
                                            LerpPosInterval, LerpScaleInterval, Parallel,
                                            Sequence, Wait)
from panda3d.core import Point3, TextNode, Vec3

from engine import Engine
from i18n import t
from kit import (BLUE, DIM, GREEN, GREY, ORANGE, WHITE, YELLOW, Fonts, array_to_texture, arrow2d,
                 curve_arrow2d, fade_in, fade_out, fill, fx, image_card, rect, text, text_width)
from tf_steps import TransformerSteps, disp

STAGES = ["Context", "Tokens", "Embedding", "Attention", "Add & Norm", "Feed Fwd", "Output", "Tool", "Answer"]

SEC_A = Point3(0, 0, 0)
SEC_P = Point3(292, 0, 0)       # the prediction loop continues the highway to the right
SEC_T = Point3(345, 0, 0)

TOOL_PANELS = {"calculator": (-1.5, 3.6), "run_terminal": (9.0, 3.6),
               "plot_chart": (-1.5, -2.6), "generate_image": (9.0, -2.6)}
PANEL_W, PANEL_H = 9.6, 5.6
CTX_W = 24.0                     # width of the context strip in the prediction area
PRED_VIEW = SEC_P + Vec3(0.6, 0, 2.6)
PRED_D = 27.0
SYSTEM_TEXT = "You are a helpful assistant. You can call tools."


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
    return disp(t)


class Story(TransformerSteps):
    def __init__(self, app, sc, deep=True, seed=None):
        self.app = app
        self.sc = sc
        self.deep = deep
        self.board = app.board
        self.engine = Engine(sc, seed)
        self.results = [r["result"] for r in self.engine.rounds if r["kind"] == "tool"]
        self.tf_setup()
        self.toks = self.ttoks
        self.steps = self._build_steps()

    # ================================================================ helpers
    def view(self, target, h=0.0, p=0.0, d=24.0):
        return Func(self.app.cam_ctl.go_to, Point3(target), h, p, d)

    def orbit(self, target, h0, h1, p, d, dur):
        return LerpFunc(lambda h: self.app.cam_ctl.go_to(Point3(target), h, p, d), fromData=h0, toData=h1,
                        duration=dur, blendType="easeInOut")

    def chat(self, who, s):
        return Func(self.app.ui.chat, who, s)

    # ================================================================ step list
    def _build_steps(self):
        st = [Step("Context", self.s_context), Step("Tokens", self.s_tokens),
              Step("Embedding", self.s_embed), Step("Embedding", self.s_position)]
        if self.deep:
            st += [Step("Attention", self.s_qkv), Step("Attention", self.s_scores),
                   Step("Attention", self.s_softmax), Step("Attention", self.s_weighted),
                   Step("Attention", self.s_head2), Step("Attention", self.s_concat),
                   Step("Add & Norm", self.s_add1), Step("Feed Fwd", self.s_ffn),
                   Step("Add & Norm", self.s_add2), Step("Attention", self.s_layer2)]
        else:
            st += [Step("Attention", self.s_quick)]
        st += [Step("Output", self.s_output)]
        n = sum(1 for r in self.engine.rounds if r["kind"] == "tool")
        for i in range(n + 1):
            if i < n:
                st += [Step("Output", lambda i=i: self.s_predict(i, "first")),
                       Step("Output", lambda i=i: self.s_fast(i, "to_name")),
                       Step("Output", lambda i=i: self.s_predict(i, "name")),
                       Step("Output", lambda i=i: self.s_fast(i, "rest")),
                       Step("Tool", lambda i=i: self.s_program(i)),
                       Step("Tool", lambda i=i: self.s_run(i)),
                       Step("Tool", lambda i=i: self.s_return(i))]
            else:
                st += [Step("Output", lambda i=i: self.s_predict(i, "first")),
                       Step("Output", lambda i=i: self.s_fast(i, "rest")),
                       Step("Answer", self.s_answer)]
        return st

    # ================================================================ A: context
    def s_context(self):
        a = self.board.attachNewNode("context")
        pr = self.sc["prompt"]
        sc_big = min(0.95, 21.0 / max(1.0, text_width(pr)))
        lines_ = [(t("story.sys"), SYSTEM_TEXT), (t("story.tools"), "calculator    run_terminal    plot_chart    generate_image")]
        hidden = a.attachNewNode("hidden")
        for i, (k, v) in enumerate(lines_):
            text(hidden, k, Point3(-10.5, 0, 7.4 - i * 0.75), 0.4, GREY, align=TextNode.ARight)
            text(hidden, v, Point3(-10.1, 0, 7.4 - i * 0.75), 0.4, GREY, align=TextNode.ALeft)
        text(hidden, t("story.user"), Point3(-10.5, 0, 5.9), 0.4, GREY, align=TextNode.ARight)
        self.hidden = hidden
        self.sentence = text(a, pr, Point3(0, 0, 4.2), sc_big, WHITE)
        self.sentence.hide()
        seq = Sequence(Func(self.app.ui.arch.highlight, "input"), self.view(SEC_A + Vec3(0, 0, 3.0), 0, 0, 24),
                       self.chat("You", pr), fade_in(hidden, 0.8), Wait(0.3), Func(self.sentence.show),
                       fade_in(self.sentence, 1.0))
        return t("story.context.caption"), seq

    # ================================================================ A: tokens
    def s_tokens(self):
        sc = 0.62
        toks = self.ttoks
        widths = [max(1.15, text_width(disp(t)) * sc + 0.45) for t in toks]
        gap = 0.18
        total = sum(widths) + gap * (len(widths) - 1)
        f = min(1.0, 22.0 / total)
        widths = [w * f for w in widths]
        gap *= f
        total = sum(widths) + gap * (len(widths) - 1)
        x = SEC_A.x - total / 2
        xs = []
        for w in widths:
            xs.append(x + w / 2)
            x += w + gap
        z = 2.0
        self.a_tokens, self.a_tok_x = [], xs
        seq = Parallel(Sequence(fade_out(self.sentence, 0.5), Func(self.sentence.hide)),
                       LerpColorScaleInterval(self.hidden, 0.6, (1, 1, 1, 0.45)))
        for i, (tk, w) in enumerate(zip(toks, widths)):
            g = self.board.attachNewNode("tok")
            c, xx = self.tcolor[i], xs[i]
            fill(g, xx - w / 2, z - 0.35 * f, xx + w / 2, z + 0.62 * f, c)
            rect(g, xx - w / 2, z - 0.35 * f, xx + w / 2, z + 0.62 * f, c, 2.2)
            text(g, disp(tk), Point3(xx, 0, z), sc * f, WHITE)
            text(g, str(self.T["ids"][i]), Point3(xx, 0, z - 0.85 * f), 0.32 * f, GREY)
            if i == self.hi:
                rect(g, xx - w / 2 - 0.12, z - 1.2 * f, xx + w / 2 + 0.12, z + 0.78 * f, YELLOW, 2.6, y=-0.03)
            g.hide()
            self.a_tokens.append(g)
            seq.append(Sequence(Wait(0.3 + 0.08 * i), Func(g.show), fade_in(g, 0.4)))
        note = text(self.board, t("story.tokens.note"), Point3(0, 0, -0.6), 0.34, GREY)
        note.hide()
        seq.append(Sequence(Wait(1.6), Func(note.show), fade_in(note, 0.5)))
        parts = [disp(t) for t in toks if not t.startswith(" ") and t.isalpha() and t not in toks[2:3]]
        extra = (t("story.tokens.part", part=parts[0]) if parts else "")
        extra += t("story.tokens.follow")
        return (t("story.tokens.caption", V=self.V) + extra,
                Sequence(Func(self.app.ui.arch.highlight, "input"), self.view(SEC_A + Vec3(0, 0, 2.0), 0, 0, 24), seq))

    # ================================================================ rounds (what the model writes)
    def round(self, i):
        r = self.engine.rounds[i]
        if r["kind"] == "tool" and "k" not in r:
            r["k"] = r["toks"].index(r["name"])
        r.setdefault("k", None)
        return r

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
        """The REAL top-5 next-token probabilities the tiny model gave at this step."""
        return self.round(i)["steps"][j]["top"]

    # ================================================================ P: prediction area
    def _build_predict_area(self):
        P = SEC_P
        a = self.board.attachNewNode("predict")
        self.pa = a
        rect(a, P.x - CTX_W / 2 - 0.3, P.z + 4.6, P.x + CTX_W / 2 + 0.3, P.z + 10.6, DIM, 1.4)
        text(a, t("story.pa.context"), Point3(P.x - CTX_W / 2 - 0.3, 0, P.z + 10.9), 0.4,
             GREY, align=TextNode.ALeft)
        arrow2d(a, (P.x, 0, P.z + 4.55), (P.x, 0, P.z + 3.95), WHITE, 1.6, 0.18)
        fill(a, P.x - 6, P.z + 1.6, P.x + 6, P.z + 3.8, (0.75, 0.75, 0.78, 1), 0.12)
        self.tf_box = rect(a, P.x - 6, P.z + 1.6, P.x + 6, P.z + 3.8, WHITE, 1.6)
        text(a, "Transformer", Point3(P.x, 0, P.z + 2.75), 0.62, WHITE)
        text(a, t("story.pa.model"), Point3(P.x, 0, P.z + 1.95), 0.36, GREY)
        arrow2d(a, (P.x, 0, P.z + 1.55), (P.x, 0, P.z + 0.65), WHITE, 1.6, 0.18)
        text(a, t("story.pa.probs"), Point3(P.x - 3.0, 0, P.z + 0.15), 0.36, GREY, align=TextNode.ALeft)
        rx = P.x + CTX_W / 2 + 1.3
        curve_arrow2d(a, [(P.x + 8.5, 0, P.z - 1.6), (rx, 0, P.z - 1.6), (rx, 0, P.z + 7.6),
                          (rx - 0.9, 0, P.z + 7.6)], YELLOW, 2.0, 0.25)
        lab = text(a, t("story.pa.append"), Point3(rx + 0.55, 0, P.z + 3.0), 0.36, YELLOW)
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
                pt.node().setText("{:.1f}%".format(p * 100) if p >= 0.001 else "<0.1%")
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
            user = []
            for k, tk in enumerate(self.engine.first_context):
                c = self.add_chip(word(tk), self.tcolor[k], ("user",))
                c.np.hide()
                user.append(c)
            seq.append(Func(self.pa.show))
            seq.append(fade_in(self.pa, 0.6))
            seq.append(Parallel(*[Sequence(Wait(0.05 * k), Func(c.np.show), fade_in(c.np, 0.3))
                                  for k, c in enumerate(user)]))
        if which == "first" and i > 1:
            seq.append(self.collapse(("res", i - 2), t("story.chip.tool_result"), GREEN))
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
            cap = t("story.predict.tool_first", p=p * 100)
            if i > 0:
                cap = t("story.predict.tool_again", p=p * 100)
        elif which == "name":
            cap = t("story.predict.name", name=r["name"], p=p * 100)
        elif i > 0:
            cap = t("story.predict.reply", tok=word(tok), p=p * 100)
        else:
            alt = [c for c, _ in cands if c != tok][0]
            cap = t("story.predict.greeting", tok=word(tok), alt=word(alt), p=p * 100)
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
            cap = t("story.fast.to_name")
        elif r["kind"] == "tool":
            cap = t("story.fast.tool_rest", n=n)
        else:
            cap = t("story.fast.reply", n=n)
        return cap, seq

    # ================================================================ T: tool area
    def _build_tool_area(self):
        T = SEC_T
        a = self.board.attachNewNode("tools")
        self.ta = a
        rect(a, T.x - 21, T.z - 4.2, T.x - 8.6, T.z + 7.2, WHITE, 1.6)
        text(a, t("story.ta.title"), Point3(T.x - 14.8, 0, T.z + 6.3), 0.48, WHITE)
        text(a, t("story.ta.sub"), Point3(T.x - 14.8, 0, T.z + 5.7), 0.28, GREY)
        self.prog_text = text(a, "", Point3(T.x - 20.5, 0, T.z + 4.8), 0.3, (0.85, 1, 0.85, 1), Fonts.mono,
                              TextNode.ALeft, wrap=39)
        self.prog_font = Fonts.mono
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
        lines_ = [t("story.prog.output"), r["text"] if len(r["text"]) < 170 else r["text"][:167] + "...", "",
                  t("story.prog.found"), t("story.prog.read"), "",
                  t("story.prog.call"), call]
        shown = []

        def add(line):
            shown.append(line)
            self.prog_text.node().setText(fx("\n".join(shown), self.prog_font))
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
        return t("story.prog.caption"), seq

    def s_run(self, i):
        r = self.round(i)
        res = r["result"]           # the engine already ran the real tool with the model's arguments
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
        return t("story.run." + r["name"]), seq

    def _diffusion(self, tex, img, body):
        import numpy as np
        rng = np.random.default_rng(3)
        h, w = img.shape[:2]
        noise = rng.integers(0, 256, size=(h, w, 4)).astype(np.float32)
        noise[..., 3] = 255
        target = img.astype(np.float32)
        steps = 25
        state = {"k": -1}

        def upd(frac):
            k = int(frac * steps)
            if k == state["k"]:
                return
            state["k"] = k
            a = (k / steps) ** 1.6
            fresh = rng.normal(0, 60 * (1 - a), size=(h, w, 1)).astype(np.float32)
            mix = np.clip(target * a + noise * (1 - a) + fresh, 0, 255).astype(np.uint8)
            tex.setRamImageAs(mix[::-1].tobytes(), "RGBA")
            body.node().setText(fx(t("story.run.denoise", k=k, steps=steps), Fonts.mono))
        return Sequence(LerpFunc(upd, fromData=0, toData=1, duration=3.0), Func(upd, 1.0),
                        Func(body.node().setText, ""))

    # ================================================================ return
    def s_return(self, i):
        r = self.round(i)
        pnl = self.panels[r["name"]]
        crate = Chip(self.board, t("story.chip.tool_result"), GREEN, ("tmp",), 0.42)
        start = Point3(pnl["cx"], 0, pnl["cz"] - PANEL_H / 2 - 0.4)
        crate.np.setPos(start)
        crate.np.hide()
        squash = self.collapse(("out", i), t("story.chip.tool_call", n=len(r["toks"])), ORANGE)
        toks = ["<tool>"] + r["result_toks"]
        new = []
        for tk in toks[:13]:
            c = self.add_chip(word(tk), GREY if tk.startswith("<") else GREEN, ("res", i))
            c.np.hide()
            new.append(c)
        if len(toks) > 13:
            c = self.add_chip(t("story.chip.more", n=len(toks) - 13), GREEN, ("res", i))
            c.np.hide()
            new.append(c)
        c = self.add_chip("<ai>", GREY, ("ai", i))
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
        return t("story.return.caption"), seq

    # ================================================================ answer
    def s_answer(self):
        n = len(self.engine.rounds) - 1
        r = self.round(n)
        self.answer_text.node().setText(fx(t("story.answer.label", text=r["text"]), Fonts.serif))
        self.answer_text.hide()
        seq = Sequence(self.view(PRED_VIEW + Vec3(0, 0, -0.8), 0, 0, PRED_D + 1), Func(self.set_bars, []),
                       Func(self.answer_text.show), fade_in(self.answer_text, 0.8), self.chat("Model", r["text"]),
                       Wait(0.5))
        loop = ("" if n == 0 else
                t("story.answer.loop", twice=t("story.answer.twice") if n > 1 else ""))
        return (t("story.answer.head") + loop + t("story.answer.tail"), seq)
