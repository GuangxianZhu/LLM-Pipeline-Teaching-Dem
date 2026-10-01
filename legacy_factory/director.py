# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
The Director turns one scenario into a list of Steps and plays them.

Each Step = which stage it belongs to, a camera view, an explanation text,
a "data" text (what the computer really has at that moment), and a function
that builds the animation (a Panda3D Interval).
"""
import json
import random

from direct.interval.IntervalGlobal import (Func, LerpFunc, LerpPosInterval,
                                            LerpScaleInterval, Parallel, Sequence, Wait)
from panda3d.core import Point3, Texture, Vec3

import scene as S
import sim
import tools
from scenarios import SYSTEM_PROMPT, tool_call_text

STAGES = ["Input", "Tokenize", "Embed", "Transformer", "Predict", "Tool call",
          "Run tool", "Return", "Answer"]


class Step:
    def __init__(self, stage, view, title, text, data, build):
        self.stage, self.view = stage, view
        self.title, self.text, self.data, self.build = title, text, data, build


def _v(x):
    return x() if callable(x) else x


def arc(np, p0, p1, height, dur, blend="easeInOut"):
    p0, p1 = Point3(p0), Point3(p1)
    ctrl = (p0 + p1) * 0.5 + Vec3(0, 0, height)

    def f(t):
        np.setPos(p0 * (1 - t) * (1 - t) + ctrl * 2 * t * (1 - t) + p1 * t * t)
    return LerpFunc(f, fromData=0.0, toData=1.0, duration=dur, blendType=blend)


def pop(np, dur=0.25, scale=1.0):
    return Sequence(Func(np.show), LerpScaleInterval(np, dur, scale, startScale=0.01, blendType="easeOut"))


def array_to_texture(arr):
    h, w = arr.shape[:2]
    tex = Texture("img")
    tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.setRamImageAs(arr[::-1].tobytes(), "RGBA")
    return tex


class Director:
    def __init__(self, app):
        self.app = app
        self.f = app.factory

    # ================================================================ setup
    def load(self, sc):
        self.sc = sc
        self.f.clear()
        self.results = []
        self.rounds = {}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "tools", "content": [t.split(":")[0] for t in tools.TOOL_SPECS]},
                         {"role": "user", "content": sc["prompt"]}]
        self.dock_lines = []
        self.batch = None
        self.user_toks = []
        self.row = []              # tokens in the context row above the transformer
        self.gen_block = None
        self.gen_count = 0
        self.tray = {}             # slot -> Tok
        self.tray_text = ""
        self.out_count = 0
        self.crate = None
        self.last_att = []
        self.prev_result, self.prev_result_n = [], 0
        self.steps = self._build_steps()
        return self.steps

    def _build_steps(self):
        sc = self.sc
        st = [self._input_step(), self._tokenize_step(), self._embed_step(), self._transformer_step()]
        n = len(sc["calls"])
        for i in range(n + 1):
            if i < n:
                st += [self._sample_step(i, "first"), self._fast_step(i, "to_name"),
                       self._sample_step(i, "name"), self._fast_step(i, "rest"),
                       self._router_step(i), self._run_step(i), self._return_step(i),
                       self._reprocess_step(i)]
            else:
                st += [self._sample_step(i, "first"), self._fast_step(i, "rest"), self._done_step()]
        return st

    # ================================================================ rounds
    def round(self, i):
        if i not in self.rounds:
            if i < len(self.sc["calls"]):
                name, args = self.sc["calls"][i](self.results)
                text = tool_call_text(name, args)
                toks = sim.tokenize(text)
                k = toks.index(name)
                self.rounds[i] = dict(kind="tool", name=name, args=args, text=text, toks=toks, k=k)
            else:
                text = self.sc["answer"](self.results)
                self.rounds[i] = dict(kind="answer", text=text, toks=sim.tokenize(text), k=None)
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
        if r["kind"] == "answer" and j == 0 and self.sc.get("first") and i == 0:
            return self.sc["first"]
        prev = r["toks"][j - 1] if j > 0 else None
        return sim.candidates(tok, seed + str(j), prev)

    # ================================================================ layout helpers
    def grid_pos(self, idx, cx):
        """Position of user token idx inside the belt batch (lines of 5)."""
        line, col = divmod(idx, 5)
        return Point3(cx + (col - 2) * 0.8, 0.4 - line * 0.85, S.BELT_Z)

    def row_layout(self):
        n = len(self.row)
        widths = [t.width for t in self.row]
        gap = 0.2
        total = sum(widths) + gap * (n - 1)
        limit = 15.0
        s = min(1.0, limit / total) if total else 1.0
        x = S.TF - total * s / 2
        out = []
        for t in self.row:
            w = t.width * s
            out.append((t, Point3(x + w / 2, 0, S.ROW_Z), s))
            x += w + gap * s
        return out

    def relayout(self, dur=0.6):
        return Parallel(*[Parallel(LerpPosInterval(t.np, dur, p, blendType="easeInOut"),
                                   LerpScaleInterval(t.np, dur, s))
                          for t, p, s in self.row_layout()])

    def flash_plates(self, each=0.22):
        seq = Sequence()
        for p in self.f.plates:
            seq.append(LerpFunc(lambda v, p=p: p.setColorScale(1 + v, 1 + v, 1 + v, 1 + v),
                                fromData=0, toData=1.4, duration=each))
            seq.append(LerpFunc(lambda v, p=p: p.setColorScale(1 + v, 1 + v, 1 + v, 1 + v),
                                fromData=1.4, toData=0, duration=each))
        return seq

    def collapse(self, toks, label):
        """Replace a run of tokens in the row by ONE block (instant layout, shrink animation)."""
        toks = [t for t in toks if t in self.row]
        if len(toks) < 2:
            return Parallel()
        n = self.prev_result_n if toks and toks[0].kind == "result" else len(toks)
        k = self.row.index(toks[0])
        block = self.f.token("blk", toks[0].kind, width=1.5, display=label.format(n))
        block.np.setPos(toks[len(toks) // 2].np.getPos())
        block.np.setScale(0.01)
        for t in toks:
            self.row.remove(t)
        self.row.insert(k, block)
        return Parallel(*[Sequence(LerpScaleInterval(t.np, 0.4, 0.01), Func(t.np.hide)) for t in toks] +
                        [Sequence(Wait(0.3), LerpScaleInterval(block.np, 0.3, 1.0))])

    def fade_row(self, alpha, dur=0.6):
        from direct.interval.IntervalGlobal import LerpColorScaleInterval
        return Parallel(*[LerpColorScaleInterval(t.np, dur, (1, 1, 1, alpha)) for t in self.row])

    def clear_beams(self):
        for b in self.f.dyn.findAllMatches("beams"):
            b.removeNode()

    def draw_beams(self, src_tok, focus, seed, top=6):
        labels = [sim.show(t.text) for t in self.row if t is not src_tok]
        others = [t for t in self.row if t is not src_tok]
        w = sim.attention(labels, focus, seed)
        ranked = sorted(zip(others, w, labels), key=lambda x: -x[1])[:top]
        src = src_tok.np.getPos() + Vec3(0, 0, 0.35)
        self.f.beams(src, [(t.np.getPos() + Vec3(0, 0, 0.35), wt) for t, wt, _ in ranked])
        self.last_att = [(lab, wt) for _, wt, lab in ranked]

    def att_text(self, who):
        if not self.last_att:
            return "(computing attention ...)"
        lines = ["attention from {} ->".format(who)]
        for lab, w in self.last_att:
            lines.append("  {:<14} {:.2f}  {}".format(lab[:14], w, "#" * int(round(w * 20))))
        return "\n".join(lines)

    def set_dock(self, line=None):
        if line:
            self.dock_lines.append(line)
        self.f.dock_screen.set_text("\n".join(self.dock_lines[-9:]))

    def chat(self, who, text):
        self.app.ui.chat(who, text)

    # ================================================================ 1 input
    def _input_step(self):
        def build():
            sc = self.sc
            self.pallet = self.f.token("SYS", "system", width=3.6, display="SYSTEM PROMPT + TOOL LIST  (~180 tokens)")
            self.pallet.np.setPos(S.DOCK, 1.2, S.BELT_Z)
            self.pallet.np.hide()
            plank_w = min(4.6, 0.6 + 0.16 * len(sc["prompt"]))
            self.plank = self.f.token(sc["prompt"], "user", width=plank_w, display=sc["prompt"])
            self.plank.np.setPos(S.DOCK, -0.4, S.BELT_Z)
            self.plank.lab.setScale(min(0.28, (plank_w + 0.6) / max(self.plank.tn.getWidth(), 1)))
            self.plank.np.hide()
            return Sequence(
                Func(self.chat, "You", sc["prompt"]),
                Func(self.set_dock, "[system] " + SYSTEM_PROMPT), Wait(0.5),
                Func(self.set_dock, "[tools]  " + ", ".join(tools.TOOLS)), Wait(0.5),
                pop(self.pallet.np, 0.4), Wait(0.3),
                Func(self.set_dock, "[user]   " + sc["prompt"]),
                pop(self.plank.np, 0.4), Wait(0.6))
        return Step("Input", "dock", "Your message becomes a CONTEXT",
                    "Your sentence is not sent to the model alone. The program wraps it into one long "
                    "text called the context:\n\n"
                    "  1. a hidden SYSTEM PROMPT (rules for the model)\n"
                    "  2. the LIST OF TOOLS it is allowed to use\n"
                    "  3. your message\n\n"
                    "Grey block = system prompt + tools. Blue block = your words.",
                    lambda: json.dumps(self.messages, indent=1)[:900], build)

    # ================================================================ 2 tokenize
    def _tokenize_step(self):
        def build():
            sc = self.sc
            self.batch = self.f.dyn.attachNewNode("batch")
            self.batch.setPos(S.DOCK, 0, 0)
            for t in (self.pallet, self.plank):
                t.np.wrtReparentTo(self.batch)
            self.user_toks = []
            pops = Sequence()
            for i, tk in enumerate(sim.tokenize(sc["prompt"])):
                t = self.f.token(tk, "user")
                t.np.reparentTo(self.batch)
                t.np.setPos(self.grid_pos(i, 0))
                t.np.hide()
                self.user_toks.append(t)
                pops.append(pop(t.np, 0.12))
            blade = self.f.blade
            return Sequence(
                LerpPosInterval(self.batch, 1.4, Point3(S.TOK, 0, 0), blendType="easeInOut"),
                LerpPosInterval(blade, 0.18, Point3(S.TOK, 0, 0.9)),
                Func(self.plank.np.hide),
                Parallel(LerpPosInterval(blade, 0.3, Point3(S.TOK, 0, 2.2)), pops),
                Wait(0.5))

        def data():
            return "token        id\n" + "\n".join(
                "{:<12} {}".format(sim.show(t.text), sim.token_id(t.text)) for t in self.user_toks)
        return Step("Tokenize", "tokenizer", "Text is cut into TOKENS",
                    "The model cannot read letters. A tokenizer cuts the text into pieces called "
                    "tokens (whole words or parts of words) and gives each piece a number (its ID).\n\n"
                    "'_' means the token starts with a space. Long or rare words are split "
                    "(e.g. temper + ature), and numbers can be split too (235 + 0)!",
                    data, build)

    # ================================================================ 3 embed
    def _embed_step(self):
        def build():
            seq = Sequence(LerpPosInterval(self.batch, 1.2, Point3(S.EMB, 0, 0), blendType="easeInOut"))
            par = Sequence()
            for t in [self.pallet] + self.user_toks:
                def add(t=t):
                    v = t.add_vector(sim.embedding(t.text) if t is not self.pallet else
                                     [round(random.Random(i).uniform(-1, 1), 2) for i in range(16)])
                    v.setScale(1, 1, 0.01)
                    LerpScaleInterval(v, 0.25, 1, startScale=(1, 1, 0.01)).start(0.0, -1.0, self.app.speed)
                par.append(Func(add))
                par.append(Wait(0.11))
            seq.append(par)
            seq.append(Wait(0.5))
            return seq

        def data():
            return "token      vector (8 of 4096 numbers)\n" + "\n".join(
                "{:<10} [{}]".format(sim.show(t.text)[:10], ", ".join("{:+.1f}".format(x) for x in sim.embedding(t.text)))
                for t in self.user_toks)
        return Step("Embed", "embed", "Each token becomes a VECTOR (a list of numbers)",
                    "Each token ID is looked up in a huge table and replaced by a list of numbers "
                    "called a vector (embedding). The coloured bars on each block show 8 of them: "
                    "red = positive, blue = negative.\n\n"
                    "Real models use thousands of numbers per token. Words with similar meaning "
                    "get similar vectors - from now on the model only computes with numbers.",
                    data, build)

    # ================================================================ 4 transformer
    def _transformer_step(self):
        def build():
            self.row = [self.pallet] + self.user_toks

            def detach():
                for t in self.row:
                    t.np.wrtReparentTo(self.f.dyn)
            lifts = Parallel()
            for k, (t, p, s) in enumerate(self.row_layout()):
                lifts.append(Sequence(Wait(0.04 * k),
                                      LerpPosInterval(t.np, 1.6, p, blendType="easeInOut"),
                                      ))
                lifts.append(Sequence(Wait(0.04 * k + 1.6), LerpScaleInterval(t.np, 0.2, s)))
            last = self.user_toks[-1]
            return Sequence(
                LerpPosInterval(self.batch, 1.0, Point3(S.TF, 0, 0), blendType="easeInOut"),
                Func(detach),
                Parallel(lifts, Sequence(Wait(0.3), self.flash_plates())),
                Func(self.draw_beams, last, self.sc["keywords"], "tf"),
                Wait(0.8))
        return Step("Transformer", "transformer", "Transformer layers + ATTENTION",
                    "The vectors pass through many stacked layers (4 drawn here, real models have "
                    "30-100+). In every layer each token 'looks at' all earlier tokens and mixes in "
                    "what is relevant. This is called ATTENTION.\n\n"
                    "The yellow beams show what the last token pays attention to. Thick beam = high "
                    "weight. Notice which words matter most for this question!",
                    lambda: self.att_text(sim.show(self.user_toks[-1].text)), build)

    # ================================================================ 5 predict
    def _ensure_gen_block(self, i):
        if self.gen_block is None:
            r = self.round(i)
            self.gen_block = self.f.token("+0", "block", width=1.3, display="output\n+0")
            self.gen_block.np.setPos(S.TF + 14, 0, S.ROW_Z)
            self.gen_block.kind_round = r["kind"]
            self.row.append(self.gen_block)
            self.gen_count = 0
            self.relayout(0.5).start(0.0, -1.0, self.app.speed)
            self.tray_text = ""
            self.f.tray_screen.set_text("")
            for t in self.tray.values():
                t.remove()
            self.tray = {}

    def tray_pos(self, slot):
        row, col = divmod(slot, 4)
        return Point3(S.TRAY - 1.2 + col * 0.8, -0.85 + row * 0.85, 0.75)

    def emit(self, i, j, fly=0.7):
        """Spawn generated token j at the chosen bar and fly it to the tray."""
        r = self.round(i)
        tok = r["toks"][j]
        slot = self.out_count % 12
        self.out_count += 1
        if slot in self.tray:
            self.tray[slot].remove()
        t = self.f.token(tok, "gen")
        self.tray[slot] = t
        cands = self._cands(i, j)
        ci = [c for c, _ in cands].index(tok) if tok in [c for c, _ in cands] else 0
        t.np.setPos(self.f.bar_top(ci))
        t.np.hide()
        dest = self.tray_pos(slot)

        def land():
            self.tray_text += tok
            self.f.tray_screen.set_text(self.tray_text)
            self.gen_count += 1
            if self.gen_block:
                self.gen_block.set_label("output\n+{}".format(self.gen_count))
        return Sequence(Func(t.np.show), arc(t.np, t.np.getPos(), dest, 1.6, fly), Func(land))

    def _sample_step(self, i, which):
        def build():
            self._ensure_gen_block(i)
            j = self._range(i, which)[0]
            cands = self._cands(i, j)
            tok = self.round(i)["toks"][j]
            ci = [c for c, _ in cands].index(tok)
            self.clear_beams()
            grow = LerpFunc(lambda g: self.f.set_bars(cands, None, g), fromData=0.01, toData=1.0,
                            duration=0.7, blendType="easeOut")
            return Sequence(
                Wait(0.6),
                Func(self.draw_beams, self.gen_block, self.sc["keywords"] + [self.round(i).get("name", "x")],
                     "s{}{}".format(i, j), 5),
                Wait(0.6), grow, Wait(0.4),
                Func(self.f.set_bars, cands, ci), Wait(0.5),
                self.emit(i, j, 1.0), Wait(0.4))

        def title():
            r = self.round(i)
            if r["kind"] == "tool" and which == "first":
                return "The model DECIDES to use a tool"
            if which == "name":
                return "Which tool? It writes the tool's NAME"
            return "The model starts its reply"

        def text():
            r = self.round(i)
            j = self._range(i, which)[0]
            cands = self._cands(i, j)
            p = dict(cands)[r["toks"][j]]
            if r["kind"] == "tool" and which == "first":
                return ("The model predicts only ONE token at a time. It gives every possible next token "
                        "a probability (top 5 shown as bars).\n\n"
                        "The winner is a special token <tool_call> ({:.0f}%). During training the model "
                        "learned that questions like this are best answered with a tool. 'Deciding to "
                        "use a tool' = predicting this one token!".format(p * 100))
            if which == "name":
                return ("Next comes the tool name. The candidates are the tool names from the hidden "
                        "tool list, so they compete with each other: '{}' wins with {:.0f}%.\n\n"
                        "The model only WRITES the name as text - it cannot run anything."
                        .format(r["name"], p * 100))
            if self.results:
                return ("The tool result is now part of the context, so this time the most likely "
                        "token is normal text, not <tool_call>.\n\nFirst token: '{}' ({:.0f}%)."
                        .format(sim.show(r["toks"][j]), p * 100))
            return ("A greeting needs no tool, so the most likely first token is ordinary text.\n\n"
                    "'{}' wins with only {:.0f}%. The model SAMPLES from these probabilities, so the "
                    "same question can start with 'Hello' next time. That is why answers vary!"
                    .format(sim.show(r["toks"][j]), p * 100))

        def data():
            j = self._range(i, which)[0]
            lines = ["next-token candidates:"]
            for c, p in self._cands(i, j):
                lines.append("  {:<14} {:5.1f}%  {}".format(sim.show(c)[:14], p * 100, "#" * int(round(p * 24))))
            lines.append("")
            lines.append(self.att_text("[next]"))
            return "\n".join(lines)
        return Step("Predict", "sampler", title, text, data, build)

    def _fast_step(self, i, which):
        def build():
            js = self._range(i, which)
            if not js:
                return Sequence(Wait(0.1))
            self.clear_beams()
            par = Parallel()
            gap = 0.16 if len(js) < 30 else 0.11
            for n, j in enumerate(js):
                cands = self._cands(i, j)
                tok = self.round(i)["toks"][j]
                ci = [c for c, _ in cands].index(tok) if tok in dict(cands) else 0
                par.append(Sequence(Wait(n * gap), Func(self.f.set_bars, cands, ci),
                                    Func(lambda i=i, j=j: self.emit(i, j, 0.45).start(0.0, -1.0, self.app.speed))))
            return Sequence(par, Wait(0.9))

        def text():
            r = self.round(i)
            n = len(self._range(i, which))
            if r["kind"] == "tool" and which == "to_name":
                return ("Again and again: each new token is added to the context and the model runs once "
                        "more to predict the next one. This loop is called AUTOREGRESSIVE generation.\n\n"
                        "{} tokens later the output already looks like JSON: {{\"name\": \"..."
                        .format(n))
            if r["kind"] == "tool":
                return ("The model finishes the JSON with the ARGUMENTS for the tool and closes it with "
                        "</tool_call>.\n\nLook at the OUTPUT screen: it is just text that follows a strict "
                        "format the program can read. ({} more tokens)".format(n))
            return ("The reply is generated token by token ({} more) until the model predicts a special "
                    "'end of text' token. Each token goes through ALL the layers again.".format(n))

        def data():
            return "output so far:\n" + self.round(i)["text"]
        return Step("Predict", "sampler", "Token by token ...", text, data, build)

    # ================================================================ 6 router
    def _router_step(self, i):
        def build():
            r = self.round(i)
            self.chat("Model", r["text"])
            self.gen_block.set_label("tool call\n{} tok".format(len(r["toks"])))
            self.gen_block = None
            crate = self.f.token("crate", "crate", width=1.3, display="{}(...)".format(r["name"]))
            crate.body.setScale(1, 1.3, 2.4)
            crate.lab.setZ(1.2)
            crate.np.setPos(S.TRAY, 0, 0.75)
            crate.np.hide()
            self.crate = crate
            bx = S.BAYS[r["name"]]
            shrink = Parallel(*[Sequence(LerpPosInterval(t.np, 0.5, Point3(S.TRAY, 0, 0.9)),
                                         Func(t.np.hide)) for t in self.tray.values()])
            lamp = self.f.router_lamp
            for t in self.row:
                t.np.setTransparency(1)
            return Sequence(
                Func(lamp.setColor, 0.8, 0.4, 1, 1), Func(self.clear_beams),
                Parallel(shrink, self.fade_row(0.0)), pop(crate.np, 0.35),
                LerpPosInterval(crate.np, 0.8, Point3(S.LANE_X, 0, S.BELT_Z)),
                LerpPosInterval(crate.np, 1.6, Point3(S.LANE_X, S.LANE_Y, S.BELT_Z)),
                LerpPosInterval(crate.np, 1.0 + abs(S.LANE_X - bx) * 0.12, Point3(bx, S.LANE_Y, S.BELT_Z)),
                LerpPosInterval(crate.np, 0.5, Point3(bx, S.BAY_Y - 0.6, 0.25)),
                Func(lamp.setColor, 0.3, 0.3, 0.3, 1), Wait(0.3))

        def data():
            r = self.round(i)
            return "parsed by the program:\n" + json.dumps({"name": r["name"], "arguments": r["args"]}, indent=1)[:700]
        return Step("Tool call", "router", "The PROGRAM reads the tool call",
                    "The model itself cannot run code, open files or draw pictures - it only produced "
                    "text.\n\nThe surrounding program (often called the 'harness' or 'agent') watches the "
                    "output. When it sees <tool_call> ... </tool_call> it stops the model, parses the "
                    "JSON and sends it to the matching Python function.",
                    data, build)

    # ================================================================ 7 run tool
    def _run_step(self, i):
        def build():
            r = self.round(i)
            res = tools.run(r["name"], r["args"])
            self.results.append(res)
            self.res = res
            screen = self.f.bay_screens[r["name"]]
            lamp = self.f.bay_lamps[r["name"]]
            seq = Sequence(Func(lamp.setColor, 0.2, 1, 0.35, 1),
                           LerpScaleInterval(self.crate.np, 0.4, 0.01), Func(self.crate.np.hide))
            if res["kind"] in ("number", "terminal", "text"):
                full = res["screen"]
                seq.append(LerpFunc(lambda v: screen.set_text(full[:int(v)]), fromData=0,
                                    toData=len(full), duration=min(2.5, 0.4 + len(full) * 0.02)))
            elif res.get("diffusion"):
                seq.append(self._diffusion(screen, res["image"]))
            else:
                tex = array_to_texture(res["image"])
                seq.append(Func(screen.set_image, tex))
                seq.append(LerpFunc(lambda a: screen.img.setColor(1, 1, 1, a), fromData=0, toData=1, duration=1.0))
                seq.append(Func(self.app.ui.show_result, tex, res["image"].shape))
            seq.append(Func(self.chat, "Tool", res["text"]))
            seq.append(Wait(1.0))
            return seq

        def text():
            name = self.round(i)["name"]
            return {
                "calculator": "A normal Python function computes the exact result.\n\nLLMs see numbers as "
                              "tokens (remember 235 + 0?) and can make arithmetic mistakes. Using a "
                              "calculator tool is much more reliable.",
                "run_terminal": "The program REALLY runs this command on your computer, right now.\n\n"
                                "Only whitelisted, read-only commands are allowed here. Giving an AI a "
                                "terminal is powerful - real systems ask the user for permission first.",
                "plot_chart": "matplotlib (a normal Python library) draws the chart from the data.\n\n"
                              "The language model never drew a single pixel - it only chose the chart "
                              "type, title and data.",
                "generate_image": "A DIFFERENT AI (an image / diffusion model) starts from pure noise and "
                                  "removes noise step by step, guided by the prompt text (simulated "
                                  "here).\n\nNote: the LLM rewrote your request into a more detailed prompt!",
            }[name]

        def data():
            return "Python call:\n  {}\n\nresult:\n{}".format(self.res["python"], self.res["text"][:600])
        return Step("Run tool", lambda: S.bay_view(self.round(i)["name"]), lambda: "Real code runs: {}()".format(self.round(i)["name"]),
                    text, data, build)

    def _diffusion(self, screen, img):
        import numpy as np
        rng = np.random.default_rng(3)
        h, w = img.shape[:2]
        noise = rng.integers(0, 256, size=(h, w, 4)).astype(np.float32)
        noise[..., 3] = 255
        target = img.astype(np.float32)
        tex = array_to_texture(img)
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
            screen.set_text("denoising step {}/{}".format(k, steps))
        return Sequence(Func(screen.set_image, tex),
                        LerpFunc(upd, fromData=0, toData=1, duration=3.0),
                        Func(upd, 1.0),
                        Func(screen.set_text, ""),
                        Func(self.app.ui.show_result, tex, img.shape))

    # ================================================================ 8 return
    def _return_step(self, i):
        def build():
            r = self.round(i)
            res = self.results[i]
            self.messages.append({"role": "assistant", "content": r["text"]})
            self.messages.append({"role": "tool", "name": r["name"], "content": res["text"]})
            bx = S.BAYS[r["name"]]
            box = self.f.token("result", "result", width=1.3, display="tool result")
            box.body.setScale(1, 1.3, 2.0)
            box.lab.setZ(1.0)
            start = Point3(bx, S.BAY_Y - 0.6, 0.25)
            box.np.setPos(start)
            box.np.hide()
            self.result_box = box
            short = res["text"].replace("\n", " ")
            return Sequence(pop(box.np, 0.3), Wait(0.2),
                            arc(box.np, start, Point3(S.DOCK, -1.2, S.BELT_Z), 9, 2.6),
                            Func(self.set_dock, "[assistant] " + r["text"][:60] + "..."),
                            Func(self.set_dock, "[tool] " + short[:120]),
                            Wait(0.8))

        def data():
            return json.dumps(self.messages[-1], indent=1)[:800]
        return Step("Return", "return", "The result goes BACK into the context",
                    "The program turns the tool's result into text and appends it to the conversation "
                    "as a new 'tool' message (see the CONTEXT screen).\n\n"
                    "In this demo the model gets a short text back (numbers, file names, a file path) - "
                    "not the picture itself.",
                    data, build)

    def _reprocess_step(self, i):
        def build():
            res = self.results[i]
            self.result_box.np.hide()
            toks = sim.tokenize("Tool result: " + res["text"].replace("\n", " "))
            shown = toks[:5]
            collapse = Parallel()
            if i >= 1:   # keep the row readable: squash older parts into single blocks
                collapse = self.collapse(self.user_toks, "your message\n{} tok")
                collapse.append(self.collapse(self.prev_result, "tool result\n{} tok"))
            new = []
            for tk in shown:
                t = self.f.token(tk, "result")
                t.np.setPos(S.DOCK, -1.2, S.BELT_Z)
                new.append(t)
            if len(toks) > 5:
                more = self.f.token("more", "result", width=1.2, display="+{} more".format(len(toks) - 5))
                more.np.setPos(S.DOCK, -1.2, S.BELT_Z)
                new.append(more)
            self.reproc_toks = toks
            self.prev_result = list(new)
            self.prev_result_n = len(toks)
            self.row.extend(new)
            lay = {id(t): (p, s) for t, p, s in self.row_layout()}
            flights = Parallel()
            for k, t in enumerate(new):
                p, s = lay[id(t)]
                t.add_vector(sim.embedding(t.text))
                flights.append(Sequence(Wait(0.12 * k), arc(t.np, t.np.getPos(), p, 5, 1.8),
                                        LerpScaleInterval(t.np, 0.2, s)))
            return Sequence(Func(self.clear_beams), self.fade_row(1.0, 0.4), collapse,
                            Parallel(self.relayout(0.8), flights, Sequence(Wait(1.2), self.flash_plates())),
                            Func(self.draw_beams, new[-1], self.sc["keywords"] + toks, "r{}".format(i)),
                            Wait(0.8))

        def data():
            return "new tokens added to context:\n" + " ".join(sim.show(t) for t in self.reproc_toks)[:700]
        return Step("Transformer", "transformer", "The model reads EVERYTHING again",
                    "The tool result is tokenized and embedded exactly like your message (green blocks) "
                    "and added to the end of the context.\n\nThen the whole model runs again over the "
                    "complete context - now it has the information it was missing. (Real systems cache "
                    "the earlier tokens so this is fast.)",
                    data, build)

    # ================================================================ 9 done
    def _done_step(self):
        def build():
            n = len(self.sc["calls"])
            r = self.round(n)
            self.messages.append({"role": "assistant", "content": r["text"]})
            if self.gen_block:
                self.gen_block.set_label("answer\n{} tok".format(len(r["toks"])))
            self.clear_beams()
            self.chat("Model", r["text"])
            return Sequence(Func(self.f.set_bars, []), Wait(0.5))

        def text():
            calls = len(self.sc["calls"])
            loop = ("" if calls == 0 else
                    "  -> <tool_call> JSON -> program runs real code -> result back into context\n"
                    "  -> (model runs again{})\n".format(" - here {} times".format(calls) if calls > 1 else ""))
            return ("Summary:\n  your words -> tokens -> vectors -> layers + attention\n"
                    "  -> next-token probabilities -> one token at a time\n" + loop +
                    "  -> final answer\n\nThe LLM itself only ever predicts the next token. "
                    "Everything else is done by ordinary programs around it.")

        def data():
            return json.dumps(self.messages, indent=1)[-900:]
        return Step("Answer", "overview", "Final answer", text, data, build)
