# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
LLM Pipeline Demo - how "what you say" becomes a tool call and a picture.

Run:   python main.py
Needs: pip install panda3d matplotlib

Keys:  Space / Right = next step     A = auto play      R = restart
       1-6 = pick a question          + / - = speed      C = camera follow on/off
       Right mouse drag = rotate      Mouse wheel = zoom  O = overview
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # shared sim/tools/scenarios

from panda3d.core import loadPrcFileData

loadPrcFileData("", """
window-title LLM Pipeline Demo
win-size 1600 900
framebuffer-multisample 1
multisamples 4
textures-power-2 none
sync-video 1
audio-library-name null
""")
if os.environ.get("LLM_DEMO_OFFSCREEN"):
    loadPrcFileData("", "window-type offscreen\n")

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame, DirectLabel   # noqa: E402
from direct.gui.OnscreenImage import OnscreenImage                           # noqa: E402
from direct.gui.OnscreenText import OnscreenText                             # noqa: E402
from direct.interval.IntervalGlobal import Func, Sequence                    # noqa: E402
from direct.showbase.ShowBase import ShowBase                                # noqa: E402
from panda3d.core import (AntialiasAttrib, Filename, Point3, TextNode,       # noqa: E402
                          TransparencyAttrib)

import scene                                                                 # noqa: E402
from director import STAGES, Director, _v                                    # noqa: E402
from scenarios import SCENARIOS                                              # noqa: E402

FONT_CANDIDATES = {
    "ui": ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
           "/System/Library/Fonts/Supplemental/Arial.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "mono": ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
             "/System/Library/Fonts/Menlo.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
}

PANEL_BG = (0.07, 0.08, 0.1, 0.88)
LEFT_W, RIGHT_W = 0.86, 1.04     # width of the side panels (aspect2d units)
ACCENT = (1.0, 0.72, 0.3, 1)


# ====================================================================== camera
class OrbitCamera:
    def __init__(self, base):
        self.base = base
        base.disableMouse()
        base.camLens.setFov(62)
        self.cur = list(self._unpack(scene.VIEWS["overview"]))
        self.want = list(self.cur)
        self.drag = None
        self.follow = True
        base.accept("mouse3", self._start_drag)
        base.accept("mouse3-up", self._stop_drag)
        base.accept("wheel_up", self.zoom, [0.88])
        base.accept("wheel_down", self.zoom, [1.14])
        base.taskMgr.add(self._task, "orbit-cam")

    @staticmethod
    def _unpack(v):
        (x, y, z), h, p, d = v
        return [x, y, z, h, p, d]

    def go(self, view, force=False):
        if self.follow or force:
            self.want = self._unpack(scene.VIEWS[view] if isinstance(view, str) else view)

    def zoom(self, f):
        self.want[5] = min(80, max(5, self.want[5] * f))

    def _start_drag(self):
        if self.base.mouseWatcherNode.hasMouse():
            m = self.base.mouseWatcherNode.getMouse()
            self.drag = (m.x, m.y)

    def _stop_drag(self):
        self.drag = None

    def _task(self, task):
        mw = self.base.mouseWatcherNode
        if self.drag and mw.hasMouse():
            m = mw.getMouse()
            dx, dy = m.x - self.drag[0], m.y - self.drag[1]
            self.drag = (m.x, m.y)
            self.want[3] -= dx * 110
            self.want[4] = min(85, max(3, self.want[4] - dy * 70))
        dt = min(self.base.clock.getDt(), 0.1)
        k = 1 - math.exp(-dt * 3.2)
        for i in range(6):
            self.cur[i] += (self.want[i] - self.cur[i]) * k
        lens = self.base.camLens
        a = self.base.getAspectRatio()
        cx = (LEFT_W - RIGHT_W) / 2.0          # centre of the free 3D area (aspect2d units)
        lens.setFilmOffset(-(cx / a) * lens.getFilmSize()[0] / 2, 0)
        x, y, z, h, p, d = self.cur
        hr, pr = math.radians(h), math.radians(p)
        cam = self.base.camera
        cam.setPos(x + d * math.cos(pr) * math.sin(hr), y - d * math.cos(pr) * math.cos(hr), z + d * math.sin(pr))
        cam.lookAt(Point3(x, y, z))
        return task.cont


# ====================================================================== 2D UI
class UI:
    def __init__(self, app):
        self.app = app
        b = app
        self.font = self._font("ui")
        self.mono = self._font("mono") or self.font
        self.chat_lines = []
        self.preview = None

        # ---- left column: questions + chat
        left = DirectFrame(parent=b.a2dTopLeft, frameColor=PANEL_BG,
                           frameSize=(0, LEFT_W - 0.04, -1.42, 0), pos=(0.02, 0, -0.02))
        self.left = left
        OnscreenText("LLM PIPELINE DEMO", parent=left, pos=(0.04, -0.08), scale=0.05,
                     fg=ACCENT, align=TextNode.ALeft, font=self.font)
        OnscreenText("Pick a question:", parent=left, pos=(0.04, -0.15), scale=0.032,
                     fg=(0.8, 0.8, 0.85, 1), align=TextNode.ALeft, font=self.font)
        self.q_buttons = []
        for i, sc in enumerate(SCENARIOS):
            btn = DirectButton(parent=left, text="{}  {}".format(i + 1, sc["prompt"]),
                               text_align=TextNode.ALeft, text_scale=0.03, text_font=self.font,
                               text_fg=(0.95, 0.95, 1, 1), text_pos=(0.02, -0.012),
                               frameSize=(0, 0.76, -0.032, 0.035), frameColor=(0.18, 0.2, 0.25, 1),
                               relief=DGG.FLAT, pos=(0.03, 0, -0.22 - i * 0.078),
                               command=app.pick, extraArgs=[i])
            self.q_buttons.append(btn)
        OnscreenText("Conversation:", parent=left, pos=(0.04, -0.73), scale=0.032,
                     fg=(0.8, 0.8, 0.85, 1), align=TextNode.ALeft, font=self.font)
        self.chat_text = OnscreenText("", parent=left, pos=(0.04, -0.79), scale=0.028,
                                      fg=(0.92, 0.92, 0.95, 1), align=TextNode.ALeft,
                                      wordwrap=26, font=self.font, mayChange=True)

        # ---- top: stage chips
        self.chips = []
        for i, st in enumerate(STAGES):
            c = DirectLabel(parent=b.a2dTopCenter, text=st, text_scale=0.026, text_font=self.font,
                            text_fg=(0.85, 0.85, 0.9, 1), frameColor=(0.16, 0.17, 0.2, 0.9),
                            frameSize=(-0.083, 0.083, -0.025, 0.04),
                            pos=((LEFT_W - RIGHT_W) / 2 + (i - 4) * 0.176, 0, -0.06))
            self.chips.append(c)

        # ---- right panel: explanation + data
        right = DirectFrame(parent=b.a2dTopRight, frameColor=PANEL_BG,
                            frameSize=(-(RIGHT_W - 0.04), 0, -1.96, 0), pos=(-0.02, 0, -0.02))
        self.step_info = OnscreenText("", parent=right, pos=(-0.96, -0.06), scale=0.03,
                                      fg=(0.7, 0.75, 0.85, 1), align=TextNode.ALeft,
                                      font=self.font, mayChange=True)
        self.title = OnscreenText("", parent=right, pos=(-0.96, -0.14), scale=0.046, fg=ACCENT,
                                  align=TextNode.ALeft, wordwrap=20, font=self.font, mayChange=True)
        self.expl = OnscreenText("", parent=right, pos=(-0.96, -0.27), scale=0.034,
                                 fg=(0.95, 0.95, 0.97, 1), align=TextNode.ALeft, wordwrap=27.5,
                                 font=self.font, mayChange=True)
        DirectFrame(parent=right, frameColor=(0.02, 0.03, 0.04, 0.95),
                    frameSize=(-0.97, -0.03, -1.93, -1.0))
        OnscreenText("INSIDE THE COMPUTER", parent=right, pos=(-0.94, -1.05), scale=0.026,
                     fg=(0.55, 0.85, 1, 1), align=TextNode.ALeft, font=self.font)
        self.data = OnscreenText("", parent=right, pos=(-0.94, -1.1), scale=0.025,
                                 fg=(0.75, 1, 0.78, 1), align=TextNode.ALeft, wordwrap=36.5,
                                 font=self.mono, mayChange=True)

        # ---- bottom: controls
        bar = b.a2dBottomCenter
        specs = [("Restart", app.restart_demo), ("Next step >", app.next),
                 ("Auto: OFF", app.toggle_auto), ("Speed 1x", app.cycle_speed),
                 ("Cam: follow", app.toggle_follow), ("Overview", app.overview)]
        self.ctrl = {}
        widths = [0.07 + len(t) * 0.0165 for t, _ in specs]
        x = (LEFT_W - RIGHT_W) / 2 - (sum(widths) + 0.02 * (len(specs) - 1)) / 2
        for (text, cmd), w in zip(specs, widths):
            btn = DirectButton(parent=bar, text=text, text_scale=0.03, text_font=self.font,
                               text_fg=(1, 1, 1, 1), frameSize=(-w / 2, w / 2, -0.035, 0.055),
                               frameColor=(0.2, 0.36, 0.6, 1) if "Next" in text else (0.2, 0.22, 0.27, 1),
                               relief=DGG.FLAT, pos=(x + w / 2, 0, 0.06), command=cmd)
            self.ctrl[text.split()[0]] = btn
            x += w + 0.02
        OnscreenText("Space: next   A: auto   R: restart   +/-: speed   C: camera   O: overview   "
                     "1-6: question   right-drag: rotate   wheel: zoom", parent=bar,
                     pos=((LEFT_W - RIGHT_W) / 2, 0.125), scale=0.024, fg=(0.6, 0.62, 0.68, 1), font=self.font)

    def _font(self, kind):
        for p in FONT_CANDIDATES[kind]:
            if os.path.exists(p):
                f = self.app.loader.loadFont(Filename.fromOsSpecific(p).getFullpath())
                if f and f.isValid():
                    f.setPixelsPerUnit(60)
                    return f
        return None

    # ---- updates
    def set_stage(self, stage):
        idx = STAGES.index(stage) if stage in STAGES else -1
        for i, c in enumerate(self.chips):
            if i == idx:
                c["frameColor"] = (0.85, 0.5, 0.15, 1)
                c["text_fg"] = (1, 1, 1, 1)
            else:
                c["frameColor"] = (0.16, 0.17, 0.2, 0.9)
                c["text_fg"] = (0.85, 0.85, 0.9, 1)

    def set_step(self, info, title, text, data):
        self.step_info.setText(info)
        self.title.setText(title)
        self.expl.setText(text)
        self.set_data(data)

    def set_data(self, data):
        lines = data.splitlines()
        if len(lines) > 30:
            lines = lines[:29] + ["..."]
        self.data.setText("\n".join(lines))

    def mark_question(self, i):
        for k, b in enumerate(self.q_buttons):
            b["frameColor"] = (0.85, 0.5, 0.15, 1) if k == i else (0.18, 0.2, 0.25, 1)

    def chat(self, who, text):
        text = " ".join(text.split())
        if len(text) > 150:
            text = text[:147] + "..."
        self.chat_lines.append("{}: {}".format(who, text))
        self.chat_text.setText("\n\n".join(self.chat_lines[-4:]))

    def clear_chat(self):
        self.chat_lines = []
        self.chat_text.setText("")

    def show_result(self, tex, shape):
        self.hide_result()
        h, w = shape[:2]
        sz = min(0.25, 0.38 * h / w)      # fits below the left panel
        sx = sz * w / h
        self.preview = OnscreenImage(image=tex, parent=self.app.a2dBottomLeft,
                                     pos=(0.02 + sx, 0, 0.03 + sz), scale=(sx, 1, sz))
        self.preview.setTransparency(TransparencyAttrib.MAlpha)

    def hide_result(self):
        if self.preview:
            self.preview.destroy()
            self.preview = None

    def set_button(self, key, text):
        self.ctrl[key]["text"] = text


# ====================================================================== app
class App(ShowBase):
    SPEEDS = [0.5, 1.0, 2.0, 4.0]

    def __init__(self):
        ShowBase.__init__(self)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.factory = scene.Factory(self)
        self.cam_ctl = OrbitCamera(self)
        self.ui = UI(self)
        self.director = Director(self)
        self.speed_i = 1
        self.auto = False
        self.ival = None
        self.idx = -1
        self.sc_i = 0
        for key, fn in [("space", self.next), ("arrow_right", self.next), ("a", self.toggle_auto),
                        ("r", self.restart_demo), ("c", self.toggle_follow), ("o", self.overview),
                        ("+", self.speed_up), ("=", self.speed_up), ("-", self.speed_down),
                        ("escape", sys.exit)]:
            self.accept(key, fn)
        for i in range(len(SCENARIOS)):
            self.accept(str(i + 1), self.pick, [i])
        self.pick(0)

    @property
    def speed(self):
        return self.SPEEDS[self.speed_i]

    # ---- scenario control
    def pick(self, i):
        self.sc_i = i
        self.restart_demo()

    def restart_demo(self):
        self._stop()
        self.ui.hide_result()
        self.ui.clear_chat()
        self.ui.mark_question(self.sc_i)
        self.steps = self.director.load(SCENARIOS[self.sc_i])
        self.idx = -1
        self.ui.set_stage(None)
        self.ui.set_step("{} steps  -  press Next (Space) or Auto (A)".format(len(self.steps)),
                         "Question: \"{}\"".format(SCENARIOS[self.sc_i]["prompt"]),
                         "Follow this sentence through the LLM factory:\n\n"
                         "1 Input  >  2 Tokenizer  >  3 Embedding  >  4 Transformer  >  "
                         "5 Next-token prediction  >  6 Output + Router  >  7 Tools  >  back to 1\n\n"
                         "Press Next to go one step at a time, or Auto to watch it all.",
                         "(the real data inside the computer will appear here)")
        self.cam_ctl.go("overview", force=True)
        if self.auto:
            self.taskMgr.doMethodLater(1.0, lambda t: self.next(), "auto-next")

    def _stop(self):
        self.taskMgr.remove("auto-next")
        if self.ival:
            self.ival.finish()
            self.ival = None

    def next(self):
        if self.ival and self.ival.isPlaying():
            self.ival.finish()          # skip to the end of the current animation
            return
        if self.idx + 1 >= len(self.steps):
            self.ui.step_info.setText("Finished - pick another question or press Restart")
            return
        self.idx += 1
        st = self.steps[self.idx]
        ival = st.build()
        self.cur_step = st
        self.ui.set_stage(st.stage)
        self.ui.set_step("Step {} / {}   -   {}".format(self.idx + 1, len(self.steps), st.stage),
                         _v(st.title), _v(st.text), _v(st.data))
        self.cam_ctl.go(_v(st.view))
        self.ival = Sequence(ival, Func(self._step_done))
        self.ival.start(0.0, -1.0, self.speed)

    def _step_done(self):
        self.ui.set_data(_v(self.cur_step.data))     # values that appeared during the animation
        if self.auto:
            self.taskMgr.doMethodLater(1.6 / self.speed, lambda t: self.next(), "auto-next")

    def toggle_auto(self):
        self.auto = not self.auto
        self.ui.set_button("Auto:", "Auto: ON" if self.auto else "Auto: OFF")
        if self.auto and not (self.ival and self.ival.isPlaying()):
            self.next()
        if not self.auto:
            self.taskMgr.remove("auto-next")

    def _apply_speed(self):
        self.ui.set_button("Speed", "Speed {:g}x".format(self.speed))
        if self.ival and self.ival.isPlaying():
            self.ival.setPlayRate(self.speed)

    def cycle_speed(self):
        self.speed_i = (self.speed_i + 1) % len(self.SPEEDS)
        self._apply_speed()

    def speed_up(self):
        self.speed_i = min(len(self.SPEEDS) - 1, self.speed_i + 1)
        self._apply_speed()

    def speed_down(self):
        self.speed_i = max(0, self.speed_i - 1)
        self._apply_speed()

    def toggle_follow(self):
        self.cam_ctl.follow = not self.cam_ctl.follow
        self.ui.set_button("Cam:", "Cam: follow" if self.cam_ctl.follow else "Cam: free")

    def overview(self):
        self.cam_ctl.go("overview", force=True)


if __name__ == "__main__":
    App().run()
