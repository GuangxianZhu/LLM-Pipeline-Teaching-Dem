# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
LLM Pipeline Demo - 3Blue1Brown-style edition.
How "what you say" becomes tokens, vectors, attention, a tool call, real code, and an answer.

Run:   python main.py
Needs: pip install panda3d matplotlib

Keys:  Space / Right = next step (skip animation if still playing)    A = auto play
       1-7 = pick a question     R = restart     + / - = speed     D = deep dive on/off
       C = camera follow on/off  G = glow on/off  H = hide/show panels
       Right mouse drag = rotate   Mouse wheel = zoom
"""
import math
import os
import sys

from panda3d.core import loadPrcFileData

loadPrcFileData("", """
window-title LLM Pipeline Demo
win-size 1600 900
framebuffer-multisample 1
multisamples 8
textures-power-2 none
sync-video 1
audio-library-name null
""")
if os.environ.get("LLM_DEMO_OFFSCREEN"):
    loadPrcFileData("", "window-type offscreen\n")

from direct.filter.CommonFilters import CommonFilters                        # noqa: E402
from direct.gui.DirectGui import DGG, DirectButton, DirectFrame               # noqa: E402
from direct.gui.OnscreenText import OnscreenText                             # noqa: E402
from direct.interval.IntervalGlobal import Func, Sequence                    # noqa: E402
from direct.showbase.ShowBase import ShowBase                                # noqa: E402
from panda3d.core import AntialiasAttrib, Point3, TextNode                   # noqa: E402

from archmap import ArchMap                                                  # noqa: E402
from kit import GREY, WHITE, YELLOW, Fonts                                   # noqa: E402
from scenarios import SCENARIOS                                              # noqa: E402
from cache_story import CacheStory                                         # noqa: E402
from story import SEC_A, STAGES, Story                                       # noqa: E402

PANEL_BG = (0.04, 0.04, 0.05, 0.9)
LEFT_W, RIGHT_W = 0.74, 0.52          # screen space taken by the side panels (aspect2d units)
SHORT = ["Hi! Who are you?", "What is 17% of 2350?", "Tank temperature chart",
         "Cat in a cleanroom suit", "Count files (terminal)", "Files -> bar chart (2 tools)",
         "Why so fast? (cache)"]
BTN_BG = (0.13, 0.13, 0.15, 0.9)
ACCENT = (0.85, 0.62, 0.15, 1)


# ====================================================================== camera
class OrbitCamera:
    """Eases toward a wanted (target, heading, pitch, distance). Right-drag rotates, wheel zooms."""

    def __init__(self, base):
        self.base = base
        base.disableMouse()
        base.camLens.setFov(52)
        base.camLens.setNearFar(0.5, 600)
        self.cur = [SEC_A.x, SEC_A.y, SEC_A.z + 3, 0.0, 0.0, 24.0]
        self.want = list(self.cur)
        self.follow = True
        self.drag = None
        self.k_dist = 1.0     # pull back so the scene fits between the side panels
        self.cx = 0.0         # centre of the free screen area (aspect2d units)
        base.accept("mouse3", self._start)
        base.accept("mouse3-up", self._stop)
        base.accept("wheel_up", self.zoom, [0.9])
        base.accept("wheel_down", self.zoom, [1.1])
        base.taskMgr.add(self._task, "orbit-cam")

    def go_to(self, target, h=0.0, p=0.0, d=24.0, force=False):
        if self.follow or force:
            self.want = [target.x, target.y, target.z, h, p, d]

    def zoom(self, f):
        self.want[5] = max(4, min(250, self.want[5] * f))

    def _start(self):
        if self.base.mouseWatcherNode.hasMouse():
            m = self.base.mouseWatcherNode.getMouse()
            self.drag = (m.x, m.y)

    def _stop(self):
        self.drag = None

    def _task(self, task):
        mw = self.base.mouseWatcherNode
        if self.drag and mw.hasMouse():
            m = mw.getMouse()
            self.want[3] -= (m.x - self.drag[0]) * 100
            self.want[4] = max(-80, min(85, self.want[4] - (m.y - self.drag[1]) * 70))
            self.drag = (m.x, m.y)
        k = 1 - math.exp(-min(self.base.clock.getDt(), 0.1) * 2.4)
        for i in range(6):
            self.cur[i] += (self.want[i] - self.cur[i]) * k
        x, y, z, h, p, d = self.cur
        d *= self.k_dist
        lens = self.base.camLens
        lens.setFilmOffset(-(self.cx / self.base.getAspectRatio()) * lens.getFilmSize()[0] / 2, 0)
        hr, pr = math.radians(h), math.radians(p)
        self.base.camera.setPos(x + d * math.cos(pr) * math.sin(hr), y - d * math.cos(pr) * math.cos(hr),
                                z + d * math.sin(pr))
        self.base.camera.lookAt(Point3(x, y, z))
        return task.cont


# ====================================================================== 2D UI
class UI:
    def __init__(self, app):
        self.app = app
        f = Fonts.serif
        self.root = app.aspect2d.attachNewNode("ui")
        # ---- left: questions + conversation
        left = DirectFrame(parent=app.a2dTopLeft, frameColor=PANEL_BG, frameSize=(0, LEFT_W - 0.04, -1.96, 0),
                           pos=(0.02, 0, -0.02))
        self.left = left
        OnscreenText("LLM Pipeline", parent=left, pos=(0.04, -0.075), scale=0.05, fg=WHITE, font=f,
                     align=TextNode.ALeft)
        self.step_info = OnscreenText("", parent=left, pos=(0.04, -0.125), scale=0.028, fg=GREY, font=f,
                                      align=TextNode.ALeft, mayChange=True)
        self.q_buttons = []
        for i, sc in enumerate(SCENARIOS):
            b = DirectButton(parent=left, text="{}   {}".format(i + 1, SHORT[i]), text_font=f,
                             text_scale=0.026, text_align=TextNode.ALeft, text_fg=(0.92, 0.92, 0.95, 1),
                             text_pos=(0.02, -0.008), frameSize=(0, LEFT_W - 0.1, -0.024, 0.034), frameColor=BTN_BG,
                             relief=DGG.FLAT, pos=(0.03, 0, -0.18 - i * 0.064), command=app.pick, extraArgs=[i])
            self.q_buttons.append(b)
        OnscreenText("model architecture", parent=left, pos=(0.04, -0.64), scale=0.026, fg=GREY, font=f,
                     align=TextNode.ALeft)
        self.arch = ArchMap(left, (0.36, 0, -0.73))
        self.arch.root.setScale(0.83)
        OnscreenText("Conversation", parent=app.a2dTopRight, pos=(-RIGHT_W + 0.04, -0.75), scale=0.028,
                     fg=GREY, font=f, align=TextNode.ALeft)
        self.chat_text = OnscreenText("", parent=app.a2dTopRight, pos=(-RIGHT_W + 0.04, -0.8), scale=0.024,
                                      fg=(0.9, 0.9, 0.93, 1), font=f, align=TextNode.ALeft, wordwrap=19.5,
                                      mayChange=True)
        self.chat_lines = []
        # ---- top: stage chips
        self.chips = []
        self.stages = list(STAGES)
        for i, st in enumerate(STAGES):
            c = OnscreenText(st, parent=app.a2dTopCenter, pos=((LEFT_W - RIGHT_W) / 2 + (i - 4) * 0.2, -0.07),
                             scale=0.03,
                             fg=(0.45, 0.45, 0.48, 1), font=f, mayChange=True)
            self.chips.append(c)
        # ---- right: controls
        self.btn = {}
        specs = [("next", "Next  (Space)", app.next), ("auto", "Auto: OFF  (A)", app.toggle_auto),
                 ("speed", "Speed 1x  (+/-)", app.cycle_speed), ("restart", "Restart  (R)", app.restart_demo),
                 ("deep", "Deep dive: ON  (D)", app.toggle_deep), ("cam", "Camera: follow  (C)", app.toggle_follow),
                 ("glow", "Glow: ON  (G)", app.toggle_glow), ("hide", "Hide panels  (H)", app.toggle_panels)]
        for i, (key, label, cmd) in enumerate(specs):
            b = DirectButton(parent=app.a2dTopRight, text=label, text_font=f, text_scale=0.028,
                             text_fg=(1, 1, 1, 1), text_pos=(0, -0.01),
                             frameSize=(-0.23, 0.23, -0.03, 0.04),
                             frameColor=(0.2, 0.32, 0.5, 0.95) if key == "next" else BTN_BG,
                             relief=DGG.FLAT, pos=(-0.27, 0, -0.07 - i * 0.082), command=cmd)
            self.btn[key] = b
        # ---- bottom: caption (the explanation, like video subtitles)
        self.caption = OnscreenText("", pos=(0, -0.79), scale=0.04, fg=WHITE, bg=(0, 0, 0, 0.62),
                                    font=Fonts.symbol, wordwrap=48, mayChange=True)
        self.title_hint = OnscreenText("right-drag: rotate    wheel: zoom", parent=app.a2dBottomRight,
                                       pos=(-0.04, 0.03), scale=0.024, fg=(0.35, 0.35, 0.38, 1), font=f,
                                       align=TextNode.ARight)

    def set_stages(self, names):
        self.stages = list(names)
        for i, c in enumerate(self.chips):
            c.setText(names[i] if i < len(names) else "")

    def set_stage(self, stage):
        for st, c in zip(self.stages, self.chips):
            c.setFg(YELLOW if st == stage else (0.45, 0.45, 0.48, 1))

    def set_caption(self, s):
        self.caption.setText(s)

    def mark_question(self, i):
        for k, b in enumerate(self.q_buttons):
            b["frameColor"] = ACCENT if k == i else BTN_BG

    def chat(self, who, s):
        s = " ".join(s.split())
        if len(s) > 120:
            s = s[:117] + "..."
        self.chat_lines.append("{}:  {}".format(who, s))
        self.chat_text.setText("\n\n".join(self.chat_lines[-4:]))

    def clear_chat(self):
        self.chat_lines = []
        self.chat_text.setText("")

    def set_btn(self, key, label):
        self.btn[key]["text"] = label


# ====================================================================== app
class App(ShowBase):
    SPEEDS = [0.5, 1.0, 1.5, 2.0, 3.0]

    def __init__(self):
        ShowBase.__init__(self)
        self.setBackgroundColor(0, 0, 0, 1)
        self.render.setAntialias(AntialiasAttrib.MMultisample)
        Fonts.load(self.loader)
        self.cam_ctl = OrbitCamera(self)
        self.filters = CommonFilters(self.win, self.cam)
        self.glow = False
        self.ui = UI(self)
        self.toggle_glow()
        self.panels_on = True
        self._frame()
        self.board = None
        self.ival = None
        self.auto = False
        self.deep = True
        self.speed_i = 1
        self.sc_i = 0
        self.panels_on = True
        for key, fn in [("space", self.next), ("arrow_right", self.next), ("a", self.toggle_auto),
                        ("r", self.restart_demo), ("d", self.toggle_deep), ("c", self.toggle_follow),
                        ("g", self.toggle_glow), ("h", self.toggle_panels), ("+", self.speed_up),
                        ("=", self.speed_up), ("-", self.speed_down), ("escape", sys.exit)]:
            self.accept(key, fn)
        for i in range(len(SCENARIOS)):
            self.accept(str(i + 1), self.pick, [i])
        self.pick(0)

    @property
    def speed(self):
        return self.SPEEDS[self.speed_i]

    # ---- question control
    def pick(self, i):
        self.sc_i = i
        self.restart_demo()

    def restart_demo(self):
        self._stop()
        if self.board:
            self.board.removeNode()
        self.board = self.render.attachNewNode("board")
        self.board.setLightOff()
        self.ui.clear_chat()
        self.ui.mark_question(self.sc_i)
        sc = SCENARIOS[self.sc_i]
        self.ui.set_stages(CacheStory.stages if sc.get("kind") == "cache" else STAGES)
        self.ui.set_stage(None)
        if sc.get("kind") == "cache":
            self.story = CacheStory(self, sc, self.deep)
            intro = self.story.intro + "\nPress Space (or Next) to start."
        else:
            self.story = Story(self, sc, self.deep)
            intro = 'Question {}:  "{}"\nPress Space (or Next) to follow it through the model.'.format(
                self.sc_i + 1, sc["prompt"])
            self.cam_ctl.go_to(SEC_A + Point3(0, 0, 3.0), 0, 0, 24, force=True)
        self.steps = self.story.steps
        self.idx = -1
        self.ui.step_info.setText("{} steps   -   Space = next,  A = auto".format(len(self.steps)))
        self.ui.set_caption(intro)
        if self.auto:
            self.taskMgr.doMethodLater(1.0, lambda t: self.next(), "auto-next")

    def _stop(self):
        self.taskMgr.remove("auto-next")
        if self.ival:
            self.ival.finish()
            self.ival = None

    def next(self):
        if self.ival and self.ival.isPlaying():
            self.ival.finish()                     # skip to the end of this step
            return
        if self.idx + 1 >= len(self.steps):
            self.ui.step_info.setText("Finished  -  pick another question (1-7) or Restart")
            return
        self.idx += 1
        st = self.steps[self.idx]
        default_map = {"Context": ("input",), "Tokens": ("input",), "Output": ("linear", "softmax", "output")}
        self.ui.arch.highlight(*default_map.get(st.stage, ()))
        caption, ival = st.build()
        self.ui.set_stage(st.stage)
        self.ui.set_caption(caption)
        self.ui.step_info.setText("Step {} / {}   -   {}".format(self.idx + 1, len(self.steps), st.stage))
        self.ival = Sequence(ival, Func(self._done))
        self.ival.start(0.0, -1.0, self.speed)

    def _done(self):
        if self.auto:
            self.taskMgr.doMethodLater(2.4 / self.speed, lambda t: self.next(), "auto-next")

    # ---- toggles
    def toggle_auto(self):
        self.auto = not self.auto
        self.ui.set_btn("auto", "Auto: ON  (A)" if self.auto else "Auto: OFF  (A)")
        if self.auto and not (self.ival and self.ival.isPlaying()):
            self.next()
        if not self.auto:
            self.taskMgr.remove("auto-next")

    def _apply_speed(self):
        self.ui.set_btn("speed", "Speed {:g}x  (+/-)".format(self.speed))
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

    def toggle_deep(self):
        self.deep = not self.deep
        self.ui.set_btn("deep", "Deep dive: ON  (D)" if self.deep else "Deep dive: OFF  (D)")
        self.restart_demo()

    def toggle_follow(self):
        self.cam_ctl.follow = not self.cam_ctl.follow
        self.ui.set_btn("cam", "Camera: follow  (C)" if self.cam_ctl.follow else "Camera: free  (C)")

    def toggle_glow(self):
        if self.glow:
            self.filters.delBloom()
            self.glow = False
        else:
            self.glow = bool(self.filters.setBloom(blend=(0.3, 0.4, 0.3, 0.0), mintrigger=0.6, maxtrigger=1.0,
                                                   desat=0.1, intensity=1.4, size="medium"))
        self.ui.set_btn("glow", "Glow: ON  (G)" if self.glow else "Glow: OFF  (G)")

    def _frame(self):
        if self.panels_on:
            free = 2 * self.getAspectRatio() - LEFT_W - RIGHT_W
            self.cam_ctl.k_dist = 2 * self.getAspectRatio() / free
            self.cam_ctl.cx = (LEFT_W - RIGHT_W) / 2
        else:
            self.cam_ctl.k_dist, self.cam_ctl.cx = 1.0, 0.0

    def toggle_panels(self):
        self.panels_on = not self.panels_on
        self._frame()
        for n in [self.ui.left] + list(self.ui.btn.values()):
            n.show() if self.panels_on else n.hide()
        self.ui.btn["hide"].show()
        self.ui.set_btn("hide", "Hide panels  (H)" if self.panels_on else "Show panels  (H)")


if __name__ == "__main__":
    App().run()
