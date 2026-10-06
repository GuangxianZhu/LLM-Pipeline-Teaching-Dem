# -*- coding: utf-8 -*-
# Claude Opus 写的
# 中英文切换：Claude 改了这个文件（--lang、L 键、界面文字、实时切换 set_language）
"""
LLM Pipeline Demo - 3Blue1Brown-style edition.
How "what you say" becomes tokens, vectors, attention, a tool call, real code, and an answer.

Run:   python main.py
Needs: pip install panda3d matplotlib

Keys:  Space / Right = next step (skip animation if still playing)    A = auto play
       1-7 = pick a question     R = restart     + / - = speed     D = deep dive on/off
       C = camera follow on/off  G = glow on/off  H = hide/show panels
       L = language English / Chinese   (start in Chinese: python main.py --lang zh)
       Right mouse drag = rotate   Mouse wheel = zoom
"""
import math
import os
import random
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

import i18n                                                                  # noqa: E402
from archmap import ArchMap                                                  # noqa: E402
from i18n import t                                                           # noqa: E402
from kit import GREY, WHITE, YELLOW, Fonts, display, ui_font, wrap_cjk      # noqa: E402
from scenarios import SCENARIOS                                              # noqa: E402
from cache_story import CacheStory                                         # noqa: E402
from story import SEC_A, STAGES, Story                                       # noqa: E402

PANEL_BG = (0.04, 0.04, 0.05, 0.9)
LEFT_W, RIGHT_W = 0.74, 0.52          # screen space taken by the side panels (aspect2d units)
BTN_BG = (0.13, 0.13, 0.15, 0.9)
BTN_PITCH = 0.074                     # distance between the control buttons (9 of them now)
CAPTION_WRAP = 44.0                   # width of a Chinese caption line in text units (English: wordwrap=48)
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
        f = ui_font()
        self.static = []          # (OnscreenText, i18n key): the labels that never change except with the language
        self.root = app.aspect2d.attachNewNode("ui")
        # ---- left: questions + conversation
        left = DirectFrame(parent=app.a2dTopLeft, frameColor=PANEL_BG, frameSize=(0, LEFT_W - 0.04, -1.96, 0),
                           pos=(0.02, 0, -0.02))
        self.left = left
        self.static.append((OnscreenText(display(t("ui.title")), parent=left, pos=(0.04, -0.075), scale=0.05,
                                         fg=WHITE, font=f, align=TextNode.ALeft, mayChange=True), "ui.title"))
        self.step_info = OnscreenText("", parent=left, pos=(0.04, -0.125), scale=0.028, fg=GREY, font=f,
                                      align=TextNode.ALeft, mayChange=True)
        self.q_buttons = []
        for i, sc in enumerate(SCENARIOS):
            b = DirectButton(parent=left, text=display(self._q_label(i)), text_font=f,
                             text_scale=0.026, text_align=TextNode.ALeft, text_fg=(0.92, 0.92, 0.95, 1),
                             text_pos=(0.02, -0.008), frameSize=(0, LEFT_W - 0.1, -0.024, 0.034), frameColor=BTN_BG,
                             relief=DGG.FLAT, pos=(0.03, 0, -0.18 - i * 0.064), command=app.pick, extraArgs=[i])
            self.q_buttons.append(b)
        self.static.append((OnscreenText(display(t("ui.arch_title")), parent=left, pos=(0.04, -0.64), scale=0.026,
                                         fg=GREY, font=f, align=TextNode.ALeft, mayChange=True), "ui.arch_title"))
        self.arch = ArchMap(left, (0.36, 0, -0.73))
        self.arch.root.setScale(0.83)
        self.static.append((OnscreenText(display(t("ui.conversation")), parent=app.a2dTopRight,
                                         pos=(-RIGHT_W + 0.04, -0.75), scale=0.028, fg=GREY, font=f,
                                         align=TextNode.ALeft, mayChange=True), "ui.conversation"))
        self.chat_text = OnscreenText("", parent=app.a2dTopRight, pos=(-RIGHT_W + 0.04, -0.8), scale=0.024,
                                      fg=(0.9, 0.9, 0.93, 1), font=f, align=TextNode.ALeft, wordwrap=19.5,
                                      mayChange=True)
        self.chat_lines = []                  # (who, message) - rendered in the current language
        # ---- bottom right: the tracker (the followed token's current 32 numbers)
        self.tracker = DirectFrame(parent=app.a2dBottomRight, frameColor=PANEL_BG,
                                   frameSize=(-RIGHT_W + 0.02, -0.02, 0.06, 0.86))
        self.trk_title = OnscreenText("", parent=self.tracker, pos=(-RIGHT_W / 2, 0.8), scale=0.03, fg=YELLOW,
                                      font=f, mayChange=True)
        self.trk_where = OnscreenText("", parent=self.tracker, pos=(-RIGHT_W + 0.2, 0.72), scale=0.026,
                                      fg=WHITE, font=f, align=TextNode.ALeft, wordwrap=11, mayChange=True)
        self.trk_hist = OnscreenText("", parent=self.tracker, pos=(-RIGHT_W + 0.2, 0.55), scale=0.022,
                                     fg=GREY, font=f, align=TextNode.ALeft, wordwrap=12.5, mayChange=True)
        self.trk_cells = None
        self.trk_token = None
        self.trk_steps = []                   # i18n keys of the places the followed token has been
        self.tracker.hide()
        # ---- top: stage chips
        self.chips = []
        self.stages = list(STAGES)
        self.cur_stage = None
        for i, st in enumerate(STAGES):
            c = OnscreenText(display(t("stage." + st)), parent=app.a2dTopCenter,
                             pos=((LEFT_W - RIGHT_W) / 2 + (i - 4) * 0.2, -0.07),
                             scale=0.03,
                             fg=(0.45, 0.45, 0.48, 1), font=f, mayChange=True)
            self.chips.append(c)
        # ---- right: controls
        self.btn = {}
        specs = [("next", t("ui.btn.next"), app.next), ("auto", t("ui.btn.auto_off"), app.toggle_auto),
                 ("speed", t("ui.btn.speed", speed=1.0), app.cycle_speed),
                 ("restart", t("ui.btn.restart"), app.restart_demo),
                 ("deep", t("ui.btn.deep_on"), app.toggle_deep), ("cam", t("ui.btn.cam_follow"), app.toggle_follow),
                 ("glow", t("ui.btn.glow_on"), app.toggle_glow), ("hide", t("ui.btn.hide"), app.toggle_panels),
                 ("lang", t("ui.btn.lang"), app.toggle_language)]
        for i, (key, label, cmd) in enumerate(specs):
            b = DirectButton(parent=app.a2dTopRight, text=display(label), text_font=f, text_scale=0.028,
                             text_fg=(1, 1, 1, 1), text_pos=(0, -0.01),
                             frameSize=(-0.23, 0.23, -0.03, 0.04),
                             frameColor=(0.2, 0.32, 0.5, 0.95) if key == "next" else BTN_BG,
                             relief=DGG.FLAT, pos=(-0.27, 0, -0.07 - i * BTN_PITCH), command=cmd)
            self.btn[key] = b
        # ---- bottom: caption (the explanation, like video subtitles)
        self.caption = OnscreenText("", pos=(0, -0.79), scale=0.04, fg=WHITE, bg=(0, 0, 0, 0.62),
                                    font=ui_font("symbol"), wordwrap=48, mayChange=True)
        self.static.append((OnscreenText(display(t("ui.hint")), parent=app.a2dBottomRight,
                                         pos=(-0.04, 0.03), scale=0.024, fg=(0.35, 0.35, 0.38, 1), font=f,
                                         align=TextNode.ARight, mayChange=True), "ui.hint"))

    @staticmethod
    def _q_label(i):
        return "{}   {}".format(i + 1, t("ui.q.{}".format(i + 1)))

    def relabel(self):
        """The language changed: every label, button text and font. (The scene is rebuilt by App.)"""
        f = ui_font()
        for o, key in self.static:
            o.setFont(f)
            o.setText(display(t(key)))
        for i, b in enumerate(self.q_buttons):
            b["text_font"] = f
            b["text"] = display(self._q_label(i))
        for o in (self.step_info, self.chat_text, self.trk_title, self.trk_where, self.trk_hist):
            o.setFont(f)
        self.caption.setFont(ui_font("symbol"))
        self.set_stages(self.stages)
        self.set_stage(self.cur_stage)
        self._show_chat()
        self._show_tracker()

    def set_info(self, key, **kw):
        self.step_info.setText(display(t(key, **kw)))

    def set_stages(self, names):
        self.stages = list(names)
        for i, c in enumerate(self.chips):
            c.setFont(ui_font())
            c.setText(display(t("stage." + names[i])) if i < len(names) else "")

    def set_stage(self, stage):
        self.cur_stage = stage
        for st, c in zip(self.stages, self.chips):
            c.setFg(YELLOW if st == stage else (0.45, 0.45, 0.48, 1))

    def set_caption(self, s):
        if i18n.LANG != "en":
            s = wrap_cjk(s, CAPTION_WRAP, ui_font("symbol"))      # Panda3D breaks lines at spaces only
        self.caption.setText(display(s, "symbol"))

    def mark_question(self, i):
        for k, b in enumerate(self.q_buttons):
            b["frameColor"] = ACCENT if k == i else BTN_BG

    def chat(self, who, s):
        s = " ".join(s.split())
        if len(s) > 120:
            s = s[:117] + "..."
        self.chat_lines.append((who, s))
        self._show_chat()

    def _show_chat(self):
        lines = [t("ui.chat_line", who=t("ui.who." + who.lower()), msg=msg) for who, msg in self.chat_lines[-3:]]
        self.chat_text.setText(display("\n\n".join(lines)))

    def clear_chat(self):
        self.chat_lines = []
        self.chat_text.setText("")

    def track(self, token, vec, scale, where):
        """Show the followed token's current vector (one column of coloured cells). `where` is an i18n key."""
        from kit import heatmap
        import numpy as np
        if self.trk_cells:
            self.trk_cells.removeNode()
        v = np.asarray(vec)[None, :].T
        h = 0.68 / len(vec)
        self.trk_cells = heatmap(self.tracker, v, -RIGHT_W + 0.06, 0.76, 0.1, h, scale, gap=0.08)
        self.trk_token = token
        if not self.trk_steps or self.trk_steps[-1] != where:
            self.trk_steps.append(where)
        self._show_tracker()
        self.tracker.show()

    def _show_tracker(self):
        if self.trk_token is None:
            return
        wrap = (lambda s, w: wrap_cjk(s, w, ui_font())) if i18n.LANG != "en" else (lambda s, w: s)
        self.trk_title.setText(display(t("ui.trk.following", token=self.trk_token)))
        self.trk_where.setText(display(wrap(t("ui.trk.now", where=t(self.trk_steps[-1])), 11)))
        past = [t(k) for k in self.trk_steps[:-1]]
        self.trk_hist.setText(display(wrap(t("ui.trk.path", path="\n".join(past[-9:])), 12.5)) if past else "")

    def reset_tracker(self):
        self.trk_steps = []
        self.trk_token = None
        if self.trk_cells:
            self.trk_cells.removeNode()
            self.trk_cells = None
        self.tracker.hide()

    def set_btn(self, key, label):
        self.btn[key]["text_font"] = ui_font()
        self.btn[key]["text"] = display(label)


# ====================================================================== app
class App(ShowBase):
    SPEEDS = [0.5, 1.0, 1.5, 2.0, 3.0]

    def __init__(self, lang=None):
        if lang:
            i18n.set_lang(lang)               # before anything is drawn
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
        self.seed = None                      # seed of the current Engine run (question 1 samples its first word)
        self.finished = False                 # the "Finished" message is showing
        self.save_lang = True                 # write settings.json when the language changes
        for key, fn in [("space", self.next), ("arrow_right", self.next), ("a", self.toggle_auto),
                        ("r", self.restart_demo), ("d", self.toggle_deep), ("c", self.toggle_follow),
                        ("g", self.toggle_glow), ("h", self.toggle_panels), ("l", self.toggle_language), ("+", self.speed_up),
                        ("=", self.speed_up), ("-", self.speed_down), ("escape", sys.exit)]:
            self.accept(key, fn)
        for i in range(len(SCENARIOS)):
            self.accept(str(i + 1), self.pick, [i])
        self.refresh_buttons()
        self.pick(0)

    @property
    def speed(self):
        return self.SPEEDS[self.speed_i]

    # ---- question control
    def pick(self, i):
        self.sc_i = i
        self.restart_demo()

    def restart_demo(self, keep_seed=False):
        """(Re)build the current question from the first step. keep_seed: the same random draws as before
        (question 1 picks Hi / Hello at random), used when only the language changed."""
        self._stop()
        self.finished = False
        if not keep_seed or self.seed is None:
            self.seed = random.randrange(10 ** 6)
        if self.board:
            self.board.removeNode()
        self.board = self.render.attachNewNode("board")
        self.board.setLightOff()
        self.ui.clear_chat()
        self.ui.reset_tracker()
        self.ui.mark_question(self.sc_i)
        sc = SCENARIOS[self.sc_i]
        self.ui.set_stages(CacheStory.stages if sc.get("kind") == "cache" else STAGES)
        self.ui.set_stage(None)
        if sc.get("kind") == "cache":
            self.story = CacheStory(self, sc, self.deep)
            intro = self.story.intro + t("ui.intro_cache_start")
        else:
            self.story = Story(self, sc, self.deep, self.seed)
            intro = t("ui.intro", i=self.sc_i + 1, prompt=sc["prompt"])
            self.cam_ctl.go_to(SEC_A + Point3(0, 0, 3.0), 0, 0, 24, force=True)
        self.steps = self.story.steps
        self.idx = -1
        self.ui.set_info("ui.info.start", n=len(self.steps))
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
            self.ui.set_info("ui.info.done")
            self.finished = True
            return
        self.idx += 1
        st = self.steps[self.idx]
        default_map = {"Context": ("input",), "Tokens": ("input",), "Output": ("linear", "softmax", "output"),
                       "Compute": ("attn",), "No cache": ("attn",), "Why same": ("attn",),
                       "KV cache": ("attn",)}
        self.ui.arch.highlight(*default_map.get(st.stage, ()))
        caption, ival = st.build()
        self.ui.set_stage(st.stage)
        self.ui.set_caption(caption)
        self.ui.set_info("ui.info.step", i=self.idx + 1, n=len(self.steps), stage=t("stage." + st.stage))
        self.ival = Sequence(ival, Func(self._done))
        self.ival.start(0.0, -1.0, self.speed)

    def _done(self):
        if self.auto:
            self.taskMgr.doMethodLater(2.4 / self.speed, lambda t: self.next(), "auto-next")

    # ---- toggles
    def toggle_auto(self):
        self.auto = not self.auto
        self.ui.set_btn("auto", t("ui.btn.auto_on") if self.auto else t("ui.btn.auto_off"))
        if self.auto and not (self.ival and self.ival.isPlaying()):
            self.next()
        if not self.auto:
            self.taskMgr.remove("auto-next")

    def _apply_speed(self):
        self.ui.set_btn("speed", t("ui.btn.speed", speed=self.speed))
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
        self.ui.set_btn("deep", t("ui.btn.deep_on") if self.deep else t("ui.btn.deep_off"))
        self.restart_demo()

    def toggle_follow(self):
        self.cam_ctl.follow = not self.cam_ctl.follow
        self.ui.set_btn("cam", t("ui.btn.cam_follow") if self.cam_ctl.follow else t("ui.btn.cam_free"))

    def toggle_glow(self):
        if self.glow:
            self.filters.delBloom()
            self.glow = False
        else:
            self.glow = bool(self.filters.setBloom(blend=(0.3, 0.4, 0.3, 0.0), mintrigger=0.6, maxtrigger=1.0,
                                                   desat=0.1, intensity=1.4, size="medium"))
        self.ui.set_btn("glow", t("ui.btn.glow_on") if self.glow else t("ui.btn.glow_off"))

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
        self.ui.set_btn("hide", t("ui.btn.hide") if self.panels_on else t("ui.btn.show"))

    # ---- language
    def refresh_buttons(self):
        """Every button label again, from the current state (the state itself is never touched)."""
        ui = self.ui
        ui.set_btn("next", t("ui.btn.next"))
        ui.set_btn("auto", t("ui.btn.auto_on") if self.auto else t("ui.btn.auto_off"))
        ui.set_btn("speed", t("ui.btn.speed", speed=self.speed))
        ui.set_btn("restart", t("ui.btn.restart"))
        ui.set_btn("deep", t("ui.btn.deep_on") if self.deep else t("ui.btn.deep_off"))
        ui.set_btn("cam", t("ui.btn.cam_follow") if self.cam_ctl.follow else t("ui.btn.cam_free"))
        ui.set_btn("glow", t("ui.btn.glow_on") if self.glow else t("ui.btn.glow_off"))
        ui.set_btn("hide", t("ui.btn.hide") if self.panels_on else t("ui.btn.show"))
        ui.set_btn("lang", t("ui.btn.lang"))

    def toggle_language(self):
        self.set_language("zh" if i18n.LANG == "en" else "en")

    def set_language(self, lang):
        """Switch the language on the fly and stay on the same step."""
        if lang not in i18n.LANGS:
            return
        i18n.set_lang(lang)
        if self.save_lang:
            i18n.save_lang(lang)
        self.ui.relabel()
        self.ui.arch.relabel()
        self.refresh_buttons()
        self._rebuild_scene()

    def _rebuild_scene(self):
        """The 3D texts are made when a step is built, so build the story again and fast-forward to the step we
        were on (same question, same random seed, camera on the last step's view)."""
        idx, finished, auto = self.idx, self.finished, self.auto
        cam = self.cam_ctl
        old_view = (list(cam.cur), list(cam.want))
        self.auto = False                     # no auto-next while fast-forwarding
        self.restart_demo(keep_seed=True)
        for _ in range(idx + 1):
            self.next()
            if self.ival:
                self.ival.finish()
        if finished:
            self.next()                       # shows the "Finished" message again
        self.auto = auto
        if cam.follow:
            cam.cur = list(cam.want)          # no fly-over: the camera is already where the step ends
        else:
            cam.cur, cam.want = old_view
        if auto:
            self.taskMgr.doMethodLater(1.0 if idx < 0 else 2.4 / self.speed, lambda task: self.next(), "auto-next")


def self_test(app):
    """LLMPipelineDemo.exe --selftest : play every question (fast) and report, e.g. to check a new build.
    Halfway and at the end of every question it also switches the language and back: the step, the caption and
    the random first word must be the same afterwards."""
    import traceback
    app.save_lang = False
    ok = True
    start_lang = i18n.LANG
    other = "zh" if start_lang == "en" else "en"
    for deep in (True, False):
        if not deep:
            app.toggle_deep()
        for i in range(len(SCENARIOS)):
            app.pick(i)
            try:
                n = len(app.steps)
                for k in range(n):
                    app.next()
                    if app.ival:
                        app.ival.finish()
                    app.taskMgr.step()
                    if k in (n // 2, n - 1):
                        before = (app.idx, app.ui.caption.getText(), app.ui.step_info.getText())
                        app.set_language(other)
                        assert app.idx == k and app.ui.caption.getText(), "language switch lost the step"
                        app.set_language(start_lang)
                        after = (app.idx, app.ui.caption.getText(), app.ui.step_info.getText())
                        assert before == after, "language round trip changed the screen:\n{}\n{}".format(before, after)
                print("selftest", "deep" if deep else "quick", i + 1, "ok", flush=True)
            except Exception:
                traceback.print_exc()
                ok = False
    print("SELFTEST", "PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


def _arg_lang(argv):
    """--lang zh   or   --lang=zh"""
    for k, a in enumerate(argv):
        if a == "--lang" and k + 1 < len(argv):
            return argv[k + 1]
        if a.startswith("--lang="):
            return a.split("=", 1)[1]
    return None


if __name__ == "__main__":
    lang = _arg_lang(sys.argv)
    if lang not in i18n.LANGS:                # no (valid) --lang: the saved choice, default English
        lang = "en" if "--selftest" in sys.argv else i18n.load_lang()
    app = App(lang)
    if "--selftest" in sys.argv:
        self_test(app)
    app.run()
