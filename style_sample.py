# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
STYLE SAMPLE v2 (3Blue1Brown-like look), one sentence, ten steps:

  1 sentence            2 tokens              3 vectors (columns of numbers)
  4 meaning space       5 big picture: Attention block -> Multilayer Perceptron
  6 Q / K table         7 scores -> attention weights
  8 weighted sum of V = delta E                9 the 'ature' arrow is pushed in 3D
 10 MLP + "repeat for many layers"

Run:  python style_sample.py
Keys: Space = next step (skips animation if still playing)   A = auto   R = restart
      B = glow on/off   Right-drag = rotate camera   Wheel = zoom
"""
import math
import os
import random
import sys

from panda3d.core import loadPrcFileData

loadPrcFileData("", """
window-title LLM Style Sample
win-size 1600 900
framebuffer-multisample 1
multisamples 8
textures-power-2 none
audio-library-name null
""")
if os.environ.get("LLM_DEMO_OFFSCREEN"):
    loadPrcFileData("", "window-type offscreen\n")

from direct.filter.CommonFilters import CommonFilters                      # noqa: E402
from direct.gui.OnscreenText import OnscreenText                           # noqa: E402
from direct.interval.IntervalGlobal import (Func, LerpColorScaleInterval,  # noqa: E402
                                            LerpFunc, LerpPosInterval, LerpScaleInterval,
                                            Parallel, Sequence, Wait)
from direct.showbase.ShowBase import ShowBase                              # noqa: E402
from panda3d.core import (AmbientLight, AntialiasAttrib, CardMaker,        # noqa: E402
                          DirectionalLight, Filename, Geom, GeomNode, GeomTriangles,
                          GeomVertexData, GeomVertexFormat, GeomVertexWriter, LineSegs,
                          Point3, TextNode, TransparencyAttrib, Vec3, Vec4)

import sim                                                                 # noqa: E402

# ------------------------------------------------------------------ palette (3B1B-like)
WHITE = (1, 1, 1, 1)
GREY = (0.55, 0.55, 0.55, 1)
DIM = (0.32, 0.32, 0.34, 1)
BLUE = (0.35, 0.77, 0.87, 1)
TEAL = (0.36, 0.81, 0.72, 1)
GREEN = (0.51, 0.76, 0.40, 1)
YELLOW = (1.0, 0.93, 0.35, 1)
GOLD = (0.94, 0.68, 0.30, 1)
RED = (0.99, 0.38, 0.33, 1)
PINK = (0.93, 0.47, 0.66, 1)
PURPLE = (0.60, 0.50, 0.95, 1)
EDGE_RED = (0.93, 0.42, 0.38, 1)
EDGE_BLUE = (0.33, 0.72, 0.88, 1)
TOKEN_COLORS = [BLUE, YELLOW, GREEN, RED, PINK, TEAL, GOLD, PURPLE]

SERIF = ["C:/Windows/Fonts/cambria.ttc", "C:/Windows/Fonts/times.ttf",
         "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
         "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]
ITALIC = ["C:/Windows/Fonts/timesi.ttf", "C:/Windows/Fonts/cambriai.ttf",
          "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
          "/usr/share/fonts/truetype/crosextra/Caladea-Italic.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"]

SYMBOL = ["C:/Windows/Fonts/cambria.ttc", "C:/Windows/Fonts/times.ttf",
          "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]          # must contain the Greek Delta

SENTENCE = "Plot the tank temperature for the last hour."
FOCUS_TOKEN = "ature"        # the token whose column we zoom into

# meaning groups -> where the arrows point in 3D, and what attends to what
GROUP = {"Plot": "action", " the": "func", " tank": "equip", " temper": "equip", "ature": "equip",
         " for": "func", " last": "time", " hour": "time", ".": "func"}
SPACE_POS = {"Plot": (0.6, -3.4, 0.9), " the": (-2.4, -1.9, -0.3), " tank": (3.3, 1.4, 0.3),
             " temper": (3.5, 0.4, 1.3), "ature": (-0.4, -0.9, 3.3), " for": (-3.2, -0.6, -0.9),
             " last": (-1.9, 2.4, 2.0), " hour": (-0.9, 3.0, 2.8), ".": (-3.4, 1.0, 0.9)}

# board sections (everything lives on one big black board, the camera flies between them)
SEC_A = Point3(0, 0, 0)          # tokens + vectors; the big-picture model grows behind it (+y)
SEC_B = Point3(30, 6, -0.5)      # 3D meaning space
SEC_C = Point3(56, 0, 0)         # Q/K table
SEC_E = Point3(58, 0, -12.5)     # weighted sum -> delta E (below the table)


# ------------------------------------------------------------------ small geometry kit
def load_font(base, paths):
    for p in paths:
        if os.path.exists(p):
            f = base.loader.loadFont(Filename.fromOsSpecific(p).getFullpath())
            if f and f.isValid():
                f.setPixelsPerUnit(80)
                return f
    return None


def text(parent, s, pos, scale, color=WHITE, font=None, align=TextNode.ACenter):
    tn = TextNode("t")
    tn.setText(s)
    tn.setAlign(align)
    tn.setTextColor(*color)
    if font:
        tn.setFont(font)
    np = parent.attachNewNode(tn)
    np.setPos(pos)
    np.setScale(scale)
    np.setY(np.getY() - 0.01)
    return np


def text_width(font, s):
    tn = TextNode("m")
    if font:
        tn.setFont(font)
    tn.setText(s)
    return tn.getWidth()


def lines(parent, pts_list, color, thick=2.0):
    ls = LineSegs()
    ls.setThickness(thick)
    ls.setColor(*color)
    for pts in pts_list:
        ls.moveTo(*pts[0])
        for p in pts[1:]:
            ls.drawTo(*p)
    return parent.attachNewNode(ls.create())


def rect(parent, x0, z0, x1, z1, color, thick=2.0, y=0):
    return lines(parent, [[(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), (x0, y, z0)]], color, thick)


def fill(parent, x0, z0, x1, z1, color, alpha=0.18, y=0.02):
    cm = CardMaker("fill")
    cm.setFrame(x0, x1, z0, z1)
    np = parent.attachNewNode(cm.generate())
    np.setY(y)
    np.setColor(color[0], color[1], color[2], alpha)
    np.setTransparency(TransparencyAttrib.MAlpha)
    np.setDepthWrite(False)
    return np


def _geom_node(name, vd, tris):
    g = Geom(vd)
    g.addPrimitive(tris)
    gn = GeomNode(name)
    gn.addGeom(g)
    return gn


def disc(parent, r, color, segs=40):
    vd = GeomVertexData("disc", GeomVertexFormat.getV3(), Geom.UHStatic)
    vw = GeomVertexWriter(vd, "vertex")
    vw.addData3(0, 0, 0)
    for i in range(segs + 1):
        a = 2 * math.pi * i / segs
        vw.addData3(r * math.cos(a), 0, r * math.sin(a))
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(1, segs + 1):
        tris.addVertices(0, i + 1, i)
    np = parent.attachNewNode(_geom_node("disc", vd, tris))
    np.setColor(*color)
    np.setTwoSided(True)
    return np


def triangle(parent, p, direction, size, color):
    """Flat arrow head in the XZ plane at point p, pointing along direction (dx, dz)."""
    dx, dz = direction
    L = math.hypot(dx, dz) or 1
    dx, dz = dx / L, dz / L
    px, pz = -dz, dx
    vd = GeomVertexData("tri", GeomVertexFormat.getV3(), Geom.UHStatic)
    vw = GeomVertexWriter(vd, "vertex")
    vw.addData3(p[0], p[1], p[2])
    vw.addData3(p[0] - dx * size + px * size * 0.5, p[1], p[2] - dz * size + pz * size * 0.5)
    vw.addData3(p[0] - dx * size - px * size * 0.5, p[1], p[2] - dz * size - pz * size * 0.5)
    tris = GeomTriangles(Geom.UHStatic)
    tris.addVertices(0, 1, 2)
    np = parent.attachNewNode(_geom_node("tri", vd, tris))
    np.setColor(*color)
    np.setTwoSided(True)
    return np


def arrow2d(parent, p0, p1, color, thick=2.0, head=0.16):
    """Flat arrow in the board plane (XZ)."""
    g = parent.attachNewNode("arrow2d")
    d = Vec3(*p1) - Vec3(*p0)
    L = d.length() or 1
    tip_base = Point3(*p1) - d / L * head * 0.9
    lines(g, [[p0, tip_base]], color, thick)
    triangle(g, p1, (d.x, d.z), head, color)
    return g


def cone(parent, r, h, color, segs=24):
    """Cone pointing along +Y, base at y=0."""
    vd = GeomVertexData("cone", GeomVertexFormat.getV3(), Geom.UHStatic)
    vw = GeomVertexWriter(vd, "vertex")
    vw.addData3(0, h, 0)
    vw.addData3(0, 0, 0)
    for i in range(segs + 1):
        a = 2 * math.pi * i / segs
        vw.addData3(r * math.cos(a), 0, r * math.sin(a))
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(2, segs + 2):
        tris.addVertices(0, i, i + 1)
        tris.addVertices(1, i + 1, i)
    np = parent.attachNewNode(_geom_node("cone", vd, tris))
    np.setColor(*color)
    np.setTwoSided(True)
    return np


def sphere(parent, r, color, rings=8, segs=14):
    vd = GeomVertexData("sph", GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vw = GeomVertexWriter(vd, "vertex")
    nw = GeomVertexWriter(vd, "normal")
    for i in range(rings + 1):
        th = math.pi * i / rings
        for j in range(segs + 1):
            ph = 2 * math.pi * j / segs
            n = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
            vw.addData3(n[0] * r, n[1] * r, n[2] * r)
            nw.addData3(*n)
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(rings):
        for j in range(segs):
            a = i * (segs + 1) + j
            b = a + segs + 1
            tris.addVertices(a, b, a + 1)
            tris.addVertices(a + 1, b, b + 1)
    np = parent.attachNewNode(_geom_node("sphere", vd, tris))
    np.setColor(*color)
    np.setTwoSided(True)
    return np


def slab(parent, x0, x1, y0, y1, z0, z1, color=(0.75, 0.75, 0.78), alpha=0.16):
    """Translucent box with faint white edges (the 'Attention' block)."""
    g = parent.attachNewNode("slab")
    vd = GeomVertexData("slab", GeomVertexFormat.getV3(), Geom.UHStatic)
    vw = GeomVertexWriter(vd, "vertex")
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for v in c:
        vw.addData3(*v)
    tris = GeomTriangles(Geom.UHStatic)
    for a, b, cc, d in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7), (0, 3, 2, 1)]:
        tris.addVertices(a, b, cc)
        tris.addVertices(a, cc, d)
    body = g.attachNewNode(_geom_node("slab", vd, tris))
    body.setColor(color[0], color[1], color[2], alpha)
    body.setTransparency(TransparencyAttrib.MAlpha)
    body.setTwoSided(True)
    body.setDepthWrite(False)
    body.setBin("transparent", 0)
    e = [[c[0], c[1], c[2], c[3], c[0]], [c[4], c[5], c[6], c[7], c[4]],
         [c[0], c[4]], [c[1], c[5]], [c[2], c[6]], [c[3], c[7]]]
    edges = lines(g, e, (0.8, 0.8, 0.85, 0.45), 1.4)
    edges.setTransparency(TransparencyAttrib.MAlpha)
    return g


class Arrow:
    """A 3B1B-style vector: thin shaft + cone tip. grow(t) animates 0..1."""

    def __init__(self, parent, start, end, color, thick=3.0):
        self.root = parent.attachNewNode("arrow")
        self.start, self.end = Point3(start), Point3(end)
        self.color, self.thick = color, thick
        self.shaft = None
        self.tip = cone(self.root, 0.12, 0.35, color)
        self.grow(0.001)

    def grow(self, t):
        if self.shaft:
            self.shaft.removeNode()
        d = self.end - self.start
        L = d.length()
        p = self.start + d * t
        back = d.normalized() * min(0.3, L * t) if L > 0 else Vec3(0)
        self.shaft = lines(self.root, [[self.start, p - back]], self.color, self.thick)
        self.tip.setPos(p - back)
        self.tip.lookAt(self.root, p + d)
        self.tip.setScale(min(1.0, 3 * t + 0.01))


def fade_in(np, dur=0.5):
    np.setTransparency(TransparencyAttrib.MAlpha)
    np.setColorScale(1, 1, 1, 0)
    return LerpColorScaleInterval(np, dur, (1, 1, 1, 1), startColorScale=(1, 1, 1, 0))


def fade_out(np, dur=0.5):
    np.setTransparency(TransparencyAttrib.MAlpha)
    return LerpColorScaleInterval(np, dur, (1, 1, 1, 0))


def num(v):
    s = "{:+.1f}".format(round(v, 1) + 0.0)
    return " 0.0" if s in ("+0.0", "-0.0") else s


# ------------------------------------------------------------------ the sample
class StyleSample(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.setBackgroundColor(0, 0, 0, 1)
        self.render.setAntialias(AntialiasAttrib.MMultisample)
        self.disableMouse()
        self.camLens.setFov(52)
        self.camLens.setNearFar(0.5, 400)
        self.serif = load_font(self, SERIF)
        self.italic = load_font(self, ITALIC) or self.serif
        self.symbol = load_font(self, SYMBOL) or self.serif
        self.filters = CommonFilters(self.win, self.cam)
        self.glow = False
        self.toggle_glow()
        self.cam_cur = [SEC_A.x, SEC_A.y, SEC_A.z + 0.5, 0.0, 0.0, 24.0]
        self.cam_want = list(self.cam_cur)
        self.drag = None
        self.taskMgr.add(self._cam_task, "cam")

        self.caption = OnscreenText("", pos=(0, -0.86), scale=0.046, fg=WHITE, bg=(0, 0, 0, 0.6),
                                    font=self.symbol, wordwrap=42, mayChange=True)
        self.info = OnscreenText("", parent=self.a2dTopLeft, pos=(0.05, -0.08), scale=0.035,
                                 fg=GREY, align=TextNode.ALeft, font=self.serif, mayChange=True)
        OnscreenText("Space: next   A: auto   R: restart   B: glow   right-drag: rotate   wheel: zoom",
                     parent=self.a2dBottomRight, pos=(-0.05, 0.04), scale=0.028, fg=(0.4, 0.4, 0.4, 1),
                     align=TextNode.ARight, font=self.serif)
        for k, fn in [("space", self.next), ("arrow_right", self.next), ("a", self.toggle_auto),
                      ("r", self.restart_sample), ("b", self.toggle_glow), ("escape", sys.exit),
                      ("mouse3", self._drag_on), ("mouse3-up", self._drag_off)]:
            self.accept(k, fn)
        self.accept("wheel_up", self._zoom, [0.9])
        self.accept("wheel_down", self._zoom, [1.1])
        self.auto = False
        self.ival = None
        self.restart_sample()

    # ---------------------------------------------------------- camera
    def view(self, target, h=0.0, p=0.0, d=24.0):
        self.cam_want = [target[0], target[1], target[2], h, p, d]

    def _zoom(self, f):
        self.cam_want[5] = max(4, min(120, self.cam_want[5] * f))

    def _drag_on(self):
        if self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            self.drag = (m.x, m.y)

    def _drag_off(self):
        self.drag = None

    def _cam_task(self, task):
        mw = self.mouseWatcherNode
        if self.drag and mw.hasMouse():
            m = mw.getMouse()
            self.cam_want[3] -= (m.x - self.drag[0]) * 100
            self.cam_want[4] = max(-80, min(85, self.cam_want[4] - (m.y - self.drag[1]) * 70))
            self.drag = (m.x, m.y)
        k = 1 - math.exp(-min(self.clock.getDt(), 0.1) * 2.4)
        for i in range(6):
            self.cam_cur[i] += (self.cam_want[i] - self.cam_cur[i]) * k
        x, y, z, h, p, d = self.cam_cur
        hr, pr = math.radians(h), math.radians(p)
        self.camera.setPos(x + d * math.cos(pr) * math.sin(hr), y - d * math.cos(pr) * math.cos(hr),
                           z + d * math.sin(pr))
        self.camera.lookAt(Point3(x, y, z))
        return task.cont

    def orbit(self, target, h0, h1, p, d, dur):
        return LerpFunc(lambda h: self.view(target, h, p, d), fromData=h0, toData=h1,
                        duration=dur, blendType="easeInOut")

    # ---------------------------------------------------------- step player
    def restart_sample(self):
        if self.ival:
            self.ival.finish()
            self.ival = None
        self.taskMgr.remove("auto")
        if hasattr(self, "board"):
            self.board.removeNode()
        self.board = self.render.attachNewNode("board")
        self.board.setLightOff()
        self.toks = sim.tokenize(SENTENCE)
        self.fq = self.toks.index(FOCUS_TOKEN)
        self.W = self._weights()
        self.idx = -1
        self.steps = [self.s_sentence, self.s_tokens, self.s_vectors, self.s_space, self.s_big_picture,
                      self.s_qk_table, self.s_weights, self.s_delta, self.s_update, self.s_mlp]
        self.view(SEC_A + Vec3(0, 0, 0.5), 0, 0, 24)
        self.caption.setText("Press Space to start")
        self.info.setText("STYLE SAMPLE  -  0 / {}".format(len(self.steps)))

    def next(self):
        if self.ival and self.ival.isPlaying():
            self.ival.finish()
            return
        if self.idx + 1 >= len(self.steps):
            return
        self.idx += 1
        caption, ival = self.steps[self.idx]()
        self.caption.setText(caption)
        self.info.setText("STYLE SAMPLE  -  {} / {}".format(self.idx + 1, len(self.steps)))
        self.ival = Sequence(ival, Func(self._done))
        self.ival.start()

    def _done(self):
        if self.auto:
            self.taskMgr.doMethodLater(2.2, lambda t: self.next(), "auto")

    def toggle_auto(self):
        self.auto = not self.auto
        if self.auto:
            self.next()
        else:
            self.taskMgr.remove("auto")

    def toggle_glow(self):
        if self.glow:
            self.filters.delBloom()
            self.glow = False
        else:
            self.glow = bool(self.filters.setBloom(blend=(0.3, 0.4, 0.3, 0.0), mintrigger=0.6,
                                                   maxtrigger=1.0, desat=0.1, intensity=1.4, size="medium"))

    # ---------------------------------------------------------- shared pieces
    def word(self, t):
        return t.strip() or t

    def token_box(self, parent, t, color, center, scale=0.34):
        """Small 3B1B token box: text with a thin coloured frame."""
        g = parent.attachNewNode("tokbox")
        w = text_width(self.serif, self.word(t)) * scale + 0.22
        h = scale * 1.25
        x, y, z = center
        fill(g, x - w / 2, z - h * 0.32, x + w / 2, z + h * 0.68, color, 0.15, y + 0.02)
        rect(g, x - w / 2, z - h * 0.32, x + w / 2, z + h * 0.68, color, 1.6, y)
        text(g, self.word(t), Point3(x, y, z), scale, WHITE, self.serif)
        return g, w

    def vlabel(self, parent, letter, sub, pos, scale, color, align="center", arrow=True):
        """A vector symbol: letter with a little arrow above and a subscript (e.g. E-arrow, 3)."""
        g = parent.attachNewNode("vlabel")
        font = self.symbol if "\u0394" in letter else self.serif
        wl = text_width(font, letter) * scale
        ws = text_width(self.serif, str(sub)) * scale * 0.55 if sub != "" else 0
        total = wl + ws + 0.02
        x0 = pos[0] - total / 2 if align == "center" else (pos[0] if align == "left" else pos[0] - total)
        y, z = pos[1], pos[2]
        text(g, letter, Point3(x0, y, z), scale, color, font, TextNode.ALeft)
        if sub != "":
            text(g, str(sub), Point3(x0 + wl + 0.02, y, z - scale * 0.22), scale * 0.55, color, self.serif,
                 TextNode.ALeft)
        if arrow:
            az = z + scale * 0.88
            ax0, ax1 = x0 + wl * 0.1, x0 + wl * 0.95
            lines(g, [[(ax0, y - 0.01, az), (ax1, y - 0.01, az)],
                      [(ax1 - scale * 0.16, y - 0.01, az + scale * 0.09), (ax1, y - 0.01, az),
                       (ax1 - scale * 0.16, y - 0.01, az - scale * 0.09)]], color, 1.4)
        return g, total

    def column(self, parent, x, z_top, vals, color=WHITE, scale=0.34, dz=0.48, y=0.0, bracket=GREY):
        col = parent.attachNewNode("col")
        k = scale / 0.34
        for r, v in enumerate(vals):
            text(col, num(v), Point3(x, y, z_top - r * dz), scale, color, self.serif)
        n = len(vals)
        for j in range(3):
            d = disc(col, 0.032 * k, color)
            d.setPos(x, y - 0.01, z_top - n * dz - j * 0.16 * k + 0.12)
        bw = 0.5 * k
        bz0, bz1 = z_top - n * dz - 0.35 * k, z_top + 0.4 * k
        for s in (-1, 1):
            bx = x + s * bw
            lines(col, [[(bx - s * 0.12, y, bz1), (bx, y, bz1), (bx, y, bz0), (bx - s * 0.12, y, bz0)]],
                  bracket, 1.6)
        return col

    def _weights(self):
        n = len(self.toks)
        rng = random.Random(11)
        W = [[0.0] * n for _ in range(n)]          # W[key][query]
        S = [[None] * n for _ in range(n)]         # raw scores (before softmax)
        for q in range(n):
            scores = []
            for k in range(q + 1):
                s = rng.uniform(0, 0.8)
                if GROUP.get(self.toks[k]) == GROUP.get(self.toks[q]) and GROUP.get(self.toks[q]) != "func":
                    s += 2.2
                if self.toks[q] == ".":
                    s += 1.5 if GROUP.get(self.toks[k]) in ("action", "equip", "time") else 0
                scores.append(s)
            p = sim.softmax(scores, 0.7)
            for k in range(q + 1):
                W[k][q] = p[k]
                S[k][q] = scores[k] / 0.7 - 1.5
        self.S = S
        return W

    # ---------------------------------------------------------- 1 sentence
    def s_sentence(self):
        self.sentence = text(self.board, SENTENCE, SEC_A + Vec3(0, 0, 4.2), 0.95, WHITE, self.serif)
        return ("Everything starts as plain text.", fade_in(self.sentence, 1.2))

    # ---------------------------------------------------------- 2 tokens
    def s_tokens(self):
        sc = 0.62
        widths = [text_width(self.serif, self.word(t)) * sc + 0.45 for t in self.toks]
        gap = 0.18
        total = sum(widths) + gap * (len(widths) - 1)
        x = SEC_A.x - total / 2
        self.tok_x = []
        for w in widths:
            self.tok_x.append(x + w / 2)
            x += w + gap
        self.tok_color = [TOKEN_COLORS[i % len(TOKEN_COLORS)] for i in range(len(self.toks))]
        seq = Parallel(Sequence(fade_out(self.sentence, 0.5), Func(self.sentence.hide)))
        z = SEC_A.z + 2.0
        for i, (t, w) in enumerate(zip(self.toks, widths)):
            g = self.board.attachNewNode("tok")
            c, x = self.tok_color[i], self.tok_x[i]
            fill(g, x - w / 2, z - 0.35, x + w / 2, z + 0.62, c)
            rect(g, x - w / 2, z - 0.35, x + w / 2, z + 0.62, c, 2.2)
            text(g, self.word(t), Point3(x, 0, z), sc, WHITE, self.serif)
            text(g, str(sim.token_id(t)), Point3(x, 0, z - 0.85), 0.32, GREY, self.serif)
            seq.append(Sequence(Wait(0.3 + 0.12 * i), fade_in(g, 0.4)))
        self.tok_z = z
        return ("The tokenizer cuts the text into tokens. Each token has an ID number. "
                "Notice: 'temperature' becomes two tokens.", seq)

    # ---------------------------------------------------------- 3 vectors
    def s_vectors(self):
        self.columns = []
        par = Parallel()
        for i, t in enumerate(self.toks):
            x = self.tok_x[i]
            top = self.tok_z - 1.5
            col = self.column(self.board, x, top, sim.embedding(t, 6), WHITE, 0.34)
            lines(col, [[(x, 0, self.tok_z - 0.95), (x, 0, top + 0.5)]], DIM, 1.2)
            self.columns.append(col)
            par.append(Sequence(Wait(0.12 * i), fade_in(col, 0.5),
                                LerpPosInterval(col, 0.5, Point3(0, 0, 0), startPos=Point3(0, 0, 0.6),
                                                blendType="easeOut")))
        return ("Each token ID is looked up in a giant table and becomes a long list of numbers - "
                "a VECTOR (6 shown, real models use thousands).",
                Sequence(Func(self.view, SEC_A + Vec3(0, 0, -0.3), 0, 0, 22), par))

    # ---------------------------------------------------------- 4 3D meaning space
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
        for i, t in enumerate(self.toks):
            end = Point3(*SPACE_POS.get(t, (0, 0, 1)))
            a = Arrow(sp, Point3(0, 0, 0), end, self.tok_color[i], 3.0)
            self.arrows.setdefault(t, a)
            grow = LerpFunc(a.grow, fromData=0.001, toData=1.0, duration=0.9, blendType="easeOut")
            steps = [Wait(1.2 + 0.25 * i), grow]
            if t not in seen:
                seen.add(t)
                lab = text(sp, self.word(t), end * 1.13 + Vec3(0, 0, 0.12), 0.34, self.tok_color[i], self.serif)
                lab.setBillboardPointEye()
                lab.hide()
                if t == FOCUS_TOKEN:
                    self.focus_label = lab
                steps.append(Func(lab.show))
                steps.append(fade_in(lab, 0.3))
            flights.append(Sequence(*steps))
        return ("Think of each vector as an ARROW in space (squeezed into 3D here). Words with related "
                "meaning point in similar directions: tank / temper are close, so are last / hour. "
                "'ature' on its own means almost nothing - it points somewhere vague.",
                Sequence(Func(self.view, SEC_B + Vec3(0, 0, 0.8), -35, 24, 16),
                         Parallel(par, flights, self.orbit(SEC_B + Vec3(0, 0, 0.8), -35, 30, 24, 16, 7.0))))

    # ---------------------------------------------------------- 5 big picture
    def s_big_picture(self):
        xs = self.tok_x
        x0, x1 = min(xs) - 0.9, max(xs) + 0.9
        top = self.tok_z - 1.5
        zt, zb = top + 0.9, top - 4.0
        xc = (x0 + x1) / 2
        bp = self.board.attachNewNode("bigpicture")
        self.bp = bp
        att = slab(bp, x0, x1, 1.6, 6.0, zb, zt)
        att_lab = text(bp, "Attention", Point3(xc, 3.8, zt + 0.6), 1.1, WHITE, self.serif)
        att_lab.setBillboardAxis()
        # after attention: the same columns, numbers changed by context
        moves = Parallel()
        for i, t in enumerate(self.toks):
            vals = [v + 0.6 * d for v, d in zip(sim.embedding(t, 6), sim.embedding("ctx" + t, 6))]
            c = self.column(bp, xs[i], top, vals, WHITE, 0.34)
            c.hide()
            moves.append(Sequence(Wait(0.07 * i), Func(c.show),
                                  LerpPosInterval(c, 2.0, Point3(0, 8.0, 0), startPos=Point3(0, 0, 0),
                                                  blendType="easeInOut")))
        # the multilayer perceptron: shaded spheres + red/blue connections
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
        for ly, n in ((10.5, 26), (13.0, 34), (15.5, 26)):
            layers.append([Point3(rng.uniform(x0 + 0.4, x1 - 0.4), ly + rng.uniform(-0.5, 0.5),
                                  rng.uniform(zb + 0.4, zt - 0.4)) for _ in range(n)])
        self.mlp_edges = []
        for a, b in zip(layers, layers[1:]):
            reds, blues = [], []
            for p in a:
                for q in rng.sample(b, 4):
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
            self.mlp_nodes.append(g)
        mlp_lab = text(bp, "Multilayer\nPerceptron", Point3(xc, 13.0, zt + 1.6), 0.95, WHITE, self.serif)
        mlp_lab.setBillboardAxis()
        self.mlp_center = Point3(xc, 13.0, (zt + zb) / 2)
        # output of the MLP: columns again, further back
        holder = bp.attachNewNode("cols3")
        holder.setY(8.0)
        out = Parallel()
        for i, t in enumerate(self.toks):
            vals = [v + 0.9 * d for v, d in zip(sim.embedding(t, 6), sim.embedding("mlp" + t, 6))]
            c = self.column(holder, xs[i], top, vals, WHITE, 0.34)
            c.hide()
            out.append(Sequence(Wait(0.06 * i), Func(c.show),
                                LerpPosInterval(c, 2.4, Point3(0, 10.5, 0), startPos=Point3(0, 0, 0),
                                                blendType="easeInOut")))
        dots = text(bp, ". . .", Point3(xc, 22.0, (zt + zb) / 2), 1.2, GREY, self.serif)
        dots.setBillboardAxis()
        rep = text(bp, "x 32 layers", Point3(xc, 22.0, zt + 0.6), 0.8, YELLOW, self.serif)
        rep.setBillboardAxis()
        for nd in (att, att_lab, mlp_lab, dots, rep, mlp):
            nd.hide()
        center = Point3(xc, 10.5, (zt + zb) / 2 + 0.5)
        seq = Sequence(
            Func(self.view, center, 62, 16, 33),
            Parallel(
                self.orbit(center, 62, 44, 16, 33, 11.0),
                Sequence(Func(att.show), fade_in(att, 0.8), Func(att_lab.show), fade_in(att_lab, 0.5),
                         moves,
                         Func(mlp.show), Parallel(*[Sequence(Wait(0.25 * k), fade_in(nd, 0.5))
                                                    for k, nd in enumerate(self.mlp_nodes + self.mlp_edges)]),
                         Func(mlp_lab.show), fade_in(mlp_lab, 0.5),
                         Parallel(out, self.pulse(0.3)),
                         Func(dots.show), Func(rep.show), Parallel(fade_in(dots, 0.5), fade_in(rep, 0.5)))))
        return ("The big picture: all the vectors flow through an ATTENTION block (words exchange "
                "information), then a MULTILAYER PERCEPTRON (each vector is processed on its own). "
                "That pair is repeated many times.", seq)

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

    # ---------------------------------------------------------- 6 Q / K table
    def s_qk_table(self):
        n = len(self.toks)
        cw, ch = 1.3, 0.8
        g = self.board.attachNewNode("qk")
        g.setPos(SEC_C)
        self.qk = g
        gx0 = -n * cw / 2 + 2.6            # grid left edge (room for the row headers)
        gz0 = 2.6                          # grid top edge
        self.cw, self.ch, self.gx0, self.gz0 = cw, ch, gx0, gz0
        sep = lines(g, [[(gx0 - 6.3, 0, gz0), (gx0 + n * cw, 0, gz0)],
                        [(gx0, 0, gz0 + 3.6), (gx0, 0, gz0 - n * ch)]], GREY, 1.6)
        cols = []
        for q, t in enumerate(self.toks):        # column headers: token -> E -> (W_Q) -> Q
            cx = gx0 + (q + 0.5) * cw
            h = g.attachNewNode("colhead")
            self.token_box(h, t, self.tok_color[q], (cx, 0, gz0 + 3.15), 0.26)
            arrow2d(h, (cx, 0, gz0 + 2.95), (cx, 0, gz0 + 2.55), WHITE, 1.4, 0.12)
            self.vlabel(h, "E", q + 1, (cx, 0, gz0 + 2.1), 0.36, WHITE)
            arrow2d(h, (cx, 0, gz0 + 1.9), (cx, 0, gz0 + 1.2), WHITE, 1.4, 0.12)
            self.vlabel(h, "W", "Q", (cx + 0.32, 0, gz0 + 1.5), 0.22, YELLOW, arrow=False)
            self.vlabel(h, "Q", q + 1, (cx, 0, gz0 + 0.65), 0.38, YELLOW)
            cols.append(h)
        rows = []
        for k, t in enumerate(self.toks):        # row headers: token -> E -> (W_K) -> K
            cz = gz0 - (k + 0.5) * ch - 0.1
            h = g.attachNewNode("rowhead")
            self.token_box(h, t, self.tok_color[k], (gx0 - 5.4, 0, cz), 0.26)
            arrow2d(h, (gx0 - 4.75, 0, cz + 0.1), (gx0 - 4.25, 0, cz + 0.1), WHITE, 1.4, 0.12)
            self.vlabel(h, "E", k + 1, (gx0 - 3.85, 0, cz), 0.32, WHITE)
            arrow2d(h, (gx0 - 3.35, 0, cz + 0.1), (gx0 - 1.75, 0, cz + 0.1), WHITE, 1.4, 0.12)
            self.vlabel(h, "W", "K", (gx0 - 2.55, 0, cz + 0.22), 0.2, GREEN, arrow=False)
            self.vlabel(h, "K", k + 1, (gx0 - 1.15, 0, cz), 0.34, GREEN)
            rows.append(h)
        # cells: K_k . Q_q   (only where the key comes before or at the query)
        self.cells = {}
        cells_anim = Parallel()
        for q in range(n):
            for k in range(q + 1):
                cx = gx0 + (q + 0.5) * cw
                cz = gz0 - (k + 0.5) * ch - 0.08
                c = g.attachNewNode("cell")
                self.vlabel(c, "K", k + 1, (cx - 0.3, 0, cz), 0.24, GREEN)
                text(c, "\u00b7", Point3(cx, 0, cz), 0.3, WHITE, self.serif)
                self.vlabel(c, "Q", q + 1, (cx + 0.3, 0, cz), 0.24, YELLOW)
                c.hide()
                self.cells[(k, q)] = c
                cells_anim.append(Sequence(Wait(2.4 + 0.12 * q + 0.04 * k), Func(c.show), fade_in(c, 0.35)))
        seq = Sequence(
            Func(self.view, SEC_C + Vec3(-1.8, 0, -0.4), 0, 0, 25),
            Parallel(fade_in(sep, 0.6),
                     Parallel(*[Sequence(Wait(0.08 * i), fade_in(h, 0.5)) for i, h in enumerate(cols)]),
                     Parallel(*[Sequence(Wait(0.8 + 0.08 * i), fade_in(h, 0.5)) for i, h in enumerate(rows)]),
                     cells_anim))
        return ("Zoom into the attention block. Each vector E is multiplied by a matrix W_Q to get a QUERY Q "
                "('what am I looking for?') and by W_K to get a KEY K ('what do I contain?'). "
                "Every cell compares a key with a query: K . Q", seq)

    # ---------------------------------------------------------- 7 scores -> weights
    def s_weights(self):
        n = len(self.toks)
        g, cw, ch, gx0, gz0 = self.qk, self.cw, self.ch, self.gx0, self.gz0
        to_scores = Parallel()
        to_weights = Parallel()
        self.wtext = {}
        for (k, q), c in self.cells.items():
            cx = gx0 + (q + 0.5) * cw
            cz = gz0 - (k + 0.5) * ch - 0.08
            s = text(g, num(self.S[k][q]), Point3(cx, 0, cz), 0.3, GREY, self.serif)
            s.hide()
            w = self.W[k][q]
            b = 0.25 + 0.75 * min(1.0, w * 1.6)
            bg = disc(g, 0.3, (b * 0.55, b * 0.55, b * 0.55, 1))
            bg.setPos(cx, 0.03, cz + 0.08)
            bg.setScale(0.001)
            wt = text(g, "{:.2f}".format(w), Point3(cx, -0.02, cz), 0.3, WHITE if w > 0.15 else GREY, self.serif)
            wt.hide()
            self.wtext[(k, q)] = wt
            to_scores.append(Sequence(fade_out(c, 0.3), Func(c.hide), Func(s.show), fade_in(s, 0.3)))
            to_weights.append(Sequence(fade_out(s, 0.3), Func(s.hide), Func(wt.show),
                                       Parallel(fade_in(wt, 0.4),
                                                LerpScaleInterval(bg, 0.5, 1.0, startScale=0.001))))
        for q in range(n):                          # cells a token may not look at: 0.00
            for k in range(q + 1, n):
                cx = gx0 + (q + 0.5) * cw
                cz = gz0 - (k + 0.5) * ch - 0.08
                z = text(g, "0.00", Point3(cx, 0, cz), 0.26, (0.25, 0.25, 0.27, 1), self.serif)
                z.hide()
                to_weights.append(Sequence(Func(z.show), fade_in(z, 0.4)))
        cx = gx0 + (self.fq + 0.5) * cw
        hl = rect(g, cx - cw / 2 + 0.04, gz0 - n * ch, cx + cw / 2 - 0.04, gz0 + 0.95, YELLOW, 3.0, y=-0.04)
        hl.hide()
        total = text(g, "sum = 1.00", Point3(cx, 0, gz0 - n * ch - 0.55), 0.3, YELLOW, self.serif)
        total.hide()
        soft = text(g, "softmax: each column -> weights that add up to 1",
                    Point3(gx0 + n * cw / 2, 0, gz0 - n * ch - 1.25), 0.34, GREY, self.serif)
        soft.hide()
        seq = Sequence(to_scores, Wait(0.6),
                       Func(soft.show), fade_in(soft, 0.4), to_weights, Wait(0.3),
                       Func(self.view, SEC_C + Vec3(1.2, 0, -0.2), -24, 6, 27),
                       Func(hl.show), fade_in(hl, 0.5), Func(total.show), fade_in(total, 0.4))
        return ("Each K . Q gives a score (bigger = better match). Then every column is normalized "
                "(softmax) into weights between 0 and 1. Look at the 'ature' column: its weight goes to "
                "'temper', itself, and 'tank'.", seq)

    # ---------------------------------------------------------- 8 weighted sum of values
    def s_delta(self):
        q = self.fq
        ks = [k for k in range(q + 1) if self.W[k][q] > 0.1]
        e = self.board.attachNewNode("delta")
        e.setPos(SEC_E)
        n = len(ks)
        gap = 3.2
        x_start = -(n * gap) / 2 - 1.0
        vals_sum = [0.0] * 6
        flights = Parallel()
        parts = Parallel()
        for j, k in enumerate(ks):
            t = self.toks[k]
            w = self.W[k][q]
            V = sim.embedding("V" + t, 6)
            vals_sum = [a + w * b for a, b in zip(vals_sum, V)]
            x = x_start + j * gap
            grp = e.attachNewNode("term")
            self.token_box(grp, t, self.tok_color[k], (x + 0.5, 0, 3.0), 0.26)
            arrow2d(grp, (x + 0.5, 0, 2.8), (x + 0.5, 0, 2.35), WHITE, 1.4, 0.12)
            self.vlabel(grp, "W", "V", (x + 0.85, 0, 2.5), 0.2, RED, arrow=False)
            self.vlabel(grp, "V", k + 1, (x + 0.5, 0, 1.85), 0.36, RED)
            self.column(grp, x + 0.5, 1.15, V, RED, 0.28, dz=0.38)
            if j < n - 1:
                text(grp, "+", Point3(x + 1.85, 0, -0.2), 0.6, WHITE, self.serif)
            grp.hide()
            parts.append(Sequence(Wait(0.3 * j), Func(grp.show), fade_in(grp, 0.5)))
            # the weight flies from its table cell to the front of this term
            src = self.wtext[(k, q)]
            fly = text(self.board, "{:.2f}".format(w), self.board.getRelativePoint(self.qk, src.getPos()),
                       0.3, YELLOW, self.serif)
            fly.hide()
            dst = self.board.getRelativePoint(e, Point3(x - 0.55, 0, -0.2))
            flights.append(Sequence(Wait(0.25 * j), Func(fly.show),
                                    Parallel(LerpPosInterval(fly, 1.4, dst, blendType="easeInOut"),
                                             LerpScaleInterval(fly, 1.4, 0.42))))
        xr = x_start + n * gap + 0.4
        res = e.attachNewNode("result")
        text(res, "=", Point3(xr - 0.9, 0, -0.2), 0.6, WHITE, self.serif)
        self.vlabel(res, "\u0394E", q + 1, (xr + 0.5, 0, 1.85), 0.38, YELLOW)
        self.column(res, xr + 0.5, 1.15, vals_sum, YELLOW, 0.28, dz=0.38)
        rect(res, xr - 0.2, -2.0, xr + 1.2, 2.5, YELLOW, 2.0)
        res.hide()
        seq = Sequence(Func(self.view, SEC_E + Vec3(0, 0, 0.6), 0, 0, 16.5),
                       Wait(0.4), parts, flights, Wait(0.2), Func(res.show), fade_in(res, 0.6))
        names = " + ".join("{:.2f} x V({})".format(self.W[k][q], self.word(self.toks[k])) for k in ks)
        return ("Each word also offers a VALUE vector V (made with W_V). The weights of the 'ature' column "
                "mix these values:  {}  =  \u0394E, a change for 'ature'.".format(names), seq)

    # ---------------------------------------------------------- 9 the arrow moves
    def s_update(self):
        sp = self.space
        base = Point3(*SPACE_POS[FOCUS_TOKEN])
        target = (Point3(*SPACE_POS[" tank"]) + Point3(*SPACE_POS[" temper"])) * 0.5 + Vec3(0.2, -0.3, 1.6)
        delta = Arrow(sp, base, target, YELLOW, 3.0)
        new = Arrow(sp, Point3(0, 0, 0), target, WHITE, 3.5)
        lab = text(sp, "ature + context = \"tank temperature\"", target + Vec3(0.35, 0, 0.35), 0.3,
                   WHITE, self.serif, TextNode.ALeft)
        lab.setBillboardPointEye()
        lab.hide()
        dl = text(sp, "\u0394E", (base + target) * 0.5 + Vec3(0, 0, -0.6), 0.4, YELLOW, self.symbol)
        dl.setBillboardPointEye()
        dl.hide()
        old = self.arrows[FOCUS_TOKEN].root
        old.setTransparency(TransparencyAttrib.MAlpha)
        self.focus_label.setTransparency(TransparencyAttrib.MAlpha)
        seq = Sequence(
            Func(self.view, SEC_B + Vec3(1.2, 0, 1.4), 15, 22, 14),
            Wait(1.0),
            LerpFunc(delta.grow, fromData=0.001, toData=1.0, duration=1.2, blendType="easeOut"),
            Func(dl.show), fade_in(dl, 0.3),
            LerpFunc(new.grow, fromData=0.001, toData=1.0, duration=1.2, blendType="easeOut"),
            Parallel(LerpColorScaleInterval(old, 0.8, (1, 1, 1, 0.25)),
                     LerpColorScaleInterval(self.focus_label, 0.8, (1, 1, 1, 0.3))),
            Func(lab.show), fade_in(lab, 0.5),
            self.orbit(SEC_B + Vec3(1.2, 0, 1.4), 15, -20, 22, 14, 4.0))
        return ("Add \u0394E to the old arrow: 'ature' alone was vague, but now its vector lands right "
                "next to tank and temper - it carries 'tank temperature'. This is what attention DOES - it moves meanings around "
                "using context.", seq)

    # ---------------------------------------------------------- 10 MLP + repeat
    def s_mlp(self):
        c = self.mlp_center
        seq = Sequence(Func(self.view, c + Vec3(0, 0, 0.5), 60, 14, 16),
                       Wait(1.0), self.pulse(0.3), self.pulse(0.25),
                       Func(self.view, c + Vec3(0, -2.5, 0), 48, 18, 34), Wait(1.0))
        return ("Then every vector goes through the MULTILAYER PERCEPTRON: millions of weights "
                "(red = negative, blue = positive) that store facts. Attention + MLP = one layer; "
                "real models repeat it 30-100 times, then predict the next token.", seq)


if __name__ == "__main__":
    StyleSample().run()
