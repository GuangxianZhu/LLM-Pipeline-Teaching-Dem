# -*- coding: utf-8 -*-
# Claude Opus 写的
# 中英文切换：Claude 改了这个文件（CJK 字体、ui_font、wrap_cjk、text 的字体选择）
"""
Drawing kit for the 3Blue1Brown-like look: black board, thin lines, serif text,
vector arrows, number columns, glowing spheres. Everything is built from code.
"""
import math
import os
import re

from direct.interval.IntervalGlobal import LerpColorScaleInterval
from panda3d.core import (CardMaker, DynamicTextGlyph, Filename, Geom, GeomNode, GeomTriangles, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, LineSegs, Point3, TextNode, TextProperties,
                          TextPropertiesManager, Texture, TransparencyAttrib, Vec3)

import i18n

# ------------------------------------------------------------------ palette
WHITE = (1, 1, 1, 1)
GREY = (0.55, 0.55, 0.55, 1)
DIM = (0.32, 0.32, 0.34, 1)
DARK = (0.18, 0.18, 0.2, 1)
BLUE = (0.35, 0.77, 0.87, 1)
TEAL = (0.36, 0.81, 0.72, 1)
GREEN = (0.51, 0.76, 0.40, 1)
YELLOW = (1.0, 0.93, 0.35, 1)
GOLD = (0.94, 0.68, 0.30, 1)
ORANGE = (0.98, 0.58, 0.22, 1)
RED = (0.99, 0.38, 0.33, 1)
PINK = (0.93, 0.47, 0.66, 1)
PURPLE = (0.60, 0.50, 0.95, 1)
EDGE_RED = (0.93, 0.42, 0.38, 1)
EDGE_BLUE = (0.33, 0.72, 0.88, 1)
SALMON = (0.98, 0.70, 0.60, 1)
TOKEN_COLORS = [BLUE, SALMON, GREEN, RED, PINK, TEAL, GOLD, PURPLE]   # no yellow: yellow = the followed token

FONTS = {
    "serif": ["C:/Windows/Fonts/cambria.ttc", "C:/Windows/Fonts/times.ttf",
              "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
              "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "italic": ["C:/Windows/Fonts/timesi.ttf", "C:/Windows/Fonts/cambriai.ttf",
               "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
               "/usr/share/fonts/truetype/crosextra/Caladea-Italic.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"],
    # must contain the Greek Delta
    "symbol": ["C:/Windows/Fonts/cambria.ttc", "C:/Windows/Fonts/times.ttf",
               "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "mono": ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
             "/System/Library/Fonts/Menlo.ttc",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
    # Chinese (Han) glyphs, used when the language is zh. The first existing file wins.
    "cjk": ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf",
            "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            # extra Linux fallbacks (after the required list)
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"],
}


class Fonts:
    serif = italic = symbol = mono = cjk = None
    has_cjk = False

    @classmethod
    def load(cls, loader):
        for kind, paths in FONTS.items():
            f = None
            for p in paths:
                if os.path.exists(p):
                    f = loader.loadFont(Filename.fromOsSpecific(p).getFullpath())
                    if f and f.isValid():
                        f.setPixelsPerUnit(120 if kind == "cjk" else 80)   # Chinese glyphs need more pixels
                        break
                    f = None
            setattr(cls, kind, f)
        cls.italic = cls.italic or cls.serif
        cls.symbol = cls.symbol or cls.serif
        cls.mono = cls.mono or cls.serif
        cls.has_cjk = cls.cjk is not None
        cls.cjk = cls.cjk or cls.serif                  # no Chinese font found: fall back (squares, but no crash)
        # fonts usable inside a string with the markup  \1cjk\1...\2  and  \1sym\1...\2  (see fx)
        tpm = TextPropertiesManager.getGlobalPtr()
        for name, font in (("cjk", cls.cjk), ("sym", cls.symbol)):
            if font:
                props = TextProperties()
                props.setFont(font)
                tpm.setProperties(name, props)


# ------------------------------------------------------------------ text + lines
_HAS = {}
_CJK_RE = re.compile("[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]")


def _has_glyph(font, ch):
    key = (id(font), ch)
    if key not in _HAS:
        _HAS[key] = ch.isspace() or isinstance(font.getGlyph(ord(ch)), DynamicTextGlyph)
    return _HAS[key]


def fx(s, font):
    """zh only: a character the base font cannot draw (e.g. the superscript T of K-transpose in a Chinese font)
    is wrapped in text markup that switches to a font that has it. English text is returned untouched."""
    if i18n.LANG == "en" or not font or s.isascii():
        return s
    out, run, run_fb = [], [], None

    def flush():
        if run:
            out.append("\1{0}\1{1}\2".format(run_fb, "".join(run)) if run_fb else "".join(run))
            run.clear()
    for ch in s:
        fb = None
        if ord(ch) >= 0x250 and ch not in "\x01\x02" and not _has_glyph(font, ch):
            if font is not Fonts.cjk and Fonts.has_cjk and _has_glyph(Fonts.cjk, ch):
                fb = "cjk"
            elif font is not Fonts.symbol and _has_glyph(Fonts.symbol, ch):
                fb = "sym"
        if fb != run_fb:
            flush()
            run_fb = fb
        run.append(ch)
    flush()
    return "".join(out)


def ui_font(kind="serif"):
    """Font for interface text: the Chinese font when the language is zh, otherwise Fonts.<kind>."""
    return Fonts.cjk if i18n.LANG != "en" and Fonts.cjk else getattr(Fonts, kind)


def display(s, kind="serif"):
    """A string ready for an OnscreenText / DirectButton that uses ui_font(kind)."""
    return fx(s, ui_font(kind))


def _scene_font(font, s):
    """Scene text keeps its usual font, except a string that contains Chinese: that one needs the Chinese font."""
    if (i18n.LANG != "en" and Fonts.cjk and font in (None, Fonts.serif, Fonts.italic, Fonts.symbol)
            and _CJK_RE.search(s)):
        return Fonts.cjk
    return font or Fonts.serif


_UNIT = re.compile("[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef\u3000-\u303f]|\\s+|[^\\s\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef\u3000-\u303f]+")
_NO_LINE_START = set("，。、；：！？）】」』》”’…％,.;:!?)]}%")
_UNIT_W = {}


def _unit_width(u, font):
    key = (id(font), u)
    if key not in _UNIT_W:
        tn = TextNode("meas")
        tn.setFont(font)
        tn.setText(fx(u, font))
        _UNIT_W[key] = tn.getWidth()
    return _UNIT_W[key]


def wrap_cjk(s, width=44.0, font=None):
    """Insert line breaks into Chinese text (Panda3D only breaks lines at spaces). `width` is in text units, the
    same unit as OnscreenText(wordwrap=...); one Chinese character is 1.0 wide. Existing newlines are kept,
    a line never breaks inside an English word and never starts with closing punctuation."""
    font = font or ui_font("symbol")
    lines_out = []
    for para in s.split("\n"):
        line, w = "", 0.0
        for u in _UNIT.findall(para):
            uw = _unit_width(u, font) if font else len(u)
            if line and w + uw > width and not u.isspace() and u[0] not in _NO_LINE_START:
                lines_out.append(line.rstrip())
                line, w = "", 0.0
            if not line and u.isspace():
                continue
            line += u
            w += uw
        lines_out.append(line.rstrip())
    return "\n".join(lines_out)


def text(parent, s, pos, scale, color=WHITE, font=None, align=TextNode.ACenter, wrap=None):
    tn = TextNode("t")
    font = _scene_font(font, s)
    tn.setText(fx(s, font))
    tn.setAlign(align)
    tn.setTextColor(*color)
    if font:
        tn.setFont(font)
    if wrap:
        tn.setWordwrap(wrap)
    np = parent.attachNewNode(tn)
    np.setPos(pos)
    np.setScale(scale)
    np.setY(np.getY() - 0.01)
    return np


def text_width(s, font=None):
    tn = TextNode("m")
    f = _scene_font(font, s)
    if f:
        tn.setFont(f)
    tn.setText(fx(s, f))
    return tn.getWidth()


def lines(parent, pts_list, color, thick=2.0):
    ls = LineSegs()
    ls.setThickness(thick)
    ls.setColor(*color)
    for pts in pts_list:
        ls.moveTo(*pts[0])
        for p in pts[1:]:
            ls.drawTo(*p)
    np = parent.attachNewNode(ls.create())
    if len(color) == 4 and color[3] < 1:
        np.setTransparency(TransparencyAttrib.MAlpha)
    return np


def rect(parent, x0, z0, x1, z1, color, thick=2.0, y=0.0):
    return lines(parent, [[(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), (x0, y, z0)]], color, thick)


def fill(parent, x0, z0, x1, z1, color, alpha=0.18, y=0.02):
    cm = CardMaker("fill")
    cm.setFrame(x0, x1, z0, z1)
    np = parent.attachNewNode(cm.generate())
    np.setY(y)
    np.setColor(color[0], color[1], color[2], alpha)
    np.setTransparency(TransparencyAttrib.MAlpha)
    np.setDepthWrite(False)
    np.setTwoSided(True)
    return np


def image_card(parent, tex, x0, z0, x1, z1, y=-0.02):
    cm = CardMaker("img")
    cm.setFrame(x0, x1, z0, z1)
    np = parent.attachNewNode(cm.generate())
    np.setY(y)
    np.setTexture(tex, 1)
    np.setTransparency(TransparencyAttrib.MAlpha)
    return np


def array_to_texture(arr):
    h, w = arr.shape[:2]
    tex = Texture("img")
    tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.setRamImageAs(arr[::-1].tobytes(), "RGBA")
    return tex


# ------------------------------------------------------------------ meshes
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
    """Flat arrow head in the XZ plane at point p, pointing along (dx, dz)."""
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


def arrow2d(parent, p0, p1, color=WHITE, thick=2.0, head=0.16):
    """Flat arrow in the board plane (XZ)."""
    g = parent.attachNewNode("arrow2d")
    d = Vec3(*p1) - Vec3(*p0)
    L = d.length() or 1
    tip_base = Point3(*p1) - d / L * head * 0.9
    lines(g, [[p0, tip_base]], color, thick)
    triangle(g, p1, (d.x, d.z), head, color)
    return g


def curve_arrow2d(parent, pts, color=WHITE, thick=2.0, head=0.2):
    """Arrow along a polyline (board plane)."""
    g = parent.attachNewNode("curve")
    lines(g, [pts[:-1] + [pts[-1]]], color, thick)
    a, b = pts[-2], pts[-1]
    triangle(g, b, (b[0] - a[0], b[2] - a[2]), head, color)
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
    lines(g, e, (0.8, 0.8, 0.85, 0.45), 1.4)
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


# ------------------------------------------------------------------ animation helpers
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


# ------------------------------------------------------------------ composite symbols
def token_box(parent, word, color, center, scale=0.34, label_color=WHITE):
    """3B1B token box: word with a thin coloured frame and faint fill. Returns (node, width)."""
    g = parent.attachNewNode("tokbox")
    w = text_width(word) * scale + 0.22
    h = scale * 1.25
    x, y, z = center
    fill(g, x - w / 2, z - h * 0.32, x + w / 2, z + h * 0.68, color, 0.15, y + 0.02)
    rect(g, x - w / 2, z - h * 0.32, x + w / 2, z + h * 0.68, color, 1.6, y)
    text(g, word, Point3(x, y, z), scale, label_color)
    return g, w


def vlabel(parent, letter, sub, pos, scale, color, align="center", arrow=True):
    """A vector symbol: letter with a small arrow above and a subscript (E-arrow, 3)."""
    g = parent.attachNewNode("vlabel")
    font = Fonts.symbol if "\u0394" in letter else Fonts.serif
    wl = text_width(letter, font) * scale
    ws = text_width(str(sub)) * scale * 0.55 if sub != "" else 0
    total = wl + ws + 0.02
    x0 = pos[0] - total / 2 if align == "center" else (pos[0] if align == "left" else pos[0] - total)
    y, z = pos[1], pos[2]
    text(g, letter, Point3(x0, y, z), scale, color, font, TextNode.ALeft)
    if sub != "":
        text(g, str(sub), Point3(x0 + wl + 0.02, y, z - scale * 0.22), scale * 0.55, color,
             align=TextNode.ALeft)
    if arrow:
        az = z + scale * 0.88
        ax0, ax1 = x0 + wl * 0.1, x0 + wl * 0.95
        lines(g, [[(ax0, y - 0.01, az), (ax1, y - 0.01, az)],
                  [(ax1 - scale * 0.16, y - 0.01, az + scale * 0.09), (ax1, y - 0.01, az),
                   (ax1 - scale * 0.16, y - 0.01, az - scale * 0.09)]], color, 1.4)
    return g, total


def column(parent, x, z_top, vals, color=WHITE, scale=0.34, dz=0.48, y=0.0, bracket=GREY):
    """A vector written as a column of numbers in square brackets, with vertical dots."""
    col = parent.attachNewNode("col")
    k = scale / 0.34
    for r, v in enumerate(vals):
        text(col, num(v), Point3(x, y, z_top - r * dz), scale, color)
    n = len(vals)
    for j in range(3):
        d = disc(col, 0.032 * k, color, 12)
        d.setPos(x, y - 0.01, z_top - n * dz - j * 0.16 * k + 0.12)
    bw = 0.5 * k
    bz0, bz1 = z_top - n * dz - 0.35 * k, z_top + 0.4 * k
    for s in (-1, 1):
        bx = x + s * bw
        lines(col, [[(bx - s * 0.12, y, bz1), (bx, y, bz1), (bx, y, bz0), (bx - s * 0.12, y, bz0)]],
              bracket, 1.6)
    return col


# ------------------------------------------------------------------ heatmaps (Claude Opus 写的)
def value_color(v, scale=1.0):
    """Diverging colour: negative = blue, zero = near black, positive = warm red/orange."""
    t = math.tanh(v / scale) if scale else 0.0
    a = abs(t)
    if t >= 0:
        return (0.10 + 0.88 * a, 0.10 + 0.40 * a, 0.12 + 0.18 * a, 1)
    return (0.10 + 0.20 * a, 0.10 + 0.52 * a, 0.12 + 0.86 * a, 1)


def heatmap(parent, M, x0, z0, cw, ch, scale=1.0, gap=0.12, y=0.0, colors=None, gapx=None,
            nan_color=(0.04, 0.04, 0.05, 1)):
    """
    Draw matrix M (rows x cols) as coloured cells in the board plane, ONE mesh.
    Cell (r, c) has its top-left corner at (x0 + c*cw, z0 - r*ch). Returns the NodePath.
    `colors` may be a function value -> rgba.
    """
    import numpy as _np
    M = _np.atleast_2d(_np.asarray(M, dtype=float))
    R, C = M.shape
    vd = GeomVertexData("heat", GeomVertexFormat.getV3c4(), Geom.UHStatic)
    vd.setNumRows(R * C * 4)
    vw = GeomVertexWriter(vd, "vertex")
    cwr = GeomVertexWriter(vd, "color")
    tris = GeomTriangles(Geom.UHStatic)
    gx, gz = cw * (gap if gapx is None else gapx), ch * gap
    i = 0
    fn = colors or (lambda v: value_color(v, scale))
    for r in range(R):
        for c in range(C):
            v = M[r, c]
            col = nan_color if not _np.isfinite(v) else fn(v)
            xa, xb = x0 + c * cw + gx / 2, x0 + (c + 1) * cw - gx / 2
            za, zb = z0 - (r + 1) * ch + gz / 2, z0 - r * ch - gz / 2
            for (px, pz) in ((xa, za), (xb, za), (xb, zb), (xa, zb)):
                vw.addData3(px, y, pz)
                cwr.addData4(*col)
            tris.addVertices(i, i + 1, i + 2)
            tris.addVertices(i, i + 2, i + 3)
            i += 4
    np_ = parent.attachNewNode(_geom_node("heat", vd, tris))
    np_.setTwoSided(True)
    return np_


def shape_label(parent, s, pos, scale=0.32, color=GREY):
    """Matrix shape in brackets, e.g. '13 x 32'."""
    return text(parent, s.replace("x", "×") if "×" not in s else s, pos, scale, color, Fonts.symbol)
