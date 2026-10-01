# -*- coding: utf-8 -*-
# Claude Opus 写的
"""
The 3D "LLM factory": stations, belts, screens, and token blocks.
Everything is built from code (boxes + cards + text), no model files needed.
"""
from panda3d.core import (AmbientLight, CardMaker, DirectionalLight, Geom, GeomNode,
                          GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
                          LineSegs, NodePath, PNMImage, Point3, SamplerState, TextNode,
                          Texture, TextureStage, TransparencyAttrib, Vec3, Vec4)

import sim

# ------------------------------------------------------------------ layout
BELT_Z = 0.6          # height of the belt surface
DOCK, TOK, EMB, TF, SAMP, TRAY = -16.0, -11.0, -6.0, 0.0, 6.0, 11.5
LANE_X = 14.8         # belt that carries tool calls to the tool bays
LANE_Y = 9.5
BAY_Y = 12.0
BAYS = {"calculator": -7.0, "run_terminal": -1.5, "plot_chart": 4.0, "generate_image": 9.5}
ROW_Z = 7.6           # height of the "context row" above the transformer

COLORS = {
    "user": (0.33, 0.6, 0.95, 1),
    "system": (0.55, 0.56, 0.62, 1),
    "gen": (0.98, 0.58, 0.22, 1),
    "result": (0.3, 0.78, 0.45, 1),
    "block": (0.75, 0.45, 0.95, 1),
    "crate": (0.62, 0.36, 0.9, 1),
    "steel": (0.42, 0.45, 0.5, 1),
    "dark": (0.16, 0.17, 0.2, 1),
}

# camera views: target, heading, pitch, distance
VIEWS = {
    "overview": ((0, 3, 1), 0, 38, 50),
    "dock": ((-15.5, 1, 2.2), -8, 22, 17),
    "tokenizer": ((-11, 0, 1.0), -4, 42, 15),
    "embed": ((-6, 0, 1.0), 4, 42, 15),
    "transformer": ((0, 0, 5.0), 0, 15, 27),
    "sampler": ((8.5, 0.5, 2.8), 0, 18, 20),
    "router": ((12.5, 4.5, 1.0), 0, 50, 23),
    "return": ((-3, 5, 2), 0, 42, 42),
}


def bay_view(name):
    return ((BAYS[name], BAY_Y + 0.3, 2.6), 0, 26, 15)


# ------------------------------------------------------------------ geometry
def make_box(sx, sy, sz, color=(1, 1, 1, 1), name="box"):
    """Box centred in x/y, sitting on z=0."""
    vd = GeomVertexData(name, GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vw = GeomVertexWriter(vd, "vertex")
    nw = GeomVertexWriter(vd, "normal")
    tris = GeomTriangles(Geom.UHStatic)
    x, y = sx / 2.0, sy / 2.0
    faces = [
        ((0, 0, 1), [(-x, -y, sz), (x, -y, sz), (x, y, sz), (-x, y, sz)]),
        ((0, 0, -1), [(-x, y, 0), (x, y, 0), (x, -y, 0), (-x, -y, 0)]),
        ((0, -1, 0), [(-x, -y, 0), (x, -y, 0), (x, -y, sz), (-x, -y, sz)]),
        ((0, 1, 0), [(x, y, 0), (-x, y, 0), (-x, y, sz), (x, y, sz)]),
        ((1, 0, 0), [(x, -y, 0), (x, y, 0), (x, y, sz), (x, -y, sz)]),
        ((-1, 0, 0), [(-x, y, 0), (-x, -y, 0), (-x, -y, sz), (-x, y, sz)]),
    ]
    i = 0
    for n, vs in faces:
        for v in vs:
            vw.addData3(*v)
            nw.addData3(*n)
        tris.addVertices(i, i + 1, i + 2)
        tris.addVertices(i, i + 2, i + 3)
        i += 4
    g = Geom(vd)
    g.addPrimitive(tris)
    gn = GeomNode(name)
    gn.addGeom(g)
    np = NodePath(gn)
    np.setColor(*color)
    if len(color) == 4 and color[3] < 1:
        np.setTransparency(TransparencyAttrib.MAlpha)
    return np


def box(parent, sx, sy, sz, pos, color, name="box"):
    np = make_box(sx, sy, sz, color, name)
    np.reparentTo(parent)
    np.setPos(*pos)
    return np


def text_node(text, color=(1, 1, 1, 1), align=TextNode.ACenter, wrap=None, card=None):
    tn = TextNode("text")
    tn.setText(text)
    tn.setAlign(align)
    tn.setTextColor(*color)
    if wrap:
        tn.setWordwrap(wrap)
    if card:
        tn.setCardColor(*card)
        tn.setCardAsMargin(0.12, 0.12, 0.05, 0.05)
    return tn


def label(parent, text, pos, scale=0.5, color=(1, 1, 1, 1), card=None, billboard=True):
    np = parent.attachNewNode(text_node(text, color, card=card))
    np.setPos(*pos)
    np.setScale(scale)
    np.setLightOff()
    if billboard:
        np.setBillboardPointEye()
    return np


def stripe_texture():
    img = PNMImage(32, 8)
    img.fill(0.2, 0.21, 0.24)
    for x in range(0, 6):
        for y in range(8):
            img.setXel(x, y, 0.33, 0.35, 0.39)
    tex = Texture("stripes")
    tex.load(img)
    tex.setWrapU(SamplerState.WM_repeat)
    tex.setWrapV(SamplerState.WM_repeat)
    return tex


def grid_texture():
    img = PNMImage(64, 64)
    img.fill(0.13, 0.14, 0.16)
    for i in range(64):
        img.setXel(i, 0, 0.19, 0.2, 0.23)
        img.setXel(0, i, 0.19, 0.2, 0.23)
    tex = Texture("grid")
    tex.load(img)
    tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    return tex


class Screen:
    """A flat monitor: dark card + text, optionally showing an image texture."""

    def __init__(self, parent, pos, w, h, title, text_scale=0.24):
        self.root = parent.attachNewNode("screen")
        self.root.setPos(*pos)
        self.root.setLightOff()
        box(self.root, w + 0.3, 0.15, h + 0.3, (0, 0.12, -0.15), COLORS["steel"]).setLightOff(0)
        cm = CardMaker("bg")
        cm.setFrame(-w / 2, w / 2, 0, h)
        self.bg = self.root.attachNewNode(cm.generate())
        self.bg.setColor(0.05, 0.07, 0.09, 1)
        cm2 = CardMaker("img")
        cm2.setFrame(-w / 2 + 0.1, w / 2 - 0.1, 0.1, h - 0.45)
        self.img = self.root.attachNewNode(cm2.generate())
        self.img.setY(-0.01)
        self.img.setTransparency(TransparencyAttrib.MAlpha)
        self.img.hide()
        self.title = self.root.attachNewNode(text_node(title, (0.6, 0.85, 1, 1)))
        self.title.setPos(0, -0.02, h - 0.35)
        self.title.setScale(0.26)
        self.tn = text_node("", (0.85, 1, 0.85, 1), TextNode.ALeft, wrap=(w - 0.3) / text_scale)
        self.text = self.root.attachNewNode(self.tn)
        self.text.setPos(-w / 2 + 0.15, -0.02, h - 0.75)
        self.text.setScale(text_scale)
        self.w, self.h = w, h

    def set_text(self, s):
        self.tn.setText(s)

    def set_image(self, tex):
        if tex is None:
            self.img.hide()
            return
        self.img.setTexture(tex, 1)
        self.img.show()
        self.img.setColor(1, 1, 1, 1)


class Tok:
    """One token block (or a bigger block standing for many tokens)."""

    def __init__(self, parent, text, kind="user", width=0.62, display=None):
        self.text = text
        self.kind = kind
        self.width = width
        self.np = parent.attachNewNode("tok")
        self.body = make_box(width, 0.62, 0.32, COLORS.get(kind, COLORS["user"]))
        self.body.reparentTo(self.np)
        shown = display if display is not None else sim.show(text)
        self.tn = text_node(shown, (1, 1, 1, 1), card=(0, 0, 0, 0.6))
        self.lab = self.np.attachNewNode(self.tn)
        self.lab.setLightOff()
        self.lab.setBillboardPointEye()
        self.lab.setPos(0, 0, 0.62)
        self.lab.setBin("fixed", 30)      # labels always readable, never hidden
        self.lab.setDepthTest(False)
        self.lab.setDepthWrite(False)
        self.fit()
        self.vec = None

    def fit(self):
        w = max(self.tn.getWidth(), 0.5)
        lines = self.tn.getNumRows() if self.tn.getNumRows() else 1
        self.lab.setScale(min(0.3, (self.width + 0.2) / w, 0.55 / lines))

    def set_label(self, s):
        self.tn.setText(s)
        self.fit()

    def add_vector(self, values):
        """Little coloured cells on top of the block = the embedding vector."""
        if self.vec:
            self.vec.removeNode()
        self.vec = self.np.attachNewNode("vec")
        n = len(values)
        cw = min(0.075, (self.width - 0.06) / n)
        for i, v in enumerate(values):
            a = abs(v)
            if v >= 0:
                c = (0.5 + 0.45 * a, 0.5 - 0.2 * a, 0.5 - 0.35 * a, 1)
            else:
                c = (0.5 - 0.32 * a, 0.5 + 0.02 * a, 0.5 + 0.42 * a, 1)
            box(self.vec, cw * 0.85, 0.4, 0.05 + 0.18 * a, ((i - (n - 1) / 2) * cw, 0, 0.32), c)
        return self.vec

    def remove(self):
        self.np.removeNode()


# ------------------------------------------------------------------ factory
class Factory:
    def __init__(self, base):
        self.base = base
        self.root = base.render.attachNewNode("factory")
        self.dyn = base.render.attachNewNode("dynamic")   # everything per-scenario
        self.belt_cards = []
        self.scroll = 0.0
        self._lights()
        self._floor()
        self._belts()
        self._stations()
        base.taskMgr.add(self._scroll_task, "belt-scroll")

    # -- environment
    def _lights(self):
        r = self.base.render
        al = AmbientLight("amb")
        al.setColor(Vec4(0.45, 0.46, 0.5, 1))
        r.setLight(r.attachNewNode(al))
        dl = DirectionalLight("sun")
        dl.setColor(Vec4(0.75, 0.73, 0.7, 1))
        dnp = r.attachNewNode(dl)
        dnp.setHpr(-30, -55, 0)
        r.setLight(dnp)
        self.base.setBackgroundColor(0.09, 0.1, 0.12, 1)

    def _floor(self):
        cm = CardMaker("floor")
        cm.setFrame(-40, 40, -25, 35)
        cm.setUvRange((0, 0), (40, 30))
        f = self.root.attachNewNode(cm.generate())
        f.setP(-90)
        f.setTexture(grid_texture())

    def belt(self, x0, y0, x1, y1, width=3.4):
        d = Vec3(x1 - x0, y1 - y0, 0)
        length = d.length()
        mid = Point3((x0 + x1) / 2, (y0 + y1) / 2, 0)
        node = self.root.attachNewNode("belt")
        node.setPos(mid)
        node.lookAt(Point3(x1, y1, 0))
        node.setH(node.getH() - 90)          # local +X along the belt
        box(node, length, width, BELT_Z - 0.02, (0, 0, 0), COLORS["dark"])
        for s in (1, -1):
            box(node, length, 0.12, 0.12, (0, s * (width / 2 + 0.06), BELT_Z - 0.06), COLORS["steel"])
        cm = CardMaker("top")
        cm.setFrame(-length / 2, length / 2, -width / 2, width / 2)
        cm.setUvRange((0, 0), (length / 0.7, 1))
        top = node.attachNewNode(cm.generate())
        top.setP(-90)
        top.setZ(BELT_Z - 0.01)
        top.setTexture(stripe_texture())
        top.setLightOff()
        self.belt_cards.append(top)

    def _belts(self):
        self.belt(DOCK - 2.2, 0, LANE_X + 1.0, 0)
        self.belt(LANE_X, 1.7, LANE_X, LANE_Y + 0.8, 1.6)
        self.belt(LANE_X - 0.8, LANE_Y, -9.5, LANE_Y, 1.6)

    def _scroll_task(self, task):
        self.scroll -= self.base.clock.getDt() * 0.8
        for c in self.belt_cards:
            c.setTexOffset(TextureStage.getDefault(), self.scroll, 0)
        return task.cont

    # -- stations
    def _sign(self, text, x, y, z, color=(1, 0.85, 0.4, 1), scale=0.42):
        np = label(self.root, text, (x, y, z), scale, color)
        np.setDepthWrite(False)
        np.setBin("transparent", 5)
        return np

    def _stations(self):
        R = self.root
        steel, dark = COLORS["steel"], COLORS["dark"]

        # 1 input dock
        box(R, 5.6, 1.4, 0.9, (DOCK, 3.0, 0), dark)
        self.dock_screen = Screen(R, (DOCK, 2.7, 1.0), 5.6, 3.6, "CONTEXT (what the model sees)", 0.2)
        self._sign("1  INPUT", DOCK, 2.7, 5.2)

        # 2 tokenizer: gantry + blade
        for s in (1, -1):
            box(R, 0.3, 0.3, 3.0, (TOK, s * 2.0, 0), steel)
        box(R, 0.5, 4.3, 0.3, (TOK, 0, 3.0), steel)
        self.blade = box(R, 0.18, 3.6, 0.7, (TOK, 0, 2.2), (0.85, 0.87, 0.9, 1))
        self._sign("2  TOKENIZER", TOK, 0, 4.2)

        # 3 embedding booth
        for s in (1, -1):
            box(R, 2.6, 0.2, 2.2, (EMB, s * 2.0, 0), (0.15, 0.55, 0.55, 1))
        box(R, 2.6, 4.2, 0.15, (EMB, 0, 2.2), (0.15, 0.55, 0.55, 0.45))
        self.emb_nozzles = [box(R, 0.25, 0.25, 0.25, (EMB + dx, 0, 1.95), (0.9, 0.5, 0.3, 1))
                            for dx in (-0.8, 0, 0.8)]
        self._sign("3  EMBEDDING", EMB, 0, 3.4)

        # 4 transformer tower
        for sx in (1, -1):
            for sy in (1, -1):
                box(R, 0.25, 0.25, 6.6, (TF + sx * 2.6, sy * 2.0, 0), steel)
        self.plates = []
        self.plate_labels = []
        for i, z in enumerate((1.7, 2.9, 4.1, 5.3)):
            p = box(R, 5.0, 3.8, 0.12, (TF, 0, z), (0.55, 0.35, 0.9, 0.35))
            p.setDepthWrite(False)
            self.plates.append(p)
            txt = "Layer {}".format(i + 1) if i < 3 else "Layer 4 ... 32"
            self.plate_labels.append(label(R, txt, (TF - 3.6, -1.9, z + 0.05), 0.3, (0.8, 0.7, 1, 1)))
        self._sign("4  TRANSFORMER (attention layers)", TF, 0, 9.4)

        # 5 sampler + probability bars
        box(R, 2.2, 2.6, 1.3, (SAMP, 0, 0), (0.5, 0.33, 0.18, 1))
        box(R, 1.0, 1.0, 0.3, (SAMP, 0, 1.3), steel)
        self.bars_root = self.root.attachNewNode("bars")
        self.bars_root.setPos(SAMP, 0, 2.0)
        self.bars_root.setBillboardAxis()
        self.bars = []
        for i in range(5):
            b = make_box(0.62, 0.3, 1.0, (0.4, 0.5, 0.65, 1))
            b.reparentTo(self.bars_root)
            b.setPos((i - 2) * 1.15, 0, 0)
            b.setSz(0.01)
            tl = label(self.bars_root, "", ((i - 2) * 1.15, -0.2, -0.3), 0.22, billboard=False)
            pl = label(self.bars_root, "", ((i - 2) * 1.15, -0.2, 0.2), 0.24, (1, 0.9, 0.6, 1), billboard=False)
            for t in (tl, pl):
                t.setBin("fixed", 31)
                t.setDepthTest(False)
                t.setDepthWrite(False)
            self.bars.append((b, tl, pl))
        self._sign("5  NEXT-TOKEN PREDICTION", SAMP, 0, 6.2)

        # 6 output tray + router
        box(R, 3.6, 2.8, 0.75, (TRAY, 0, 0), steel)
        self.tray_screen = Screen(R, (TRAY - 0.4, 2.7, 1.0), 4.6, 3.6, "MODEL OUTPUT (text)", 0.2)
        self.router_lamp = box(R, 0.5, 0.5, 0.5, (TRAY + 2.3, -1.4, 0.75), (0.3, 0.3, 0.3, 1))
        self._sign("6  OUTPUT + ROUTER", TRAY, 0, 5.4)

        # 7 tool bays
        self.bay_screens = {}
        self.bay_lamps = {}
        names = {"calculator": "calculator()", "run_terminal": "run_terminal()",
                 "plot_chart": "plot_chart()", "generate_image": "generate_image()"}
        for name, bx in BAYS.items():
            box(R, 4.6, 3.4, 0.25, (bx, BAY_Y + 0.4, 0), dark)
            box(R, 4.6, 0.2, 4.9, (bx, BAY_Y + 2.0, 0), (0.22, 0.24, 0.28, 1))
            self.bay_screens[name] = Screen(R, (bx, BAY_Y + 1.7, 1.2), 4.2, 3.4, names[name], 0.19)
            self.bay_lamps[name] = box(R, 0.45, 0.45, 0.35, (bx + 1.8, BAY_Y - 0.9, 0.25), (0.3, 0.3, 0.3, 1))
        self._sign("7  TOOLS (normal Python code)", 1.0, BAY_Y + 2.0, 6.3)
        label(R, "tool calls ->", (LANE_X - 3, LANE_Y - 1.3, 0.9), 0.32, (0.85, 0.75, 1, 1))

    # -- per-scenario helpers
    def clear(self):
        self.dyn.removeNode()
        self.dyn = self.base.render.attachNewNode("dynamic")
        for b, tl, pl in self.bars:
            b.setSz(0.01)
            tl.node().setText("")
            pl.node().setText("")
        for s in list(self.bay_screens.values()) + [self.dock_screen, self.tray_screen]:
            s.set_text("")
            s.set_image(None)
        for l in self.bay_lamps.values():
            l.setColor(0.3, 0.3, 0.3, 1)
        self.router_lamp.setColor(0.3, 0.3, 0.3, 1)
        self.blade.setZ(2.2)

    def token(self, text, kind="user", width=0.62, display=None):
        return Tok(self.dyn, text, kind, width, display)

    def set_bars(self, cands, chosen_index=None, grow=1.0):
        top = max([p for _, p in cands] or [1.0])
        for i, (b, tl, pl) in enumerate(self.bars):
            if i < len(cands):
                t, p = cands[i]
                h = max(0.03, 3.2 * p / max(top, 1e-6)) * grow
                b.setSz(h)
                tl.node().setText(sim.show(t))
                tw = max(tl.node().getWidth(), 0.1)
                tl.setScale(min(0.24, 1.05 / tw))
                pl.node().setText("{:.0f}%".format(p * 100) if p >= 0.01 else "<1%")
                pl.setZ(h + 0.12)
                hot = chosen_index is not None and i == chosen_index
                b.setColor(*(COLORS["gen"] if hot else (0.4, 0.5, 0.65, 1)))
            else:
                b.setSz(0.01)
                tl.node().setText("")
                pl.node().setText("")

    def bar_top(self, i):
        b = self.bars[i][0]
        return self.base.render.getRelativePoint(self.bars_root, Point3((i - 2) * 1.15, 0, b.getSz() + 0.3))

    def beams(self, src, targets):
        """Curved attention beams from src to each (point, weight). Returns NodePath."""
        root = self.dyn.attachNewNode("beams")
        root.setLightOff()
        root.setTransparency(TransparencyAttrib.MAlpha)
        root.setDepthWrite(False)
        root.setBin("fixed", 10)
        wmax = max(w for _, w in targets) if targets else 1
        for p, w in targets:
            k = w / wmax
            ls = LineSegs()
            ls.setThickness(1.5 + 7 * k)
            ls.setColor(1, 0.85 * (0.6 + 0.4 * k), 0.25, 0.25 + 0.75 * k)
            dist = (p - src).length()
            ctrl = (src + p) * 0.5 + Vec3(0, 0, 0.8 + dist * 0.18)
            for i in range(21):
                t = i / 20.0
                q = src * (1 - t) * (1 - t) + ctrl * 2 * t * (1 - t) + p * t * t
                if i == 0:
                    ls.moveTo(q)
                else:
                    ls.drawTo(q)
            root.attachNewNode(ls.create())
            if k > 0.35:
                mid = src * 0.25 + ctrl * 0.5 + p * 0.25
                label(root, "{:.2f}".format(w), (mid.x, mid.y, mid.z + 0.2), 0.26, (1, 0.95, 0.6, 1),
                      card=(0, 0, 0, 0.6))
        return root
