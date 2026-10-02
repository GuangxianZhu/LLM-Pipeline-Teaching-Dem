# -*- coding: utf-8 -*-
# Claude Opus 写的
# 中英文切换：Claude 改了这个文件（方框文字改用 i18n.t，新增 relabel）
"""
The always-visible architecture map (the classic GPT / decoder-only Transformer diagram).
Bottom to top, like in the papers. highlight(key) lights up the block being shown.
"""
from direct.gui.DirectGui import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import LineSegs, TextNode

from i18n import t
from kit import display, ui_font

BOX_BG = (0.12, 0.12, 0.14, 0.95)
HOT_BG = (0.55, 0.42, 0.08, 0.95)
EDGE = (0.5, 0.5, 0.55, 1)
TXT = (0.85, 0.85, 0.9, 1)
DIMTXT = (0.55, 0.55, 0.6, 1)

# key, x-centre, z-centre, width, height   (in the map's own units); the label is t("arch.<key>")
BLOCKS = [
    ("output", 0.0, 0.00, 0.56, 0.07),
    ("softmax", 0.0, -0.13, 0.34, 0.07),
    ("linear", 0.0, -0.26, 0.40, 0.07),
    ("add2", 0.0, -0.43, 0.42, 0.07),
    ("ffn", 0.0, -0.56, 0.42, 0.08),
    ("add1", 0.0, -0.73, 0.42, 0.07),
    ("attn", 0.0, -0.88, 0.42, 0.11),
    ("pos", -0.235, -1.07, 0.17, 0.10),
    ("embed", 0.06, -1.20, 0.40, 0.07),
    ("input", 0.06, -1.33, 0.40, 0.07),
]


class ArchMap:
    def __init__(self, parent, pos):
        self.root = parent.attachNewNode("archmap")
        self.root.setPos(pos)
        self.boxes, self.labels = {}, {}
        f = ui_font()
        ls = LineSegs()
        ls.setThickness(1.4)
        ls.setColor(*EDGE)
        # main arrows (bottom -> top)
        chain = [("input", "embed"), ("embed", "plus"), ("plus", "attn"), ("attn", "add1"), ("add1", "ffn"),
                 ("ffn", "add2"), ("add2", "linear"), ("linear", "softmax"), ("softmax", "output")]
        geo = {k: (x, z, w, h) for k, x, z, w, h in BLOCKS}
        geo["plus"] = (0.06, -1.07, 0.04, 0.04)
        for a, b in chain:
            xa, za, wa, ha = geo[a]
            xb, zb, wb, hb = geo[b]
            x = xb if b != "plus" else 0.06
            if a == "plus":
                x = 0.0
                ls.moveTo(0.06, 0, za + ha / 2)
                ls.drawTo(0.06, 0, za + 0.06)
                ls.drawTo(0.0, 0, za + 0.06)
                ls.drawTo(0.0, 0, zb - hb / 2)
            else:
                ls.moveTo(x, 0, za + ha / 2)
                ls.drawTo(x, 0, zb - hb / 2)
            self._head(ls, x, zb - hb / 2)
        # positional encoding -> plus
        ls.moveTo(-0.15, 0, -1.07)
        ls.drawTo(0.04, 0, -1.07)
        # residual (skip) connections on the right side
        for lo, hi in (("plus_out", "add1"), ("add1", "add2")):
            z0 = -1.0 if lo == "plus_out" else geo["add1"][1] - 0.0
            x_, z1, w1, h1 = geo[hi]
            ls.moveTo(0.0 if lo == "plus_out" else 0.21, 0, z0 + (0.0 if lo == "plus_out" else -0.06))
            ls.drawTo(0.28, 0, z0 + (0.0 if lo == "plus_out" else -0.06))
            ls.drawTo(0.28, 0, z1)
            ls.drawTo(0.21, 0, z1)
        self.root.attachNewNode(ls.create())
        # the N x frame around one layer
        frame = LineSegs()
        frame.setThickness(1.2)
        frame.setColor(0.4, 0.4, 0.45, 1)
        x0, x1, z0, z1 = -0.3, 0.33, -0.98, -0.36
        for a, b in (((x0, z0), (x1, z0)), ((x1, z0), (x1, z1)), ((x1, z1), (x0, z1)), ((x0, z1), (x0, z0))):
            frame.moveTo(a[0], 0, a[1])
            frame.drawTo(b[0], 0, b[1])
        self.root.attachNewNode(frame.create())
        self.layer_key = "arch.layers"
        self.layer_text = OnscreenText(display(t(self.layer_key)), parent=self.root, pos=(-0.29, -0.395), scale=0.03,
                                       fg=DIMTXT, font=f, align=TextNode.ALeft, mayChange=True)
        plus = OnscreenText("+", parent=self.root, pos=(0.06, -1.083), scale=0.05, fg=TXT, font=f)
        self.plus = plus
        self.geo = geo
        for key, x, z, w, h in BLOCKS:
            b = DirectFrame(parent=self.root, frameColor=BOX_BG, frameSize=(-w / 2, w / 2, -h / 2, h / 2),
                            pos=(x, 0, z))
            label = t("arch." + key)
            lines = label.count("\n") + 1
            lab = OnscreenText(display(label), parent=self.root, pos=(x, z - 0.01 + 0.014 * (lines - 1)),
                               scale=0.026, fg=TXT, font=f, mayChange=True)
            self.boxes[key], self.labels[key] = b, lab
        self.current = None

    @staticmethod
    def _head(ls, x, z):
        ls.moveTo(x - 0.012, 0, z - 0.018)
        ls.drawTo(x, 0, z)
        ls.drawTo(x + 0.012, 0, z - 0.018)

    def highlight(self, *keys):
        for k, b in self.boxes.items():
            hot = k in keys
            b["frameColor"] = HOT_BG if hot else BOX_BG
            self.labels[k].setFg((1, 1, 1, 1) if hot else TXT)
        self.current = keys

    def set_layer(self, key):
        """key: an i18n key ("arch.layers" or "arch.layer2")."""
        self.layer_key = key
        self.layer_text.setText(display(t(key)))

    def relabel(self):
        """New language: every box label, the layer text, and the font."""
        f = ui_font()
        for key, x, z, w, h in BLOCKS:
            label = t("arch." + key)
            lab = self.labels[key]
            lab.setFont(f)
            lab.setText(display(label))
            lab.setPos(x, z - 0.01 + 0.014 * label.count("\n"))
        self.layer_text.setFont(f)
        self.layer_text.setText(display(t(self.layer_key)))
        self.plus.setFont(f)

    def show(self):
        self.root.show()

    def hide(self):
        self.root.hide()
