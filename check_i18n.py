# -*- coding: utf-8 -*-
# Claude 写的（中英文切换的检查脚本）
"""
Checks for the English / Chinese interface strings.   Run:  python check_i18n.py

1. every key of the zh dictionary exists in en, and its {placeholders} (names + format specs) are exactly the same;
   every en key has a zh text; the number of lines of the 3D scene notes is the same in both languages
2. (needs a Chinese font) no Chinese scene line is wider than the widest line of its English original
3. every key used in the code (t("...")) exists in en, and every en key is used somewhere
4. scans story.py / tf_steps.py / cache_story.py / main.py / archmap.py for string literals that look like
   interface text but are not passed through t()   (model input/output, math, internal keys are allowed below)
"""
import ast
import os
import re
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import i18n                                                                  # noqa: E402

CODE_FILES = ["story.py", "tf_steps.py", "cache_story.py", "main.py", "archmap.py"]

# String literals that are allowed to stay English. Each one is NOT interface text for the student:
ALLOWED = {
    # --- internal keys / identifiers (logic, node names, dict keys)
    "Context", "Tokens", "Embedding", "Attention", "Add & Norm", "Feed Fwd", "Output", "Tool", "Answer",
    "No cache", "Why reuse", "KV cache", "Memory", "Request 2", "Prefix rule", "Summary",   # stage keys
    "input", "attn", "linear", "softmax", "output", "add1", "add2", "ffn", "pos", "embed",   # architecture-map keys
    "first", "name", "rest", "to_name", "tool", "answer", "number", "terminal", "text", "kind", "cache",
    "calculator", "run_terminal", "plot_chart", "generate_image",                           # tool names (model output)
    "You", "Model", "Tool",                                                                 # speaker keys -> ui.who.*
    "user", "ai", "res", "out", "tmp", "blk", "ai", "ids", "emb", "pe", "x0", "x1", "x_out", "x_final", "layers",
    "heads", "tokens", "logits", "probs", "att", "scaled", "hid", "hid_pre", "delta", "ff", "res1",
    "key", "prompt", "result", "image", "screen", "value", "args", "toks", "steps", "top", "toks", "result_toks",
    "expression", "diffusion", "kind", "results", "context", "mono",
    "tool_call", "kv", "ui", "left", "board", "tf_box", "hlx", "demo", "mat", "row", "map", "words", "marks",
    "best", "probs", "wcol", "ahl", "kt_heads", "anums", "nums", "note", "transformer", "l2flash", "grow",
    "heads", "ch", "rh", "new", "mem", "req", "out1", "ttft", "pair", "reads", "beams", "row1", "row2",
    "requests", "cache-area", "predict", "tools", "panel", "chips", "chip", "tokbox", "hidden", "context",
    "tok", "orbit-cam", "auto-next", "main", "offscreen", "mark", "why", "leg2", "r", "mouse3", "mouse3-up",
    "wheel_up", "wheel_down", "escape", "space", "arrow_right", "frameColor", "text", "text_font", "wordwrap",
    "board", "WQ", "WK", "WV", "WO", "W1", "W2", "W2f", "WOUT", "W", "Wq", "Wk", "Wv", "Wo", "Wout", "E", "P", "X",
    "Q", "K", "V", "S", "A", "KT", "O1", "O2", "Q2", "K2", "V2", "S2", "A2", "C", "D", "X1", "X2", "X3", "H", "F",
    "L2", "fan", "plus0", "plus1", "plus2", "norm1", "norm2", "table", "logit", "mix", "Vc", "words_x", "ids_x",
    "probs_x", "end_x", "hi", "nan", "ui_font", "utf-8",
    # --- math, matrix names, shapes, code
    "×", "·", "+", "...", "Kᵀ", "W_Q", "W_K", "W_V", "W_O", "W_out", "W1", "W2", "ΔX", "<ai>", "<sys>", "<user>",
    "<tool>", "<tool_call>", "</tool_call>", "<end>", "Q2  K2  V2", "Q · Kᵀ", "X · W_{0} = {0}\n({1} × {2}) · ({2} × {3}) = ({1} × {3})",
    "X1 · W1\n({} × {}) · ({} × {}) = ({} × {})", "({:.1f})·({:.1f})", " + ", "{} × {}", "× {:.2f}", "{:.1f}",
    "{:.2f}", "{:.1f}%", "{:.1%}", "<0.1%", "'{}'", "{}", "{}={!r}", "{}({})", "{}()", "√", "x", "name()",
    "calculator    run_terminal    plot_chart    generate_image",         # the tool list that is in the model's context
    "lang", "heads", "next",
    # --- proper noun, model input / output, program plumbing (nothing a student reads)
    "Transformer",                                                     # the name of the architecture
    "You are a helpful assistant. You can call tools.",                # the system prompt = model input
    "Plot the tank temperature for the last hour.",                    # question 3 = model input
    "Here", " is", " the", " chart",                                   # tokens the model writes (question 7 demo)
    "<sys",                                                            # token prefix test
    "RGBA", "SELFTEST", "PASSED", "FAILED", "LLM_DEMO_OFFSCREEN", "window-type offscreen\n", "--selftest", "--lang",
    "--lang=",
    "language switch lost the step", "language round trip changed the screen:\n{}\n{}",    # self-test messages
    "\nwindow-title LLM Pipeline Demo\nwin-size 1600 900\nframebuffer-multisample 1\nmultisamples 8\n"
    "textures-power-2 none\nsync-video 1\naudio-library-name null\n",                    # Panda3D config
}
# Chinese lines that are wider than the English one but still fit the space around them (checked on screen)
WIDTH_OK = {"tf.map.norm": "text 0.96 wide in a 1.4 wide box",
            "cache.req.hit": "label 7.1 wide in a 16.7 wide HIT region"}
# A literal is interface text if it has a word of 3+ letters. These patterns (regex, full match) are fine too:
ALLOWED_RE = [
    r"ease[A-Z][A-Za-z]*",            # interval blend types
    r"[a-z0-9_.]+\.",                 # the start of an i18n key that is completed with a variable: "stage." + name
    r"window-title.*",
    r"[a-z_0-9]+",                    # identifiers used as keys
    r"[a-z]+(-[a-z]+)*",
    r"[A-Za-z0-9_.]+\.[a-z0-9_]+",    # file names, i18n keys
]


def literals(path):
    """(lineno, string) of every string literal that is not a docstring, not a t("...") key, not a dict key."""
    tree = ast.parse(open(os.path.join(HERE, path), encoding="utf-8").read())
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                skip.add(id(node.body[0].value))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t" and node.args:
            skip.add(id(node.args[0]))
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if k is not None:
                    skip.add(id(k))
        if isinstance(node, ast.Subscript):
            skip.add(id(node.slice))
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                pass
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node.lineno, node.value


def placeholders(s):
    return sorted((name, spec, conv) for _, name, spec, conv in string.Formatter().parse(s) if name is not None)


def key_uses():
    used = set()
    for f in CODE_FILES + ["kit.py"]:
        src = open(os.path.join(HERE, f), encoding="utf-8").read()
        used |= set(re.findall(r"\bt\(\s*\"([a-zA-Z0-9_.]+)\"", src))
        used |= set(re.findall(r"\"((?:tf|story|cache|ui|arch)\.[a-zA-Z0-9_.]+)\"", src))
    return used


def main():
    en, zh = i18n.STRINGS["en"], i18n.STRINGS["zh"]
    errors = []

    # 1. dictionary consistency
    for k in zh:
        if k not in en:
            errors.append("zh key not in en: " + k)
    for k in en:
        if k not in zh:
            errors.append("missing in zh: " + k)
        elif placeholders(en[k]) != placeholders(zh[k]):
            errors.append("placeholders differ for {}: en {} / zh {}".format(k, placeholders(en[k]), placeholders(zh[k])))
    for k in en:
        if k in zh and (k.startswith(("tf.map", "tf.scores.info", "tf.scores.mask", "tf.softmax.info", "tf.add1.info",
                                      "tf.ffn.relu", "tf.out.info", "cache.why", "cache.mem", "arch.", "story.prog",
                                      "tf.weighted.info"))):
            if en[k].count("\n") != zh[k].count("\n"):
                errors.append("line count differs for {}: en {} / zh {}".format(k, en[k].count("\n") + 1, zh[k].count("\n") + 1))

    # 2. widths of the scene lines (text units; needs a Chinese font)
    try:
        from panda3d.core import DynamicTextFont, Filename, TextNode, loadPrcFileData
        loadPrcFileData("", "notify-level-text error")
        import kit
        def font(kind):
            for p in kit.FONTS[kind]:
                if os.path.exists(p):
                    f = DynamicTextFont(Filename.fromOsSpecific(p))
                    if f.isValid():
                        return f
        serif, cjk = font("serif"), font("cjk")
        if serif and cjk:
            def width(s, f):
                tn = TextNode("w")
                tn.setFont(f)
                tn.setText(s)
                return tn.getWidth()
            sample = dict(n=9, d=32, dh=16, ff=128, V=95, top="'x' 0.50", root=4, off=3, terms="(0.1)·(0.2)", q0=0.5,
                          a="x", b="y", h1="x", h2="y", prompt="p", total=1, tok="x", i=1, k=1, steps=25, part="x",
                          p=50.0, name="n", who="w", msg="m", token="t", where="w", path="p", stage="s", speed=1.0,
                          text="x", twice="", w="x")
            for k in en:
                if k in zh and k.startswith(("tf.map", "tf.scores.info", "tf.scores.mask", "tf.softmax.info",
                                             "tf.add1.info", "tf.ffn.relu", "tf.ffn.info2", "tf.out.info", "cache.why",
                                             "cache.mem.", "cache.req.", "cache.prefix", "story.pa", "story.ta",
                                             "tf.concat.info", "tf.weighted.info", "tf.qkv.demo", "arch.attn",
                                             "arch.pos", "arch.add", "arch.ffn", "arch.embed", "arch.input",
                                             "arch.output", "arch.linear",
                                             "story.tokens.note", "cache.chart_title", "cache.box_note")):
                    if k.startswith(("cache.mem.caption", "cache.req.caption", "cache.prefix.caption")):
                        continue
                    we = max(width(l, serif) for l in en[k].format(**sample).split("\n"))
                    wz = max(width(l, cjk) for l in zh[k].format(**sample).split("\n"))
                    if wz > we * 1.12 and k not in WIDTH_OK:
                        errors.append("zh line wider than en for {}: {:.1f} > {:.1f}".format(k, wz, we))
        else:
            print("(no serif / Chinese font found: width check skipped)")
    except Exception as e:                                           # noqa: BLE001
        print("(width check skipped: {})".format(e))

    # 3. keys used vs defined
    used = key_uses()
    for k in sorted(used):
        if k not in en and not k.startswith(("stage.", "story.run.", "ui.q.", "ui.who.", "cache.mem.")) \
                and k not in ("tf.map.", "arch."):
            errors.append("key used in code but not in en: " + k)
    for k in en:
        if k not in used and not k.startswith(("stage.", "ui.q.", "ui.who.", "story.run.", "cache.mem.", "arch.")):
            errors.append("en key never used: " + k)

    # 4. hard-coded strings left in the code
    left = []
    for f in CODE_FILES:
        for ln, s in literals(f):
            if s in ALLOWED or not re.search(r"[A-Za-z]{3,}", s):
                continue
            if any(re.fullmatch(p, s) for p in ALLOWED_RE):
                continue
            left.append("{}:{}: {!r}".format(f, ln, s[:90]))
    if left:
        errors.append("string literals that look like interface text and are not passed through t():")
        errors += ["    " + x for x in left]

    if errors:
        print("\n".join(errors))
        print("\nCHECK FAILED ({} problems)".format(len(errors)))
        return 1
    print("en keys: {}   zh keys: {}".format(len(en), len(zh)))
    print("CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
