# -*- coding: utf-8 -*-
# Claude Opus 写的；中英文切换由 Claude 新增（本文件为新增）
"""
Interface language (English / Chinese).

    t("ui.btn.next")                  -> the string in the current language
    t("tf.qkv.caption", n=9, dh=16)   -> the same, with {n} and {dh} filled in (both languages use the same names)

NOT translated (on purpose): what goes into or comes out of the model (the question, tokens, chat messages, the
tool-call JSON, tool results, the answer - the small model only knows English), math names and shapes (Q, K, V,
W_Q, softmax, "9 x 32" ...), numbers, code, and the internal keys (the stage names "Attention", "Add & Norm" ...
are only used for logic; what is shown is t("stage.<name>")).

A key that is missing in zh falls back to en; a key that is missing in both is returned as it is.
Keys are dotted names. English text must stay exactly as it was before the translation was added.
"""
import json
import os
import sys

LANG = "en"
LANGS = ("en", "zh")


def set_lang(lang):
    global LANG
    LANG = lang if lang in STRINGS else "en"


def t(key, **kw):
    s = STRINGS.get(LANG, {}).get(key)
    if s is None:
        s = STRINGS["en"].get(key)
    if s is None:
        return key
    return s.format(**kw)


# ------------------------------------------------------------------ settings.json (next to main.py)
def _settings_path():
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "settings.json")


def load_lang():
    """The saved language, or "en" if there is none or the file cannot be read."""
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            lang = json.load(f).get("lang")
        return lang if lang in LANGS else "en"
    except Exception:
        return "en"


def save_lang(lang):
    """Remember the language for the next start. Failures are ignored."""
    try:
        data = {}
        try:
            with open(_settings_path(), encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        data["lang"] = lang
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ====================================================================== English
EN = {
    # ---------------------------------------------------------------- main.py: panels, buttons
    "ui.title": "LLM Pipeline",
    "ui.arch_title": "model architecture",
    "ui.conversation": "Conversation",
    "ui.hint": "right-drag: rotate    wheel: zoom",
    "ui.btn.next": "Next  (Space)",
    "ui.btn.auto_off": "Auto: OFF  (A)",
    "ui.btn.auto_on": "Auto: ON  (A)",
    "ui.btn.speed": "Speed {speed:g}x  (+/-)",
    "ui.btn.restart": "Restart  (R)",
    "ui.btn.deep_on": "Deep dive: ON  (D)",
    "ui.btn.deep_off": "Deep dive: OFF  (D)",
    "ui.btn.cam_follow": "Camera: follow  (C)",
    "ui.btn.cam_free": "Camera: free  (C)",
    "ui.btn.glow_on": "Glow: ON  (G)",
    "ui.btn.glow_off": "Glow: OFF  (G)",
    "ui.btn.hide": "Hide panels  (H)",
    "ui.btn.show": "Show panels  (H)",
    "ui.btn.lang": "Language: English (L)",
    "ui.q.1": "Hi! Who are you?",
    "ui.q.2": "What is 17% of 2350?",
    "ui.q.3": "Tank temperature chart",
    "ui.q.4": "Cat in a cleanroom suit",
    "ui.q.5": "Count files (terminal)",
    "ui.q.6": "Files -> bar chart (2 tools)",
    "ui.q.7": "Why so fast? (cache)",
    "ui.who.you": "You",
    "ui.who.model": "Model",
    "ui.who.tool": "Tool",
    "ui.chat_line": "{who}:  {msg}",
    "ui.trk.following": "following: {token}",
    "ui.trk.now": "now:\n{where}",
    "ui.trk.path": "path so far:\n{path}",
    "ui.info.start": "{n} steps   -   Space = next,  A = auto",
    "ui.info.step": "Step {i} / {n}   -   {stage}",
    "ui.info.done": "Finished  -  pick another question (1-7) or Restart",
    "ui.intro": 'Question {i}:  "{prompt}"\nPress Space (or Next) to follow it through the model.',
    "ui.intro_cache_start": "\nPress Space (or Next) to start.",
    # ---------------------------------------------------------------- stage chips (the key is the English stage name)
    "stage.Context": "Context",
    "stage.Tokens": "Tokens",
    "stage.Embedding": "Embedding",
    "stage.Attention": "Attention",
    "stage.Add & Norm": "Add & Norm",
    "stage.Feed Fwd": "Feed Fwd",
    "stage.Output": "Output",
    "stage.Tool": "Tool",
    "stage.Answer": "Answer",
    "stage.Compute": "Compute",
    "stage.No cache": "No cache",
    "stage.Why same": "Why same",
    "stage.KV cache": "KV cache",
    "stage.Work": "Work",
    "stage.Cost": "Money, power",
    "stage.Next request": "Next request",
    "stage.Cache miss": "Cache miss",
    "stage.Summary": "Summary",
    # ---------------------------------------------------------------- archmap.py
    "arch.output": "next-token probabilities",
    "arch.softmax": "Softmax",
    "arch.linear": "Linear  (x W_out)",
    "arch.add2": "Add & Norm",
    "arch.ffn": "Feed Forward",
    "arch.add1": "Add & Norm",
    "arch.attn": "Masked Multi-Head\nAttention",
    "arch.pos": "Positional\nEncoding",
    "arch.embed": "Token Embedding",
    "arch.input": "input tokens",
    "arch.layers": "x 2 layers",
    "arch.layer2": "layer 2 of 2",
    # ---------------------------------------------------------------- story.py
    "story.sys": "system:",
    "story.tools": "tools:",
    "story.user": "user:",
    "story.context.caption": (
        "Your message is never sent alone. A hidden SYSTEM PROMPT (rules + the list of tools) comes first. "
        "Together they form the CONTEXT - the only thing the model sees. Everything from here on is computed "
        "by a REAL tiny Transformer trained for this demo."),
    "story.tokens.note": ("<sys> = the whole system prompt, squeezed into ONE token in this tiny model.   "
                          "<user> / <ai> mark who is speaking."),
    "story.tokens.part": " Look: '{part}' is only part of a word.",
    "story.tokens.follow": " The yellow one, <ai>, is the token we will follow all the way through the model.",
    "story.tokens.caption": (
        "TOKENIZER: the text is cut into TOKENS (words or pieces of words) and each token gets its ID number "
        "in the model's vocabulary ({V} tokens)."),
    "story.pa.context": "context  (everything the model reads)",
    "story.pa.model": "the whole model:  2 layers x (attention + feed-forward)",
    "story.pa.probs": "next-token probabilities",
    "story.pa.append": "append the new token, run again",
    "story.predict.tool_first": (
        "The winning token is APPENDED to the context, and to get the next token the whole model runs "
        "again (every step you just saw). The first token here is the special <tool_call> ({p:.1f}%) - "
        "'deciding to use a tool' is just predicting this token!"),
    "story.predict.tool_again": (
        "The tool result is now in the context, and the model runs again on EVERYTHING. It decides "
        "it needs another tool: <tool_call> again ({p:.1f}%)."),
    "story.predict.name": (
        "Which tool? The model writes the tool's NAME as a token: '{name}' ({p:.1f}%). Writing the name does "
        "not run anything yet - it is still just text."),
    "story.predict.reply": (
        "The tool result is now part of the context, so this time the most likely first token is normal "
        "text: '{tok}' ({p:.1f}%). The model starts its reply."),
    "story.predict.greeting": (
        "A greeting needs no tool. Here two tokens are almost equally likely: '{tok}' and '{alt}' (about "
        "{p:.0f}% each). The model SAMPLES - it rolled '{tok}' this time. Press R: next time it may start "
        "with '{alt}'. That is why answers vary."),
    "story.fast.to_name": (
        "Again and again: each new token is appended to the context and the whole model runs once more to "
        "predict the next one (orange chips). This loop is called AUTOREGRESSIVE generation."),
    "story.fast.tool_rest": (
        "The model finishes the tool call: the arguments ({n} more tokens), then </tool_call>. It is still "
        "only text - but text in a strict format that a program can read."),
    "story.fast.reply": (
        "The reply is generated token by token ({n} more), until the model predicts a special "
        "'end of text' token."),
    "story.ta.title": "program  (the harness)",
    "story.ta.sub": "ordinary code around the model",
    "story.prog.output": "model output:",
    "story.prog.found": "found <tool_call> -> stop the model",
    "story.prog.read": "read the tool name and arguments",
    "story.prog.call": "call:",
    "story.prog.caption": (
        "The model itself cannot run code, open files or draw pictures - it only produced text. The program "
        "around it (the 'harness') watches the output. When it sees <tool_call>, it stops the model, reads "
        "the tool name and arguments, and calls the matching Python function."),
    "story.run.calculator": (
        "Ordinary Python computes the exact result. LLMs see numbers as tokens (remember 235 + 0?) "
        "and can make arithmetic mistakes - a calculator tool is far more reliable."),
    "story.run.run_terminal": (
        "The program REALLY runs this command on your computer, right now. Only whitelisted, "
        "read-only commands are allowed - giving an AI a terminal is powerful, so real systems "
        "ask the user first."),
    "story.run.plot_chart": (
        "matplotlib (a normal Python library) draws the chart. The language model never drew a "
        "single pixel - it only chose the chart type, title and data."),
    "story.run.generate_image": (
        "A DIFFERENT AI (an image / diffusion model) starts from pure noise and removes noise "
        "step by step, guided by the prompt (simulated here). Note: the LLM rewrote your "
        "request into a more detailed prompt!"),
    "story.run.denoise": "denoising step {k}/{steps}",
    "story.chip.tool_result": "tool result",
    "story.chip.tool_call": "tool call ({n} tokens)",
    "story.chip.more": "+{n} more",
    "story.return.caption": (
        "The program turns the tool's result into text and appends it to the context as a new message "
        "(green chips). The model gets back a short text - numbers, file names, a file path - not the "
        "picture itself."),
    "story.answer.label": "Answer:  {text}",
    "story.answer.head": ("Summary:  your words -> tokens -> vectors (+ position) -> 2 x [attention, add & norm, "
                          "feed-forward, add & norm] -> next-token probabilities\n"),
    "story.answer.loop": "  -> <tool_call> -> the program runs real code -> result back into the context{twice}\n",
    "story.answer.twice": " (twice here)",
    "story.answer.tail": "  -> final answer.   The LLM itself only ever predicts the next token.",
    # ---------------------------------------------------------------- tf_steps.py: the map
    "tf.map.P": "P  (position)",
    "tf.map.S": "scores",
    "tf.map.A": "weights A",
    "tf.map.O1": "head 1 out",
    "tf.map.S2": "scores (head 2)",
    "tf.map.A2": "weights (head 2)",
    "tf.map.O2": "head 2 out",
    "tf.map.C": "concat",
    "tf.map.D": "ΔX  (change)",
    "tf.map.H": "hidden",
    "tf.map.X2": "X2  (layer 1 out)",
    "tf.map.X3": "X3  (final)",
    "tf.map.own": "× its own\nW_Q, W_K, W_V",
    "tf.map.head2": "head 2",
    "tf.map.head1": "head 1",
    "tf.map.table": "embedding table",
    "tf.map.norm": "norm",
    "tf.map.layer2": "layer 2",
    "tf.map.layer2_note": ("the same steps again\n(attention, add & norm,\nfeed forward, add & norm)\n"
                           "with its own weights"),
    "tf.map.layer1": "layer 1",
    "tf.map.attention": "ATTENTION",
    "tf.map.ffn": "FEED FORWARD",
    "tf.map.output": "OUTPUT",
    "tf.map.res1": "residual: X skips attention and is ADDED back",
    "tf.map.res2": "residual: X1 skips feed forward",
    # ---------------------------------------------------------------- tf_steps.py: tracker
    "tf.track.embed": "token embedding",
    "tf.track.pos": "added position",
    "tf.track.add1": "layer 1: add & norm",
    "tf.track.l1end": "end of layer 1",
    "tf.track.l2end": "end of layer 2",
    "tf.track.pred": "used for the prediction",
    # ---------------------------------------------------------------- tf_steps.py: steps
    "tf.embed.ids": "token   ID",
    "tf.embed.caption": (
        "THE MAP: everything the Transformer does, from left to right. Every token is one ROW of numbers and "
        "keeps its row the whole way; the last row (yellow) is <ai>, whose row will predict the next word.  "
        "STEP 1, EMBEDDING: each token ID picks its row of {d} numbers from a learned table ({V} x {d}). "
        "Stacked up, the rows form matrix E ({n} x {d}). Red = positive, blue = negative."),
    "tf.pos.caption": (
        "STEP 2, POSITION: so far the model would not know the ORDER of the rows. Each position gets a fixed "
        "pattern of numbers (P, made of sine waves) and it is simply ADDED, number by number: X = E + P. "
        "X ({n} x {d}) goes into the attention block."),
    "tf.qkv.demo1": "<ai>'s row of X  ·  first column of W_Q  =  first number of <ai>'s row of Q",
    "tf.qkv.demo2": "{terms} + ...   ({d} products added up)  =  {q0:.2f}",
    "tf.qkv.caption": (
        "STEP 3, Q K V: three ordinary matrix multiplications with learned matrices: X · W_Q = Q, X · W_K = "
        "K, X · W_V = V (each {n} x {dh}). One number is worked out on top: <ai>'s row of X times the first "
        "column of W_Q. The idea: Q = what each token is LOOKING FOR, K = what it OFFERS, V = what it will "
        "PASS ON."),
    "tf.scores.info": "Q · Kᵀ = scores\n({n} × {dh}) · ({dh} × {n}) = ({n} × {n})",
    "tf.scores.info2": ("cell (i, j) = how well token i's\nquestion (Q) matches token j (K).\n"
                        "Then all ÷ √{dh} = {root:.0f}"),
    "tf.scores.mask": "MASK: no looking at LATER\ntokens -> grey (−∞)",
    "tf.scores.caption": (
        "STEP 4, SCORES: Q · Kᵀ is again a matrix multiplication. The K rows turn into columns (Kᵀ, on "
        "top), so cell (i, j) = row i of Q · column j of Kᵀ = how well token i's question matches token j. "
        "The yellow row is <ai> compared with every token. Then the MASK: no token may look at tokens that "
        "come AFTER it (grey). <ai> is last, so it sees everything."),
    "tf.softmax.info": "softmax, row by row:\nall weights ≥ 0,\nevery row adds up to 1\n\nbright = big weight",
    "tf.softmax.caption": (
        "STEP 5, WEIGHTS: softmax turns every row of scores into WEIGHTS (e^score, then divide by the row's "
        "sum): all positive, every row adds up to 1, bright = big. The yellow row says how much <ai> pays "
        "attention to each token - most to {top}."),
    "tf.weighted.info": "A · V = head output\n({n} × {n}) · ({n} × {dh}) = ({n} × {dh})",
    "tf.weighted.caption": (
        "STEP 6, MIX: A · V is a matrix multiplication again, and what it does is MIX the V rows: <ai>'s new "
        "row = (its weight for token 1) × V row 1 + (weight 2) × V row 2 + ... Strong weights stay bright, "
        "weak ones fade, then all are added up. So <ai> now carries information from '{a}' and '{b}'. Every "
        "row does the same with its own weights."),
    "tf.head2.caption": (
        "STEP 7, HEAD 2: attention is done twice side by side, with a second set of W_Q, W_K, W_V - the same "
        "steps 3 to 6. Each head can look for a different kind of relation. <ai> - head 1: {h1};  head 2: {h2}."),
    "tf.concat.heads": "head 1 | head 2",
    "tf.concat.info": "concat · W_O = ΔX    ({n} × {d}) · ({d} × {d}) = ({n} × {d})",
    "tf.concat.caption": (
        "STEP 8: the two head outputs are put side by side (16 + 16 = {d} numbers per row) and multiplied by "
        "one more learned matrix W_O. The result ΔX is the CHANGE that attention wants to make to every "
        "token's row."),
    "tf.add1.info": "X + ΔX, then NORM each row:\nminus its mean, ÷ its spread,\n× gain + bias (learned)",
    "tf.add1.caption": (
        "STEP 9, ADD & NORM: X itself jumps over the whole attention block (red line, the 'residual') and "
        "is ADDED to ΔX: X + ΔX. So attention only adds a correction and nothing is lost. Then every row is "
        "normalized (mean 0, spread 1) to keep the numbers in a stable range. Result: X1 ({n} x {d})."),
    "tf.ffn.relu": "ReLU: every negative number -> 0\n(<ai>: {off} of {ff} switched off)",
    "tf.ffn.info2": "hidden · W2\n({n} × {ff}) · ({ff} × {d}) = ({n} × {d})",
    "tf.ffn.caption": (
        "STEP 10, FEED FORWARD: two more matrix multiplications: X1 · W1 gives {ff} numbers per token, ReLU sets "
        "every negative one to 0 (dark), then · W2 brings it back to {d}. Each row is processed ON ITS OWN - "
        "no mixing between rows. Attention = tokens talk to each other; feed forward = each token thinks "
        "for itself."),
    "tf.add2.caption": (
        "STEP 11, ADD & NORM again: X1 jumps over the feed forward block and is added to F, then every row "
        "is normalized. X2 is the output of LAYER 1 - the same shape as X ({n} x {d}), but now every row knows "
        "something about the other tokens."),
    "tf.layer2.caption": (
        "STEP 12, LAYER 2: steps 3-11 once more, with its own learned matrices. In layer 2, <ai> looks most "
        "at: {top}. Real models stack 30-100 such layers."),
    "tf.out.probs": "softmax -> probabilities",
    "tf.out.info": ("only <ai>'s row:  (1 × {d}) · ({d} × {V}) = (1 × {V})\n"
                    "one score for every token the model knows"),
    "tf.out.caption": (
        "STEP 13, OUTPUT: only <ai>'s row - the yellow row we followed all the way - is used now. One last "
        "matrix multiplication with W_out gives one score for each of the {V} tokens the model knows; softmax "
        "turns the scores into probabilities. Winner: '{w}' ({p:.1%})."),
    "tf.out.extra": (" (Almost 100%: this tiny model learned these few conversations by heart - big models are much "
                     "less certain.)"),
    "tf.quick.caption": (
        "Inside the Transformer (deep dive OFF - press D to see every step): attention lets the token rows "
        "exchange information, feed forward processes each row, and that twice (2 layers). Follow the "
        "yellow <ai> row."),
    # ---------------------------------------------------------------- cache_story.py
    "cache.intro": (
        'Question 7: what does a cache actually save? We follow the green K matrix of question 3 '
        '("{prompt}") - every number is the real tiny model. Step by step: what "computing" is, what '
        '"computing again" is, how the cache saves it - and what that means in money and electricity.'),
    "cache.x.note": "{n} tokens, each one a row of {d} numbers  (the X of question 3)",
    "cache.x.caption": (
        "First a reminder. The {n} tokens of the question each became a row of {d} numbers; stacked up they "
        "are the matrix X. We want to find out what the model really does when it \"computes\" - and what "
        "\"computing again\" means."),
    "cache.k.demo": "one number of K = {terms} + ...   ({d} multiplications)  = {v:.2f}",
    "cache.k.count": "{rows} rows done x {per} = {n} multiplications",
    "cache.k.count2": "K and V together: {n} multiplications  (and that is only 1 head)",
    "cache.k.caption": (
        "THIS is computing: one row of X ({d} numbers) times one column of W_K ({d} numbers) - {d} "
        "multiplications, added up - gives ONE number of K. A row of K has {dh} numbers = {per} "
        "multiplications. Watch the {n} rows appear one by one. V is made the same way with W_V. Inside a "
        "real model every layer is full of exactly these multiplications."),
    "cache.first.out": "the model's first new token: '{tok}'  ({p:.1f} %)",
    "cache.first.note": "with the new token X has {n} rows. Must the rows above be computed again?",
    "cache.first.caption": (
        "The model has predicted its first new token: '{tok}'. To write the SECOND token, '{tok}' is added "
        "to the input: X now has {n} rows (the new one at the bottom, yellow). The question is: what about "
        "the rows above - compute them again?"),
    "cache.redo.label": "K  (2nd time)",
    "cache.redo.arrow": "no cache: everything again",
    "cache.redo.caption": (
        "WITHOUT a cache: all {n} rows go through the model again, multiplied from scratch. The orange K on "
        "the right is the result of this second pass. Look at the counter - for K alone that is {n} x 512 "
        "multiplications, and the model does the same for Q, V, both heads, the feed forward and layer 2."),
    "cache.cmp.same": "= difference {d:.3f}",
    "cache.cmp.new": "new row",
    "cache.cmp.waste": ("these {n0} rows: wasted work\n"
                        "for K alone: {k} multiplications again\n"
                        "the whole model (2 layers, every matrix):\n"
                        "   all rows again:  {all}\n"
                        "   only the new row:  {new}"),
    "cache.cmp.caption": (
        "Now compare row by row. The first {n0} rows are EXACTLY the old ones: difference 0.000 (same input "
        "rows, same W_K). Only the last row is new. So \"computing again\" means: multiplying these {n0} "
        "rows once more and getting not a single different number. Pure waste."),
    "cache.mask.label": "attention weights  (layer 1, head 1)",
    "cache.mask.legend": "black = masked, not allowed to look;  brighter = looks more",
    "cache.mod.diff": "difference to A",
    "cache.mask.rows": "who looks",
    "cache.mask.cols": "is looked at  ->",
    "cache.mask.col_note": ("column of the new token (yellow):\n"
                            "the {n0} rows above are black = masked\n"
                            "old tokens never look at the new one"),
    "cache.mask.row_note": "only the new row\nlooks at every token",
    "cache.mask.caption": (
        "Why MUST they be the same? The attention table: every row may only look at itself and the rows "
        "above (the black upper-right part is masked). The new token is the last row and the last column. "
        "In that column every old row is black - old tokens never read the new one, so nothing about them "
        "can change."),
    "cache.l2.old": "layer 2 K  (1st time)",
    "cache.l2.new": "layer 2 K  (2nd time)",
    "cache.l2.diff": "difference",
    "cache.l2.why": ("in layer 1 every row only looked at itself and the rows above,\n"
                     "so the old rows enter layer 2 unchanged  ->  layer 2's K and V are the same.\n"
                     "The feed forward works on each row alone: also unchanged."),
    "cache.l2.caption": (
        "Layer 2 too: the old rows leave layer 1 unchanged (they never saw the new token), so they enter "
        "layer 2 unchanged, and layer 2's K and V are identical again. The feed forward handles every row on "
        "its own. So in EVERY layer, the K and V of old tokens can be kept."),
    "cache.store.shelf": "KV cache (the notebook)  -  kept in GPU memory",
    "cache.store.l1h2": "layer 1  head 2",
    "cache.store.l2h1": "layer 2  head 1",
    "cache.store.l2h2": "layer 2  head 2",
    "cache.store.mem": "per token: {per} numbers (K and V x 2 layers x 2 heads x 16);  {n} tokens: {tot}",
    "cache.store.caption": (
        "They never change - so keep them. The K and V of every token, in every layer and every head, go "
        "onto the shelf: the KV CACHE, the model's notebook. Only K and V: a new token later needs the old "
        "tokens' K (to compare with) and V (to mix) - nothing else of them."),
    "cache.one.xrow": "only the row of the new token '{tok}'",
    "cache.one.q": "q = x · W_Q  (the new token's query)",
    "cache.one.scores": "ONE row of scores: q against the {n} K on the shelf -> softmax",
    "cache.one.result": ("next token: '{tok}'\n"
                         "with cache {pc:.4f} %,  without cache {pf:.4f} %\n"
                         "largest difference {diff:.0e} (computer rounding) - the same result"),
    "cache.one.count": ("multiplications (whole model):\n"
                        "   no cache, all 14 rows again:  {all}\n"
                        "   with cache, 1 row:  {new}"),
    "cache.one.caption": (
        "WITH the cache, writing the second token: only the new row is computed. x · W_K and x · W_V give its "
        "K and V - they go to the bottom of the shelf. x · W_Q gives q, which is compared with all {n1} K on "
        "the shelf: ONE row of scores, not a whole table. The prediction is exactly the same, for about a "
        "tenth of the work."),
    "cache.bars.title": "multiplications to write each new token  (the whole tiny model)",
    "cache.bars.tok": "token {k}\n({n} rows)",
    "cache.bars.legend": "grey = no cache (all rows again)\nyellow = KV cache (one row)",
    "cache.bars.total": ("the whole tool call, {k} tokens:\n"
                         "   no cache  {no} multiplications\n"
                         "   with cache  {ca}\n"
                         "   {x:.0f} times less"),
    "cache.bars.caption": (
        "Token by token: without a cache (grey) every new token recomputes ALL rows, so the work keeps "
        "growing. With the cache (yellow) it is one row each time - almost flat. Over the whole tool call the "
        "gap gets huge."),
    "cache.m1.title": "a chat already has {hist} tokens; you send {new} more",
    "cache.m1.no": "no cache",
    "cache.m1.hit": "cache hit",
    "cache.m1.r_tok": "tokens to compute",
    "cache.m1.r_usd": "input price (US$)",
    "cache.m1.r_yen": "in yen (about)",
    "cache.m1.r_time": "time on one GPU",
    "cache.m1.r_energy": "electricity (estimate)",
    "cache.m1.bar_no": "{n} tokens computed again",
    "cache.m1.bar_hit": "only {n}",
    "cache.m1.how": (
        "How: price = tokens x the Claude Sonnet 5.5 price list (Oct 2026, per million tokens:\n"
        "   input $2, writing the cache $2.50, reading the cache $0.20);  US$ 1 = 150 yen.\n"
        "Electricity is an ESTIMATE: a made-up model with 70 billion parameters (Claude's size is\n"
        "   not public) needs about 70 billion multiplications per token; one H100 GPU, about 700 W,\n"
        "   200 trillion multiply-adds per second  ->  about {J:.2f} J and {ms:.2f} ms per token (no cooling).\n"
        "Without the cache this one message uses about {phone:.0f} % of a phone battery.\n"
        "Reading the cache moves data, so it is not free - but far less."),
    "cache.m1.caption": (
        "The same idea at real size: a chat with 20,000 tokens so far (system prompt, tool descriptions, "
        "messages), and you send one more sentence. Without a cache 20,050 tokens are computed again; with a "
        "cache hit only the new 50. The input price drops about {x:.0f} times."),
    "cache.m2.title": "a whole chat: {turns} rounds.  system + tools {sys} tokens, every round adds {per}",
    "cache.m2.axis": "round  (input price of each round: grey = no cache, yellow = cache)",
    "cache.m2.sum": ("input price of the whole chat:\n"
                     "   no cache  {no}  (about {noy} yen)\n"
                     "   cache  {hit}  (about {hity} yen),  {x:.1f} times cheaper\n"
                     "tokens computed:  {tn}  vs  {th}\n"
                     "electricity (estimate):  {wn}  vs  {wh}\n"
                     "(round 1 costs 25 % more: writing the cache; from round 2 it pays back)\n"
                     "(the output is the same in both, not counted)"),
    "cache.m2.class": ("a class of {c}, everyone has such a chat once:\n"
                       "   no cache  {no}  (about {noy} yen),  {kn:.2f} kWh\n"
                       "   cache  {hit}  (about {hity} yen),  {kh:.2f} kWh\n"
                       "   {kn:.2f} kWh costs about {en:.0f} yen of electricity (31 yen/kWh)"),
    "cache.m2.caption": (
        "A whole chat of 30 rounds: the context gets longer every round. Without a cache every round reads "
        "the whole, growing history again (the grey bars keep rising). With a cache only the new part is paid "
        "in full, the old part at a tenth of the price. This is why long chats need the cache."),
    "cache.req.r1": "request 1",
    "cache.req.r2": "request 2",
    "cache.req.call": "tool call  {n} tokens",
    "cache.req.result": "tool result  {n} tokens",
    "cache.req.hit": "cache hit: {n} tokens use the K and V on the shelf",
    "cache.req.miss": "computed: {n}",
    "cache.req.why": ("compared token by token from the start; identical = use the stored K and V.\n"
                      "That is why APIs like Claude's charge only a tenth for reading the cache: almost no computing."),
    "cache.req.caption": (
        "The NEXT request (prompt cache): the tool ran, the program appends its result and asks again. The "
        "first {n} tokens are exactly as before, and the server still has their K and V (by default about 5 "
        "minutes) - they are reused. Only the {m} new tokens are computed."),
    "cache.chg.edit": "one word changed: tank -> wafer",
    "cache.chg.layer": "layer {l} K",
    "cache.chg.n1": "layer 1: only the changed row is different",
    "cache.chg.n2": "layer 2: EVERY row from the change down is different\n(the rows below saw it in attention)",
    "cache.chg.rule": ("the cache can only be used up to the first different token.\n"
                       "So: things that never change (system prompt, tools) FIRST,\n"
                       "things that change (the time, the new question) at the END."),
    "cache.chg.caption": (
        "Now change one early word (tank -> wafer). In layer 1 only that row changes. But in layer 2 every "
        "row from the change downwards is different - the rows below looked at it in attention. So the cache "
        "is valid only up to the first different token; everything after it is computed again."),
    "cache.mod.a": "K of model A",
    "cache.mod.b": "K of model B\n(other weights)",
    "cache.mod.note": ("same tokens, different weights  ->  every row of K is different.\n"
                       "the K and V stored by the other model are useless here.\n"
                       "example, a chat with {hist} tokens:\n"
                       "   same model, cache hit:  {hit}  (about {hity} yen)\n"
                       "   switch models:  {sw}  (about {swy} yen), everything written to the cache again"),
    "cache.mod.caption": (
        "Switch to another model: the same tokens, but a different W_K, so K comes out completely different. "
        "The notebook of the old model cannot be read by the new one - the whole chat must be computed again. "
        "That is why switching models in the middle of a long chat costs more tokens."),
    "cache.sum.mem": "the price: the cache takes GPU memory, so the server keeps it only a short time\n"
                     "(Claude API: 5 minutes by default)",
    "cache.sum.caption": (
        "Summary:  computing = multiplying every row by the weights. A new token never changes the old rows' "
        "K and V, so they are stored (KV cache) and each new token needs one row. A next request with the "
        "same beginning reuses them (prompt cache). Change an early word, switch models or wait too long: "
        "everything again - slower, pricier, more electricity."),
}

# ====================================================================== 中文
ZH = {
    # ---------------------------------------------------------------- main.py: panels, buttons
    "ui.title": "LLM 流水线",
    "ui.arch_title": "模型结构",
    "ui.conversation": "对话",
    "ui.hint": "右键拖动：旋转    滚轮：缩放",
    "ui.btn.next": "下一步  (空格)",
    "ui.btn.auto_off": "自动播放：关  (A)",
    "ui.btn.auto_on": "自动播放：开  (A)",
    "ui.btn.speed": "速度 {speed:g}x  (+/-)",
    "ui.btn.restart": "重新开始  (R)",
    "ui.btn.deep_on": "深入讲解：开  (D)",
    "ui.btn.deep_off": "深入讲解：关  (D)",
    "ui.btn.cam_follow": "相机：跟随  (C)",
    "ui.btn.cam_free": "相机：自由  (C)",
    "ui.btn.glow_on": "辉光：开  (G)",
    "ui.btn.glow_off": "辉光：关  (G)",
    "ui.btn.hide": "隐藏面板  (H)",
    "ui.btn.show": "显示面板  (H)",
    "ui.btn.lang": "语言：中文 (L)",
    "ui.q.1": "你好！你是谁？",
    "ui.q.2": "2350 的 17% 是多少？",
    "ui.q.3": "水箱温度曲线图",
    "ui.q.4": "穿洁净服的猫",
    "ui.q.5": "数文件（终端）",
    "ui.q.6": "文件 -> 柱状图（2 个工具）",
    "ui.q.7": "为什么这么快？（缓存）",
    "ui.who.you": "你",
    "ui.who.model": "模型",
    "ui.who.tool": "工具",
    "ui.chat_line": "{who}：{msg}",
    "ui.trk.following": "跟踪：{token}",
    "ui.trk.now": "现在：\n{where}",
    "ui.trk.path": "已走过：\n{path}",
    "ui.info.start": "共 {n} 步   -   空格 = 下一步，A = 自动",
    "ui.info.step": "第 {i} / {n} 步   -   {stage}",
    "ui.info.done": "已结束  -  请选另一道题 (1-7) 或点重新开始",
    "ui.intro": '第 {i} 题：  "{prompt}"\n按空格键（或点“下一步”），跟着它走一遍模型。',
    "ui.intro_cache_start": "\n按空格键（或点“下一步”）开始。",
    # ---------------------------------------------------------------- stage chips
    "stage.Context": "上下文",
    "stage.Tokens": "词元",
    "stage.Embedding": "嵌入",
    "stage.Attention": "注意力",
    "stage.Add & Norm": "残差+归一",
    "stage.Feed Fwd": "前馈网络",
    "stage.Output": "输出",
    "stage.Tool": "工具",
    "stage.Answer": "回答",
    "stage.Compute": "计算",
    "stage.No cache": "无缓存",
    "stage.Why same": "为何相同",
    "stage.KV cache": "KV 缓存",
    "stage.Work": "工作量",
    "stage.Cost": "钱和电",
    "stage.Next request": "下次请求",
    "stage.Cache miss": "缓存失效",
    "stage.Summary": "总结",
    # ---------------------------------------------------------------- archmap.py
    "arch.output": "下一个词元的概率",
    "arch.softmax": "Softmax",
    "arch.linear": "线性层  (x W_out)",
    "arch.add2": "残差 + 归一化",
    "arch.ffn": "前馈网络",
    "arch.add1": "残差 + 归一化",
    "arch.attn": "多头注意力\n（带遮挡）",
    "arch.pos": "位置\n编码",
    "arch.embed": "词元嵌入",
    "arch.input": "输入词元",
    "arch.layers": "x 2 层",
    "arch.layer2": "第 2 层 / 共 2 层",
    # ---------------------------------------------------------------- story.py
    "story.sys": "系统：",
    "story.tools": "工具：",
    "story.user": "用户：",
    "story.context.caption": (
        "你的消息从来不会单独发给模型。前面还藏着一段系统提示词（规则 + 工具清单）。"
        "两部分合在一起就是上下文——模型能看到的全部内容。从这里开始的所有计算，"
        "都来自一个为本演示训练的真实的迷你 Transformer。"),
    "story.tokens.note": ("<sys> = 整段系统提示词，在这个迷你模型里被压成了一个词元。   "
                          "<user> / <ai> 标明是谁在说话。"),
    "story.tokens.part": "注意：'{part}' 只是一个词的一部分。",
    "story.tokens.follow": "黄色的 <ai> 就是我们要一路跟着走完整个模型的词元。",
    "story.tokens.caption": (
        "分词器：把文字切成词元 token（整个词，或词的一部分），每个词元在模型的词表里都有一个编号"
        "（词表共 {V} 个词元）。"),
    "story.pa.context": "上下文  (模型读到的全部内容)",
    "story.pa.model": "整个模型：  2 层 x (注意力 + 前馈网络)",
    "story.pa.probs": "下一个词元的概率",
    "story.pa.append": "把新词元接到末尾，再跑一遍",
    "story.predict.tool_first": (
        "得票最高的词元会被接到上下文末尾。要得到下一个词元，整个模型要重新跑一遍（就是你刚看过的每一步）。"
        "这里的第一个词元是特殊的 <tool_call>（{p:.1f}%）——“决定调用工具”，其实就是预测出这个词元！"),
    "story.predict.tool_again": (
        "工具结果现在已经在上下文里了，模型把全部内容再读一遍。它决定还要再用一个工具："
        "又是 <tool_call>（{p:.1f}%）。"),
    "story.predict.name": (
        "用哪个工具？模型把工具的名字也写成一个词元：'{name}'（{p:.1f}%）。写出名字还不会运行任何东西——"
        "它仍然只是文字。"),
    "story.predict.reply": (
        "工具结果已经是上下文的一部分，所以这次最可能的第一个词元是普通文字：'{tok}'（{p:.1f}%）。"
        "模型开始写回答。"),
    "story.predict.greeting": (
        "打招呼不需要工具。这里有两个词元几乎一样可能：'{tok}' 和 '{alt}'（各约 {p:.0f}%）。"
        "模型是按概率抽样的——这次抽到了 '{tok}'。按 R 重来：下次可能以 '{alt}' 开头。"
        "这就是回答会变的原因。"),
    "story.fast.to_name": (
        "一遍又一遍：每个新词元都接到上下文末尾，整个模型再跑一次，预测下一个词元（橙色小块）。"
        "这个循环叫自回归生成。"),
    "story.fast.tool_rest": (
        "模型把工具调用写完：先是参数（还有 {n} 个词元），最后是 </tool_call>。"
        "它仍然只是文字——不过是格式很严格、程序能读懂的文字。"),
    "story.fast.reply": (
        "回答一个词元一个词元地生成（还有 {n} 个），直到模型预测出特殊的“结束”词元。"),
    "story.ta.title": "程序  (调度程序)",
    "story.ta.sub": "模型外面的普通代码",
    "story.prog.output": "模型输出：",
    "story.prog.found": "发现 <tool_call> -> 停下模型",
    "story.prog.read": "读出工具名和参数",
    "story.prog.call": "调用：",
    "story.prog.caption": (
        "模型自己不能运行代码、打开文件或画图——它只会输出文字。模型外面的程序（调度程序）一直盯着输出。"
        "一看到 <tool_call>，就停下模型，读出工具名和参数，再调用对应的 Python 函数。"),
    "story.run.calculator": (
        "普通的 Python 算出精确的结果。大模型把数字也看成词元（还记得 235 + 0 吗？），算术会出错——"
        "计算器工具可靠得多。"),
    "story.run.run_terminal": (
        "程序真的会在你的电脑上运行这条命令，就是现在。只允许白名单里的只读命令——"
        "给 AI 一个终端，权力很大，所以真实的系统会先问用户。"),
    "story.run.plot_chart": (
        "画图的是 matplotlib（一个普通的 Python 库）。语言模型一个像素都没画——"
        "它只选了图表类型、标题和数据。"),
    "story.run.generate_image": (
        "这是另一个 AI（图像扩散模型）：从纯噪声开始，一步步去掉噪声，由提示词来引导（这里是模拟的）。"
        "注意：大模型把你的请求改写成了更详细的提示词！"),
    "story.run.denoise": "去噪第 {k}/{steps} 步",
    "story.chip.tool_result": "工具结果",
    "story.chip.tool_call": "工具调用（{n} 个词元）",
    "story.chip.more": "还有 {n} 个",
    "story.return.caption": (
        "程序把工具的结果变成文字，作为一条新消息接到上下文末尾（绿色小块）。"
        "模型拿回来的只是一小段文字——数字、文件名、文件路径——而不是图片本身。"),
    "story.answer.label": "回答：{text}",
    "story.answer.head": ("总结：你的话 -> 词元 -> 向量（+ 位置）-> 2 x [注意力、残差 + 归一化、"
                          "前馈网络、残差 + 归一化] -> 下一个词元的概率\n"),
    "story.answer.loop": "  -> <tool_call> -> 程序运行真实的代码 -> 结果回到上下文{twice}\n",
    "story.answer.twice": "（这里重复了两次）",
    "story.answer.tail": "  -> 最终回答。   大模型本身只做一件事：预测下一个词元。",
    # ---------------------------------------------------------------- tf_steps.py: the map
    "tf.map.P": "P  (位置)",
    "tf.map.S": "分数",
    "tf.map.A": "权重 A",
    "tf.map.O1": "头 1 输出",
    "tf.map.S2": "分数 (头 2)",
    "tf.map.A2": "权重 (头 2)",
    "tf.map.O2": "头 2 输出",
    "tf.map.C": "拼接",
    "tf.map.D": "ΔX  (变化量)",
    "tf.map.H": "隐藏层",
    "tf.map.X2": "X2  (第 1 层输出)",
    "tf.map.X3": "X3  (最终)",
    "tf.map.own": "× 各自的\nW_Q, W_K, W_V",
    "tf.map.head2": "头 2",
    "tf.map.head1": "头 1",
    "tf.map.table": "嵌入表",
    "tf.map.norm": "归一化",
    "tf.map.layer2": "第 2 层",
    "tf.map.layer2_note": ("同样的步骤再做一遍\n(注意力、残差 + 归一化、\n前馈网络、残差 + 归一化)\n"
                           "用这一层自己的权重"),
    "tf.map.layer1": "第 1 层",
    "tf.map.attention": "注意力",
    "tf.map.ffn": "前馈网络",
    "tf.map.output": "输出",
    "tf.map.res1": "残差：X 跳过注意力，再加回来",
    "tf.map.res2": "残差：X1 跳过前馈网络",
    # ---------------------------------------------------------------- tf_steps.py: tracker
    "tf.track.embed": "词元嵌入",
    "tf.track.pos": "加上位置",
    "tf.track.add1": "第 1 层：残差+归一化",
    "tf.track.l1end": "第 1 层结束",
    "tf.track.l2end": "第 2 层结束",
    "tf.track.pred": "用来做预测",
    # ---------------------------------------------------------------- tf_steps.py: steps
    "tf.embed.ids": "词元   ID",
    "tf.embed.caption": (
        "地图：Transformer 做的所有事，从左到右一目了然。每个词元是一行数字，一路上都守着自己的那一行；"
        "最后一行（黄色）是 <ai>，用它这一行来预测下一个词。  "
        "第 1 步，嵌入：每个词元的编号，会从一张学出来的表（{V} x {d}）里取出自己那一行，共 {d} 个数。"
        "把这些行叠起来，就是矩阵 E（{n} x {d}）。红色 = 正数，蓝色 = 负数。"),
    "tf.pos.caption": (
        "第 2 步，位置：到现在为止，模型还不知道这些行的先后顺序。每个位置都有一组固定的数字"
        "（位置编码 P，由正弦波生成），逐个数直接加上去：X = E + P。"
        "X（{n} x {d}）接着进入注意力模块。"),
    "tf.qkv.demo1": "<ai> 这一行 X  ·  W_Q 的第一列  =  <ai> 这一行 Q 的第一个数",
    "tf.qkv.demo2": "{terms} + ...   (共 {d} 个乘积相加)  =  {q0:.2f}",
    "tf.qkv.caption": (
        "第 3 步，Q K V：用学出来的矩阵做三次普通的矩阵乘法：X · W_Q = Q，X · W_K = K，X · W_V = V"
        "（每个都是 {n} x {dh}）。上方手算了其中一个数：<ai> 这一行 X 乘以 W_Q 的第一列。"
        "可以这样理解：Q（查询）= 每个词元在找什么，K（键）= 它能提供什么，V（值）= 它要传出去什么。"),
    "tf.scores.info": "Q · Kᵀ = 分数\n({n} × {dh}) · ({dh} × {n}) = ({n} × {n})",
    "tf.scores.info2": ("格子 (i, j) = 词元 i 的问题 (Q)\n和词元 j (K) 匹配得有多好。\n"
                        "然后全部 ÷ √{dh} = {root:.0f}"),
    "tf.scores.mask": "遮挡：不许看后面的\n词元 -> 灰色 (−∞)",
    "tf.scores.caption": (
        "第 4 步，分数：Q · Kᵀ 还是一次矩阵乘法。K 的行转成列（Kᵀ，在上方），"
        "所以格子 (i, j) = Q 的第 i 行 · Kᵀ 的第 j 列 = 第 i 个词元的“问题”和第 j 个词元匹配得有多好。"
        "黄色那一行是 <ai> 和每个词元比较的结果。然后是遮挡（掩码）：任何词元都不许看排在它后面的词元（灰色）。"
        "<ai> 排在最后，所以它什么都看得到。"),
    "tf.softmax.info": "softmax，逐行计算：\n所有权重 ≥ 0，\n每一行加起来等于 1\n\n越亮 = 权重越大",
    "tf.softmax.caption": (
        "第 5 步，权重：softmax 把每一行分数变成权重（先算 e^分数，再除以这一行的总和）："
        "全是正数，每行加起来等于 1，越亮越大。黄色这一行表示 <ai> 对每个词元的注意力有多少——最多给了 {top}。"),
    "tf.weighted.info": "A · V = 头的输出\n({n} × {n}) · ({n} × {dh}) = ({n} × {dh})",
    "tf.weighted.caption": (
        "第 6 步，混合：A · V 又是一次矩阵乘法，它做的事是把 V 的各行混合起来："
        "<ai> 的新一行 =（它对词元 1 的权重）× V 第 1 行 +（权重 2）× V 第 2 行 + ... "
        "权重大的保持明亮，权重小的变暗，最后全部加起来。所以 <ai> 现在带上了 '{a}' 和 '{b}' 的信息。"
        "每一行都用自己的权重做同样的事。"),
    "tf.head2.caption": (
        "第 7 步，第 2 个头：注意力并排做了两次，第二次用另一组 W_Q、W_K、W_V——步骤和第 3 到 6 步完全一样。"
        "每个头可以去找不同类型的关系。<ai> —— 头 1：{h1}；头 2：{h2}。"),
    "tf.concat.heads": "头 1 | 头 2",
    "tf.concat.info": "拼接 · W_O = ΔX    ({n} × {d}) · ({d} × {d}) = ({n} × {d})",
    "tf.concat.caption": (
        "第 8 步：把两个头的输出并排拼起来（每行 16 + 16 = {d} 个数），再乘以一个学出来的矩阵 W_O。"
        "结果 ΔX 就是注意力想对每个词元那一行做的修改量。"),
    "tf.add1.info": "X + ΔX，然后对每一行做归一化：\n减去均值，÷ 离散程度，\n× 增益 + 偏置（学出来的）",
    "tf.add1.caption": (
        "第 9 步，残差 + 归一化：X 自己跳过整个注意力模块（红线，这就是“残差”），加到 ΔX 上：X + ΔX。"
        "这样注意力只是添加一个修正量，原来的信息一点也不会丢。然后每一行做归一化"
        "（均值变成 0，离散程度变成 1），让数字保持在稳定的范围内。结果：X1（{n} x {d}）。"),
    "tf.ffn.relu": "ReLU：每个负数 -> 0\n(<ai>：{ff} 个里有 {off} 个被关掉)",
    "tf.ffn.info2": "隐藏层 · W2\n({n} × {ff}) · ({ff} × {d}) = ({n} × {d})",
    "tf.ffn.caption": (
        "第 10 步，前馈网络：再做两次矩阵乘法：X1 · W1 让每个词元变成 {ff} 个数，"
        "ReLU 把每个负数变成 0（变暗），再 · W2 变回 {d} 个数。每一行都是单独处理的——行与行之间不混合。"
        "注意力 = 词元之间互相交流；前馈网络 = 每个词元自己思考。"),
    "tf.add2.caption": (
        "第 11 步，再做一次残差 + 归一化：X1 跳过前馈网络，加到 F 上，然后每一行做归一化。"
        "X2 是第 1 层的输出——形状和 X 一样（{n} x {d}），但现在每一行都已经知道了一些其他词元的信息。"),
    "tf.layer2.caption": (
        "第 12 步，第 2 层：把第 3 到 11 步再做一遍，用的是这一层自己学出来的矩阵。"
        "在第 2 层里，<ai> 最关注的是：{top}。真实的模型会叠 30 到 100 层。"),
    "tf.out.probs": "softmax -> 概率",
    "tf.out.info": ("只用 <ai> 这一行：  (1 × {d}) · ({d} × {V}) = (1 × {V})\n"
                    "模型认识的每个词元各得一个分数"),
    "tf.out.caption": (
        "第 13 步，输出：现在只用 <ai> 这一行——就是我们一路跟着走的黄色那一行。"
        "最后再和 W_out 做一次矩阵乘法，模型认识的 {V} 个词元每个得到一个分数；"
        "softmax 把分数变成概率。得票最高的是：'{w}'（{p:.1%}）。"),
    "tf.out.extra": "（几乎是 100%：这个迷你模型把这几段对话背下来了——大模型要不确定得多。）",
    "tf.quick.caption": (
        "Transformer 内部（深入讲解已关闭——按 D 可以看到每一步）：注意力让各个词元的行互相交换信息，"
        "前馈网络单独处理每一行，这样重复两次（2 层）。请跟着黄色的 <ai> 那一行看。"),
    # ---------------------------------------------------------------- cache_story.py
    "cache.intro": (
        '第 7 题：缓存到底省了什么？我们跟着第 3 题那张绿色的 K 矩阵走（问题："{prompt}"），'
        '全部是迷你模型的真实数字。一步一步看：什么叫"算"，什么叫"重新算"，缓存怎么省，最后换算成钱和电。'),
    "cache.x.note": "{n} 个词元，每个是一行 {d} 个数（第 3 题见过的 X）",
    "cache.x.caption": (
        "先回忆一下：问题里的 {n} 个词元，每个变成一行 {d} 个数，叠起来就是矩阵 X。"
        "我们要弄清楚：模型\"算\"的时候到底在算什么，\"重新算\"又是什么意思。"),
    "cache.k.demo": "K 的一个数 = {terms} + ...（共 {d} 次乘法）= {v:.2f}",
    "cache.k.count": "已算 {rows} 行 × {per} 次 = {n} 次乘法",
    "cache.k.count2": "K 和 V 一共 {n} 次乘法（这还只是 1 个头）",
    "cache.k.caption": (
        "这就是\"算\"：X 的一行（{d} 个数）乘 W_K 的一列（{d} 个数），乘 {d} 次再加起来，得到 K 的 1 个数。"
        "K 的一行有 {dh} 个数 = {per} 次乘法。看 {n} 行一行一行出来。V 用 W_V 同样算。"
        "真实模型的每一层里，全是这种乘法。"),
    "cache.first.out": "模型预测的第一个新词元：'{tok}'（{p:.1f}%）",
    "cache.first.note": "接上新词元后 X 有 {n} 行。上面那些行要不要再算一遍？",
    "cache.first.caption": (
        "模型预测出了第一个新词元 '{tok}'。要写第 2 个词元，就把 '{tok}' 接到输入后面："
        "X 现在有 {n} 行（新的一行在最下面，黄框）。问题来了：上面那些行，要不要再算一遍？"),
    "cache.redo.label": "K（第 2 次）",
    "cache.redo.arrow": "不用缓存：全部再算一遍",
    "cache.redo.caption": (
        "不用缓存的做法：{n} 行全部重新送进模型，从头再乘。右边橙色的 K 就是第二次算出来的。"
        "看计数器：光 K 这一项就是 {n} × 512 次乘法；Q、V、两个头、前馈网络、第 2 层也都要再算。"),
    "cache.cmp.same": "= 差 {d:.3f}",
    "cache.cmp.new": "新的一行",
    "cache.cmp.waste": ("这 {n0} 行：白算了\n"
                        "光 K 这一项就白乘了 {k} 次\n"
                        "整个模型（2 层、所有矩阵）：\n"
                        "   全部重算：{all} 次\n"
                        "   只算新的一行：{new} 次"),
    "cache.cmp.caption": (
        "逐行对比：前 {n0} 行和第一次算的完全一样，差 0.000（同样的输入行，同样的 W_K）。只有最后一行是新的。"
        "所谓\"重新算\"，就是把这 {n0} 行又乘了一遍，结果一个数都没变——纯属浪费。"),
    "cache.mask.label": "注意力权重（第 1 层 第 1 个头）",
    "cache.mask.legend": "黑格 = 被遮住，不准看；越亮 = 看得越多",
    "cache.mod.diff": "和模型 A 的差",
    "cache.mask.rows": "谁在看",
    "cache.mask.cols": "被看的词元 ->",
    "cache.mask.col_note": ("新词元这一列（黄框）：\n"
                            "上面 {n0} 行全是黑的 = 被遮住\n"
                            "旧词元从来不看新词元"),
    "cache.mask.row_note": "只有新的这一行\n会看全部词元",
    "cache.mask.caption": (
        "为什么一定一样？看注意力表：每一行只能看自己和上面的行（右上角黑的部分被遮住了）。"
        "新词元是最后一行、最后一列。这一列里旧的行全是黑的——旧词元从来不读新词元，所以它们的结果不可能变。"),
    "cache.l2.old": "第 2 层 K（第 1 次）",
    "cache.l2.new": "第 2 层 K（第 2 次）",
    "cache.l2.diff": "和上次的差",
    "cache.l2.why": ("第 1 层里每一行只看了自己和上面的行，\n"
                     "所以旧行进入第 2 层时没变 -> 第 2 层的 K、V 也一样。\n"
                     "前馈网络是一行一行单独算的，也不变。"),
    "cache.l2.caption": (
        "第 2 层也一样：旧的行从第 1 层出来没变（它们没看到新词元），进入第 2 层也没变，"
        "第 2 层的 K 和 V 又是一模一样。前馈网络每一行单独算。所以每一层里，旧词元的 K、V 都可以留着。"),
    "cache.store.shelf": "KV 缓存（笔记本）- 存在 GPU 显存里",
    "cache.store.l1h2": "第1层 头2",
    "cache.store.l2h1": "第2层 头1",
    "cache.store.l2h2": "第2层 头2",
    "cache.store.mem": "每个词元存 {per} 个数（K 和 V × 2 层 × 2 个头 × 16）；{n} 个词元共 {tot} 个",
    "cache.store.caption": (
        "既然不会变，就存起来：每个词元在每一层、每个头的 K 和 V 都放到架子上——这就是 KV 缓存，模型的笔记本。"
        "只存 K 和 V：以后新词元只需要旧词元的 K（拿来比较）和 V（拿来加权），别的都用不上。"),
    "cache.one.xrow": "只有新词元 '{tok}' 这一行",
    "cache.one.q": "q = x · W_Q（新词元的查询）",
    "cache.one.scores": "只有 1 行分数：q 和架子上 {n} 个 K 比较 -> softmax",
    "cache.one.result": ("下一个词元：'{tok}'\n"
                         "用缓存 {pc:.4f}%，不用缓存 {pf:.4f}%\n"
                         "最大差值 {diff:.0e}（电脑的舍入误差）——结果一样"),
    "cache.one.count": ("乘法次数（整个模型）：\n"
                        "   不用缓存，14 行全部重算：{all}\n"
                        "   用缓存，只算 1 行：{new}"),
    "cache.one.caption": (
        "用缓存写第 2 个词元：只算新的这一行。x · W_K、x · W_V 得到它的 K、V，放到架子最下面；"
        "x · W_Q 得到 q，和架子上全部 {n1} 个 K 比较——只出 1 行分数，不是整张表。"
        "预测结果完全一样，工作量只要大约十分之一。"),
    "cache.bars.title": "写每个新词元要多少次乘法（整个迷你模型）",
    "cache.bars.tok": "第 {k} 个\n({n} 行)",
    "cache.bars.legend": "灰色 = 不用缓存（全部重算）\n黄色 = 用 KV 缓存（只算一行）",
    "cache.bars.total": ("整个工具调用，{k} 个词元：\n"
                         "   不用缓存 {no} 次乘法\n"
                         "   用缓存 {ca} 次\n"
                         "   少了 {x:.0f} 倍"),
    "cache.bars.caption": (
        "一个词元一个词元地比：不用缓存（灰色），每写一个新词元都要把全部行重算，越往后越多；"
        "用缓存（黄色），每次只算一行，几乎不变。整个工具调用算下来，差距非常大。"),
    "cache.m1.title": "一段对话已经有 {hist} 个词元，你再发 {new} 个",
    "cache.m1.no": "不用缓存",
    "cache.m1.hit": "命中缓存",
    "cache.m1.r_tok": "要计算的词元",
    "cache.m1.r_usd": "输入费用（美元）",
    "cache.m1.r_yen": "约合日元",
    "cache.m1.r_time": "一块 GPU 的计算时间",
    "cache.m1.r_energy": "耗电（估算）",
    "cache.m1.bar_no": "{n} 个词元全部重算",
    "cache.m1.bar_hit": "只算 {n} 个",
    "cache.m1.how": (
        "怎么算的：费用 = 词元数 × Claude Sonnet 5.5 价目表（2026 年 10 月，每百万词元：\n"
        "   输入 $2，写缓存 $2.5，读缓存 $0.2）；汇率按 1 美元 = 150 日元。\n"
        "耗电是估算：假设一个 700 亿参数的模型（Claude 的大小没有公开），\n"
        "   读一个词元约 700 亿次乘法；一块 H100 GPU 约 700 W，每秒约 200 万亿次乘加\n"
        "   -> 每个词元约 {J:.2f} 焦耳、{ms:.2f} 毫秒（不含散热）。\n"
        "不用缓存时，这一句话的耗电约等于手机电池的 {phone:.0f}%。\n"
        "读缓存也要搬数据，不完全是零，但小得多。"),
    "cache.m1.caption": (
        "放大到真实世界：一段已经有 2 万个词元的对话（系统提示、工具说明、聊天记录），你再发一句话。"
        "不用缓存要把 20,050 个词元全部重算；命中缓存只算新的 50 个。输入费用便宜约 {x:.0f} 倍。"),
    "cache.m2.title": "一整段对话：{turns} 轮。系统提示 + 工具 {sys} 个词元，每轮新增 {per} 个",
    "cache.m2.axis": "第几轮（每轮的输入费用：灰 = 不用缓存，黄 = 用缓存）",
    "cache.m2.sum": ("整段对话的输入费用：\n"
                     "   不用缓存 {no}（约 {noy} 日元）\n"
                     "   用缓存 {hit}（约 {hity} 日元），便宜 {x:.1f} 倍\n"
                     "要计算的词元：{tn} 对 {th}\n"
                     "耗电（估算）：{wn} 对 {wh}\n"
                     "（第 1 轮写缓存要多付 25%，从第 2 轮起就赚回来）\n"
                     "（输出部分两边一样，没算在里面）"),
    "cache.m2.class": ("一个 {c} 人的班，每人这样聊一次：\n"
                       "   不用缓存 {no}（约 {noy} 日元），{kn:.2f} kWh\n"
                       "   用缓存 {hit}（约 {hity} 日元），{kh:.2f} kWh\n"
                       "   {kn:.2f} kWh 的电费约 {en:.0f} 日元（按 31 日元/kWh）"),
    "cache.m2.caption": (
        "一整段对话，30 轮：前文每一轮都在变长。不用缓存，每一轮都要把越来越长的前文重读一遍（灰柱越来越高）；"
        "用缓存，每轮只有新增的部分付全价，旧的按一成价读取。所以长对话一定要用缓存。"),
    "cache.req.r1": "请求 1",
    "cache.req.r2": "请求 2",
    "cache.req.call": "工具调用 {n} 个",
    "cache.req.result": "工具结果 {n} 个词元",
    "cache.req.hit": "缓存命中：{n} 个词元直接用架子上的 K、V",
    "cache.req.miss": "要算：{n} 个",
    "cache.req.why": ("从头开始逐个词元比较；一模一样就用存好的 K、V。\n"
                      "所以 Claude 这类 API 读缓存只收一成价：几乎不用算。"),
    "cache.req.caption": (
        "下一次请求（提示缓存）：工具运行完，程序接上结果再问一次。开头 {n} 个词元和上次一字不差，"
        "服务器上还留着它们的 K、V（默认约 5 分钟），直接拿来用；只有新增的 {m} 个词元要算。"),
    "cache.chg.edit": "改了一个词：tank -> wafer",
    "cache.chg.layer": "第 {l} 层 K",
    "cache.chg.n1": "第 1 层：只有被改的那一行变了",
    "cache.chg.n2": "第 2 层：从改动处往下全变了\n（下面的行在注意力里看到了它）",
    "cache.chg.rule": ("缓存只能用到第一个不同的词元为止。\n"
                       "所以：不变的内容（系统提示、工具）放最前面，会变的（时间、新问题）放最后面。"),
    "cache.chg.caption": (
        "把前面的一个词改掉（tank -> wafer）：第 1 层只有那一行变了；但到了第 2 层，从它往下每一行都变了——"
        "下面的行在注意力里看到了它。所以缓存只能用到第一个不同的词元为止，后面全部重算。"),
    "cache.mod.a": "模型 A 的 K",
    "cache.mod.b": "模型 B 的 K\n（另一组权重）",
    "cache.mod.note": ("同样的词元，权重不同 -> K 的每一行都不同。\n"
                       "另一个模型存的 K、V 在这里完全用不上。\n"
                       "例：对话已有 {hist} 个词元时——\n"
                       "   继续用同一个模型，命中缓存：{hit}（约 {hity} 日元）\n"
                       "   中途换模型：{sw}（约 {swy} 日元），全部重新写进缓存"),
    "cache.mod.caption": (
        "换一个模型：同样的词元，但 W_K 不一样，算出来的 K 完全不同。上一个模型的笔记，新模型一个字都读不了，"
        "整段对话必须从头再算。这就是为什么长对话中途换模型会多花 token。"),
    "cache.sum.mem": "代价：缓存占 GPU 显存，所以服务器只留一小会儿\n（Claude API 默认 5 分钟）",
    "cache.sum.caption": (
        "总结：\"算\" = 每一行乘一遍权重。新词元不会改变旧行的 K、V，所以存起来（KV 缓存），每个新词元只算一行；"
        "下一次请求开头一样，也能直接用（提示缓存）。改了前面的词、换了模型、隔太久过期，就只能从头再算。"),
}

STRINGS = {"en": EN, "zh": ZH}
