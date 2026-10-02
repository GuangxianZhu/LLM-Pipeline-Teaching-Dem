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
    "stage.No cache": "No cache",
    "stage.Why reuse": "Why reuse",
    "stage.KV cache": "KV cache",
    "stage.Memory": "Memory",
    "stage.Request 2": "Request 2",
    "stage.Prefix rule": "Prefix rule",
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
        'Question 7 continues question 3 ("{prompt}"). Why can a model with billions of numbers write '
        'its answer so fast - and why is the second request after a tool call even faster? '
        'The answer is CACHING.'),
    "cache.context": "context",
    "cache.box_note": "computes K and V for every token it processes",
    "cache.chart_title": "work for each new token  (tokens pushed through the Transformer)",
    "cache.new_token": "new token {i}",
    "cache.no_cache": "without cache",
    "cache.with_cache": "with KV cache",
    "cache.processing": "processing {n} tokens ...",
    "cache.generated": "generated '{tok}'  -  that took {n} tokens of work",
    "cache.total": "total work: {total}",
    "cache.nocache.caption": (
        "Without any cache: to write each new token (yellow), the model pushes the WHOLE context through "
        "the Transformer again - all its layers, for every token. The work grows with every token "
        "(bars), and for a long answer that is a huge waste."),
    "cache.why.old": "all of this was computed\nin the previous step\n= same numbers again",
    "cache.why.new": "only this column is new:\nthe new token's Q\ncompares with all K",
    "cache.why.mask": "old tokens never look at\nnew ones (no arrows back)",
    "cache.why.caption": (
        "Why is recomputing a waste? Remember the attention table. When a new token arrives, only ONE new "
        "column appears: its query compares with all keys. Old tokens may only look backwards, so their "
        "K and V can never change. Same input, same numbers - so just keep them!"),
    "cache.shelf": "KV cache  (stored K and V)",
    "cache.prefill": "PREFILL: process the prompt once ({n} tokens)",
    "cache.only_new": "only the NEW token is processed; old K, V are read from the cache",
    "cache.same_answer": "same answer, a fraction of the work",
    "cache.cache.caption": (
        "With a KV CACHE: the prompt is processed once (PREFILL) and every token's K and V are stored on "
        "the shelf. After that each new token needs only ONE pass; old K and V are simply read from "
        "memory (DECODE). Compare the bars."),
    "cache.mem.1": "for EVERY token the cache keeps:",
    "cache.mem.2": "our tiny model:  K and V  x  2 layers  x  32 numbers  =  128",
    "cache.mem.3": "a 7B model:  K and V  x  32 layers  x  4096 numbers",
    "cache.mem.4": "= 262,144 numbers  =  about 0.5 MB",
    "cache.mem.5": "10,000 tokens  ->  about 5 GB of GPU memory",
    "cache.mem.caption": (
        "The price: memory. The cache must hold K and V for every token, in every layer - the K and V "
        "columns you saw inside attention (here for our tiny model and for a typical 7-billion-parameter "
        "model, 16-bit). That is why long conversations are expensive, why "
        "models have a context limit, and why newer models use tricks to shrink the cache."),
    "cache.seg.system": "system + tools",
    "cache.seg.user": "user message",
    "cache.seg.call": "tool call",
    "cache.seg.result": "tool result",
    "cache.req1": "request 1",
    "cache.server": "server cache",
    "cache.req2": "request 2",
    "cache.req2b": "request 2'",
    "cache.req.out1": "-> tool call -> tool runs ...",
    "cache.req.hit": "HIT: 190 tokens reused",
    "cache.req.miss": "MISS: 55 computed",
    "cache.req.ttft": "work before the first new token",
    "cache.req.nocache": "no cache",
    "cache.req.cachehit": "cache hit",
    "cache.req.n245": "245 tokens",
    "cache.req.n55": "55 tokens  -> faster, and cached tokens are usually billed much cheaper",
    "cache.req.caption": (
        "Across requests: after the tool runs, the program sends a SECOND request. Its beginning is exactly "
        "the same as request 1. The server kept the K and V of that prefix (prompt cache), so it compares "
        "token by token from the start: everything identical is a CACHE HIT; only the new part is computed."),
    "cache.prefix.changed": 'someone added "Time: 14:47" here',
    "cache.prefix.miss": "MISS from the change to the end: about 240 tokens computed again",
    "cache.prefix.rule": "rule: put things that never change FIRST, things that change at the END",
    "cache.prefix.caption": (
        "Why only the PREFIX? Every token's K and V depend on ALL tokens before it (attention mixed them "
        "in). Change one early token and every K and V after it is different - the cache is useless from "
        "there on. One small change at the start = a full miss."),
    "cache.summary.caption": (
        "Summary:  KV cache = inside one answer, store each token's K and V so every new token needs only "
        "one pass.  Prompt cache = between requests, reuse the stored K and V of an identical prefix "
        "(hit), compute only the new part (miss).  Cost: GPU memory."),
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
    "stage.No cache": "无缓存",
    "stage.Why reuse": "为何复用",
    "stage.KV cache": "KV 缓存",
    "stage.Memory": "内存",
    "stage.Request 2": "请求 2",
    "stage.Prefix rule": "前缀规则",
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
        '第 7 题接着第 3 题（"{prompt}"）。一个有几十亿个数的模型，为什么能写得这么快？'
        '为什么工具调用之后的第二次请求还要更快？答案是：缓存。'),
    "cache.context": "上下文",
    "cache.box_note": "为它处理的每个词元计算 K 和 V",
    "cache.chart_title": "生成每个新词元的工作量  (送进 Transformer 的词元数)",
    "cache.new_token": "新词元 {i}",
    "cache.no_cache": "不用缓存",
    "cache.with_cache": "使用 KV 缓存",
    "cache.processing": "正在处理 {n} 个词元 ...",
    "cache.generated": "生成了 '{tok}'  -  花了 {n} 个词元的工作量",
    "cache.total": "总工作量：{total}",
    "cache.nocache.caption": (
        "没有任何缓存：每写一个新词元（黄色），模型都要把整个上下文重新送进 Transformer——"
        "所有的层，每个词元都要算一遍。工作量随着词元增多越来越大（柱子），回答一长，浪费就很大。"),
    "cache.why.old": "这些在上一步\n已经算过了\n= 又是一样的数",
    "cache.why.new": "只有这一列是新的：\n新词元的 Q\n和所有的 K 比较",
    "cache.why.mask": "旧词元不会去看\n新词元（没有往回的箭头）",
    "cache.why.caption": (
        "为什么重新计算是浪费？回想一下注意力表。来了一个新词元，只会多出一列：它的查询和所有的键比较。"
        "旧词元只能往前看，所以它们的 K 和 V 永远不会变。输入一样，数就一样——那就存起来！"),
    "cache.shelf": "KV 缓存  (存下来的 K 和 V)",
    "cache.prefill": "预填充：把提示词处理一遍（{n} 个词元）",
    "cache.only_new": "只处理新词元；旧的 K、V 直接从缓存里读",
    "cache.same_answer": "答案一样，工作量少得多",
    "cache.cache.caption": (
        "用了 KV 缓存：提示词只处理一次（预填充），每个词元的 K 和 V 都存到架子上。"
        "之后每个新词元只需要算一次；旧的 K 和 V 直接从内存里读（解码）。比一比柱子。"),
    "cache.mem.1": "缓存为每个词元保存：",
    "cache.mem.2": "我们的迷你模型：  K 和 V  x  2 层  x  32 个数  =  128",
    "cache.mem.3": "70 亿参数的模型：  K 和 V  x  32 层  x  4096 个数",
    "cache.mem.4": "= 262,144 个数  =  约 0.5 MB",
    "cache.mem.5": "10,000 个词元  ->  约 5 GB 显存",
    "cache.mem.caption": (
        "代价是内存。缓存必须为每个词元、每一层都存下 K 和 V——就是你在注意力里看到的那些 K 和 V 列"
        "（这里分别是我们的迷你模型，和一个典型的 70 亿参数模型，16 位）。"
        "所以长对话很贵，所以模型有上下文长度限制，所以新模型会想办法把缓存压小。"),
    "cache.seg.system": "系统提示词 + 工具",
    "cache.seg.user": "用户消息",
    "cache.seg.call": "工具调用",
    "cache.seg.result": "工具结果",
    "cache.req1": "请求 1",
    "cache.server": "服务器缓存",
    "cache.req2": "请求 2",
    "cache.req2b": "请求 2'",
    "cache.req.out1": "-> 工具调用 -> 工具运行中 ...",
    "cache.req.hit": "提示缓存命中：复用了 190 个词元",
    "cache.req.miss": "未命中：重算了 55 个",
    "cache.req.ttft": "输出第一个新词元之前的工作量",
    "cache.req.nocache": "不用缓存",
    "cache.req.cachehit": "缓存命中",
    "cache.req.n245": "245 个词元",
    "cache.req.n55": "55 个词元  -> 更快，而且缓存的词元通常收费便宜得多",
    "cache.req.caption": (
        "跨请求：工具运行完后，程序会发出第二次请求。它的开头和请求 1 一模一样。"
        "服务器把这段前缀的 K 和 V 留着了（提示缓存），所以从头开始一个词元一个词元地比较："
        "完全相同的部分就是提示缓存命中；只需要计算新的部分。"),
    "cache.prefix.changed": '有人在这里加了 "Time: 14:47"',
    "cache.prefix.miss": "从改动处到结尾都未命中：约 240 个词元要重新计算",
    "cache.prefix.rule": "规则：不变的内容放在最前面，会变的内容放在最后",
    "cache.prefix.caption": (
        "为什么只能复用前缀？因为每个词元的 K 和 V 都取决于它前面的所有词元（注意力把它们混了进来）。"
        "改掉前面的一个词元，后面所有的 K 和 V 就都变了——从那里往后，缓存就没用了。"
        "开头改一点点 = 整个未命中。"),
    "cache.summary.caption": (
        "总结：KV 缓存 = 在一次回答内部，存下每个词元的 K 和 V，这样每个新词元只需要算一遍。"
        "提示缓存 = 在不同请求之间，复用相同前缀已存的 K 和 V（命中），只计算新的部分（未命中）。"
        "代价：显存。"),
}

STRINGS = {"en": EN, "zh": ZH}
