# LLM Pipeline Demo (LLM 可视化教学演示)

Claude Opus 写的 — a pure-Python 3D teaching demo that shows **what happens inside a Transformer**, and how
what you type becomes **tokens → vectors → attention → feed-forward → next-token prediction → a tool call →
real code → a result → the answer**.

Everything you see inside the Transformer is computed by a **real, trained tiny GPT** (2 layers, 2 heads,
32 numbers per token). The numbers on screen are its actual numbers.

## Install & run

```
pip install -r requirements.txt
python main.py
```

Only `panda3d`, `numpy` and `matplotlib` are needed. Works offline. Each student runs it on their own
computer (Windows / macOS / Linux).

## Windows version for students (no Python needed)

Give students `dist/LLMPipelineDemo-win64.zip`: unzip, double-click `LLMPipelineDemo.exe`
(see `HOW TO RUN.txt` inside). To rebuild it after changing the code (works on Windows, macOS or Linux):

```
pip install panda3d
python build_windows.py          # -> build/win_amd64/  and  dist/LLMPipelineDemo-win64.zip
```

`LLMPipelineDemo.exe --selftest` plays every question quickly and reports PASSED / FAILED.
Errors of the .exe go to `%LOCALAPPDATA%\LLMPipelineDemo\output.log`.

## Controls

| Key / mouse | Action |
|---|---|
| Space or → | Next step (press during an animation = skip to its end) |
| A | Auto play on/off |
| + / - | Speed 0.5x – 3x |
| 1 – 7 | Pick a question |
| R | Restart current question |
| D | Deep dive on/off (off = one quick pass through the Transformer) |
| C | Camera follow on/off (off = explore freely) |
| G | Glow on/off (turn off on slow computers) |
| H | Hide/show the side panels (for projecting) |
| Right-drag / wheel | Rotate / zoom |

## How to follow the data

* **The highway.** Every token is one column of 32 numbers (red = positive, blue = negative). This row of
  columns — the *residual stream* — travels to the right along the top of the board. At each station a
  computation unfolds below it, and its result rises back up into the stream.
* **The yellow column** is the last token, `<ai>`. Its column is the one that finally predicts the next token,
  so it is framed in yellow at every station.
* **The tracker** (bottom right) always shows `<ai>`'s current 32 numbers, where it is now, and the path so far.
* **The architecture map** (left) lights up the block being shown (classic GPT / decoder-only diagram).
* All highway vectors share one colour scale, so the same number always has the same colour.

## The story

1. **Context** – hidden system prompt + your message, as one sequence
2. **Tokens** – text cut into tokens with IDs
3. **Embedding** – each ID picks its row in the embedding table (181 × 32) → the highway starts
4. **Positional encoding** – a fixed wave pattern is added to every column
5. **Attention, head 1** – columns drop into W_Q, W_K, W_V → Q, K, V (13 × 16)
6. **Scores** – Q·K table → ÷√16 → mask (no looking ahead) → softmax weights
7. **Weighted sum** – V columns × `<ai>`'s weights, added up → head output
8. **Head 2** – the same steps with its own matrices
9. **Concat × W_O** → ΔE, the change every token wants to make
10. **Add** – ΔE rises into the highway (plus a 3D view where every arrow moves at once)
11. **Norm** – mean 0, spread 1, then learned gain and bias (shown as bars)
12. **Feed forward** – 32 → 128 neurons → ReLU → 32, the same network for every token
13. **Add & Norm** – layer 1 done
14. **Layer 2** – the same again with its own matrices
15. **Output** – only `<ai>`'s column × W_out → 181 scores → softmax → next token
16. **Loop / tool / answer** – the token is appended and the model runs again; when it writes a
    `<tool_call>`, the program runs the real tool, appends the result, and the model continues

## The 7 questions

| # | Question | Tool |
|---|---|---|
| 1 | Hi! Who are you? | none (the first token is sampled, so "Hi" / "Hello" can change) |
| 2 | What is 17% of 2350? | calculator |
| 3 | Plot the tank temperature for the last hour. | matplotlib line chart |
| 4 | Draw a cat wearing a cleanroom suit. | image model (denoising) |
| 5 | How many files are in the demo folder? | real terminal command |
| 6 | List the files here and make a bar chart by file type. | terminal, then chart (2 calls) |
| 7 | Why is it fast? (cache) | — (KV cache + prompt-cache hit, see below) |

## Question 7: caching

1. **No cache** – every new token pushes the whole context through the Transformer again (work 10, 11, 12, 13…)
2. **Why reuse** – the attention table grows by only one column; old tokens never look forward, so their K and V never change
3. **KV cache** – prefill once, then one pass per token; old K/V are read from the cache shelf
4. **Memory** – K and V × layers × numbers per token → long chats cost GPU memory
5. **Request 2** – after a tool call the next request starts with the same prefix → cache HIT for the prefix, only the new part is computed
6. **Prefix rule** – change one early token and everything after it misses; keep stable content first, changing content last

## What is real

* **The model is real.** `tiny/weights.npz` is a tiny GPT trained on these conversations
  (`<sys> <user> question <ai> <tool_call>{json}</tool_call> <tool> result <ai> answer <end>`).
  The demo runs it with numpy and shows its real intermediate matrices. It writes the tool calls and
  answers itself, token by token.
* It is *tiny*: it has learned these few conversations by heart (hence ~100% probabilities), it does
  not understand new questions. Big models are the same machine with far more numbers and layers.
* **The tools are real.** The program parses the JSON, then runs the tool. The terminal really runs one
  read-only whitelisted command (`dir /b /a-d` on Windows, `ls -p` elsewhere) inside `demo_files/`.
  Charts are drawn by matplotlib. The "AI image" is drawn by code; only the denoising is simulated.
  Generated pictures are saved in `outputs/`.
* If you change the files in `demo_files/`, the tool output changes, but the tiny model's answer was
  learned for the original files and may then be wrong.
* Retraining (teacher only, needs `pip install torch`): `python tiny/train.py`.

## Files

| File | Purpose |
|---|---|
| `main.py` | window, panels (questions, architecture map, conversation, tracker), camera, step player |
| `story.py` | turns a question into animated steps + captions (context, tokens, prediction loop, tools) |
| `tf_steps.py` | inside the Transformer: the highway and every station, with the real numbers |
| `archmap.py` | the GPT architecture map on the left |
| `engine.py` | runs the whole conversation with the real model and the real tools |
| `tiny/` | the tiny GPT: `model.py` (numpy inference + trace), `data.py`, `train.py`, weights, vocab |
| `cache_story.py` | question 7: KV cache and prompt-cache hits |
| `kit.py` | 3B1B-style drawing kit (heatmaps, arrows, token boxes, text) |
| `sim.py` | tokenizer helpers |
| `tools.py` | the four real tools |
| `scenarios.py` | the seven questions |
| `setup.py`, `build_windows.py`, `build-requirements.txt` | packaging the Windows .exe |
| `demo_files/` | sample files the terminal tool lists |
| `style_sample.py` | the standalone style sample |
| `legacy_factory/` | the first version (factory / conveyor-belt look), kept for reference |
