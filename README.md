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

## Windows: one-click start (for students)

1. Install Python 3.9 or newer from https://www.python.org/downloads/ (tick **"Add python.exe to PATH"**).
2. Download this project (GitHub: **Code → Download ZIP**) and unzip it.
3. Double-click **`run_windows.bat`**.

The first start installs panda3d, numpy and matplotlib into a private `.venv` folder inside the project
(needs internet, 1–3 minutes). After that it starts right away and works offline. If anything goes wrong,
the black window shows the error and waits. To reinstall, delete the `.venv` folder.

(`setup.py` / `build_windows.py` are an unfinished attempt at a stand-alone .exe; not needed.)

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

The plan of the whole flow is `docs/flow.drawio` (open it in draw.io / diagrams.net; `docs/flow.png` is a preview).

* **One fixed strip, left to right.** The inside of the Transformer is laid out once, like the drawing:
  embedding -> + position -> Q, K, V -> scores -> weights -> mix -> head 2 -> concat · W_O -> add & norm ->
  feed forward -> add & norm -> layer 2 -> output. Nothing moves away; every result stays where it was made
  and the camera travels along. The strip is first shown as an empty map, then filled in step by step.
* **Every token is a ROW of numbers** (red = positive, blue = negative), in the same order everywhere.
  The last row is always `<ai>` (yellow frame): the row that finally predicts the next token.
* **Every "· W" is an ordinary matrix multiplication**, with the shapes written next to it. One number is
  worked out in detail (a row of X times a column of W_Q); Q · Kᵀ is drawn with Kᵀ on top of the table.
* **The red lines are the residual connections**: a copy of X travels over the attention block and is added back.
* **The tracker** (bottom right) shows `<ai>`'s current 32 numbers; **the architecture map** (left) lights up
  the block being shown.

## The story

1. **Context** – hidden system prompt + your message, as one sequence
2. **Tokens** – text cut into tokens with IDs
3. **Embedding** – the map of the whole strip, then each ID picks its row of the table (181 × 32) -> E
4. **Position** – X = E + P
5. **Q K V** – X · W_Q = Q, X · W_K = K, X · W_V = V (one number worked out)
6. **Scores** – Q · Kᵀ, ÷ √16, mask
7. **Weights** – softmax per row
8. **Mix** – A · V: <ai>'s row becomes a weighted sum of the V rows
9. **Head 2** – the same with its own matrices
10. **Concat · W_O** -> ΔX
11. **Add & norm** – X + ΔX (residual), normalized -> X1
12. **Feed forward** – X1 · W1, ReLU, · W2 -> F
13. **Add & norm** -> X2 (layer 1 output)
14. **Layer 2** – the same once more -> X3
15. **Output** – only <ai>'s row · W_out -> 181 scores -> softmax -> next token
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
| `tf_steps.py` | inside the Transformer: the fixed left-to-right strip, with the real numbers |
| `docs/` | `flow.drawio` / `flow.png`: the plan of the data flow (`make_flow.py` draws both) |
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
