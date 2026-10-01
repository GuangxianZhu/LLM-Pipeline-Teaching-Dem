# LLM Pipeline Demo (LLM 可视化教学演示) — 3Blue1Brown-style edition

Claude Opus 写的 — a pure-Python 3D teaching demo that shows how **what you say** becomes
**tokens → vectors → attention → next-token prediction → a tool call → real code → a result → the answer**.

## Install & run

```
pip install -r requirements.txt
python main.py
```

Works offline. Each student runs it on their own computer (Windows / macOS / Linux).
Fonts: Cambria / Times New Roman on Windows (automatic fallback on other systems).

## Controls

| Key / mouse | Action |
|---|---|
| Space or → | Next step (press during an animation = skip to its end) |
| A | Auto play on/off |
| + / - | Speed 0.5x – 3x |
| 1 – 7 | Pick a question |
| R | Restart current question |
| D | Deep dive on/off (inside attention: Q/K table, weights, ΔE, MLP) |
| C | Camera follow on/off (off = explore freely) |
| G | Glow on/off (turn off on slow computers) |
| H | Hide/show the side panels (for projecting) |
| Right-drag / wheel | Rotate / zoom |

## The story (one big black board, the camera flies between parts)

1. **Context** – hidden system prompt + tool list + your message
2. **Tokens** – text cut into tokens with IDs
3. **Vectors** – each token becomes a column of numbers
4. **Meaning space** – vectors as 3D arrows; related words point the same way
5. **Big picture** – vectors flow through the Attention block and the Multilayer Perceptron, × 32 layers
6. *Deep dive (D)* – Q/K table → K·Q scores → softmax weights → weighted sum of V = ΔE →
   the vague token's arrow is pushed next to the words it belongs with → MLP signal
7. **Predict** – context → Transformer → next-token probabilities → token appended, run again
8. **Tool** – the program (harness) reads `<tool_call>` and calls a real Python function
9. **Return** – the result goes back into the context; the model runs again
10. **Answer**

## The 6 questions

| # | Question | Attention example | Tool |
|---|---|---|---|
| 1 | Hi! Who are you? | "you" | none (sampling randomness) |
| 2 | What is 17% of 2350? | "0" belongs to "2350" | calculator |
| 3 | Plot the tank temperature for the last hour. | "ature" → tank temperature | matplotlib line chart |
| 4 | Draw a cat wearing a cleanroom suit. | "room" → cleanroom suit, not a room | image model (denoising) |
| 5 | How many files are in the demo folder? | "folder" → the demo folder | real terminal command |
| 6 | List the files here and make a bar chart by file type. | "chart" → bar chart | terminal, then chart (2 calls) |
| 7 | Why is it fast? (cache) | — | — (KV cache + prompt-cache hit, see below) |

## Question 7: caching

1. **No cache** – every new token pushes the whole context through the Transformer again (work 10, 11, 12, 13…)
2. **Why reuse** – the attention table grows by only one column; old tokens never look forward, so their K and V never change
3. **KV cache** – prefill once, then one pass per token; old K/V are read from the cache shelf (total work 46 → 13)
4. **Memory** – K and V × 32 layers × 4096 numbers ≈ 0.5 MB per token (7B-size model, 16-bit) → long chats cost GPU memory
5. **Request 2** – after a tool call the next request starts with the same prefix → cache HIT for the prefix, only the new part is computed
6. **Prefix rule** – change one early token and everything after it misses; keep stable content first, changing content last

## What is real and what is simulated

* **Simulated (no real LLM inside):** tokens, IDs, vectors, attention weights and probabilities are
  deterministic toy values, so every student sees the same thing. The ΔE sum is computed from those values.
* **Real:** the tool-call format, the parsing, and all tool outputs. The terminal really runs one
  read-only whitelisted command (`dir /b /a-d` on Windows, `ls -p` elsewhere) inside `demo_files/`.
  Charts are drawn by matplotlib. The "AI image" is drawn by code; only the denoising is simulated.
* Generated pictures are saved in `outputs/`.

## Files

| File | Purpose |
|---|---|
| `main.py` | window, panels, camera, step player |
| `story.py` | turns a question into animated steps + captions |
| `cache_story.py` | question 7: KV cache and prompt-cache hits |
| `kit.py` | 3B1B-style drawing kit (arrows, number columns, token boxes, spheres) |
| `sim.py` | toy tokenizer, embeddings, attention, probabilities |
| `tools.py` | the four real tools |
| `scenarios.py` | the six questions — add your own here (prompt, word groups, focus word, tool calls) |
| `demo_files/` | sample files the terminal tool lists (change them freely) |
| `style_sample.py` | the standalone style sample (one sentence, attention only) |
| `legacy_factory/` | the first version (factory / conveyor-belt look), kept for reference |
