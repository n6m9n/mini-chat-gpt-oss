# mini-chat-gpt-oss

A tiny (~250M-parameter) re-implementation of the **gpt-oss** architecture,
trained on **TinyStories**, in two variants so you can compare them:

| Folder | What it is | Params |
|---|---|---|
| [`mini-gpt-oss/`](./mini-gpt-oss) | Baseline: the gpt-oss architecture (RMSNorm, GQA + attention sinks + sliding window, RoPE, **MoE**, SwiGLU) | ~249M |
| [`mini-gpt-oss-engram/`](./mini-gpt-oss-engram) | Baseline **+ Engram** conditional-memory lookup (hashed N-gram tables + gating + causal conv), from arXiv:2601.07372 | ~267M |

Both share the exact same backbone, tokenizer (o200k_harmony), and training
recipe — the **only** difference is the Engram module — so training both gives a
clean **MoE vs MoE + Engram** comparison.

## Quick start
Each folder is self-contained with its own README:
- **[mini-gpt-oss/README.md](./mini-gpt-oss/README.md)** — baseline setup, training, generation.
- **[mini-gpt-oss-engram/README.md](./mini-gpt-oss-engram/README.md)** — Engram module details + how to run the comparison.

```bash
cd mini-gpt-oss            # or  cd mini-gpt-oss-engram
pip install -r requirements.txt
python model.py            # sanity-check the model + parameter count
python data.py --split train
python train.py --device cuda --resume
python generate.py --prompt "Once upon a time"
```

Training checkpoints (`out/`), tokenized data (`data/`), and diagram files are
git-ignored; both trainers support `--resume` and `DATA_DIR` / `OUT_DIR` env
overrides for multi-GPU / Kaggle runs (see the per-folder READMEs).
