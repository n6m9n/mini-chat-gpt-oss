# mini-gpt-oss (~250M parameters)

A faithful, scaled-down re-implementation of the **gpt-oss** architecture
(OpenAI's open-weight 20B/120B models, as reproduced in
[VizuaraAILabs/nano-gpt-oss](https://github.com/VizuaraAILabs/nano-gpt-oss)).
It keeps every architectural component of gpt-oss but shrinks the dimensions to
land in the **200–300M parameter** range, then trains on **TinyStories** with the
real **o200k_harmony** BPE tokenizer.

---

## Architecture — same as gpt-oss

| Component | gpt-oss (20B) | this mini |
|---|---|---|
| Norm | RMSNorm | RMSNorm |
| Position | RoPE + YaRN | RoPE (+YaRN code, inactive) |
| Attention | Grouped-Query (64 Q / 8 KV) | GQA (12 Q / 4 KV) |
| Attention sinks | learned per-head sink logit | same |
| Sliding window | alternating layers (128) | alternating layers (128) |
| MLP | Mixture-of-Experts (32 experts, top-4) | MoE (8 experts, top-2) |
| Activation | clamped SwiGLU (α=1.702, limit=7) | identical |
| Tokenizer | o200k_harmony (vocab 201,088) | **identical** |

**Pre-norm residual block:**
`x = x + Attention(RMSNorm(x))` then `x = x + MoE(RMSNorm(x))`.
Attention uses GQA, RoPE, a causal mask (+ sliding window on even layers), and an
**attention sink** (a learned extra logit appended before softmax). The MoE
**router** picks the top-2 of 8 experts per token; each expert is a 2-layer FFN
with **clamped SwiGLU**, combined by softmax routing weights.

### Two deliberate, documented changes
1. **Weight tying** (embedding ↔ output head). The o200k vocab (201,088) is huge
   and almost entirely unused on TinyStories, so untied embeddings would waste
   ~309M params on two near-dead tables. Tying halves that cost and redirects the
   budget to real transformer capacity — much better performance per parameter.
2. **MoE load-balancing aux loss** (Switch-Transformer style, weight 0.01). The
   reference omits it; with top-k routing it keeps the router from collapsing
   onto a few experts. Set `aux_loss_coef=0` to disable.

(Also: fp32 + device-agnostic, and a `[B, T]` batch dimension so training is
efficient — the reference processes a single ungrouped token stream.)

---

## Configuration & parameter budget

```
hidden=768, layers=6, head_dim=64, heads=12, kv_heads=4,
experts=8, top-2, intermediate=768, vocab=201,088, tied embeddings

embedding (tied)        201,088 x 768                    = 154,435,584
per layer:
  attention   qkv 768x1280 + out 768x768 + norms + sinks =   1,575,692
  MoE         gate + 8 x (up 768x1536 + down 768x768)    =  14,181,128
              ------------------------------------------- = 15,756,820
  x 6 layers                                              =  94,540,920
final norm                                                =         768
------------------------------------------------------------------------
TOTAL                                                     ≈ 249,000,000  (~249M)
```

Only **~30M parameters are active per token** (top-2 of 8 experts), so it is
much cheaper to run than its size suggests. Run `python model.py` to print the
exact count. To move within 200–300M, the main levers are `num_hidden_layers`,
`num_experts`, `intermediate_size`, and `hidden_size`.

---

## How to run

```bash
# 0. setup (Python 3.10–3.13; torch needs a supported version)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. sanity check: model builds, ~249M params, one forward pass
python model.py
python tokenizer.py          # verify o200k_harmony loads (vocab 201,088)

# 2. prepare data (TinyStories -> tokenized uint32 bins)
python data.py --split validation         # quick start (smaller split)
#   python data.py --split train           # full corpus (recommended for quality)
#   python data.py --sample                # offline tiny corpus to smoke-test the pipeline

# 3. train  (a CUDA GPU is strongly recommended at this scale)
python train.py --device cuda --batch_size 16 --grad_accum 4 --max_iters 20000
#   Apple Silicon:  --device mps  (smaller batch);   CPU works but is very slow.

# 4. generate
python generate.py --prompt "Once upon a time" --temperature 0.8 --top_k 50
```

`train.py` saves the best checkpoint to `out/ckpt.pt` (weights + `ModelConfig`),
so `generate.py` is fully self-contained.

---

## Files
- `model.py` — config + full gpt-oss architecture (RMSNorm, RoPE/YaRN, GQA +
  sinks + sliding window, MoE + SwiGLU, load-balancing aux loss). Run directly
  to print the parameter count.
- `tokenizer.py` — the o200k_harmony BPE (tiktoken), exactly as in the repo.
- `data.py` — loads/tokenizes TinyStories into `uint32` token bins.
- `train.py` — AdamW, cosine LR, AMP, gradient accumulation, checkpointing.
- `generate.py` — autoregressive sampling (temperature + top-k), stops on EOT.

## Practical notes
- Token ids reach 201,087, so the data bins are **uint32** (uint16 cannot hold them).
- At ~250M params, expect to train on a GPU. On CPU/MPS use a small `--batch_size`
  and `--block_size` just to verify the pipeline end-to-end.
- To push toward 300M for more capacity, add layers or experts; to drop toward
  200M, reduce `intermediate_size` or `num_experts`.
