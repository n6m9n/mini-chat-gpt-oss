# mini-gpt-oss + Engram (~267M)

This folder is the **MoE + Engram** variant of mini-gpt-oss. It is identical to
the baseline (`../mini gpt oss`) except for one added module: **Engram**, the
conditional-memory lookup from *"Conditional Memory via Scalable Lookup"*
(DeepSeek-AI, arXiv:2601.07372). Keeping it in a separate folder lets you train
both and compare **MoE vs MoE+Engram** cleanly.

## What Engram adds

At two early blocks (**1 and 3**), before attention, an `EngramBlock`:
1. builds **suffix {2,3}-grams** from the token ids (after *tokenizer compression*),
2. hashes each with **K=2** multiply-XOR heads into embedding tables (O(1) lookup),
3. **gates** the retrieved static memory with the current hidden state (so a
   wrong/colliding lookup gets suppressed),
4. refines with a **depthwise causal conv** and adds it to the residual stream.

It's a second axis of sparsity: MoE = conditional *computation*, Engram =
conditional *memory*. TinyStories is highly formulaic ("Once upon a time…"),
which is exactly the local-pattern regime Engram is meant to offload.

## Parameter budget

```
Engram tables : 4 tables × 16,381 × 128 × 2 blocks ≈ 16.8M   (sparse: 4 rows read/token)
W_K/W_V/norms/conv                                 ≈  1.6M
-----------------------------------------------------------
Engram total                                       ≈ 18.4M
model total : ~249M (baseline)  ->  ~267M
active/token added: ~0.3M  (4 hash lookups + two 512×768 matmuls + conv)
```

## Design choices (from the paper's ablations, Sec. 6.2)

| Choice | Reason |
|---|---|
| inject at blocks **1 & 3** | scaled from their layers 2 & 6 of 12; layer 2 optimal, needs ≥1 attention round for a good gating query |
| **{2,3}-grams** only | 4-grams *hurt* under a fixed budget (dilute frequent 2/3-gram capacity) |
| **K=2** hash heads | reduce hash collisions |
| **gating + tokenizer compression** | their 2 most important components |
| tables: **5× LR, no weight decay** | Sec. 4.1 optimizer setting |
| **conv zero-init** | Engram's conv branch starts as identity → smooth ramp-in |

## How to run

```bash
pip install -r requirements.txt

# 0. (once) build the tokenizer-compression map  V -> V'  (~1-2 min, decodes vocab)
python build_compression.py                 # writes data/canonical_ids.npy

# 1. sanity check (~267M total, ~18M Engram)
python model.py

# 2. tokenize TinyStories (shared with baseline)
python data.py --split validation

# 3a. train MoE + Engram
python train.py --device cuda --max_iters 20000
# 3b. train the pure-MoE baseline *in this same folder* (Engram off) for A/B:
python train.py --device cuda --max_iters 20000 --no_engram

# 4. generate
python generate.py --prompt "Once upon a time"
```

### Resuming (Kaggle 12h limit) & shared data
`train.py` saves `out/ckpt.pt` (best, for generate) + `out/latest.pt` (full state).
Resume with the **same flags** you started with:
```bash
python train.py --device cuda --max_iters 20000 --resume            # Engram
python train.py --device cuda --max_iters 20000 --resume --no_engram # if you started with --no_engram
```
On Kaggle, tokenize once and point both models at the shared data + a writable out dir:
```bash
DATA_DIR=/kaggle/working/data python build_compression.py
DATA_DIR=/kaggle/working/data python data.py --split train
CUDA_VISIBLE_DEVICES=1 DATA_DIR=/kaggle/working/data OUT_DIR=/kaggle/working/out_engram \
  python train.py --device cuda --resume
```

## Comparing MoE vs MoE+Engram
Two clean options:
- **Same code, flag off:** run `train.py --no_engram` here vs the default — isolates
  the Engram module exactly (identical everything else).
- **Two folders:** train `../mini gpt oss` (baseline) and this one, compare `out/ckpt.pt`
  val losses / sample quality.

For a fair *iso-parameter* comparison (the paper's U-curve), later reduce
`num_experts` (8→6) here so total params match the baseline; for now this is the
simpler **add-on** version (~267M > 249M).

## Files (only these differ from baseline)
- `engram.py` — the `EngramBlock` (hashing + tables + gating + conv).
- `model.py` — baseline model + Engram config, token-id threading, injection at blocks 1 & 3.
- `train.py` — adds the 5×-LR / no-weight-decay optimizer group for the tables; `--no_engram` flag.
- `build_compression.py` — precompute the vocab→canonical id map.
