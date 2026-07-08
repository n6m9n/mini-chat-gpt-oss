"""
Evaluate a trained mini-gpt-oss checkpoint.

Measures:
  1. Val loss & perplexity on the TinyStories held-out split
  2. Generation quality (fixed prompts, deterministic seed)
  3. Optional: lm-evaluation-harness benchmarks (HellaSwag, ARC, PIQA, WinoGrande)

Usage:
    python evaluate.py --ckpt out/ckpt.pt --device cuda
    python evaluate.py --ckpt out/ckpt.pt --device cuda --benchmarks   # run lm-eval benchmarks
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# ---- model imports (try engram variant first, fall back to baseline) ----
try:
    from model import MiniGPTOSS, ModelConfig
    from tokenizer import EOT, decode, encode, VOCAB_SIZE
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import MiniGPTOSS, ModelConfig
    from tokenizer import EOT, decode, encode, VOCAB_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MiniGPTOSS(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt.get("val_loss", None)


@torch.no_grad()
def compute_val_loss(model, cfg, device, eval_iters=200):
    """Compute average val loss over many batches."""
    val_path = os.path.join(DATA_DIR, "val.bin")
    if not os.path.exists(val_path):
        print(f"  [WARN] {val_path} not found, skipping val loss")
        return None
    data = np.memmap(val_path, dtype=np.uint32, mode="r")
    batch_size = 8
    losses = []
    for k in range(eval_iters):
        ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i+1:i+1 + cfg.block_size].astype(np.int64)) for i in ix]).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def generate_samples(model, device, prompts, max_tokens=200, temperature=0.8, top_k=50, seed=1337):
    """Generate text from fixed prompts for qualitative comparison."""
    results = {}
    for prompt in prompts:
        torch.manual_seed(seed)
        ids = encode(prompt) or [EOT]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_tokens, temperature=temperature, top_k=top_k, eot_token=EOT)
        results[prompt] = decode(out[0].tolist())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ckpt.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--benchmarks", action="store_true", help="run lm-eval-harness benchmarks")
    ap.add_argument("--label", default="", help="label for this model (e.g. 'baseline' or 'engram')")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")
    print(f"loading checkpoint: {args.ckpt}")

    model, cfg, saved_val = load_model(args.ckpt, device)

    # ---- Model info ----
    has_engram = hasattr(cfg, "use_engram") and cfg.use_engram
    label = args.label or ("engram" if has_engram else "baseline")
    total_params = model.num_params()
    non_emb = model.num_params(non_embedding=True)
    engram_params = model.num_params(engram_only=True) if hasattr(model, "num_params") and "engram_only" in model.num_params.__code__.co_varnames else 0

    print(f"\n{'='*60}")
    print(f"  Model: {label}")
    print(f"  Total params:         {total_params / 1e6:.1f}M")
    print(f"  Non-embedding params: {non_emb / 1e6:.1f}M")
    if engram_params > 0:
        print(f"  Engram params:        {engram_params / 1e6:.1f}M")
    print(f"  Block size:           {cfg.block_size}")
    print(f"  Saved val loss:       {saved_val:.4f}" if saved_val else "  Saved val loss:       N/A")
    print(f"{'='*60}\n")

    # ---- 1. Val loss & perplexity ----
    print("[1/3] Computing val loss...")
    t0 = time.time()
    val_loss = compute_val_loss(model, cfg, device, args.eval_iters)
    if val_loss is not None:
        val_ppl = math.exp(val_loss)
        print(f"  val loss:       {val_loss:.4f}")
        print(f"  val perplexity: {val_ppl:.2f}")
        print(f"  ({time.time() - t0:.1f}s)\n")
    else:
        val_ppl = None

    # ---- 2. Generation samples ----
    print("[2/3] Generating samples...")
    prompts = [
        "Once upon a time",
        "The little robot walked",
        "A princess lived in a big",
        "One day, the sun was",
        "There was a small dog named",
    ]
    samples = generate_samples(model, device, prompts)
    for prompt, text in samples.items():
        print(f"\n  PROMPT: \"{prompt}\"")
        print(f"  OUTPUT: {text[:300]}{'...' if len(text) > 300 else ''}")

    # ---- 3. Benchmarks (optional) ----
    if args.benchmarks:
        print("\n[3/3] Running lm-eval benchmarks...")
        try:
            import lm_eval
            # lm-eval-harness integration would go here
            # For 250M models, these will be near-random but still show relative differences
            print("  [TODO] lm-eval-harness integration — install with: pip install lm-eval")
            print("  For 250M models trained on TinyStories, standard benchmarks will be low.")
            print("  The primary comparison metric is val loss + generation quality.")
        except ImportError:
            print("  [SKIP] lm-eval not installed. pip install lm-eval")
    else:
        print("\n[3/3] Skipping benchmarks (use --benchmarks to enable)")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {label}")
    print(f"  Total params:   {total_params / 1e6:.1f}M")
    if val_loss is not None:
        print(f"  Val loss:       {val_loss:.4f}")
        print(f"  Val perplexity: {val_ppl:.2f}")
    print(f"{'='*60}")

    # Save results to a JSON for compare.py
    import json
    results = {
        "label": label,
        "ckpt": args.ckpt,
        "total_params_M": total_params / 1e6,
        "non_emb_params_M": non_emb / 1e6,
        "engram_params_M": engram_params / 1e6,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "samples": samples,
    }
    out_path = os.path.join(os.path.dirname(args.ckpt), f"eval_{label}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
"""
