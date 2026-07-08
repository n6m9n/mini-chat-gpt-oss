"""
Side-by-side comparison of baseline MoE vs MoE+Engram.

Loads both checkpoints, runs evaluation, and prints a comparison table
following the methodology from arXiv:2601.07372 (Sections 4 & 6).

Usage:
    python compare.py \\
        --baseline_ckpt out_baseline/ckpt.pt \\
        --engram_ckpt out_engram/ckpt.pt \\
        --device cuda
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# We need both models. Add both directories to path.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "mini-gpt-oss", "data"))


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, model_dir, device):
    """Load a model from ckpt, using the model.py from model_dir."""
    # Temporarily add model_dir to path for imports
    sys.path.insert(0, model_dir)
    # Force reimport
    if "model" in sys.modules:
        del sys.modules["model"]
    if "tokenizer" in sys.modules:
        del sys.modules["tokenizer"]
    if "engram" in sys.modules:
        del sys.modules["engram"]

    from model import MiniGPTOSS, ModelConfig
    from tokenizer import EOT, VOCAB_SIZE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MiniGPTOSS(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sys.path.remove(model_dir)
    return model, cfg, ckpt.get("val_loss", None), EOT, VOCAB_SIZE


@torch.no_grad()
def compute_val_loss(model, cfg, device, data_dir, eval_iters=200, batch_size=8):
    val_path = os.path.join(data_dir, "val.bin")
    if not os.path.exists(val_path):
        return None
    data = np.memmap(val_path, dtype=np.uint32, mode="r")
    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i+1:i+1 + cfg.block_size].astype(np.int64)) for i in ix]).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def generate_sample(model, prompt_ids, eot, device, max_tokens=200, temperature=0.8, top_k=50, seed=1337):
    torch.manual_seed(seed)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_tokens, temperature=temperature, top_k=top_k, eot_token=eot)
    return out[0].tolist()


def measure_throughput(model, cfg, device, n_iters=50, batch_size=8):
    """Measure inference throughput in tokens/sec."""
    x = torch.randint(0, 1000, (batch_size, cfg.block_size), device=device)
    # warmup
    for _ in range(5):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    total_tokens = n_iters * batch_size * cfg.block_size
    return total_tokens / elapsed


def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--engram_ckpt", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--data_dir", default=None, help="shared data directory")
    args = ap.parse_args()

    device = pick_device(args.device)
    data_dir = args.data_dir or DATA_DIR

    baseline_dir = os.path.join(HERE, "mini-gpt-oss")
    engram_dir = os.path.join(HERE, "mini-gpt-oss-engram")

    # ---- Load baseline ----
    print("Loading baseline model...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model_b, cfg_b, saved_val_b, eot_b, _ = load_model(args.baseline_ckpt, baseline_dir, device)
    baseline_mem = get_gpu_memory()

    # ---- Load tokenizer for decoding ----
    sys.path.insert(0, baseline_dir)
    if "tokenizer" in sys.modules:
        del sys.modules["tokenizer"]
    from tokenizer import decode, encode
    sys.path.remove(baseline_dir)

    # ---- Evaluate baseline ----
    print("Evaluating baseline...")
    val_loss_b = compute_val_loss(model_b, cfg_b, device, data_dir, args.eval_iters)
    tps_b = measure_throughput(model_b, cfg_b, device)
    params_b = model_b.num_params()

    # Free baseline
    del model_b
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- Load engram ----
    print("Loading engram model...")
    model_e, cfg_e, saved_val_e, eot_e, _ = load_model(args.engram_ckpt, engram_dir, device)
    engram_mem = get_gpu_memory()

    # ---- Evaluate engram ----
    print("Evaluating engram...")
    val_loss_e = compute_val_loss(model_e, cfg_e, device, data_dir, args.eval_iters)
    tps_e = measure_throughput(model_e, cfg_e, device)
    params_e = model_e.num_params()
    try:
        engram_params = model_e.num_params(engram_only=True)
    except TypeError:
        engram_params = 0

    # ---- Generate samples ----
    prompts_text = [
        "Once upon a time",
        "The little robot walked",
        "A princess lived in a big",
        "One day, the sun was",
        "There was a small dog named",
    ]

    print("\nGenerating samples from both models...")
    # Reload baseline for generation
    model_b2, _, _, _, _ = load_model(args.baseline_ckpt, baseline_dir, device)

    samples_b, samples_e = {}, {}
    for prompt in prompts_text:
        ids = encode(prompt) or [eot_b]
        out_b = generate_sample(model_b2, ids, eot_b, device)
        out_e = generate_sample(model_e, ids, eot_e, device)
        samples_b[prompt] = decode(out_b)
        samples_e[prompt] = decode(out_e)

    del model_b2, model_e
    if device == "cuda":
        torch.cuda.empty_cache()

    # ======================================================================
    # PRINT COMPARISON
    # ======================================================================
    print("\n" + "=" * 72)
    print("  COMPARISON: Baseline MoE  vs  MoE + Engram")
    print("  (Following arXiv:2601.07372 evaluation methodology)")
    print("=" * 72)

    # Parameter table
    print(f"\n{'─'*72}")
    print(f"  {'Metric':<30} {'Baseline MoE':>15} {'MoE+Engram':>15} {'Δ':>8}")
    print(f"{'─'*72}")
    print(f"  {'Total params':<30} {params_b/1e6:>14.1f}M {params_e/1e6:>14.1f}M {(params_e-params_b)/1e6:>+7.1f}M")
    if engram_params > 0:
        print(f"  {'Engram params':<30} {'0':>15} {engram_params/1e6:>14.1f}M {'':>8}")

    # Performance table
    print(f"{'─'*72}")
    if val_loss_b is not None and val_loss_e is not None:
        ppl_b, ppl_e = math.exp(val_loss_b), math.exp(val_loss_e)
        delta_loss = val_loss_e - val_loss_b
        delta_ppl = ppl_e - ppl_b
        print(f"  {'Val loss':<30} {val_loss_b:>15.4f} {val_loss_e:>15.4f} {delta_loss:>+8.4f}")
        print(f"  {'Val perplexity':<30} {ppl_b:>15.2f} {ppl_e:>15.2f} {delta_ppl:>+8.2f}")
    elif saved_val_b is not None and saved_val_e is not None:
        print(f"  {'Best val loss (saved)':<30} {saved_val_b:>15.4f} {saved_val_e:>15.4f} {saved_val_e-saved_val_b:>+8.4f}")

    # Throughput
    print(f"{'─'*72}")
    delta_tps = ((tps_e - tps_b) / tps_b) * 100
    print(f"  {'Inference tok/s':<30} {tps_b:>15,.0f} {tps_e:>15,.0f} {delta_tps:>+7.1f}%")
    print(f"  {'Peak GPU memory (GB)':<30} {baseline_mem:>15.2f} {engram_mem:>15.2f} {engram_mem-baseline_mem:>+8.2f}")
    print(f"{'─'*72}")

    # Generation samples
    print(f"\n{'─'*72}")
    print("  GENERATION SAMPLES (same prompt, seed=1337, temp=0.8, top_k=50)")
    print(f"{'─'*72}")
    for prompt in prompts_text:
        print(f"\n  PROMPT: \"{prompt}\"")
        b_text = samples_b[prompt][:250]
        e_text = samples_e[prompt][:250]
        print(f"  BASELINE: {b_text}{'...' if len(samples_b[prompt]) > 250 else ''}")
        print(f"  ENGRAM:   {e_text}{'...' if len(samples_e[prompt]) > 250 else ''}")

    print(f"\n{'='*72}")
    print("  To visualize training curves, open your W&B dashboard:")
    print("  https://wandb.ai → project: mini-gpt-oss-comparison")
    print("  Overlay both runs to compare loss/perplexity over training steps.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
"""
