"""
LogitLens analysis for mini-gpt-oss: compare prediction convergence across layers.

Implements the analysis from arXiv:2601.07372 Section 6.1.1:
  - Projects each layer's hidden state through the final LM head
  - Computes KL divergence between each layer's output distribution and the
    final layer's distribution
  - Shows that the Engram model reaches prediction-ready representations at
    earlier layers (= "effectively deeper" network)

Usage:
    python logitlens.py \\
        --baseline_ckpt out_baseline/ckpt.pt \\
        --engram_ckpt out_engram/ckpt.pt \\
        --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

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
    sys.path.insert(0, model_dir)
    for mod in ("model", "tokenizer", "engram"):
        if mod in sys.modules:
            del sys.modules[mod]
    from model import MiniGPTOSS
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MiniGPTOSS(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sys.path.remove(model_dir)
    return model, cfg


@torch.no_grad()
def logitlens_kl(model, cfg, device, data_dir, n_batches=50, batch_size=4):
    """
    For each transformer layer, project the hidden state through the LM head
    and compute KL(layer_dist || final_dist). Returns per-layer mean KL.
    """
    val_path = os.path.join(data_dir, "val.bin")
    if not os.path.exists(val_path):
        print(f"  [WARN] {val_path} not found")
        return None
    data = np.memmap(val_path, dtype=np.uint32, mode="r")

    n_layers = cfg.num_hidden_layers
    kl_sums = [0.0] * n_layers
    count = 0

    for _ in range(n_batches):
        ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix]).to(device)

        # Forward pass, collecting hidden states at each layer
        # We need to hook into the model to get per-layer hidden states
        hidden_states = []

        # Run embedding
        h = model.embedding(x)

        # Check if model uses engram (accepts token ids in blocks)
        has_engram = hasattr(cfg, "use_engram") and cfg.use_engram
        if has_engram:
            c = model.canonical[x]

        for i, block in enumerate(model.blocks):
            if has_engram:
                h, _ = block(h, c)
            else:
                h, _ = block(h)
            hidden_states.append(h)

        # Final norm + LM head on the LAST layer's output -> "final" distribution
        h_final = model.norm(hidden_states[-1])
        logits_final = model.unembedding(h_final).float()
        p_final = F.softmax(logits_final, dim=-1)  # [B, T, V]

        # For each intermediate layer, project through norm + LM head -> layer distribution
        for layer_idx in range(n_layers):
            h_layer = model.norm(hidden_states[layer_idx])
            logits_layer = model.unembedding(h_layer).float()
            p_layer = F.log_softmax(logits_layer, dim=-1)

            # KL(final || layer) = sum p_final * (log p_final - log p_layer)
            kl = F.kl_div(p_layer, p_final, reduction="batchmean", log_target=False)
            kl_sums[layer_idx] += kl.item()

        count += 1

    return [s / count for s in kl_sums]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--engram_ckpt", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--n_batches", type=int, default=50)
    args = ap.parse_args()

    device = pick_device(args.device)
    data_dir = args.data_dir or DATA_DIR
    baseline_dir = os.path.join(HERE, "mini-gpt-oss")
    engram_dir = os.path.join(HERE, "mini-gpt-oss-engram")

    # ---- Baseline ----
    print("Loading baseline model...")
    model_b, cfg_b = load_model(args.baseline_ckpt, baseline_dir, device)
    print(f"  {cfg_b.num_hidden_layers} layers, {model_b.num_params()/1e6:.1f}M params")
    print("Computing LogitLens KL divergence for baseline...")
    kl_b = logitlens_kl(model_b, cfg_b, device, data_dir, args.n_batches)
    del model_b
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Engram ----
    print("Loading engram model...")
    model_e, cfg_e = load_model(args.engram_ckpt, engram_dir, device)
    print(f"  {cfg_e.num_hidden_layers} layers, {model_e.num_params()/1e6:.1f}M params")
    print("Computing LogitLens KL divergence for engram...")
    kl_e = logitlens_kl(model_e, cfg_e, device, data_dir, args.n_batches)
    del model_e
    if device == "cuda":
        torch.cuda.empty_cache()

    if kl_b is None or kl_e is None:
        print("Could not compute KL — val data missing.")
        return

    # ---- Print results ----
    n_layers = len(kl_b)
    print(f"\n{'='*60}")
    print("  LogitLens Analysis (arXiv:2601.07372, Section 6.1.1)")
    print(f"  KL(final_layer || layer_i) — lower = closer to final prediction")
    print(f"{'='*60}")
    print(f"\n  {'Layer':<8} {'Baseline KL':>14} {'Engram KL':>14} {'Δ':>10}")
    print(f"  {'─'*46}")
    for i in range(n_layers):
        delta = kl_e[i] - kl_b[i]
        marker = " ◀ Engram closer" if delta < -0.01 else ""
        print(f"  {i+1:<8} {kl_b[i]:>14.4f} {kl_e[i]:>14.4f} {delta:>+10.4f}{marker}")

    print(f"\n  Interpretation:")
    print(f"  - Lower KL = hidden state is closer to the final prediction")
    print(f"  - If Engram KL is lower at early layers, it means Engram")
    print(f"    reaches prediction-ready representations FASTER")
    print(f"  - This confirms the paper's finding: Engram 'effectively deepens'")
    print(f"    the network by offloading static pattern reconstruction")
    print(f"{'='*60}\n")

    # ---- Save for plotting ----
    import json
    out = {"baseline_kl": kl_b, "engram_kl": kl_e, "n_layers": n_layers}
    out_path = os.path.join(os.path.dirname(args.baseline_ckpt), "logitlens_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
"""
