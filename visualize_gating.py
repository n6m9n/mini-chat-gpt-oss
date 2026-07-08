"""
Gating visualization for the Engram model.

Implements the analysis from arXiv:2601.07372 Section 6.5:
  - Runs inference on sample prompts
  - Extracts the gating scalar alpha_t for each token position at each Engram layer
  - Prints a text-based heatmap showing where Engram activates
  - High activation (red) = Engram memory is being used (formulaic/entity patterns)
  - Low activation = Engram suppressed (novel/context-dependent content)

Usage:
    python visualize_gating.py --ckpt out_engram/ckpt.pt --device cuda
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    engram_dir = os.path.join(HERE, "mini-gpt-oss-engram")
    sys.path.insert(0, engram_dir)
    for mod in ("model", "tokenizer", "engram"):
        if mod in sys.modules:
            del sys.modules[mod]
    from model import MiniGPTOSS
    from tokenizer import encode, get_tokenizer

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MiniGPTOSS(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sys.path.remove(engram_dir)
    return model, cfg, get_tokenizer()


@torch.no_grad()
def extract_gating(model, cfg, x, device):
    """
    Run a forward pass and extract gating scalars alpha_t from each Engram layer.
    Returns a dict: {layer_idx: alpha tensor [B, T]}.
    """
    gating_values = {}

    h = model.embedding(x)
    c = model.canonical[x] if cfg.use_engram else x

    for layer_idx, block in enumerate(model.blocks):
        if block.engram is not None:
            # Manually run engram forward to capture alpha
            engram = block.engram
            B, T, D = h.shape

            # Build N-gram embeddings
            embs, ti = [], 0
            for n in engram.orders:
                c_pad = torch.nn.functional.pad(c, (n - 1, 0))
                comps = [c_pad[:, j:j + T] for j in range(n)]
                for _ in range(engram.K):
                    idx = engram._hash(comps, engram.seeds[ti], h.device, (B, T))
                    embs.append(engram.tables[ti](idx))
                    ti += 1
            e = torch.cat(embs, dim=-1)

            kt, vt = engram.WK(e), engram.WV(e)
            alpha = torch.sigmoid(
                (engram.rms_q(h) * engram.rms_k(kt)).sum(-1, keepdim=True) * engram.inv_sqrt_d
            )
            gating_values[layer_idx] = alpha.squeeze(-1).cpu()  # [B, T]

            # Complete the engram forward (so the rest of the model gets correct input)
            vtil = alpha * vt
            z = engram.rms_v(vtil).transpose(1, 2)
            z = torch.nn.functional.pad(z, (engram.pad, 0))
            z = engram.conv(z).transpose(1, 2)
            y = torch.nn.functional.silu(z) + vtil
            h = h + y

        # Run attention + MLP
        if hasattr(cfg, 'use_engram') and cfg.use_engram:
            h, _ = block(model.embedding(x) if False else h, c)  # block.forward includes engram
            # Actually we already did engram, need to skip it. Let's just run attn+mlp manually.
            pass

        # We need to handle this properly - run attn and mlp only
        h_attn = block.attn(h)
        h_attn, aux = block.mlp(h_attn)
        h = h_attn

    return gating_values


@torch.no_grad()
def extract_gating_simple(model, cfg, x, device):
    """
    Simpler approach: hook into the engram blocks to capture alpha values.
    """
    gating_values = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # EngramBlock forward: x, c -> x + y
            # We need to recompute alpha. Instead, let's store it during forward.
            pass
        return hook_fn

    # Monkey-patch engram forward to capture alpha
    original_forwards = {}
    for layer_idx, block in enumerate(model.blocks):
        if block.engram is not None:
            engram = block.engram
            original_forward = engram.forward

            def patched_forward(self, x, c, _layer_idx=layer_idx):
                B, T, D = x.shape
                embs, ti = [], 0
                for n in self.orders:
                    c_pad = torch.nn.functional.pad(c, (n - 1, 0))
                    comps = [c_pad[:, j:j + T] for j in range(n)]
                    for _ in range(self.K):
                        idx = self._hash(comps, self.seeds[ti], x.device, (B, T))
                        embs.append(self.tables[ti](idx))
                        ti += 1
                e = torch.cat(embs, dim=-1)

                kt, vt = self.WK(e), self.WV(e)
                alpha = torch.sigmoid(
                    (self.rms_q(x) * self.rms_k(kt)).sum(-1, keepdim=True) * self.inv_sqrt_d
                )
                # Store alpha for visualization
                gating_values[_layer_idx] = alpha.squeeze(-1).cpu()

                vtil = alpha * vt
                z = self.rms_v(vtil).transpose(1, 2)
                z = torch.nn.functional.pad(z, (self.pad, 0))
                z = self.conv(z).transpose(1, 2)
                y = torch.nn.functional.silu(z) + vtil
                return x + y

            import types
            engram.forward = types.MethodType(patched_forward, engram)
            original_forwards[layer_idx] = original_forward

    # Run forward pass
    model(x)

    # Restore original forwards
    for layer_idx, block in enumerate(model.blocks):
        if layer_idx in original_forwards:
            block.engram.forward = original_forwards[layer_idx]

    return gating_values


def alpha_to_bar(alpha, width=20):
    """Convert alpha (0-1) to a text bar with intensity."""
    filled = int(alpha * width)
    return "█" * filled + "░" * (width - filled)


def alpha_to_color_char(alpha):
    """Map alpha to intensity character."""
    chars = " ░▒▓█"
    idx = min(int(alpha * len(chars)), len(chars) - 1)
    return chars[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to engram ckpt.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompts", nargs="+", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg, tokenizer = load_model(args.ckpt, device)

    if not (hasattr(cfg, "use_engram") and cfg.use_engram):
        print("ERROR: This checkpoint does not have Engram enabled.")
        print("Use a checkpoint from mini-gpt-oss-engram/")
        return

    prompts = args.prompts or [
        "Once upon a time, there was a little cat named Tom.",
        "The princess lived in a big castle with her friends.",
        "One day, the sun was shining and the birds were singing.",
        "A small dog named Max ran to the park to play.",
    ]

    engram_layers = [i for i, b in enumerate(model.blocks) if b.engram is not None]
    print(f"Engram layers: {engram_layers}")

    print(f"\n{'='*72}")
    print(f"  Gating Visualization (arXiv:2601.07372, Section 6.5)")
    print(f"  alpha_t: high = Engram memory active, low = suppressed")
    print(f"{'='*72}")

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        tokens = [tokenizer.decode([t]) for t in ids]

        gating = extract_gating_simple(model, cfg, x, device)

        print(f"\n  PROMPT: \"{prompt}\"")

        for layer_idx in sorted(gating.keys()):
            alphas = gating[layer_idx][0]  # first batch element
            print(f"\n  Layer {layer_idx} gating (alpha_t):")
            print(f"  {'Token':<20} {'alpha':>6}  {'Visualization'}")
            print(f"  {'─'*55}")

            for t_idx, (tok, alpha) in enumerate(zip(tokens, alphas)):
                a = alpha.item()
                bar = alpha_to_bar(a, width=25)
                marker = " ← HIGH" if a > 0.6 else ""
                print(f"  {tok:<20} {a:>6.3f}  {bar}{marker}")

            mean_alpha = alphas.mean().item()
            max_alpha = alphas.max().item()
            min_alpha = alphas.min().item()
            print(f"  {'─'*55}")
            print(f"  mean={mean_alpha:.3f}  max={max_alpha:.3f}  min={min_alpha:.3f}")

    print(f"\n{'='*72}")
    print(f"  Interpretation (from the paper):")
    print(f"  - High alpha on multi-token entities and formulaic phrases")
    print(f"    means Engram is retrieving stored patterns")
    print(f"  - Low alpha means the model relies on attention/computation")
    print(f"  - TinyStories should show high gating on: 'Once upon a time',")
    print(f"    character names, 'The end', formulaic story elements")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
"""
