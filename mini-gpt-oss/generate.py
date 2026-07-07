"""
Inference / text generation for mini-gpt-oss.

    python generate.py --prompt "Once upon a time" --max_new_tokens 300

Loads out/ckpt.pt (weights + ModelConfig) and the o200k_harmony tokenizer.
"""

import argparse
import os

import torch

from model import MiniGPTOSS
from tokenizer import EOT, decode, encode

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT_DIR, "ckpt.pt"))
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = MiniGPTOSS(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    start = encode(args.prompt) or [EOT]
    idx = torch.tensor([start], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k, eot_token=EOT)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
