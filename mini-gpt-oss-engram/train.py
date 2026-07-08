"""
Training loop for mini-gpt-oss + Engram.

Same as the baseline trainer, with one Engram-specific change: the Engram
embedding TABLES get their own optimizer group with a 5x learning rate and no
weight decay (as in the paper, Sec. 4.1). All other params train normally.

    python build_compression.py                 # once: enables tokenizer compression
    python data.py --split validation           # tokenize TinyStories
    python train.py --device cuda --max_iters 20000
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from model import MiniGPTOSS, ModelConfig
from tokenizer import VOCAB_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
# override via env for shared data / writable output (e.g. Kaggle /kaggle/working)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(split, cfg, batch_size, device):
    data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint32, mode="r")
    ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, cfg, batch_size, device, ctx, eval_iters=50):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, cfg, batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, warmup, max_iters, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, (max_iters - warmup))
    coeff = 0.5 * (1.0 + np.cos(np.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def fmt_hms(seconds):
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def save_full(path, model, optimizer, scaler, it, best_val, cfg):
    """Full training state, so a run can resume after Kaggle's 12h session limit."""
    raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    torch.save({"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "iter": it, "best_val": best_val,
                "config": cfg}, path)


def build_optimizer(model, lr, weight_decay):
    """Two groups: Engram tables (5x LR, no weight decay) and everything else."""
    table_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "engram" in n and "tables" in n:
            table_params.append(p)
        else:
            other_params.append(p)
    groups = [
        {"params": other_params, "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": table_params, "weight_decay": 0.0, "lr_scale": 5.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    print(f"optimizer groups: {len(other_params)} tensors @1x LR | "
          f"{len(table_params)} Engram-table tensors @5x LR, wd=0")
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=20000)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--no_engram", action="store_true", help="disable Engram (pure-MoE baseline in this folder)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume from out/latest.pt if present")
    ap.add_argument("--log_interval", type=int, default=20, help="print a progress line every N iters")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    device_type = "cuda" if "cuda" in device else ("mps" if device == "mps" else "cpu")
    print(f"device: {device}")

    use_amp = device_type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    cfg = ModelConfig(vocab_size=VOCAB_SIZE, block_size=args.block_size,
                      initial_context_length=args.block_size, use_engram=not args.no_engram)
    model = MiniGPTOSS(cfg).to(device)
    print(f"parameters: {model.num_params() / 1e6:.1f}M total "
          f"({model.num_params(engram_only=True) / 1e6:.1f}M Engram) | "
          f"{model.num_params(non_embedding=True) / 1e6:.1f}M non-embedding")
    if args.compile:
        model = torch.compile(model)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    # ---- Weights & Biases (activate via WANDB_PROJECT env var) ----
    use_wandb = wandb is not None and os.environ.get("WANDB_PROJECT")
    if use_wandb:
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=os.environ.get("WANDB_NAME", "engram-moe"),
            config={
                "model": "mini-gpt-oss-engram",
                "use_engram": not args.no_engram,
                "params_M": model.num_params() / 1e6,
                "engram_params_M": model.num_params(engram_only=True) / 1e6,
                "non_emb_params_M": model.num_params(non_embedding=True) / 1e6,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "block_size": args.block_size,
                "max_iters": args.max_iters,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "warmup": args.warmup,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "device": device,
                "seed": args.seed,
            },
        )
        print("[wandb] logging enabled")

    os.makedirs(OUT_DIR, exist_ok=True)
    latest_path = os.path.join(OUT_DIR, "latest.pt")
    best_path = os.path.join(OUT_DIR, "ckpt.pt")
    start_iter, best_val = 0, float("inf")

    if args.resume and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        getattr(model, "_orig_mod", model).load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_iter, best_val = ck["iter"] + 1, ck["best_val"]
        print(f"resumed from {latest_path} at iter {start_iter} (best_val {best_val:.4f})")

    tokens_per_iter = args.batch_size * args.block_size * args.grad_accum
    t0 = t_log = time.time()
    model.train()

    for it in range(start_iter, args.max_iters + 1):
        lr = get_lr(it, args.warmup, args.max_iters, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr * g["lr_scale"]           # keep the 5x ratio for the table group

        if it % args.eval_interval == 0:
            losses = estimate_loss(model, cfg, args.batch_size, device, ctx)
            print(f"[eval] iter {it:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | elapsed {fmt_hms(time.time() - t0)}", flush=True)
            if losses["val"] < best_val:
                best_val = losses["val"]
                raw = getattr(model, "_orig_mod", model)
                torch.save({"model": raw.state_dict(), "config": cfg, "val_loss": best_val}, best_path)
            # rolling full-state checkpoint for resume (overwrites, bounded disk)
            save_full(latest_path, model, optimizer, scaler, it, best_val, cfg)
            if use_wandb:
                log = {
                    "eval/train_loss": losses["train"],
                    "eval/val_loss": losses["val"],
                    "eval/train_ppl": math.exp(losses["train"]),
                    "eval/val_ppl": math.exp(losses["val"]),
                    "eval/best_val_loss": best_val,
                    "lr": lr,
                }
                if device_type == "cuda":
                    log["gpu/mem_allocated_GB"] = torch.cuda.memory_allocated() / 1e9
                    log["gpu/mem_reserved_GB"] = torch.cuda.memory_reserved() / 1e9
                wandb.log(log, step=it)

        if it == args.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch("train", cfg, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # live progress line
        if it % args.log_interval == 0:
            now = time.time()
            tps = args.log_interval * tokens_per_iter / (now - t_log)
            t_log = now
            done = it - start_iter + 1
            eta = (args.max_iters - it) * (now - t0) / max(done, 1)
            print(f"iter {it:6d}/{args.max_iters} ({100 * it / args.max_iters:4.1f}%) | "
                  f"loss {loss.item() * args.grad_accum:.3f} | {tps:6.0f} tok/s | "
                  f"elapsed {fmt_hms(now - t0)} | eta {fmt_hms(eta)}", flush=True)
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item() * args.grad_accum,
                    "train/tok_per_sec": tps,
                    "lr": lr,
                }, step=it)

    print(f"done. best val loss: {best_val:.4f}  ->  {best_path}", flush=True)
    if use_wandb:
        wandb.log({"final/best_val_loss": best_val, "final/best_val_ppl": math.exp(best_val)})
        wandb.finish()


if __name__ == "__main__":
    main()
