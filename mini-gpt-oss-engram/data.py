"""
Data preparation for mini-gpt-oss.

Loads TinyStories from HuggingFace, tokenizes it with the o200k_harmony BPE
(tokenizer.py), separates stories with <|endoftext|>, and writes flat token
streams that train.py memory-maps.

NOTE: token ids go up to 201,087, so they do NOT fit in uint16 -> we use uint32.

Usage:
    python data.py                       # full valid split (fast, ~2.7M stories? no: valid is small)
    python data.py --split train          # the big split (lots of stories)
    python data.py --max_docs 100000      # cap for a quicker run
    python data.py --sample               # tiny built-in corpus, no download
"""

import argparse
import os

import numpy as np

from tokenizer import EOT, get_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))

SAMPLE_DOCS = [
    "Once upon a time, there was a little cat named Tom. Tom liked to play in the "
    "sun. One day he saw a big red ball and kicked it high into the sky. Tom was happy.",
    "Lily had a small blue boat. She sailed it on the pond every morning. A friendly "
    "duck swam beside her and they watched the fish together. The end.",
] * 500


def write_split(name: str, ids: np.ndarray):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.bin")
    ids.astype(np.uint32).tofile(path)
    print(f"  wrote {name:5s}: {len(ids):>12,} tokens -> {path}")


def tokenize_docs(docs):
    """Encode each doc and append <|endoftext|>; return one flat uint32 array."""
    enc = get_tokenizer()
    chunks = []
    total = 0
    for i, txt in enumerate(docs):
        toks = enc.encode(txt)
        toks.append(EOT)
        chunks.append(np.array(toks, dtype=np.uint32))
        total += len(toks)
        if (i + 1) % 20000 == 0:
            print(f"    tokenized {i + 1:,} docs ({total:,} tokens)")
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train", "validation"],
                    help="TinyStories split to use as the training corpus")
    ap.add_argument("--max_docs", type=int, default=0, help="cap number of stories (0 = all)")
    ap.add_argument("--val_frac", type=float, default=0.02, help="fraction held out for val")
    ap.add_argument("--sample", action="store_true", help="use the built-in sample (no download)")
    args = ap.parse_args()

    if args.sample:
        docs = SAMPLE_DOCS
    else:
        from datasets import load_dataset
        print(f"loading TinyStories split='{args.split}' ...")
        ds = load_dataset("roneneldan/TinyStories", split=args.split)
        docs = ds["text"]
        if args.max_docs:
            docs = docs[: args.max_docs]
    print(f"documents: {len(docs):,}")

    ids = tokenize_docs(docs)
    print(f"total tokens: {len(ids):,}")

    n_val = int(len(ids) * args.val_frac)
    write_split("val", ids[:n_val])
    write_split("train", ids[n_val:])
    print("done.")


if __name__ == "__main__":
    main()
