"""
Precompute the "tokenizer compression" map  P : V -> V'  (Section 2.2 of the paper).

Many o200k tokens are semantically equivalent but have distinct ids
("Apple" vs " apple" vs "apple"). Collapsing them before hashing increases the
semantic density of the N-gram tables. We map each raw token id to a canonical
representative id by normalizing its text (NFKC + lowercase + strip).

Run ONCE (takes ~1-2 min, decodes the whole 201,088 vocab):
    python build_compression.py

Writes data/canonical_ids.npy  (int64 array of length vocab_size).
model.py loads it automatically if present; otherwise it falls back to identity.
"""

import os
import unicodedata

import numpy as np

from tokenizer import VOCAB_SIZE, get_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
OUT = os.path.join(DATA_DIR, "canonical_ids.npy")


def normalize(b: bytes) -> str:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return ""                      # partial-byte token: leave it unique
    s = unicodedata.normalize("NFKC", s)
    return s.strip().lower()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    enc = get_tokenizer()
    canonical = np.arange(VOCAB_SIZE, dtype=np.int64)
    rep = {}                           # normalized text -> representative id
    collapsed = 0
    for tid in range(VOCAB_SIZE):
        try:
            b = enc.decode_single_token_bytes(tid)
        except Exception:
            continue                   # special/reserved token -> keep itself
        key = normalize(b)
        if not key:
            continue
        if key in rep:
            canonical[tid] = rep[key]
            collapsed += 1
        else:
            rep[key] = tid
        if (tid + 1) % 40000 == 0:
            print(f"  {tid + 1:,}/{VOCAB_SIZE:,}  collapsed so far: {collapsed:,}")

    np.save(OUT, canonical)
    uniq = len(np.unique(canonical))
    print(f"done. collapsed {collapsed:,} ids -> {uniq:,} canonical "
          f"({100 * (1 - uniq / VOCAB_SIZE):.1f}% reduction)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
