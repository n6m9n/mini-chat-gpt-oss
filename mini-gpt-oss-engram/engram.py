"""
Engram : conditional-memory module for mini-gpt-oss.

Implements the lookup module from "Conditional Memory via Scalable Lookup"
(DeepSeek-AI, arXiv:2601.07372), scaled down for our ~250M model.

Per token position t, at a chosen layer, the module:
  1. builds suffix N-grams from the (compressed) token ids: (x_{t-1},x_t), (x_{t-2..t})
  2. hashes each N-gram with K deterministic multiply-XOR heads into embedding
     tables (Eq. 1-2)  ->  a static memory vector e_t
  3. gates that static memory with the current hidden state h_t so contradictory
     lookups are suppressed (Eq. 3-4)
  4. refines it with a short depthwise causal conv (Eq. 5) and adds it back to the
     residual stream:  H <- H + Y

Design choices (from the paper's ablations, Sec. 6.2):
  * orders {2,3} only (4-grams hurt under a fixed budget)
  * context-aware gating + tokenizer compression are kept (top-3 components)
  * conv is zero-initialised so Y = V~ at start (identity of the conv branch)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dtype = x.dtype
        t = x.float()
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


class EngramBlock(nn.Module):
    # 61-bit masked FNV-style multiplicative-XOR hash (fits signed int64 safely)
    PRIME = 1099511628211
    MASK = (1 << 61) - 1

    def __init__(self, config):
        super().__init__()
        self.orders = list(config.ngram_orders)          # e.g. [2, 3]
        self.K = config.hash_heads                        # hash heads per order
        self.M = config.engram_table_size                 # rows per table (prime)
        self.slot = config.engram_slot_dim                # dim per retrieved vector
        D = config.hidden_size
        self.inv_sqrt_d = 1.0 / math.sqrt(D)

        n_tables = len(self.orders) * self.K
        self.tables = nn.ModuleList([nn.Embedding(self.M, self.slot) for _ in range(n_tables)])
        # distinct deterministic seed per (order, head)
        self.seeds = [(0x9E3779B97F4A7C15 * (i + 1)) & self.MASK for i in range(n_tables)]

        d_mem = n_tables * self.slot
        self.WK = nn.Linear(d_mem, D, bias=False)         # memory -> Key   (Eq. 3)
        self.WV = nn.Linear(d_mem, D, bias=False)         # memory -> Value (Eq. 3)
        self.rms_q = _RMSNorm(D)                          # RMSNorm(h_t)
        self.rms_k = _RMSNorm(D)                          # RMSNorm(k_t)
        self.rms_v = _RMSNorm(D)                          # before conv

        self.kernel = config.engram_conv_kernel
        self.dil = max(self.orders)
        self.pad = self.dil * (self.kernel - 1)
        self.conv = nn.Conv1d(D, D, self.kernel, groups=D, dilation=self.dil, bias=True)

        self._init()

    def _init(self):
        for t in self.tables:
            nn.init.normal_(t.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.WK.weight, std=0.02)
        nn.init.normal_(self.WV.weight, std=0.02)
        nn.init.zeros_(self.conv.weight)                  # conv branch = identity at init
        nn.init.zeros_(self.conv.bias)

    def _hash(self, comps, seed, device, shape):
        """multiplicative-XOR hash of the N-gram components -> row index in [0, M)."""
        h = torch.full(shape, seed, dtype=torch.long, device=device)
        for comp in comps:
            h = ((h ^ comp) * self.PRIME) & self.MASK
            h = (h ^ (h >> 29)) & self.MASK
        return h % self.M

    def forward(self, x, c):
        # x: [B, T, D] hidden states ; c: [B, T] compressed (canonical) token ids
        B, T, D = x.shape
        embs, ti = [], 0
        for n in self.orders:
            c_pad = F.pad(c, (n - 1, 0))                  # left-pad so suffix ends at t
            comps = [c_pad[:, j:j + T] for j in range(n)] # component j = token t-(n-1)+j
            for _ in range(self.K):
                idx = self._hash(comps, self.seeds[ti], x.device, (B, T))
                embs.append(self.tables[ti](idx))         # [B, T, slot]
                ti += 1
        e = torch.cat(embs, dim=-1)                       # [B, T, d_mem]

        kt, vt = self.WK(e), self.WV(e)                   # [B, T, D]
        alpha = torch.sigmoid((self.rms_q(x) * self.rms_k(kt)).sum(-1, keepdim=True) * self.inv_sqrt_d)
        vtil = alpha * vt                                 # gated memory (Eq. 4)

        z = self.rms_v(vtil).transpose(1, 2)              # [B, D, T]
        z = F.pad(z, (self.pad, 0))                       # causal
        z = self.conv(z).transpose(1, 2)                  # [B, T, D]
        y = F.silu(z) + vtil                              # Eq. 5
        return x + y                                      # residual integration
