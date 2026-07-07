"""
mini-gpt-oss + Engram : the ~250M gpt-oss model with a conditional-memory
(Engram) lookup module added at two early layers.

Everything below the Engram additions is identical to the baseline mini-gpt-oss
(same RMSNorm, RoPE, GQA + sinks + sliding window, MoE + SwiGLU), so the two
folders differ ONLY by the Engram module -> a clean MoE vs MoE+Engram comparison.

Engram additions:
  * ModelConfig gains engram_* fields
  * token ids are threaded through the blocks (Engram needs them to build N-grams)
  * an EngramBlock runs at the start of the blocks in `engram_layers` (default 1,3)
  * a precomputed vocab->canonical id map ("tokenizer compression") collapses
    equivalent tokens before hashing
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engram import EngramBlock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
CANON_PATH = os.path.join(DATA_DIR, "canonical_ids.npy")


@dataclass
class ModelConfig:
    # vocabulary / context
    vocab_size: int = 201088
    block_size: int = 512

    # transformer size
    num_hidden_layers: int = 6
    hidden_size: int = 768
    tie_embeddings: bool = True

    # attention (GQA)
    head_dim: int = 64
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    sliding_window: int = 128

    # Mixture of Experts
    num_experts: int = 8
    experts_per_token: int = 2
    intermediate_size: int = 768
    swiglu_limit: float = 7.0
    aux_loss_coef: float = 0.01

    # RoPE / YaRN
    rope_theta: float = 10000.0
    initial_context_length: int = 512
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    # ---- Engram (conditional memory) ----
    use_engram: bool = True
    engram_layers: tuple = (1, 3)          # inject after 1 attn round + mid (paper: layers 2 & 6 of 12)
    ngram_orders: tuple = (2, 3)           # {2,3}-grams (4-grams hurt under fixed budget)
    hash_heads: int = 2                    # K distinct hash heads per order
    engram_slot_dim: int = 128             # dim of each retrieved vector
    engram_table_size: int = 16381         # rows per table (prime)
    engram_conv_kernel: int = 4
    use_tokenizer_compression: bool = True


# --------------------------------------------------------------------------- #
#  RMSNorm                                                                     #
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        dtype = x.dtype
        t = x.float()
        t = t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


# --------------------------------------------------------------------------- #
#  Rotary embeddings (RoPE + YaRN path)                                        #
# --------------------------------------------------------------------------- #
def _apply_rotary_emb(x, cos, sin):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.base = config.rope_theta
        self.initial_context_length = config.initial_context_length
        self.scaling_factor = config.rope_scaling_factor
        self.ntk_alpha = config.rope_ntk_alpha
        self.ntk_beta = config.rope_ntk_beta

    def _concentration_and_inv_freq(self, device):
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device) / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
            assert 0 < low < high < d_half - 1
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (torch.arange(d_half, dtype=torch.float32, device=device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        return concentration, inv_freq

    def forward(self, q, k):
        T, device = q.shape[1], q.device
        concentration, inv_freq = self._concentration_and_inv_freq(device)
        t = torch.arange(T, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = (freqs.cos() * concentration).to(q.dtype)
        sin = (freqs.sin() * concentration).to(q.dtype)
        return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


# --------------------------------------------------------------------------- #
#  Attention: GQA + sliding window + sinks                                     #
# --------------------------------------------------------------------------- #
class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.nh = config.num_attention_heads
        self.nkv = config.num_key_value_heads
        self.hd = config.head_dim
        self.q_mult = self.nh // self.nkv
        assert self.nh % self.nkv == 0
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0

        self.norm = RMSNorm(config.hidden_size)
        qkv_dim = self.hd * (self.nh + 2 * self.nkv)
        self.qkv = nn.Linear(config.hidden_size, qkv_dim)
        self.out = nn.Linear(self.hd * self.nh, config.hidden_size)
        self.sinks = nn.Parameter(torch.zeros(self.nh))
        self.sm_scale = 1.0 / math.sqrt(self.hd)
        self.rope = RotaryEmbedding(config)

    def forward(self, x):
        B, T, D = x.shape
        t = self.norm(x)
        qkv = self.qkv(t)
        nh, nkv, hd = self.nh, self.nkv, self.hd
        q = qkv[..., : nh * hd].view(B, T, nh, hd)
        k = qkv[..., nh * hd : (nh + nkv) * hd].view(B, T, nkv, hd)
        v = qkv[..., (nh + nkv) * hd :].view(B, T, nkv, hd)
        q, k = self.rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.repeat_interleave(self.q_mult, dim=1)
        v = v.repeat_interleave(self.q_mult, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.sm_scale
        mask = torch.triu(x.new_full((T, T), float("-inf")), diagonal=1)
        if self.sliding_window > 0:
            mask = mask + torch.tril(x.new_full((T, T), float("-inf")), diagonal=-self.sliding_window)
        scores = scores + mask
        sink = self.sinks.view(1, nh, 1, 1).expand(B, nh, T, 1)
        scores = torch.cat([scores, sink], dim=-1)
        w = torch.softmax(scores, dim=-1)[..., :-1]
        out = torch.matmul(w, v)
        out = out.transpose(1, 2).reshape(B, T, nh * hd)
        return x + self.out(out)


# --------------------------------------------------------------------------- #
#  MoE MLP with clamped SwiGLU                                                 #
# --------------------------------------------------------------------------- #
def swiglu(x, alpha=1.702, limit=7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    return (x_glu * torch.sigmoid(alpha * x_glu)) * (x_linear + 1)


class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.norm = RMSNorm(config.hidden_size)
        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        self.experts = nn.ModuleList(
            nn.ModuleList([
                nn.Linear(config.hidden_size, config.intermediate_size * 2),
                nn.Linear(config.intermediate_size, config.hidden_size),
            ])
            for _ in range(config.num_experts)
        )

    def forward(self, x):
        B, T, D = x.shape
        t = self.norm(x).view(B * T, D)
        logits = self.gate(t)
        router_probs = torch.softmax(logits, dim=-1)
        top = torch.topk(logits, self.experts_per_token, dim=-1)
        weights = torch.softmax(top.values, dim=-1)
        indices = top.indices
        out = torch.zeros_like(t)
        for e in range(self.num_experts):
            hit = (indices == e).any(dim=-1)
            if not hit.any():
                continue
            sel = hit.nonzero(as_tuple=True)[0]
            pos = (indices[sel] == e).float().argmax(dim=-1)
            w_e = weights[sel, pos].unsqueeze(-1)
            up, down = self.experts[e]
            h = down(swiglu(up(t[sel]), limit=self.swiglu_limit))
            out[sel] += h * w_e
        one_hot = F.one_hot(indices, self.num_experts).sum(dim=1).float()
        f = one_hot.mean(dim=0)
        P = router_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f * P)
        return x + out.view(B, T, D), aux_loss


# --------------------------------------------------------------------------- #
#  Block (optionally with an Engram module in front) + full model             #
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = AttentionBlock(config, layer_idx)
        self.mlp = MLPBlock(config)
        use = config.use_engram and (layer_idx in tuple(config.engram_layers))
        self.engram = EngramBlock(config) if use else None

    def forward(self, x, c):
        if self.engram is not None:            # H <- H + Engram(H)  before attention
            x = self.engram(x, c)
        x = self.attn(x)
        x, aux = self.mlp(x)
        return x, aux


class MiniGPTOSS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.unembedding.weight = self.embedding.weight

        # vocab -> canonical id map (tokenizer compression). Loaded if present,
        # else identity (build it once with build_compression.py). Saved in ckpt.
        canon = torch.arange(config.vocab_size, dtype=torch.long)
        if config.use_engram and config.use_tokenizer_compression:
            if os.path.exists(CANON_PATH):
                canon = torch.from_numpy(np.load(CANON_PATH)).long()
            else:
                print(f"[engram] {CANON_PATH} not found -> using identity map "
                      f"(run build_compression.py to enable tokenizer compression)")
        self.register_buffer("canonical", canon, persistent=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        c = self.canonical[idx] if self.config.use_engram else idx
        x = self.embedding(idx)
        aux_total = idx.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            x, aux = block(x, c)
            aux_total = aux_total + aux
        x = self.norm(x)
        logits = self.unembedding(x)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = ce + (aux_total / len(self.blocks)) * self.config.aux_loss_coef
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eot_token=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eot_token is not None and (nxt == eot_token).all():
                break
        return idx

    def num_params(self, non_embedding=False, engram_only=False):
        if engram_only:
            return sum(p.numel() for n, p in self.named_parameters() if "engram" in n)
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embedding.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.unembedding.weight.numel()
        return n


if __name__ == "__main__":
    cfg = ModelConfig()
    model = MiniGPTOSS(cfg)
    print(f"total parameters     : {model.num_params() / 1e6:.1f}M  ({model.num_params():,})")
    print(f"  of which Engram     : {model.num_params(engram_only=True) / 1e6:.1f}M")
    print(f"non-embedding params : {model.num_params(non_embedding=True) / 1e6:.1f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(x, x)
    print(f"logits shape         : {tuple(logits.shape)}")
    print(f"loss (random init)   : {loss.item():.3f}")
