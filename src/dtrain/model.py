import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LlamaConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    intermediate_size: int = 172
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def get_cos_sin(seq_length: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    angles = torch.arange(seq_length).float().unsqueeze(1) * theta
    return torch.cos(angles).repeat(1, 2), torch.sin(angles).repeat(1, 2)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
    v = v.repeat_interleave(q.size(1) // v.size(1), dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def init_linear(weight: torch.Tensor) -> None:
    bound = math.sqrt(1 / weight.size(1))
    nn.init.uniform_(weight, -bound, bound)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            init_linear(proj.weight)
        self.attend = causal_attention

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos[:seq], sin[:seq])
        k = apply_rotary_pos_emb(k, cos[:seq], sin[:seq])
        out = self.attend(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(batch, seq, -1))


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        for proj in (self.up_proj, self.gate_proj, self.down_proj):
            init_linear(proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig, recompute: bool = False):
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        self.config = config
        self.recompute = recompute
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.final_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        init_linear(self.final_proj.weight)
        cos, sin = get_cos_sin(config.max_position_embeddings, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            if self.recompute and self.training:
                x = checkpoint(layer, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = layer(x, self.cos, self.sin)
        return self.final_proj(self.final_norm(x))
