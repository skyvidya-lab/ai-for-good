"""
ChaosAttention — a lightweight physics-informed attention variant.

Adapted from `dynamis-finance-ai/backend/core/pi_attention.py` (V4.2) for the
geospatial/temporal crop classification problem.

Mechanism:
    Attention = Softmax( (Q K^T / sqrt(d)) + PhysicsBias ) V

Where PhysicsBias is driven by per-sample physics state:
    - chaos_score: mean innovation magnitude (0..1)
    - hurst: temporal Hurst exponent (0..1)

The adapter maps these 2 scalars to per-head bias terms. Small MLP, cheap.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChaosAttention(nn.Module):
    """
    Multi-head self-attention modulated by a per-sample physics vector.

    Args:
        embed_dim: feature dimension of inputs (must be divisible by num_heads).
        num_heads: number of attention heads.
        n_physics: size of the physics-state vector (default 2: chaos + hurst).
        dropout: attention dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        n_physics: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Physics adapter: (B, n_physics) -> (B, num_heads) bias per head
        self.physics_adapter = nn.Sequential(
            nn.Linear(n_physics, 16),
            nn.GELU(),
            nn.Linear(16, num_heads),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        physics_state: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)
            physics_state: (B, n_physics) — chaos_score, hurst (possibly more).
            attn_mask: optional (B, T) boolean mask (True = keep, False = ignore).

        Returns:
            (B, T, embed_dim)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Physics bias — broadcast across T x T
        phys_bias = self.physics_adapter(physics_state)  # (B, H)
        phys_bias = phys_bias.view(B, self.num_heads, 1, 1)
        scores = scores + phys_bias

        if attn_mask is not None:
            mask_bool = attn_mask.to(dtype=torch.bool)
            key_mask = mask_bool.view(B, 1, 1, T)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out)


__all__ = ["ChaosAttention"]
