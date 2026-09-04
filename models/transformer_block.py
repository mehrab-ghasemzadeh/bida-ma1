"""Transformer block with an MoE (or plain FFN) sub-layer (Step 4 of the spec).

Post-norm formulation, exactly as specified:

    H'    = LN(H  + MHSA(H))
    H_out = LN(H' + MoE(H'))

A pre-norm variant is available for ablation, since post-norm can be harder to
train at greater depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .moe import build_ffn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_cfg: dict | None = None,
        norm_style: str = "post",
    ):
        super().__init__()
        if norm_style not in {"post", "pre"}:
            raise ValueError(f"Unknown norm_style: {norm_style}")
        self.norm_style = norm_style

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = build_ffn(dim, ffn_cfg or {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] -> [B, N, D]."""
        if self.norm_style == "post":
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.norm1(x + self.drop(attn_out))
            x = self.norm2(x + self.ffn(x))
            return x

        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x
