"""Stack of Transformer blocks used as the per-domain encoder (Steps 4-5).

One instance is created per domain; by default source and target encoders keep
independent parameters (section 9 of the specification).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_cfg: dict | None = None,
        norm_style: str = "post",
        final_norm: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    ffn_cfg=ffn_cfg,
                    norm_style=norm_style,
                )
                for _ in range(depth)
            ]
        )
        # A post-norm stack already ends in a LayerNorm; a pre-norm stack needs one.
        self.norm = nn.LayerNorm(dim) if (final_norm or norm_style == "pre") else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] -> [B, N, D]."""
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def pool_tokens(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Reduce [B, N, D] token sequences to a single [B, D] patch representation."""
    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "max":
        return tokens.max(dim=1).values
    if mode == "cls":
        return tokens[:, 0]
    raise ValueError(f"Unknown pooling mode: {mode}")
