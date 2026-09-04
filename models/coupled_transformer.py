"""Bidirectional coupled Transformer (Step 11 of the specification).

Takes the two independent encoder outputs H_s and H_t and exchanges information
between the domains in both directions simultaneously:

    C_{s<-t} : source tokens query target tokens
    C_{t<-s} : target tokens query source tokens

With depth > 1 the two streams keep alternating; at every layer both directions
read the previous layer's output of the *other* stream, so the coupling stays
symmetric and neither direction sees an already-updated partner.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .coupled_attention import BidirectionalCrossAttentionLayer


class CoupledTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_cfg: dict | None = None,
        norm_style: str = "post",
        store_attention: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BidirectionalCrossAttentionLayer(
                    dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    ffn_cfg=ffn_cfg,
                    norm_style=norm_style,
                    store_attention=store_attention,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """h_s, h_t: [B, N, D] -> (C_{s<-t}, C_{t<-s}), both [B, N, D]."""
        c_s, c_t = h_s, h_t
        for layer in self.layers:
            c_s, c_t = layer(c_s, c_t)
        return c_s, c_t
