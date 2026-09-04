"""Cross-domain attention block (Steps 12-13 of the specification).

One direction of the coupled Transformer. The query stream attends to the other
domain's key/value stream, the residual connection keeps the query stream, and an
MoE sub-layer follows:

    A       = Attention(Q_query, K_context, V_context)
    C_prime = LN(H_query + A)
    C       = LN(C_prime + MoE(C_prime))
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .moe import build_ffn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_cfg: dict | None = None,
        norm_style: str = "post",
        store_attention: bool = False,
    ):
        super().__init__()
        if norm_style not in {"post", "pre"}:
            raise ValueError(f"Unknown norm_style: {norm_style}")
        self.norm_style = norm_style
        self.store_attention = store_attention

        # Separate Q / K / V projections per direction are exactly W_Q^s, W_K^t, W_V^t.
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        if norm_style == "pre":
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = build_ffn(dim, ffn_cfg or {})
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """query/context: [B, N, D] -> [B, N, D] (shape of `query`)."""
        need_weights = self.store_attention
        if self.norm_style == "post":
            attn_out, weights = self.attn(
                query, context, context, need_weights=need_weights,
                average_attn_weights=True,
            )
            out = self.norm1(query + self.drop(attn_out))
            out = self.norm2(out + self.ffn(out))
        else:
            q_n, kv_n = self.norm_q(query), self.norm_kv(context)
            attn_out, weights = self.attn(
                q_n, kv_n, kv_n, need_weights=need_weights, average_attn_weights=True,
            )
            out = query + self.drop(attn_out)
            out = out + self.ffn(self.norm2(out))

        if need_weights and weights is not None:
            self.last_attention = weights.detach()
        return out


class BidirectionalCrossAttentionLayer(nn.Module):
    """One layer of bidirectional coupling: source queries target and target queries
    source, both computed from the same incoming pair of streams."""

    def __init__(self, dim: int, **block_kwargs):
        super().__init__()
        self.s_from_t = CrossAttentionBlock(dim, **block_kwargs)
        self.t_from_s = CrossAttentionBlock(dim, **block_kwargs)

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c_s = self.s_from_t(h_s, h_t)   # source queries target
        c_t = self.t_from_s(h_t, h_s)   # target queries source
        return c_s, c_t
