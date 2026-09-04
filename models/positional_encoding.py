"""Positional encoding for the semantic tokens (Step 3 of the specification).

Z_tilde = Z + P with P in R^{num_tokens x D}, a learnable parameter.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """P = nn.Parameter(torch.randn(1, num_tokens, dim)), added to the tokens."""

    def __init__(self, num_tokens: int = 5, dim: int = 64, std: float = 0.02,
                 dropout: float = 0.0):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * std)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.pos.shape[1]:
            raise ValueError(
                f"Expected {self.pos.shape[1]} tokens, got {tokens.shape[1]}"
            )
        return self.dropout(tokens + self.pos)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal alternative, kept for ablation."""

    def __init__(self, num_tokens: int = 5, dim: int = 64, dropout: float = 0.0):
        super().__init__()
        pos = torch.arange(num_tokens).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        table = torch.zeros(1, num_tokens, dim)
        table[0, :, 0::2] = torch.sin(pos * div)
        table[0, :, 1::2] = torch.cos(pos * div)[:, : table[0, :, 1::2].shape[1]]
        self.register_buffer("pos", table)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(tokens + self.pos)


def build_positional_encoding(num_tokens: int, dim: int, kind: str = "learned",
                              dropout: float = 0.0) -> nn.Module:
    if kind == "learned":
        return LearnedPositionalEncoding(num_tokens, dim, dropout=dropout)
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(num_tokens, dim, dropout=dropout)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown positional encoding: {kind}")
