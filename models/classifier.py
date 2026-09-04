"""Classification head (Step 10 of the specification).

    Linear(D -> D/2) -> LeakyReLU -> Dropout -> Linear(D/2 -> num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        hidden_ratio: float = 0.5,
        dropout: float = 0.1,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        hidden = max(int(round(dim * hidden_ratio)), 1)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, D] -> logits [B, num_classes]."""
        return self.net(h)
