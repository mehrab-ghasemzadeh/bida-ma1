"""Source classification loss (Step 10 of the specification).

Cross entropy on the labelled source domain only. Optional label smoothing and
class weighting help with the heavily imbalanced class counts typical of
hyperspectral scenes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        return nn.functional.cross_entropy(
            logits, targets, weight=weight, label_smoothing=self.label_smoothing
        )
