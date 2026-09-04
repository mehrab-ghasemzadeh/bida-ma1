"""MoE load-balancing loss (Step 8 of the specification).

    L_balance = sum_k (p_k - 1/K)^2

The per-layer terms are produced inside `models.moe.MoELayer` during the forward
pass (so the router probabilities are available where they are computed); this
module just gathers them from the model and averages over the layers.

Note on p_k: the specification defines it as the fraction of tokens assigned to
expert k. Hard assignment counts are piecewise constant in the router weights and
therefore carry no gradient, so the default `balance_mode="soft"` uses the mean
router probability instead - the usual differentiable surrogate. Set
`model.moe.balance_mode` to "hard" or "switch" to change that.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.moe import collect_moe_aux, moe_usage


class MoEBalanceLoss(nn.Module):
    def forward(self, model: nn.Module,
                device: Optional[torch.device] = None) -> torch.Tensor:
        return collect_moe_aux(model, device=device)

    @staticmethod
    def usage(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Per-layer expert usage fractions, for the expert-specialisation study."""
        return moe_usage(model)
