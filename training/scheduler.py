"""Loss-weight schedules and learning-rate schedules.

The distillation warm-up of Step 17 is

    lambda_dist(e) = lambda_max * min(1, e / E_warmup)

evaluated per epoch by default, or per optimiser step (`granularity="step"`) for
a smoother ramp on long epochs.
"""

from __future__ import annotations

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR, _LRScheduler


class WarmupWeight:
    """Linearly ramps a loss weight from 0 to `max_weight` over `warmup` units.

    Args:
        max_weight: lambda_max.
        warmup: warm-up length in epochs (or steps, see `granularity`). 0 disables
            the ramp and the weight is `max_weight` from the start.
        delay: units to wait at weight 0 before the ramp begins.
        granularity: "epoch" or "step".
    """

    def __init__(
        self,
        max_weight: float,
        warmup: int = 10,
        delay: int = 0,
        granularity: str = "epoch",
    ):
        if granularity not in {"epoch", "step"}:
            raise ValueError(f"Unknown granularity: {granularity}")
        self.max_weight = float(max_weight)
        self.warmup = int(warmup)
        self.delay = int(delay)
        self.granularity = granularity

    def at(self, epoch: int, step: Optional[int] = None) -> float:
        unit = epoch if self.granularity == "epoch" else (step if step is not None else 0)
        progress = unit - self.delay
        if progress <= 0:
            return 0.0 if self.warmup > 0 or self.delay > 0 else self.max_weight
        if self.warmup <= 0:
            return self.max_weight
        return self.max_weight * min(1.0, progress / self.warmup)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"WarmupWeight(max={self.max_weight}, warmup={self.warmup}, "
            f"delay={self.delay}, per={self.granularity})"
        )


def build_lr_scheduler(
    optimizer: Optimizer,
    kind: str = "cosine",
    epochs: int = 100,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.01,
    step_size: int = 30,
    gamma: float = 0.5,
) -> Optional[_LRScheduler]:
    """Epoch-level learning-rate schedule."""
    kind = (kind or "none").lower()
    if kind == "none":
        return None

    if kind == "cosine":
        if warmup_epochs > 0:
            def factor(epoch: int) -> float:
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            return LambdaLR(optimizer, factor)
        base_lr = optimizer.param_groups[0]["lr"]
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * min_lr_ratio)

    if kind == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unknown lr scheduler: {kind}")
