"""Bidirectional teacher-student representation distillation (Steps 14-15).

    L_{s<-t} = D( sg(h_s_teacher), h_{s<-t} )
    L_{t<-s} = D( sg(h_t_teacher), h_{t<-s} )
    L_dist   = L_{s<-t} + L_{t<-s}

with D the cosine distance by default. The teacher side is always detached, so no
gradient reaches the teacher through this loss.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distance(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """1 - cos(a, b), averaged over the batch."""
    return (1.0 - F.cosine_similarity(teacher, student, dim=-1)).mean()


def mse_distance(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student, teacher)


def smooth_l1_distance(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(student, teacher)


def normalised_mse(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """MSE between L2-normalised vectors (equivalent to cosine distance up to 2x)."""
    return F.mse_loss(F.normalize(student, dim=-1), F.normalize(teacher, dim=-1))


_DISTANCES = {
    "cosine": cosine_distance,
    "mse": mse_distance,
    "smooth_l1": smooth_l1_distance,
    "normalised_mse": normalised_mse,
}


class DistillationLoss(nn.Module):
    """Sum of the two directional distillation terms.

    Args:
        distance: cosine (default) | mse | smooth_l1 | normalised_mse.
        symmetric: also pull the teacher-side representation towards the student
            with the roles swapped and detached the other way round. Off by default,
            which matches the specification.
    """

    def __init__(self, distance: str = "cosine", symmetric: bool = False):
        super().__init__()
        if distance not in _DISTANCES:
            raise ValueError(f"Unknown distillation distance: {distance}")
        self.distance_name = distance
        self.distance = _DISTANCES[distance]
        self.symmetric = symmetric

    def _directional(self, teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
        loss = self.distance(teacher.detach(), student)
        if self.symmetric:
            loss = 0.5 * (loss + self.distance(student.detach(), teacher))
        return loss

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expects h_s_teacher / h_s_from_t and h_t_teacher / h_t_from_s in `outputs`."""
        device = next(iter(outputs.values())).device
        zero = torch.zeros((), device=device)

        loss_s = zero
        if "h_s_teacher" in outputs and "h_s_from_t" in outputs:
            loss_s = self._directional(outputs["h_s_teacher"], outputs["h_s_from_t"])

        loss_t = zero
        if "h_t_teacher" in outputs and "h_t_from_s" in outputs:
            loss_t = self._directional(outputs["h_t_teacher"], outputs["h_t_from_s"])

        return {"dist_s_from_t": loss_s, "dist_t_from_s": loss_t, "dist": loss_s + loss_t}
