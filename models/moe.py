"""Mixture-of-Experts layer with Top-k routing (Steps 6-8 of the specification).

The standard Transformer FFN is replaced by K small expert MLPs. A linear router
produces a probability over experts; the top-k are selected and their outputs are
combined with renormalised gate values:

    MoE(x) = g_i_hat * E_i(x) + g_j_hat * E_j(x),   g_i_hat = g_i / (g_i + g_j)

Every forward pass records a load-balancing auxiliary loss and hard expert-usage
statistics; `collect_moe_aux` / `moe_usage` gather them from a whole model.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert: E_k(x) = W_k2 * sigma(W_k1 x + b_k1) + b_k2."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1,
                 negative_slope: float = 0.01):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class MoELayer(nn.Module):
    """Sparse Top-k mixture of experts.

    Args:
        dim: token embedding dimension D.
        num_experts: K.
        top_k: number of experts activated per token (2 in the main configuration).
        hidden_mult: expert hidden dimension = hidden_mult * D.
        balance_mode: how p_k in L_balance is estimated.
            "soft"   - p_k = mean router probability (differentiable, the default).
            "hard"   - p_k = fraction of token slots dispatched to expert k. This is
                       the literal reading of the specification, but hard counts carry
                       no gradient, so it only makes sense for logging / ablation.
            "switch" - standard sparse-MoE auxiliary loss K * sum_k f_k * P_k, the
                       "later replacement" mentioned in section 12 of the spec.
        noise_std: optional Gaussian noise on router logits during training, which
            helps break ties early in training.
    """

    MAX_USAGE_RECORDS = 4096   # cap on the per-step usage log, to bound memory

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: float = 2.0,
        dropout: float = 0.1,
        balance_mode: str = "soft",
        noise_std: float = 0.0,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts]; got {top_k}/{num_experts}")
        if balance_mode not in {"soft", "hard", "switch"}:
            raise ValueError(f"Unknown balance_mode: {balance_mode}")

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_mode = balance_mode
        self.noise_std = noise_std

        hidden_dim = int(round(hidden_mult * dim))
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim, dropout, negative_slope) for _ in range(num_experts)]
        )

        # Per-step recordings, cleared by reset_stats() at the start of each step.
        self._aux_losses: List[torch.Tensor] = []
        self._usage: List[torch.Tensor] = []

    # ------------------------------------------------------------------ stats
    def reset_stats(self) -> None:
        self._aux_losses = []
        self._usage = []

    @property
    def aux_losses(self) -> List[torch.Tensor]:
        return self._aux_losses

    @property
    def usage(self) -> List[torch.Tensor]:
        return self._usage

    def _balance_loss(self, probs: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """probs: [T, K] router probabilities; top_idx: [T, k] selected experts."""
        num_slots = top_idx.numel()
        counts = torch.zeros(self.num_experts, device=probs.device, dtype=probs.dtype)
        counts.scatter_add_(
            0,
            top_idx.reshape(-1),
            torch.ones(num_slots, device=probs.device, dtype=probs.dtype),
        )
        hard_frac = counts / max(num_slots, 1)

        if self.balance_mode == "hard":
            p_k = hard_frac
        elif self.balance_mode == "switch":
            mean_prob = probs.mean(dim=0)
            return self.num_experts * torch.sum(hard_frac.detach() * mean_prob)
        else:  # soft
            p_k = probs.mean(dim=0)

        target = 1.0 / self.num_experts
        return torch.sum((p_k - target) ** 2)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., D] -> [..., D]."""
        in_shape = x.shape
        flat = x.reshape(-1, self.dim)                                  # [T, D]

        logits = self.router(flat)                                      # [T, K]
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = F.softmax(logits, dim=-1)

        top_val, top_idx = probs.topk(self.top_k, dim=-1)               # [T, k]
        gates = top_val / (top_val.sum(dim=-1, keepdim=True) + 1e-9)    # renormalised

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            slot_mask = top_idx == expert_id                            # [T, k]
            token_ids, slot_ids = slot_mask.nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            expert_out = expert(flat[token_ids])
            weight = gates[token_ids, slot_ids].unsqueeze(-1).to(expert_out.dtype)
            out.index_add_(0, token_ids, expert_out * weight)

        # The auxiliary loss is only meaningful where it can be optimised; usage is
        # recorded in evaluation too, for the expert-specialisation analysis.
        if self.training and torch.is_grad_enabled():
            self._aux_losses.append(self._balance_loss(probs, top_idx))
        if len(self._usage) < self.MAX_USAGE_RECORDS:
            with torch.no_grad():
                counts = torch.zeros(self.num_experts, device=flat.device, dtype=torch.float32)
                counts.scatter_add_(
                    0,
                    top_idx.reshape(-1),
                    torch.ones(top_idx.numel(), device=flat.device),
                )
                self._usage.append(counts / max(top_idx.numel(), 1))

        return out.reshape(in_shape)


class FFN(nn.Module):
    """Plain Transformer feed-forward network, the no-MoE ablation baseline."""

    def __init__(self, dim: int, hidden_mult: float = 2.0, dropout: float = 0.1,
                 negative_slope: float = 0.01):
        super().__init__()
        hidden_dim = int(round(hidden_mult * dim))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_ffn(dim: int, cfg) -> nn.Module:
    """Build either an MoE layer or a plain FFN, according to cfg.use_moe."""
    if not cfg.get("use_moe", True):
        return FFN(
            dim,
            hidden_mult=cfg.get("hidden_mult", 2.0),
            dropout=cfg.get("dropout", 0.1),
            negative_slope=cfg.get("negative_slope", 0.01),
        )
    return MoELayer(
        dim,
        num_experts=cfg.get("num_experts", 4),
        top_k=cfg.get("top_k", 2),
        hidden_mult=cfg.get("hidden_mult", 2.0),
        dropout=cfg.get("dropout", 0.1),
        balance_mode=cfg.get("balance_mode", "soft"),
        noise_std=cfg.get("noise_std", 0.0),
        negative_slope=cfg.get("negative_slope", 0.01),
    )


# --------------------------------------------------------------- model-level
def iter_moe_layers(module: nn.Module) -> Iterable[MoELayer]:
    for sub in module.modules():
        if isinstance(sub, MoELayer):
            yield sub


def reset_moe_stats(module: nn.Module) -> None:
    for layer in iter_moe_layers(module):
        layer.reset_stats()


def collect_moe_aux(module: nn.Module, device: Optional[torch.device] = None) -> torch.Tensor:
    """Mean auxiliary balancing loss over every MoE call recorded since the last reset."""
    losses = [loss for layer in iter_moe_layers(module) for loss in layer.aux_losses]
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def moe_usage(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Per-MoE-layer expert usage fractions, for the expert-specialisation study (Q3)."""
    usage: Dict[str, torch.Tensor] = {}
    for name, sub in module.named_modules():
        if isinstance(sub, MoELayer) and sub.usage:
            usage[name] = torch.stack(sub.usage).mean(dim=0)
    return usage
