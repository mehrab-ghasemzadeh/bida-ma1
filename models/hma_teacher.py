"""Hull Moving Average teacher (Step 16 of the specification).

The teacher is a frozen copy of a student branch whose weights are a temporally
smoothed version of the student's weight trajectory. With a window n:

    HMA(n) = WMA( 2 * WMA(n/2) - WMA(n), sqrt(n) )

which is implemented literally: a rolling history of the last n student parameter
snapshots gives WMA(n/2) and WMA(n); their combination

    raw_t = 2 * WMA(n/2) - WMA(n)

is pushed into a second rolling history of length round(sqrt(n)), and the teacher
parameters are the WMA of that second history.

Compared with an EMA the HMA reacts faster to the student's trend while still
suppressing step-to-step noise, at the cost of holding n parameter snapshots.
`EMA`, `frozen` and `copy` modes are provided for the ablation in section 29 (Q6).
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

TeacherMode = str  # "hma" | "ema" | "frozen" | "copy"


def smoothable_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Floating point parameters and buffers, i.e. the state that can be averaged.

    Integer buffers (e.g. BatchNorm `num_batches_tracked`) are excluded here and
    copied verbatim instead.
    """
    state: Dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        state[name] = param.detach()
    for name, buf in module.named_buffers():
        if buf is not None and buf.is_floating_point():
            state[name] = buf.detach()
    return state


def _weighted_moving_average(
    history: Iterable[Dict[str, torch.Tensor]], window: int, keys: List[str]
) -> Dict[str, torch.Tensor]:
    """Linearly weighted moving average over the most recent `window` snapshots.

    Weight i (1-based, oldest first within the window) is i, so the newest snapshot
    carries the largest weight; weights are normalised to sum to one. If fewer than
    `window` snapshots exist, the available ones are used.
    """
    snapshots = list(history)[-window:]
    if not snapshots:
        raise ValueError("Cannot compute a moving average over an empty history")

    length = len(snapshots)
    weights = [float(i + 1) for i in range(length)]
    total = sum(weights)

    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        acc = snapshots[0][key].to(torch.float32) * (weights[0] / total)
        for idx in range(1, length):
            acc = acc + snapshots[idx][key].to(torch.float32) * (weights[idx] / total)
        out[key] = acc
    return out


class TeacherBranch(nn.Module):
    """A gradient-free copy of `student` whose weights are a smoothed version of it.

    Args:
        student: the online module to mirror (its architecture is deep-copied).
        mode: "hma", "ema", "frozen" or "copy".
        window: HMA window n.
        ema_decay: decay for the EMA mode.
        update_every: apply an update every k optimiser steps (the history is only
            appended on those steps, so the effective HMA window spans k*n steps).
        history_device: where to keep the parameter history ("same" or "cpu"). CPU
            storage saves GPU memory at the cost of host/device transfers.
    """

    def __init__(
        self,
        student: nn.Module,
        mode: TeacherMode = "hma",
        window: int = 16,
        ema_decay: float = 0.999,
        update_every: int = 1,
        history_device: str = "same",
    ):
        super().__init__()
        if mode not in {"hma", "ema", "frozen", "copy"}:
            raise ValueError(f"Unknown teacher mode: {mode}")
        if window < 2:
            raise ValueError("The HMA window must be at least 2")

        self.mode = mode
        self.window = window
        self.half_window = max(int(round(window / 2)), 1)
        self.sqrt_window = max(int(round(math.sqrt(window))), 1)
        self.ema_decay = ema_decay
        self.update_every = max(int(update_every), 1)
        self.history_device = history_device

        self.module = copy.deepcopy(student)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

        self._keys: List[str] = sorted(smoothable_state(student).keys())
        self._param_history: Deque[Dict[str, torch.Tensor]] = deque(maxlen=window)
        self._raw_history: Deque[Dict[str, torch.Tensor]] = deque(maxlen=self.sqrt_window)
        self._step = 0

        # Teacher parameters start equal to the student parameters.
        self.sync(student)

    # ------------------------------------------------------------------ state
    def _snapshot(self, student: nn.Module) -> Dict[str, torch.Tensor]:
        state = smoothable_state(student)
        snapshot = {}
        for key in self._keys:
            tensor = state[key].detach().to(torch.float32)
            if self.history_device == "cpu":
                tensor = tensor.cpu()
            snapshot[key] = tensor.clone()
        return snapshot

    @torch.no_grad()
    def _write(self, values: Dict[str, torch.Tensor]) -> None:
        targets = dict(self.module.named_parameters())
        targets.update(dict(self.module.named_buffers()))
        for key, value in values.items():
            target = targets[key]
            target.copy_(value.to(device=target.device, dtype=target.dtype))

    @torch.no_grad()
    def sync(self, student: nn.Module) -> None:
        """Hard-copy the student weights into the teacher and reset the history."""
        self.module.load_state_dict(student.state_dict())
        for param in self.module.parameters():
            param.requires_grad_(False)
        self._param_history.clear()
        self._raw_history.clear()
        self._step = 0

    @torch.no_grad()
    def _copy_int_buffers(self, student: nn.Module) -> None:
        student_buffers = dict(student.named_buffers())
        for name, buf in self.module.named_buffers():
            src = student_buffers.get(name)
            if src is not None and not buf.is_floating_point():
                buf.copy_(src)

    # ----------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, student: nn.Module) -> bool:
        """Advance the teacher after an optimiser step. Returns True if it changed."""
        if self.mode == "frozen":
            return False

        self._step += 1
        if self._step % self.update_every != 0:
            return False

        if self.mode == "copy":
            self.module.load_state_dict(student.state_dict())
            return True

        if self.mode == "ema":
            student_state = smoothable_state(student)
            targets = dict(self.module.named_parameters())
            targets.update(dict(self.module.named_buffers()))
            for key in self._keys:
                target = targets[key]
                target.mul_(self.ema_decay).add_(
                    student_state[key].to(device=target.device, dtype=target.dtype),
                    alpha=1.0 - self.ema_decay,
                )
            self._copy_int_buffers(student)
            return True

        # --- HMA -------------------------------------------------------------
        self._param_history.append(self._snapshot(student))

        wma_half = _weighted_moving_average(self._param_history, self.half_window, self._keys)
        wma_full = _weighted_moving_average(self._param_history, self.window, self._keys)
        raw = {key: 2.0 * wma_half[key] - wma_full[key] for key in self._keys}

        self._raw_history.append(raw)
        smoothed = _weighted_moving_average(self._raw_history, self.sqrt_window, self._keys)

        self._write(smoothed)
        self._copy_int_buffers(student)
        return True

    # ---------------------------------------------------------------- forward
    def train(self, mode: bool = True):  # noqa: D102 - the teacher always stays in eval
        super().train(False)
        self.module.eval()
        return self

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def drift(self, student: nn.Module) -> float:
        """L2 distance between teacher and student parameter vectors.

        Used in Phase 9 of the implementation plan to check that the teacher evolves
        smoothly rather than tracking the student exactly or lagging indefinitely.
        """
        student_state = smoothable_state(student)
        teacher_state = smoothable_state(self.module)
        total = 0.0
        for key in self._keys:
            diff = teacher_state[key].to(torch.float32) - student_state[key].to(
                device=teacher_state[key].device, dtype=torch.float32
            )
            total += float(torch.sum(diff * diff))
        return math.sqrt(total)

    def state_dict_extra(self) -> Dict[str, Optional[int]]:
        return {"mode": self.mode, "window": self.window, "step": self._step}
