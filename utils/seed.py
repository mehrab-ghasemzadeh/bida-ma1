"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch. `deterministic` trades speed for exact reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give every dataloader worker a distinct but reproducible numpy seed."""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
