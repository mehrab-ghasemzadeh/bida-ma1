"""Labelled source-domain dataset.

The source domain provides the only labels used during training. Every labelled
pixel is a sample by default; `samples_per_class` reproduces the "N samples per
class" protocol common in the cross-scene literature.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .preprocessing import (
    HSIPatchDataset,
    labelled_coordinates,
    pad_cube,
    sample_per_class,
)


class SourceDataset(HSIPatchDataset):
    """Patches around labelled source pixels, with their class labels."""

    def __init__(
        self,
        cube: np.ndarray,
        gt: np.ndarray,
        patch_size: int = 13,
        samples_per_class: Optional[int] = None,
        augment: bool = False,
        pad_mode: str = "reflect",
        seed: int = 0,
    ):
        coords, labels = labelled_coordinates(gt)
        if samples_per_class is not None:
            coords, labels = sample_per_class(coords, labels, samples_per_class, seed=seed)
        super().__init__(
            cube=pad_cube(cube, patch_size, pad_mode),
            coords=coords,
            labels=labels,
            patch_size=patch_size,
            already_padded=True,
            augment=augment,
            seed=seed,
        )


def split_train_val(
    dataset: SourceDataset, val_fraction: float = 0.0, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified index split of a source dataset, for optional source validation."""
    if val_fraction <= 0:
        return np.arange(len(dataset)), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(dataset.labels):
        idx = np.flatnonzero(dataset.labels == cls)
        rng.shuffle(idx)
        cut = max(int(round(len(idx) * val_fraction)), 1)
        val_idx.append(idx[:cut])
        train_idx.append(idx[cut:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))
