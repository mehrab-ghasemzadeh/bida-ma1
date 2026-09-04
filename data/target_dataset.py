"""Unlabelled target-domain dataset (training) and labelled target set (evaluation).

During training the target labels are never read: the trainer only consumes the
patches. The same coordinates are reused at evaluation time, where the labels are
compared against the predictions.

`pixels="labelled"` restricts the target training pool to pixels that carry a
ground-truth label (the usual transductive protocol of the cross-scene
benchmarks); `pixels="all"` uses every pixel of the scene, which is a strictly
unsupervised - and much larger - target pool.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .preprocessing import (
    HSIPatchDataset,
    all_coordinates,
    labelled_coordinates,
    pad_cube,
)


class TargetDataset(HSIPatchDataset):
    """Target patches. Labels are carried for evaluation but never used in training."""

    def __init__(
        self,
        cube: np.ndarray,
        gt: np.ndarray,
        patch_size: int = 13,
        pixels: str = "labelled",
        max_samples: Optional[int] = None,
        augment: bool = False,
        pad_mode: str = "reflect",
        seed: int = 0,
    ):
        if pixels == "labelled":
            coords, labels = labelled_coordinates(gt)
        elif pixels == "all":
            coords, labels = all_coordinates(gt)
        else:
            raise ValueError(f"Unknown pixel selection: {pixels}")

        if max_samples is not None and len(coords) > max_samples:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(coords), max_samples, replace=False))
            coords, labels = coords[keep], labels[keep]

        super().__init__(
            cube=pad_cube(cube, patch_size, pad_mode),
            coords=coords,
            labels=labels,
            patch_size=patch_size,
            already_padded=True,
            augment=augment,
            seed=seed,
        )


class InfiniteLoader:
    """Endless iterator over a DataLoader, for pairing target with source batches.

    The source loader defines the length of an epoch; the target loader is cycled
    so that every source batch gets a target batch regardless of the two dataset
    sizes.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iterator: Optional[Iterator] = None

    def __iter__(self):
        return self

    def next(self):
        if self._iterator is None:
            self._iterator = iter(self.loader)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)

    __next__ = next


def target_full_scene_dataset(
    cube: np.ndarray, gt: np.ndarray, patch_size: int = 13, pad_mode: str = "reflect"
) -> TargetDataset:
    """Every pixel of the target scene, for producing a full classification map."""
    return TargetDataset(cube, gt, patch_size=patch_size, pixels="all", pad_mode=pad_mode)
