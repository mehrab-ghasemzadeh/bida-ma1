"""Builds the source/target datasets and dataloaders from a config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.seed import worker_init_fn

from .preprocessing import ScenePair, prepare_scene_pair
from .registry import resolve_dataset
from .source_dataset import SourceDataset, split_train_val
from .target_dataset import InfiniteLoader, TargetDataset


@dataclass
class DataBundle:
    scene: ScenePair
    source_train: Any
    source_val: Optional[Any]
    target_train: TargetDataset
    target_eval: TargetDataset
    num_classes: int
    class_names: List[str]

    def describe(self) -> str:
        lines = [
            f"scene: {self.scene.summary()}",
            f"source train patches: {len(self.source_train)}",
            f"target train patches: {len(self.target_train)} (labels unused)",
            f"target eval patches:  {len(self.target_eval)}",
        ]
        if self.source_val is not None:
            lines.insert(2, f"source val patches:   {len(self.source_val)}")
        return "\n".join(lines)


def synthetic_scene_pair(
    num_bands: int = 13,
    num_classes: int = 7,
    size: int = 96,
    shift: float = 0.6,
    seed: int = 0,
) -> ScenePair:
    """Two small random scenes with a shared class structure and a domain shift.

    Only for shape checks and smoke tests - it is not a substitute for real data.
    """
    rng = np.random.default_rng(seed)
    signatures = rng.normal(size=(num_classes, num_bands)).astype(np.float32) * 2.0

    def make(scene_seed: int, gain: np.ndarray, offset: np.ndarray) -> tuple:
        local = np.random.default_rng(scene_seed)
        gt = local.integers(1, num_classes + 1, size=(size, size)).astype(np.int64)
        # Smooth the label map so patches contain spatially coherent regions.
        block = 8
        coarse = local.integers(1, num_classes + 1, size=(size // block + 1, size // block + 1))
        gt = np.kron(coarse, np.ones((block, block), dtype=np.int64))[:size, :size]
        cube = signatures[gt - 1] * gain + offset
        cube = cube + local.normal(scale=0.3, size=cube.shape).astype(np.float32)
        return cube.astype(np.float32), gt

    gain_s = np.ones(num_bands, dtype=np.float32)
    gain_t = (1.0 + shift * rng.normal(size=num_bands)).astype(np.float32)
    offset_t = (shift * rng.normal(size=num_bands)).astype(np.float32)

    src_cube, src_gt = make(seed + 1, gain_s, np.zeros(num_bands, dtype=np.float32))
    tgt_cube, tgt_gt = make(seed + 2, gain_t, offset_t)

    return ScenePair(
        source_cube=src_cube,
        source_gt=src_gt,
        target_cube=tgt_cube,
        target_gt=tgt_gt,
        num_classes=num_classes,
        class_names=[f"class {i + 1}" for i in range(num_classes)],
    )


def build_scene(cfg: Dict[str, Any]) -> ScenePair:
    data_cfg = cfg.get("dataset", {})
    num_bands = int(cfg.get("model", {}).get("num_bands", data_cfg.get("num_bands", 13)))

    if data_cfg.get("synthetic", False):
        return synthetic_scene_pair(
            num_bands=num_bands,
            num_classes=int(data_cfg.get("num_classes", 7) or 7),
            size=int(data_cfg.get("synthetic_size", 96)),
            seed=int(cfg.get("seed", 0)),
        )

    resolved = resolve_dataset(data_cfg, data_cfg.get("root", "data/raw"))
    paths = resolved["paths"]
    scene = prepare_scene_pair(
        source_image=paths["source_image"],
        source_gt=paths["source_gt"],
        target_image=paths["target_image"],
        target_gt=paths["target_gt"],
        num_bands=num_bands,
        normalisation=data_cfg.get("normalisation", "standard"),
        spectral_reduction=data_cfg.get("spectral_reduction", "pca"),
        keys=resolved["keys"],
        class_names=resolved["class_names"],
        seed=int(cfg.get("seed", 0)),
    )
    if resolved.get("num_classes"):
        expected = int(resolved["num_classes"])
        if scene.num_classes != expected:
            raise ValueError(
                f"Ground truth has {scene.num_classes} classes but the '{resolved['name']}' "
                f"benchmark expects {expected}. Check that the correct gt files are used."
            )
    return scene


def build_data(cfg: Dict[str, Any]) -> DataBundle:
    data_cfg = cfg.get("dataset", {})
    seed = int(cfg.get("seed", 0))
    patch_size = int(cfg.get("model", {}).get("patch_size", data_cfg.get("patch_size", 13)))

    scene = build_scene(cfg)

    source_full = SourceDataset(
        scene.source_cube,
        scene.source_gt,
        patch_size=patch_size,
        samples_per_class=data_cfg.get("samples_per_class"),
        augment=bool(data_cfg.get("augment", False)),
        seed=seed,
    )
    train_idx, val_idx = split_train_val(
        source_full, float(data_cfg.get("val_fraction", 0.0) or 0.0), seed=seed
    )
    source_train = Subset(source_full, train_idx) if len(val_idx) else source_full
    source_val = Subset(source_full, val_idx) if len(val_idx) else None

    target_train = TargetDataset(
        scene.target_cube,
        scene.target_gt,
        patch_size=patch_size,
        pixels=data_cfg.get("target_pixels", "labelled"),
        max_samples=data_cfg.get("target_max_samples"),
        augment=bool(data_cfg.get("augment", False)),
        seed=seed,
    )
    target_eval = TargetDataset(
        scene.target_cube,
        scene.target_gt,
        patch_size=patch_size,
        pixels="labelled",
        augment=False,
        seed=seed,
    )

    class_names = list(scene.class_names) or [f"class {i + 1}" for i in range(scene.num_classes)]
    return DataBundle(
        scene=scene,
        source_train=source_train,
        source_val=source_val,
        target_train=target_train,
        target_eval=target_eval,
        num_classes=scene.num_classes,
        class_names=class_names,
    )


def build_dataloaders(cfg: Dict[str, Any], bundle: DataBundle) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 64))
    eval_batch_size = int(train_cfg.get("eval_batch_size", max(batch_size, 256)))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", torch.cuda.is_available()))

    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # Dropping the last batch keeps the source/target batch sizes equal, but must
    # not empty a loader that is smaller than one batch.
    source_loader = DataLoader(
        bundle.source_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(bundle.source_train) > batch_size,
        **common,
    )
    target_loader = DataLoader(
        bundle.target_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(bundle.target_train) > batch_size,
        **common,
    )
    target_eval_loader = DataLoader(
        bundle.target_eval, batch_size=eval_batch_size, shuffle=False, **common
    )
    loaders = {
        "source": source_loader,
        "target": InfiniteLoader(target_loader),
        "target_raw": target_loader,
        "target_eval": target_eval_loader,
    }
    if bundle.source_val is not None:
        loaders["source_val"] = DataLoader(
            bundle.source_val, batch_size=eval_batch_size, shuffle=False, **common
        )
    return loaders
