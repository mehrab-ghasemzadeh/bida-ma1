"""Target-domain evaluation and classification-map generation (Step 20)."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_metrics


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    domain: str = "target",
    source_batch: Optional[torch.Tensor] = None,
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return (predictions, labels)."""
    model.eval()
    preds, labels = [], []
    autocast = torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda")

    for patches, targets in loader:
        patches = patches.to(device, non_blocking=True)
        with autocast:
            if domain == "target":
                logits = model.predict_target(patches, x_s=source_batch)
            else:
                out = model(x_s=patches, with_teacher=False)
                logits = out["logits_s"]
        preds.append(logits.float().argmax(dim=1).cpu().numpy())
        labels.append(targets.numpy())

    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    domain: str = "target",
    source_batch: Optional[torch.Tensor] = None,
    amp: bool = False,
) -> Dict[str, object]:
    preds, labels = predict_loader(
        model, loader, device, domain=domain, source_batch=source_batch, amp=amp
    )
    metrics = compute_metrics(labels, preds, num_classes)
    metrics["predictions"] = preds
    metrics["labels"] = labels
    return metrics


@torch.no_grad()
def classification_map(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    scene_shape: Tuple[int, int],
    batch_size: int = 512,
    source_batch: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Predict every pixel of a scene and fold the predictions back into a map.

    The returned map holds 1-based class indices, matching the ground-truth
    convention (0 is reserved for "not predicted").
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for patches, _ in loader:
        patches = patches.to(device, non_blocking=True)
        logits = model.predict_target(patches, x_s=source_batch)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    out = np.zeros(scene_shape, dtype=np.int64)
    coords = dataset.coords
    out[coords[:, 0], coords[:, 1]] = preds + 1
    return out


def summarise_expert_usage(usage: Dict[str, torch.Tensor]) -> str:
    """Readable dump of per-layer MoE expert usage (research question Q3)."""
    lines = []
    for name, fractions in sorted(usage.items()):
        values = ", ".join(f"{v * 100:5.1f}%" for v in fractions.tolist())
        lines.append(f"  {name:<45} [{values}]")
    return "\n".join(lines) if lines else "  (no MoE layers)"
