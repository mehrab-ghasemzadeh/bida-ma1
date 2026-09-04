"""Classification metrics for hyperspectral scenes.

Overall accuracy (OA), average (per-class mean) accuracy (AA) and the Cohen kappa
coefficient are the standard triple reported in the cross-scene literature;
per-class accuracies and the confusion matrix are also returned since the class
counts in these scenes are very unbalanced.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    valid = (y_true >= 0) & (y_true < num_classes)
    y_true, y_pred = y_true[valid], y_pred[valid]
    index = y_true * num_classes + y_pred
    counts = np.bincount(index, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


def overall_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    return acc


def average_accuracy(cm: np.ndarray) -> float:
    acc = per_class_accuracy(cm)
    valid = acc[~np.isnan(acc)]
    return float(valid.mean()) if valid.size else 0.0


def kappa_score(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    observed = np.trace(cm) / total
    expected = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / float(total**2)
    denom = 1.0 - expected
    return float((observed - expected) / denom) if abs(denom) > 1e-12 else 0.0


def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall /
                      np.maximum(precision + recall, 1e-12), 0.0)
    support = cm.sum(axis=1)
    return float(f1[support > 0].mean()) if np.any(support > 0) else 0.0


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return {
        "OA": overall_accuracy(cm),
        "AA": average_accuracy(cm),
        "Kappa": kappa_score(cm),
        "F1": macro_f1(cm),
        "per_class": per_class_accuracy(cm),
        "confusion_matrix": cm,
        "support": cm.sum(axis=1),
    }


def format_metrics(
    metrics: Dict[str, object], class_names: Optional[Sequence[str]] = None
) -> str:
    lines: List[str] = [
        f"OA    : {metrics['OA'] * 100:.2f}%",
        f"AA    : {metrics['AA'] * 100:.2f}%",
        f"Kappa : {metrics['Kappa'] * 100:.2f}",
        f"F1    : {metrics['F1'] * 100:.2f}",
    ]
    per_class = np.asarray(metrics["per_class"], dtype=float)
    support = np.asarray(metrics["support"], dtype=int)
    names = list(class_names) if class_names else [f"class {i + 1}" for i in range(len(per_class))]
    width = max((len(n) for n in names), default=10)
    lines.append("per-class accuracy:")
    for name, acc, count in zip(names, per_class, support):
        value = "  n/a " if np.isnan(acc) else f"{acc * 100:6.2f}%"
        lines.append(f"  {name:<{width}}  {value}  (n={count})")
    return "\n".join(lines)


def format_confusion_matrix(cm: np.ndarray, class_names: Optional[Sequence[str]] = None) -> str:
    n = cm.shape[0]
    names = list(class_names) if class_names else [f"c{i + 1}" for i in range(n)]
    short = [name[:8] for name in names]
    width = max(6, max(len(s) for s in short) + 1)
    header = " " * (width + 1) + "".join(f"{s:>{width}}" for s in short)
    rows = [header]
    for i in range(n):
        row = f"{short[i]:<{width}} " + "".join(f"{int(v):>{width}}" for v in cm[i])
        rows.append(row)
    return "\n".join(rows)
