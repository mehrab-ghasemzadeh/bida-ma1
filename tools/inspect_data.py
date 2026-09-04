"""Inspect the downloaded .mat files before training.

Prints the arrays inside each file with their shapes and dtypes, and - once a
dataset resolves - the class distribution of both scenes. Use this first when a
new download does not load.

    python tools/inspect_data.py --root data/raw
    python tools/inspect_data.py --config configs/houston.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.preprocessing import load_mat  # noqa: E402
from data.registry import DATASETS, resolve_dataset  # noqa: E402
from utils import load_config  # noqa: E402


def describe_file(path: Path) -> None:
    print(f"\n{path}")
    try:
        from scipy.io import loadmat

        mat = loadmat(str(path))
        arrays = {k: v for k, v in mat.items()
                  if not k.startswith("__") and isinstance(v, np.ndarray)}
    except Exception:  # noqa: BLE001 - fall back to HDF5
        try:
            import h5py

            with h5py.File(path, "r") as handle:
                arrays = {k: np.array(handle[k]).T for k in handle.keys()}
        except Exception as exc:  # noqa: BLE001
            print(f"  could not read: {exc}")
            return
    for key, value in sorted(arrays.items()):
        extra = ""
        if value.ndim == 2 and np.issubdtype(value.dtype, np.integer):
            labels = np.unique(value)
            extra = f"  labels={labels[:15].tolist()}{'...' if labels.size > 15 else ''}"
        print(f"  {key:<16} {str(value.shape):<20} {value.dtype}{extra}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--config", default=None,
                        help="also try to resolve the dataset from this config")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"data root '{root}' does not exist")
        return 1

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".mat", ".npy", ".npz"})
    if not files:
        print(f"no .mat / .npy / .npz files under '{root}'")
        return 1

    print(f"found {len(files)} data file(s) under {root}")
    for path in files:
        describe_file(path)

    print("\nknown benchmarks:")
    for name, spec in DATASETS.items():
        print(f"  {name:<20} {spec['description']}")

    if args.config:
        cfg = load_config(args.config)
        print(f"\nresolving dataset from {args.config} ...")
        try:
            resolved = resolve_dataset(cfg.get("dataset", {}), root)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {exc}")
            return 1
        for field, path in resolved["paths"].items():
            print(f"  {field:<14} {path}")

        gt_src = load_mat(resolved["paths"]["source_gt"],
                          resolved["keys"].get("source_gt"), ndim=2)
        gt_tgt = load_mat(resolved["paths"]["target_gt"],
                          resolved["keys"].get("target_gt"), ndim=2)
        names = resolved["class_names"] or []
        print("\nclass distribution (source / target):")
        for cls in range(1, int(max(gt_src.max(), gt_tgt.max())) + 1):
            label = names[cls - 1] if cls - 1 < len(names) else f"class {cls}"
            print(f"  {label:<32} {int((gt_src == cls).sum()):>8} "
                  f"{int((gt_tgt == cls).sum()):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
