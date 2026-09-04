"""Write synthetic scenes to .mat files, to exercise the real file-loading path.

Useful before the real data arrives: it produces files in the same layout as a
downloaded benchmark, so `tools/inspect_data.py`, the registry and the PCA
spectral reduction can all be tested end to end.

    python tools/make_synthetic_data.py --out data/raw/synthetic --bands 48
    python train.py --config configs/synthetic_files.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.builder import synthetic_scene_pair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/synthetic")
    parser.add_argument("--bands", type=int, default=48)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unlabelled-fraction", type=float, default=0.3,
                        help="fraction of pixels set to 0 in the ground-truth maps")
    args = parser.parse_args()

    from scipy.io import savemat

    scene = synthetic_scene_pair(
        num_bands=args.bands, num_classes=args.classes, size=args.size, seed=args.seed
    )

    rng = np.random.default_rng(args.seed)
    src_gt = scene.source_gt.copy()
    tgt_gt = scene.target_gt.copy()
    src_gt[rng.random(src_gt.shape) < args.unlabelled_fraction] = 0
    tgt_gt[rng.random(tgt_gt.shape) < args.unlabelled_fraction] = 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    savemat(out / "SynthSource.mat", {"ori_data": scene.source_cube})
    savemat(out / "SynthSource_gt.mat", {"map": src_gt})
    savemat(out / "SynthTarget.mat", {"ori_data": scene.target_cube})
    savemat(out / "SynthTarget_gt.mat", {"map": tgt_gt})

    print(f"wrote 4 files to {out}")
    print(f"  cube {scene.source_cube.shape}, {args.classes} classes")
    print("\nUse them with a custom dataset config:")
    print(
        "\ndataset:\n  name: custom\n"
        f"  source_image: {out / 'SynthSource.mat'}\n"
        f"  source_gt: {out / 'SynthSource_gt.mat'}\n"
        f"  target_image: {out / 'SynthTarget.mat'}\n"
        f"  target_gt: {out / 'SynthTarget_gt.mat'}\n"
        "  keys: {source_image: ori_data, source_gt: map, "
        "target_image: ori_data, target_gt: map}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
