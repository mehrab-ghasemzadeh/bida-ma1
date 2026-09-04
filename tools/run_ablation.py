"""Run the ablation study of section 28 and collect the results into one table.

    python tools/run_ablation.py --dataset houston --seeds 0 1 2
    python tools/run_ablation.py --configs configs/ablation/full.yaml --seeds 0

Each experiment is a separate `train.py` invocation, so a crash in one row does
not lose the others; results are read back from each run's summary.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

MAIN_TABLE = [
    ("Source only", "configs/ablation/source_only.yaml"),
    ("Transformer baseline", "configs/ablation/transformer.yaml"),
    ("MoE", "configs/ablation/moe.yaml"),
    ("Coupled", "configs/ablation/coupled.yaml"),
    ("Distillation", "configs/ablation/distillation.yaml"),
    ("Full model (HMA)", "configs/ablation/full.yaml"),
]

EXTRA = [
    ("EMA teacher", "configs/ablation/ema_teacher.yaml"),
    ("Frozen teacher", "configs/ablation/frozen_teacher.yaml"),
    ("Top-1 routing", "configs/ablation/top1_routing.yaml"),
    ("2 experts", "configs/ablation/experts_2.yaml"),
    ("8 experts", "configs/ablation/experts_8.yaml"),
    ("1 semantic token", "configs/ablation/tokens_1.yaml"),
    ("10 semantic tokens", "configs/ablation/tokens_10.yaml"),
    ("Inference option B", "configs/ablation/inference_b.yaml"),
    ("Shared branches", "configs/ablation/shared_branches.yaml"),
    ("Aux CE on coupled", "configs/ablation/aux_cls_coupled.yaml"),
]


def run_one(config: str, dataset: str, seed: int, out_root: Path,
            extra: List[str]) -> Dict[str, float]:
    name = Path(config).stem
    out_dir = out_root / dataset / name / f"seed{seed}"
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--config", str(ROOT / config),
        "--output-dir", str(out_dir),
        "--seed", str(seed),
        "--override", f"dataset.name={dataset}",
    ] + extra
    print(f"\n>>> {name} | {dataset} | seed {seed}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        return {}
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    return summary.get("best", summary.get("last", {}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="houston")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--configs", nargs="*", default=None,
                        help="explicit config list (default: the full ablation table)")
    parser.add_argument("--include-extra", action="store_true",
                        help="also run the routing / expert-count / token-count studies")
    parser.add_argument("--out-root", default="runs/ablation")
    parser.add_argument("--override", nargs="*", default=[],
                        help="extra overrides forwarded to train.py")
    args = parser.parse_args()

    if args.configs:
        table = [(Path(c).stem, c) for c in args.configs]
    else:
        table = list(MAIN_TABLE) + (list(EXTRA) if args.include_extra else [])

    out_root = Path(args.out_root)
    extra_overrides = ["--override"] + args.override if args.override else []
    results: Dict[str, List[Dict[str, float]]] = {}

    for label, config in table:
        runs = []
        for seed in args.seeds:
            metrics = run_one(config, args.dataset, seed, out_root, extra_overrides)
            if metrics:
                runs.append(metrics)
        results[label] = runs

    print("\n\n=== ablation summary: " + args.dataset + " ===")
    header = f"{'experiment':<24}{'OA':>16}{'AA':>16}{'Kappa':>16}"
    print(header)
    print("-" * len(header))
    for label, runs in results.items():
        if not runs:
            print(f"{label:<24}{'(failed)':>16}")
            continue
        cells = []
        for key in ("OA", "AA", "Kappa"):
            values = [run[key] * 100 for run in runs if key in run]
            if not values:
                cells.append("n/a")
            elif len(values) == 1:
                cells.append(f"{values[0]:.2f}")
            else:
                mean = sum(values) / len(values)
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                cells.append(f"{mean:.2f}+-{var ** 0.5:.2f}")
        print(f"{label:<24}" + "".join(f"{c:>16}" for c in cells))

    out_path = out_root / f"{args.dataset}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
