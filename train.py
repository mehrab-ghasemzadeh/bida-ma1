"""Train the cross-domain hyperspectral classifier.

Examples
--------
    python train.py --config configs/houston.yaml
    python train.py --config configs/smoke.yaml --override train.epochs=1
    python train.py --config configs/ablation/moe.yaml --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_data, build_dataloaders
from models import build_model
from training import Trainer
from utils import get_logger, load_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="path to a YAML config")
    parser.add_argument("--override", nargs="*", default=[],
                        help="config overrides, e.g. train.epochs=50 model.embed_dim=128")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    parser.add_argument("--output-dir", default=None, help="overrides output_dir from the config")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir:
        cfg.output_dir = args.output_dir

    output_dir = Path(cfg.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("cdhsi", output_dir / "train.log")

    set_seed(int(cfg.get("seed", 0)), bool(cfg.get("deterministic", False)))
    device = resolve_device(args.device)

    logger.info(f"config: {args.config}")
    if args.override:
        logger.info(f"overrides: {' '.join(args.override)}")

    bundle = build_data(cfg)
    logger.info("data:\n" + bundle.describe())

    loaders = build_dataloaders(cfg, bundle)

    model = build_model(cfg.get("model", {}), bundle.num_classes)
    logger.info(
        "model: "
        f"D={cfg.get_path('model.embed_dim')} tokens={cfg.get_path('model.num_tokens')} "
        f"depth={cfg.get_path('model.depth')} coupled={cfg.get_path('model.use_coupled')} "
        f"moe={cfg.get_path('model.use_moe')} "
        f"distill={cfg.get_path('model.use_distillation')} "
        f"teacher={cfg.get_path('model.teacher.mode')}"
    )

    trainer = Trainer(
        model=model,
        loaders=loaders,
        cfg=cfg,
        num_classes=bundle.num_classes,
        class_names=bundle.class_names,
        device=device,
        output_dir=output_dir,
    )

    cfg.dump(output_dir / "config.yaml")

    if args.resume:
        trainer.load_checkpoint(args.resume, resume=True)

    summary = trainer.fit()
    logger.info(f"done. summary written to {output_dir / 'summary.json'}")
    logger.info(
        "last-epoch target OA {:.2f}% / AA {:.2f}% / Kappa {:.2f}".format(
            summary["last"]["OA"] * 100,
            summary["last"]["AA"] * 100,
            summary["last"]["Kappa"] * 100,
        )
    )


if __name__ == "__main__":
    main()
