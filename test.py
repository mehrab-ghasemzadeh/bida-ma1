"""Evaluate a trained checkpoint on the target domain.

Examples
--------
    python test.py --checkpoint runs/houston/best.pt
    python test.py --checkpoint runs/houston/best.pt --save-map runs/houston/map.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from data import build_data, build_dataloaders
from data.target_dataset import target_full_scene_dataset
from evaluation import classification_map, evaluate, format_confusion_matrix, format_metrics
from evaluation.evaluate import summarise_expert_usage
from models import build_model
from models.moe import moe_usage
from utils import Config, get_logger, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--config", default=None,
                        help="config to use (default: the one stored in the checkpoint)")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-map", default=None,
                        help="also predict the whole target scene and save it as .npy")
    parser.add_argument("--save-predictions", default=None,
                        help="save the target predictions and labels as .npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("cdhsi-test")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.config:
        cfg = load_config(args.config, args.override)
    else:
        cfg = Config(state["config"])
        for override in args.override:
            key, _, value = override.partition("=")
            cfg.set_path(key.strip(), yaml.safe_load(value.strip()))

    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    bundle = build_data(cfg)
    loaders = build_dataloaders(cfg, bundle)

    model = build_model(cfg.get("model", {}), bundle.num_classes).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    logger.info(f"loaded {args.checkpoint} (epoch {state.get('epoch')})")

    source_batch = None
    if getattr(model, "inference_mode", "A") == "B":
        patches, _ = next(iter(loaders["source"]))
        source_batch = patches.to(device)

    metrics = evaluate(
        model, loaders["target_eval"], device, bundle.num_classes,
        domain="target", source_batch=source_batch,
    )
    logger.info("target-domain results:")
    print(format_metrics(metrics, bundle.class_names))
    print()
    print(format_confusion_matrix(metrics["confusion_matrix"], bundle.class_names))

    usage = moe_usage(model)
    if usage:
        print("\nMoE expert usage:")
        print(summarise_expert_usage(usage))

    if args.save_predictions:
        Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_predictions,
            predictions=metrics["predictions"],
            labels=metrics["labels"],
            confusion_matrix=metrics["confusion_matrix"],
        )
        logger.info(f"predictions saved to {args.save_predictions}")

    if args.save_map:
        patch_size = int(cfg.get_path("model.patch_size", 13))
        full = target_full_scene_dataset(
            bundle.scene.target_cube, bundle.scene.target_gt, patch_size=patch_size
        )
        cmap = classification_map(
            model, full, device, bundle.scene.target_gt.shape,
            batch_size=int(cfg.get_path("train.eval_batch_size", 512)),
            source_batch=source_batch,
        )
        Path(args.save_map).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_map, cmap)
        logger.info(f"classification map saved to {args.save_map} (shape {cmap.shape})")


if __name__ == "__main__":
    main()
