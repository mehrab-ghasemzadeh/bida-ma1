"""End-to-end smoke test on synthetic scenes - no real data or GPU required.

Prints the tensor shape at every stage of the pipeline, runs a few training
iterations of the complete model (tokenizer, MoE Transformers, coupled
cross-attention, HMA teachers), evaluates the target domain, and round-trips a
checkpoint.

    python smoke_test.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from data import build_data, build_dataloaders
from evaluation import format_metrics
from evaluation.evaluate import summarise_expert_usage
from models import build_model
from models.moe import moe_usage
from training import Trainer
from utils import load_config, set_seed


def trace_shapes(model, x_s: torch.Tensor, x_t: torch.Tensor) -> None:
    print("\n--- forward shape trace ---")
    print(f"input patch                     {tuple(x_s.shape)}")

    tok = model.source_branch.tokenizer
    feat = tok.conv3d_1(x_s)
    print(f"after Conv3D(2,2,2) + BN + LReLU {tuple(feat.shape)}")
    feat = tok.conv3d_2(feat)
    print(f"after Conv3D(2,2,2) + BN + LReLU {tuple(feat.shape)}")
    b, c, s, h, w = feat.shape
    reshaped = feat.reshape(b, c * s, h, w)
    print(f"reshaped to 2D                   {tuple(reshaped.shape)}")
    conv2d = tok.conv2d(reshaped)
    print(f"after Conv2D(2,2) + BN + LReLU   {tuple(conv2d.shape)}")

    tokens = tok(x_s)
    print(f"semantic tokens                  {tuple(tokens.shape)}")
    tokens = model.source_branch.pos_enc(tokens)
    print(f"+ positional encoding            {tuple(tokens.shape)}")

    out = model(x_s=x_s, x_t=x_t)
    print(f"H_s (source encoder)             {tuple(out['H_s'].shape)}")
    print(f"H_t (target encoder)             {tuple(out['H_t'].shape)}")
    print(f"h_s (mean pooled)                {tuple(out['h_s'].shape)}")
    print(f"logits_s                         {tuple(out['logits_s'].shape)}")
    print(f"C_s<-t (source queries target)   {tuple(out['C_s_from_t'].shape)}")
    print(f"C_t<-s (target queries source)   {tuple(out['C_t_from_s'].shape)}")
    print(f"h_s teacher (no grad)            {tuple(out['h_s_teacher'].shape)}"
          f"  requires_grad={out['h_s_teacher'].requires_grad}")
    print(f"h_t teacher (no grad)            {tuple(out['h_t_teacher'].shape)}"
          f"  requires_grad={out['h_t_teacher'].requires_grad}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep", action="store_true", help="keep the run directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(args.device)

    output_dir = Path(cfg.get("output_dir", "runs/smoke"))
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print("=== data ===")
    bundle = build_data(cfg)
    print(bundle.describe())
    loaders = build_dataloaders(cfg, bundle)

    print("\n=== model ===")
    model = build_model(cfg.model, bundle.num_classes).to(device)
    trainable = sum(p.numel() for p in model.student_parameters())
    teacher_params = sum(
        p.numel()
        for teacher in (model.source_teacher, model.target_teacher)
        if teacher is not None
        for p in teacher.parameters()
    )
    print(f"trainable parameters: {trainable:,}")
    print(f"teacher parameters (frozen): {teacher_params:,}")

    x_s = next(iter(loaders["source"]))[0][:4].to(device)
    x_t = loaders["target"].next()[0][:4].to(device)
    model.eval()
    with torch.no_grad():
        trace_shapes(model, x_s, x_t)

    print("\n=== training ===")
    trainer = Trainer(
        model=model,
        loaders=loaders,
        cfg=cfg,
        num_classes=bundle.num_classes,
        class_names=bundle.class_names,
        device=device,
        output_dir=output_dir,
    )
    summary = trainer.fit()

    print("\n=== target metrics (last epoch) ===")
    metrics = trainer.evaluate_target()
    print(format_metrics(metrics, bundle.class_names))

    usage = moe_usage(model)
    if usage:
        print("\n=== MoE expert usage ===")
        print(summarise_expert_usage(usage))

    print("\n=== checkpoint round-trip ===")
    checkpoint = output_dir / "last.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded = build_model(cfg.model, bundle.num_classes).to(device)
    reloaded.load_state_dict(state["model"])
    reloaded.eval()
    with torch.no_grad():
        a = model.predict_target(x_t)
        b = reloaded.predict_target(x_t)
    assert torch.allclose(a, b, atol=1e-5), "reloaded model disagrees with the trained one"
    print(f"reloaded {checkpoint} and reproduced identical target logits")

    if not args.keep:
        shutil.rmtree(output_dir, ignore_errors=True)

    print(
        "\nsmoke test passed: target OA "
        f"{summary['last']['OA'] * 100:.2f}% after {cfg.get_path('train.epochs')} epochs "
        "on synthetic data"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
