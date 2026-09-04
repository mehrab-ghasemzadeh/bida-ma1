"""Verification tests for Phases 1-10 of the implementation plan (section 27).

Run everything with either:

    python -m tests.test_phases
    pytest tests/test_phases.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import build_data, build_dataloaders, synthetic_scene_pair  # noqa: E402
from data.source_dataset import SourceDataset  # noqa: E402
from losses import DistillationLoss  # noqa: E402
from models import (  # noqa: E402
    CoupledTransformer,
    MoELayer,
    SemanticTokenizer,
    TransformerEncoder,
    build_model,
)
from models.hma_teacher import TeacherBranch, _weighted_moving_average  # noqa: E402
from models.moe import collect_moe_aux, reset_moe_stats  # noqa: E402
from training.scheduler import WarmupWeight  # noqa: E402
from utils import Config  # noqa: E402

PATCH, BANDS, DIM, TOKENS, CLASSES = 13, 13, 64, 5, 7


def base_config(**overrides) -> Config:
    cfg = Config(
        {
            "seed": 0,
            "dataset": {
                "synthetic": True,
                "synthetic_size": 48,
                "num_classes": CLASSES,
                "spectral_reduction": "none",
                "target_pixels": "labelled",
            },
            "model": {
                "patch_size": PATCH,
                "num_bands": BANDS,
                "embed_dim": DIM,
                "num_tokens": TOKENS,
                "depth": 2,
                "coupled_depth": 1,
                "num_heads": 4,
                "dropout": 0.1,
                "use_moe": True,
                "use_coupled": True,
                "use_distillation": True,
                "moe": {"num_experts": 4, "top_k": 2, "hidden_mult": 2.0},
                "teacher": {"mode": "hma", "window": 4},
            },
            "train": {"batch_size": 8, "eval_batch_size": 32, "num_workers": 0},
            "loss": {},
        }
    )
    for key, value in overrides.items():
        cfg.set_path(key, value)
    return cfg


# ------------------------------------------------------------------- Phase 1
def test_phase1_data_shapes():
    """Input patches are [B, 1, 13, 13, 13]."""
    scene = synthetic_scene_pair(num_bands=BANDS, num_classes=CLASSES, size=48, seed=0)
    dataset = SourceDataset(scene.source_cube, scene.source_gt, patch_size=PATCH)
    patch, label = dataset[0]
    assert patch.shape == (1, BANDS, PATCH, PATCH), patch.shape
    assert patch.dtype == torch.float32
    assert 0 <= int(label) < CLASSES

    bundle = build_data(base_config())
    loaders = build_dataloaders(base_config(), bundle)
    batch, labels = next(iter(loaders["source"]))
    assert batch.shape == (8, 1, BANDS, PATCH, PATCH), batch.shape
    assert labels.shape == (8,)

    target_batch = loaders["target"].next()[0]
    assert target_batch.shape == (8, 1, BANDS, PATCH, PATCH)
    print("phase 1 ok: patches are [B, 1, 13, 13, 13]")


# ------------------------------------------------------------------- Phase 2
def test_phase2_tokenizer():
    """Tokenizer: [B, 1, 13, 13, 13] -> [B, 5, D]."""
    tokenizer = SemanticTokenizer(PATCH, BANDS, DIM, TOKENS)
    x = torch.randn(4, 1, BANDS, PATCH, PATCH)
    tokens = tokenizer(x)
    assert tokens.shape == (4, TOKENS, DIM), tokens.shape

    # The pooling attention is a convex combination over spatial positions.
    attn = tokenizer.pool.last_attention
    assert attn.shape == (4, tokenizer.num_spatial, TOKENS)
    assert torch.allclose(attn.sum(dim=1), torch.ones(4, TOKENS), atol=1e-5)

    tokens.sum().backward()
    assert tokenizer.pool.attend.weight.grad is not None
    print(f"phase 2 ok: tokens {tuple(tokens.shape)} from {tokenizer.num_spatial} positions")


# ------------------------------------------------------------------- Phase 3
def test_phase3_transformer_without_moe():
    """Plain Transformer: [B, 5, D] -> [B, 5, D]."""
    encoder = TransformerEncoder(DIM, depth=2, num_heads=4, ffn_cfg={"use_moe": False})
    x = torch.randn(4, TOKENS, DIM)
    out = encoder(x)
    assert out.shape == x.shape
    assert not [m for m in encoder.modules() if isinstance(m, MoELayer)]
    print("phase 3 ok: FFN Transformer preserves [B, 5, D]")


# ------------------------------------------------------------------- Phase 4
def test_phase4_top2_moe():
    """Top-2 routing activates exactly two experts and renormalises their gates."""
    torch.manual_seed(0)
    moe = MoELayer(DIM, num_experts=4, top_k=2, hidden_mult=2.0, dropout=0.0)
    moe.eval()
    x = torch.randn(3, TOKENS, DIM)
    out = moe(x)
    assert out.shape == x.shape

    flat = x.reshape(-1, DIM)
    probs = F.softmax(moe.router(flat), dim=-1)
    top_val, top_idx = probs.topk(2, dim=-1)
    gates = top_val / top_val.sum(dim=-1, keepdim=True)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(flat.shape[0]), atol=1e-6)

    manual = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(2):
            expert = moe.experts[int(top_idx[token, slot])]
            manual[token] += gates[token, slot] * expert(flat[token])
    assert torch.allclose(out.reshape(-1, DIM), manual, atol=1e-5), \
        (out.reshape(-1, DIM) - manual).abs().max()

    # The balancing loss is zero exactly at the uniform distribution.
    moe.train()
    moe.reset_stats()
    moe(x)
    aux = collect_moe_aux(moe)
    assert aux.requires_grad and float(aux) >= 0.0
    uniform = torch.full((10, 4), 0.25)
    assert float(moe._balance_loss(uniform, uniform.topk(2, -1).indices)) < 1e-12
    print(f"phase 4 ok: top-2 routing verified, balance loss {float(aux):.6f}")


# ------------------------------------------------------------------- Phase 5
def test_phase5_independent_branches():
    """Source and target branches run independently and hold separate parameters."""
    model = build_model(base_config().model, CLASSES)
    x_s = torch.randn(4, 1, BANDS, PATCH, PATCH)
    x_t = torch.randn(6, 1, BANDS, PATCH, PATCH)

    out_s = model(x_s=x_s, with_teacher=False)
    assert out_s["H_s"].shape == (4, TOKENS, DIM)
    assert out_s["logits_s"].shape == (4, CLASSES)
    assert "H_t" not in out_s

    out_t = model(x_t=x_t, with_teacher=False)
    assert out_t["H_t"].shape == (6, TOKENS, DIM)
    assert "H_s" not in out_t

    src_ids = {id(p) for p in model.source_branch.parameters()}
    tgt_ids = {id(p) for p in model.target_branch.parameters()}
    assert src_ids.isdisjoint(tgt_ids), "branches must not share parameters by default"
    print("phase 5 ok: independent source and target branches")


# ------------------------------------------------------------------- Phase 6
def test_phase6_coupled_cross_attention():
    """Both coupling directions return [B, 5, D]."""
    coupled = CoupledTransformer(DIM, depth=2, num_heads=4, ffn_cfg={"use_moe": False})
    h_s = torch.randn(4, TOKENS, DIM)
    h_t = torch.randn(4, TOKENS, DIM)
    c_s, c_t = coupled(h_s, h_t)
    assert c_s.shape == (4, TOKENS, DIM) and c_t.shape == (4, TOKENS, DIM)

    # The two directions are different computations.
    assert not torch.allclose(c_s, c_t)

    # C_{s<-t} must depend on the target stream, and vice versa.
    c_s_alt, _ = coupled(h_s, torch.randn_like(h_t))
    assert not torch.allclose(c_s, c_s_alt, atol=1e-6)
    print("phase 6 ok: bidirectional cross-attention shapes and dependencies")


# ------------------------------------------------------------------- Phase 7
def test_phase7_moe_in_coupled_and_balance_loss():
    """MoE inside the coupled Transformer contributes to the balancing loss."""
    model = build_model(base_config().model, CLASSES)
    model.train()
    reset_moe_stats(model)

    x_s = torch.randn(4, 1, BANDS, PATCH, PATCH)
    x_t = torch.randn(4, 1, BANDS, PATCH, PATCH)
    out = model(x_s=x_s, x_t=x_t)

    assert out["C_s_from_t"].shape == (4, TOKENS, DIM)
    assert out["C_t_from_s"].shape == (4, TOKENS, DIM)

    coupled_aux = [loss for m in model.coupled.modules() if isinstance(m, MoELayer)
                   for loss in m.aux_losses]
    assert coupled_aux, "the coupled Transformer recorded no MoE statistics"

    aux = collect_moe_aux(model)
    assert aux.requires_grad
    aux.backward()
    router = next(m for m in model.coupled.modules() if isinstance(m, MoELayer)).router
    assert router.weight.grad is not None and router.weight.grad.abs().sum() > 0
    print(f"phase 7 ok: {len(coupled_aux)} coupled MoE calls, aux {float(aux):.6f}")


# ------------------------------------------------------------------- Phase 8
def test_phase8_stop_gradient():
    """The teacher receives no gradient; the student does."""
    model = build_model(base_config().model, CLASSES)
    model.train()
    x_s = torch.randn(4, 1, BANDS, PATCH, PATCH)
    x_t = torch.randn(4, 1, BANDS, PATCH, PATCH)

    for teacher in (model.source_teacher, model.target_teacher):
        assert all(not p.requires_grad for p in teacher.parameters())
        assert not teacher.module.training, "the teacher must stay in eval mode"

    out = model(x_s=x_s, x_t=x_t)
    assert not out["h_s_teacher"].requires_grad
    assert out["h_s_from_t"].requires_grad

    loss = DistillationLoss("cosine")(out)["dist"]
    loss.backward()

    teacher_grads = [p.grad for p in model.source_teacher.parameters() if p.grad is not None]
    assert not teacher_grads, "teacher parameters received gradients"

    student_grads = [
        p.grad for p in model.coupled.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert student_grads, "the coupled student received no gradients"
    print(f"phase 8 ok: stop-gradient holds, distillation loss {float(loss):.4f}")


# ------------------------------------------------------------------- Phase 9
def test_phase9_hma_teacher():
    """The HMA teacher reproduces the WMA cascade and evolves smoothly."""
    torch.manual_seed(0)

    # (a) exactness on a one-parameter module against a hand-computed HMA
    student = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(0.0)
    teacher = TeacherBranch(student, mode="hma", window=4)

    series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    history, raw_history = [], []
    for value in series:
        with torch.no_grad():
            student.weight.fill_(value)
        teacher.update(student)

        history.append(value)
        window = history[-4:]
        half = history[-2:]
        wma = lambda seq: sum((i + 1) * v for i, v in enumerate(seq)) / sum(
            i + 1 for i in range(len(seq))
        )
        raw_history.append(2 * wma(half) - wma(window))
        expected = wma(raw_history[-2:])   # sqrt(4) = 2
        assert abs(float(teacher.module.weight) - expected) < 1e-5, (
            float(teacher.module.weight), expected
        )

    # (b) the teacher lags a noisy student but tracks its trend
    student = torch.nn.Linear(8, 8, bias=False)
    teacher = TeacherBranch(student, mode="hma", window=8)
    drifts = []
    for step in range(30):
        with torch.no_grad():
            student.weight.add_(torch.randn_like(student.weight) * 0.05)
        teacher.update(student)
        drifts.append(teacher.drift(student))
    assert all(np.isfinite(drifts))
    assert max(drifts) < 10.0, "the teacher diverged from the student"

    # (c) the teacher smooths: its own step-to-step movement is smaller than
    #     the student's under pure zero-mean noise
    student = torch.nn.Linear(16, 16, bias=False)
    teacher = TeacherBranch(student, mode="hma", window=8)
    base = student.weight.detach().clone()
    student_steps, teacher_steps = [], []
    previous_teacher = teacher.module.weight.detach().clone()
    previous_student = student.weight.detach().clone()
    for step in range(40):
        with torch.no_grad():
            student.weight.copy_(base + torch.randn_like(base) * 0.1)
        teacher.update(student)
        student_steps.append(float((student.weight - previous_student).norm()))
        teacher_steps.append(float((teacher.module.weight - previous_teacher).norm()))
        previous_student = student.weight.detach().clone()
        previous_teacher = teacher.module.weight.detach().clone()
    mean_student = float(np.mean(student_steps[8:]))
    mean_teacher = float(np.mean(teacher_steps[8:]))
    assert mean_teacher < mean_student, (mean_teacher, mean_student)

    # (d) EMA and frozen modes behave as expected
    student = torch.nn.Linear(4, 4, bias=False)
    frozen = TeacherBranch(student, mode="frozen")
    before = frozen.module.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(1.0)
    frozen.update(student)
    assert torch.allclose(frozen.module.weight, before)

    ema = TeacherBranch(student, mode="ema", ema_decay=0.5)
    start = ema.module.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(2.0)
    ema.update(student)
    assert torch.allclose(ema.module.weight, 0.5 * start + 0.5 * student.weight, atol=1e-6)

    print(
        "phase 9 ok: HMA matches the WMA cascade; teacher step "
        f"{mean_teacher:.4f} < student step {mean_student:.4f}"
    )


def test_weighted_moving_average():
    """WMA weights are linear and normalised, newest sample heaviest."""
    history = [{"w": torch.tensor([float(v)])} for v in (1.0, 2.0, 3.0)]
    out = _weighted_moving_average(history, 3, ["w"])
    expected = (1 * 1.0 + 2 * 2.0 + 3 * 3.0) / 6.0
    assert abs(float(out["w"]) - expected) < 1e-6
    partial = _weighted_moving_average(history, 10, ["w"])
    assert abs(float(partial["w"]) - expected) < 1e-6


# ------------------------------------------------------------------ Phase 10
def test_phase10_warmup_and_training_step():
    """Distillation warm-up follows lambda_max * min(1, e / E_warmup)."""
    schedule = WarmupWeight(max_weight=1.0, warmup=20)
    assert schedule.at(0) == 0.0
    assert abs(schedule.at(10) - 0.5) < 1e-9
    assert abs(schedule.at(20) - 1.0) < 1e-9
    assert abs(schedule.at(40) - 1.0) < 1e-9

    delayed = WarmupWeight(max_weight=0.5, warmup=10, delay=5)
    assert delayed.at(5) == 0.0
    assert abs(delayed.at(10) - 0.25) < 1e-9

    # A full optimisation step reduces the loss on a fixed batch.
    from training import Trainer

    cfg = base_config()
    cfg.set_path("train.epochs", 1)
    cfg.set_path("train.lr", 1e-3)
    cfg.set_path("loss.dist_warmup_epochs", 0)
    bundle = build_data(cfg)
    loaders = build_dataloaders(cfg, bundle)
    model = build_model(cfg.model, bundle.num_classes)
    trainer = Trainer(model, loaders, cfg, bundle.num_classes, bundle.class_names,
                      device=torch.device("cpu"), output_dir="runs/_test")
    trainer.log_every = 0

    source_batch = next(iter(loaders["source"]))
    target_batch = loaders["target"].next()

    first = trainer.train_step(source_batch, target_batch)
    for _ in range(30):
        stats = trainer.train_step(source_batch, target_batch)
    assert stats["loss"] < first["loss"], (first["loss"], stats["loss"])
    assert stats["lambda_dist"] > 0
    assert "teacher_drift_s" in stats
    print(
        f"phase 10 ok: loss {first['loss']:.4f} -> {stats['loss']:.4f}, "
        f"teacher drift {stats['teacher_drift_s']:.4f}"
    )


def test_inference_options():
    """Option A needs no source data; Option B consumes source tokens."""
    model = build_model(base_config().model, CLASSES)
    model.eval()
    x_t = torch.randn(5, 1, BANDS, PATCH, PATCH)
    logits_a = model.predict_target(x_t)
    assert logits_a.shape == (5, CLASSES)

    model.inference_mode = "B"
    x_s = torch.randn(3, 1, BANDS, PATCH, PATCH)   # deliberately a different batch size
    logits_b = model.predict_target(x_t, x_s=x_s)
    assert logits_b.shape == (5, CLASSES)
    print("inference ok: option A and option B both produce target logits")


def main() -> int:
    tests = [
        test_phase1_data_shapes,
        test_phase2_tokenizer,
        test_phase3_transformer_without_moe,
        test_phase4_top2_moe,
        test_phase5_independent_branches,
        test_phase6_coupled_cross_attention,
        test_phase7_moe_in_coupled_and_balance_loss,
        test_phase8_stop_gradient,
        test_phase9_hma_teacher,
        test_weighted_moving_average,
        test_phase10_warmup_and_training_step,
        test_inference_options,
    ]
    failures = 0
    for test in tests:
        torch.manual_seed(0)
        np.random.seed(0)
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
    total = len(tests)
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
