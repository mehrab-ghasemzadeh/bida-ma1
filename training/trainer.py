"""Training loop (Step 19 of the specification).

Each iteration:
    1. sample a labelled source batch and an unlabelled target batch;
    2. tokenize both, add positional encodings, run the two independent encoders;
    3. classify the source representation                       -> L_cls
    4. run the bidirectional coupled cross-attention;
    5. run the HMA teachers without gradients                   -> h_s^T, h_t^T
    6. bidirectional representation distillation                -> L_dist
    7. MoE load balancing                                       -> L_balance
    8. L_total = L_cls + lambda_dist * L_dist + lambda_balance * L_balance
    9. backpropagate and step the optimiser (teachers get no gradients);
   10. push the new student weights through the HMA teacher update.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from evaluation.evaluate import evaluate, summarise_expert_usage
from evaluation.metrics import format_confusion_matrix, format_metrics
from losses import ClassificationLoss, DistillationLoss
from models.moe import collect_moe_aux, moe_usage, reset_moe_stats
from utils.logging import CSVLogger, get_logger

from .scheduler import WarmupWeight, build_lr_scheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, Any],
        cfg: Dict[str, Any],
        num_classes: int,
        class_names: Optional[list] = None,
        device: Optional[torch.device] = None,
        output_dir: str | Path = "runs/default",
    ):
        self.cfg = cfg
        train_cfg = cfg.get("train", {})
        loss_cfg = cfg.get("loss", {})

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loaders = loaders
        self.num_classes = num_classes
        self.class_names = class_names or [f"class {i + 1}" for i in range(num_classes)]

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("cdhsi", self.output_dir / "train.log")
        self.csv = CSVLogger(self.output_dir / "metrics.csv")

        self.epochs = int(train_cfg.get("epochs", 100))
        self.source_only = bool(train_cfg.get("source_only", False))
        self.grad_clip = float(train_cfg.get("grad_clip", 0.0) or 0.0)
        self.eval_every = int(train_cfg.get("eval_every", 1))
        self.log_every = int(train_cfg.get("log_every", 50))
        self.model_selection = train_cfg.get("model_selection", "target_oa")
        self.amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"

        # ------------------------------------------------------------ losses
        class_weights = None
        if loss_cfg.get("class_weights"):
            class_weights = torch.tensor(loss_cfg["class_weights"], dtype=torch.float32)
        self.cls_loss = ClassificationLoss(
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            class_weights=class_weights,
        ).to(self.device)
        self.dist_loss = DistillationLoss(
            distance=loss_cfg.get("distillation_distance", "cosine"),
            symmetric=bool(loss_cfg.get("symmetric_distillation", False)),
        ).to(self.device)

        self.lambda_dist = WarmupWeight(
            max_weight=float(loss_cfg.get("lambda_dist", 1.0)),
            warmup=int(loss_cfg.get("dist_warmup_epochs", 10)),
            delay=int(loss_cfg.get("dist_delay_epochs", 0)),
            granularity=loss_cfg.get("dist_warmup_granularity", "epoch"),
        )
        self.lambda_balance = float(loss_cfg.get("lambda_balance", 0.01))
        self.lambda_aux_cls = float(loss_cfg.get("lambda_aux_cls", 0.0))

        # --------------------------------------------------------- optimiser
        params = (
            self.model.student_parameters()
            if hasattr(self.model, "student_parameters")
            else [p for p in self.model.parameters() if p.requires_grad]
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        )
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            kind=train_cfg.get("lr_scheduler", "cosine"),
            epochs=self.epochs,
            warmup_epochs=int(train_cfg.get("lr_warmup_epochs", 0)),
            min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.01)),
            step_size=int(train_cfg.get("lr_step_size", 30)),
            gamma=float(train_cfg.get("lr_gamma", 0.5)),
        )
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):  # older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.epoch = 0
        self.global_step = 0
        self.best_metric = -1.0
        self.best_epoch = -1
        self.history: list = []

    # ------------------------------------------------------------------ step
    def _autocast(self):
        return torch.autocast(device_type=self.device.type, enabled=self.amp)

    def train_step(self, source_batch, target_batch) -> Dict[str, float]:
        x_s, y_s = source_batch
        x_s = x_s.to(self.device, non_blocking=True)
        y_s = y_s.to(self.device, non_blocking=True)

        x_t = None
        if not self.source_only and target_batch is not None:
            x_t = target_batch[0].to(self.device, non_blocking=True)

        reset_moe_stats(self.model)
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast():
            out = self.model(x_s=x_s, x_t=x_t, with_teacher=not self.source_only)

            loss_cls = self.cls_loss(out["logits_s"], y_s)

            dist_terms = self.dist_loss(out) if not self.source_only else {}
            loss_dist = dist_terms.get("dist", torch.zeros((), device=self.device))

            loss_balance = collect_moe_aux(self.model, device=self.device)

            weight_dist = self.lambda_dist.at(self.epoch, self.global_step)
            total = loss_cls + weight_dist * loss_dist + self.lambda_balance * loss_balance

            if self.lambda_aux_cls > 0 and "logits_s_from_t" in out:
                aux_cls = self.cls_loss(out["logits_s_from_t"], y_s)
                total = total + self.lambda_aux_cls * aux_cls
            else:
                aux_cls = torch.zeros((), device=self.device)

        self.scaler.scale(total).backward()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                self.grad_clip,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Teacher update happens after the optimiser step, on the new student weights.
        teacher_stats = {}
        if not self.source_only and getattr(self.model, "use_distillation", False):
            teacher_stats = self.model.update_teachers()

        self.global_step += 1

        with torch.no_grad():
            acc = (out["logits_s"].argmax(dim=1) == y_s).float().mean().item()

        stats = {
            "loss": float(total.detach()),
            "loss_cls": float(loss_cls.detach()),
            "loss_dist": float(loss_dist.detach()) if torch.is_tensor(loss_dist) else 0.0,
            "loss_balance": float(loss_balance.detach()),
            "loss_aux_cls": float(aux_cls.detach()),
            "lambda_dist": weight_dist,
            "source_acc": acc,
        }
        for key in ("dist_s_from_t", "dist_t_from_s"):
            if key in dist_terms and torch.is_tensor(dist_terms[key]):
                stats[key] = float(dist_terms[key].detach())
        stats.update(teacher_stats)
        return stats

    # ----------------------------------------------------------------- epoch
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = defaultdict(float)
        count = 0
        start = time.time()

        source_loader = self.loaders["source"]
        target_loader = self.loaders.get("target")

        for batch_idx, source_batch in enumerate(source_loader):
            target_batch = None
            if not self.source_only and target_loader is not None:
                target_batch = target_loader.next()

            stats = self.train_step(source_batch, target_batch)
            for key, value in stats.items():
                totals[key] += value
            count += 1

            if self.log_every and batch_idx % self.log_every == 0:
                self.logger.info(
                    f"epoch {self.epoch:3d} | batch {batch_idx:4d}/{len(source_loader)} | "
                    f"loss {stats['loss']:.4f} (cls {stats['loss_cls']:.4f}, "
                    f"dist {stats['loss_dist']:.4f}, bal {stats['loss_balance']:.4f}) | "
                    f"src acc {stats['source_acc'] * 100:.1f}%"
                )

        averaged = {key: value / max(count, 1) for key, value in totals.items()}
        averaged["epoch_time"] = time.time() - start
        averaged["lr"] = self.optimizer.param_groups[0]["lr"]
        return averaged

    # ------------------------------------------------------------------ loop
    def fit(self) -> Dict[str, Any]:
        self.logger.info(f"device: {self.device}")
        total_params = sum(p.numel() for p in self.model.student_parameters())
        self.logger.info(f"trainable parameters: {total_params:,}")
        self.logger.info(f"distillation schedule: {self.lambda_dist}")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            train_stats = self.train_epoch()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # The evaluation keys are always present so the CSV header, which is
            # fixed by the first row, covers epochs that are not evaluated.
            row: Dict[str, Any] = {
                "epoch": epoch,
                **{k: round(v, 6) for k, v in train_stats.items()},
                "target_OA": "",
                "target_AA": "",
                "target_Kappa": "",
                "source_val_OA": "",
            }

            if self.eval_every and (epoch + 1) % self.eval_every == 0:
                eval_metrics = self.evaluate_target()
                row.update(
                    {
                        "target_OA": eval_metrics["OA"],
                        "target_AA": eval_metrics["AA"],
                        "target_Kappa": eval_metrics["Kappa"],
                    }
                )
                if "source_val" in self.loaders:
                    val = evaluate(
                        self.model, self.loaders["source_val"], self.device,
                        self.num_classes, domain="source", amp=self.amp,
                    )
                    row["source_val_OA"] = val["OA"]

                self.logger.info(
                    f"epoch {epoch:3d} | train loss {train_stats['loss']:.4f} | "
                    f"src acc {train_stats['source_acc'] * 100:.2f}% | "
                    f"target OA {eval_metrics['OA'] * 100:.2f}% "
                    f"AA {eval_metrics['AA'] * 100:.2f}% "
                    f"Kappa {eval_metrics['Kappa'] * 100:.2f} | "
                    f"{train_stats['epoch_time']:.1f}s"
                )
                self._maybe_save_best(row)

            self.csv.log(row)
            self.history.append(row)
            self.save_checkpoint(self.output_dir / "last.pt")

        return self.finalise()

    def _selection_value(self, row: Dict[str, Any]) -> Optional[float]:
        key = {"target_oa": "target_OA", "source_val_oa": "source_val_OA"}.get(
            self.model_selection
        )
        value = row.get(key) if key else None
        return float(value) if isinstance(value, (int, float)) else None

    def _maybe_save_best(self, row: Dict[str, Any]) -> None:
        value = self._selection_value(row)
        if value is None:
            return
        if value > self.best_metric:
            self.best_metric = value
            self.best_epoch = int(row["epoch"])
            self.save_checkpoint(self.output_dir / "best.pt")

    # ------------------------------------------------------------ evaluation
    def evaluate_target(self) -> Dict[str, Any]:
        source_batch = None
        if getattr(self.model, "inference_mode", "A") == "B":
            patches, _ = next(iter(self.loaders["source"]))
            source_batch = patches.to(self.device)
        return evaluate(
            self.model,
            self.loaders["target_eval"],
            self.device,
            self.num_classes,
            domain="target",
            source_batch=source_batch,
            amp=self.amp,
        )

    def finalise(self) -> Dict[str, Any]:
        # Captured before evaluation so it reflects the last training epoch.
        usage = moe_usage(self.model)

        metrics = self.evaluate_target()
        self.logger.info("final target-domain results (last epoch):")
        self.logger.info("\n" + format_metrics(metrics, self.class_names))
        self.logger.info(
            "\n" + format_confusion_matrix(metrics["confusion_matrix"], self.class_names)
        )

        if usage:
            self.logger.info("MoE expert usage (last training epoch):")
            self.logger.info("\n" + summarise_expert_usage(usage))

        summary = {
            "last": {k: float(metrics[k]) for k in ("OA", "AA", "Kappa", "F1")},
            "per_class": np.asarray(metrics["per_class"], dtype=float).tolist(),
            "class_names": self.class_names,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "model_selection": self.model_selection,
            "epochs": self.epochs,
        }

        best_path = self.output_dir / "best.pt"
        if best_path.exists() and self.best_epoch >= 0:
            state = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            best_metrics = self.evaluate_target()
            self.logger.info(
                f"best checkpoint (epoch {self.best_epoch}): "
                f"OA {best_metrics['OA'] * 100:.2f}% AA {best_metrics['AA'] * 100:.2f}% "
                f"Kappa {best_metrics['Kappa'] * 100:.2f}"
            )
            summary["best"] = {k: float(best_metrics[k]) for k in ("OA", "AA", "Kappa", "F1")}

        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    # ----------------------------------------------------------- checkpoints
    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "config": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, resume: bool = True) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        if resume:
            self.optimizer.load_state_dict(state["optimizer"])
            self.epoch = int(state.get("epoch", 0)) + 1
            self.global_step = int(state.get("global_step", 0))
            self.best_metric = float(state.get("best_metric", -1.0))
            self.best_epoch = int(state.get("best_epoch", -1))
        self.logger.info(f"loaded checkpoint {path} (resume={resume})")
