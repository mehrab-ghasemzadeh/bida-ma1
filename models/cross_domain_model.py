"""The complete cross-domain model (section 30 of the specification).

    X -> semantic tokenizer -> 5 tokens -> +positional encoding
      -> domain Transformer (+ Top-2 MoE) -> H -> mean pool -> h

The source branch feeds the classifier. The coupled Transformer exchanges
information between H_s and H_t in both directions, and the resulting coupled
representations are pulled towards the HMA teachers of the two independent
branches by the distillation loss.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .classifier import Classifier
from .coupled_transformer import CoupledTransformer
from .hma_teacher import TeacherBranch
from .positional_encoding import build_positional_encoding
from .semantic_tokenizer import SemanticTokenizer
from .transformer_encoder import TransformerEncoder, pool_tokens


class DomainBranch(nn.Module):
    """Tokenizer + positional encoding + Transformer encoder for one domain."""

    def __init__(self, tokenizer: nn.Module, pos_enc: nn.Module, encoder: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer
        self.pos_enc = pos_enc
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, S, H, W] -> token sequence H: [B, num_tokens, D]."""
        tokens = self.tokenizer(x)
        tokens = self.pos_enc(tokens)
        return self.encoder(tokens)


class CrossDomainModel(nn.Module):
    """Config-driven assembly of the whole architecture.

    Ablation switches (all under the model config):
        use_moe          - MoE sub-layers instead of a plain FFN.
        use_coupled      - the bidirectional coupled Transformer.
        use_distillation - the teacher/student representation distillation.
        teacher_mode     - hma | ema | frozen | copy.
        share_tokenizer / share_positional / share_encoder - parameter sharing
                           between the source and target branches.
    """

    def __init__(self, cfg: Dict[str, Any], num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        dim = int(cfg.get("embed_dim", 64))
        num_tokens = int(cfg.get("num_tokens", 5))
        patch_size = int(cfg.get("patch_size", 13))
        num_bands = int(cfg.get("num_bands", 13))
        dropout = float(cfg.get("dropout", 0.1))

        self.dim = dim
        self.num_tokens = num_tokens
        self.pool_mode = cfg.get("pool", "mean")
        self.use_coupled = bool(cfg.get("use_coupled", True))
        self.use_distillation = bool(cfg.get("use_distillation", True)) and self.use_coupled
        self.inference_mode = cfg.get("inference_mode", "A").upper()
        self.aux_cls_on_coupled = bool(cfg.get("aux_cls_on_coupled", False))

        moe_cfg = dict(cfg.get("moe", {}) or {})
        moe_cfg.setdefault("use_moe", cfg.get("use_moe", True))
        moe_cfg.setdefault("dropout", dropout)
        self.moe_cfg = moe_cfg

        tok_cfg = dict(cfg.get("tokenizer", {}) or {})

        def make_tokenizer() -> SemanticTokenizer:
            return SemanticTokenizer(
                patch_size=patch_size,
                num_bands=num_bands,
                embed_dim=dim,
                num_tokens=num_tokens,
                conv3d_channels=tuple(tok_cfg.get("conv3d_channels", (8, 16))),
                kernel3d=int(tok_cfg.get("kernel3d", 2)),
                kernel2d=int(tok_cfg.get("kernel2d", 2)),
                dropout=float(tok_cfg.get("dropout", 0.0)),
                softmax_over=tok_cfg.get("softmax_over", "spatial"),
            )

        def make_pos() -> nn.Module:
            return build_positional_encoding(
                num_tokens, dim, cfg.get("positional_encoding", "learned"), dropout=0.0
            )

        def make_encoder() -> TransformerEncoder:
            return TransformerEncoder(
                dim=dim,
                depth=int(cfg.get("depth", 2)),
                num_heads=int(cfg.get("num_heads", 4)),
                dropout=dropout,
                attn_dropout=float(cfg.get("attn_dropout", dropout)),
                ffn_cfg=moe_cfg,
                norm_style=cfg.get("norm_style", "post"),
            )

        src_tokenizer, src_pos, src_encoder = make_tokenizer(), make_pos(), make_encoder()
        tgt_tokenizer = src_tokenizer if cfg.get("share_tokenizer", False) else make_tokenizer()
        tgt_pos = src_pos if cfg.get("share_positional", False) else make_pos()
        tgt_encoder = src_encoder if cfg.get("share_encoder", False) else make_encoder()

        self.source_branch = DomainBranch(src_tokenizer, src_pos, src_encoder)
        self.target_branch = DomainBranch(tgt_tokenizer, tgt_pos, tgt_encoder)

        if self.use_coupled:
            self.coupled = CoupledTransformer(
                dim=dim,
                depth=int(cfg.get("coupled_depth", 1)),
                num_heads=int(cfg.get("num_heads", 4)),
                dropout=dropout,
                attn_dropout=float(cfg.get("attn_dropout", dropout)),
                ffn_cfg=moe_cfg,
                norm_style=cfg.get("norm_style", "post"),
                store_attention=bool(cfg.get("store_attention", False)),
            )
        else:
            self.coupled = None

        self.classifier = Classifier(
            dim=dim,
            num_classes=num_classes,
            hidden_ratio=float(cfg.get("classifier_hidden_ratio", 0.5)),
            dropout=dropout,
        )

        self.source_teacher: Optional[TeacherBranch] = None
        self.target_teacher: Optional[TeacherBranch] = None
        if self.use_distillation:
            teacher_cfg = dict(cfg.get("teacher", {}) or {})
            mode = teacher_cfg.get("mode", "hma")
            self.source_teacher = TeacherBranch(
                self.source_branch,
                mode=mode,
                window=int(teacher_cfg.get("window", 16)),
                ema_decay=float(teacher_cfg.get("ema_decay", 0.999)),
                update_every=int(teacher_cfg.get("update_every", 1)),
                history_device=teacher_cfg.get("history_device", "same"),
            )
            self.target_teacher = TeacherBranch(
                self.target_branch,
                mode=mode,
                window=int(teacher_cfg.get("window", 16)),
                ema_decay=float(teacher_cfg.get("ema_decay", 0.999)),
                update_every=int(teacher_cfg.get("update_every", 1)),
                history_device=teacher_cfg.get("history_device", "same"),
            )

    # --------------------------------------------------------------- helpers
    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return pool_tokens(tokens, self.pool_mode)

    def student_parameters(self):
        """Parameters that receive gradients (everything except the teachers)."""
        teacher_ids = set()
        for teacher in (self.source_teacher, self.target_teacher):
            if teacher is not None:
                teacher_ids.update(id(p) for p in teacher.parameters())
        return [p for p in self.parameters() if id(p) not in teacher_ids and p.requires_grad]

    # --------------------------------------------------------------- forward
    def encode_source(self, x_s: torch.Tensor) -> torch.Tensor:
        return self.source_branch(x_s)

    def encode_target(self, x_t: torch.Tensor) -> torch.Tensor:
        return self.target_branch(x_t)

    def forward(
        self,
        x_s: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        with_teacher: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run whichever branches the given inputs allow.

        Returns a dictionary with (depending on the inputs and the configuration):
            H_s, h_s, logits_s      - source branch and its classification logits
            H_t, h_t, logits_t      - target branch and its logits under Option A
            C_s_from_t, h_s_from_t  - coupled student, source queries target
            C_t_from_s, h_t_from_s  - coupled student, target queries source
            h_s_teacher, h_t_teacher- detached HMA teacher representations
        """
        out: Dict[str, torch.Tensor] = {}

        if x_s is not None:
            h_tokens_s = self.encode_source(x_s)
            out["H_s"] = h_tokens_s
            out["h_s"] = self.pool(h_tokens_s)
            out["logits_s"] = self.classifier(out["h_s"])

        if x_t is not None:
            h_tokens_t = self.encode_target(x_t)
            out["H_t"] = h_tokens_t
            out["h_t"] = self.pool(h_tokens_t)
            out["logits_t"] = self.classifier(out["h_t"])

        if self.use_coupled and x_s is not None and x_t is not None:
            c_s, c_t = self.coupled(out["H_s"], out["H_t"])
            out["C_s_from_t"], out["C_t_from_s"] = c_s, c_t
            out["h_s_from_t"] = self.pool(c_s)
            out["h_t_from_s"] = self.pool(c_t)
            if self.aux_cls_on_coupled:
                out["logits_s_from_t"] = self.classifier(out["h_s_from_t"])
                out["logits_t_from_s"] = self.classifier(out["h_t_from_s"])

        if with_teacher and self.use_distillation:
            with torch.no_grad():
                if x_s is not None and self.source_teacher is not None:
                    out["h_s_teacher"] = self.pool(self.source_teacher(x_s)).detach()
                if x_t is not None and self.target_teacher is not None:
                    out["h_t_teacher"] = self.pool(self.target_teacher(x_t)).detach()

        return out

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_target(
        self, x_t: torch.Tensor, x_s: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Target-domain logits.

        Option A (default) classifies the independent target representation h_t and
        needs no source data. Option B classifies the coupled representation
        h_{t<-s} and therefore requires a batch of source tokens.
        """
        tokens_t = self.encode_target(x_t)
        if self.inference_mode == "B":
            if not self.use_coupled:
                raise RuntimeError("Option B inference requires the coupled Transformer")
            if x_s is None:
                raise ValueError("Option B inference requires source patches (x_s)")
            tokens_s = self.encode_source(x_s)
            if tokens_s.shape[0] != tokens_t.shape[0]:
                repeat = (tokens_t.shape[0] + tokens_s.shape[0] - 1) // tokens_s.shape[0]
                tokens_s = tokens_s.repeat(repeat, 1, 1)[: tokens_t.shape[0]]
            _, c_t = self.coupled(tokens_s, tokens_t)
            return self.classifier(self.pool(c_t))
        return self.classifier(self.pool(tokens_t))

    # --------------------------------------------------------------- teacher
    @torch.no_grad()
    def update_teachers(self) -> Dict[str, float]:
        """Push the current student weights through the HMA/EMA teacher update."""
        stats: Dict[str, float] = {}
        if self.source_teacher is not None:
            self.source_teacher.update(self.source_branch)
            stats["teacher_drift_s"] = self.source_teacher.drift(self.source_branch)
        if self.target_teacher is not None:
            self.target_teacher.update(self.target_branch)
            stats["teacher_drift_t"] = self.target_teacher.drift(self.target_branch)
        return stats


def build_model(cfg: Dict[str, Any], num_classes: int) -> CrossDomainModel:
    return CrossDomainModel(cfg, num_classes)
