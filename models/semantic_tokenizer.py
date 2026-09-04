"""3D/2D CNN semantic tokenizer (Steps 1-2 of the specification).

Turns a hyperspectral patch [B, 1, S, H, W] into exactly `num_tokens` semantic
tokens [B, num_tokens, D]:

    Conv3D(2,2,2) -> BN -> LeakyReLU
    Conv3D(2,2,2) -> BN -> LeakyReLU
    reshape [B, C, S', H', W'] -> [B, C*S', H', W']
    Conv2D(2,2)   -> BN -> LeakyReLU
    learned attention pooling into `num_tokens` tokens
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTokenPooling(nn.Module):
    """Learned attention-based semantic token pooling.

    Given a feature map flattened to F' in R^{N x C} (N = H'*W'), a learned
    projection produces `num_tokens` attention maps A in R^{N x num_tokens} and the
    tokens are Z = A^T F' in R^{num_tokens x C}.

    The softmax is taken over the N spatial positions, so each token is a convex
    combination of spatial features (`softmax_over="spatial"`, the default and the
    behaviour of the semantic tokenizer this is modelled on). `softmax_over="token"`
    normalises across the tokens instead, i.e. each spatial position is distributed
    over the tokens; it is provided for ablation.
    """

    def __init__(self, in_channels: int, num_tokens: int = 5, softmax_over: str = "spatial"):
        super().__init__()
        if softmax_over not in {"spatial", "token"}:
            raise ValueError(f"Unknown softmax_over: {softmax_over}")
        self.num_tokens = num_tokens
        self.softmax_over = softmax_over
        self.attend = nn.Linear(in_channels, num_tokens, bias=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, C, H, W] -> tokens [B, num_tokens, C] (attention maps cached)."""
        b, c, h, w = feat.shape
        flat = feat.flatten(2).transpose(1, 2)               # [B, N, C], N = H*W
        logits = self.attend(flat)                           # [B, N, num_tokens]
        dim = 1 if self.softmax_over == "spatial" else 2
        attn = F.softmax(logits, dim=dim)
        self.last_attention = attn.detach()                  # [B, N, num_tokens]
        tokens = attn.transpose(1, 2) @ flat                 # [B, num_tokens, C]
        return tokens


class SemanticTokenizer(nn.Module):
    """CNN tokenizer producing [B, num_tokens, embed_dim] from [B, 1, S, H, W]."""

    def __init__(
        self,
        patch_size: int = 13,
        num_bands: int = 13,
        embed_dim: int = 64,
        num_tokens: int = 5,
        conv3d_channels: tuple[int, int] = (8, 16),
        kernel3d: int = 2,
        kernel2d: int = 2,
        dropout: float = 0.0,
        negative_slope: float = 0.01,
        softmax_over: str = "spatial",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        c1, c2 = conv3d_channels
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=kernel3d),
            nn.BatchNorm3d(c1),
            nn.LeakyReLU(negative_slope),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=kernel3d),
            nn.BatchNorm3d(c2),
            nn.LeakyReLU(negative_slope),
        )

        # Shapes after two valid 3D convolutions with unit stride.
        spectral_out = num_bands - 2 * (kernel3d - 1)
        spatial_out = patch_size - 2 * (kernel3d - 1)
        if spectral_out < 1 or spatial_out < kernel2d:
            raise ValueError(
                f"patch_size={patch_size} / num_bands={num_bands} are too small for "
                f"two {kernel3d}^3 convolutions followed by a {kernel2d}^2 convolution"
            )

        self.conv2d = nn.Sequential(
            nn.Conv2d(c2 * spectral_out, embed_dim, kernel_size=kernel2d),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(negative_slope),
        )
        self.spatial_out = spatial_out - (kernel2d - 1)
        self.num_spatial = self.spatial_out**2

        self.pool = SemanticTokenPooling(embed_dim, num_tokens, softmax_over)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, S, H, W] (or [B, S, H, W]) -> [B, num_tokens, embed_dim]."""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected a 4D or 5D input tensor, got shape {tuple(x.shape)}")

        feat = self.conv3d_1(x)                              # [B, c1, S-1, H-1, W-1]
        feat = self.conv3d_2(feat)                           # [B, c2, S-2, H-2, W-2]

        b, c, s, h, w = feat.shape
        feat = feat.reshape(b, c * s, h, w)                  # spectral depth -> channels
        feat = self.conv2d(feat)                             # [B, D, H', W']

        tokens = self.pool(feat)                             # [B, num_tokens, D]
        return self.dropout(tokens)
