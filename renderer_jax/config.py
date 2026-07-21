"""Hyperparameters / dims for the renderer (replaces torch's argparse ``args``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RendererConfig:
    feature_dims: Tuple[int, ...] = (32, 64, 128, 256, 512, 512)
    spatial_dims: Tuple[int, ...] = (256, 128, 64, 32, 16, 8)
    num_heads: int = 8
    window_size: int = 8
    swin_res_threshold: int = 128
    latent_dim: int = 32
    const_dim: int = 32


DEFAULT = RendererConfig()
