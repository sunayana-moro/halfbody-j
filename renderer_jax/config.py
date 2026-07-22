from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RendererConfig:
    # -----------------------------------------------------------------------------------
    # Feature pyramid (IdentityEncoder output + SynthesisNetwork + imt)
    # -----------------------------------------------------------------------------------
    feature_dims: Tuple[int, ...] = (32, 64, 128, 256, 512, 512)
    spatial_dims: Tuple[int, ...] = (256, 128, 64, 32, 16, 8)

    # -----------------------------------------------------------------------------------
    # IdentityEncoder (appearance)
    # -----------------------------------------------------------------------------------
    id_enc_init: int = 32        # initial 7x7 conv channels
    id_dim: int = 512            # identity vector dim (== IdentidyAdaptive dim_app)

    # -----------------------------------------------------------------------------------
    # MotionEncoder (motion)
    # -----------------------------------------------------------------------------------
    motion_enc_init: int = 64                                        # conv1 channels
    motion_enc_channels: Tuple[int, ...] = (128, 256, 512, 512, 512)  # ResBlock stages

    # -----------------------------------------------------------------------------------
    # motion latent / decoder
    # -----------------------------------------------------------------------------------
    latent_dim: int = 32         # motion latent (MotionEncoder dm, MotionDecoder + adapt dim_mot)
    const_dim: int = 32          # MotionDecoder learned constant channels

    # -----------------------------------------------------------------------------------
    # IdentidyAdaptive
    # -----------------------------------------------------------------------------------
    adapt_depth: int = 4         # number of hidden EqualLinear layers

    # -----------------------------------------------------------------------------------
    # Attention
    # -----------------------------------------------------------------------------------
    num_heads: int = 8
    window_size: int = 8
    swin_res_threshold: int = 128


DEFAULT = RendererConfig()
