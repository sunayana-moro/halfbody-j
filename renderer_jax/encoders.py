"""Appearance & motion encoders. torch: models.IdentityEncoder / MotionEncoder."""

from __future__ import annotations

from flax import nnx


class IdentityEncoder(nnx.Module):
    """Multi-scale appearance features + identity vector. torch: models.IdentityEncoder.

    QUIRK to reproduce: silently drops the 32 in output_channels (builds 64->512).
    """

    def __init__(self, in_channels=3, output_channels=(64, 128, 256, 512, 512, 512),
                 initial_channels=32, dm=512, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, use_running_average=None):
        raise NotImplementedError


class MotionEncoder(nnx.Module):
    """Per-frame motion latent. torch: models.MotionEncoder."""

    def __init__(self, initial_channels=64, output_channels=(128, 256, 512, 512, 512),
                 dm=32, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError
