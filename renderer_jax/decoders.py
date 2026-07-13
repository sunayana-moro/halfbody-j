"""Motion decoder & synthesis network. torch: models.MotionDecoder / SynthesisNetwork."""

from __future__ import annotations

from flax import nnx


class MotionDecoder(nnx.Module):
    """Learned const + StyledConv x13 -> 4 motion maps. torch: models.MotionDecoder."""

    def __init__(self, latent_dim=32, const_dim=32, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, t):
        raise NotImplementedError


class SynthesisNetwork(nnx.Module):
    """Upconv + resblocks + self-attn -> RGB frame. torch: models.SynthesisNetwork."""

    def __init__(self, config, feature_dims, spatial_dims, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, features_align, use_running_average=None):
        raise NotImplementedError
