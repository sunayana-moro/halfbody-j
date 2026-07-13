"""Composite conv blocks (the U-net body). torch: modules.py (+ lia-style ResBlock)."""

from __future__ import annotations

from flax import nnx


class ConvBlock(nnx.Module):
    """Conv -> NormLayer -> activation. torch: modules.ConvBlock."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 norm_type="batch", activation=True, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, use_running_average=None):
        raise NotImplementedError


class FeatResBlock(nnx.Module):
    """Two ConvBlocks + residual. torch: modules.FeatResBlock."""

    def __init__(self, channels, norm_type="batch", activation=True, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, use_running_average=None):
        raise NotImplementedError


class ResBlock(nnx.Module):
    """lia-style downsampling resblock (ConvLayer x3). torch: modules.ResBlock."""

    def __init__(self, in_ch, out_ch, blur_kernel=(1, 3, 3, 1), *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


class ConvResBlock(nnx.Module):
    """conv -> norm -> act -> conv -> FeatResBlock x2. torch: modules.ConvResBlock."""

    def __init__(self, in_ch, out_ch, norm_type="batch", *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, use_running_average=None):
        raise NotImplementedError


class DownConvResBlock(nnx.Module):
    """ConvResBlock + AvgPool2d downsample. torch: modules.DownConvResBlock."""

    def __init__(self, in_ch, out_ch, norm_type="batch", *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, use_running_average=None):
        raise NotImplementedError


class UpConvResBlock(nnx.Module):
    """nearest upsample -> conv blocks. torch: modules.UpConvResBlock."""

    def __init__(self, in_ch, out_ch, norm_type="batch", *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, use_running_average=None):
        raise NotImplementedError
