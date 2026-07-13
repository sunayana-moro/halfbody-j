"""Leaf nnx.Modules that own parameters. torch: lia_resblocks + modules.NormLayer."""

from __future__ import annotations

from flax import nnx


class EqualLinear(nnx.Module):
    """Equalized-LR linear (scale applied at call time). torch: lia_resblocks.EqualLinear."""

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0.0, lr_mul=1.0,
                 activation=None, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


class EqualConv2d(nnx.Module):
    """Equalized-LR conv (weight * scale at call time). torch: lia_resblocks.EqualConv2d."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True,
                 *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


class FusedLeakyReLU(nnx.Module):
    """Per-channel bias then leaky_relu * sqrt(2). torch: lia_resblocks.FusedLeakyReLU."""

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


class Blur(nnx.Module):
    """FIR blur over a non-learned kernel constant. torch: lia_resblocks.Blur."""

    def __init__(self, kernel, pad, upsample_factor=1):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


class NoiseInjection(nnx.Module):
    """image + weight * noise (noise=None -> passthrough). torch: lia_resblocks.NoiseInjection."""

    def __init__(self, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, image, noise=None):
        raise NotImplementedError


class NormLayer(nnx.Module):
    """batch | instance | layer(group) norm dispatch. torch: modules.NormLayer."""

    def __init__(self, num_features, norm_type="batch", *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, use_running_average=None):
        raise NotImplementedError


class ConvLayer(nnx.Module):
    """Blur(if downsample) + EqualConv2d + activation. torch: lia_resblocks.ConvLayer."""

    def __init__(self, in_ch, out_ch, kernel_size, downsample=False,
                 blur_kernel=(1, 3, 3, 1), bias=True, activate=True, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError
