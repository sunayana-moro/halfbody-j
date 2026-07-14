"""Composite conv blocks (the U-net body). torch: modules.py (+ lia-style ResBlock).


Conventions:
- **Layout is NHWC** (Flax-idiomatic): tensors are (batch, H, W, channels).
- Plain blocks activate with LeakyReLU at slope ``0.01`` — matching torch's
  ``nn.LeakyReLU`` default (the 0.2 slope only lives on the equalized-LR path).
- Batchnorm state is threaded via ``use_running_average`` (None → module default).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from renderer_jax.layers import NormLayer, ConvLayer

# torch nn.LeakyReLU default slope (modules.py blocks instantiate it bare)
_NEG_SLOPE = 0.01


def _leaky(x):
    return jax.nn.leaky_relu(x, _NEG_SLOPE)


def _avg_pool2x2(x):
    """Non-overlapping 2x2 mean, stride 2 (NHWC). torch: nn.AvgPool2d(2, 2)."""
    win = (1, 2, 2, 1)
    summed = jax.lax.reduce_window(x, 0.0, jax.lax.add, win, win, "VALID")
    return summed / 4.0


def _upsample_nearest2x(x):
    """Nearest-neighbour 2x upsample (NHWC). torch: nn.Upsample(2, 'nearest')."""
    return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)


class ConvBlock(nnx.Module):
    """Conv -> NormLayer -> (LeakyReLU)"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 norm_type="batch", activation=True, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_ch, out_ch, (kernel_size, kernel_size),
                             strides=stride, padding=padding, use_bias=False, rngs=rngs)
        self.norm = NormLayer(out_ch, norm_type, rngs=rngs)
        self.activation = activation

    def __call__(self, x, use_running_average=None):
        x = self.conv(x)
        x = self.norm(x, use_running_average=use_running_average)
        if self.activation:
            x = _leaky(x)
        return x


class FeatResBlock(nnx.Module):

    def __init__(self, channels, norm_type="batch", activation=True, *, rngs: nnx.Rngs):
        self.conv1 = ConvBlock(channels, channels, norm_type=norm_type, activation=True, rngs=rngs)
        self.conv2 = ConvBlock(channels, channels, norm_type=norm_type, activation=False, rngs=rngs)
        self.activation = activation

    def __call__(self, x, use_running_average=None):
        residual = x
        out = self.conv1(x, use_running_average=use_running_average)
        out = self.conv2(out, use_running_average=use_running_average)
        out = out + residual
        if self.activation:
            out = _leaky(out)
        return out


class ResBlock(nnx.Module):
    """lia-style downsampling resblock (ConvLayer x3). torch: modules.ResBlock.

    conv1(in->in) -> conv2(in->out, down); skip(in->out, down); out + skip.
    (No /sqrt(2) here — that's the discriminator's lia ResBlock, not this one.)
    """

    def __init__(self, in_ch, out_ch, blur_kernel=(1, 3, 3, 1), *, rngs: nnx.Rngs):
        self.conv1 = ConvLayer(in_ch, in_ch, 3, blur_kernel=blur_kernel, rngs=rngs)
        self.conv2 = ConvLayer(in_ch, out_ch, 3, downsample=True, blur_kernel=blur_kernel, rngs=rngs)
        self.skip = ConvLayer(in_ch, out_ch, 3, downsample=True, blur_kernel=blur_kernel, rngs=rngs)

    def __call__(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        return out + skip


class ConvResBlock(nnx.Module):
    """conv -> norm -> act -> conv -> FeatResBlock x2. torch: modules.ConvResBlock."""

    def __init__(self, in_ch, out_ch, norm_type="batch", *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.norm = NormLayer(out_ch, norm_type, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.feat1 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)
        self.feat2 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)

    def __call__(self, x, use_running_average=None):
        out = self.conv1(x)
        out = self.norm(out, use_running_average=use_running_average)
        out = _leaky(out)
        out = self.conv2(out)
        out = self.feat1(out, use_running_average=use_running_average)
        out = self.feat2(out, use_running_average=use_running_average)
        return out


class DownConvResBlock(nnx.Module):
    """conv -> norm -> act -> avgpool(2) -> conv -> FeatResBlock x2. torch: modules.DownConvResBlock."""

    def __init__(self, in_ch, out_ch, norm_type="batch", *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.norm = NormLayer(out_ch, norm_type, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.feat1 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)
        self.feat2 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)

    def __call__(self, x, use_running_average=None):
        out = self.conv1(x)
        out = self.norm(out, use_running_average=use_running_average)
        out = _leaky(out)
        out = _avg_pool2x2(out)
        out = self.conv2(out)
        out = self.feat1(out, use_running_average=use_running_average)
        out = self.feat2(out, use_running_average=use_running_average)
        return out


class UpConvResBlock(nnx.Module):
    """upsample(2, nearest) -> conv -> norm -> act -> conv -> FeatResBlock x2. torch: modules.UpConvResBlock."""

    def __init__(self, in_ch, out_ch, norm_type="batch", *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.norm = NormLayer(out_ch, norm_type, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.feat1 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)
        self.feat2 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)

    def __call__(self, x, use_running_average=None):
        out = _upsample_nearest2x(x)
        out = self.conv1(out)
        out = self.norm(out, use_running_average=use_running_average)
        out = _leaky(out)
        out = self.conv2(out)
        out = self.feat1(out, use_running_average=use_running_average)
        out = self.feat2(out, use_running_average=use_running_average)
        return out
