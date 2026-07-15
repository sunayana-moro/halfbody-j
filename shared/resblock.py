from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from shared.layers import NormLayer, ConvLayer

_NEG_SLOPE = 0.01  


# ---------------------------------------------------------------------------
# pure helpers (no params)
# ---------------------------------------------------------------------------
def _leaky(x):
    return jax.nn.leaky_relu(x, _NEG_SLOPE)


def _avg_pool2x2(x):
    """Non-overlapping 2x2 mean, stride 2 (NHWC). Converted from: torch: nn.AvgPool2d(2, 2)."""
    win = (1, 2, 2, 1)
    return jax.lax.reduce_window(x, 0.0, jax.lax.add, win, win, "VALID") / 4.0


def _upsample_nearest2x(x):
    """Nearest-neighbour 2x upsample (NHWC). Converted from: torch: nn.Upsample(2, 'nearest')."""
    return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)

# ---------------------------------------------------------------------------
# ConvBlock : Conv(bias=False) -> NormLayer -> (LeakyReLU)
# ---------------------------------------------------------------------------
class ConvBlock(nnx.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 norm_type="batch", activation=True, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_ch, out_ch, (kernel_size, kernel_size),
                             strides=stride, padding=padding, use_bias=False, rngs=rngs)
        self.norm = NormLayer(out_ch, norm_type, rngs=rngs)
        self.activation = activation

    def __call__(self, x, use_running_average=None):
        x = self.conv(x)
        x = self.norm(x, use_running_average=use_running_average)
        return _leaky(x) if self.activation else x

# --------------------------------------------------------------------------------------------------
# FeatResBlock : ConvBlock(activation) -> ConvBlock(no activation) -> +residual -> (activation)
# --------------------------------------------------------------------------------------------------
class FeatResBlock(nnx.Module):
    def __init__(self, channels, norm_type="batch",dropout_rate=0, activation=True, *, rngs: nnx.Rngs):
        self.conv1 = ConvBlock(channels, channels, norm_type=norm_type, activation=True, rngs=rngs)
        self.conv2 = ConvBlock(channels, channels, norm_type=norm_type, activation=False, rngs=rngs)
        self.activation = activation

    def __call__(self, x, use_running_average=None):
        out = self.conv1(x, use_running_average=use_running_average)
        out = self.conv2(out, use_running_average=use_running_average)
        out = out + x                                   # residual
        return _leaky(out) if self.activation else out

# ---------------------------------------------------------------------------
# ResBlock : ConvLayer x3 (used by MotionEncoder)
# ---------------------------------------------------------------------------
class ResBlock(nnx.Module):
    def __init__(self, in_ch, out_ch,dropout_rate=0, blur_kernel=(1, 3, 3, 1), *, rngs: nnx.Rngs):
        self.conv1 = ConvLayer(in_ch, in_ch, 3, blur_kernel=blur_kernel, rngs=rngs)
        self.conv2 = ConvLayer(in_ch, out_ch, 3, downsample=True, blur_kernel=blur_kernel, rngs=rngs)
        self.skip = ConvLayer(in_ch, out_ch, 3, downsample=True, blur_kernel=blur_kernel, rngs=rngs)

    def __call__(self, x):
        out = self.conv2(self.conv1(x))
        return out + self.skip(x)
    
# ---------------------------------------------------------------------------
# ConvResBlock : conv -> norm -> activation -> conv -> FeatResBlock x2 
# (used by SynthesisNetwork)
# ---------------------------------------------------------------------------
class ConvResBlock(nnx.Module):
    def __init__(self, in_ch, out_ch,dropout_rate=0, norm_type="batch", *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.norm = NormLayer(out_ch, norm_type, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.feat_res_block1 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)
        self.feat_res_block2 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)

    def __call__(self, x, use_running_average=None):
        out = _leaky(self.norm(self.conv1(x), use_running_average=use_running_average))
        out = self.conv2(out)
        out = self.feat_res_block1(out, use_running_average=use_running_average)
        return self.feat_res_block2(out, use_running_average=use_running_average)

# ---------------------------------------------------------------------------
# DownConvResBlock : conv -> norm -> activation -> avgpool(2) -> conv -> FeatResBlock x2
# (used by IdentityEncoder) , AvgPool AFTER activation
# ---------------------------------------------------------------------------
class DownConvResBlock(nnx.Module):
    def __init__(self, in_ch, out_ch,dropout_rate=0, norm_type="batch", *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.norm = NormLayer(out_ch, norm_type, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.feat_res_block1 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)
        self.feat_res_block2 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)

    def __call__(self, x, use_running_average=None):
        out = _leaky(self.norm(self.conv1(x), use_running_average=use_running_average))
        out = _avg_pool2x2(out)
        out = self.conv2(out)
        out = self.feat_res_block1(out, use_running_average=use_running_average)
        return self.feat_res_block2(out, use_running_average=use_running_average)

# ---------------------------------------------------------------------------
# UpConvResBlock : upsample(2, nearest) -> conv -> norm -> activation -> conv -> FeatResBlock x2
# (used by SynthesisNetwork)
# ---------------------------------------------------------------------------
class UpConvResBlock(nnx.Module):
    def __init__(self, in_ch, out_ch, norm_type="batch", *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.norm = NormLayer(out_ch, norm_type, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), strides=1, padding=1, use_bias=False, rngs=rngs)
        self.feat_res_block1 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)
        self.feat_res_block2 = FeatResBlock(out_ch, norm_type=norm_type, rngs=rngs)

    def __call__(self, x, use_running_average=None):
        out = _upsample_nearest2x(x)
        out = _leaky(self.norm(self.conv1(out), use_running_average=use_running_average))
        out = self.conv2(out)
        out = self.feat_res_block1(out, use_running_average=use_running_average)
        return self.feat_res_block2(out, use_running_average=use_running_average)