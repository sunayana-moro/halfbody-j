from __future__ import annotations
import math
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

# ------------------------------------------------------------------------------------------------------------
# Useful Functions
# ------------------------------------------------------------------------------------------------------------
def make_kernel(k):
    """1D or 2D list -> a normalized 2D numpy FIR kernel (sums to 1)."""
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    return k / k.sum()


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    """
    Upsample → apply a FIR (Finite Impulse Response) filter → Dnsample

    Input (H * W) -> Insert zeros (Upsample) -> Pad / Crop -> Convolve with FIR kernel -> Take every d-th pixel (Downsample) -> Output
    """
    B, H, W, C = x.shape
    kh, kw = kernel.shape

    # 1) upsample by zero-insertion:
    if up > 1:
        up_x = jnp.zeros((B, H * up, W * up, C), dtype=x.dtype)
        x = up_x.at[:, ::up, ::up, :].set(x)

    # 2) pad (can be negative -> crop, matching torch's pad/slice combo)
    p0, p1 = pad
    x = jnp.pad(x, ((0, 0), (max(p0, 0), max(p1, 0)), (max(p0, 0), max(p1, 0)), (0, 0)))
    Hc, Wc = x.shape[1], x.shape[2]
    x = x[:, max(-p0, 0):Hc - max(-p1, 0), max(-p0, 0):Wc - max(-p1, 0), :]

    # 3) depthwise TRUE convolution
    flipped = jnp.flip(jnp.asarray(kernel), (0, 1)).reshape(kh, kw, 1, 1)
    filt = jnp.broadcast_to(flipped, (kh, kw, 1, C))
    x = jax.lax.conv_general_dilated(
        x, filt, window_strides=(1, 1), padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=C,
    )

    # 4) downsample by strided slicing
    return x[:, ::down, ::down, :]

def fused_leaky_relu(x, bias, negative_slope=0.2, scale=2 ** 0.5):
    """leaky_relu(x + bias) * scale. torch: lia_resblocks.fused_leaky_relu."""
    return jax.nn.leaky_relu(x + bias, negative_slope) * scale

# ---------------------------------------------------------------------------------------------------
# EqualLinear (Equalised Learning Rate Linear Layer)
# ---------------------------------------------------------------------------------------------------
class EqualLinear(nnx.Module):
    """
    Linear layer with equalized learning rate: weight scaled at runtime by gain/√fan_in.
    -> like every layer trains at same effective rate regardless of fan-in
    """

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0.0, lr_mul=1.0,
                 activation=None, *, rngs: nnx.Rngs):
        w = jax.random.normal(rngs.params(), (in_dim, out_dim)) / lr_mul
        self.weight = nnx.Param(w)
        self.bias = nnx.Param(jnp.full((out_dim,), bias_init)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def __call__(self, x):
        if self.activation:
            out = x @ (self.weight.value * self.scale)
            out = fused_leaky_relu(out, self.bias.value * self.lr_mul)
        else:
            out = x @ (self.weight.value * self.scale) + self.bias.value * self.lr_mul
        return out

# ---------------------------------------------------------------------------------------------------
# EqualConv2d (Equalised Learning Rate Conv2d)
# ---------------------------------------------------------------------------------------------------
class EqualConv2d(nnx.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True,
                 *, rngs: nnx.Rngs):
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)

        w = jax.random.normal(rngs.params(), (kernel_size, kernel_size, in_ch, out_ch))
        self.weight = nnx.Param(w)
        self.bias = nnx.Param(jnp.zeros((out_ch,))) if bias else None

    def __call__(self, x):
        w = self.weight.value * self.scale
        out = jax.lax.conv_general_dilated(
            x, w, window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding)] * 2,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias is not None:
            out = out + self.bias.value.reshape(1, 1, 1, -1)
        return out

# ---------------------------------------------------------------------------------------------------
# Fused-Leaky-ReLU
# ---------------------------------------------------------------------------------------------------
class FusedLeakyReLU(nnx.Module):
    """
    ReLU -> max(0,x)
    LeakyReLU -> max(x,alpha*x)
    FusedLeakyReLU -> root(2)*LeakyReLU(x+b)
    """

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, *, rngs: nnx.Rngs):
        self.bias = nnx.Param(jnp.zeros((1, 1, 1, channel)))
        self.negative_slope = negative_slope
        self.scale = scale

    def __call__(self, x):
        return fused_leaky_relu(x, self.bias.value, self.negative_slope, self.scale)


# ---------------------------------------------------------------------------------------------------
# Blur (a fixed low-pass filter i.e blur on image or feature map)
# ---------------------------------------------------------------------------------------------------
class Blur(nnx.Module):
    """
    -> In StyleGan2, before and after changing resolution, we need blur.
    -> Let say, after upsampling, Nearest-neighbor upsampling creates blocky artifacts, so applying blur smooths these transitions.
    -> Similarly, before downsampling, blur removes high-frequency content to prevent aliasing.
    """

    def __init__(self, kernel, pad, upsample_factor=1):
        k = make_kernel(kernel)
        if upsample_factor > 1:
            k = k * (upsample_factor ** 2)
        self.kernel = k          # numpy constant, NOT nnx.Param --> never trained
        self.pad = pad

    def __call__(self, x):
        return upfirdn2d(x, self.kernel, pad=self.pad)

# ---------------------------------------------------------------------------------------------------
# NoiseInjection (Injects per-pixel random noise into the input feature map.)
# ---------------------------------------------------------------------------------------------------
class NoiseInjection(nnx.Module):
    """
    -> The learned weight allows the network to control how much noise is injected.
    -> If no noise is provided, the input is returned unchanged (identity operation).
    """

    def __init__(self, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.zeros((1,)))

    def __call__(self, image, noise=None):
        if noise is None:
            return image
        return image + self.weight.value * noise
# ---------------------------------------------------------------------------------------------------
# NormLayers
# ---------------------------------------------------------------------------------------------------
class NormLayer(nnx.Module):
    def __init__(self, num_features, norm_type="batch", *, rngs: nnx.Rngs):
        self.norm_type = norm_type
        if norm_type == "batch":
            self.norm = nnx.BatchNorm(num_features, rngs=rngs)

        elif norm_type == "instance":
            self.norm = nnx.GroupNorm(num_features, num_groups=num_features,
                                      use_scale=False, use_bias=False, rngs=rngs)
        elif norm_type == "layer":
            self.norm = nnx.GroupNorm(num_features, num_groups=1, rngs=rngs)

        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def __call__(self, x, use_running_average=None):
        if self.norm_type == "batch":
            return self.norm(x, use_running_average=use_running_average)
        return self.norm(x)

# ---------------------------------------------------------------------------------------------------
# ConvLayer -- composes Blur(if downsample) + EqualConv2d + activation
# ---------------------------------------------------------------------------------------------------
class _ScaledLeakyReLU(nnx.Module):
    """
    Plain leaky_relu, no learned bias.
    If bias present, then we use FusedLeakyReLU else ScaledLeakyReLU
    """

    def __init__(self, negative_slope=0.2):
        self.negative_slope = negative_slope

    def __call__(self, x):
        return jax.nn.leaky_relu(x, self.negative_slope)

class ConvLayer(nnx.Module):
    """
    This convolution layer involves: Blur(if downsample) + EqualConv2d + activation 
    """

    def __init__(self, in_ch, out_ch, kernel_size, downsample=False,
                 blur_kernel=(1, 3, 3, 1), bias=True, activate=True, *, rngs: nnx.Rngs):
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0, pad1 = (p + 1) // 2, p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
            stride, padding = 2, 0
        else:
            self.blur = None
            stride, padding = 1, kernel_size // 2

        self.conv = EqualConv2d(in_ch, out_ch, kernel_size, stride=stride,
                                padding=padding, bias=bias and not activate, rngs=rngs)

        if activate:
            self.act = FusedLeakyReLU(out_ch, rngs=rngs) if bias else _ScaledLeakyReLU(0.2)
        else:
            self.act = None

    def __call__(self, x):
        if self.blur is not None:
            x = self.blur(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x