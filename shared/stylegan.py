from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from shared.layers import EqualLinear, Blur, NoiseInjection, FusedLeakyReLU

# -----------------------------------------------------------------------------------------------------
# ModulatedConv2d 
# -----------------------------------------------------------------------------------------------------
class ModulatedConv2d(nnx.Module):
    """
    - Modulates convolution weights using a per-sample style vector.
    - Optionally demodulates weights to normalize feature magnitudes.
    - Supports standard, upsampling, and downsampling convolutions.
    - Applies anti-aliasing blur during up/downsampling.
    - Performs grouped convolution so each sample uses its own modulated weights.
    
    Input:
        x: Input feature map of shape (B, H, W, Cin).
        style: Style vector of shape (B, style_dim).

    Returns:
        Output feature map of shape (B, H_out, W_out, Cout).
    """

    def __init__(self, in_ch, out_ch, kernel_size, style_dim, demodulate=True,
                 upsample=False, downsample=False, blur_kernel=(1, 3, 3, 1), *, rngs: nnx.Rngs):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (kernel_size, kernel_size, in_ch, out_ch)))
        self.modulation = EqualLinear(style_dim, in_ch, bias_init=1.0, rngs=rngs)

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
        elif downsample:  # never used though
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
        else:
            self.blur = None

    def __call__(self, x, style):
        B, H, W, Cin = x.shape
        Cout, k = self.out_ch, self.kernel_size

        s = self.modulation(style)                                  # (B, Cin)
        w = self.scale * self.weight.value[None] * s[:, None, None, :, None]   # (B, k, k, Cin, Cout)
        if self.demodulate:
            demod = jax.lax.rsqrt(jnp.sum(w ** 2, axis=(1, 2, 3)) + 1e-8)      # (B, Cout)
            w = w * demod[:, None, None, None, :]

        if self.upsample:
            xr = jnp.transpose(x, (1, 2, 0, 3)).reshape(1, H, W, B * Cin)   # fold batch->channels
            wf = w[:, ::-1, ::-1, :, :]                             # rot180 spatial
            wk = jnp.transpose(wf, (1, 2, 3, 0, 4)).reshape(k, k, Cin, B * Cout)
            out = jax.lax.conv_general_dilated(
                xr, wk, window_strides=(1, 1), padding=[(k - 1, k - 1)] * 2,
                lhs_dilation=(2, 2), dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=B)
        elif self.downsample:
            x = self.blur(x)
            Hb, Wb = x.shape[1], x.shape[2]
            xr = jnp.transpose(x, (1, 2, 0, 3)).reshape(1, Hb, Wb, B * Cin)
            wk = jnp.transpose(w, (1, 2, 3, 0, 4)).reshape(k, k, Cin, B * Cout)
            out = jax.lax.conv_general_dilated(
                xr, wk, window_strides=(2, 2), padding="VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=B)
        else:
            xr = jnp.transpose(x, (1, 2, 0, 3)).reshape(1, H, W, B * Cin)   # fold batch->channels
            wk = jnp.transpose(w, (1, 2, 3, 0, 4)).reshape(k, k, Cin, B * Cout)
            out = jax.lax.conv_general_dilated(
                xr, wk, window_strides=(1, 1), padding=[(k // 2, k // 2)] * 2,
                dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=B)

        oh, ow = out.shape[1], out.shape[2]
        out = jnp.transpose(out.reshape(oh, ow, B, Cout), (2, 0, 1, 3))   # (B, oh, ow, Cout)

        if self.upsample:
            out = self.blur(out)
        return out

# -----------------------------------------------------------------------------------------------------
# StyleConv (Style-modulated convolution layer used in StyleGAN2)
# -----------------------------------------------------------------------------------------------------
class StyledConv(nnx.Module):
    """
    ModulatedConv2d -> NoiseInjection -> FusedLeakyReLU 
    """

    def __init__(self, in_ch, out_ch, kernel_size, style_dim, upsample=False,
                 blur_kernel=(1, 3, 3, 1), demodulate=True, *, rngs: nnx.Rngs):
        self.conv = ModulatedConv2d(in_ch, out_ch, kernel_size, style_dim,
                                    demodulate=demodulate, upsample=upsample,
                                    blur_kernel=blur_kernel, rngs=rngs)
        self.noise = NoiseInjection(rngs=rngs)
        self.activate = FusedLeakyReLU(out_ch, rngs=rngs)

    def __call__(self, x, style, noise=None):
        x = self.conv(x, style)
        x = self.noise(x, noise=noise)
        return self.activate(x)

