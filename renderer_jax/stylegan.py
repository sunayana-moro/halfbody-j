"""StyleGAN modulated / styled convs. torch: lia_resblocks.ModulatedConv2d / StyledConv."""

from __future__ import annotations

from flax import nnx


class ModulatedConv2d(nnx.Module):
    """Style-modulated conv w/ grouped-batch trick. EMERGENCY op (grouped-batch conv)."""

    def __init__(self, in_ch, out_ch, kernel_size, style_dim, demodulate=True,
                 upsample=False, downsample=False, blur_kernel=(1, 3, 3, 1),
                 *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, style):
        raise NotImplementedError


class StyledConv(nnx.Module):
    """ModulatedConv2d -> NoiseInjection -> FusedLeakyReLU. torch: lia_resblocks.StyledConv."""

    def __init__(self, in_ch, out_ch, kernel_size, style_dim, upsample=False,
                 blur_kernel=(1, 3, 3, 1), demodulate=True, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, x, style, noise=None):
        raise NotImplementedError
