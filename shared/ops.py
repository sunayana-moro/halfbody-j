"""Pure functions — no learned params. torch source: lia_resblocks free functions."""

from __future__ import annotations


def make_kernel(k):
    """Normalized 2D FIR kernel from a 1D list. torch: lia_resblocks.make_kernel."""
    raise NotImplementedError


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    """Upsample-pad-FIR-downsample (StyleGAN2 blur). EMERGENCY op — provided later."""
    raise NotImplementedError


def fused_leaky_relu(x, bias, negative_slope=0.2, scale=2 ** 0.5):
    """leaky_relu(x + bias) * scale. torch: lia_resblocks.fused_leaky_relu."""
    raise NotImplementedError


def scaled_leaky_relu(x, negative_slope=0.2):
    """leaky_relu(x). torch: lia_resblocks.ScaledLeakyReLU."""
    raise NotImplementedError


def pixel_shuffle(x, upscale_factor):
    """Depth-to-space. torch: nn.PixelShuffle (SynthesisNetwork.final_conv)."""
    raise NotImplementedError


def resize_nearest(x, scale):
    """Nearest-neighbour upsample. torch: nn.Upsample(mode='nearest') (UpConvResBlock)."""
    raise NotImplementedError
