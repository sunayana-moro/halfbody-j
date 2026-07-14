"""
Tests the ../shared/layers.py
"""

import os
import sys

# put the repo root on sys.path so `import shared.layers` resolves
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
import jax.numpy as jnp
from flax import nnx

from shared.layers import (
    make_kernel,
    upfirdn2d,
    EqualLinear,
    EqualConv2d,
    FusedLeakyReLU,
    Blur,
    NoiseInjection,
    NormLayer,
    ConvLayer,
)


def randn(shape, seed):
    """A fresh seeded random matrix (deterministic per seed)."""
    return jax.random.normal(nnx.Rngs(seed).params(), shape)


def main():
    rngs = nnx.Rngs(0)

    # upfirdn2d / make_kernel: identity kernel [1] with no pad returns input unchanged
    x = randn((2, 6, 6, 3), 0)
    assert jnp.allclose(upfirdn2d(x, make_kernel([1]), pad=(0, 0)), x, atol=1e-6)
    print("upfirdn2d: identity kernel is a no-op  OK")

    # EqualLinear: plain and with fused-leaky activation
    lin = EqualLinear(16, 32, rngs=rngs)
    assert lin(randn((4, 16), 1)).shape == (4, 32)
    lin_act = EqualLinear(16, 32, activation="fused_lrelu", rngs=rngs)
    assert lin_act(randn((4, 16), 1)).shape == (4, 32)
    print("EqualLinear: plain + activation  OK")

    # EqualConv2d: same-padding conv preserves spatial size, changes channels
    conv = EqualConv2d(8, 16, 3, padding=1, rngs=rngs)
    assert conv(randn((2, 10, 10, 8), 2)).shape == (2, 10, 10, 16)
    print("EqualConv2d: shape OK")

    # FusedLeakyReLU: shape preserved
    flr = FusedLeakyReLU(8, rngs=rngs)
    x8 = randn((2, 8, 8, 8), 3)
    assert flr(x8).shape == x8.shape
    print("FusedLeakyReLU: shape preserved  OK")

    # Blur: channel count preserved (spatial may change from padding)
    blur = Blur([1, 3, 3, 1], pad=(1, 1))
    bx = randn((2, 8, 8, 4), 4)
    by = blur(bx)
    assert by.shape[-1] == 4
    print(f"Blur: {bx.shape} -> {by.shape}  OK")

    ni = NoiseInjection(rngs=rngs)
    img = randn((2, 8, 8, 4), 5)
    assert jnp.array_equal(ni(img, noise=None), img)
    assert ni(img, noise=randn((2, 8, 8, 4), 6)).shape == img.shape
    print("NoiseInjection: noise=None -> identity, else additive  OK")

    # NormLayer: batch / instance / layer all run and preserve shape
    x4 = randn((2, 8, 8, 4), 7)
    for nt in ("batch", "instance", "layer"):
        norm = NormLayer(4, nt, rngs=rngs)
        y = norm(x4, use_running_average=True) if nt == "batch" else norm(x4)
        assert y.shape == x4.shape
    print("NormLayer: batch/instance/layer all OK")

    # ConvLayer: plain, downsample, and the skip-style (no activation, no bias)
    cl = ConvLayer(4, 8, 3, rngs=rngs)
    cl_down = ConvLayer(4, 8, 3, downsample=True, rngs=rngs)
    cl_skip = ConvLayer(4, 8, 1, downsample=True, activate=False, bias=False, rngs=rngs)
    assert cl(x4).shape == (2, 8, 8, 8)
    assert cl_down(x4).shape == (2, 4, 4, 8)
    assert cl_skip(x4).shape == (2, 4, 4, 8)
    print("ConvLayer: plain + downsample + skip(no-act,no-bias)  OK")

    print("all shared/layers.py blocks OK")


if __name__ == "__main__":
    main()
