"""
Test for ../shared/layers.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax array creation

import numpy as np
import jax.numpy as jnp
from flax import nnx

from shared.layers import (
    make_kernel, upfirdn2d,
    EqualLinear, EqualConv2d, FusedLeakyReLU, Blur, NoiseInjection, NormLayer, ConvLayer,
)

ATOL = 1e-9


# -------------------------------------------------------------------------------
# Smoke Test Utilities
# -------------------------------------------------------------------------------

def _randn(shape, seed):
    return jax.random.normal(nnx.Rngs(seed).params(), shape)


# -------------------------------------------------------------------------------
# Smoke Test
# -------------------------------------------------------------------------------

def smoke():
    print("-- smoke (pure jax: shapes + invariants) " + "-" * 42)
    rngs = nnx.Rngs(0)

    assert jnp.allclose(upfirdn2d(_randn((2, 6, 6, 3), 0), make_kernel([1]), pad=(0, 0)),
                        _randn((2, 6, 6, 3), 0), atol=1e-6)
    print("  upfirdn2d identity kernel is a no-op                         OK")

    assert EqualLinear(16, 32, rngs=rngs)(_randn((4, 16), 1)).shape == (4, 32)
    assert EqualConv2d(8, 16, 3, padding=1, rngs=rngs)(_randn((2, 10, 10, 8), 2)).shape == (2, 10, 10, 16)
    print("  EqualLinear / EqualConv2d shapes                             OK")

    x8 = _randn((2, 8, 8, 8), 3)
    assert FusedLeakyReLU(8, rngs=rngs)(x8).shape == x8.shape
    assert Blur([1, 3, 3, 1], pad=(1, 1))(_randn((2, 8, 8, 4), 4)).shape[-1] == 4
    print("  FusedLeakyReLU / Blur shapes                                 OK")

    ni = NoiseInjection(rngs=rngs)
    img = _randn((2, 8, 8, 4), 5)
    assert jnp.array_equal(ni(img, noise=None), img)
    print("  NoiseInjection noise=None -> identity                        OK")

    x4 = _randn((2, 8, 8, 4), 7)
    for nt in ("batch", "instance", "layer"):
        n = NormLayer(4, nt, rngs=rngs)
        y = n(x4, use_running_average=True) if nt == "batch" else n(x4)
        assert y.shape == x4.shape
    assert ConvLayer(4, 8, 3, rngs=rngs)(x4).shape == (2, 8, 8, 8)
    assert ConvLayer(4, 8, 3, downsample=True, rngs=rngs)(x4).shape == (2, 4, 4, 8)
    print("  NormLayer (batch/instance/layer) / ConvLayer shapes          OK")
    print("  smoke: all OK\n")


# -------------------------------------------------------------------------------
# Parity Test Imports & Utilities
# -------------------------------------------------------------------------------

import torch                                                     # noqa: E402
from renderer.lia_resblocks import (                             # noqa: E402
    EqualLinear as TEqualLinear, EqualConv2d as TEqualConv2d,
    FusedLeakyReLU as TFusedLeakyReLU, Blur as TBlur,
    NoiseInjection as TNoiseInjection, ConvLayer as TConvLayer,
)
from renderer.modules import NormLayer as TNormLayer            # noqa: E402


def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def rnd(shape, seed):
    return np.random.RandomState(seed).randn(*shape).astype(np.float64)


def _fmt(s):
    return "(" + ", ".join(str(d) for d in s) + ")"


def report(idx, title, desc, in_desc, maps, yt, yj):
    """yt, yj: numpy arrays already in the SAME layout."""
    print(f"[{idx}] {title}")
    print(f"      what      : {desc}")
    print(f"      input     : {in_desc} | dtype float64")
    if maps:
        print(f"      weights   : ported torch -> jax (shared, not random):")
        for m in maps:
            print(f"                    - {m}")
    else:
        print(f"      weights   : none (parameter-free)")
    print(f"      torch out : {_fmt(yt.shape)}")
    print(f"      jax out   : {_fmt(yj.shape)}")
    if yt.shape != yj.shape:
        print(f"      result    : SHAPE MISMATCH -> FAIL\n")
        return False, float("inf")
    d = np.abs(yt - yj)
    mx, mean = float(d.max()), float(d.mean())
    ok = mx < ATOL
    print(f"      diff      : max|Δ|={mx:.2e}   mean|Δ|={mean:.2e}")
    print(f"      condition : max|Δ| ({mx:.2e}) < atol ({ATOL:.0e})")
    print(f"      result    : {'PASS' if ok else 'FAIL'}\n")
    return ok, mx


# -------------------------------------------------------------------------------
# Parity Tests
# -------------------------------------------------------------------------------

def parity():
    print("=" * 84)
    print(" torch <-> jax parity : shared/layers.py")
    print("=" * 84)
    print(" torch : renderer.lia_resblocks + renderer.modules (NCHW)")
    print(" jax   : shared.layers (NHWC, transposed at boundary) | float64 | ported weights")
    print(f" pass  : max|torch - jax| < {ATOL:.0e}\n")

    R = []

    # -------------------------------------------------------------------------------
    # EqualLinear (plain)
    # -------------------------------------------------------------------------------
    t = TEqualLinear(16, 32).double().eval()
    j = EqualLinear(16, 32, rngs=nnx.Rngs(0))
    j.weight.value = jnp.asarray(to_np(t.weight).T)          # (out,in)->(in,out)
    j.bias.value = jnp.asarray(to_np(t.bias))
    x = rnd((4, 16), 0)
    with torch.no_grad():
        yt = to_np(t(torch.from_numpy(x)))
    yj = to_np(j(jnp.asarray(x)))
    R.append(report(1, "EqualLinear (plain)", "equalized-LR linear, scale at call",
                    "x (4, 16)  seed 0",
                    [f"weight torch {_fmt(to_np(t.weight).shape)} -> jax {_fmt(j.weight.value.shape)}  [.T]",
                     f"bias   torch {_fmt(to_np(t.bias).shape)} -> jax {_fmt(j.bias.value.shape)}"], yt, yj))

    # -------------------------------------------------------------------------------
    # EqualLinear (fused-lrelu activation)
    # -------------------------------------------------------------------------------
    t = TEqualLinear(16, 32, activation="fused_lrelu").double().eval()
    j = EqualLinear(16, 32, activation="fused_lrelu", rngs=nnx.Rngs(0))
    j.weight.value = jnp.asarray(to_np(t.weight).T)
    j.bias.value = jnp.asarray(to_np(t.bias))
    with torch.no_grad():
        yt = to_np(t(torch.from_numpy(x)))
    yj = to_np(j(jnp.asarray(x)))
    R.append(report(2, "EqualLinear (fused_lrelu)", "linear + fused leaky-relu",
                    "x (4, 16)  seed 0",
                    ["weight .T, bias (as above)"], yt, yj))

    # -------------------------------------------------------------------------------
    # EqualConv2d
    # -------------------------------------------------------------------------------
    t = TEqualConv2d(8, 16, 3, padding=1).double().eval()
    j = EqualConv2d(8, 16, 3, padding=1, rngs=nnx.Rngs(0))
    j.weight.value = jnp.asarray(to_np(t.weight).transpose(2, 3, 1, 0))   # OIHW->HWIO
    j.bias.value = jnp.asarray(to_np(t.bias))
    x = rnd((2, 8, 10, 10), 1)                                # NCHW
    with torch.no_grad():
        yt = to_np(t(torch.from_numpy(x)))                    # (2,16,10,10)
    yj = to_np(j(jnp.asarray(x.transpose(0, 2, 3, 1)))).transpose(0, 3, 1, 2)
    R.append(report(3, "EqualConv2d", "equalized-LR conv, padding=1",
                    "x (2, 8, 10, 10) NCHW  seed 1",
                    [f"weight torch {_fmt(to_np(t.weight).shape)} -> jax {_fmt(j.weight.value.shape)}  [OIHW->HWIO]",
                     f"bias   torch {_fmt(to_np(t.bias).shape)} -> jax {_fmt(j.bias.value.shape)}"], yt, yj))

    # -------------------------------------------------------------------------------
    # FusedLeakyReLU
    # -------------------------------------------------------------------------------
    t = TFusedLeakyReLU(8).double().eval()
    with torch.no_grad():
        t.bias.add_(torch.randn_like(t.bias))                 # non-zero bias so it matters
    j = FusedLeakyReLU(8, rngs=nnx.Rngs(0))
    j.bias.value = jnp.asarray(to_np(t.bias).reshape(1, 1, 1, -1))   # (1,C,1,1)->(1,1,1,C)
    x = rnd((2, 8, 8, 8), 2)
    with torch.no_grad():
        yt = to_np(t(torch.from_numpy(x)))
    yj = to_np(j(jnp.asarray(x.transpose(0, 2, 3, 1)))).transpose(0, 3, 1, 2)
    R.append(report(4, "FusedLeakyReLU", "leaky_relu(x + bias) * sqrt(2)",
                    "x (2, 8, 8, 8) NCHW  seed 2",
                    [f"bias torch (1,8,1,1) -> jax (1,1,1,8)"], yt, yj))

    # -------------------------------------------------------------------------------
    # Blur (parameter-free)
    # -------------------------------------------------------------------------------
    t = TBlur([1, 3, 3, 1], pad=(1, 1)).double().eval()
    j = Blur([1, 3, 3, 1], pad=(1, 1))
    x = rnd((2, 4, 8, 8), 3)
    with torch.no_grad():
        yt = to_np(t(torch.from_numpy(x)))
    yj = to_np(j(jnp.asarray(x.transpose(0, 2, 3, 1)))).transpose(0, 3, 1, 2)
    R.append(report(5, "Blur", "FIR blur (upfirdn2d), pad=(1,1)",
                    "x (2, 4, 8, 8) NCHW  seed 3", [], yt, yj))

    # -------------------------------------------------------------------------------
    # NoiseInjection (add path)
    # -------------------------------------------------------------------------------
    t = TNoiseInjection().double().eval()
    with torch.no_grad():
        t.weight.add_(0.7)                                    # non-zero so noise matters
    j = NoiseInjection(rngs=nnx.Rngs(0))
    j.weight.value = jnp.asarray(to_np(t.weight))
    img = rnd((2, 4, 8, 8), 4)
    noise = rnd((2, 1, 8, 8), 5)                              # torch noise broadcasts over C
    with torch.no_grad():
        yt = to_np(t(torch.from_numpy(img), noise=torch.from_numpy(noise)))
    yj = to_np(j(jnp.asarray(img.transpose(0, 2, 3, 1)),
                 noise=jnp.asarray(noise.transpose(0, 2, 3, 1)))).transpose(0, 3, 1, 2)
    R.append(report(6, "NoiseInjection (add)", "image + weight * noise",
                    "img (2,4,8,8) seed 4 | noise (2,1,8,8) seed 5",
                    [f"weight torch (1,) -> jax (1,)  (set 0.7)"], yt, yj))

    # -------------------------------------------------------------------------------
    # NormLayer batch / instance / layer
    # -------------------------------------------------------------------------------
    for i, nt in enumerate(("batch", "instance", "layer"), start=7):
        t = TNormLayer(4, nt).double().eval()
        maps = []
        if nt == "batch":
            with torch.no_grad():                            # non-trivial running stats + affine
                t.norm.running_mean.add_(torch.randn(4).double())
                t.norm.running_var.mul_(0).add_(torch.rand(4).double() + 0.5)
                t.norm.weight.mul_(0).add_(torch.randn(4).double() + 1)
                t.norm.bias.add_(torch.randn(4).double())
            maps = ["running_mean/var, weight->scale, bias (all 4-vectors)"]
        elif nt == "layer":
            with torch.no_grad():
                t.norm.weight.add_(torch.randn(4).double())
                t.norm.bias.add_(torch.randn(4).double())
            maps = ["GroupNorm(1,C) weight->scale, bias"]
        j = NormLayer(4, nt, rngs=nnx.Rngs(0))
        if nt == "batch":
            j.norm.scale.value = jnp.asarray(to_np(t.norm.weight))
            j.norm.bias.value = jnp.asarray(to_np(t.norm.bias))
            j.norm.mean.value = jnp.asarray(to_np(t.norm.running_mean))
            j.norm.var.value = jnp.asarray(to_np(t.norm.running_var))
        elif nt == "layer":
            j.norm.scale.value = jnp.asarray(to_np(t.norm.weight))
            j.norm.bias.value = jnp.asarray(to_np(t.norm.bias))
        x = rnd((2, 4, 8, 8), 6)
        with torch.no_grad():
            yt = to_np(t(torch.from_numpy(x)))
        xj = jnp.asarray(x.transpose(0, 2, 3, 1))
        yj = to_np(j(xj, use_running_average=True) if nt == "batch" else j(xj)).transpose(0, 3, 1, 2)
        R.append(report(i, f"NormLayer ({nt})",
                        {"batch": "BatchNorm2d eval (running stats)",
                         "instance": "InstanceNorm2d, no affine, eps=1e-5",
                         "layer": "GroupNorm(1,C), affine, eps=1e-5"}[nt],
                        "x (2, 4, 8, 8) NCHW  seed 6", maps, yt, yj))

    # -------------------------------------------------------------------------------
    # ConvLayer plain / downsample
    # -------------------------------------------------------------------------------
    for i, (down, sh) in enumerate([(False, (2, 4, 8, 8)), (True, (2, 4, 8, 8))], start=10):
        t = TConvLayer(4, 8, 3, downsample=down).double().eval()
        j = ConvLayer(4, 8, 3, downsample=down, rngs=nnx.Rngs(0))
        # ConvLayer = [Blur?] + EqualConv2d + FusedLeakyReLU; port conv + activation bias
        tconv = t[1] if down else t[0]                       # nn.Sequential index of EqualConv2d
        tact = t[2] if down else t[1]                        # FusedLeakyReLU
        j.conv.weight.value = jnp.asarray(to_np(tconv.weight).transpose(2, 3, 1, 0))
        if j.conv.bias is not None and tconv.bias is not None:
            j.conv.bias.value = jnp.asarray(to_np(tconv.bias))
        j.act.bias.value = jnp.asarray(to_np(tact.bias).reshape(1, 1, 1, -1))
        x = rnd(sh, 7)
        with torch.no_grad():
            yt = to_np(t(torch.from_numpy(x)))
        yj = to_np(j(jnp.asarray(x.transpose(0, 2, 3, 1)))).transpose(0, 3, 1, 2)
        R.append(report(i, f"ConvLayer ({'downsample' if down else 'plain'})",
                        "Blur(if down) + EqualConv2d + FusedLeakyReLU",
                        f"x {_fmt(sh)} NCHW  seed 7",
                        ["conv.weight OIHW->HWIO, act.bias (1,C,1,1)->(1,1,1,C)"], yt, yj))

    n_pass = sum(1 for ok, _ in R if ok)
    worst = max((mx for _, mx in R), default=0.0)
    print("=" * 84)
    print(f" SUMMARY : {n_pass}/{len(R)} PASS   |   worst max|Δ| = {worst:.2e}   |   threshold = {ATOL:.0e}")
    print("=" * 84)
    return n_pass == len(R)


# -------------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------------

def main():
    smoke()
    ok = parity()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())