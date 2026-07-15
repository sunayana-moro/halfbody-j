"""
Test for ../shared/stylegan.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax array creation

import numpy as np
import jax.numpy as jnp
import torch
from flax import nnx

from renderer.lia_resblocks import ModulatedConv2d as TMod, StyledConv as TStyled
from shared.stylegan import ModulatedConv2d as JMod, StyledConv as JStyled

ATOL = 1e-9          # float64 pass threshold
SEED_X = 0
SEED_STYLE = 1


# -------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rand(shape, seed):
    return np.random.RandomState(seed).randn(*shape).astype(np.float64)


def _fmt(shape):
    return "(" + ", ".join(str(d) for d in shape) + ")"


# -------------------------------------------------------------------------------
# Weight Porting (Torch -> JAX)
# -------------------------------------------------------------------------------

def port_mod(t, j):
    w = to_np(t.weight)                                           # (1, out, in, k, k)
    j.weight.value = jnp.asarray(w[0].transpose(2, 3, 1, 0))      # (k, k, in, out)
    mw = to_np(t.modulation.weight)                              # (in_ch, style)
    j.modulation.weight.value = jnp.asarray(mw.T)                # (style, in_ch)
    j.modulation.bias.value = jnp.asarray(to_np(t.modulation.bias))
    return [
        f"conv.weight  torch {_fmt(w.shape)}  -> jax {_fmt(j.weight.value.shape)}   [drop dim0, transpose OIHW->HWIO]",
        f"modulation.w torch {_fmt(mw.shape)}  -> jax {_fmt(j.modulation.weight.value.shape)}   [EqualLinear (O,I)->(I,O) = .T]",
        f"modulation.b torch {_fmt(to_np(t.modulation.bias).shape)}  -> jax {_fmt(j.modulation.bias.value.shape)}",
    ]


def port_styled(t, j):
    maps = port_mod(t.conv, j.conv)
    j.noise.weight.value = jnp.asarray(to_np(t.noise.weight))     # scalar (1,)
    b = to_np(t.activate.bias)                                    # (1, out, 1, 1)
    j.activate.bias.value = jnp.asarray(b.reshape(1, 1, 1, -1))   # (1, 1, 1, out)
    maps.append(f"noise.weight torch {_fmt(to_np(t.noise.weight).shape)}  -> jax {_fmt(j.noise.weight.value.shape)}   [scalar, 0-init]")
    maps.append(f"activate.bias torch {_fmt(b.shape)}  -> jax {_fmt(j.activate.bias.value.shape)}   [NCHW->NHWC channel]")
    return maps


# -------------------------------------------------------------------------------
# Test Runner
# -------------------------------------------------------------------------------

def run_case(idx, title, desc, tmod, jmod, x_nchw, style, styled):
    tmod = tmod.double().eval()
    maps = (port_styled if styled else port_mod)(tmod, jmod)

    with torch.no_grad():
        yt = to_np(tmod(torch.from_numpy(x_nchw), torch.from_numpy(style)))    # (B, out, oh, ow) NCHW
    yj_nhwc = to_np(jmod(jnp.asarray(x_nchw.transpose(0, 2, 3, 1)), jnp.asarray(style)))   # (B, oh, ow, out)
    yj = yj_nhwc.transpose(0, 3, 1, 2)                                          # -> NCHW

    print(f"[{idx}] {title}")
    print(f"      what      : {desc}")
    print(f"      input     : x {_fmt(x_nchw.shape)} NCHW (seed {SEED_X}) | style {_fmt(style.shape)} (seed {SEED_STYLE}) | dtype float64")
    print(f"      weights   : ported torch -> jax (shared, not random):")
    for m in maps:
        print(f"                    - {m}")
    print(f"      torch out : {_fmt(yt.shape)} NCHW")
    print(f"      jax out   : {_fmt(yj_nhwc.shape)} NHWC -> {_fmt(yj.shape)} NCHW")

    if yt.shape != yj.shape:
        print(f"      result    : SHAPE MISMATCH  -> FAIL\n")
        return False, float("inf")

    diff = np.abs(yt - yj)
    mx, mean = float(diff.max()), float(diff.mean())
    ok = mx < ATOL
    print(f"      diff      : max|Δ|={mx:.2e}   mean|Δ|={mean:.2e}")
    print(f"      condition : max|Δ| ({mx:.2e}) < atol ({ATOL:.0e})")
    print(f"      result    : {'PASS' if ok else 'FAIL'}\n")
    return ok, mx


# -------------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------------

def main():
    B, Cin, Cout, k, S, H, W = 2, 3, 4, 3, 8, 8, 8
    x = rand((B, Cin, H, W), SEED_X)
    style = rand((B, S), SEED_STYLE)

    print("=" * 84)
    print(" torch <-> jax parity : ModulatedConv2d + StyledConv")
    print("=" * 84)
    print(" torch reference : renderer.lia_resblocks   (NCHW)")
    print(" jax port        : shared.demo_stylegan     (NHWC, transposed at boundary)")
    print(" dtype           : float64 both sides   (real inference is float32 ~1e-6; float64 is a crisp gate)")
    print(" weights         : PORTED torch -> jax (shared weights, NOT independent random init)")
    print(" input           : seeded numpy, identical for both sides")
    print(f" pass condition  : max|torch - jax| < {ATOL:.0e}")
    print(f" dims            : B={B} Cin={Cin} Cout={Cout} k={k} style_dim={S} spatial={H}x{W}")
    print("=" * 84 + "\n")

    results = []
    idx = 1
    
    # -------------------------------------------------------------------------------
    # ModulatedConv2d Tests
    # -------------------------------------------------------------------------------
    for demod in (True, False):
        for mode in ("plain", "upsample", "downsample"):
            up, down = (mode == "upsample"), (mode == "downsample")
            desc = {
                "plain": "grouped conv, stride 1, padding k//2 (feature_group_count=B)",
                "upsample": "transposed conv (lhs_dilation=2, rot180 kernel) then Blur",
                "downsample": "Blur first, then grouped conv stride 2 VALID",
            }[mode] + f" | demodulate={demod}"
            t = TMod(Cin, Cout, k, S, demodulate=demod, upsample=up, downsample=down)
            j = JMod(Cin, Cout, k, S, demodulate=demod, upsample=up, downsample=down, rngs=nnx.Rngs(0))
            results.append(run_case(idx, f"ModulatedConv2d  mode={mode}  demod={demod}", desc,
                                    t, j, x, style, styled=False))
            idx += 1

    # -------------------------------------------------------------------------------
    # StyledConv Tests
    # -------------------------------------------------------------------------------
    for mode in ("plain", "upsample"):                   # torch StyledConv has no downsample
        up = (mode == "upsample")
        desc = f"ModulatedConv2d({mode}) -> NoiseInjection(noise=None, passthrough) -> FusedLeakyReLU"
        t = TStyled(Cin, Cout, k, S, upsample=up)
        j = JStyled(Cin, Cout, k, S, upsample=up, rngs=nnx.Rngs(0))
        results.append(run_case(idx, f"StyledConv  mode={mode}", desc, t, j, x, style, styled=True))
        idx += 1

    n_pass = sum(1 for ok, _ in results if ok)
    worst = max((mx for _, mx in results), default=0.0)
    print("=" * 84)
    print(f" SUMMARY : {n_pass}/{len(results)} PASS   |   worst max|Δ| = {worst:.2e}   |   threshold = {ATOL:.0e}")
    print("=" * 84)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())