"""Tests for shared/resblock.py — pure-jax smoke test AND a real torch parity test.

- smoke()  : builds each jax block on random NHWC input, asserts shapes. No torch.
- parity() : real torch (renderer.modules) vs jax (shared.resblock), float64,
             ported weights, transparent per-case report.

The fiddly part here is **nested BatchNorm**: each ConvBlock/FeatResBlock carries a
BatchNorm2d. We PERTURB the torch running-stats (else they're 0/1 = near-identity
and BN isn't really exercised), then port scale/bias/mean/var into flax and run in
eval (use_running_average=True). ResBlock has no BN (equalized ConvLayer path).

Run (parity needs torch):
    test_env/bin/python test/shared/resblock.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax array creation

import numpy as np
import jax.numpy as jnp
from flax import nnx

from shared.resblock import (
    ConvBlock, FeatResBlock, ResBlock, ConvResBlock, DownConvResBlock, UpConvResBlock,
)

ATOL = 1e-9


# ======================================================================== smoke
def _randn(shape, seed):
    return jax.random.normal(nnx.Rngs(seed).params(), shape)


def smoke():
    print("-- smoke (pure jax: shapes) " + "-" * 55)
    rngs = nnx.Rngs(0)
    x = _randn((2, 8, 8, 4), 0)
    assert ConvBlock(4, 8, rngs=rngs)(x, use_running_average=True).shape == (2, 8, 8, 8)
    assert FeatResBlock(4, rngs=rngs)(x, use_running_average=True).shape == (2, 8, 8, 4)
    assert ResBlock(4, 8, rngs=rngs)(x).shape == (2, 4, 4, 8)               # downsample
    assert ConvResBlock(4, 8, rngs=rngs)(x, use_running_average=True).shape == (2, 8, 8, 8)
    assert DownConvResBlock(4, 8, rngs=rngs)(x, use_running_average=True).shape == (2, 4, 4, 8)
    assert UpConvResBlock(4, 8, rngs=rngs)(x, use_running_average=True).shape == (2, 16, 16, 8)
    print("  ConvBlock/FeatResBlock/ResBlock/Conv/Down/Up shapes          OK")
    print("  smoke: all OK\n")


# ======================================================================= parity
import torch                                                     # noqa: E402
from renderer.modules import (                                   # noqa: E402
    ConvBlock as TConvBlock, FeatResBlock as TFeatResBlock, ResBlock as TResBlock,
    ConvResBlock as TConvResBlock, DownConvResBlock as TDownConvResBlock,
    UpConvResBlock as TUpConvResBlock,
)
from renderer.lia_resblocks import EqualConv2d as TEqualConv2d, FusedLeakyReLU as TFusedLeakyReLU  # noqa: E402


def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def rnd(shape, seed):
    return np.random.RandomState(seed).randn(*shape).astype(np.float64)


def _fmt(s):
    return "(" + ", ".join(str(d) for d in s) + ")"


def perturb_bn(tmod):
    """Randomize every BatchNorm2d's running-stats + affine so BN is exercised."""
    with torch.no_grad():
        for m in tmod.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0.0, 1.0)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.normal_(1.0, 0.3)
                m.bias.normal_(0.0, 0.3)
    return tmod


# ---- atomic porters (torch -> jax) ----
def port_conv(tc, jc):                                # nn.Conv2d (bias=False) -> nnx.Conv
    jc.kernel.value = jnp.asarray(to_np(tc.weight).transpose(2, 3, 1, 0))   # OIHW->HWIO


def port_bn(tn, jn):                                  # NormLayer(batch) -> NormLayer(batch)
    jn.norm.scale.value = jnp.asarray(to_np(tn.norm.weight))
    jn.norm.bias.value = jnp.asarray(to_np(tn.norm.bias))
    jn.norm.mean.value = jnp.asarray(to_np(tn.norm.running_mean))
    jn.norm.var.value = jnp.asarray(to_np(tn.norm.running_var))


def port_cb(tb, jb):                                  # ConvBlock
    port_conv(tb.conv, jb.conv)
    port_bn(tb.norm, jb.norm)


def port_fr(tf, jf):                                  # FeatResBlock
    port_cb(tf.conv1, jf.conv1)
    port_cb(tf.conv2, jf.conv2)


def port_cl(tcl, jcl):                                # ConvLayer (nn.Sequential -> our ConvLayer)
    for layer in tcl:
        if isinstance(layer, TEqualConv2d):
            jcl.conv.weight.value = jnp.asarray(to_np(layer.weight).transpose(2, 3, 1, 0))
            if jcl.conv.bias is not None and layer.bias is not None:
                jcl.conv.bias.value = jnp.asarray(to_np(layer.bias))
        elif isinstance(layer, TFusedLeakyReLU):
            jcl.act.bias.value = jnp.asarray(to_np(layer.bias).reshape(1, 1, 1, -1))
        # Blur is parameter-free


# ---- per-block porters ----
def port_ConvBlock(t, j):
    port_cb(t, j)


def port_FeatResBlock(t, j):
    port_fr(t, j)


def port_ResBlock(t, j):
    port_cl(t.conv1, j.conv1)
    port_cl(t.conv2, j.conv2)
    port_cl(t.skip, j.skip)


def port_ConvResBlock(t, j):                          # also Down/Up (same params; pool/upsample free)
    port_conv(t.conv1, j.conv1)
    port_bn(t.norm, j.norm)
    port_conv(t.conv2, j.conv2)
    port_fr(t.feat_res_block1, j.feat_res_block1)
    port_fr(t.feat_res_block2, j.feat_res_block2)


def report(idx, title, desc, in_desc, note, yt, yj):
    print(f"[{idx}] {title}")
    print(f"      what      : {desc}")
    print(f"      input     : {in_desc} | dtype float64")
    print(f"      weights   : ported torch -> jax: {note}")
    print(f"      torch out : {_fmt(yt.shape)}   jax out : {_fmt(yj.shape)}")
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


def run_block(idx, name, desc, tblock, jblock, port_fn, shape, note, bn, ura):
    if bn:
        perturb_bn(tblock)
    tblock = tblock.double().eval()
    port_fn(tblock, jblock)
    x = rnd(shape, 0)
    with torch.no_grad():
        yt = to_np(tblock(torch.from_numpy(x)))                  # NCHW
    xj = jnp.asarray(x.transpose(0, 2, 3, 1))
    yj = jblock(xj, use_running_average=True) if ura else jblock(xj)
    yj = to_np(yj).transpose(0, 3, 1, 2)                         # -> NCHW
    return report(idx, name, desc, f"x {_fmt(shape)} NCHW seed 0", note, yt, yj)


def parity():
    print("=" * 84)
    print(" torch <-> jax parity : shared/resblock.py")
    print("=" * 84)
    print(" torch : renderer.modules (NCHW) | jax : shared.resblock (NHWC) | float64 | ported weights")
    print(" note  : BatchNorm running-stats are randomized so BN is actually exercised")
    print(f" pass  : max|torch - jax| < {ATOL:.0e}\n")

    rngs = nnx.Rngs(0)
    Cin, Cout = 4, 8
    R = []
    R.append(run_block(1, "ConvBlock", "conv(bias=F) -> BN -> LeakyReLU(0.01)",
                       TConvBlock(Cin, Cout), ConvBlock(Cin, Cout, rngs=rngs),
                       port_ConvBlock, (2, Cin, 8, 8), "conv OIHW->HWIO, BN scale/bias/mean/var",
                       bn=True, ura=True))
    R.append(run_block(2, "FeatResBlock", "ConvBlock(act)->ConvBlock(no act)->+res->act",
                       TFeatResBlock(Cout), FeatResBlock(Cout, rngs=rngs),
                       port_FeatResBlock, (2, Cout, 8, 8), "2x (conv + BN)",
                       bn=True, ura=True))
    R.append(run_block(3, "ResBlock", "ConvLayer x3 (equalized), out+skip, downsample",
                       TResBlock(Cin, Cout), ResBlock(Cin, Cout, rngs=rngs),
                       port_ResBlock, (2, Cin, 8, 8), "3x ConvLayer: EqualConv2d + FusedLeakyReLU bias",
                       bn=False, ura=False))
    R.append(run_block(4, "ConvResBlock", "conv->BN->act->conv->FeatResBlock x2",
                       TConvResBlock(Cin, Cout), ConvResBlock(Cin, Cout, rngs=rngs),
                       port_ConvResBlock, (2, Cin, 8, 8), "convs + BN + 2 FeatResBlock",
                       bn=True, ura=True))
    R.append(run_block(5, "DownConvResBlock", "conv->BN->act->avgpool(2)->conv->FeatResBlock x2",
                       TDownConvResBlock(Cin, Cout), DownConvResBlock(Cin, Cout, rngs=rngs),
                       port_ConvResBlock, (2, Cin, 8, 8), "convs + BN + 2 FeatResBlock (avgpool free)",
                       bn=True, ura=True))
    R.append(run_block(6, "UpConvResBlock", "upsample(2,nearest)->conv->BN->act->conv->FeatResBlock x2",
                       TUpConvResBlock(Cin, Cout), UpConvResBlock(Cin, Cout, rngs=rngs),
                       port_ConvResBlock, (2, Cin, 8, 8), "convs + BN + 2 FeatResBlock (upsample free)",
                       bn=True, ura=True))

    n_pass = sum(1 for ok, _ in R if ok)
    worst = max((mx for _, mx in R), default=0.0)
    print("=" * 84)
    print(f" SUMMARY : {n_pass}/{len(R)} PASS   |   worst max|Δ| = {worst:.2e}   |   threshold = {ATOL:.0e}")
    print("=" * 84)
    return n_pass == len(R)


def main():
    smoke()
    ok = parity()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
