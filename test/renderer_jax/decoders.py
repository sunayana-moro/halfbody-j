"""Tests for renderer_jax/decoders.py — smoke + real torch parity (under jax.jit).

- smoke()  : build MotionDecoder + SynthesisNetwork on random input, check shapes.
- parity() : MotionDecoder — torch vs jax, float64, ported weights, JAX under jax.jit.

SynthesisNetwork parity is DEFERRED (smoke only). Reason: it contains SelfAttention,
whose composite blocks (UnifiedTransformerBlock / UnifiedSwinBlock) are not yet
parity-verified, and the Swin blocks hold bare-numpy rel_index/attn_mask that break
nnx.split (so it can't be jitted via split/merge yet). Verify the attention
composites in test/shared/attention.py first, then enable SynthesisNetwork parity.

Run:  test_env/bin/python test/renderer_jax/decoders.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.dirname(__file__))             # for _porters (append: never shadow a real package)

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from flax import nnx

from renderer_jax.config import RendererConfig
from renderer_jax.decoders import MotionDecoder, SynthesisNetwork
from renderer_jax.encoders import IdentityEncoder

ATOL = 1e-9


def _randn(shape, seed):
    return jax.random.normal(nnx.Rngs(seed).params(), shape)


def jf(m, *args, **static_kw):
    graphdef, state = nnx.split(m)
    run = jax.jit(lambda st, *a: nnx.merge(graphdef, st)(*a, **static_kw))
    return run(state, *args)


# ------------------------------------------------------------------ smoke
def smoke():
    print("-- smoke (pure jax: shapes) " + "-" * 55)
    rngs = nnx.Rngs(0)

    md = MotionDecoder(rngs=rngs)
    t = _randn((2, 32), 0)                                # motion latent (B, latent_dim)
    m1, m2, m3, m4 = md(t)
    assert m1.shape == (2, 8, 8, 512) and m2.shape == (2, 16, 16, 512)
    assert m3.shape == (2, 32, 32, 256) and m4.shape == (2, 64, 64, 128)
    print(f"  MotionDecoder: m1 {m1.shape} m2 {m2.shape} m3 {m3.shape} m4 {m4.shape}  OK")

    # SynthesisNetwork smoke: source its input from the REAL producer (IdentityEncoder),
    # not hand-typed dims -> the 6 feature-pyramid sizes are guaranteed correct and can't
    # drift from config. (Heavy: full 512 encoder + synthesis incl. Swin -> run on GPU.)
    cfg = RendererConfig()
    ie = IdentityEncoder(output_channels=cfg.feature_dims, initial_channels=cfg.id_enc_init,
                         dm=cfg.id_dim, rngs=rngs)
    f_r, i_r = ie(_randn((1, 512, 512, 3), 9), use_running_average=True)   # 6 maps, coarse..fine
    print("  SynthesisNetwork input (from IdentityEncoder), NHWC:")
    for k, fm in enumerate(f_r):
        print(f"      features_align[{k}] : {fm.shape}")

    sn = SynthesisNetwork(cfg, cfg.feature_dims, cfg.spatial_dims, rngs=rngs)
    out = sn(f_r, use_running_average=True)
    assert out.shape == (1, 512, 512, 3), out.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0            # sigmoid range
    print(f"  SynthesisNetwork output frame : {out.shape}, range [0,1]  OK")
    print("  smoke: all OK\n")


# ------------------------------------------------------------------ parity
import torch                                                       # noqa: E402
from renderer.models import MotionDecoder as TMotionDecoder        # noqa: E402
from _porters import to_np, rnd, fmt, port_motion_decoder          # noqa: E402


def report(idx, title, desc, yt, yj):
    print(f"[{idx}] {title}")
    print(f"      what   : {desc}")
    print(f"      torch  : {fmt(yt.shape)}    jax : {fmt(yj.shape)}")
    if yt.shape != yj.shape:
        print(f"      result : SHAPE MISMATCH -> FAIL\n")
        return False, float("inf")
    d = np.abs(yt - yj)
    mx, mean = float(d.max()), float(d.mean())
    ok = mx < ATOL
    print(f"      diff   : max|Δ|={mx:.2e}  mean|Δ|={mean:.2e}")
    print(f"      result : {'PASS' if ok else 'FAIL'} ( < {ATOL:.0e} )\n")
    return ok, mx


def parity():
    print("=" * 84)
    print(" torch <-> jax parity : renderer_jax/decoders.py  (JAX side under jax.jit, float64)")
    print(" note : SynthesisNetwork parity deferred (attention composites unverified) -> smoke only")
    print("=" * 84 + "\n")
    R = []

    # MotionDecoder -- compare all 4 motion maps m1..m4 (NCHW->NHWC on torch side)
    t = TMotionDecoder().double().eval()
    j = MotionDecoder(rngs=nnx.Rngs(0))
    port_motion_decoder(t, j)
    style = rnd((2, 32), 0)                               # motion latent
    with torch.no_grad():
        t_ms = t(torch.from_numpy(style))                # tuple of (B,C,H,W)
    j_ms = jf(j, jnp.asarray(style))                     # tuple of (B,H,W,C)
    for k, (tm, jm) in enumerate(zip(t_ms, j_ms), start=1):
        tm_nhwc = to_np(tm).transpose(0, 2, 3, 1)
        ok, mx = report(k, f"MotionDecoder m{k}", "motion pyramid map", tm_nhwc, to_np(jm))
        R.append((ok, mx))

    n_pass = sum(1 for ok, _ in R if ok)
    worst = max((mx for _, mx in R), default=0.0)
    print("=" * 84)
    print(f" SUMMARY : {n_pass}/{len(R)} PASS   |   worst max|Δ| = {worst:.2e}   |   threshold = {ATOL:.0e}")
    print("           (SynthesisNetwork: smoke-tested only — parity deferred)")
    print("=" * 84)
    return n_pass == len(R)


def main():
    smoke()
    return 0 if parity() else 1


if __name__ == "__main__":
    raise SystemExit(main())
