"""Tests for renderer_jax/encoders.py — smoke + real torch parity (under jax.jit).

- smoke()  : build each jax encoder on random NHWC input, check output shapes.
- parity() : torch (renderer.models) vs jax (renderer_jax.encoders), float64,
             ported weights, run the JAX side under jax.jit (the inference path),
             report max|Δ| per network.

Run:  test_env/bin/python test/renderer_jax/encoders.py
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

from renderer_jax.encoders import IdentityEncoder, MotionEncoder

ATOL = 1e-9


def _randn(shape, seed):
    return jax.random.normal(nnx.Rngs(seed).params(), shape)


def jf(m, *args, **static_kw):
    """JAX module under jax.jit via nnx.split/merge (nnx.jit is broken on this
    flax/jax combo). Works because module constants are static (Blur kernel = tuple)."""
    graphdef, state = nnx.split(m)
    run = jax.jit(lambda st, *a: nnx.merge(graphdef, st)(*a, **static_kw))
    return run(state, *args)


# ------------------------------------------------------------------ smoke
def smoke():
    print("-- smoke (pure jax: shapes) " + "-" * 55)
    rngs = nnx.Rngs(0)
    # small square input; encoders are fully-conv + global-pool so any size works
    x = _randn((2, 64, 64, 3), 0)

    ie = IdentityEncoder(output_channels=(32, 64, 128, 256, 512, 512), rngs=rngs)
    feats, idv = ie(x, use_running_average=True)
    assert idv.shape == (2, 512)
    assert len(feats) == 6 and feats[0].shape[-1] == 512 and feats[-1].shape[-1] == 32
    print(f"  IdentityEncoder: {len(feats)} feats (coarse {feats[0].shape} .. fine {feats[-1].shape}), id {idv.shape}  OK")

    me = MotionEncoder(initial_channels=64, output_channels=(128, 256, 512, 512, 512), rngs=rngs)
    mot = me(x)
    assert mot.shape == (2, 32)
    print(f"  MotionEncoder: motion latent {mot.shape}  OK")
    print("  smoke: all OK\n")


# ------------------------------------------------------------------ parity
import torch                                                       # noqa: E402
from renderer.models import IdentityEncoder as TIdentityEncoder, MotionEncoder as TMotionEncoder  # noqa: E402
from _porters import (to_np, rnd, fmt, port_identity_encoder, port_motion_encoder)  # noqa: E402


def perturb_bn(tmod):
    with torch.no_grad():
        for m in tmod.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0.0, 1.0)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.normal_(1.0, 0.3)
                m.bias.normal_(0.0, 0.3)
    return tmod


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
    print(" torch <-> jax parity : renderer_jax/encoders.py  (JAX side under jax.jit, float64)")
    print("=" * 84 + "\n")
    R = []
    x = rnd((2, 64, 64, 3), 0)                           # NHWC seed 0
    x_nchw = torch.from_numpy(x.transpose(0, 3, 1, 2))   # -> NCHW for torch

    # 1. IdentityEncoder -- compare identity vector + every feature map
    t = perturb_bn(TIdentityEncoder(output_channels=[32, 64, 128, 256, 512, 512])).double().eval()
    j = IdentityEncoder(output_channels=(32, 64, 128, 256, 512, 512), rngs=nnx.Rngs(0))
    port_identity_encoder(t, j)
    with torch.no_grad():
        t_feats, t_id = t(x_nchw)
    j_feats, j_id = jf(j, jnp.asarray(x), use_running_average=True)
    ok, mx = report(1, "IdentityEncoder (identity vector)", "reference frame -> 512-d id vec",
                    to_np(t_id), to_np(j_id)); R.append((ok, mx))
    for k, (tf, jff) in enumerate(zip(t_feats, j_feats)):   # both already coarse..fine (features[::-1])
        tf_nhwc = to_np(tf).transpose(0, 2, 3, 1)
        ok, mx = report(f"1.{k}", f"IdentityEncoder feature[{k}]", "pyramid level (coarse..fine)",
                        tf_nhwc, to_np(jff)); R.append((ok, mx))

    # 2. MotionEncoder -- motion latent
    t = TMotionEncoder(initial_channels=64, output_channels=[128, 256, 512, 512, 512]).double().eval()
    j = MotionEncoder(initial_channels=64, output_channels=(128, 256, 512, 512, 512), rngs=nnx.Rngs(0))
    port_motion_encoder(t, j)
    with torch.no_grad():
        t_mot = t(x_nchw)
    j_mot = jf(j, jnp.asarray(x))
    ok, mx = report(2, "MotionEncoder (motion latent)", "frame -> 32-d motion latent",
                    to_np(t_mot), to_np(j_mot)); R.append((ok, mx))

    n_pass = sum(1 for ok, _ in R if ok)
    worst = max((mx for _, mx in R), default=0.0)
    print("=" * 84)
    print(f" SUMMARY : {n_pass}/{len(R)} PASS   |   worst max|Δ| = {worst:.2e}   |   threshold = {ATOL:.0e}")
    print("=" * 84)
    return n_pass == len(R)


def main():
    smoke()
    return 0 if parity() else 1


if __name__ == "__main__":
    raise SystemExit(main())
