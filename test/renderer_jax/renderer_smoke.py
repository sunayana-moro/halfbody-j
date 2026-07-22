"""Smoke test for renderer_jax/renderer.py (IdentidyAdaptive + IMTRenderer).

Smoke ONLY (no torch parity) — the real end-to-end validation is loading
renderer.ckpt into IMTRenderer and comparing to torch inference, which comes with
inference.py + the checkpoint-porting harness. Here we just confirm the full model
builds, wires together, and produces correct shapes / value ranges.

Runs EAGER in float32 (inputs are float32 even if x64 is enabled by a sibling suite),
batch=1, so the heavy full forward (incl. the res-64 attention) stays manageable on CPU.

Run:  test_env/bin/python test/renderer_jax/renderer.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
import jax.numpy as jnp
from flax import nnx

from renderer_jax.config import RendererConfig
from renderer_jax.renderer import IdentidyAdaptive, IMTRenderer


def randn(shape, seed):
    return jax.random.normal(nnx.Rngs(seed).params(), shape)   # float32


def main():
    print("-- smoke: renderer_jax/renderer.py " + "-" * 48)
    rngs = nnx.Rngs(0)
    cfg = RendererConfig()

    # 1. IdentidyAdaptive (cheap) --------------------------------------------------
    ia = IdentidyAdaptive(dim_mot=cfg.latent_dim, dim_app=cfg.id_dim,
                          depth=cfg.adapt_depth, rngs=rngs)
    out = ia(randn((2, cfg.latent_dim), 0), randn((2, cfg.id_dim), 1))
    assert out.shape == (2, cfg.latent_dim)
    print(f"  IdentidyAdaptive: (mot {cfg.latent_dim}, app {cfg.id_dim}) -> {out.shape}  OK")

    # 2. IMTRenderer build + decomposed sub-steps ---------------------------------
    m = IMTRenderer(cfg, rngs=rngs)
    x = randn((1, 512, 512, 3), 2)                    # batch 1, 512x512 RGB (NHWC)

    f_r, i_r = m.app_encode(x, use_running_average=True)
    assert i_r.shape == (1, cfg.id_dim) and len(f_r) == 6
    assert f_r[0].shape == (1, 8, 8, 512) and f_r[-1].shape == (1, 256, 256, 32)
    print(f"  app_encode: -> id {i_r.shape}, {len(f_r)} feats (coarse {f_r[0].shape} .. fine {f_r[-1].shape})  OK")

    t_c = m.mot_encode(x)
    assert t_c.shape == (1, cfg.latent_dim)
    ta = m.id_adapt(t_c, i_r)
    assert ta.shape == (1, cfg.latent_dim)
    ma = m.mot_decode(ta)
    assert len(ma) == 4 and ma[0].shape == (1, 8, 8, 512) and ma[3].shape == (1, 64, 64, 128)
    print(f"  mot_encode/id_adapt/mot_decode: latent {t_c.shape}, motion maps {[tuple(mm.shape) for mm in ma]}  OK")

    # 3. full forward (HEAVY: full 512 pipeline incl. coarse->fine attention) ------
    print("  running full IMTRenderer forward (may take a bit)...")
    frame, t_c2 = m(x, x, use_running_average=True)
    assert frame.shape == (1, 512, 512, 3), frame.shape
    assert t_c2.shape == (1, cfg.latent_dim)
    lo, hi = float(frame.min()), float(frame.max())
    assert 0.0 <= lo and hi <= 1.0, (lo, hi)
    print(f"  forward(x_current, x_reference) -> frame {frame.shape} range [{lo:.3f}, {hi:.3f}], t_c {t_c2.shape}  OK")

    print("  smoke: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
