"""
Test for ../shared/attention.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any jax array creation

import numpy as np
import jax.numpy as jnp
from flax import nnx

from shared.attention import (
    StandardUnifiedAttention, UnifiedTransformerBlock, UnifiedSwinBlock,
    SelfAttention, CrossAttention, window_partition, window_reverse,
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
    B, C, heads = 2, 32, 4

    attn = StandardUnifiedAttention(C, heads, rngs=rngs)
    _, amap = attn(_randn((B, 64, C), 1), _randn((B, 64, C), 1), _randn((B, 64, C), 1))
    assert jnp.allclose(amap.sum(-1), 1.0, atol=1e-5)
    print(f"  StandardUnifiedAttention: attn rows sum to 1                 OK")

    x8 = _randn((B, 8, 8, C), 2)
    tblock = UnifiedTransformerBlock(C, 8, heads, rngs=rngs)
    assert tblock(x8).shape == x8.shape and tblock(x8, key=x8, value=x8).shape == x8.shape
    print(f"  UnifiedTransformerBlock: self & cross shape preserved        OK")

    x16 = _randn((B, 16, 16, C), 3)
    s0 = UnifiedSwinBlock(C, 16, heads, window_size=4, shift_size=0, rngs=rngs)
    s1 = UnifiedSwinBlock(C, 16, heads, window_size=4, shift_size=2, rngs=rngs)
    assert s0(x16).shape == x16.shape and s1(x16).shape == x16.shape
    print(f"  UnifiedSwinBlock: shift 0 & 2 shape preserved                OK")

    sa = SelfAttention(C, 16, heads, window_size=4, swin_res_threshold=8, rngs=rngs)
    assert sa(x16).shape == x16.shape
    ca_c = CrossAttention(C, 4, heads, swin_res_threshold=8, rngs=rngs)
    co, amap = ca_c.coarse_stage(_randn((B, 4, 4, C), 4), _randn((B, 4, 4, C), 5), _randn((B, 4, 4, C), 6))
    ca_f = CrossAttention(C, 16, heads, swin_res_threshold=8, rngs=rngs)
    assert ca_f.fine_stage(_randn((B, 16, 16, C), 7), amap).shape == (B, 16, 16, C)
    print(f"  SelfAttention / CrossAttention coarse->fine                  OK")

    assert jnp.allclose(window_reverse(window_partition(x16, 4), 4, 16, 16), x16)
    print(f"  window_partition/reverse round-trip                          OK")
    print("  smoke: all OK\n")


# -------------------------------------------------------------------------------
# Parity Test Imports & Utilities
# -------------------------------------------------------------------------------

import torch                                                         # noqa: E402
from renderer.attention_modules import (                             # noqa: E402
    StandardUnifiedAttention as TStd,
    window_partition as t_wp, window_reverse as t_wr,
)


def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def rnd(shape, seed):
    return np.random.RandomState(seed).randn(*shape).astype(np.float64)


def _fmt(s):
    return "(" + ", ".join(str(d) for d in s) + ")"


def jf(m, *args, **static_kw):
    """Run the JAX module under jax.jit via nnx.split/merge (nnx.jit is broken with
    flax 0.10.2 + jax 0.10.0: it passes abstracted_axes which jax.jit rejects).
    graphdef static, state + args traced, kwargs baked static."""
    graphdef, state = nnx.split(m)
    run = jax.jit(lambda st, *a: nnx.merge(graphdef, st)(*a, **static_kw))
    return run(state, *args)


def report(idx, title, desc, in_desc, maps, yt, yj):
    print(f"[{idx}] {title}")
    print(f"      what      : {desc}")
    print(f"      input     : {in_desc} | dtype float64")
    print(f"      weights   : " + ("parameter-free" if not maps else "ported torch -> jax:"))
    for m in maps:
        print(f"                    - {m}")
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
    print(" torch <-> jax parity : shared/attention.py (core primitives)")
    print("=" * 84)
    print(" torch : renderer.attention_modules   | jax : shared.attention")
    print(" dtype : float64 both sides | ported weights | sequences (B,N,C), no layout transpose")
    print(f" pass  : max|torch - jax| < {ATOL:.0e}\n")

    R = []
    B, N, C, heads = 2, 16, 32, 4

    # -------------------------------------------------------------------------------
    # StandardUnifiedAttention: output x
    # -------------------------------------------------------------------------------
    t = TStd(C, heads).double().eval()
    j = StandardUnifiedAttention(C, heads, rngs=nnx.Rngs(0))
    maps = []
    for nm in ("q_proj", "k_proj", "v_proj", "proj"):
        tw, tb = getattr(t, nm).weight, getattr(t, nm).bias
        getattr(j, nm).kernel.value = jnp.asarray(to_np(tw).T)   # (out,in)->(in,out)
        getattr(j, nm).bias.value = jnp.asarray(to_np(tb))
        maps.append(f"{nm}.weight {_fmt(to_np(tw).shape)}->{_fmt(getattr(j,nm).kernel.value.shape)} [.T], bias")
    q, k, v = rnd((B, N, C), 0), rnd((B, N, C), 1), rnd((B, N, C), 2)
    with torch.no_grad():
        xt, at = t(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v))
    xj, aj = jf(j, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v))   # under jax.jit
    R.append(report(1, "StandardUnifiedAttention (output)",
                    "multi-head attention q,k,v -> projected output",
                    f"q,k,v {_fmt((B, N, C))} seeds 0/1/2", maps, to_np(xt), to_np(xj)))

    # -------------------------------------------------------------------------------
    # StandardUnifiedAttention: attn_map
    # -------------------------------------------------------------------------------
    R.append(report(2, "StandardUnifiedAttention (attn_map)",
                    "softmax(q k^T / sqrt(d)) attention weights",
                    f"q,k,v {_fmt((B, N, C))} seeds 0/1/2",
                    ["(same ported projections as [1])"], to_np(at), to_np(aj)))

    # -------------------------------------------------------------------------------
    # window_partition (parameter-free)
    # -------------------------------------------------------------------------------
    x = rnd((2, 8, 8, 4), 3)                                 # (B,H,W,C) — same layout both sides
    with torch.no_grad():
        wt = to_np(t_wp(torch.from_numpy(x), 4))
    wj = to_np(jax.jit(lambda a: window_partition(a, 4))(jnp.asarray(x)))   # pure fn under jax.jit
    R.append(report(3, "window_partition", "(B,H,W,C) -> (nW*B, ws*ws, C)",
                    "x (2, 8, 8, 4) seed 3", [], wt, wj))

    # -------------------------------------------------------------------------------
    # window_reverse (parameter-free)
    # -------------------------------------------------------------------------------
    with torch.no_grad():
        rt = to_np(t_wr(torch.from_numpy(wt), 4, 8, 8))
    rj = to_np(jax.jit(lambda a: window_reverse(a, 4, 8, 8))(jnp.asarray(wj)))   # pure fn under jax.jit
    R.append(report(4, "window_reverse", "inverse of window_partition",
                    "windows (8, 16, 4) from [3]", [], rt, rj))

    n_pass = sum(1 for ok, _ in R if ok)
    worst = max((mx for _, mx in R), default=0.0)
    print("=" * 84)
    print(f" SUMMARY : {n_pass}/{len(R)} PASS   |   worst max|Δ| = {worst:.2e}   |   threshold = {ATOL:.0e}")
    print(" (composite Transformer/Swin/Self/Cross blocks are smoke-tested above)")
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
