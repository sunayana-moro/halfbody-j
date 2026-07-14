"""
Tests the ../shared/attention.py
"""

import os
import sys

# put the repo root on sys.path so `import shared.attention` resolves
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
import jax.numpy as jnp
from flax import nnx

from shared.attention import (
    StandardUnifiedAttention,
    UnifiedTransformerBlock,
    UnifiedSwinBlock,
    SelfAttention,
    CrossAttention,
    window_partition,
    window_reverse,
)


def randn(shape, seed):
    """A fresh seeded random matrix (deterministic per seed)."""
    return jax.random.normal(nnx.Rngs(seed).params(), shape)


def main():
    rngs = nnx.Rngs(0)
    B, C, heads = 2, 32, 4

    # StandardUnifiedAttention: attention rows must sum to 1
    attn = StandardUnifiedAttention(C, heads, rngs=rngs)
    seq = randn((B, 64, C), 1)
    _, amap = attn(seq, seq, seq)
    assert jnp.allclose(amap.sum(-1), 1.0, atol=1e-5)
    print(f"StandardUnifiedAttention: attn_map {amap.shape}, rows sum to 1  OK")

    # UnifiedTransformerBlock (low-res path): shape preserved, self & cross
    x8 = randn((B, 8, 8, C), 2)
    tblock = UnifiedTransformerBlock(C, 8, heads, rngs=rngs)
    assert tblock(x8).shape == x8.shape
    assert tblock(x8, key=x8, value=x8).shape == x8.shape
    print(f"UnifiedTransformerBlock: {x8.shape} -> {tblock(x8).shape}  (self & cross)  OK")

    # UnifiedSwinBlock (high-res path): unshifted and shifted
    x16 = randn((B, 16, 16, C), 3)
    swin0 = UnifiedSwinBlock(C, 16, heads, window_size=4, shift_size=0, rngs=rngs)
    swin1 = UnifiedSwinBlock(C, 16, heads, window_size=4, shift_size=2, rngs=rngs)
    assert swin0(x16).shape == x16.shape and swin1(x16).shape == x16.shape
    print(f"UnifiedSwinBlock: {x16.shape} -> {swin0(x16).shape}  (shift 0 & 2)  OK")

    # SelfAttention dispatcher: both paths (swin at high res, transformer at low res)
    sa_swin = SelfAttention(C, 16, heads, window_size=4, swin_res_threshold=8, rngs=rngs)
    sa_std = SelfAttention(C, 8, heads, window_size=4, swin_res_threshold=16, rngs=rngs)
    assert sa_swin(x16).shape == x16.shape and sa_std(x8).shape == x8.shape
    print(f"SelfAttention: swin {sa_swin(x16).shape}, std {sa_std(x8).shape}  OK")

    # CrossAttention coarse -> fine (the IMT reuse of the coarse map)
    A = randn((B, 4, 4, C), 4)
    Bk = randn((B, 4, 4, C), 5)
    Cv = randn((B, 4, 4, C), 6)
    ca_coarse = CrossAttention(C, 4, heads, swin_res_threshold=8, rngs=rngs)
    coarse_out, attn_map = ca_coarse.coarse_stage(A, Bk, Cv)          # attn_map (B, heads, 16, 16)
    ca_fine = CrossAttention(C, 16, heads, swin_res_threshold=8, rngs=rngs)  # ratio = 2*(16/8) = 4
    fine_out = ca_fine.fine_stage(randn((B, 16, 16, C), 7), attn_map)
    assert fine_out.shape == (B, 16, 16, C)
    print(f"CrossAttention: coarse {coarse_out.shape} (+map {attn_map.shape}) -> fine {fine_out.shape}  OK")

    # window helpers round-trip
    assert jnp.allclose(window_reverse(window_partition(x16, 4), 4, 16, 16), x16)
    print("window_partition/reverse round-trip  OK")

    print("all shared/attention blocks OK")


if __name__ == "__main__":
    main()
