"""Attention family. torch: attention_modules.py (timm helpers inlined later)."""

from __future__ import annotations

from flax import nnx


def window_partition(x, window_size):
    """(B,H,W,C) -> (nW*B, ws*ws, C). torch: attention_modules.window_partition."""
    raise NotImplementedError


def window_reverse(windows, window_size, H, W):
    """Inverse of window_partition. torch: attention_modules.window_reverse."""
    raise NotImplementedError


class StandardUnifiedAttention(nnx.Module):
    """MHA cross/self attn; returns (x, attn_map). torch: StandardUnifiedAttention."""

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, query, key, value, mask=None):
        raise NotImplementedError


class GuidedResampler(nnx.Module):
    """Sparse gather warp (no params). EMERGENCY op (index-heavy). torch: GuidedResampler."""

    def __init__(self, dim, downsample_ratio=4, k_top_samples=1):
        raise NotImplementedError

    def __call__(self, v_high_feat, coarse_attn_map):
        raise NotImplementedError


class SwinUnifiedAttention(nnx.Module):
    """Windowed attn w/ relative position bias. torch: SwinUnifiedAttention."""

    def __init__(self, dim, num_heads, window_size, qkv_bias=True, attn_drop=0.0,
                 proj_drop=0.0, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, query, key, value, mask=None):
        raise NotImplementedError


class UnifiedTransformerBlock(nnx.Module):
    """Pre-norm transformer w/ learned pos-embeddings. torch: UnifiedTransformerBlock."""

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=2.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, query, key=None, value=None):
        raise NotImplementedError


class UnifiedSwinBlock(nnx.Module):
    """Swin block w/ cyclic shift. torch: UnifiedSwinBlock."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, query, key=None, value=None):
        raise NotImplementedError


class CrossAttention(nnx.Module):
    """The `imt` blocks — coarse/fine dispatch. torch: attention_modules.CrossAttention."""

    def __init__(self, config, dim, resolution, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def coarse_stage(self, A, B, C, attn=None):
        raise NotImplementedError

    def fine_stage(self, C, attn=None):
        raise NotImplementedError

    def __call__(self, A, B, C, D=None, attn=None):
        raise NotImplementedError


class SelfAttention(nnx.Module):
    """SynthesisNetwork transformer blocks (swin or standard). torch: SelfAttention."""

    def __init__(self, config, dim, resolution, *, rngs: nnx.Rngs):
        raise NotImplementedError

    def __call__(self, query, key=None, value=None):
        raise NotImplementedError
