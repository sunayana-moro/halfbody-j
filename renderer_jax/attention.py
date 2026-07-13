"""Attention family. torch: attention_modules.py (timm helpers inlined later)."""

from __future__ import annotations

from flax import nnx


def window_partition(x, window_size):



def window_reverse(windows, window_size, H, W):



class StandardUnifiedAttention(nnx.Module):



class GuidedResampler(nnx.Module):



class SwinUnifiedAttention(nnx.Module):



class UnifiedTransformerBlock(nnx.Module):



class UnifiedSwinBlock(nnx.Module):



class CrossAttention(nnx.Module):



class SelfAttention(nnx.Module):

