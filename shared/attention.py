from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _split_heads(x, num_heads):
    # (B, N, C) -> (B, heads, N, head_dim)
    B, N, C = x.shape
    x = x.reshape(B, N, num_heads, C // num_heads)
    return jnp.transpose(x, (0, 2, 1, 3))

def merge_heads(x):
    """(B, heads, N, head_dim) -> (B, N, C)."""
    B, h, N, hd = x.shape
    return jnp.transpose(x, (0, 2, 1, 3)).reshape(B, N, h * hd)


def window_partition(x, ws):
    """(B, H, W, C) -> (num_windows*B, ws*ws, C)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(-1, ws * ws, C)


def window_reverse(windows, ws, H, W):
    """(num_windows*B, ws*ws, C) -> (B, H, W, C)."""
    B = windows.shape[0] // ((H // ws) * (W // ws))
    x = windows.reshape(B, H // ws, W // ws, ws, ws, -1)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(B, H, W, -1)

# -----------------------------------------------------------------------------
# MultiHead Attention
# -----------------------------------------------------------------------------
class StandardUnifiedAttention(nnx.Module):
    """Q = Query tensor (B, N, C)
       K = Key tensor (B, N, C)
       V = Value tensor (B, N, C)
       return: (B, N, C)
       attn_mask = mask tensor (B, N, N), The attn_map is returned because the renderer'scoarse stage reuses it in the fine stage
       where B=batches, N=no of tokens, C=channels
    """

    def __init__(self, dim, num_heads, qkv_bias=True, *, rngs: nnx.Rngs, attn_drop=0.0, proj_drop=0.0):
        assert dim % num_heads == 0, "dim must divide evenly into heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)

        self.attn_drop = nnx.Dropout(rate=attn_drop, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.proj_drop = nnx.Dropout(rate=proj_drop, rngs=rngs)

    def __call__(self, query, key, value, mask=None):
        B, N, C = query.shape

        q = _split_heads(self.q_proj(query), self.num_heads)
        k = _split_heads(self.k_proj(key), self.num_heads)
        v = _split_heads(self.v_proj(value), self.num_heads)

        attn = (q @ jnp.swapaxes(k, -1, -2)) * self.scale
        if mask is not None:                          # torch: masked_fill(mask == 0, -inf)
            attn = jnp.where(mask == 0, -jnp.inf, attn)
        attn_map = jax.nn.softmax(attn, axis=-1)
        attn_map_dropped = self.attn_drop(attn_map)

        x = attn_map_dropped @ v
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_map

# -----------------------------------------------------------------------------
# Transformer Block = Norm -> Attention -> Norm -> MLP -> Residual Connection
# -----------------------------------------------------------------------------
class UnifiedTransformerBlock(nnx.Module):
    """
    Transformer block for image feature maps:
    - Converts a feature map into a sequence of tokens
    - adds learned positional embeddings, applies self-attention or cross-attention
    - then feed-forward network with residual connections
    - return transformed feature map

    Input:
    query: (B, H, W, C)
    key: Optional (B, H, W, C)
    value: Optional (B, H, W, C)

    Output:
    Feature map of shape (B, H, W, C).
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=2.0, qkv_bias=True, drop=0., attn_drop=0., drop_path=0., *, rngs: nnx.Rngs):
        H = W = input_resolution
        self.H, self.W = H, W
        n_tokens = H * W # no of dimensions

        self.norm_q = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_ffn = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = StandardUnifiedAttention(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, rngs=rngs)

        self.q_pos_embedding = nnx.Param(jax.random.normal(rngs.params(), (1, n_tokens, dim)))
        self.k_pos_embedding = nnx.Param(jax.random.normal(rngs.params(), (1, n_tokens, dim)))

        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def _mlp(self, x):
        x = jax.nn.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def __call__(self, query, key=None, value=None):
        B, H, W, C = query.shape
        if key is None:   # measn self-attention as there is no key.
            key = value = query

        # feature map -> token sequence: (B, H, W, C) -> (B, N, C)
        q = query.reshape(B, H * W, C)
        k = key.reshape(B, H * W, C)
        v = value.reshape(B, H * W, C)

        shortcut = v   # residual is anchored on value
        q = self.norm_q(q + self.q_pos_embedding.value)
        k = self.norm_kv(k + self.k_pos_embedding.value)
        v = self.norm_kv(v)
        # normalized all k,q,v

        attn_out, _ = self.attn(q, k, v)
        x = shortcut + attn_out                      # residual 1
        x = x + self._mlp(self.norm_ffn(x))          # residual 2
        return x.reshape(B, H, W, C)



# ---------------------------------------------------------------------------
# Swin attention
# ---------------------------------------------------------------------------
def _relative_position_index(ws):
    """
    Computes the lookup index for relative position bias.
    - Computes the relative spatial offset between every pair of tokens
      in a window of size (ws * ws).
    - Converts each 2D offset into a unique integer index.
    - The resulting index is used to gather values from the learned
      relative position bias table during attention.
    - The output is a fixed, non-trainable lookup table.

    Input:
        ws: Window size.

    Returns:
        Relative position index of shape (ws², ws²) with dtype int32.
    """
    coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing="ij"))
    coords = coords.reshape(2, -1)
    rel = coords[:, :, None] - coords[:, None, :]
    rel = np.transpose(rel, (1, 2, 0))

    idx = (rel[..., 0] + ws - 1) * (2 * ws - 1) + (rel[..., 1] + ws - 1)
    return idx.astype(np.int32)

class SwinUnifiedAttention(nnx.Module):
    """
    Multi-head attention within a single window using learned relative position bias.

    - Projects the input into q,k,v representations.
    - Computes scaled dot-product attention within each window.
    - Adds a learned relative position bias to encode the spatial relationship
      between tokens.
    - Optionally applies an attention mask (used for shifted-window attention).
    - Applies attention and output projection dropout.
    - Projects the attended features back to the original embedding dimension.

    Input:
        query: Tensor of shape (num_windows * B, N, C).
        key: Tensor of shape (num_windows * B, N, C).
        value: Tensor of shape (num_windows * B, N, C).
        mask: Optional attention mask of shape (num_windows, N, N).

    Returns:
        Tensor of shape (num_windows * B, N, C) containing the attended
        window features.
    """

    def __init__(self, dim, num_heads, window_size, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.q = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.k = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.v = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

        self.attn_drop = nnx.Dropout(rate=attn_drop, rngs=rngs)
        self.proj_drop = nnx.Dropout(rate=proj_drop, rngs=rngs)

        n_rel = (2 * window_size - 1) ** 2
        table = jax.random.normal(rngs.params(), (n_rel, num_heads)) * 0.02  # ~ torch trunc_normal_(std=.02)

        self.rel_bias_table = nnx.Param(table)
        self.rel_index = _relative_position_index(window_size)   # numpy constant, not a Param

    def __call__(self, query, key, value, mask=None, *, deterministic=True):
        Bn, N, C = query.shape

        q = _split_heads(self.q(query), self.num_heads)
        k = _split_heads(self.k(key), self.num_heads)
        v = _split_heads(self.v(value), self.num_heads)

        attn = (q * self.scale) @ jnp.swapaxes(k, -1, -2)

        # relative position bias -> (heads, N, N)
        bias = self.rel_bias_table.value[self.rel_index.reshape(-1)]
        bias = jnp.transpose(bias.reshape(N, N, self.num_heads), (2, 0, 1))
        attn = attn + bias[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(Bn // nW, nW, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, deterministic=deterministic)

        x = merge_heads(attn @ v)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)
        return x

# --------------------------------------------------------------------------------------------------------------------
# Swin Block
# --------------------------------------------------------------------------------------------------------------------
def _build_attn_mask(H, W, ws, shift):
    """Mask so a shifted window never attends across the wrapped image edge.
    Returns numpy (num_windows, ws*ws, ws*ws) of {0, -100}, or None if shift==0."""
    if shift == 0:
        return None
    img = np.zeros((1, H, W, 1), dtype=np.float32)
    slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    cnt = 0
    for hs in slices:
        for wsl in slices:
            img[:, hs, wsl, :] = cnt
            cnt += 1
    mw = np.asarray(window_partition(jnp.asarray(img), ws)).reshape(-1, ws * ws)  # (nW, ws^2)
    m = mw[:, None, :] - mw[:, :, None]          # (nW, ws^2, ws^2)
    return np.where(m != 0, -100.0, 0.0).astype(np.float32)

class UnifiedSwinBlock(nnx.Module):
    """
    Swin Transformer block for image feature maps.
    - Applies pre-normalization to the input feature maps.
    - Optionally performs a cyclic shift to enable shifted-window attention.
    - Partitions the feature maps into non-overlapping windows and applies
    window-based multi-head self- or cross-attention with relative position bias.
    - Reconstructs the feature map, reverses the cyclic shift (if applied),
    and applies residual connections and a feed-forward network (MLP).

    Input:
        query: Feature map of shape (B, H, W, C).
        key: Optional feature map of shape (B, H, W, C). If None, self-attention
            is performed using `query`.
        value: Optional feature map of shape (B, H, W, C). If None, `query` is
            used as the value.

    Returns:
        Feature map of shape (B, H, W, C) containing the transformed features.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2.0, drop=0.0, attn_drop=0.0, *, rngs: nnx.Rngs):
        if input_resolution <= window_size:
            shift_size = 0
            window_size = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm_q = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_ffn = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = SwinUnifiedAttention(dim, num_heads, window_size,
                                         attn_drop=attn_drop, proj_drop=drop, rngs=rngs)
        hidden = int(dim * mlp_ratio)

        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
        self.attn_mask = _build_attn_mask(input_resolution, input_resolution,
                                          window_size, shift_size)  # numpy const or None

    def _mlp(self, x):
        return self.drop(self.fc2(jax.nn.gelu(self.fc1(x))))   # torch swin MLP: Linear,GELU,Linear,Dropout

    def __call__(self, query, key=None, value=None):
        B, H, W, C = query.shape
        ws, shift = self.window_size, self.shift_size
        if key is None:
            key = value = query

        q = self.norm_q(query.reshape(B, H * W, C)).reshape(B, H, W, C)
        k = self.norm_kv(key.reshape(B, H * W, C)).reshape(B, H, W, C)
        v = self.norm_kv(value.reshape(B, H * W, C)).reshape(B, H, W, C)
        shortcut = value.reshape(B, H * W, C)

        if shift > 0:    
            q, k, v = [jnp.roll(t, (-shift, -shift), axis=(1, 2)) for t in (q, k, v)]

        # windowed attention
        aw = self.attn(window_partition(q, ws), window_partition(k, ws),
                       window_partition(v, ws), mask=self.attn_mask)
        x = window_reverse(aw, ws, H, W)           # (B, H, W, C)

        if shift > 0:                              # undo the shift
            x = jnp.roll(x, (shift, shift), axis=(1, 2))

        x = shortcut + x.reshape(B, H * W, C)      # residual 1
        x = x + self._mlp(self.norm_ffn(x))        # residual 2
        return x.reshape(B, H, W, C)


# ---------------------------------------------------------------------------------------------
# GuidedResampler
# ---------------------------------------------------------------------------------------------
class GuidedResampler(nnx.Module):
    """Warp a high-res value map using a coarse (low-res) attention map: 
    - for each high-res position, gather the top-k coarse matches (expanded to their r*r high-res blocks) and take their softmax-weighted average. 
    - No parameters, so __init__ takes no rngs."""

    def __init__(self, dim, downsample_ratio=4, k_top_samples=1):
        self.ratio = downsample_ratio
        self.k = k_top_samples

    def __call__(self, v_high_feat, coarse_attn_map):
        B, H, W, C = v_high_feat.shape
        r, k = self.ratio, self.k
        H_low, W_low = H // r, W // r
        N_high, N_low = H * W, H_low * W_low
        assert coarse_attn_map.shape == (B, N_low, N_low), coarse_attn_map.shape

        v_seq = v_high_feat.reshape(B, N_high, C)             # NHWC flattens straight to a sequence

        # top-k coarse keys per low-res query
        topk_vals, topk_idx = jax.lax.top_k(coarse_attn_map, k)   # (B, N_low, k)
        tl_row = (topk_idx // W_low) * r                          # top-left of each r*r block
        tl_col = (topk_idx % W_low) * r

        # expand each key to its r*r high-res block -> flat indices (B, N_low, k*r*r)
        dh, dw = jnp.meshgrid(jnp.arange(r), jnp.arange(r), indexing="ij")
        delta = jnp.stack([dh.reshape(-1), dw.reshape(-1)], axis=-1)  # (r*r, 2)
        s_row = tl_row[..., None] + delta[:, 0]
        s_col = tl_col[..., None] + delta[:, 1]
        sparse_1d = (s_row * W + s_col).reshape(B, N_low, k * r * r)

        # each high-res position -> its low-res cell -> gather that cell's indices
        qh, qw = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        low_cell = ((qh // r) * W_low + (qw // r)).reshape(N_high)    # (N_high,)
        K = sparse_1d.shape[-1]
        g_idx = jnp.broadcast_to(low_cell[None, :, None], (B, N_high, K))
        final_idx = jnp.take_along_axis(sparse_1d, g_idx, axis=1)     # (B, N_high, K)

        # gather the actual high-res values (vmap the batch)
        v_sparse = jax.vmap(lambda seq, idx: seq[idx])(v_seq, final_idx)  # (B, N_high, K, C)

        # softmax weights over the k coarse matches, spread across the r*r block
        w = jax.nn.softmax(topk_vals, axis=-1)                        # (B, N_low, k)
        wg_idx = jnp.broadcast_to(low_cell[None, :, None], (B, N_high, k))
        w_high = jnp.take_along_axis(w, wg_idx, axis=1)              # (B, N_high, k)
        v_sparse = v_sparse.reshape(B, N_high, k, r * r, C)
        w_bcast = (w_high / (r * r)).reshape(B, N_high, k, 1, 1)
        warped = (v_sparse * w_bcast).sum(axis=(2, 3))              # (B, N_high, C)
        return warped.reshape(B, H, W, C)

# ----------------------------------------------------------------------------------------------
# Self Attention
# ----------------------------------------------------------------------------------------------
class SelfAttention(nnx.Module):
    def __init__(self, dim, resolution, num_heads, window_size, swin_res_threshold,
                 *, rngs: nnx.Rngs):
        if resolution >= swin_res_threshold:
            self.blocks = [
                UnifiedSwinBlock(dim, resolution, num_heads, window_size, shift_size=0, rngs=rngs),
                UnifiedSwinBlock(dim, resolution, num_heads, window_size,
                                 shift_size=window_size // 2, rngs=rngs),
            ]
        else:
            self.blocks = [UnifiedTransformerBlock(dim, resolution, num_heads, rngs=rngs)]

    def __call__(self, query, key=None, value=None):
        if key is not None:                       # cross: q,k fixed, value threads through
            out = value
            for b in self.blocks:
                out = b(query, key, out)
            return out
        x = query                                 # self-attention then
        for b in self.blocks:
            x = b(x)
        return x

# ----------------------------------------------------------------------------------------------
# Cross Attention
# ----------------------------------------------------------------------------------------------
class CrossAttention(nnx.Module):
    def __init__(self, dim, resolution, num_heads, swin_res_threshold, *, rngs: nnx.Rngs):
        self.is_standard = resolution < swin_res_threshold
        if self.is_standard:
            self.block_efc = StandardUnifiedAttention(dim, num_heads, rngs=rngs)
        else:
            ratio = 2 * (resolution / swin_res_threshold)
            assert ratio >= 1 and float(ratio).is_integer(), "fine res must be a multiple of anchor"
            self.block = GuidedResampler(dim, downsample_ratio=int(ratio))

    def coarse_stage(self, A, Bk, Cv):
        B, H, W, C = A.shape
        out, attn_map = self.block_efc(A.reshape(B, H * W, C),
                                       Bk.reshape(B, H * W, C),
                                       Cv.reshape(B, H * W, C))
        return out.reshape(B, H, W, C), attn_map

    def fine_stage(self, Cv, attn):
        return self.block(Cv, attn.mean(axis=1))   # average the coarse map over heads





















