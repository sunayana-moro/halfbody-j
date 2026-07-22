"""One-time torch -> jax checkpoint migration for the renderer.

Reads renderer.ckpt (a torch artifact) and copies its trained weights into the
JAX/Flax IMTRenderer, applying the layout translations (conv OIHW->HWIO, linear
(O,I)->(I,O), FusedLeakyReLU bias (1,C,1,1)->(1,1,1,C), etc.)
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import torch

from renderer_jax.renderer import IMTRenderer


# =============================================================================
# Weight porting (torch -> jax).  torch (O,I,H,W)->flax (H,W,I,O); linear (O,I)->(I,O).
# =============================================================================
def _np(x):
    return x.detach().cpu().numpy()

# ---- atomic ----
def port_conv(tc, jc):                                   # nn.Conv2d -> nnx.Conv
    jc.kernel.value = jnp.asarray(_np(tc.weight).transpose(2, 3, 1, 0))
    if getattr(tc, "bias", None) is not None and jc.bias is not None:
        jc.bias.value = jnp.asarray(_np(tc.bias))

def port_bn(tn, jn):                                     # BatchNorm2d -> nnx.BatchNorm
    jn.scale.value = jnp.asarray(_np(tn.weight))
    jn.bias.value = jnp.asarray(_np(tn.bias))
    jn.mean.value = jnp.asarray(_np(tn.running_mean))
    jn.var.value = jnp.asarray(_np(tn.running_var))

def port_normlayer(tnl, jnl):                            # NormLayer(batch) -> NormLayer
    port_bn(tnl.norm, jnl.norm)

def port_layernorm(tln, jln):                            # nn.LayerNorm -> nnx.LayerNorm
    jln.scale.value = jnp.asarray(_np(tln.weight))
    jln.bias.value = jnp.asarray(_np(tln.bias))

def port_linear(tl, jl):                                 # nn.Linear -> nnx.Linear
    jl.kernel.value = jnp.asarray(_np(tl.weight).T)      # (O,I) -> (I,O)
    if tl.bias is not None and jl.bias is not None:
        jl.bias.value = jnp.asarray(_np(tl.bias))

def port_equal_linear(tl, jl):                           # EqualLinear (weight stored (in,out))
    jl.weight.value = jnp.asarray(_np(tl.weight).T)
    if tl.bias is not None and jl.bias is not None:
        jl.bias.value = jnp.asarray(_np(tl.bias))

def port_equal_conv(tc, jc):                             # EqualConv2d
    jc.weight.value = jnp.asarray(_np(tc.weight).transpose(2, 3, 1, 0))
    if tc.bias is not None and jc.bias is not None:
        jc.bias.value = jnp.asarray(_np(tc.bias))

def port_fused_lrelu(ta, ja):                            # FusedLeakyReLU bias (1,C,1,1)->(1,1,1,C)
    ja.bias.value = jnp.asarray(_np(ta.bias).reshape(1, 1, 1, -1))

def port_conv_layer(tcl, jcl):                           # lia ConvLayer (nn.Sequential)
    for layer in tcl:
        name = type(layer).__name__
        if name == "EqualConv2d":
            jcl.conv.weight.value = jnp.asarray(_np(layer.weight).transpose(2, 3, 1, 0))
            if getattr(layer, "bias", None) is not None and jcl.conv.bias is not None:
                jcl.conv.bias.value = jnp.asarray(_np(layer.bias))
        elif name == "FusedLeakyReLU":
            jcl.act.bias.value = jnp.asarray(_np(layer.bias).reshape(1, 1, 1, -1))
        # Blur / ScaledLeakyReLU: no params

def port_conv_block(tcb, jcb):
    port_conv(tcb.conv, jcb.conv)
    port_normlayer(tcb.norm, jcb.norm)

def port_feat_res_block(tfr, jfr):
    port_conv_block(tfr.conv1, jfr.conv1)
    port_conv_block(tfr.conv2, jfr.conv2)

def port_res_block(trb, jrb):                            # modules.ResBlock (ConvLayer x3)
    port_conv_layer(trb.conv1, jrb.conv1)
    port_conv_layer(trb.conv2, jrb.conv2)
    port_conv_layer(trb.skip, jrb.skip)

def port_conv_res_block(t, j):                           # Conv/Down/Up-ConvResBlock
    port_conv(t.conv1, j.conv1)
    port_normlayer(t.norm, j.norm)
    port_conv(t.conv2, j.conv2)
    port_feat_res_block(t.feat_res_block1, j.feat_res_block1)
    port_feat_res_block(t.feat_res_block2, j.feat_res_block2)

def port_modulated_conv(tmc, jmc):
    jmc.weight.value = jnp.asarray(_np(tmc.weight)[0].transpose(2, 3, 1, 0))   # (1,O,I,k,k)->(k,k,I,O)
    port_equal_linear(tmc.modulation, jmc.modulation)


def port_styled_conv(tsc, jsc):
    port_modulated_conv(tsc.conv, jsc.conv)
    jsc.noise.weight.value = jnp.asarray(_np(tsc.noise.weight))
    port_fused_lrelu(tsc.activate, jsc.activate)


# ---- attention (UNVERIFIED) ----
def port_std_attn(t, j):                                 # StandardUnifiedAttention
    port_linear(t.q_proj, j.q_proj)
    port_linear(t.k_proj, j.k_proj)
    port_linear(t.v_proj, j.v_proj)
    port_linear(t.proj, j.proj)


def port_swin_attn(t, j):                                # SwinUnifiedAttention
    port_linear(t.q, j.q)
    port_linear(t.k, j.k)
    port_linear(t.v, j.v)
    port_linear(t.proj, j.proj)
    j.rel_bias_table.value = jnp.asarray(_np(t.relative_position_bias_table))  # (n_rel, heads) direct


def port_unified_transformer_block(t, j):
    port_layernorm(t.norm_q, j.norm_q)
    port_layernorm(t.norm_kv, j.norm_kv)
    port_layernorm(t.norm_ffn, j.norm_ffn)
    port_std_attn(t.attn, j.attn)
    j.q_pos_embedding.value = jnp.asarray(_np(t.q_pos_embedding))   # (1,N,dim) direct
    j.k_pos_embedding.value = jnp.asarray(_np(t.k_pos_embedding))
    port_linear(t.mlp[0], j.fc1)                          # Sequential(Linear, GELU, Dropout, Linear, Dropout)
    port_linear(t.mlp[3], j.fc2)


def port_unified_swin_block(t, j):
    port_layernorm(t.norm_q, j.norm_q)
    port_layernorm(t.norm_kv, j.norm_kv)
    port_layernorm(t.norm_ffn, j.norm_ffn)
    port_swin_attn(t.attn, j.attn)
    port_linear(t.mlp[0], j.fc1)                          # Sequential(Linear, GELU, Linear, Dropout)
    port_linear(t.mlp[2], j.fc2)


def port_self_attention(t, j):
    for tb, jb in zip(t.blocks, j.blocks):
        if type(jb).__name__ == "UnifiedSwinBlock":
            port_unified_swin_block(tb, jb)
        else:
            port_unified_transformer_block(tb, jb)


def port_cross_attention(t, j):
    if j.is_standard:
        port_std_attn(t.block_efc, j.block_efc)
    # else: fine stage uses GuidedResampler -> no parameters


# ---- networks ----
def port_identity_encoder(t, j):
    port_conv(t.initial_conv[0], j.initial_conv)         # Conv2d (bias)
    port_bn(t.initial_conv[1], j.initial_norm)           # BatchNorm2d
    port_conv_res_block(t.down_block_0, j.down_block_0)
    for tb, jb in zip(t.down_blocks, j.down_blocks):
        port_conv_res_block(tb, jb)
    # t.equalconv unused in forward -> skip
    for tl, jl in zip(t.linear_layers, j.linear_layers):
        port_equal_linear(tl, jl)
    port_equal_linear(t.final_linear, j.final_linear)


def port_motion_encoder(t, j):
    port_conv(t.conv1, j.conv1)                          # Conv2d (bias)
    for trb, jrb in zip(t.res_blocks, j.res_blocks):
        port_res_block(trb, jrb)
    port_equal_conv(t.equalconv, j.equalconv)            # USED
    for tl, jl in zip(t.linear_layers, j.linear_layers):
        port_equal_linear(tl, jl)
    port_equal_linear(t.final_linear, j.final_linear)


def port_motion_decoder(t, j):
    j.const.value = jnp.asarray(_np(t.const).transpose(0, 2, 3, 1))   # (1,C,4,4)->(1,4,4,C)
    for tsc, jsc in zip(t.style_conv_layers, j.style_conv_layers):
        port_styled_conv(tsc, jsc)


def port_synthesis(t, j):
    for tb, jb in zip(t.upconv_blocks, j.upconv_blocks):
        port_conv_res_block(tb, jb)
    for tb, jb in zip(t.resblocks, j.resblocks):
        port_conv_res_block(tb, jb)
    for tb, jb in zip(t.transformer_blocks, j.transformer_blocks):
        port_self_attention(tb, jb)
    port_conv(t.final_conv[1], j.final_conv)             # Sequential(LeakyReLU, Conv2d, PixelShuffle, Sigmoid)


def port_identidy_adaptive(t, j):
    port_equal_linear(t.in_layer, j.in_layer)
    for tl, jl in zip(t.linear_layers, j.linear_layers):
        port_equal_linear(tl, jl)
    port_equal_linear(t.final_linear, j.final_linear)


def port_imt_renderer(t, j):
    port_identity_encoder(t.dense_feature_encoder, j.dense_feature_encoder)
    port_motion_encoder(t.latent_token_encoder, j.latent_token_encoder)
    port_motion_decoder(t.latent_token_decoder, j.latent_token_decoder)
    port_synthesis(t.frame_decoder, j.frame_decoder)
    port_identidy_adaptive(t.adapt, j.adapt)
    for tb, jb in zip(t.imt, j.imt):
        port_cross_attention(tb, jb)


# =============================================================================
# Load model: build jax IMTRenderer, load torch ckpt, port weights
# =============================================================================
def load_model(ckpt_path, config):
    from renderer.models import IMTRenderer as TIMTRenderer   # torch oracle (has .ckpt keys)

    jax_model = IMTRenderer(config, rngs=nnx.Rngs(0))

    args = SimpleNamespace(num_heads=config.num_heads, window_size=config.window_size,
                           swin_res_threshold=config.swin_res_threshold)
    tmodel = TIMTRenderer(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace("gen.", ""): v for k, v in sd.items() if k.startswith("gen.")} or sd
    tmodel.load_state_dict(sd, strict=False)
    tmodel.double().eval()                                # match parity-style precision

    port_imt_renderer(tmodel, jax_model)
    print(f"[INFO] loaded + ported weights from {ckpt_path}")
    return jax_model
