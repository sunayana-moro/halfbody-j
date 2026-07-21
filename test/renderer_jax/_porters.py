"""Shared torch -> jax weight porters + helpers for renderer_jax parity tests.

Atomic porters compose into whole-network porters. All follow the layout rules:
  conv   torch (O,I,H,W) -> flax (H,W,I,O)   [.transpose(2,3,1,0)]
  linear torch (O,I)     -> flax (I,O)        [.T]
  EqualLinear weight stored (in,out) -> torch (out,in).T
  FusedLeakyReLU bias (1,C,1,1) -> (1,1,1,C)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import torch


def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def rnd(shape, seed):
    return np.random.RandomState(seed).randn(*shape).astype(np.float64)


def fmt(s):
    return "(" + ", ".join(str(d) for d in s) + ")"


# ---- atomic ----
def port_conv(tc, jc):
    """torch nn.Conv2d -> flax nnx.Conv (handles optional bias)."""
    jc.kernel.value = jnp.asarray(to_np(tc.weight).transpose(2, 3, 1, 0))   # OIHW -> HWIO
    if getattr(tc, "bias", None) is not None and jc.bias is not None:
        jc.bias.value = jnp.asarray(to_np(tc.bias))


def port_bn(tn, jn):
    """torch BatchNorm2d -> flax nnx.BatchNorm."""
    jn.scale.value = jnp.asarray(to_np(tn.weight))
    jn.bias.value = jnp.asarray(to_np(tn.bias))
    jn.mean.value = jnp.asarray(to_np(tn.running_mean))
    jn.var.value = jnp.asarray(to_np(tn.running_var))


def port_normlayer(tnl, jnl):
    """torch modules.NormLayer(batch) -> jax NormLayer (delegates to BN)."""
    port_bn(tnl.norm, jnl.norm)


def port_equal_linear(tl, jl):
    jl.weight.value = jnp.asarray(to_np(tl.weight).T)          # (O,I) -> (I,O)
    if tl.bias is not None and jl.bias is not None:
        jl.bias.value = jnp.asarray(to_np(tl.bias))


def port_equal_conv(tc, jc):
    jc.weight.value = jnp.asarray(to_np(tc.weight).transpose(2, 3, 1, 0))   # OIHW -> HWIO
    if tc.bias is not None and jc.bias is not None:
        jc.bias.value = jnp.asarray(to_np(tc.bias))


def port_fused_lrelu(ta, ja):
    ja.bias.value = jnp.asarray(to_np(ta.bias).reshape(1, 1, 1, -1))        # (1,C,1,1)->(1,1,1,C)


def port_conv_layer(tcl, jcl):
    """torch lia ConvLayer (nn.Sequential) -> jax ConvLayer. Blur is param-free."""
    # torch EqualConv2d / FusedLeakyReLU live inside the Sequential; match by type name
    for layer in tcl:
        name = type(layer).__name__
        if name == "EqualConv2d":
            jcl.conv.weight.value = jnp.asarray(to_np(layer.weight).transpose(2, 3, 1, 0))
            if getattr(layer, "bias", None) is not None and jcl.conv.bias is not None:
                jcl.conv.bias.value = jnp.asarray(to_np(layer.bias))
        elif name == "FusedLeakyReLU":
            jcl.act.bias.value = jnp.asarray(to_np(layer.bias).reshape(1, 1, 1, -1))
        # Blur / ScaledLeakyReLU: no params


def port_conv_block(tcb, jcb):
    port_conv(tcb.conv, jcb.conv)          # Conv2d bias=False both sides
    port_normlayer(tcb.norm, jcb.norm)


def port_feat_res_block(tfr, jfr):
    port_conv_block(tfr.conv1, jfr.conv1)
    port_conv_block(tfr.conv2, jfr.conv2)


def port_res_block(trb, jrb):
    """torch modules.ResBlock (lia ConvLayer x3)."""
    port_conv_layer(trb.conv1, jrb.conv1)
    port_conv_layer(trb.conv2, jrb.conv2)
    port_conv_layer(trb.skip, jrb.skip)


def port_conv_res_block(t, j):
    """ConvResBlock / DownConvResBlock / UpConvResBlock — same param structure
    (avgpool / upsample are param-free)."""
    port_conv(t.conv1, j.conv1)
    port_normlayer(t.norm, j.norm)
    port_conv(t.conv2, j.conv2)
    port_feat_res_block(t.feat_res_block1, j.feat_res_block1)
    port_feat_res_block(t.feat_res_block2, j.feat_res_block2)


def port_modulated_conv(tmc, jmc):
    jmc.weight.value = jnp.asarray(to_np(tmc.weight)[0].transpose(2, 3, 1, 0))   # (1,O,I,k,k)->(k,k,I,O)
    jmc.modulation.weight.value = jnp.asarray(to_np(tmc.modulation.weight).T)
    jmc.modulation.bias.value = jnp.asarray(to_np(tmc.modulation.bias))


def port_styled_conv(tsc, jsc):
    port_modulated_conv(tsc.conv, jsc.conv)
    jsc.noise.weight.value = jnp.asarray(to_np(tsc.noise.weight))                # scalar (1,)
    port_fused_lrelu(tsc.activate, jsc.activate)


# ---- network-level ----
def port_identity_encoder(t, j):
    port_conv(t.initial_conv[0], j.initial_conv)     # Conv2d (has bias)
    port_bn(t.initial_conv[1], j.initial_norm)       # BatchNorm2d
    port_conv_res_block(t.down_block_0, j.down_block_0)
    for tb, jb in zip(t.down_blocks, j.down_blocks):
        port_conv_res_block(tb, jb)
    # t.equalconv unused in forward -> skip
    for tl, jl in zip(t.linear_layers, j.linear_layers):
        port_equal_linear(tl, jl)
    port_equal_linear(t.final_linear, j.final_linear)


def port_motion_encoder(t, j):
    port_conv(t.conv1, j.conv1)                      # Conv2d (has bias)
    for trb, jrb in zip(t.res_blocks, j.res_blocks):
        port_res_block(trb, jrb)
    port_equal_conv(t.equalconv, j.equalconv)        # USED
    for tl, jl in zip(t.linear_layers, j.linear_layers):
        port_equal_linear(tl, jl)
    port_equal_linear(t.final_linear, j.final_linear)


def port_motion_decoder(t, j):
    # torch const (1,C,4,4) NCHW -> jax (1,4,4,C) NHWC
    j.const.value = jnp.asarray(to_np(t.const).transpose(0, 2, 3, 1))
    for tsc, jsc in zip(t.style_conv_layers, j.style_conv_layers):
        port_styled_conv(tsc, jsc)
