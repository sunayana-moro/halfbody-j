from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from shared.stylegan import StyledConv
from shared.resblock import UpConvResBlock, ConvResBlock
from shared.attention import SelfAttention

# ------------------------------------------------------------------------------------------------------------
# pixel_shuffle is a replacedment of nn.PixelShuffle()
# ------------------------------------------------------------------------------------------------------------
def pixel_shuffle(x, r):
    """
    python replacement for nn.PixelShuffle()
    """
    B, H, W, C = x.shape
    Co = C // (r * r)
    x = x.reshape(B, H, W, Co, r, r)              
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))     
    return x.reshape(B, H * r, W * r, Co)

# ------------------------------------------------------------------------------------------------------------
# Motion Decoder
# ------------------------------------------------------------------------------------------------------------
class MotionDecoder(nnx.Module):
    """
    -> Constant (4*4*32)
    -> StyledConv (4*4*512)
    -> Upsample StyledConv (8*8*512)
    -> StyledConv (8*8*512)
    -> StyledConv (8*8*512)   ← m1
    -> Upsample StyledConv (16*16*512)
    -> StyledConv (16*16*512)
    -> StyledConv (16*16*512)  ← m2
    -> Upsample StyledConv (32*32*256)
    -> StyledConv (32*32*256)
    -> StyledConv (32*32*256)  ← m3
    -> Upsample StyledConv (64*64*128)
    -> StyledConv (64*64*128)
    -> StyledConv (64*64*128)  ← m4
    """

    _SPEC = (
        (32, 512, False), (512, 512, True), (512, 512, False), (512, 512, False),   # 0,1,2,3 -> m1
        (512, 512, True), (512, 512, False), (512, 512, False),                     # 4,5,6 -> m2
        (512, 256, True), (256, 256, False), (256, 256, False),                     # 7,8,9 -> m3
        (256, 128, True), (128, 128, False), (128, 128, False),                     # 10,11,12 -> m4
    )

    def __init__(self, latent_dim=32, const_dim=32, *, rngs: nnx.Rngs):
        # torch const is (1, C, 4, 4) NCHW -> (1, 4, 4, C) NHWC
        self.const = nnx.Param(jax.random.normal(rngs.params(), (1, 4, 4, const_dim)))
        self.style_conv_layers = [
            StyledConv(i, o, 3, latent_dim, upsample=u, rngs=rngs) for (i, o, u) in self._SPEC
        ]

    def __call__(self, t):
        B = t.shape[0]
        c = self.const.value
        x = jnp.broadcast_to(c, (B, c.shape[1], c.shape[2], c.shape[3])) 
        m1 = m2 = m3 = m4 = None
        for idx, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)                          
            if idx == 3:
                m1 = x
            elif idx == 6:
                m2 = x
            elif idx == 9:
                m3 = x
            elif idx == 12:
                m4 = x
        return m1, m2, m3, m4

class SynthesisNetwork(nnx.Module):
    """
    Aligned Feature Pyramid (8*8*512)
    -> UpConvResBlock (16*16*512)
    -> Concatenate Skip Feature (16*16*1024)
    -> ConvResBlock (16*16*512)
    -> SelfAttention
    -> UpConvResBlock (32*32*256)
    -> Concatenate Skip Feature (32*32*512)
    -> ConvResBlock (32*32*256)
    -> SelfAttention
    -> UpConvResBlock (64*64*128)
    -> Concatenate Skip Feature (64*64*256)
    -> ConvResBlock (64*64*128)
    -> SelfAttention
    -> UpConvResBlock (128*128*64)
    -> Concatenate Skip Feature (128*128*128)
    -> ConvResBlock (128*128*64)
    -> SelfAttention
    -> UpConvResBlock (256*256*32)
    -> Concatenate Skip Feature (256*256*64)
    -> ConvResBlock (256*256*32)
    -> SelfAttention
    -> LeakyReLU
    -> 3*3 Conv (256*256*12)
    -> PixelShuffle *2 (512*512*3)
    -> Sigmoid
    -> Synthesized RGB Image
    """
    def __init__(self, config, feature_dims, spatial_dims, *, rngs: nnx.Rngs):
        fdr = tuple(feature_dims)[::-1]              # [512,512,256,128,64,32]
        sdr = tuple(spatial_dims)[::-1]              # [8,16,32,64,128,256]
        n = len(fdr) - 1                             

        self.upconv_blocks = [UpConvResBlock(fdr[i], fdr[i + 1], rngs=rngs) for i in range(n)]
        self.resblocks = [ConvResBlock(fdr[i + 1] * 2, fdr[i + 1], rngs=rngs) for i in range(n)]
        self.transformer_blocks = [
            SelfAttention(fdr[i + 1], sdr[i + 1], config.num_heads,
                          window_size=config.window_size,
                          swin_res_threshold=config.swin_res_threshold, rngs=rngs)
            for i in range(n)
        ]
        self.final_conv = nnx.Conv(fdr[-1], 3 * 4, (3, 3), strides=1, padding=1, rngs=rngs)

    def __call__(self, features_align, use_running_average=None):
        x = features_align[0]
        for i in range(len(self.upconv_blocks)):
            x = self.upconv_blocks[i](x, use_running_average=use_running_average)
            x = jnp.concatenate([x, features_align[i + 1]], axis=-1)   # skip concat (channel)
            x = self.resblocks[i](x, use_running_average=use_running_average)
            x = self.transformer_blocks[i](x)                          # self-attention (no BN)
        # final_conv: LeakyReLU(0.2) -> Conv2d(fdr[-1] -> 3*4, k3, p1) -> PixelShuffle(2) -> Sigmoid
        x = jax.nn.leaky_relu(x, 0.2)
        x = self.final_conv(x)
        x = pixel_shuffle(x, 2)
        return jax.nn.sigmoid(x)