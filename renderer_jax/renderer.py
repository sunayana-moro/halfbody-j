from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from shared.layers import EqualLinear
from shared.attention import CrossAttention
from renderer_jax.encoders import IdentityEncoder, MotionEncoder
from renderer_jax.decoders import MotionDecoder, SynthesisNetwork

class IdentidyAdaptive(nnx.Module):
    """
    Fuse a motion latent with an identity vector -> adapted motion latent.

    motion latent (B, 32) , identity vector (B, 512)
        -> Concatenate (B, 544)                         
        -> EqualLinear (B, 512)                         
        -> 4* (EqualLinear + LeakyReLU 0.2) (B, 512)    
        -> Final EqualLinear (B, 32)              
    """

    def __init__(self, dim_mot=32, dim_app=512, depth=4, *, rngs: nnx.Rngs):
        self.in_layer = EqualLinear(dim_app + dim_mot, dim_app, rngs=rngs)
        self.linear_layers = [EqualLinear(dim_app, dim_app, rngs=rngs) for _ in range(depth)]
        self.final_linear = EqualLinear(dim_app, dim_mot, rngs=rngs)

    def __call__(self, mot, app):
        x = jnp.concatenate((mot, app), axis=-1)      # (B, dim_mot + dim_app)
        x = self.in_layer(x)
        for lin in self.linear_layers:
            x = jax.nn.leaky_relu(lin(x), 0.2)
        return self.final_linear(x)

class IMTRenderer(nnx.Module):
    """
    forward(x_current, x_reference) -> (output_frame, t_c):
        reference -> IdentityEncoder  -> (f_r feature pyramid, i_r identity vec)
        reference -> MotionEncoder    -> t_r     ;  current -> MotionEncoder -> t_c
        adapt(t_r, i_r) -> ta_r       ;  adapt(t_c, i_r) -> ta_c              [IdentidyAdaptive]
        MotionDecoder(ta_r) -> ma_r   ;  MotionDecoder(ta_c) -> ma_c          [4 motion maps each]
        decode(ma_c, ma_r, f_r) -> output_frame                              [imt align + synthesis]
        return (output_frame, t_c)
    """

    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        self.feature_dims = tuple(config.feature_dims)     # (32,64,128,256,512,512)
        self.spatial_dims = tuple(config.spatial_dims)     # (256,128,64,32,16,8)
        
        self.dense_feature_encoder = IdentityEncoder(
            output_channels=self.feature_dims, initial_channels=config.id_enc_init,
            dm=config.id_dim, rngs=rngs)
        self.latent_token_encoder = MotionEncoder(
            initial_channels=config.motion_enc_init,
            output_channels=tuple(config.motion_enc_channels), dm=config.latent_dim, rngs=rngs)
        self.latent_token_decoder = MotionDecoder(
            latent_dim=config.latent_dim, const_dim=config.const_dim, rngs=rngs)
        self.frame_decoder = SynthesisNetwork(config, self.feature_dims, self.spatial_dims, rngs=rngs)
        self.adapt = IdentidyAdaptive(
            dim_mot=config.latent_dim, dim_app=config.id_dim, depth=config.adapt_depth, rngs=rngs)

        self.imt = [
            CrossAttention(dim, s_dim, config.num_heads, config.swin_res_threshold, rngs=rngs)
            for dim, s_dim in zip(self.feature_dims[::-1], self.spatial_dims[::-1])
        ]

    def app_encode(self, x, use_running_average=None):
        return self.dense_feature_encoder(x, use_running_average=use_running_average)

    def mot_encode(self, x):
        return self.latent_token_encoder(x)

    def mot_decode(self, x):
        return self.latent_token_decoder(x)

    def id_adapt(self, t, id):
        return self.adapt(t, id)

    def decode(self, A, B, C, use_running_average=None):
        """
        Align the 6 pyramid levels via the imt CrossAttention blocks, then synthesize.
        A = ma_c (4 motion maps), B = ma_r (4 motion maps), C = f_r (6 features).
        Per level i (resolutions [8,16,32,64,128,256], threshold 128):
        """
        num_levels = len(self.spatial_dims)
        aligned = [None] * num_levels
        attn_map = None
        for i in range(num_levels):
            block = self.imt[i]
            if block.is_standard:
                aligned[i], attn_map = block.coarse_stage(A[i], B[i], C[i])
            else:
                aligned[i] = block.fine_stage(C[i], attn_map)
        return self.frame_decoder(aligned, use_running_average=use_running_average)

    def __call__(self, x_current, x_reference, use_running_average=None):
        f_r, i_r = self.app_encode(x_reference, use_running_average=use_running_average)
        t_r = self.mot_encode(x_reference)
        t_c = self.mot_encode(x_current)
        ta_r = self.adapt(t_r, i_r)
        ta_c = self.adapt(t_c, i_r)
        ma_r = self.mot_decode(ta_r)
        ma_c = self.mot_decode(ta_c)
        output_frame = self.decode(ma_c, ma_r, f_r, use_running_average=use_running_average)
        return output_frame, t_c
