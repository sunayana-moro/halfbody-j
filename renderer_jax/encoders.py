from __future__ import annotations

import jax
from flax import nnx

from shared.layers import EqualConv2d, EqualLinear
from shared.resblock import DownConvResBlock, ResBlock

# ------------------------------------------------------------------------------------------------------------
# Identity Encoder
# ------------------------------------------------------------------------------------------------------------

class IdentityEncoder(nnx.Module):
    """
    Input Image (512 * 512 * 3)
        -> 7*7 Conv -> BatchNorm -> ReLU (512 * 512 * 32)
        -> DownConvResBlock (256 * 256 * 32)   [down_block_0, feature]
        -> DownConvResBlock (128 * 128 * 64)   [feature]
        -> DownConvResBlock (64 * 64 * 128)    [feature]
        -> DownConvResBlock (32 * 32 * 256)    [feature]
        -> DownConvResBlock (16 * 16 * 512)    [feature]
        -> DownConvResBlock (8 * 8 * 512)      [feature]
        -> Global Average Pooling (512)
        -> 4* (EqualLinear + LeakyReLU 0.2)
        -> Final EqualLinear
        -> Returns (feature pyramid [coarse..fine], 512-dim identity vector)
    """

    def __init__(self, in_channels=3, output_channels=(64, 128, 256, 512, 512, 512),
                 initial_channels=32, dm=512, *, rngs: nnx.Rngs):
        # initial 7×7 convolution to extract low-level features.
        self.initial_conv = nnx.Conv(in_channels, initial_channels, (7, 7),
                                     strides=1, padding=3, rngs=rngs)
        self.initial_norm = nnx.BatchNorm(initial_channels, rngs=rngs)

        self.down_block_0 = DownConvResBlock(initial_channels, initial_channels, rngs=rngs)
        self.down_blocks = []
        cur = initial_channels
        for oc in output_channels:
            if oc == 32:                 
                continue
            self.down_blocks.append(DownConvResBlock(cur, oc, rngs=rngs))
            cur = oc

        last = output_channels[-1]
        self.equalconv = EqualConv2d(last, last, 3, stride=1, padding=1, rngs=rngs)  # but not used in forward
        self.linear_layers = [EqualLinear(last, last, rngs=rngs) for _ in range(4)]
        self.final_linear = EqualLinear(last, dm, rngs=rngs)

    def __call__(self, x, use_running_average=None):
        features = []
        x = self.initial_conv(x)
        x = self.initial_norm(x, use_running_average=use_running_average)
        x = jax.nn.relu(x)                                          
        x = self.down_block_0(x, use_running_average=use_running_average)
        features.append(x)
        for block in self.down_blocks:
            x = block(x, use_running_average=use_running_average)
            features.append(x)

        x = x.mean(axis=(1, 2))                                  
        for lin in self.linear_layers:
            x = jax.nn.leaky_relu(lin(x), 0.2)                   
        x = self.final_linear(x)
        return features[::-1], x    

# ------------------------------------------------------------------------------------------------------------
# Motion Encoder
# ------------------------------------------------------------------------------------------------------------

class MotionEncoder(nnx.Module):
    """
    Input Image (256 * 256 * 3)
        -> 3*3 Conv
        -> LeakyReLU (256 * 256 * 64)
        -> ResBlock (128 * 128 * 64)
        -> ResBlock (64 * 64 * 128)
        -> ResBlock (32 * 32 * 256)
        -> ResBlock (16 * 16 * 512)
        -> ResBlock 8 * 8 * 512)
        -> ResBlock (4 * 4 * 512)
        -> EqualConv2d (4 * 4 * 512)
        -> Global Average Pooling (512)
        -> EqualLinear + LeakyReLU (512)
        -> EqualLinear + LeakyReLU (512)
        -> EqualLinear + LeakyReLU (512)
        -> EqualLinear + LeakyReLU (512)
        -> Final EqualLinear (32)
        -> Motion latent embedding
    """

    def __init__(self, initial_channels=64, output_channels=(64, 128, 256, 512, 512, 512),
                 dm=32, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, initial_channels, (3, 3), strides=1, padding=1, rngs=rngs)
        self.res_blocks = []
        cur = initial_channels
        for oc in output_channels:
            self.res_blocks.append(ResBlock(cur, oc, rngs=rngs))
            cur = oc

        last = output_channels[-1]
        self.equalconv = EqualConv2d(last, last, 3, stride=1, padding=1, rngs=rngs)  
        self.linear_layers = [EqualLinear(last, last, rngs=rngs) for _ in range(4)]
        self.final_linear = EqualLinear(last, dm, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.leaky_relu(self.conv1(x), 0.2)
        for rb in self.res_blocks:
            x = rb(x)                                               
        x = self.equalconv(x)
        x = x.mean(axis=(1, 2))                                    
        for lin in self.linear_layers:
            x = jax.nn.leaky_relu(lin(x), 0.2)
        x = self.final_linear(x)
        return x