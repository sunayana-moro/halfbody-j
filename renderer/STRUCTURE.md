# Renderer — PyTorch structure map

Full tree, expanded down to functions/methods and their internal call chains.

**Legend**
- `→ Name` = calls our own class/function (follow the chain)
- `→[lib]` = terminal library call (torch / cv2 / PIL / timm / torchvision) — chain stops here
- `◄` = note / hotspot ·  `‹DEAD›` = not reachable from `IMTRenderer` (skip when porting)
- `·param` learned tensor · `·buffer` non-learned constant · `·sub` submodule

**Dependency direction**
```
train.py / inference.py
        │
        ▼
    models.py ───► modules.py ───► lia_resblocks.py
        │                              ▲
        └──────────────────────────────┘
   (train also ──► discriminator.py + vgg19_mask.py)
```

---

## lia_resblocks.py — StyleGAN2 / LIA primitives (the foundation)

```
lia_resblocks.py
│
├── module functions
│   ├── fused_leaky_relu()          →[lib] F.leaky_relu
│   ├── upfirdn2d_native()          →[lib] F.pad, torch.flip, F.conv2d, strided-slice
│   ├── upfirdn2d()                 → upfirdn2d_native
│   └── make_kernel()               →[lib] torch.tensor
│
├── LIVE classes (used by IMTRenderer)
│   ├── FusedLeakyReLU              ·param bias
│   │   └── forward                 → fused_leaky_relu
│   ├── Blur                        ·buffer kernel
│   │   └── forward                 → upfirdn2d
│   ├── ScaledLeakyReLU
│   │   └── forward                 →[lib] F.leaky_relu
│   ├── EqualConv2d                 ·param weight, bias ·const scale        ◄ EQUALIZED-LR
│   │   └── forward                 →[lib] F.conv2d   (weight*scale at call time)
│   ├── EqualLinear                 ·param weight, bias ·const scale, lr_mul ◄ EQUALIZED-LR
│   │   └── forward                 →[lib] F.linear  (+ fused_leaky_relu if activation)
│   ├── ConvLayer(nn.Sequential)    = Blur(if downsample) + EqualConv2d + FusedLeakyReLU/ScaledLeakyReLU
│   ├── ModulatedConv2d             ·param weight ·sub modulation=EqualLinear, blur=Blur  ◄ HARD (grouped-batch conv)
│   │   └── forward(input, style)   → modulation →[lib] F.conv2d / F.conv_transpose2d (groups=batch) → blur
│   ├── NoiseInjection              ·param weight
│   │   └── forward(image, noise)   → image + weight*noise   (or image)
│   └── StyledConv                  ·sub conv=ModulatedConv2d, noise=NoiseInjection, activate=FusedLeakyReLU
│       └── forward(input, style)   → conv → noise → activate
│
└── ‹DEAD› classes (skip)
    ├── PixelNorm, MotionPixelNorm  →[lib] torch.rsqrt
    ├── Upsample, Downsample        ·buffer kernel → upfirdn2d
    ├── ConstantInput               ·param input
    ├── ToRGB                       → ConvLayer, Upsample
    ├── ToFlow                      → ModulatedConv2d, Upsample, [lib] F.grid_sample, numpy grid, .cuda()  ◄ grid_sample
    ├── Direction                   ·param weight →[lib] torch.qr, torch.diag_embed                        ◄ QR
    └── Synthesis                   → ConstantInput, StyledConv, ToRGB, ToFlow, Direction
```

---

## modules.py — CNN building blocks

```
modules.py
│
├── NormLayer                       ·sub norm
│   └── forward                     →[lib] BatchNorm2d | InstanceNorm2d | GroupNorm(1,C)
├── ConvBlock                       ·sub conv[lib], norm=NormLayer, activation[lib]
│   └── forward                     → conv → NormLayer → activation
├── FeatResBlock                    ·sub conv1, conv2 = ConvBlock
│   └── forward                     → ConvBlock ×2 → out+=residual (in-place) → activation
├── ResBlock                        ·sub conv1, conv2, skip = ConvLayer(lia)   (used by MotionEncoder)
│   └── forward                     → ConvLayer ×3 → out+skip
├── ConvResBlock                    ·sub Conv2d, NormLayer, FeatResBlock ×2    (used by SynthesisNetwork)
│   └── forward                     → conv → norm → act → conv → FeatResBlock ×2
├── DownConvResBlock                ·sub Conv2d, NormLayer, AvgPool2d, FeatResBlock ×2  (used by IdentityEncoder)
│   └── forward                     → conv → norm → act → avgpool → conv → FeatResBlock ×2
├── UpConvResBlock                  ·sub Upsample(nearest), Conv2d, NormLayer, FeatResBlock ×2 (used by SynthesisNetwork)
│   └── forward                     → upsample → conv → norm → act → conv → FeatResBlock ×2
│
└── ‹DEAD› SPADE, SPADEResnetBlock (spectral_norm), SPADEDecoder
```

---

## attention_modules.py — attention family (needs timm to_2tuple / trunc_normal_)

```
attention_modules.py
│
├── window_partition() / window_reverse()   →[lib] view/permute
│
├── StandardUnifiedAttention        ·sub q/k/v_proj, proj [lib Linear]
│   └── forward(q, k, v, mask)      → qkv projs → (q@kᵀ)·scale →[lib] softmax → @v → proj
│                                     returns (x, attn_map)   ◄ attn_map reused coarse→fine
├── GuidedResampler                 (NO params — pure gather)                 ◄ HARD (index-heavy)
│   └── forward(v_high, coarse_map) →[lib] torch.topk, meshgrid, gather, softmax → warped_feat
├── SwinUnifiedAttention            ·param relative_position_bias_table ·buffer relative_position_index
│   └── forward(q, k, v, mask)      → qkv → attn + rel-pos-bias → softmax → @v → proj
├── UnifiedTransformerBlock         ·param q_pos_embedding, k_pos_embedding ·sub norms, attn=StandardUnifiedAttention, mlp
│   └── forward(q, k, v)            → flatten BCHW→B(HW)C → +pos → norm → StandardUnifiedAttention → residual → mlp
├── UnifiedSwinBlock                ·buffer attn_mask ·sub norms, attn=SwinUnifiedAttention, mlp
│   └── forward(q, k, v)            → norm →[lib] torch.roll → window_partition → SwinUnifiedAttention → window_reverse → mlp
├── CrossAttention  (the imt blocks)   ·is_standard_attention = res < swin_res_threshold
│   ├── (standard) ·sub block_efc=StandardUnifiedAttention   |   (fine) ·sub block=GuidedResampler
│   ├── coarse_stage(A, B, C)       → block_efc → (out, attn_map)
│   ├── fine_stage(C, attn)         → block(C, attn.mean(dim=1))
│   └── forward(A, B, C, D, attn)   → dispatch coarse / fine
└── SelfAttention  (SynthesisNetwork transformer blocks)   ·sub blocks=ModuleList
    ├── res ≥ threshold → UnifiedSwinBlock ×2 (shift 0 & ws//2)   |   else → UnifiedTransformerBlock ×1
    └── forward(q, k, v)            → chain blocks (self or cross)
```

---

## models.py — assembled networks (top model IMTRenderer)

```
models.py
│
├── IdentityEncoder                 ·sub initial_conv[Conv7/BN/ReLU], down_blocks=DownConvResBlock,
│   │                                    equalconv‹unused›, linear_layers/final_linear=EqualLinear
│   └── forward(x)                  → initial_conv → DownConvResBlock ×N (collect features)
│                                     → global-avg-pool → EqualLinear ×4 → final_linear
│                                     ⇒ (features[::-1], identity_vec)
│                                     ◄ QUIRK: `if out_channels==32: continue` drops the 32 (builds 64→512)
├── MotionEncoder                   ·sub conv1[Conv], res_blocks=ResBlock, equalconv=EqualConv2d, EqualLinear×
│   └── forward(x)                  → conv1 → ResBlock ×N → EqualConv2d → avg-pool → EqualLinear ×4 → final_linear ⇒ latent(32)
├── MotionDecoder                   ·param const ·sub style_conv_layers=StyledConv ×13
│   └── forward(t)                  → const.repeat → StyledConv ×13 (tap idx 3/6/9/12) ⇒ (m1,m2,m3,m4)
├── SynthesisNetwork                ·sub upconv_blocks=UpConvResBlock, resblocks=ConvResBlock,
│   │                                    transformer_blocks=SelfAttention, final_conv[LeakyReLU/Conv/PixelShuffle/Sigmoid]
│   └── forward(features_align)     → per level: UpConvResBlock → cat skip → ConvResBlock → SelfAttention → final_conv
├── IdentidyAdaptive                ·sub in_layer/linear_layers/final_linear=EqualLinear, scale_activation‹unused›
│   └── forward(mot, app)           → cat → EqualLinear MLP ⇒ out(32)
│
└── IMTRenderer                     dims: feature=[32,64,128,256,512,512], spatial=[256,128,64,32,16,8]
    │   ·sub dense_feature_encoder=IdentityEncoder, latent_token_encoder=MotionEncoder,
    │        latent_token_decoder=MotionDecoder, frame_decoder=SynthesisNetwork,
    │        adapt=IdentidyAdaptive, imt=ModuleList[CrossAttention]
    ├── app_encode(x)               → IdentityEncoder ⇒ (f_r, id)
    ├── mot_encode(x)               → MotionEncoder ⇒ latent
    ├── mot_decode(x)               → MotionDecoder ⇒ motion maps
    ├── id_adapt(t, id)             → self.adapt(t, id)   (== inference's gen.adapt)
    ├── decode(A, B, C)             → per level: CrossAttention.coarse_stage (captures attn_map)
    │                                 OR .fine_stage(attn) → SynthesisNetwork ⇒ frame
    └── forward(x_current, x_ref)   → app_encode(ref) → mot_encode(ref,cur) → id_adapt
                                      → mot_decode → decode ⇒ (frame, t_c)
```

---

## discriminator.py — two-scale StyleGAN2 D (training only)

> Re-declares its OWN copies of fused_leaky_relu, FusedLeakyReLU, upfirdn2d, Blur, ScaledLeakyReLU,
> EqualConv2d, EqualLinear, ConvLayer, ResBlock (skip = /√2).

```
discriminator.py
│
├── Discriminator(size)             ·sub convs=[ConvLayer + ResBlock ×log], final_conv=ConvLayer, final_linear=EqualLinear ×2
│   └── forward(input)              → convs → minibatch-stddev concat[lib var] → final_conv → final_linear ⇒ scalar
└── PatchDiscriminator              ·sub scale1=Discriminator(512), scale2=Discriminator(256)
    └── forward(x)                  → scale1(x), scale2([lib F.interpolate ×0.5]) ⇒ [out1, out2]
```

---

## vgg19_mask.py — masked perceptual loss (training only)

```
vgg19_mask.py
│
├── AntiAliasInterpolation2d        ·buffer weight (gaussian)
│   └── forward                     →[lib] F.pad, F.conv2d(groups) → strided-slice
├── ImagePyramide                   ·sub downs=ModuleDict[AntiAliasInterpolation2d]
│   └── forward(x)                  ⇒ dict of downsampled
├── Vgg19                           [lib] torchvision vgg19(pretrained) sliced 1..5 ·param mean/std
│   └── forward(X)                  → clamp[-1,1] → X/2+0.5 → normalize → slices ⇒ 5 feature maps  ◄ assumes [-1,1] input
└── VGGLoss_mask                    ·sub pyramid=ImagePyramide, vgg=Vgg19
    └── forward(recon, real, mask)  → pyramids → Vgg19 per scale ⇒ L1 feat loss (loss_all) + mask-weighted (loss_face)
```

---

## train.py — PyTorch Lightning (training entrypoint)

```
train.py
│
├── IMFSystem(pl.LightningModule)   ·sub gen=IMTRenderer, disc=PatchDiscriminator, criterion_vgg=VGGLoss_mask
│   │                                ·automatic_optimization = False
│   ├── training_step               → gen.app_encode/mot_encode/id_adapt/mot_decode/decode
│   │                                 → D step (calculate_gan_loss, manual_backward, clip_grad_norm)
│   │                                 → G step (F.l1_loss, criterion_vgg, dist loss, calculate_gan_loss, backward, clip)
│   ├── validation_step             → gen(...) full forward ×2 (pred, recon) → criterion_vgg →[lib] logger.add_images
│   ├── calculate_gan_loss()        →[lib] F.softplus
│   ├── configure_optimizers()      →[lib] optim.Adam ×2, CosineAnnealingLR ×2
│   └── load_ckpt → _safe_load      →[lib] torch.load, load_state_dict(strict=False)  (prefix strip + shape check)
├── DataModule(pl.LightningDataModule)  → TFDataset, [lib] DataLoader
└── __main__                        → argparse → IMFSystem + DataModule →[lib] pl.Trainer.fit
```

---

## inference.py — video inference

```
inference.py
│
├── DataProcessor                   [lib] face_alignment, torchvision transforms
│   ├── process_img                 →[lib] fa.face_detector, cv2 crop
│   └── load_image                  →[lib] cv2
├── save_video()                    →[lib] cv2.VideoWriter
├── Demo(nn.Module)                 ·sub gen, processor=DataProcessor
│   ├── process_single              → encode source once (app_encode/mot_encode/adapt/mot_decode)
│   │                                 → per driving frame (mot_encode/adapt/mot_decode/decode) → save_video
│   ├── process_batch               → process_single loop
│   └── run                         → dispatch
└── __main__                        → IMTRenderer →[lib] torch.load → Demo.run
```

---

## dataset.py — data (stays torch/numpy at port time)

```
dataset.py
│
├── create_eye_mouth_mask()         → to_px_coords, fill_polygon →[lib] cv2.fillConvexPoly/convexHull/erode/dilate
└── TFDataset(Dataset)
    ├── __init__                    →[lib] transforms.Compose → _load_metadata
    ├── _load_metadata             →[lib] Path.iterdir, tqdm   (first 500 clips = train)
    ├── __getitem__                → read_landmark_info + create_eye_mouth_mask
    │                                +[lib] Image.open, transform, np.random ⇒ dict(image_0/1, neg_image, masks)
    └── read_landmark_info()       →[lib] open, np.array
```

---

# Target — `renderer_jax/` (the rewrite)

**Stack:** Flax NNX · jax 0.10.0 · flax 0.10.2.
**Name:** `renderer_jax/` — mirrors `renderer/`, names the *thing* not the framework.
**Category rule:** by **statefulness** — pure math → functions (`ops.py`); anything with learned params → `nnx.Module`.

```
renderer_jax/
├── __init__.py
├── config.py      # single source of dims/hparams: feature_dims=[32,64,128,256,512,512],
│                  #   spatial_dims=[256,128,64,32,16,8], num_heads=8, window_size=8,
│                  #   swin_res_threshold=128, latent_dim=32
├── ops.py         # PURE fns (no params): upfirdn2d, make_kernel, fused_leaky_relu,
│                  #   scaled_leaky_relu, pixel_shuffle, resize_nearest
├── layers.py      # leaf nnx.Modules: EqualLinear, EqualConv2d, Blur, FusedLeakyReLU,
│                  #   NoiseInjection, NormLayer
├── stylegan.py    # ModulatedConv2d (grouped-batch conv), StyledConv
├── resblocks.py   # ConvBlock, FeatResBlock, ResBlock, ConvResBlock, DownConvResBlock, UpConvResBlock
├── attention.py   # window_partition/reverse, Standard/Swin attn, UnifiedTransformer/SwinBlock,
│                  #   GuidedResampler, CrossAttention, SelfAttention
├── encoders.py    # IdentityEncoder, MotionEncoder
├── decoders.py    # MotionDecoder, SynthesisNetwork
├── renderer.py    # IdentidyAdaptive + IMTRenderer   (top model)
├── parity.py      # THE harness: torch→flax weight port + same-input allclose
└── tests/         # one per category, each ~3 lines calling parity.py
    ├── test_ops.py        test_layers.py     test_stylegan.py
    ├── test_resblocks.py  test_attention.py  test_encoders.py
    └── test_decoders.py   test_renderer.py

siblings for later (deferred to training phase):  losses/   training/   inference/
```

### Dev flow — per unit, bottom-up, gated

```
   write flax module  (you)
           │
           ▼
   parity.port_weights:  torch state_dict ──► nnx params
           │             conv (O,I,H,W)→flax layout ; linear (O,I)→(I,O)
           ▼
   x = np.random(seed)  ──►  torch_layer.eval()(x) ──┐
                        └──►  flax_module(jnp(x))   ──┤
                                                      ▼
                         np.allclose(a, b, atol)  ──►  ✅ commit   /   ❌ fix
```
numpy is the **bridge** (both sides → numpy → compare); the **test** is `allclose` *after weights are ported in*.

### Parity rules (in parity.py + every test)

- ONE seeded numpy input feeds BOTH sides — never sample torch and jax separately
- weights always flow **torch → flax**; torch layer in `.eval()` (BatchNorm/dropout)
- test the path the model actually uses (`noise=None`, dropout=0)
- realistic shapes from `config.py` (batch > 1, real channel/spatial dims — toy 4×4 hides bugs)
- **layout fixed once:** NHWC internally (idiomatic) + transpose at model boundary; harness transposes to torch's NCHW
- `atol ≈ 1e-5` for leaves, loosens with depth; **one small commit per passing unit** (bisectable)
- leaf passes before composite → a composite-only failure means wiring/reshape, not the leaf

### Notes

- **Skip (‹DEAD›):** all `SPADE*`, `Synthesis`/`ToFlow`/`Direction`/`ToRGB`/`ConstantInput`/`PixelNorm*`
- **Emergency ops (I provide):** grouped-batch conv (`ModulatedConv2d`), `upfirdn2d`, `GuidedResampler`
- **Build order (each parity-gated):** config → ops → layers → stylegan → resblocks → attention → encoders/decoders → renderer
- **Oracle gate:** tests need torch importable (local venv) OR golden `.npz` dumps from a torch box — set up before executing
