# Conversion notes — PyTorch → JAX/Flax

Problems hit while porting `renderer/` (torch) → `shared/` (JAX/Flax NNX), and
whether a parity test settled them. Check here first when something breaks.

**Conventions:** NHWC internally (torch is NCHW, transpose at boundary) · weight
port: conv `(O,I,H,W)→(H,W,I,O)`, linear `(O,I)→(I,O)=.T` · explicit `rngs` (no
global RNG) · non-trainable constants (blur kernels, position indices) kept as
**numpy** (not `nnx.Param`) · `shared/` is torch-free; the torch-importing parity
harness lives in `test/shared/`.

**Parity tests** (`test/shared/*.py`) run real torch vs real jax in **float64**
(correct code matches ~1e-15; float32's ~1e-6 noise would hide subtle bugs).

---

## 1. ModulatedConv2d — per-sample convolution (the emergency op)
StyleGAN2 modulates the conv weight **per sample** → B different kernels, which is
not a standard conv. torch fakes it with `F.conv2d(groups=batch)`; we mirror that
with `conv_general_dilated(feature_group_count=B)`. The difficulty was translation
(no 1:1 primitive), not math or randomness.
**→ SOLVED:** `test/shared/stylegan.py` — 8/8 at ~1e-15.

## 2. Upsample = transposed conv (the silent-parity killer)
torch uses `conv_transpose2d`; we emulate it as `conv_general_dilated` +
`lhs_dilation=(2,2)`. Two traps: the kernel needs a **rot180 spatial flip**, and
torch's `weight.transpose(1,2)` (in↔out swap) is a `conv_transpose2d` convention we
**bypass** (forward conv keeps natural `(kh,kw,Cin,Cout)`, so no swap). Blur is
applied AFTER (downsample blurs BEFORE) — both faithful to torch.
**→ SOLVED:** numpy-verified, then confirmed real-XLA by `test/shared/stylegan.py`
upsample case at ~1e-15 (removed the "confirm-in-venv" doubt).

## 3. Downsample — dead code, ported anyway
Never reached in the model (StyledConv has no downsample). It's the *simplest* path
(blur-first + forward grouped conv, stride 2, VALID). Added for full parity.
**→ SOLVED:** `test/shared/stylegan.py` downsample cases at ~1e-15.

## 4. Blur dtype trap
`make_kernel` returns **float32**, but jax `conv_general_dilated` requires lhs/rhs
same dtype (torch auto-promotes; jax does not). float32 hid it; float64 crashed.
Fix: cast the FIR kernel to `x.dtype` in `upfirdn2d`.
**→ CAUGHT by the float64 parity test**, then fixed. (Why we test in float64.)

## 5. NormLayer epsilon + affine traps
- **epsilon:** flax `nnx.GroupNorm` defaults `1e-6`; torch Instance/GroupNorm use
  `1e-5` (verified in flax 0.10.2 source). Instance/layer need explicit
  `epsilon=1e-5`. (flax `BatchNorm` already defaults `1e-5` = torch.)
- **affine:** torch `InstanceNorm2d` has NO affine; `GroupNorm(1,C)` ('layer') DOES.
Both fixed in `shared/layers.py`.
**→ to be confirmed by** `test/shared/layers.py` NormLayer cases.

## 6. Attention decisions (shared/attention.py)
- **Dropout:** no global train/eval in jax → thread an explicit `deterministic`
  flag into `nnx.Dropout` (rates are 0 here, so no-op now).
- **args → plain params:** blocks take `num_heads/window_size/swin_res_threshold`,
  not a config object, so `shared/` stays reusable by the generator too.
- **CrossAttention has no `__call__`:** torch's `forward` is a dead duplicate; the
  real API is `coarse_stage`/`fine_stage` (decode threads the attn_map coarse→fine).
- **Swin vs standard by resolution:** high-res uses windowed attention (global
  attention is O(N²) → 256² ≈ 4B-entry matrix); low-res uses global transformer.
**→ core (StandardUnifiedAttention + window ops) parity-tested; composites are
smoke-tested (built from verified pieces + stock flax layers).**

## 7. Smaller gotchas
- **Equalized-LR** (EqualConv2d/Linear): init `N(0,1)`, apply `1/√fan_in` scale at
  **call time**, never fold into the stored weight.
- **EqualLinear bias=False** would crash (`self.bias.value` on None) — never hit
  (always bias=True); guard if ever needed.
- **Two torch `ResBlock`s collide by name, differ in math:** `modules.ResBlock`
  (`out+skip`) vs `discriminator.ResBlock` (`(out+skip)/√2`). Don't merge.
- **`ops.py` deleted** — pure fns live with consumers; `pixel_shuffle` still owed
  (goes in `decoders.py`).
