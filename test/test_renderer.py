"""test_renderer.py — THE final torch<->jax end-to-end validation.

Loads renderer.ckpt into BOTH the torch IMTRenderer (oracle) and the JAX
IMTRenderer (weights ported from the same torch model), runs the FULL video-driven
pipeline on both, and:

  1. logs max|Δ| / mean|Δ| at EVERY pipeline stage (app_encode, each feature-pyramid
     level, mot_encode, id_adapt, mot_decode maps, decoded frame) — the FIRST stage
     that diverges pinpoints where the port is wrong;
  2. writes a side-by-side mp4 (PyTorch | JAX, labelled).

The JAX model runs UNDER jax.jit (the real deployment path): the forward sub-steps
are jitted once via nnx.split/merge and reused across frames. float32 (real inference
precision). torch uses CUDA if available; JAX uses its default backend.

    test_env/bin/python test/test_renderer.py \
        --renderer_path checkpoints/renderer.ckpt \
        --source_path assets/source_1.png \
        --driving_path assets/driving_1.mp4 \
        --save_path results/ [--crop] [--max_frames N]

Reading the result: a correct port shows ~1e-5..1e-6 per stage; a real bug shows
~1e-1+. First stage above ~1e-3 is the culprit. Everything below the full model is
block-level parity-tested EXCEPT the attention composites (Swin/Transformer/Self/
Cross) — so divergence first appearing after attention => attention is prime suspect,
then pixel_shuffle, then decode() wiring.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

import cv2
import torch
from PIL import Image
from tqdm import tqdm

from types import SimpleNamespace

from renderer_jax.config import RendererConfig
from renderer_jax.renderer import IMTRenderer
from renderer_jax.inference import preprocess          # reuse the exact preprocess
from scripts.port_weights import port_imt_renderer


# ---------------------------------------------------------------- diff logging
def _to_nhwc_np(x):
    """torch NCHW / jax NHWC / vectors -> numpy in a comparable layout."""
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().numpy()
        if a.ndim == 4:                                 # NCHW -> NHWC
            a = a.transpose(0, 2, 3, 1)
        return a
    return np.asarray(x)


def diff(name, t, j):
    a, b = _to_nhwc_np(t), _to_nhwc_np(j)
    if a.shape != b.shape:
        print(f"    {name:22s} SHAPE MISMATCH  torch {a.shape}  jax {b.shape}")
        return float("inf")
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mx, mean = float(d.max()), float(d.mean())
    flag = "  <-- DIVERGES" if mx > 1e-3 else ""
    print(f"    {name:22s} shape {str(a.shape):22s} max|Δ|={mx:.3e}  mean|Δ|={mean:.3e}{flag}")
    return mx


# ---------------------------------------------------------------- model loading
def build_torch(ckpt_path, cfg):
    from renderer.models import IMTRenderer as TIMT
    args = SimpleNamespace(num_heads=cfg.num_heads, window_size=cfg.window_size,
                           swin_res_threshold=cfg.swin_res_threshold)
    m = TIMT(args)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace("gen.", ""): v for k, v in sd.items() if k.startswith("gen.")} or sd
        msg = m.load_state_dict(sd, strict=False)
        print(f"[INFO] torch loaded ckpt: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    else:
        print("[WARN] no checkpoint -> RANDOM torch weights (still a valid port+forward parity check)")
    return m.eval()


def torch_frame_to_np(out):
    return out.detach().cpu().numpy()[0].transpose(1, 2, 0)   # (1,3,H,W)->(H,W,3)


def jax_frame_to_np(out):
    return np.asarray(out)[0]                                 # (1,H,W,3)->(H,W,3)


# ---------------------------------------------------------------- side-by-side video
def label(img_u8, text):
    """Draw a centered title banner at the top of a panel."""
    img = img_u8.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 44), (0, 0, 0), -1)
    scale, thick = 1.1, 2
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max((w - tw) // 2, 8)                       # centered
    cv2.putText(img, text, (x, 32), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="torch<->jax end-to-end renderer parity + side-by-side video")
    p.add_argument("--renderer_path", default="checkpoints/renderer.ckpt")
    p.add_argument("--source_path", required=True)
    p.add_argument("--driving_path", required=True)
    p.add_argument("--save_path", default="results/")
    p.add_argument("--crop", action="store_true")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None,
                   help="cap frames for a quick run (default: process ALL frames)")
    args = p.parse_args()

    cfg = RendererConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f" torch <-> jax END-TO-END parity  (float32, jax under jit, torch on {device})")
    print("=" * 78)

    # --- build both models from the SAME weights (port on CPU, then move torch to device) ---
    tm = build_torch(args.renderer_path, cfg)
    jm = IMTRenderer(cfg, rngs=nnx.Rngs(0))
    port_imt_renderer(tm, jm)
    tm = tm.to(device)
    print("[INFO] ported torch weights -> jax\n")

    # --- jitted jax sub-steps (compile once via split/merge, reuse every frame) ---
    gd, state = nnx.split(jm)
    j_app    = jax.jit(lambda s, x: nnx.merge(gd, s).app_encode(x, use_running_average=True))
    j_mot    = jax.jit(lambda s, x: nnx.merge(gd, s).mot_encode(x))
    j_adapt  = jax.jit(lambda s, t, i: nnx.merge(gd, s).id_adapt(t, i))
    j_mdec   = jax.jit(lambda s, t: nnx.merge(gd, s).mot_decode(t))
    j_decode = jax.jit(lambda s, A, B, C: nnx.merge(gd, s).decode(A, B, C, use_running_average=True))

    def to_t(nhwc):                                     # np NHWC -> torch NCHW on device
        return torch.from_numpy(nhwc.transpose(0, 3, 1, 2)).to(device)

    # --- source input (identical to both) ---
    fa = None
    if args.crop:
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    src_nhwc = preprocess(Image.open(args.source_path), fa=fa, crop=args.crop)
    src_t, src_j = to_t(src_nhwc), jnp.asarray(src_nhwc)

    # --- SOURCE encode: stage-by-stage diff ---
    print("[SOURCE ENCODE] per-stage torch vs jax:")
    with torch.no_grad():
        t_fr, t_ir = tm.app_encode(src_t)
        t_tr = tm.mot_encode(src_t)
        t_tar = tm.id_adapt(t_tr, t_ir)
        t_mar = tm.mot_decode(t_tar)
    j_fr, j_ir = j_app(state, src_j)
    j_tr = j_mot(state, src_j)
    j_tar = j_adapt(state, j_tr, j_ir)
    j_mar = j_mdec(state, j_tar)

    diff("identity vec i_r", t_ir, j_ir)
    for k in range(len(j_fr)):
        diff(f"feature f_r[{k}]", t_fr[k], j_fr[k])
    diff("motion t_r", t_tr, j_tr)
    diff("adapted ta_r", t_tar, j_tar)
    for k in range(len(j_mar)):
        diff(f"motion map ma_r[{k}]", t_mar[k], j_mar[k])
    print()

    # --- driving video: per frame ---
    cap = cv2.VideoCapture(args.driving_path)
    out_fps = cap.get(cv2.CAP_PROP_FPS) if args.fps is None else args.fps
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = n_total
    if args.max_frames and args.max_frames < n_total:
        total = args.max_frames
        print(f"[INFO] --max_frames set: processing {total} of {n_total} frames "
              f"(video truncated; omit --max_frames for the full clip)")

    save_name = f"{os.path.splitext(os.path.basename(args.source_path))[0]}_" \
                f"{os.path.splitext(os.path.basename(args.driving_path))[0]}_cmp.mp4"
    out_path = os.path.join(args.save_path, save_name)
    os.makedirs(args.save_path, exist_ok=True)
    writer = None
    frame_diffs = []

    print(f"[DRIVING] {total} frames -> side-by-side video {out_path}")
    pbar = tqdm(total=total, desc="frames")
    idx = 0
    while idx < total:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        drv_nhwc = preprocess(Image.fromarray(rgb))
        drv_t, drv_j = to_t(drv_nhwc), jnp.asarray(drv_nhwc)

        with torch.no_grad():
            t_tc = tm.mot_encode(drv_t)
            t_tac = tm.id_adapt(t_tc, t_ir)
            t_mac = tm.mot_decode(t_tac)
            t_out = tm.decode(t_mac, t_mar, t_fr)
        j_tc = j_mot(state, drv_j)
        j_tac = j_adapt(state, j_tc, j_ir)
        j_mac = j_mdec(state, j_tac)
        j_out = j_decode(state, j_mac, j_mar, j_fr)

        if idx == 0:                                    # detailed per-stage log for frame 0
            print("\n[FRAME 0] per-stage torch vs jax:")
            diff("motion t_c", t_tc, j_tc)
            diff("adapted ta_c", t_tac, j_tac)
            for k in range(len(j_mac)):
                diff(f"motion map ma_c[{k}]", t_mac[k], j_mac[k])
            diff("decoded frame", t_out, j_out)
            print()

        tf = torch_frame_to_np(t_out)
        jf = jax_frame_to_np(j_out)
        frame_diffs.append(float(np.abs(tf.astype(np.float64) - jf.astype(np.float64)).max()))

        tf_u8 = label((np.clip(tf, 0, 1) * 255).astype(np.uint8), "Torch")
        jf_u8 = label((np.clip(jf, 0, 1) * 255).astype(np.uint8), "JAX")
        panel = np.concatenate([tf_u8, jf_u8], axis=1)  # (H, 2W, 3) RGB
        if writer is None:
            H, W = panel.shape[0], panel.shape[1]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        idx += 1
        pbar.update(1)
    cap.release(); pbar.close()
    if writer is not None:
        writer.release()

    # --- verdict ---
    print("\n" + "=" * 78)
    if frame_diffs:
        worst = max(frame_diffs)
        avg = sum(frame_diffs) / len(frame_diffs)
        print(f" FRAMES: {len(frame_diffs)}  |  worst frame max|Δ|={worst:.3e}  |  avg={avg:.3e}")
        if worst < 1e-3:
            print(" VERDICT: MATCH (within float32 noise) -- port looks correct")
        else:
            print(" VERDICT: DIVERGENCE -- see the per-stage log above; the FIRST stage")
            print("          exceeding ~1e-3 localizes the bug.")
        print(f" side-by-side video: {out_path}")
    else:
        print(" no frames processed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
