"""inference.py — load renderer.ckpt into the JAX IMTRenderer and run
video-driven inference (source image + driving video -> mp4).

    test_env/bin/python renderer_jax/inference.py \
        --renderer_path checkpoints/renderer.ckpt \
        --source_path assets/source_1.png \
        --driving_path assets/driving_1.mp4 \
        --save_path results/ [--crop]

Pure inference pipeline + I/O — imports no torch. The torch->jax weight migration
lives in scripts/port_weights.py (load_model).
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import jax.numpy as jnp

import cv2
from PIL import Image
from tqdm import tqdm

from renderer_jax.config import RendererConfig
from scripts.port_weights import load_model

# -----------------------------------------------------------------------------
# Preprocess / postprocess  (I/O layer)
# -----------------------------------------------------------------------------
def preprocess(pil_img, size=512, fa=None, crop=False):
    """PIL RGB -> (1, size, size, 3) float32 NHWC in [0,1]."""
    if crop and fa is not None:
        pil_img = _face_crop(pil_img, fa, size)
    img = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # (H,W,3) [0,1]
    return arr[None]                                     # (1,H,W,3)


def _face_crop(pil_img, fa, size):
    img = np.array(pil_img)
    bboxes = fa.face_detector.detect_from_image(img)
    valid = [b for b in bboxes if b[4] > 0.95]
    if not valid:
        return pil_img
    x1, y1, x2, y2, _ = valid[0]
    bs = int(max((y2 - y1) / 2, (x2 - x1) / 2) * 1.6)
    my, mx = int((y1 + y2) / 2), int((x1 + x2) / 2)
    img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
    my, mx = my + bs, mx + bs
    return Image.fromarray(img[my - bs:my + bs, mx - bs:mx + bs])


def save_video(frames_nhwc, path, fps):
    """frames_nhwc: list/array of (H,W,3) float [0,1] -> mp4."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    vid = (np.clip(np.stack(frames_nhwc), 0, 1) * 255).astype(np.uint8)
    T, H, W, _ = vid.shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in vid:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[Success] video saved to {path}  ({T} frames)")

# -----------------------------------------------------------------------------
# Inference pipeline (mirror torch renderer/inference.py)
# -----------------------------------------------------------------------------
def process_single(model, source_path, driving_path, save_path, crop=False, fps=None):
    fa = None
    if crop:
        import face_alignment                                  # lazy: only needed with --crop
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    ura = True   

    # --- source: encode ONCE ---
    src = Image.open(source_path)
    src_x = jnp.asarray(preprocess(src, fa=fa, crop=crop))
    f_r, i_r = model.app_encode(src_x, use_running_average=ura)
    t_r = model.mot_encode(src_x)
    ta_r = model.id_adapt(t_r, i_r)
    ma_r = model.mot_decode(ta_r)

    # --- driving video: per frame ---
    cap = cv2.VideoCapture(driving_path)
    out_fps = cap.get(cv2.CAP_PROP_FPS) if fps is None else fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    pbar = tqdm(total=total, desc="Inferencing")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        drv_x = jnp.asarray(preprocess(Image.fromarray(rgb)))
        t_c = model.mot_encode(drv_x)
        ta_c = model.id_adapt(t_c, i_r)
        ma_c = model.mot_decode(ta_c)
        out = model.decode(ma_c, ma_r, f_r, use_running_average=ura)   # (1,512,512,3)
        frames.append(np.asarray(out)[0])
        pbar.update(1)
    cap.release(); pbar.close()

    if not frames:
        print("[Error] no frames generated."); return
    name = f"{os.path.splitext(os.path.basename(source_path))[0]}_" \
           f"{os.path.splitext(os.path.basename(driving_path))[0]}.mp4"
    save_video(frames, os.path.join(save_path, name), out_fps)


def main():
    p = argparse.ArgumentParser(description="JAX IMTRenderer video-driven inference demo")
    p.add_argument("--renderer_path", default="checkpoints/renderer.ckpt")
    p.add_argument("--source_path", required=True)
    p.add_argument("--driving_path", required=True)
    p.add_argument("--save_path", default="results/")
    p.add_argument("--crop", action="store_true")
    p.add_argument("--fps", type=int, default=None)
    args = p.parse_args()

    model = load_model(args.renderer_path, RendererConfig())
    process_single(model, args.source_path, args.driving_path,
                   args.save_path, crop=args.crop, fps=args.fps)


if __name__ == "__main__":
    main()
