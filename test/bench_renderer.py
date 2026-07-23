"""bench_renderer.py — real-video SPEED benchmark: jit-JAX vs Torch.

Measures steady-state per-frame throughput of the SAME renderer weights on both
frameworks, on real driving video(s), and writes a side-by-side "Torch | JAX" video
with total time + FPS captioned under each. Correctness is validated separately
(test/test_renderer.py ~1e-6); this is SPEED only.

    test_env/bin/python test/bench_renderer.py \
        --renderer_path checkpoints/renderer.ckpt \
        --source_path assets/source_1.png \
        --driving_paths assets/driving_1.mp4 assets/driving_2.mp4 assets/driving_3.mp4 \
        --save_path results/

HONEST MEASUREMENT (why naive wall-clock lies):
  * GPU work is ASYNC -> we jax.block_until_ready / torch.cuda.synchronize before
    stopping any timer, else we'd time dispatch not compute.
  * WARMUP frames (jax XLA compile + torch cudnn autotune) are discarded from the
    steady FPS and reported separately as "load/compile".
  * Torch and JAX run SEQUENTIALLY (torch fully -> free GPU -> jax) so they never
    contend for the GPU.
  * Torch runs at its DEFAULT precision (TF32 on Ampere) = real deployment speed;
    precision parity was checked elsewhere, not here.
  * If JAX has no CUDA backend (falls back to CPU) the comparison is meaningless ->
    the script warns and aborts unless --allow_cpu.
"""

import os
import sys
import gc
import time
import json
import argparse
import statistics

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
from renderer_jax.inference import preprocess
from scripts.port_weights import port_imt_renderer


# ---------------------------------------------------------------- helpers
def build_torch(ckpt_path, cfg, device):
    from renderer.models import IMTRenderer as TIMT
    args = SimpleNamespace(num_heads=cfg.num_heads, window_size=cfg.window_size,
                           swin_res_threshold=cfg.swin_res_threshold)
    m = TIMT(args)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k.replace("gen.", ""): v for k, v in sd.items() if k.startswith("gen.")} or sd
        m.load_state_dict(sd, strict=False)
        print("[INFO] torch ckpt loaded")
    else:
        print("[WARN] no ckpt -> random weights (speed numbers still valid; video not meaningful)")
    return m.eval().to(device)


def read_all_frames(driving_paths, max_frames=None):
    """Decode + preprocess every frame up front (NOT part of model timing)."""
    frames, fps = [], None
    for path in driving_paths:
        cap = cv2.VideoCapture(path)
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(rgb)))    # (1,512,512,3) np
            if max_frames and len(frames) >= max_frames:
                cap.release()
                return frames, fps
        cap.release()
    return frames, fps


def label(img_u8, top, bottom):
    """Top title banner + bottom caption on a panel."""
    img = img_u8.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 44), (0, 0, 0), -1)
    (tw, _), _ = cv2.getTextSize(top, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(img, top, (max((w - tw) // 2, 8), 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
    (bw, _), _ = cv2.getTextSize(bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(img, bottom, (max((w - bw) // 2, 8), h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def summarize(ms):
    med = statistics.median(ms)
    return med, sum(ms) / 1000.0, 1000.0 / med                 # median_ms, total_s, fps


def jax_peak_vram_gb():
    """Real peak GPU usage (BFC 'peak_bytes_in_use'), NOT jax's ~75% preallocated pool.
    Returns nan on backends without memory stats (e.g. CPU)."""
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- torch bench
def bench_torch(ckpt_path, cfg, frames_np, warmup, device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tm = build_torch(ckpt_path, cfg, device)
    src_t = torch.from_numpy(frames_np[0].transpose(0, 3, 1, 2)).to(device)   # frame[0] as source
    with torch.no_grad():
        f_r, i_r = tm.app_encode(src_t)
        ma_r = tm.mot_decode(tm.id_adapt(tm.mot_encode(src_t), i_r))
    if device == "cuda":
        torch.cuda.synchronize()
    t_load = time.perf_counter() - t0

    def one(drv_np):
        drv = torch.from_numpy(drv_np.transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad():
            ma_c = tm.mot_decode(tm.id_adapt(tm.mot_encode(drv), i_r))
            out = tm.decode(ma_c, ma_r, f_r)
        if device == "cuda":
            torch.cuda.synchronize()
        return out.detach().cpu().numpy()[0].transpose(1, 2, 0)   # (H,W,3)

    for i in range(min(warmup, len(frames_np))):               # warmup: cudnn autotune / alloc
        one(frames_np[i])

    per_ms, outs = [], []
    for drv in tqdm(frames_np, desc="torch"):
        s = time.perf_counter()
        outs.append(one(drv))
        per_ms.append((time.perf_counter() - s) * 1000.0)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    del tm
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return t_load, per_ms, outs, peak_gb


# ---------------------------------------------------------------- jax bench
def bench_jax(ckpt_path, cfg, frames_np, warmup):
    t0 = time.perf_counter()
    # build jax model + port the SAME torch ckpt weights (port source freed after)
    tm = build_torch(ckpt_path, cfg, "cpu")
    jm = IMTRenderer(cfg, rngs=nnx.Rngs(0))
    port_imt_renderer(tm, jm)
    del tm
    gc.collect()

    gd, state = nnx.split(jm)

    # clean jitted helpers: source-encode sub-steps + one whole per-frame chain
    j_app   = jax.jit(lambda s, x: nnx.merge(gd, s).app_encode(x, use_running_average=True))
    j_mot   = jax.jit(lambda s, x: nnx.merge(gd, s).mot_encode(x))
    j_adapt = jax.jit(lambda s, t, i: nnx.merge(gd, s).id_adapt(t, i))
    j_mdec  = jax.jit(lambda s, t: nnx.merge(gd, s).mot_decode(t))

    def _frame(s, drv, i_r, ma_r, f_r):                        # full per-frame chain, one fused fn
        mdl = nnx.merge(gd, s)
        ma_c = mdl.mot_decode(mdl.id_adapt(mdl.mot_encode(drv), i_r))
        return mdl.decode(ma_c, ma_r, f_r, use_running_average=True)
    j_frame = jax.jit(_frame)

    src = jnp.asarray(frames_np[0])
    f_r, i_r = j_app(state, src)
    ma_r = j_mdec(state, j_adapt(state, j_mot(state, src), i_r))
    jax.block_until_ready((f_r, i_r, ma_r))
    t_load = time.perf_counter() - t0                          # build + port (XLA compile lands in warmup)

    def one(drv_np):
        out = j_frame(state, jnp.asarray(drv_np), i_r, ma_r, f_r)
        jax.block_until_ready(out)
        return np.asarray(out)[0]                              # (H,W,3)

    for i in range(min(warmup, len(frames_np))):               # warmup: triggers XLA compile of j_frame
        one(frames_np[i])

    per_ms, outs = [], []
    for drv in tqdm(frames_np, desc="jax"):
        s = time.perf_counter()
        outs.append(one(drv))
        per_ms.append((time.perf_counter() - s) * 1000.0)
    return t_load, per_ms, outs, jax_peak_vram_gb()


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="jit-JAX vs Torch renderer speed benchmark")
    p.add_argument("--renderer_path", default="checkpoints/renderer.ckpt")
    p.add_argument("--source_path", required=True)             # CLI symmetry; frame[0] used as source
    p.add_argument("--driving_paths", nargs="+", required=True)
    p.add_argument("--save_path", default="results/")
    p.add_argument("--run_name", default=None, help="folder name under save_path (default: bench_<timestamp>)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--allow_cpu", action="store_true")
    args = p.parse_args()

    cfg = RendererConfig()
    tdev = "cuda" if torch.cuda.is_available() else "cpu"
    jdev = jax.devices()[0].platform
    print("=" * 78)
    print(f" SPEED benchmark: jit-JAX ({jdev}) vs Torch ({tdev})  |  float32")
    print("=" * 78)
    if jdev == "cpu" and not args.allow_cpu:
        print(" ABORT: JAX is on CPU (no cuda jaxlib) -> jax-vs-torch(gpu) speed compare is")
        print("        meaningless. Install 'jax[cuda12]==0.10.0', or pass --allow_cpu to force.")
        return 1

    frames, fps = read_all_frames(args.driving_paths, args.max_frames)
    if not frames:
        print(" no frames decoded from --driving_paths."); return 1
    print(f"[INFO] {len(frames)} frames from {len(args.driving_paths)} video(s) @ {fps:.1f} fps\n")

    # sequential: torch fully, freed, then jax
    tl_t, ms_t, out_t, vram_t = bench_torch(args.renderer_path, cfg, frames, args.warmup, tdev)
    tl_j, ms_j, out_j, vram_j = bench_jax(args.renderer_path, cfg, frames, args.warmup)

    med_t, tot_t, fps_t = summarize(ms_t)
    med_j, tot_j, fps_j = summarize(ms_j)

    print("\n" + "=" * 78)
    print(f" {'side':10s} {'load(s)':>9s} {'median ms/f':>12s} {'FPS':>8s} {'total(s)':>10s} {'VRAM(GB)':>10s}")
    print(f" {'Torch':10s} {tl_t:9.2f} {med_t:12.2f} {fps_t:8.2f} {tot_t:10.2f} {vram_t:10.2f}")
    print(f" {'JAX(jit)':10s} {tl_j:9.2f} {med_j:12.2f} {fps_j:8.2f} {tot_j:10.2f} {vram_j:10.2f}")
    ratio = fps_j / fps_t if fps_t else float("inf")
    print("-" * 78)
    print(f" STEADY-STATE SPEEDUP (JAX FPS / Torch FPS): {ratio:.2f}x  "
          f"({'JAX faster' if ratio > 1 else 'Torch faster'})")
    print(" (load/compile is one-time; steady FPS is what 'runs persistently' measures)")
    print("=" * 78)

    # --- persist this run: results/<run_name>/{comparison.mp4, results.json} ---
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_path, args.run_name or f"bench_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "comparison.mp4")

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({
            "timestamp": stamp,
            "torch_device": tdev, "jax_device": jdev,
            "n_frames": len(frames), "video_fps": fps, "warmup": args.warmup,
            "torch": {"load_s": tl_t, "median_ms": med_t, "fps": fps_t, "total_s": tot_t, "peak_vram_gb": vram_t},
            "jax":   {"load_s": tl_j, "median_ms": med_j, "fps": fps_j, "total_s": tot_j, "peak_vram_gb": vram_j},
            "speedup_jax_over_torch": ratio,
        }, f, indent=2)
    cap_t = f"total {tot_t:.1f}s | {fps_t:.1f} FPS"
    cap_j = f"total {tot_j:.1f}s | {fps_j:.1f} FPS"
    writer = None
    for tf, jf in zip(out_t, out_j):
        tp = label((np.clip(tf, 0, 1) * 255).astype(np.uint8), "Torch", cap_t)
        jp = label((np.clip(jf, 0, 1) * 255).astype(np.uint8), "JAX", cap_j)
        panel = np.concatenate([tp, jp], axis=1)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (panel.shape[1], panel.shape[0]))
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.release()
    print(f" saved run -> {run_dir}/  (comparison.mp4 + results.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
