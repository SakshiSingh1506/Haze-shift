"""
Classical Dark Channel Prior baseline (He, Sun & Tang, CVPR 2009).
No training -- sets the metric floor that any learned method should clear.

Usage:
    python src/baseline_dcp.py --input_dir data/NH-HAZE/hazy --output_dir results/dcp_baseline/nh_haze

Then score the output_dir against the matching clear/ folder using the same
PSNR/SSIM/LPIPS/NIQE functions in src/eval.py to get a comparable row for
metrics.csv.
"""
import numpy as np


def dark_channel_np(img, patch=15):
    """Min over RGB, then a min-filter over a patch (scipy, vectorized)."""
    from scipy.ndimage import minimum_filter
    return minimum_filter(img.min(axis=2), size=patch, mode="nearest")


def estimate_airlight(img, dark, top_percent=0.001):
    h, w = dark.shape
    n_pixels = max(int(h * w * top_percent), 1)
    flat_dark = dark.flatten()
    idx = np.argpartition(flat_dark, -n_pixels)[-n_pixels:]
    flat_img = img.reshape(-1, 3)
    candidates = flat_img[idx]
    A = candidates[np.argmax(candidates.sum(axis=1))]
    return A


def estimate_transmission(img, A, omega=0.95, patch=15):
    norm_img = img / (A + 1e-6)
    t = 1 - omega * dark_channel_np(norm_img, patch)
    return t


def guided_filter(I, p, radius=40, eps=1e-3):
    """Guided filter (He et al.) for transmission refinement, box means via uniform_filter."""
    from scipy.ndimage import uniform_filter
    size = 2 * radius + 1
    mean_I = uniform_filter(I, size)
    mean_p = uniform_filter(p, size)
    cov_Ip = uniform_filter(I * p, size) - mean_I * mean_p
    var_I = uniform_filter(I * I, size) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return uniform_filter(a, size) * I + uniform_filter(b, size)


def dcp_dehaze(img_uint8, patch=15, omega=0.95, t0=0.1, refine=True):
    """img_uint8: HxWx3 uint8 RGB array. Returns dehazed HxWx3 float array in [0,1]."""
    img = img_uint8.astype(np.float64) / 255.0
    dark = dark_channel_np(img, patch)
    A = estimate_airlight(img, dark)
    t = estimate_transmission(img, A, omega, patch)

    if refine:
        gray = img.mean(axis=2)
        t = guided_filter(gray, t, radius=40, eps=1e-3)

    t = np.clip(t, t0, 1.0)
    J = np.zeros_like(img)
    for c in range(3):
        J[:, :, c] = (img[:, :, c] - A[c]) / t + A[c]
    return np.clip(J, 0, 1)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PIL import Image

    parser = argparse.ArgumentParser(description="Run DCP baseline on a folder of hazy images.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(args.input_dir).glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = np.array(Image.open(p).convert("RGB"))
        result = dcp_dehaze(img)
        Image.fromarray((result * 255).astype(np.uint8)).save(out_dir / p.name)
        print(f"[dcp] {p.name} -> {out_dir / p.name}")
