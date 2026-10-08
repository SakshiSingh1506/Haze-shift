"""
Evaluation: PSNR / SSIM / LPIPS on SOTS (in-domain) and NH-HAZE official test
(zero-shot, 5 pairs), plus NIQE (no-reference) on the full 55-image NH-HAZE
set and on RTTS as a steadier secondary check. Also reports the
cross-domain drop (in-domain PSNR - NH-HAZE PSNR) for each run, so
protocol A vs B, stage1-only vs stage1+stage2, and DCP can be compared in one
table (<results>/metrics.csv). Per-image NH-HAZE scores go to
<results>/per_image_<tag>.csv for failure analysis.

Usage:
    python src/eval.py --config configs/protocol_a.yaml --checkpoint checkpoints/protocol_a/stage2_last.pt --tag A_stage2
    python src/eval.py --config configs/protocol_a.yaml --checkpoint checkpoints/protocol_a/stage1_best.pt --tag A_stage1_only
    python src/eval.py --config configs/protocol_a.yaml --dcp --tag A_DCP
    python src/eval.py --config configs/protocol_a.yaml --identity --tag A_hazy_input   # no dehazing at all

Requires: pip install lpips pyiqa
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import (
    PairedHazeDataset, sots_clean_key, load_nh_haze_official_test, load_full_nh_haze,
    UnlabeledHazeDataset,
)
from model import build_model
from losses import ssim_loss
from utils import load_config, get_device, amp_settings


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 99.0
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim_metric(pred, target):
    return 1 - ssim_loss(pred, target).item()


def get_lpips_fn(device):
    try:
        import lpips
        fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        return lambda pred, target: fn(pred * 2 - 1, target * 2 - 1).mean().item()
    except ImportError:
        print("[eval] WARNING: `lpips` not installed, skipping LPIPS (pip install lpips)")
        return None


def get_niqe_fn(device):
    try:
        import pyiqa
        fn = pyiqa.create_metric("niqe", device=device)
        return lambda img: fn(img).mean().item()
    except ImportError:
        print("[eval] WARNING: `pyiqa` not installed, skipping NIQE (pip install pyiqa)")
        return None


def make_predictor(args, cfg, device, use_amp, amp_dtype):
    """Returns f(hazy_tensor[1,3,H,W] in [0,1]) -> dehazed tensor, for either the model or DCP."""
    if args.identity:
        # reference row: score the hazy input itself, i.e. "do nothing"
        return lambda hazy: hazy.clamp(0, 1)
    if args.dcp:
        from baseline_dcp import dcp_dehaze

        def predict(hazy):
            img = (hazy[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            out = dcp_dehaze(img)
            return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).float().to(device)
        return predict

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"[eval] loaded {args.checkpoint}")

    @torch.no_grad()
    def predict(hazy):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(hazy)
        return out["J_hat"].float().clamp(0, 1)
    return predict


@torch.no_grad()
def eval_paired(predict, loader, device, lpips_fn, max_images=None):
    rows = []
    for i, (hazy, clean, name) in enumerate(loader):
        if max_images and i >= max_images:
            break
        hazy, clean = hazy.to(device), clean.to(device)
        pred = predict(hazy)
        row = {"image": name[0], "psnr": psnr(pred, clean), "ssim": ssim_metric(pred, clean)}
        if lpips_fn is not None:
            row["lpips"] = lpips_fn(pred, clean)
        rows.append(row)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "image"}
    summary.update({f"{k}_std": float(np.std([r[k] for r in rows])) for k in rows[0] if k != "image"})
    return summary, rows


@torch.no_grad()
def eval_niqe(predict, loader, niqe_fn, device, max_images=None):
    scores = []
    for i, batch in enumerate(loader):
        if max_images and i >= max_images:
            break
        scores.append(niqe_fn(predict(batch[0].to(device))))
    return float(np.mean(scores)) if scores else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="model weights (state_dict); not needed with --dcp")
    parser.add_argument("--dcp", action="store_true", help="evaluate the classical DCP baseline instead of a model")
    parser.add_argument("--identity", action="store_true", help="score the hazy input itself (no dehazing) as a reference")
    parser.add_argument("--tag", required=True, help="label for this run in metrics.csv, e.g. A_stage2")
    args = parser.parse_args()
    if not (args.dcp or args.identity or args.checkpoint):
        parser.error("--checkpoint is required unless --dcp or --identity is given")

    cfg = load_config(args.config)
    device = get_device()
    use_amp, amp_dtype = amp_settings(cfg, device)
    predict = make_predictor(args, cfg, device, use_amp, amp_dtype)

    lpips_fn = get_lpips_fn(device)
    niqe_fn = get_niqe_fn(device)

    # --- in-domain SOTS ---
    key = "sots_indoor" if cfg["protocol"] == "A" else "sots_outdoor"
    in_domain_name = "SOTS-indoor" if cfg["protocol"] == "A" else "SOTS-outdoor"
    sots_root = Path(cfg["paths"][key])
    sots_ds = PairedHazeDataset(sots_root / "hazy", sots_root / "clear", crop_size=None, train=False,
                                clean_key_fn=sots_clean_key)
    in_domain, sots_rows = eval_paired(predict, DataLoader(sots_ds, batch_size=1), device, lpips_fn,
                                       cfg["eval"].get("sots_max_images"))  # None = all 500
    print(f"[eval] {in_domain_name} (n={len(sots_rows)}): {in_domain}")

    # --- zero-shot NH-HAZE official test (5 pairs) ---
    nh_test_ds = load_nh_haze_official_test(cfg["paths"]["nh_haze"])
    nh, nh_rows = eval_paired(predict, DataLoader(nh_test_ds, batch_size=1), device, lpips_fn)
    print(f"[eval] NH-HAZE official test (n={len(nh_test_ds)}): {nh}")

    # --- secondary: NIQE on outputs, no GT needed ---
    niqe_full_nh = niqe_rtts = None
    if niqe_fn is not None:
        full_nh_ds = load_full_nh_haze(cfg["paths"]["nh_haze"])
        niqe_full_nh = eval_niqe(predict, DataLoader(full_nh_ds, batch_size=1), niqe_fn, device)
        print(f"[eval] NIQE over full NH-HAZE (n={len(full_nh_ds)}): {niqe_full_nh:.4f}")

        rtts_max = cfg["eval"].get("rtts_niqe_max_images")
        rtts_ds = UnlabeledHazeDataset(cfg["paths"]["rtts"], crop_size=None, train=False)
        niqe_rtts = eval_niqe(predict, DataLoader(rtts_ds, batch_size=1), niqe_fn, device, rtts_max)
        print(f"[eval] NIQE over RTTS (first {rtts_max or len(rtts_ds)}): {niqe_rtts:.4f}")

    drop = in_domain["psnr"] - nh["psnr"]
    print(f"[eval] cross-domain PSNR drop ({in_domain_name} -> NH-HAZE): {drop:.2f} dB")

    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "tag": args.tag, "protocol": cfg["protocol"], "in_domain_set": in_domain_name,
        "in_domain_psnr": in_domain["psnr"], "in_domain_ssim": in_domain["ssim"],
        "in_domain_lpips": in_domain.get("lpips"),
        "nh_haze_psnr": nh["psnr"], "nh_haze_psnr_std": nh["psnr_std"],
        "nh_haze_ssim": nh["ssim"], "nh_haze_lpips": nh.get("lpips"),
        "niqe_full_nh_haze": niqe_full_nh, "niqe_rtts": niqe_rtts,
        "cross_domain_psnr_drop": drop,
    }
    out_csv = results_dir / "metrics.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    with open(results_dir / f"per_image_{args.tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(nh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(nh_rows)
    print(f"[eval] appended results to {out_csv}")


if __name__ == "__main__":
    main()
