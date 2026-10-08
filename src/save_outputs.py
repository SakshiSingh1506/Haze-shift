"""
Saves qualitative outputs for the report and the failure analysis.

For each NH-HAZE official test image (51-55) and a spread of RTTS images, it
writes the result of every available method, plus each learned model's
predicted transmission map and transmission-uncertainty map (exp(t_logvar),
min-max scaled per image so the spatial pattern is visible).

    python src/save_outputs.py --config configs/h200/protocol_b.yaml
    python src/save_outputs.py --config configs/h200/protocol_a.yaml --rtts 8

Output in <results>/visuals/:
    <image>/hazy.png, gt.png (NH-HAZE only), DCP.png, <method>.png,
    <method>_transmission.png, <method>_uncertainty.png
    <image>_grid.jpg               hazy | DCP | each model | ground truth
    <image>_uncertainty_grid.jpg   hazy | each model's uncertainty map
(grids are small JPEGs for sharing; full-size PNGs are in the per-image folders)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from baseline_dcp import dcp_dehaze
from datasets import UnlabeledHazeDataset, load_nh_haze_official_test
from model import build_model
from utils import amp_settings, get_device, load_config

METHODS = [
    ("stage1_only", "stage1_best.pt"),
    ("stage2", "stage2_last.pt"),
    ("stage2_nounc", "stage2_nounc_last.pt"),
    ("stage2_noema", "stage2_noema_last.pt"),
]


def to_img(t):
    """(3,H,W) or (1,H,W) tensor in [0,1] -> PIL image."""
    a = t.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    a = (a * 255).round().astype(np.uint8)
    return Image.fromarray(a[:, :, 0] if a.shape[2] == 1 else a)


def minmax(m):
    m = m.float()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def grid(images, labels, width=400):
    tiles = []
    for im, label in zip(images, labels):
        im = im.convert("RGB")
        h = int(im.height * width / im.width)
        tile = Image.new("RGB", (width, h + 24), "white")
        tile.paste(im.resize((width, h), Image.BICUBIC), (0, 24))
        ImageDraw.Draw(tile).text((6, 5), label, fill="black")
        tiles.append(tile)
    out = Image.new("RGB", (width * len(tiles), max(t.height for t in tiles)), "white")
    for i, tile in enumerate(tiles):
        out.paste(tile, (i * width, 0))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rtts", type=int, default=8, help="number of RTTS images, spread evenly over the set")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    use_amp, amp_dtype = amp_settings(cfg, device)
    ck = Path(cfg["paths"]["checkpoints"])
    out_root = Path(cfg["paths"]["results"]) / "visuals"

    models = {}
    for name, fname in METHODS:
        if not (ck / fname).exists():
            print(f"[save_outputs] skip {name}: {ck / fname} not found")
            continue
        m = build_model(cfg["model"]).to(device)
        m.load_state_dict(torch.load(ck / fname, map_location=device))
        models[name] = m.eval()

    samples = []
    nh = load_nh_haze_official_test(cfg["paths"]["nh_haze"])
    for i in range(len(nh)):
        hazy, clean, name = nh[i]
        samples.append((f"nh_{Path(name).stem}", hazy, clean))
    if args.rtts > 0:
        rtts = UnlabeledHazeDataset(cfg["paths"]["rtts"], crop_size=None, train=False)
        for j in np.linspace(0, len(rtts) - 1, args.rtts).astype(int):
            hazy, name = rtts[int(j)]
            samples.append((f"rtts_{Path(name).stem}", hazy, None))

    for tag, hazy, clean in samples:
        d = out_root / tag
        d.mkdir(parents=True, exist_ok=True)
        hazy_img = to_img(hazy)
        hazy_img.save(d / "hazy.png")
        images, labels = [hazy_img], ["hazy"]

        dcp = dcp_dehaze((hazy.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        dcp_img = Image.fromarray((dcp * 255).round().astype(np.uint8))
        dcp_img.save(d / "DCP.png")
        images.append(dcp_img)
        labels.append("DCP")

        x = hazy.unsqueeze(0).to(device)
        unc_images, unc_labels = [hazy_img], ["hazy"]
        for name, m in models.items():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                o = m(x)
            j_img = to_img(o["J_hat"][0])
            j_img.save(d / f"{name}.png")
            to_img(o["t_mean"][0]).save(d / f"{name}_transmission.png")
            u_img = to_img(minmax(torch.exp(o["t_logvar"][0].float())))
            u_img.save(d / f"{name}_uncertainty.png")
            images.append(j_img)
            labels.append(name)
            unc_images.append(u_img)
            unc_labels.append(f"{name} uncertainty")

        if clean is not None:
            gt = to_img(clean)
            gt.save(d / "gt.png")
            images.append(gt)
            labels.append("ground truth")

        grid(images, labels).save(out_root / f"{tag}_grid.jpg", quality=90)
        if len(unc_images) > 1:
            grid(unc_images, unc_labels).save(out_root / f"{tag}_uncertainty_grid.jpg", quality=90)
        print(f"[save_outputs] {tag}: {len(models)} model(s) + DCP -> {d}")

    print(f"[save_outputs] done -> {out_root}")


if __name__ == "__main__":
    main()
