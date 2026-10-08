"""
Stage 1: supervised pretraining on synthetic RESIDE pairs.

Protocol A: ITS (indoor)       -> validated on SOTS-indoor
Protocol B: RESIDE-6K outdoor  -> validated on SOTS-outdoor

Features for unreliable Colab sessions:
  - mixed precision (AMP) for T4 tensor cores
  - full training state saved every epoch (stage1_last.pt) and auto-resumed
  - best model weights saved separately (stage1_best.pt, plain state_dict)
  - per-epoch log appended to <checkpoints>/train_log.txt
  - writes <checkpoints>/stage1_done when finished, so pipelines can skip it

Usage:
    python src/train_stage1.py --config configs/protocol_a.yaml
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import PairedHazeDataset, reside_its_clean_key, ots6k_clean_key, sots_clean_key
from model import build_model
from losses import Stage1Loss
from utils import load_config, set_seed, get_device, AverageMeterDict, log_line, amp_settings


def build_stage1_loaders(cfg):
    protocol = cfg["protocol"]
    crop = cfg["data"]["crop_size"]
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]

    if protocol == "A":
        train_root, clean_key_fn = Path(cfg["paths"]["its_train"]), reside_its_clean_key
        val_root = Path(cfg["paths"]["sots_indoor"])
    else:
        train_root, clean_key_fn = Path(cfg["paths"]["ots_train"]), ots6k_clean_key
        val_root = Path(cfg["paths"]["sots_outdoor"])

    train_ds = PairedHazeDataset(train_root / "hazy", train_root / "clear", crop_size=crop,
                                 train=True, clean_key_fn=clean_key_fn)
    val_ds = PairedHazeDataset(val_root / "hazy", val_root / "clear", crop_size=crop,
                               train=False, clean_key_fn=sots_clean_key)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw)
    return train_loader, val_loader


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 99.0
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


@torch.no_grad()
def validate(model, val_loader, device, use_amp, amp_dtype, max_images=None):
    model.eval()
    total, n = 0.0, 0
    for i, (hazy, clean, _) in enumerate(val_loader):
        if max_images and i >= max_images:
            break
        hazy, clean = hazy.to(device), clean.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(hazy)
        total += psnr(out["J_hat"].float().clamp(0, 1), clean)
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no_resume", action="store_true", help="ignore stage1_last.pt and start over")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    use_amp, amp_dtype = amp_settings(cfg, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.txt"

    train_loader, val_loader = build_stage1_loaders(cfg)
    model = build_model(cfg["model"]).to(device)
    if cfg.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    criterion = Stage1Loss(cfg["stage1"]["loss_weights"], device=device)

    epochs = cfg["stage1"]["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["stage1"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    start_epoch, best_psnr = 0, -1.0
    last_path = ckpt_dir / "stage1_last.pt"
    if last_path.exists() and not args.no_resume:
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_psnr = state["epoch"] + 1, state["best_psnr"]
        log_line(log_path, f"[stage1] resumed from epoch {start_epoch} (best_psnr={best_psnr:.2f})")

    log_line(log_path, f"[stage1] protocol={cfg['protocol']} device={device} amp={use_amp} "
                       f"train_size={len(train_loader.dataset)} steps/epoch={len(train_loader)}")

    val_max = cfg["stage1"].get("val_max_images")
    max_steps = cfg["stage1"].get("max_steps_per_epoch")  # debug/dry-run only
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        meter = AverageMeterDict()
        for step, (hazy, clean, _) in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            hazy, clean = hazy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(hazy)
            # losses in fp32 for numerical stability (SSIM / log-variance terms)
            out = {k: v.float() for k, v in out.items()}
            total, losses = criterion(out, clean, hazy)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            meter.update(losses)

        scheduler.step()
        train_time = time.time() - t0
        val_psnr = validate(model, val_loader, device, use_amp, amp_dtype, max_images=val_max)

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            torch.save(model.state_dict(), ckpt_dir / "stage1_best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_psnr": best_psnr}, last_path)

        log_line(log_path, f"[stage1][epoch {epoch+1}/{epochs}] {meter} | val_psnr={val_psnr:.2f}"
                           f"{' (best)' if is_best else ''} | train_time={train_time/60:.1f}min "
                           f"total_time={(time.time()-t0)/60:.1f}min")

    (ckpt_dir / "stage1_done").write_text(f"best_psnr={best_psnr:.4f}\n")
    log_line(log_path, f"[stage1] done. best val_psnr={best_psnr:.2f}")


if __name__ == "__main__":
    main()
