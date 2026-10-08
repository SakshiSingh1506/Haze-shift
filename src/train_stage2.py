"""
Stage 2: unsupervised fine-tuning on real, unlabeled haze (RTTS).

No ground truth is used here -- supervision comes entirely from the
uncertainty-weighted physical-consistency loss and the prior losses
(dark/bright channel, transmission smoothness) in src/losses.py. This is the
stage that is meant to close the synthetic->real domain gap.

Same robustness features as stage 1: AMP, full state saved every epoch
(stage2_last_state.pt, includes the EMA teacher) with auto-resume, log file,
and a stage2_done marker.

Usage (after train_stage1.py):
    python src/train_stage2.py --config configs/protocol_a.yaml
    python src/train_stage2.py --config configs/protocol_a.yaml --no_ema   # ablation
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import UnlabeledHazeDataset
from model import build_model
from losses import Stage2Loss
from utils import load_config, set_seed, get_device, AverageMeterDict, EMA, log_line, amp_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no_ema", action="store_true", help="ablation: disable EMA teacher distillation")
    parser.add_argument("--no_uncertainty", action="store_true",
                        help="ablation: plain (unweighted) physical consistency instead of uncertainty-weighted")
    parser.add_argument("--no_resume", action="store_true")
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
    log_path = ckpt_dir / "train_log.txt"
    stage1_ckpt = ckpt_dir / "stage1_best.pt"
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"{stage1_ckpt} not found -- run train_stage1.py first.")

    use_ema = cfg["model"].get("ema_teacher", True) and not args.no_ema
    tag = "stage2" + ("" if use_ema else "_noema") + ("_nounc" if args.no_uncertainty else "")

    model = build_model(cfg["model"]).to(device)
    if cfg.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
    ema = EMA(model, decay=cfg["model"].get("ema_decay", 0.999)) if use_ema else None

    rtts_ds = UnlabeledHazeDataset(cfg["paths"]["rtts"], crop_size=cfg["data"]["crop_size"], train=True)
    nw = cfg["data"]["num_workers"]
    loader = DataLoader(rtts_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=nw,
                        pin_memory=True, drop_last=True, persistent_workers=nw > 0)

    weights = dict(cfg["stage2"]["loss_weights"])
    if not use_ema:
        weights["ema_distillation"] = 0.0
    if args.no_uncertainty:
        weights["plain_consistency"] = weights.pop("uncertainty_consistency")
    criterion = Stage2Loss(weights)

    epochs = cfg["stage2"]["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["stage2"]["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    start_epoch = 0
    state_path = ckpt_dir / f"{tag}_last_state.pt"
    if state_path.exists() and not args.no_resume:
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema") is not None:
            ema.teacher.load_state_dict(state["ema"])
        start_epoch = state["epoch"] + 1
        log_line(log_path, f"[{tag}] resumed from epoch {start_epoch}")

    log_line(log_path, f"[{tag}] protocol={cfg['protocol']} amp={use_amp} rtts_size={len(rtts_ds)} "
                       f"steps/epoch={len(loader)} from={stage1_ckpt.name}")

    max_steps = cfg["stage2"].get("max_steps_per_epoch")  # debug/dry-run only
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        meter = AverageMeterDict()
        for step, (hazy, _) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            hazy = hazy.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(hazy)
                teacher_out = None
                if ema is not None:
                    with torch.no_grad():
                        teacher_out = ema.teacher(hazy)
            out = {k: v.float() for k, v in out.items()}
            if teacher_out is not None:
                teacher_out = {k: v.float() for k, v in teacher_out.items()}

            total, losses = criterion(out, hazy, teacher_out)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            meter.update(losses)

        # final weights used for eval: plain state_dict
        torch.save(model.state_dict(), ckpt_dir / f"{tag}_last.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "ema": ema.teacher.state_dict() if ema is not None else None}, state_path)
        log_line(log_path, f"[{tag}][epoch {epoch+1}/{epochs}] {meter} | time={(time.time()-t0)/60:.1f}min")

    (ckpt_dir / f"{tag}_done").write_text("ok\n")
    log_line(log_path, f"[{tag}] done.")


if __name__ == "__main__":
    main()
