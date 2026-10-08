"""
Loss functions for HazeShiftNet.

Stage 1 (supervised, synthetic pairs): l1_loss + ssim_loss + perceptual_loss
 + physical_consistency_loss(I_hat, hazy_input)

Stage 2 (unsupervised, real unlabeled haze): uncertainty_weighted_consistency_loss
 + dark_channel_prior_loss + bright_channel_prior_loss + tv_loss(t_mean)
 + optional ema_distillation_loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------------------
# Stage 1: supervised losses
# ---------------------------------------------------------------------------

def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def _gaussian_window(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window2d = g @ g.t()
    window = window2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim_loss(pred, target, window_size=11):
    """1 - SSIM, averaged over the batch. Self-contained (no external dep)."""
    channels = pred.shape[1]
    window = _gaussian_window(window_size, channels=channels, device=pred.device)
    pad = window_size // 2

    mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_y = F.conv2d(target, window, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return 1 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 relu2_2/relu3_3 features. Frozen pretrained VGG."""

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(9)])    # up to relu2_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(9, 16)])  # up to relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        p1 = self.slice1(pred_n)
        t1 = self.slice1(target_n)
        p2 = self.slice2(p1)
        t2 = self.slice2(t1)
        return F.l1_loss(p1, t1) + F.l1_loss(p2, t2)


def physical_consistency_loss(I_hat, I_input):
    """Re-synthesized hazy image should match the actual hazy input (ASM sanity check)."""
    return F.l1_loss(I_hat, I_input)


# ---------------------------------------------------------------------------
# Stage 2: unsupervised losses on real, unlabeled haze
# ---------------------------------------------------------------------------

def uncertainty_weighted_consistency_loss(I_hat, I_input, t_logvar, a_logvar):
    """
    Heteroscedastic loss: where the model is uncertain about the haze estimate
    (high predicted variance), the reconstruction error is down-weighted, but
    the log-variance term penalizes claiming high uncertainty everywhere.

        loss = error * exp(-logvar) + logvar

    This is the mechanism that lets the model say "I'm not confident about
    this patch of real haze" instead of confidently applying a synthetic-
    haze-shaped correction to an out-of-distribution input.
    """
    error = (I_hat - I_input).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)
    precision_t = torch.exp(-t_logvar)
    loss_t = (error * precision_t + t_logvar).mean()

    # airlight uncertainty: penalize on the global (per-image) reconstruction error
    img_error = (I_hat - I_input).pow(2).mean(dim=[1, 2, 3])  # (B,)
    precision_a = torch.exp(-a_logvar.squeeze(-1))
    loss_a = (img_error * precision_a + a_logvar.squeeze(-1)).mean()

    return loss_t + 0.1 * loss_a


def dark_channel(img, patch_size=15):
    """Dark channel = min over RGB, then min-pooled over a local patch."""
    min_rgb, _ = img.min(dim=1, keepdim=True)
    pad = patch_size // 2
    dark = -F.max_pool2d(-min_rgb, kernel_size=patch_size, stride=1, padding=pad)
    return dark


def dark_channel_prior_loss(J_hat, patch_size=15):
    """Clean, haze-free images should have a dark channel close to 0 (He et al. 2009).
    Used here as an unsupervised regularizer (PSD-style) since no GT exists on real data."""
    dc = dark_channel(J_hat, patch_size)
    return dc.mean()


def bright_channel_prior_loss(J_hat, I_input, patch_size=15):
    """Bright channel prior (PSD, CVPR 2021): the output's bright channel should not
    exceed the input's, discouraging over-brightened / washed-out corrections."""
    max_rgb_out, _ = J_hat.max(dim=1, keepdim=True)
    max_rgb_in, _ = I_input.max(dim=1, keepdim=True)
    pad = patch_size // 2
    bright_out = F.max_pool2d(max_rgb_out, kernel_size=patch_size, stride=1, padding=pad)
    bright_in = F.max_pool2d(max_rgb_in, kernel_size=patch_size, stride=1, padding=pad)
    return F.relu(bright_out - bright_in).mean()


def tv_loss(x):
    """Total variation smoothness, applied to the transmission map."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def ema_distillation_loss(student_out, teacher_out):
    """Student's restoration should stay close to a slowly-updated EMA teacher's
    restoration -- stabilizes stage-2 unsupervised updates (standard trick in
    unsupervised domain adaptation / mean-teacher setups)."""
    return F.l1_loss(student_out["J_hat"], teacher_out["J_hat"].detach())


# ---------------------------------------------------------------------------
# Aggregators used directly by train_stage1.py / train_stage2.py
# ---------------------------------------------------------------------------

class Stage1Loss(nn.Module):
    def __init__(self, weights: dict, device="cpu"):
        super().__init__()
        self.w = weights
        self.perceptual = VGGPerceptualLoss().to(device) if weights.get("perceptual", 0) > 0 else None

    def forward(self, out, J_gt, I_input):
        losses = {}
        losses["l1"] = l1_loss(out["J_hat"], J_gt)
        losses["ssim"] = ssim_loss(out["J_hat"], J_gt)
        if self.perceptual is not None:
            losses["perceptual"] = self.perceptual(out["J_hat"], J_gt)
        losses["physical_consistency"] = physical_consistency_loss(out["I_hat"], I_input)

        total = sum(self.w.get(k, 0.0) * v for k, v in losses.items())
        losses["total"] = total
        return total, losses


class Stage2Loss(nn.Module):
    def __init__(self, weights: dict):
        super().__init__()
        self.w = weights

    def forward(self, out, I_input, teacher_out=None):
        losses = {}
        if self.w.get("uncertainty_consistency", 0) > 0:
            losses["uncertainty_consistency"] = uncertainty_weighted_consistency_loss(
                out["I_hat"], I_input, out["t_logvar"], out["a_logvar"]
            )
        if self.w.get("plain_consistency", 0) > 0:
            # ablation: same physical re-synthesis check, no uncertainty weighting (PSD-style)
            losses["plain_consistency"] = physical_consistency_loss(out["I_hat"], I_input)
        losses["dark_channel_prior"] = dark_channel_prior_loss(out["J_hat"])
        losses["bright_channel_prior"] = bright_channel_prior_loss(out["J_hat"], I_input)
        losses["tv_transmission"] = tv_loss(out["t_mean"])
        if teacher_out is not None and self.w.get("ema_distillation", 0) > 0:
            losses["ema_distillation"] = ema_distillation_loss(out, teacher_out)

        total = sum(self.w.get(k, 0.0) * v for k, v in losses.items())
        losses["total"] = total
        return total, losses
