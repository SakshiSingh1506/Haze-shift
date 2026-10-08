"""
HazeShiftNet: a physics-guided dehazing network with an uncertainty-aware
transmission/airlight representation.

Architecture
------------
A shared encoder feeds three heads:
  - restoration head  -> J_hat        (clean image, main output)
  - transmission head -> t_mean, t_logvar  (per-pixel haze density + uncertainty)
  - airlight head     -> A, a_logvar       (global atmospheric light + uncertainty)

Physical reconstruction (atmospheric scattering model):
    I_hat = J_hat * t_mean + A * (1 - t_mean)

`I_hat` is never used as the final output -- it exists purely so we can
compute a physical-consistency loss against the (synthetic or real) hazy
input, without needing ground truth on the real side.

Backbone
--------
The encoder/decoder trunk can be swapped for a pretrained backbone (e.g. a
DehazeFormer encoder) by replacing `build_encoder` / `build_decoder` below --
see colab/setup_colab.py for downloading public checkpoints. Default here is
a compact conv encoder-decoder (U-Net-ish) so the code runs standalone
without external weights, which is useful for quick local sanity checks on
the Mac (CPU/MPS) before moving to Colab.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.GELU(),
    )


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(conv_block(ch, ch), nn.Conv2d(ch, ch, 3, padding=1))
        self.norm = nn.InstanceNorm2d(ch, affine=True)

    def forward(self, x):
        return x + self.norm(self.body(x))


class Encoder(nn.Module):
    """Downsamples 256x256 -> 32x32 over 3 stages, with residual blocks per stage."""

    def __init__(self, base=48, n_res=3):
        super().__init__()
        self.stem = conv_block(3, base)
        self.down1 = conv_block(base, base * 2, stride=2)
        self.res1 = nn.Sequential(*[ResBlock(base * 2) for _ in range(n_res)])
        self.down2 = conv_block(base * 2, base * 4, stride=2)
        self.res2 = nn.Sequential(*[ResBlock(base * 4) for _ in range(n_res)])
        self.down3 = conv_block(base * 4, base * 8, stride=2)
        self.res3 = nn.Sequential(*[ResBlock(base * 8) for _ in range(n_res)])

    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.res1(self.down1(f0))
        f2 = self.res2(self.down2(f1))
        f3 = self.res3(self.down3(f2))
        return f0, f1, f2, f3  # skip connections + bottleneck


def up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        conv_block(in_ch, out_ch),
    )


class RestorationDecoder(nn.Module):
    """Decodes bottleneck + skips back to a clean RGB image."""

    def __init__(self, base=48):
        super().__init__()
        self.up3 = up_block(base * 8, base * 4)
        self.up2 = up_block(base * 4 * 2, base * 2)
        self.up1 = up_block(base * 2 * 2, base)
        self.out = nn.Conv2d(base * 2, 3, 3, padding=1)

    def forward(self, f0, f1, f2, f3):
        x = self.up3(f3)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up1(torch.cat([x, f1], dim=1))
        x = self.out(torch.cat([x, f0], dim=1))
        return torch.sigmoid(x)


class TransmissionHead(nn.Module):
    """Predicts per-pixel transmission mean + log-variance (heteroscedastic uncertainty)."""

    def __init__(self, base=48):
        super().__init__()
        self.up3 = up_block(base * 8, base * 4)
        self.up2 = up_block(base * 4 * 2, base * 2)
        self.up1 = up_block(base * 2 * 2, base)
        self.mean_head = nn.Conv2d(base * 2, 1, 3, padding=1)
        self.logvar_head = nn.Conv2d(base * 2, 1, 3, padding=1)

    def forward(self, f0, f1, f2, f3):
        x = self.up3(f3)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = self.up1(torch.cat([x, f1], dim=1))
        feat = torch.cat([x, f0], dim=1)
        t_mean = torch.sigmoid(self.mean_head(feat)).clamp(min=0.05, max=1.0)
        t_logvar = self.logvar_head(feat).clamp(min=-10, max=4)
        return t_mean, t_logvar


class AirlightHead(nn.Module):
    """Predicts global atmospheric light (3-vector) + scalar log-variance from pooled bottleneck features."""

    def __init__(self, base=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, base * 2),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(base * 2, 3)
        self.logvar_head = nn.Linear(base * 2, 1)

    def forward(self, f3):
        feat = self.net(f3)
        a_mean = torch.sigmoid(self.mean_head(feat))  # (B,3) in [0,1]
        a_logvar = self.logvar_head(feat).clamp(min=-10, max=4)  # (B,1)
        return a_mean, a_logvar


class HazeShiftNet(nn.Module):
    def __init__(self, base=48, n_res=3, use_uncertainty=True):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.encoder = Encoder(base=base, n_res=n_res)
        self.restoration_decoder = RestorationDecoder(base=base)
        self.transmission_head = TransmissionHead(base=base)
        self.airlight_head = AirlightHead(base=base)

    def forward(self, hazy):
        """
        hazy: (B,3,H,W) in [0,1]
        Returns dict with J_hat, t_mean, t_logvar, A, a_logvar, I_hat (physical reconstruction).
        """
        f0, f1, f2, f3 = self.encoder(hazy)
        J_hat = self.restoration_decoder(f0, f1, f2, f3)
        t_mean, t_logvar = self.transmission_head(f0, f1, f2, f3)
        A, a_logvar = self.airlight_head(f3)

        B, _, H, W = hazy.shape
        A_map = A.view(B, 3, 1, 1).expand(B, 3, H, W)
        I_hat = J_hat * t_mean + A_map * (1 - t_mean)

        return {
            "J_hat": J_hat,
            "t_mean": t_mean,
            "t_logvar": t_logvar,
            "A": A,
            "a_logvar": a_logvar,
            "I_hat": I_hat,
        }

    def load_pretrained_backbone(self, state_dict, strict=False):
        """Load a public pretrained checkpoint (e.g. DehazeFormer) into the encoder/
        restoration_decoder only; transmission/airlight heads stay randomly
        initialized and are trained from scratch in stage 1. See colab/setup_colab.py
        for how checkpoints are fetched and remapped into this schema."""
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        return missing, unexpected


def build_model(cfg_model: dict) -> HazeShiftNet:
    model = HazeShiftNet(
        base=cfg_model.get("base", 48),
        n_res=cfg_model.get("n_res", 3),
        use_uncertainty=cfg_model.get("uncertainty", True),
    )
    ckpt_path = cfg_model.get("pretrained_checkpoint")
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        missing, unexpected = model.load_pretrained_backbone(state, strict=False)
        print(f"[build_model] loaded pretrained backbone, missing={len(missing)} unexpected={len(unexpected)}")
    return model


if __name__ == "__main__":
    # quick sanity check, runs on CPU/MPS for local dev before moving to Colab
    m = HazeShiftNet(base=16, n_res=1)  # tiny for a fast local smoke test
    x = torch.rand(2, 3, 64, 64)
    out = m(x)
    for k, v in out.items():
        print(k, tuple(v.shape))
