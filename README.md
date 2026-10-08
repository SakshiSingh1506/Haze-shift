# HazeShift-X: Domain-Generalized Real Image Dehazing

Course project (CV). Train on synthetic haze (RESIDE), generalize zero-shot to
real, spatially non-uniform haze (NH-HAZE) without ever seeing real
hazy/clean pairs during training.

## Problem

Dehazing networks trained on synthetic data overfit the simple atmospheric
scattering model (ASM): `I = J*t + A*(1-t)`, with globally uniform haze. Real
haze is spatially varying, mixed with illumination effects, and has no large
paired dataset. We need a model that:

1. Learns a **domain-general atmospheric representation** (transmission +
   airlight) rather than memorizing synthetic haze statistics.
2. Expresses **uncertainty** about that representation on out-of-distribution
   (real) input.
3. Uses **physical consistency** (re-synthesizing the hazy image from its own
   prediction) as a self-supervised signal on *unlabeled* real haze, since we
   have no real ground truth to supervise against.

## Method: HazeShiftNet

A shared encoder feeds three heads:

- **Restoration head** → predicts clean image `J_hat` directly (main path,
  supervised on synthetic pairs).
- **Transmission head** → predicts `t_mean`, `t_logvar` per pixel
  (heteroscedastic uncertainty on local haze density).
- **Airlight head** → predicts global `A` (+ small logvar) from pooled
  features.

Physical reconstruction: `I_hat = J_hat * t_mean + A * (1 - t_mean)`.

**Stage 1 (supervised, synthetic only):** L1 + SSIM + VGG perceptual loss on
`J_hat` vs ground truth, plus a physical-consistency loss between `I_hat` and
the synthetic hazy input (teaches `t`/`A` to stay physically meaningful, not
just backed out from a black box).

**Stage 2 (unsupervised, real unlabeled haze — RTTS):** no ground truth
exists, so we fine-tune with:
- **Uncertainty-weighted physical consistency**: reconstruction error between
  `I_hat` and the real hazy input, weighted by predicted variance
  (heteroscedastic loss: `error/exp(logvar) + logvar`) — this is what lets
  the model say "I'm unsure" on real haze patterns it hasn't seen, instead of
  confidently producing a wrong synthetic-style correction.
- **Dark Channel Prior (DCP) loss** and **bright channel prior loss** (from
  PSD, CVPR 2021) on `J_hat`, as a physics-grounded regularizer when no GT
  exists.
- **Transmission smoothness (TV) loss** on `t_mean`.
- An EMA teacher for stabilizing stage-2 updates (common in unsupervised
  domain adaptation; optional, toggled in config).

The network (13.5M parameters) is trained from scratch. Public DehazeFormer /
FFA-Net checkpoints use different architectures, so their weights cannot be
loaded into this encoder. The only pretrained component is the frozen VGG16
used for the perceptual loss.

## Literature this builds on

- Chen et al., **PSD: Principled Synthetic-to-Real Dehazing Guided by
  Physical Priors**, CVPR 2021 — unsupervised real-domain fine-tuning with a
  prior-loss committee (dark/bright channel, CLAHE). We adopt the prior
  losses but replace the fixed physical priors with a *learned, uncertainty-
  weighted* consistency term.
- Wu et al., **RIDCP: Revitalizing Real Image Dehazing via High-Quality
  Codebook Priors**, CVPR 2023 — argues the ASM is too simple for real haze
  and uses a learned high-quality codebook prior instead. We keep the ASM
  but make its parameters uncertainty-aware rather than discarding physics.
- Ancuti et al., **NH-HAZE: An Image Dehazing Benchmark with Non-Homogeneous
  Hazy and Haze-Free Images**, CVPRW/NTIRE 2020 — our zero-shot real test
  set (55 paired outdoor scenes, non-uniform haze generated with a
  professional haze machine so GT exists, unlike most real datasets).
- Variational/Bayesian haze modeling (e.g. "Deep Variational Bayesian
  Modeling of Haze Degradation Process", 2024) — motivates treating
  transmission/airlight as distributions rather than point estimates.

## Two training protocols (controlled comparison)

Everything (architecture, losses, stage-2 real data, eval set) is identical
between protocols — the **only** variable is the synthetic training domain.
This isolates the effect of indoor-vs-outdoor synthetic pretraining on
real-world (outdoor) generalization.

| | Protocol A (lightweight) | Protocol B (literature-standard) |
|---|---|---|
| Stage-1 synthetic train | RESIDE **ITS** (indoor, 13,990 imgs) | RESIDE **OTS** (outdoor, subsampled ~40k imgs) |
| Stage-1 in-domain test | SOTS-**indoor** (500 imgs) | SOTS-**outdoor** (500 imgs) |
| Stage-2 real unlabeled | RTTS (4,322 imgs, shared) | RTTS (identical) |
| Zero-shot real test | NH-HAZE official test split (5 imgs), never trained on | identical |
| Rationale | Fast, fits Colab free T4, harder domain shift (indoor→outdoor) makes a clean DG story | Matches convention in real-haze literature (RIDCP/PSD-style work trains outdoor since real benchmarks are outdoor) |

NH-HAZE's 45 training / 5 validation pairs are **never used** for training —
only the 5 official test pairs are touched, and only at evaluation time, to
support a true zero-shot domain-generalization claim.

## Metrics

- **In-domain** (SOTS-indoor or -outdoor): PSNR ↑, SSIM ↑
- **Zero-shot real** (NH-HAZE official test, 5 pairs, has GT): PSNR ↑, SSIM ↑, LPIPS ↓
- **No-reference** (additionally on full NH-HAZE 55 + RTTS, no GT needed): NIQE ↓
- **Cross-domain drop**: in-domain PSNR − NH-HAZE PSNR (lower drop = better generalization). Reported for both protocols and for a from-scratch baseline (no stage-2) to show what the uncertainty/consistency stage buys you.

Because the official NH-HAZE test split is only 5 images, PSNR/SSIM there
will be noisy — we report it as the primary comparable number (matches your
chosen protocol) but also quote NIQE over the full 55-image NH-HAZE set as a
secondary, statistically steadier check.

## Baselines

- **DCP** (Dark Channel Prior, He et al. 2009) — classical, no training, sets a floor.
- **Supervised-only backbone** (stage 1 only, no stage-2 real fine-tuning) — isolates the contribution of the uncertainty/physical-consistency stage.
- **PSD-style fine-tune** (stage 2 with fixed priors, no uncertainty weighting) — ablation isolating the uncertainty contribution specifically.

## Repo layout

```
data/            dataset download scripts
src/datasets.py  PyTorch Dataset classes for ITS/OTS/SOTS/RTTS/NH-HAZE
src/model.py     HazeShiftNet (encoder + 3 heads)
src/losses.py    reconstruction, SSIM, VGG perceptual, DCP/bright-channel, uncertainty-weighted consistency, TV
src/train_stage1.py   supervised synthetic pretraining
src/train_stage2.py   unsupervised real fine-tuning
src/eval.py      PSNR/SSIM/LPIPS/NIQE + cross-domain-drop report
configs/protocol_a.yaml, protocol_b.yaml
colab/setup_colab.py  one-shot environment + data setup for Colab
```

## Running on Colab

Everything runs on a Colab T4; the Mac only orchestrates and keeps backups.

```bash
# one-time: Kaggle API token at ~/.kaggle/access_token, colab CLI logged in

bash scripts/start_colab_run.sh        # start or resume the whole plan on Colab
python3 scripts/sync_from_colab.py     # every 10 min: copy checkpoints/results/logs to this Mac
tail -f logs/pipeline.log              # (after a sync) follow progress
```

- `colab/bootstrap_vm.py` prepares a fresh VM: code, packages, datasets via
  Kaggle + the ETH NH-HAZE link, then checks every file count.
- `scripts/run_pipeline.py` runs: Protocol A (stage 1 → stage 2 → evals), then
  Protocol B, then the ablations. Finished stages are skipped and training
  resumes from the last epoch, so after a Colab disconnect just run
  `start_colab_run.sh` again. It uploads the Mac's synced checkpoints and
  continues where it stopped.
- Mixed precision (AMP) is on; measured 0.95 s/step at batch 16 on a T4.
  Both protocols get an equal ~10,500-step Stage-1 budget (A: 12 × 874,
  B: 28 × 375), about 2.8 h each.

## Running on an H200 (or any Linux CUDA server)

```bash
export KAGGLE_API_TOKEN=KGAT_...      # your Kaggle API token
bash run_h200.sh --smoke              # ~2 min: checks env, data and the whole chain
bash run_h200.sh                      # full plan in the background (survives logout)
tail -f logs/h200_pipeline.log
```

`run_h200.sh` creates `.venv`, installs PyTorch + requirements, downloads and
verifies all datasets, then runs `scripts/run_pipeline.py --config_dir configs/h200`.
H200 settings (`configs/h200/`): batch 32, bf16 autocast, channels-last,
sqrt-scaled learning rates, equal ~13,100-step Stage-1 budgets (A 30 x 437,
B 70 x 187), Stage 2 15 epochs, full SOTS validation each epoch. Outputs go to
`checkpoints/h200/` and `results/h200/`, separate from the Colab T4 run. Re-run
the script after any interruption; it resumes.

## Experiment matrix (all rows land in `results/protocol_*/metrics.csv`)

| Tag | What it isolates |
|---|---|
| `{A,B}_DCP` | Classical prior, no learning (floor) |
| `{A,B}_stage1_only` | Synthetic supervision only: the domain gap without adaptation |
| `{A,B}_stage2` | Full method (uncertainty-weighted consistency + priors + EMA teacher) |
| `{A,B}_stage2_nounc` | Same, but plain consistency (PSD-style): the value of the uncertainty weighting |
| `{A,B}_stage2_noema` | Same, without EMA teacher: the value of the stabilizer |

Per-image NH-HAZE scores are in `results/protocol_*/per_image_<tag>.csv` for
the failure analysis (5 test images, so report mean ± std).
