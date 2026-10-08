"""
Dataset classes for HazeShift-X.

Expected directory layout after running data/download_*.py:

  data/RESIDE/ITS/{hazy,clear}/*.png                 (indoor synthetic pairs)
  data/RESIDE/OTS/{hazy,clear}/*.jpg                 (outdoor synthetic pairs, subsampled)
  data/RESIDE/SOTS/indoor/{hazy,clear}/*.png
  data/RESIDE/SOTS/outdoor/{hazy,clear}/*.png
  data/RESIDE/RTTS/*.png                             (real, unlabeled, no GT)
  data/NH-HAZE/{hazy,clear}/*.png                    (55 pairs; split file selects the official 5-image test set)
  data/NH-HAZE/official_test_list.txt                (filenames of the 5 NTIRE-2020 official test pairs)

All images are read as RGB, cropped/resized, and returned as float tensors in [0,1].
"""
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


def _list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])


def _load_rgb(path):
    return Image.open(path).convert("RGB")


class PairedHazeDataset(Dataset):
    """Generic paired (hazy, clean) dataset for ITS / OTS / SOTS / NH-HAZE.

    Assumes `hazy_dir` and `clean_dir` contain files that match 1:1 either by
    identical filename, or by RESIDE's `<id>_<idx>_<beta>.png` hazy naming
    convention against `<id>.png` clean naming (handled by `clean_key_fn`).
    """

    def __init__(self, hazy_dir, clean_dir, crop_size=256, train=True,
                 clean_key_fn=None, file_list=None):
        self.hazy_dir = Path(hazy_dir)
        self.clean_dir = Path(clean_dir)
        self.crop_size = crop_size
        self.train = train
        self.clean_key_fn = clean_key_fn or (lambda name: name)

        if file_list is not None:
            self.hazy_files = [self.hazy_dir / f for f in file_list]
        else:
            self.hazy_files = _list_images(self.hazy_dir)

    def __len__(self):
        return len(self.hazy_files)

    def _clean_path_for(self, hazy_path):
        key = self.clean_key_fn(hazy_path.name)
        candidate = self.clean_dir / key
        if candidate.exists():
            return candidate
        # fall back: try matching by stem with any known extension
        for ext in (".png", ".jpg", ".jpeg"):
            c = self.clean_dir / (Path(key).stem + ext)
            if c.exists():
                return c
        raise FileNotFoundError(f"No matching clean image for {hazy_path} (looked for {candidate})")

    def __getitem__(self, idx):
        hazy_path = self.hazy_files[idx]
        clean_path = self._clean_path_for(hazy_path)

        hazy = _load_rgb(hazy_path)
        clean = _load_rgb(clean_path)

        if self.train:
            hazy, clean = self._paired_random_crop_flip(hazy, clean)
        else:
            hazy, clean = self._center_crop_or_resize(hazy, clean)

        return TF.to_tensor(hazy), TF.to_tensor(clean), hazy_path.name

    def _paired_random_crop_flip(self, hazy, clean):
        w, h = hazy.size
        cs = self.crop_size
        if w < cs or h < cs:
            hazy = hazy.resize((max(w, cs), max(h, cs)), Image.BICUBIC)
            clean = clean.resize((max(w, cs), max(h, cs)), Image.BICUBIC)
            w, h = hazy.size
        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)
        hazy = hazy.crop((x, y, x + cs, y + cs))
        clean = clean.crop((x, y, x + cs, y + cs))
        if random.random() < 0.5:
            hazy = TF.hflip(hazy)
            clean = TF.hflip(clean)
        return hazy, clean

    def _center_crop_or_resize(self, hazy, clean):
        # Eval: resize to a multiple of 16 on the shorter side, keep full image
        # (no random crop) so metrics are computed on the full test image.
        w, h = hazy.size
        new_w, new_h = (w // 16) * 16, (h // 16) * 16
        new_w, new_h = max(new_w, 16), max(new_h, 16)
        hazy = hazy.resize((new_w, new_h), Image.BICUBIC)
        clean = clean.resize((new_w, new_h), Image.BICUBIC)
        return hazy, clean


def reside_its_clean_key(hazy_filename):
    """RESIDE ITS hazy files are named like '1_1_0.90179.png' -> clean is '1.png'."""
    return hazy_filename.split("_")[0] + ".png"


def reside_ots_clean_key(hazy_filename):
    """RESIDE OTS hazy files are named like '0001_0.8_0.2.jpg' -> clean is '0001.jpg'."""
    return hazy_filename.split("_")[0] + ".jpg"


def ots6k_clean_key(hazy_filename):
    """RESIDE-6K (Kaggle mirror kmljts/reside-6k) train split: hazy and clean share
    the exact same filename (e.g. '1.jpg' -> '1.jpg'), unlike the full OTS naming above.
    Verified against the actual downloaded archive, 2026-10-07."""
    return hazy_filename


def sots_clean_key(hazy_filename):
    """SOTS hazy/clean files share the same stem, e.g. '1400_10.png' -> '1400.png'."""
    return hazy_filename.split("_")[0] + ".png"


class UnlabeledHazeDataset(Dataset):
    """RTTS (or any unlabeled real hazy folder): no ground truth, used in stage 2."""

    def __init__(self, hazy_dir, crop_size=256, train=True):
        self.files = _list_images(hazy_dir)
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = _load_rgb(path)
        w, h = img.size
        cs = self.crop_size
        if self.train:
            if w < cs or h < cs:
                img = img.resize((max(w, cs), max(h, cs)), Image.BICUBIC)
                w, h = img.size
            x = random.randint(0, w - cs)
            y = random.randint(0, h - cs)
            img = img.crop((x, y, x + cs, y + cs))
            if random.random() < 0.5:
                img = TF.hflip(img)
        else:
            new_w, new_h = max((w // 16) * 16, 16), max((h // 16) * 16, 16)
            img = img.resize((new_w, new_h), Image.BICUBIC)
        return TF.to_tensor(img), path.name


def load_nh_haze_official_test(nh_haze_root):
    """Returns a PairedHazeDataset restricted to the 5 official NTIRE-2020 test pairs.

    `nh_haze_root/official_test_list.txt` must list the 5 hazy filenames, one
    per line (fetched/verified by data/download_nh_haze.py against the
    official NTIRE split -- do NOT guess this list, it determines the
    zero-shot test set).
    """
    root = Path(nh_haze_root)
    list_file = root / "official_test_list.txt"
    if not list_file.exists():
        raise FileNotFoundError(
            f"{list_file} missing. Run data/download_nh_haze.py first to fetch "
            "NH-HAZE and write the verified official 5-image test list."
        )
    file_list = [l.strip() for l in list_file.read_text().splitlines() if l.strip()]
    return PairedHazeDataset(
        hazy_dir=root / "hazy",
        clean_dir=root / "clear",
        crop_size=None,
        train=False,
        clean_key_fn=lambda name: name,  # NH-HAZE hazy/clear share filenames
        file_list=file_list,
    )


def load_full_nh_haze(nh_haze_root):
    """All 55 pairs, used only for the secondary no-reference NIQE stat, never for training."""
    root = Path(nh_haze_root)
    return PairedHazeDataset(
        hazy_dir=root / "hazy",
        clean_dir=root / "clear",
        crop_size=None,
        train=False,
        clean_key_fn=lambda name: name,
    )
