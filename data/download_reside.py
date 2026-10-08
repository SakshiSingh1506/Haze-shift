"""
RESIDE download helper -- uses verified Kaggle mirrors (confirmed working via
`kaggle` CLI + API token, 2026-10-07). The official RESIDE site only links to
Box.com shared folders, which require an interactive browser click (no public
API without a Box OAuth app), so these Kaggle mirrors are what this project
actually downloads from.

Verified subsets and exact counts (confirmed by listing the real downloaded
files, not assumed from documentation):

    ITS   (balraj98/indoor-training-set-its-residestandard)
          -> hazy/ (13,990 files, 'id_idx_beta.png'), clear/ (1,399 files, 'id.png')
    SOTS  (balraj98/synthetic-objective-testing-set-sots-reside)
          -> indoor/{hazy,clear}  (500 / 50 files; hazy 'id_idx.png' -> clear 'id.png')
          -> outdoor/{hazy,clear} (500 / 492 files; hazy 'id_beta_A.jpg' -> clear 'id.png')
    RTTS  (tuncnguyn/rtts-dataset)
          -> RTTS/JPEGImages/ (4,322 files, unlabeled, matches official RESIDE RTTS count)
    RESIDE-6K (kmljts/reside-6k) -- used as protocol B's outdoor training set
    (OTS6K) instead of the full 313,950-image / ~45GB OTS, since it's a
    well-established curated outdoor subset and tractable on one Colab GPU:
          -> train/{hazy,clear} (6,000 / 6,000 files, matching filenames 'id.jpg')
          -> test/{hazy,clear}  (1,000 / 1,000 files, matching filenames 'id_beta_A.jpg')

Requires `pip install kaggle` and a Kaggle API token. Either the new-style
token at ~/.kaggle/access_token (just the token string, e.g. 'KGAT_...'), or
the legacy ~/.kaggle/kaggle.json ({"username":..., "key":...}).

Usage:
    python data/download_reside.py --subset its
    python data/download_reside.py --subset sots
    python data/download_reside.py --subset rtts
    python data/download_reside.py --subset ots6k
    python data/download_reside.py --subset all
"""
import argparse
import shutil
import subprocess
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "RESIDE"
RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "RESIDE_raw"

KAGGLE_SLUGS = {
    "its": "balraj98/indoor-training-set-its-residestandard",
    "sots": "balraj98/synthetic-objective-testing-set-sots-reside",
    "rtts": "tuncnguyn/rtts-dataset",
    "ots6k": "kmljts/reside-6k",
}


def kaggle_download(slug, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(target_dir), "--unzip"],
        check=True,
    )


def move_if_missing(src, dst):
    if dst.exists():
        print(f"[download_reside] {dst} already exists, skipping move")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def get_its():
    raw = RAW_ROOT / "ITS"
    if not (raw / "hazy").exists() and not (DATA_ROOT / "ITS" / "hazy").exists():
        kaggle_download(KAGGLE_SLUGS["its"], raw)
    move_if_missing(raw, DATA_ROOT / "ITS")


def get_sots():
    raw = RAW_ROOT / "SOTS"
    if not (DATA_ROOT / "SOTS" / "indoor").exists():
        kaggle_download(KAGGLE_SLUGS["sots"], raw)
        move_if_missing(raw, DATA_ROOT / "SOTS")


def get_rtts():
    raw = RAW_ROOT / "RTTS"
    if not (DATA_ROOT / "RTTS").exists():
        kaggle_download(KAGGLE_SLUGS["rtts"], raw)
        # Kaggle mirror nests an extra RTTS/ level; JPEGImages holds the actual images
        nested = raw / "RTTS" / "JPEGImages"
        move_if_missing(nested, DATA_ROOT / "RTTS")


def get_ots6k():
    raw = RAW_ROOT / "RESIDE-6K"
    out_dir = DATA_ROOT / "OTS6K" / "train"
    if not out_dir.exists():
        kaggle_download(KAGGLE_SLUGS["ots6k"], raw)
        # Kaggle mirror nests an extra RESIDE-6K/ level
        nested = raw / "RESIDE-6K" / "train"
        out_dir.mkdir(parents=True, exist_ok=True)
        move_if_missing(nested / "hazy", out_dir / "hazy")
        move_if_missing(nested / "GT", out_dir / "clear")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["its", "sots", "rtts", "ots6k", "all"])
    args = parser.parse_args()

    fns = {"its": get_its, "sots": get_sots, "rtts": get_rtts, "ots6k": get_ots6k}
    todo = fns.values() if args.subset == "all" else [fns[args.subset]]
    for fn in todo:
        fn()
    print("[download_reside] done")


if __name__ == "__main__":
    main()
