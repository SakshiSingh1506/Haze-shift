"""
Prepares a fresh Colab VM (runs ON the VM, from /content). Idempotent: each
step is skipped if its output already exists, so it is safe to re-run.

Expects these uploaded by scripts/start_colab_run.sh (colab upload lands
files at the VM root '/'):
    /haze_code.zip           src/, configs/, scripts/, colab/, data/*.py
    /kaggle_access_token     Kaggle API token
    /haze_state.zip          (optional) checkpoints/, results/, logs/ synced
                             back from the Mac, to resume after a disconnect
"""
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/content")
DATA = ROOT / "data"


def log(msg):
    print(f"[bootstrap] {msg}", flush=True)


def sh(cmd):
    subprocess.run(cmd, check=True)


def main():
    os.chdir(ROOT)

    # 1. code
    with zipfile.ZipFile("/haze_code.zip") as z:
        z.extractall(ROOT)
    log("code extracted")

    # 2. previous run state (checkpoints/results/logs), if provided
    if Path("/haze_state.zip").exists() and not (ROOT / "checkpoints").exists():
        with zipfile.ZipFile("/haze_state.zip") as z:
            z.extractall(ROOT)
        log("restored checkpoints/results/logs from /haze_state.zip")
    (ROOT / "logs").mkdir(exist_ok=True)

    # 3. packages
    sh([sys.executable, "-m", "pip", "install", "-q", "kaggle", "lpips", "pyiqa"])
    log("packages installed")

    # 4. kaggle token
    kdir = Path("/root/.kaggle")
    kdir.mkdir(exist_ok=True)
    shutil.copy("/kaggle_access_token", kdir / "access_token")
    os.chmod(kdir / "access_token", 0o600)

    # 5. datasets (each helper skips what already exists)
    sh([sys.executable, "data/download_reside.py", "--subset", "all"])
    sh([sys.executable, "data/download_nh_haze.py"])

    # 6. sanity counts -- abort if anything is off rather than train on bad data
    sh([sys.executable, "scripts/verify_data.py"])
    log("all dataset counts verified")


if __name__ == "__main__":
    main()
