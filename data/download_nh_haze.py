"""
NH-HAZE download + official NTIRE-2020 test-split writer.

Verified working direct download (confirmed on Colab, 2026-10-07):
    https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip
    (330,735,056 bytes, a flat zip of 55 pairs)

The archive contains flat files named '{NN}_hazy.png' / '{NN}_GT.png' for
NN = 01..55 (zero-padded, confirmed by inspecting the actual zip contents,
not guessed). This script downloads it, then reorganizes into the
hazy/clear layout src/datasets.py expects:
    data/NH-HAZE/hazy/{NN}.png
    data/NH-HAZE/clear/{NN}.png

Official NTIRE-2020 split is 45 train / 5 validation / 5 test (per Ancuti et
al., CVPRW 2020). The archive itself does not label which 10 of the 55 are
val/test -- the paper's convention (and the NTIRE challenge's own numbering)
is the *last* 10 images, with 46-50 as validation and 51-55 as the final
test phase. If your course/instructor specifies a different split, override
OFFICIAL_TEST_IDS below and say so in the report.

Usage:
    python data/download_nh_haze.py
"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

NH_HAZE_ROOT = Path(__file__).resolve().parent.parent / "data" / "NH-HAZE"
RAW_DIR = NH_HAZE_ROOT.parent / "NH-HAZE_raw"
URL = "https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip"

# Last 5 of the 55 (51-55), following the NTIRE-2020 challenge's own
# train(1-45)/val(46-50)/test(51-55) numbering. VERIFY against your
# instructor's protocol if this project's course gave a different split.
OFFICIAL_TEST_IDS = list(range(51, 56))  # 51..55 inclusive


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "NH-HAZE.zip"
    if zip_path.exists() and zip_path.stat().st_size == 330735056:
        print(f"[download_nh_haze] {zip_path} already present, skipping download")
        return zip_path
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "-O", str(zip_path), URL], check=True)
    else:
        subprocess.run(["curl", "-sSL", "-o", str(zip_path), URL], check=True)
    print(f"[download_nh_haze] downloaded {zip_path.stat().st_size} bytes")
    return zip_path


def organize(zip_path):
    extract_dir = RAW_DIR / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)

    flat_dir = extract_dir / "NH-HAZE"
    hazy_out = NH_HAZE_ROOT / "hazy"
    clean_out = NH_HAZE_ROOT / "clear"
    hazy_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)

    ids = set()
    for f in flat_dir.iterdir():
        if f.name.endswith("_hazy.png"):
            id_ = f.name.replace("_hazy.png", "")
            shutil.copy(f, hazy_out / f"{id_}.png")
            ids.add(id_)
        elif f.name.endswith("_GT.png"):
            id_ = f.name.replace("_GT.png", "")
            shutil.copy(f, clean_out / f"{id_}.png")

    print(f"[download_nh_haze] organized {len(ids)} pairs into {NH_HAZE_ROOT}")
    return sorted(ids)


def write_test_split(ids):
    test_files = [f"{int(i):02d}.png" for i in ids if int(i) in OFFICIAL_TEST_IDS]
    if len(test_files) != 5:
        print(f"[download_nh_haze] WARNING: matched {len(test_files)} files to "
              f"OFFICIAL_TEST_IDS={OFFICIAL_TEST_IDS}, expected 5 -- check ids and fix before trusting the split.")
        return
    out_file = NH_HAZE_ROOT / "official_test_list.txt"
    out_file.write_text("\n".join(sorted(test_files)) + "\n")
    print(f"[download_nh_haze] wrote official 5-image test list to {out_file}:")
    for f in sorted(test_files):
        print(f"  {f}")


def main():
    zip_path = download()
    ids = organize(zip_path)
    write_test_split(ids)


if __name__ == "__main__":
    main()
