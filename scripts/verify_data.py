"""Checks every dataset folder has the expected number of files; exits non-zero otherwise.

    python scripts/verify_data.py            # checks ./data
"""
import os
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
EXPECTED = {
    "RESIDE/ITS/hazy": 13990, "RESIDE/ITS/clear": 1399,
    "RESIDE/SOTS/indoor/hazy": 500, "RESIDE/SOTS/outdoor/hazy": 500,
    "RESIDE/RTTS": 4322, "RESIDE/OTS6K/train/hazy": 6000, "RESIDE/OTS6K/train/clear": 6000,
    "NH-HAZE/hazy": 55, "NH-HAZE/clear": 55,
}


def main():
    bad = []
    for rel, n in EXPECTED.items():
        p = DATA / rel
        got = len(os.listdir(p)) if p.exists() else 0
        status = "ok" if got == n else "MISMATCH"
        print(f"[verify_data] {rel:28s} {got:6d} / {n:6d}  {status}")
        if got != n:
            bad.append(rel)
    if not (DATA / "NH-HAZE" / "official_test_list.txt").exists():
        bad.append("NH-HAZE/official_test_list.txt")
    if bad:
        sys.exit(f"[verify_data] FAILED: {bad}")
    print("[verify_data] all datasets verified")


if __name__ == "__main__":
    main()
