"""
Copies checkpoints/, results/ and logs/ from the Colab VM to this Mac, so a
dropped Colab session (whose disk is wiped) costs at most one epoch.
Only files whose size or modification time changed are downloaded.

    python3 scripts/sync_from_colab.py            # loop every 10 min until PIPELINE_DONE
    python3 scripts/sync_from_colab.py --once     # single sync
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLAB = os.environ.get("COLAB", str(Path.home() / "Library/Python/3.13/bin/colab"))
SESSION = "haze"
STATE_FILE = ROOT / ".sync_state.json"
REMOTE_DIRS = ["checkpoints", "results", "logs"]

MANIFEST_CODE = """
import os, json
out = {}
for d in %r:
    base = os.path.join('/content', d)
    for root, _, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out[os.path.relpath(p, '/content')] = [st.st_size, int(st.st_mtime)]
print('MANIFEST' + json.dumps(out))
""" % (REMOTE_DIRS,)


def env():
    e = dict(os.environ)
    e["SSL_CERT_FILE"] = subprocess.run([sys.executable, "-m", "certifi"], capture_output=True,
                                        text=True).stdout.strip()
    return e


def remote_manifest():
    r = subprocess.run([COLAB, "exec", "-s", SESSION], input=MANIFEST_CODE, capture_output=True,
                       text=True, env=env(), timeout=120)
    for line in r.stdout.splitlines():
        if line.startswith("MANIFEST"):
            return json.loads(line[len("MANIFEST"):])
    raise RuntimeError(f"could not read remote manifest:\n{r.stdout[-500:]}\n{r.stderr[-500:]}")


def sync_once():
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    manifest = remote_manifest()
    changed = [p for p, meta in manifest.items() if state.get(p) != meta]
    # small files (logs, csv, markers) first so progress is visible even if a big download fails
    changed.sort(key=lambda p: manifest[p][0])
    for rel in changed:
        local = ROOT / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        r = subprocess.run([COLAB, "download", "-s", SESSION, f"/content/{rel}", str(tmp)],
                           capture_output=True, text=True, env=env(), timeout=1800)
        if r.returncode != 0 or not tmp.exists():
            print(f"[sync] FAILED {rel}: {r.stderr[-300:]}")
            continue
        tmp.replace(local)  # atomic: never leave a half-written checkpoint
        state[rel] = manifest[rel]
        STATE_FILE.write_text(json.dumps(state, indent=1))
        print(f"[sync] {rel} ({manifest[rel][0] / 1e6:.1f} MB)")
    print(f"[sync] {time.strftime('%H:%M:%S')} up to date ({len(changed)} file(s) updated)")
    return "logs/PIPELINE_DONE" in manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=600, help="seconds between syncs")
    args = ap.parse_args()
    while True:
        try:
            done = sync_once()
        except Exception as e:  # session gone / network blip: keep what we have, retry later
            print(f"[sync] {time.strftime('%H:%M:%S')} error: {e}")
            done = False
        if args.once or done:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
