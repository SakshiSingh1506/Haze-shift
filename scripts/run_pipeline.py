"""
Runs the whole experiment plan in priority order, skipping anything already
finished. Safe to re-run after a Colab disconnect: training stages resume
from their last epoch checkpoint, finished stages are skipped via *_done
markers, and evals are skipped if their tag is already in metrics.csv.

Order (core results first, ablations last, so a time-out still leaves the
main comparison complete):
  1. Protocol A: stage 1 -> stage 2 -> eval (DCP, stage1-only, stage2)
  2. Protocol B: stage 1 -> stage 2 -> eval (DCP, stage1-only, stage2)
  3. Ablations per protocol: stage 2 without uncertainty weighting,
     stage 2 without EMA teacher -> eval

Launch detached on the VM (see scripts/colab_launch.sh):
    nohup python scripts/run_pipeline.py > logs/pipeline.log 2>&1 &
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(ROOT / "src"))


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [pipeline] {msg}", flush=True)


def run(args):
    log("RUN " + " ".join(args))
    t = time.time()
    subprocess.run([PY] + args, cwd=ROOT, check=True)
    log(f"OK ({(time.time() - t) / 60:.1f} min)")


def evaluated(results_dir, tag):
    f = ROOT / results_dir / "metrics.csv"
    if not f.exists():
        return False
    with open(f) as fh:
        return any(row["tag"] == tag for row in csv.DictReader(fh))


def protocol_steps(p, config_dir):
    from utils import load_config
    cfg = f"{config_dir}/protocol_{p.lower()}.yaml"
    paths = load_config(ROOT / cfg)["paths"]
    ck = Path(paths["checkpoints"])
    res = paths["results"]
    core = [
        (ck / "stage1_done", ["src/train_stage1.py", "--config", cfg]),
        (ck / "stage2_done", ["src/train_stage2.py", "--config", cfg]),
        (("eval", res, f"{p}_DCP"), ["src/eval.py", "--config", cfg, "--dcp", "--tag", f"{p}_DCP"]),
        (("eval", res, f"{p}_stage1_only"), ["src/eval.py", "--config", cfg, "--checkpoint",
                                             str(ck / "stage1_best.pt"), "--tag", f"{p}_stage1_only"]),
        (("eval", res, f"{p}_stage2"), ["src/eval.py", "--config", cfg, "--checkpoint",
                                        str(ck / "stage2_last.pt"), "--tag", f"{p}_stage2"]),
    ]
    ablations = [
        (ck / "stage2_nounc_done", ["src/train_stage2.py", "--config", cfg, "--no_uncertainty"]),
        (("eval", res, f"{p}_stage2_nounc"), ["src/eval.py", "--config", cfg, "--checkpoint",
                                              str(ck / "stage2_nounc_last.pt"), "--tag", f"{p}_stage2_nounc"]),
        (ck / "stage2_noema_done", ["src/train_stage2.py", "--config", cfg, "--no_ema"]),
        (("eval", res, f"{p}_stage2_noema"), ["src/eval.py", "--config", cfg, "--checkpoint",
                                              str(ck / "stage2_noema_last.pt"), "--tag", f"{p}_stage2_noema"]),
    ]
    return core, ablations


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_dir", default="configs", help="e.g. configs/h200 for the H200 settings")
    ap.add_argument("--core_only", action="store_true", help="skip the ablations")
    a = ap.parse_args()
    core_a, abl_a = protocol_steps("A", a.config_dir)
    core_b, abl_b = protocol_steps("B", a.config_dir)
    plan = core_a + core_b + ([] if a.core_only else abl_a + abl_b)

    for done_check, args in plan:
        if isinstance(done_check, tuple):
            _, res, tag = done_check
            if evaluated(res, tag):
                log(f"skip eval {tag} (already in metrics.csv)")
                continue
        elif (ROOT / done_check).exists():
            log(f"skip {args[0]} {' '.join(args[1:])} ({done_check.name} exists)")
            continue
        run(args)

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "PIPELINE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log("ALL DONE")


if __name__ == "__main__":
    main()
