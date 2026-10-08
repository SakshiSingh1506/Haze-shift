#!/usr/bin/env bash
# HazeShift-X: full experiment run on an H200 (or any Linux CUDA GPU server).
#
#   bash run_h200.sh                 # setup + data + full plan (core results, then ablations), in the background
#   bash run_h200.sh --core-only     # skip the ablations
#   bash run_h200.sh --foreground    # run in this terminal instead of in the background
#   bash run_h200.sh --setup-only    # environment + data only, no training
#   bash run_h200.sh --smoke         # ~2-minute end-to-end test on tiny budgets (recommended first)
#
# Needs: Linux, NVIDIA driver, python3 (3.10+), internet, ~20 GB free disk,
# and a Kaggle API token, given one of these ways:
#   export KAGGLE_API_TOKEN=KGAT_...          (written to ~/.kaggle/access_token), or
#   ~/.kaggle/access_token already present, or
#   ~/.kaggle/kaggle.json (legacy username/key)
#
# Safe to re-run: datasets that already exist are skipped, finished stages are
# skipped, and an interrupted training stage resumes from its last epoch.
# Outputs: checkpoints/h200/, results/h200/, logs/h200_pipeline.log
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
CORE_ONLY=""; FOREGROUND=0; SETUP_ONLY=0; SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --core-only)  CORE_ONLY="--core_only" ;;
    --foreground) FOREGROUND=1 ;;
    --setup-only) SETUP_ONLY=1 ;;
    --smoke)      SMOKE=1 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

say() { echo -e "\n== $*"; }

# ---------------------------------------------------------------- 1. checks
say "checking GPU and python"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver on this machine"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "need python >= 3.10 (set PYTHON=/path/to/python3.x)"; exit 1; }

# ---------------------------------------------------------------- 2. python env
if [ ! -d .venv ]; then
  say "creating virtualenv .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
say "installing packages (PyTorch with CUDA, then project requirements)"
pip install -q --upgrade pip
pip install -q torch torchvision           # Linux x86_64 wheels from PyPI ship with CUDA
pip install -q -r requirements.txt
python - << 'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU -- check the driver / CUDA wheel"
print(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)} | bf16 {torch.cuda.is_bf16_supported()}")
EOF

# ---------------------------------------------------------------- 3. kaggle token
mkdir -p ~/.kaggle
if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
  printf '%s' "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
fi
if [ ! -f ~/.kaggle/access_token ] && [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "No Kaggle token. Run:  export KAGGLE_API_TOKEN=KGAT_...   then re-run this script."
  exit 1
fi
chmod 600 ~/.kaggle/access_token ~/.kaggle/kaggle.json 2>/dev/null || true

# ---------------------------------------------------------------- 4. data
say "downloading datasets (skips anything already present)"
python data/download_reside.py --subset all
python data/download_nh_haze.py
python scripts/verify_data.py

# ---------------------------------------------------------------- 5. match workers to CPUs
CORES=$(nproc)
WORKERS=$(( CORES > 16 ? 16 : CORES - 1 ))
WORKERS=$(( WORKERS < 1 ? 1 : WORKERS ))
sed -i "s/^  num_workers: [0-9]*/  num_workers: $WORKERS/" configs/h200/protocol_a.yaml configs/h200/protocol_b.yaml
echo "data loader workers: $WORKERS (machine has $CORES CPU cores)"

if [ "$SETUP_ONLY" = 1 ]; then
  say "setup done (--setup-only); start training with: bash run_h200.sh"
  exit 0
fi

if [ "$SMOKE" = 1 ]; then
  say "smoke test: full chain on tiny budgets (configs/dryrun_h200, outputs in checkpoints/dryrun_h200)"
  rm -rf checkpoints/dryrun_h200 results/dryrun_h200
  python scripts/run_pipeline.py --config_dir configs/dryrun_h200 --core_only
  rm -f logs/PIPELINE_DONE
  say "smoke test passed -- start the real run with: bash run_h200.sh"
  exit 0
fi

# ---------------------------------------------------------------- 6. run
mkdir -p logs
LOG="$ROOT/logs/h200_pipeline.log"
CMD="python scripts/run_pipeline.py --config_dir configs/h200 $CORE_ONLY"

if pgrep -f "run_pipeline.py --config_dir configs/h200" >/dev/null; then
  say "pipeline already running (pid $(pgrep -f 'run_pipeline.py --config_dir configs/h200' | head -1)); follow it with: tail -f $LOG"
  exit 0
fi

if [ "$FOREGROUND" = 1 ]; then
  say "running in foreground; log also at $LOG"
  $CMD 2>&1 | tee -a "$LOG"
else
  nohup $CMD >> "$LOG" 2>&1 &
  say "pipeline started in the background (pid $!); it keeps running if you disconnect"
  echo "  follow:   tail -f $LOG"
  echo "  epochs:   tail -f checkpoints/h200/protocol_a/train_log.txt"
  echo "  results:  results/h200/protocol_a/metrics.csv, results/h200/protocol_b/metrics.csv"
fi
