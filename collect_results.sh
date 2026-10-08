#!/usr/bin/env bash
# Runs the extra analysis on the finished H200 run and packs everything needed
# for the write-up into ONE file: results_bundle_h200.tar.gz (~30 MB).
#
#   bash collect_results.sh
#
# 1. "hazy input" reference rows (no dehazing) in both metrics.csv files
# 2. comparison images + uncertainty maps for NH-HAZE 51-55 and 8 RTTS images
# 3. bundles: metrics, per-image scores, comparison grids, training logs,
#    pipeline logs, the exact configs used, and environment info.
#    Model weights (.pt) are NOT included (too big, not needed).
set -uo pipefail

cd "$(dirname "$0")"
[ -f .venv/bin/activate ] || { echo "No .venv here -- run this from the folder where run_h200.sh ran."; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate

say() { echo -e "\n== $*"; }

for P in a b; do
  CFG="configs/h200/protocol_$P.yaml"
  TAG="$(echo "$P" | tr a-z A-Z)_hazy_input"
  CSV="results/h200/protocol_$P/metrics.csv"
  if [ ! -f "$CSV" ]; then
    echo "skip protocol $P: $CSV not found"; continue
  fi
  if grep -q "^$TAG," "$CSV"; then
    echo "$TAG already in $CSV"
  else
    say "protocol $P: scoring the hazy input itself (no-dehazing reference)"
    python src/eval.py --config "$CFG" --identity --tag "$TAG" || echo "WARNING: $TAG eval failed (continuing)"
  fi
  say "protocol $P: saving comparison images and uncertainty maps"
  python src/save_outputs.py --config "$CFG" --rtts 8 || echo "WARNING: save_outputs failed for protocol $P (continuing)"
done

say "collecting environment info"
mkdir -p results/h200
{
  echo "date: $(date)"; echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  python -c "import torch,sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  echo "cpu cores: $(nproc)"
  pip freeze 2>/dev/null
} > results/h200/environment.txt 2>&1

say "packing"
OUT="results_bundle_h200.tar.gz"
FILES=(results/h200/environment.txt configs/h200)
for f in results/h200/protocol_*/metrics.csv results/h200/protocol_*/per_image_*.csv \
         results/h200/protocol_*/visuals/*_grid.jpg \
         checkpoints/h200/protocol_*/train_log.txt logs/h200_pipeline*.log; do
  [ -e "$f" ] && FILES+=("$f")
done
tar -czf "$OUT" "${FILES[@]}"

echo
echo "Done: $(pwd)/$OUT  ($(du -h "$OUT" | cut -f1), ${#FILES[@]} entries)"
echo "Send this one file back."
