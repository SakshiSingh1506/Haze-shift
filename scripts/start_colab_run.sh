#!/usr/bin/env bash
# Start (or resume) the full experiment pipeline on a Colab T4, from the Mac.
#
#   bash scripts/start_colab_run.sh
#
# 1. creates Colab session "haze" (T4) if none is running
# 2. uploads code, Kaggle token, and any locally-synced checkpoints/results
# 3. on the VM, runs colab/bootstrap_vm.py (datasets etc.) then
#    scripts/run_pipeline.py, detached with nohup so it keeps going without us
#
# Then keep checkpoints safe on the Mac with:
#   python3 scripts/sync_from_colab.py
set -euo pipefail

cd "$(dirname "$0")/.."
export SSL_CERT_FILE="$(python3 -m certifi)"   # colab-cli websocket SSL fix (python.org Python)
COLAB="${COLAB:-$HOME/Library/Python/3.13/bin/colab}"
S=haze

if ! "$COLAB" status -s "$S" >/dev/null 2>&1 || "$COLAB" status -s "$S" 2>&1 | grep -q "not found"; then
  echo "== creating Colab session $S (T4)"
  "$COLAB" new -s "$S" --gpu T4
fi

echo "== packaging code"
rm -f /tmp/haze_code.zip /tmp/haze_state.zip
zip -rq /tmp/haze_code.zip src configs scripts colab data/*.py requirements.txt
"$COLAB" upload -s "$S" /tmp/haze_code.zip haze_code.zip
"$COLAB" upload -s "$S" "$HOME/.kaggle/access_token" kaggle_access_token

if [ -d checkpoints ] && [ -n "$(ls -A checkpoints 2>/dev/null)" ]; then
  echo "== uploading local checkpoints/results/logs to resume"
  zip -rq /tmp/haze_state.zip checkpoints results logs 2>/dev/null || true
  "$COLAB" upload -s "$S" /tmp/haze_state.zip haze_state.zip
fi

echo "== launching bootstrap + pipeline (detached)"
cat > /tmp/haze_launch.py << 'EOF'
import os, subprocess
os.makedirs('/content/logs', exist_ok=True)
running = subprocess.run(['pgrep', '-f', 'run_pipeline.py'], capture_output=True, text=True).stdout.strip()
if running:
    print('pipeline already running, pid', running)
else:
    import zipfile; zipfile.ZipFile('/haze_code.zip').extractall('/content')
    cmd = 'cd /content && python colab/bootstrap_vm.py && python scripts/run_pipeline.py'
    with open('/content/logs/pipeline.log', 'a') as log:
        p = subprocess.Popen(['bash', '-c', cmd], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('launched pid', p.pid)
EOF
"$COLAB" exec -s "$S" -f /tmp/haze_launch.py
echo "== done. Follow progress with: python3 scripts/sync_from_colab.py --once && tail logs/pipeline.log"
