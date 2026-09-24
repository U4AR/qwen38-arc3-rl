#!/usr/bin/env bash
# Quick A/B after an update: pause the loop, play the train games with the new adapter at a small
# token budget, compare token-matched against the previous policy's rollouts, then resume the loop.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"
IT=${1:-1}; BUDGET=${2:-40000}; PASSES=${3:-4}
until python3 -c "import json,sys;sys.exit(0 if json.load(open('$CKPT_ROOT/state.json'))['iteration']>=$IT else 1)" 2>/dev/null; do sleep 15; done
NAME=$(python3 -c "import json;print(json.load(open('$CKPT_ROOT/state.json'))['name'])")
echo "$(date +%H:%M) update saved; policy=$NAME; pausing loop"
scripts/stop.sh
rm -rf "$RUNS_ROOT/train/$(printf iter%03d $((IT + 1)))"
OUT=$RUNS_ROOT/quick/$NAME-$BUDGET; rm -rf "$OUT"
"$SERVE_VENV/bin/python" - <<PY
from pathlib import Path
import json
from arc3rl.rollout import RolloutConfig, run_rollouts
split = json.loads(Path("configs/split.json").read_text())
cfg = RolloutConfig(model="$NAME", servers=["http://127.0.0.1:1234/v1", "http://127.0.0.1:1235/v1"],
                    max_generated_tokens=$BUDGET, concurrent_per_server=12)
run_rollouts(cfg, split["train"], $PASSES, Path("$OUT"))
PY
echo "$(date +%H:%M) quick rollouts done"
"$SERVE_VENV/bin/python" -m arc3rl.quicktest --budget "$BUDGET" \
  "previous=$RUNS_ROOT/train/$(printf iter%03d $IT)" "$NAME=$OUT" | tee "$OUT/comparison.txt"
echo "$(date +%H:%M) resuming loop"
(nohup "$SERVE_VENV/bin/python" -m arc3rl.loop train >> /cache/nvme0/Qwen/logs/train.log 2>&1 &)
