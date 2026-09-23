#!/usr/bin/env bash
# Wait for both baselines, fix the split, write the report, start RL from the Terse LoRA.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"
until [ -f "$RUNS_ROOT/eval/baseline/rollout_done.json" ] && [ -f "$RUNS_ROOT/eval/baseline-terse/rollout_done.json" ] \
      && [ -f results/eval_baseline.json ] && [ -f results/eval_baseline-terse.json ]; do sleep 60; done
"$SERVE_VENV/bin/python" -m arc3rl.split 8
"$SERVE_VENV/bin/python" -m arc3rl.report > /dev/null
git add results configs && git commit -qm "Baselines (base, base+Terse) and train/held-out split

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
exec "$SERVE_VENV/bin/python" -m arc3rl.loop train --init-adapter "$CKPT_ROOT/terse-hf"
