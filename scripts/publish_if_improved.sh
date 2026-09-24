#!/usr/bin/env bash
# Wait until iteration N's rollouts (made with the adapter trained in iteration N-1) are recorded,
# compare them with iteration 1 (the Terse starting policy), and publish to GitHub only if better:
# not worse on levels/episode and RHAE, and better on at least one.
#   scripts/publish_if_improved.sh [N]   (default 2)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"
N=${1:-2}
H=results/train_history.jsonl
until [ -f "$H" ] && [ "$(wc -l < "$H")" -ge "$N" ]; do sleep 60; done

verdict=$(python3 - "$H" "$N" <<'EOF'
import json, sys
hist = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = int(sys.argv[2])
base, new = hist[0]["rollout"], hist[n - 1]["rollout"]
if hist[n - 1].get("rolled_back"):
    print("REJECT collapse guard rolled back")
    sys.exit()
lv0, lv1 = base["levels_per_episode"], new["levels_per_episode"]
s0, s1 = base["mean_score"], new["mean_score"]
ok = lv1 >= lv0 and s1 >= s0 and (lv1 > lv0 or s1 > s0)
print(f"{'PUBLISH' if ok else 'REJECT'} levels/ep {lv0:.2f}->{lv1:.2f} RHAE {s0:.2f}->{s1:.2f} policy {hist[n - 1]['policy_used_for_rollouts']}")
EOF
)
echo "$(date -u +%FT%TZ) $verdict"
case "$verdict" in PUBLISH*) ;; *) exit 0 ;; esac

# The adapter that produced iteration N's rollouts was trained in iteration N-1.
PREV=$(printf "iter%03d" $((N - 1)))
ADAPTER="$CKPT_ROOT/$PREV"
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "missing $ADAPTER"; exit 1; }

# master: history, report, state snapshot
TS=$(date -u +%Y%m%dT%H%MZ); S=results/snapshots/$TS; mkdir -p "$S"
cp configs/*.json "$S"/; cp "$CKPT_ROOT/state.json" "$S"/
echo "$verdict" > "$S/verdict.txt"
( cd "$CKPT_ROOT" && sha256sum "$PREV/adapter_model.safetensors" ) > "$S/adapter_sha256.txt"
"$SERVE_VENV/bin/python" -m arc3rl.report > /dev/null || true
git add results configs
git commit -qm "Run 3b: $PREV adapter improves on Terse ($verdict)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
git push -q origin master

# adapters branch (Git LFS) via a separate worktree, so the running loop's checkout is untouched
W=$(mktemp -d)/adapters
git fetch -q origin adapters
git worktree add -q "$W" origin/adapters
(
  cd "$W"
  git checkout -q -B adapters origin/adapters
  git lfs install --local > /dev/null
  mkdir -p "run3b-$PREV"
  cp "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" "$ADAPTER/train_stats.json" "run3b-$PREV/"
  sha256sum */adapter_model.safetensors > SHA256SUMS
  git add -A
  git commit -qm "run3b-$PREV adapter ($verdict)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  git push -q origin adapters
)
git worktree remove --force "$W"
echo "$(date -u +%FT%TZ) published $PREV"
