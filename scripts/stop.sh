#!/usr/bin/env bash
# Stop rollout orchestration (and optionally the vLLM servers) without pattern-matching the caller.
#   scripts/stop.sh            # loop + runners
#   scripts/stop.sh --servers  # also vLLM
targets='/bin/python -m arc3rl[.]loop|/bin/inference-taaf-run'
[ "${1:-}" = "--servers" ] && targets="$targets|/bin/vllm serve"
pids=$(ps -u "$USER" -o pid=,args= | awk -v me=$$ -v re="$targets" '$1 != me && $2 ~ /python|vllm/ && $0 ~ re {print $1}')
[ -z "$pids" ] && { echo "nothing to stop"; exit 0; }
kill $pids 2>/dev/null; sleep 8
left=$(ps -o pid= -p $(echo $pids | tr ' ' ',') 2>/dev/null)
[ -n "$left" ] && kill -9 $left 2>/dev/null
echo "stopped: $pids"
