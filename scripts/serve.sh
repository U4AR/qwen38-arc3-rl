#!/usr/bin/env bash
# One BF16 vLLM server per GPU with runtime LoRA loading and sleep mode, so the
# trainer can borrow the GPUs between rollout phases.
#   scripts/serve.sh <gpu> <port>
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU=$1; PORT=$2
export CUDA_VISIBLE_DEVICES=$GPU
# Per-server compile caches: two servers compiling into one cache dir corrupted it.
export VLLM_CACHE_ROOT=/cache/nvme0/vllm-cache/gpu$GPU
export TORCHINDUCTOR_CACHE_DIR=/cache/nvme0/vllm-cache/gpu$GPU/inductor
export TRITON_CACHE_DIR=/cache/nvme0/vllm-cache/gpu$GPU/triton
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1   # /v1/load_lora_adapter
export VLLM_SERVER_DEV_MODE=1               # /sleep, /wake_up
SPEC=${SPEC_TOKENS:-2}
SPEC_ARGS=()
if [ "$SPEC" != "0" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC}")
fi
"$SERVE_VENV/bin/python" "$(dirname "$0")/patch_vllm.py"
exec "$SERVE_VENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8192}" \
  --enable-prefix-caching \
  --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  --enable-lora --max-lora-rank "${MAX_LORA_RANK:-32}" --max-loras 1 \
  --enable-sleep-mode \
  --limit-mm-per-prompt '{"image": 48, "video": 0}' \
  --mm-processor-cache-gb 0 \
  "${SPEC_ARGS[@]}"
