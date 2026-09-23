# Shared environment for every script in this repo. Source it; don't execute.
export REPO=/data/projects/qwen38-arc3-rl
export DUCK=$REPO/vendor/duck-harness/ARC3-Inference
export MODEL_PATH=/cache/nvme0/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export SERVED_NAME=Qwen3.8-27B
export ENVS_DIR=/data/projects/TestARC/environment_files
# Disposable, regenerable: venvs, rollouts, compile caches (NVMe).
export SERVE_VENV=/cache/nvme0/venvs/arc3
export TRAIN_VENV=/cache/nvme0/venvs/train
export RUNS_ROOT=/cache/nvme0/Qwen/runs
# Persistent: LoRA checkpoints (small) and summaries.
export CKPT_ROOT=/data/checkpoints/qwen38-arc3-rl
export HF_HOME=/cache/nvme0/huggingface
export TMPDIR=/cache/nvme1/tmp
export UV_CACHE_DIR=/cache/nvme0/uv-cache
export TRITON_CACHE_DIR=/cache/nvme0/triton-cache
export VLLM_CACHE_ROOT=/cache/nvme0/vllm-cache
mkdir -p "$RUNS_ROOT" "$CKPT_ROOT" "$TMPDIR" /cache/nvme0/Qwen/logs
