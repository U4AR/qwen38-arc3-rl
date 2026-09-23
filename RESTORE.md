# Restoring this RL run on a 2× H100 machine

This repo is the complete code and state of the run. Large artifacts are not in git:
- **Base model weights** come from Hugging Face.
- **LoRA adapters** are GitHub release assets.
- **Rollout trajectories** (tens of GB) are regenerable and not backed up.

## 1. Environment

Tested with: 2× H100 NVL 94 GB, NVIDIA driver 575 (CUDA 12.9), Python 3.12.12, `uv`.

```bash
git clone https://github.com/U4AR/qwen38-arc3-rl.git && cd qwen38-arc3-rl
# Edit scripts/env.sh paths (REPO, MODEL_PATH, ENVS_DIR, RUNS_ROOT, CKPT_ROOT, venvs) for the new machine.
source scripts/env.sh

# Serving + harness venv (vLLM 0.17.2rc1 pinned by the Duck's uv.lock)
(cd vendor/duck-harness/ARC3-Inference && UV_PROJECT_ENVIRONMENT=$SERVE_VENV uv sync --locked --extra server --extra dev)

# Training venv. Everything is cu128 to match the driver; causal-conv1d must build without isolation.
uv venv -p 3.12.12 $TRAIN_VENV
VIRTUAL_ENV=$TRAIN_VENV uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" transformers==5.17.0 peft==0.21.0 accelerate \
  flash-linear-attention==0.5.2 tilelang safetensors pillow numpy ninja packaging setuptools wheel
CAUSAL_CONV1D_FORCE_BUILD=FALSE TORCH_CUDA_ARCH_LIST=9.0 VIRTUAL_ENV=$TRAIN_VENV uv pip install --no-build-isolation causal-conv1d
```

**Gotcha:** always run the trainer with `LD_LIBRARY_PATH=`. System CUDA libraries on the path break cuBLAS in the training venv. `loop.py` already does this.

## 2. Model, games and adapters

```bash
hf download Qwen/Qwen3.8-27B            # BF16, ~52 GB -> set MODEL_PATH to the snapshot dir
# ARC-AGI-3 offline environment files (25 public games, layout <game>/<version>/{<game>.py,metadata.json}) -> ENVS_DIR
mkdir -p $CKPT_ROOT/terse-hf
gh release download state-20260923 -R U4AR/qwen38-arc3-rl -p 'terse-hf*' -D /tmp/rel
```

- **`terse-hf`** is `Shockem/Qwen3.8-27b-Terse-Coder-LoRA` with keys renamed to HF module paths (`model.language_model.layers.*`). It is the RL starting policy. Unpack its release asset into `$CKPT_ROOT/terse-hf/` (files `adapter_config.json`, `adapter_model.safetensors`).
- **`run1-noimage-iter001`** is the only trained adapter so far. It came from run 1 (no per-turn board images, 60k budget, flat reward), which is archived and superseded.
- Checksums are in `results/snapshots/*/adapter_sha256.txt`.

## 3. Resume training

State at backup: **run 2, iteration 1, rollout phase in progress** (see `results/snapshots/<latest>/`). No run-2 adapter exists yet, so resuming means starting run 2 from Terse:

```bash
scripts/serve.sh 0 1234 &  scripts/serve.sh 1 1235 &     # wait for /health on both
cp results/snapshots/<latest>/loop_overrides.json configs/  # lr 1e-5
python -m arc3rl.loop train --init-adapter $CKPT_ROOT/terse-hf   # resumable via $CKPT_ROOT/state.json
```

`configs/split.json` fixes the train and held-out games. Once `$CKPT_ROOT/state.json` exists, the loop resumes from it; do not pass `--init-adapter` again.

Operational notes are in README "Stability notes". The main ones:
- one torch.compile cache per server,
- 12 episodes per server,
- `--gpu-memory-utilization 0.90`,
- `--mm-processor-cache-gb 0`,
- stop processes with `scripts/stop.sh`, never with a `pkill -f` pattern that can match your own shell.
