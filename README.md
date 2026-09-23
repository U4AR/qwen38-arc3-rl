# Qwen3.8-27B × Duck harness: RL for ARC-AGI-3

LoRA reinforcement learning of **Qwen3.8-27B** (BF16) playing **ARC-AGI-3** games
through Tufa Labs' **Duck** harness, on one machine with 2× H100 NVL.

Reward: **solving levels**, plus a bonus for solving them with **fewer generated tokens**.
The model trains on a subset of the 25 public games; the rest are held out.

## Starting policy: Terse-Coder LoRA

RL starts from [`Shockem/Qwen3.8-27b-Terse-Coder-LoRA`](https://huggingface.co/Shockem/Qwen3.8-27b-Terse-Coder-LoRA): a rank-16 DPO LoRA that shortens Qwen3.8's chain of thought on coding tasks. It targets attention, MLP and the GatedDeltaNet linear-attention projections.
- **Key rename:** its keys use vLLM-style module paths (`language_model.model.layers.*`), so `terse-hf/` is a copy renamed to HF paths (`model.language_model.layers.*`). Without this, PEFT would silently leave the adapter unapplied.
- **Verified identical in vLLM and HF:** on sampled tokens, vLLM+LoRA matches HF+PEFT to 0.002 nats per token (0.011 without the adapter), with r = 0.95 correlation of the adapter's effect.

Two baselines are measured on all 25 games: the plain base model and base + Terse.

## What changed in the Duck (`vendor/duck-harness`, upstream commit 7652836)

| Change | Why |
| --- | --- |
| `watch_video` tool (`inference/agent/video_tool.py`), `VIDEO_TOOL=1` | Optional: when the model is *totally unclear* what to do, it can request a filmstrip image of **every rendered frame** of its last N actions, including the intermediate animation frames the Duck normally drops. Per-turn grid images are **off**; vision is opt-in only. |
| RL trajectory log (`inference/agent/rl_log.py`), `RL_TRAJECTORY_LOG=1` | Per request: exact vLLM `prompt_token_ids`, sampled `token_ids` and their logprobs, and the level on screen. |
| Verbatim replayed history | Upstream stripped blank lines from the model's reasoning before replaying it. Keeping it verbatim means each request's prompt extends the previous one token-for-token. This improves vLLM prefix-cache hits and lets the trainer pack a whole turn into one sequence (~4.7× fewer forward passes). |
| Per-episode generated-token budget, `EPISODE_MAX_GENERATED_TOKENS` | Load-independent episode limit, so baseline vs. trained comparisons don't depend on server congestion. Wall-clock time is only a safety cap. |
| Image-aware context-length estimate | Base64 data-URLs were counted as text (~40k "tokens" per image). |

## Method (`arc3rl/`)

1. **Rollouts** (`rollout.py`): one BF16 vLLM server per GPU (MTP speculative decoding, LoRA hot-loading, sleep mode). The current policy plays every training game `P` times.
2. **Reward and credit** (`reward.py`): each request is credited to the level on screen when it was issued. A level segment earns
   `r = solved · (1 + α · max(0, 1 − tokens/T_ref))`.
   Advantages are GRPO-style (Dr.GRPO, mean-centred only) within the group of all segments of the same *(game, level)*. They all start from the same layout, so they are directly comparable.
3. **Packing** (`data.py`): prefix-consecutive requests become one sequence with several trained spans. Packs are sampled ∝ |advantage| up to a token budget.
4. **Update** (`train.py`): the vLLM servers sleep, and `torchrun` on both GPUs runs a clipped importance-ratio policy gradient against the vLLM sampling logprobs. When starting from Terse, its r=16 LoRA (attention, MLP, linear attention) keeps training; from scratch it is r=32 on attention and MLP. 32k-token contexts fit in one H100 via CPU-offloaded layer checkpointing (peak ~74 GB).
5. The servers wake, hot-load the new adapter, and the next iteration starts (`loop.py`). The loop evaluates every few iterations on train and held-out games.

Stability notes:
- Each vLLM server needs its own torch.compile cache; a shared one got corrupted.
- 12 concurrent episodes per server: hybrid-attention state is large, and more thrashes the KV cache.
- `--gpu-memory-utilization 0.90`: 0.92 and above OOM at runtime.
- A watchdog in `loop.py` restarts a dead server and reloads its adapters.
- `--mm-processor-cache-gb 0`: with sleep/wake, the API server's image cache went out of sync with the engine (`Expected a cached item for mm_hash`), killing the engine.
- vLLM `prompt_logprobs` is unreliable on this model with prefix caching, so every check uses sampled-token logprobs.

Verified: HF-side logprobs match vLLM sampling logprobs (mean ratio ≈ 1.00), and a trained adapter shifts vLLM and HF logprobs identically (Pearson r = 0.96 per token).

## Running

```bash
source scripts/env.sh
scripts/serve.sh 0 1234 &   scripts/serve.sh 1 1235 &
python -m arc3rl.loop baseline        # base model, all 25 games
python -m arc3rl.loop eval --adapter $CKPT_ROOT/terse-hf --tag baseline-terse --all-games
python -m arc3rl.loop train --init-adapter $CKPT_ROOT/terse-hf   # resumable; state in $CKPT_ROOT/state.json
python -m arc3rl.loop eval --adapter $CKPT_ROOT/iter006
python -m arc3rl.report               # results/REPORT.md
```

Storage: weights, venvs and raw rollouts live on NVMe (`/cache/nvme0`, disposable and regenerable). LoRA checkpoints go to `/data/checkpoints/qwen38-arc3-rl`. Summaries in `results/` are committed.

## Results

See [`results/REPORT.md`](results/REPORT.md).
