---
license: apache-2.0
library_name: peft
base_model:
  - Qwen/Qwen3.8-27B
  - agentionai/Signal-3.8-27B
  - Shockem/Signal-3.8-27b-Heretic-ara
tags:
  - lora
  - dpo
  - reasoning
  - coding
  - qwen3
---

# Qwen3.8-27B Terse-Coder LoRA

Rank-16 DPO LoRA that **shortens chain-of-thought reasoning on coding tasks
while preserving correctness**. Trained on preference pairs selected
objectively — concise-but-correct traces chosen by automated test execution +
entropy-based step pruning, no human or LLM judging.

> **This is the final round.** The adapter at this repo root is round 8
> (checkpoint-salvage epoch 1) and concludes the Terse-Coder study — no
> further rounds are planned. Earlier rounds remain available under
> `archive/` (round 7 is the immediately previous root; round 6 is the
> pass-neutral option for heretic-ara bases).

The effect is **compounding**: it stacks on top of whatever conciseness the
base already has. Recommended pairings, in order:

1. **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** (full
   precision) or
   [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
   — stock base. This is where the adapter shines: near-direct answers on
   coding problems (round 8 measures **−95%** vs the NVIDIA quant's verbose
   baseline, mean ~41 reasoning tokens vs ~700) with pass rate intact.
2. [agentionai/Signal-3.8-27B](https://huggingface.co/agentionai/Signal-3.8-27B)
   (full precision) or
   [Shockem/Signal-3.8-27B-NVFP4](https://huggingface.co/Shockem/Signal-3.8-27B-NVFP4)
   — the training-lineage base; **−40%** on top of Signal's already-short
   reasoning, and pass rate *improves*.
3. [Shockem/Signal-3.8-27b-Heretic-ara](https://huggingface.co/Shockem/Signal-3.8-27b-Heretic-ara)
   (full precision) or
   [Shockem/Signal-3.8-27b-Heretic-ara-NVFP4](https://huggingface.co/Shockem/Signal-3.8-27b-Heretic-ara-NVFP4)
   — optional: the original training target. Round 8 cuts deeper but costs
   pass on this base; if you run heretic-ara and want the pass-neutral cut,
   round 6 in `archive/r6/` (−36%, pass intact) is the better fit for that
   specific base.

## Background

Qwen-3.8-27b is an excellent dense model. It's a breakthrough in locally hosted models on consumer-grade hardware. The biggest challenge it faces is that its reasoning can be, at times, overly verbose. This isn't necessarily an issue, as it can pull itself out of hallucinations, but with user hardware around this model size, it ends up causing long waits, oftentimes into hours, before any outputs or edits occur.

Recently, agentionai released Signal-3.8-27B. This opened the gates to the idea of lowering reasoning by finetuning the model, rather than handling it via templates or configuration. The outcome was a model that thought significantly less than the base model, with marginal differences in error. This sent me down a rabbit hole of testing its reasoning and how it affected the model's output. To my surprise, it was incredibly close to the base, even its traces were very close, just cleaner overall.

My experiment is to continue this research and push it further. So far, I've trained on traces from Signal, the base NVIDIA provided NVFP4 quant, and my own abliterated variant, against HumanEval, 600 questions per round. This is now round 7, and it has shown significant improvement. We are now at a 95% reduction in reasoning against the NVIDIA quant, and 52% against Signal. This was originally created as a LoRA adapter that is then merged into a custom recipe for Qwen-3.8-27b. I have provided both a merged model and a LoRA adapter. Thank you for testing and providing feedback!

## Results

Round 8 (this adapter) measured on **nvidia/Qwen3.8-27B-NVFP4** with the
adapter loaded at runtime (vLLM 0.28, LoRA, FP8 KV): held-out 40 problems
(20 HumanEval + 20 MBPP-sanitized, disjoint from training), temp 0.6 /
top_k 20 / top_p 0.95 / rep-penalty 1.05, pass@1 by automated test
execution, reasoning tokens from `completion_tokens_details.reasoning_tokens`:

| Round 8 (LoRA, NVIDIA NVFP4 base) | Value |
|---|---|
| Held-out-40 pass | **70.0%** (28/40; base ~72%) |
| Held-out-40 reasoning tokens | **mean 41, median 32** |
| MBPP+ (EvalPlus official, 378 tasks, greedy) | **80.4%** |
| HumanEval+ (merged fp32→fp16→v3-NVFP4 build, greedy) | **91.5%** |
| GSM8K-200 (merged build) | **98.5%** |
| GPQA-Diamond, n=198 (merged build) | **79.8%** |

Round 7 (previous root, now in `archive/r7/`) median-of-runs numbers, same
protocol on NVFP4 bases (n≥3 per problem, paired permutation — single runs
are heavy-tailed, run means swing ±20%):

| Base (all NVFP4) | Serving | Reasoning tokens | p | pass@1 base → +LoRA |
|---|---|---|---|---|
| **nvidia/Qwen3.8-27B-NVFP4** (stock) | LoRA, 3 runs | **−94.7%** | 0.0001 | 72.5% → 70.8% |
| **House stock NVFP4 v3** (NVIDIA-style Hessian solve) | LoRA, 6 runs | **−92.4%** | 0.0001 | 69.2% → 67.9% |
| House stock NVFP4 v2 (absmax W4A4) | LoRA, 3 runs | −49.5% | 0.0001 | 70.8% → 69.2% |
| **Signal** (no heretic tuning) | LoRA, 3 runs | **−40.0%** | 0.0002 | 64.2% → 65.8% |
| **Signal Heretic-ara** | merged quant, 6 runs | **−42.2%** | 0.0001 | 62.5% → 60.0% ⚠️ |

Effect *size* varies by base quant calibration. NVIDIA's checkpoint (and the
v3 recipe, which replicates its Local-Hessian solve on MLP/lm_head) shows the
full ~−93% cut; an absmax-calibrated quant of the *same weights* (v2) only
cut −49.5%. Direction and pass-neutrality hold on every base tested. If you
build your own W4A16 quant to pair with this adapter, Hessian-weighted
calibration on MLP/lm_head is what unlocks the full effect.

**Prefer a merged model over a runtime adapter?** The round-7 merge remains
available pre-merged into stock Qwen3.8-27B as
[Shockem/Qwen3.8-27b-Terse-Coder](https://huggingface.co/Shockem/Qwen3.8-27b-Terse-Coder)
(fp16) and
[Shockem/Qwen3.8-27b-Terse-Coder-NVFP4](https://huggingface.co/Shockem/Qwen3.8-27b-Terse-Coder-NVFP4)
(v3-recipe W4A16): 67.5% pass with ~38 reasoning tokens/problem baked in —
same behavior, zero LoRA plumbing. **Do not stack this adapter on top of the
merged model** (double application over-shortens: 63% pass with `no_code`
failures in our testing).

⚠️ On heretic-ara specifically, round 6 (`archive/r6/`) is the pass-neutral
option (−36.0%, 62.5% → 63.7%). On stock and Signal, the later rounds
dominate.

## Runtime LoRA is the full-strength deployment form

The weight deltas are deliberately tiny (‖Δ‖/‖W‖ ≈ 4e-4–1e-3) —
**below bf16's per-element resolution**. Measured delta survival when merging
into the base: 31–61% in bf16, 94–99.9% in fp16, and merged-to-NVFP4
attenuates on top of that. So: **load the adapter at runtime** on any base
quant (the delta applies in bf16 at compute time — full strength). If you
merge, merge in fp32 and store fp16; any merged 4-bit artifact loses some of
the effect.

**Round 8 measured this directly.** The same round-8 epoch-1 adapter scores
70% on the held-out-40 as a runtime LoRA on the NVIDIA base, but only
60–62% after the identical deltas go through fp32-merge → fp16 → v3-NVFP4
re-quant (two independent runs; every other battery leg — HumanEval+ 91.5,
MBPP+ 79.4, GSM8K 98.5, GPQA-Diamond 79.8 — stayed at or above the round-7
merged profile on the same merged artifact). If you need the terse behavior
at full strength, serve the adapter; treat merged 4-bit builds as the
convenience form with a small capability tax.

## Serve it with speculative decoding (recommended)

The adapter does **not** touch the MTP draft head, so MTP speculative
decoding keeps working at full acceptance (measured 0.43 with and without
the adapter on the NVIDIA checkpoint — outputs are target-verified, so spec
decode is lossless). Measured on 2× RTX 5060 Ti 16 GB (vLLM 0.28, FP8 KV,
`num_speculative_tokens: 3`):

| Config | Wall tok/s |
|---|---|
| NVIDIA NVFP4 base, MTP spec on | 54.1 |
| House stock NVFP4 v3 base, MTP spec on | 55.5 |
| House stock NVFP4 v2 base, MTP spec on | 53.7 |
| **NVIDIA NVFP4 + this adapter, MTP spec on** | **48.9** |
| House stock NVFP4 v3 + this adapter, MTP spec on | 49.1 |
| House stock NVFP4 v2 + this adapter, MTP spec on | 45.0 |
| NVIDIA NVFP4 + this adapter, spec OFF | 27.4 |
| Signal NVFP4 (house quant) + this adapter, spec on | 50.4 |
| **Terse-Coder-NVFP4 (round-7 merged, no adapter), MTP spec on** | **54.5** |

**Turn spec decode ON** — it is +78% wall speed with the adapter loaded. On
2×16 GB cards with the adapter + spec, cap context at ~175k (200k needs more
KV headroom than the two cards have; single-card 24 GB+ rigs are unaffected).

## Variants

This page holds the round-8 bf16 adapter at the repo root. The round-7
quantized variants (fp8 / int4 / nvfp4 / gguf adapter quants) are archived
under `archive/r7/` — they quantize the *round-7* deltas and are kept for
reproducibility; the bf16 root adapter is the servable, measured artifact.

Direct download:

| Variant | Format | Size | Download | Use |
|---|---|---|---|---|
| **bf16** (repo root) | PEFT bf16, rank 16, alpha 32 (vLLM-native VL key layout) | ~228 MB | [adapter_model.safetensors](https://huggingface.co/Shockem/Qwen3.8-27b-Terse-Coder-LoRA/resolve/main/adapter_model.safetensors) | **Servable artifact** — vLLM/PEFT load this directly; powers `PeftModel.from_pretrained(model, "Shockem/Qwen3.8-27b-Terse-Coder-LoRA")` |

The bf16 adapter sits at the repo root (standard PEFT layout, powers the
Hub's "Use this model" snippet) with the vLLM-native
(`language_model.model.layers.*`) key layout.

## Serving with vLLM (tested path)

Tested on vLLM 0.28, including alongside **MTP speculative decoding** and on
modelopt NVFP4 bases. The released adapter carries vLLM-native
(`language_model.model.layers.*`) key layout — it attaches correctly to the
`Qwen3_5ForConditionalGeneration` module tree. (Plain PEFT exports from a
text-only `AutoModelForCausalLM` run silently attach **zero** weights in vLLM
— basename checks pass, nothing is applied. If you re-export this adapter
yourself, keep the VL-layout keys.)

CLI:

```bash
vllm serve nvidia/Qwen3.8-27B-NVFP4 \
  --enable-lora \
  --lora-modules cot-lora=/path/to/bf16 \
  --max-lora-rank 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Then request with `model: "cot-lora"` (adapted) or the base model id
(unadapted) — both are live on the same server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="cot-lora",  # terse reasoning
    messages=[{"role": "user", "content": "Write a Python lru_cache."}],
    temperature=0.6, top_p=0.95,
)
print(resp.usage.completion_tokens_details.reasoning_tokens)
```

Hot-loading on an already-running server also works:
`POST /v1/load_lora_adapter {"lora_name": "cot-lora", "lora_path": "..."}`.

## Other bases, quants, formats, backends

The adapter is a behavioral edit to the shared Qwen3.8-27B text stack (all
attention + MLP + gated-deltanet `linear_attn` projections; no vision, no MTP
weights), so it **should stack fine on other quantizations (FP8/INT4/GGUF
bases), full-precision bases, and other backends (SGLang, TabbyAPI/EXL3,
llama.cpp)** — the preference it encodes is not quantization-specific. That
said, only the NVFP4 + vLLM combination above has been measured; treat other
combinations as untested and validate before relying on them.

## Training (summary)

- **Round 8 data:** a 301-problem pool (HumanEval + MBPP-sanitized, plus
  mined failures from the round-7 EvalPlus runs) with two pair channels:
  entropy-pruned (short+pass vs long+pass) and hard-negative (long+pass vs
  short+fail) — 70 stock pairs + 430 Signal pairs, min-capped to 122 training
  rows.
- **Round 8 method:** DPO (trl), β 0.05, lr 1e-5, 3 epochs, effective batch
  8, initialized from the round-7 adapter. Round 7's epoch-3 checkpoint
  overfit (reward accuracy 1.0, held-out collapse after merge), so round 8
  is a **checkpoint-selection retrain** of the same run: per-epoch adapter
  checkpoints were evaluated independently on the NVIDIA base (held-out-40 +
  MBPP+), and **epoch 1** was selected — the earliest checkpoint before the
  overfit regime. It is the least-trained checkpoint that still clears the
  quality gates, which is why its deltas are the smallest of any round.
- **Earlier rounds:** see `archive/r7/README.md` and `archive/r6/README.md`
  for their training summaries.

## Caveats

- Targeted at coding tasks with thinking enabled. Behavioral LoRA, not a
  knowledge edit.
- The preference is "shorter reasoning, identical answer" — if a task needs
  long derivation, raise `reasoning_effort` as usual.
- If you serve with speculative decoding, make sure the generation config has
  **no `min_p`** — vLLM 0.28 rejects min_p under spec decode.

## Attributions & licenses

This adapter is trained against, and licensed for use with,
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), © Qwen Team,
Alibaba Cloud, **Apache 2.0**; this adapter is likewise Apache 2.0 and the
upstream license and copyright notices are retained. Credits:

- **Qwen Team (Alibaba Cloud)** — the Qwen3.8-27B base model (Apache 2.0).
- **[agentionai](https://huggingface.co/agentionai/Signal-3.8-27B)** —
  Signal-3.8-27B, one of the trace-generation policies.
- **[p-e-w](https://github.com/p-e-w/heretic)** — the Heretic tool; a
  heretic-ara abliterated variant of Signal was another trace-generation
  policy.
- **NVIDIA** — [Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
  (third policy and eval baseline) and [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
  0.45 (Apache 2.0, quant tooling).
- **OpenAI** — [HumanEval](https://github.com/openai/human-eval) (MIT), and
  **Google** — [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
  (CC-BY 4.0): prompt sources for training and held-out evaluation.
- **Hugging Face [TRL](https://github.com/huggingface/trl)** (Apache 2.0) —
  the DPO trainer; **[llama.cpp](https://github.com/ggml-org/llama.cpp)**
  (MIT) — GGUF adapter conversion; **[Datacurve](https://huggingface.co/datasets/datacurve/deep-swe)**
  — DeepSWE, independent evaluation only.
