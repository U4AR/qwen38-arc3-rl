"""LoRA policy-gradient update on packed rollout samples.

Launch with torchrun, one process per GPU:

    torchrun --nproc_per_node 2 -m arc3rl.train --packs packs.jsonl --out adapter_dir [--init adapter_dir]

Objective per sampled token (clipped importance ratio against the vLLM
sampling logprobs, advantage shared by every token of a level segment):

    L = -min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A),  r = exp(logp - logp_old)

summed over tokens and divided by the global number of sampled tokens in the
optimizer step. Only completion positions go through the LM head, in
checkpointed chunks, so 32k-token contexts fit next to the 27B weights.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LORA_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
LOGIT_CHUNK = 2048


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="adapter to continue from")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--packs-per-step", type=int, default=4, help="per GPU")
    ap.add_argument("--eps-low", type=float, default=0.2)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--kl-stop", type=float, default=0.05,
                    help="end the update once a step's approx KL to the sampling policy exceeds this")
    return ap.parse_args()


def load_images(urls: list[str]):
    from PIL import Image

    return [Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))).convert("RGB") for u in urls]


class OffloadedCheckpoint(torch.autograd.Function):
    """Activation checkpointing whose saved layer input lives in pinned CPU memory.

    With 64 layers x 32k tokens x 5120 dims the checkpointed inputs alone are
    ~21 GB; next to 54 GB of bf16 weights that does not fit on one GPU.
    """

    @staticmethod
    def forward(ctx, run_fn, hidden, *args):
        ctx.run_fn, ctx.args, ctx.device = run_fn, args, hidden.device
        ctx.hidden_cpu = torch.empty(hidden.shape, dtype=hidden.dtype, pin_memory=True)
        ctx.hidden_cpu.copy_(hidden.detach(), non_blocking=True)
        with torch.no_grad():
            return run_fn(hidden, *args)

    @staticmethod
    def backward(ctx, grad_out):
        hidden = ctx.hidden_cpu.to(ctx.device, non_blocking=True).requires_grad_(True)
        with torch.enable_grad():
            out = ctx.run_fn(hidden, *ctx.args)
        torch.autograd.backward(out, grad_out)
        return (None, hidden.grad) + (None,) * len(ctx.args)


def offloaded_checkpoint(run_fn, hidden, *args):
    return OffloadedCheckpoint.apply(run_fn, hidden, *args)


def chunk_logprobs(hidden: torch.Tensor, lm_head: torch.nn.Module, targets: torch.Tensor) -> torch.Tensor:
    """log p(target) for each row of `hidden`, never materialising more than one chunk of logits."""

    def fn(h, t):
        logits = lm_head(h).float()
        return -F.cross_entropy(logits, t, reduction="none")

    out = []
    for s in range(0, hidden.shape[0], LOGIT_CHUNK):
        out.append(checkpoint(fn, hidden[s : s + LOGIT_CHUNK], targets[s : s + LOGIT_CHUNK], use_reentrant=False))
    return torch.cat(out)


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234 + rank)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.init:
        model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none"),
        )
    for layer in model.get_base_model().model.language_model.layers:
        layer._gradient_checkpointing_func = offloaded_checkpoint
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)

    base = model.get_base_model()
    backbone, lm_head = base.model, base.lm_head
    image_token_id = base.config.image_token_id

    packs = [json.loads(line) for line in open(args.packs)]
    packs = [p for p in packs if len(p["input_ids"]) <= args.max_len]
    # Balance by length: longest first, dealt round-robin; equal count per rank.
    packs.sort(key=lambda p: len(p["input_ids"]), reverse=True)
    n_per_rank = len(packs) // world
    mine = packs[rank : n_per_rank * world : world]
    steps = math.ceil(n_per_rank / args.packs_per_step)
    if rank == 0:
        print(f"packs={len(packs)} per_rank={n_per_rank} steps={steps}", flush=True)

    stats_all = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        batch = mine[step * args.packs_per_step : (step + 1) * args.packs_per_step]
        n_tok = torch.tensor(float(sum(e - s for p in batch for s, e in p["spans"])), device=device)
        dist.all_reduce(n_tok)
        acc = {"loss": 0.0, "tokens": 0, "clipped": 0, "ratio_sum": 0.0, "kl_sum": 0.0, "skipped": 0}
        for p in batch:
            ids = torch.tensor(p["input_ids"], device=device)[None]
            extra = {}
            if p["images"]:
                img = processor.image_processor(images=load_images(p["images"]), return_tensors="pt")
                merge = processor.image_processor.merge_size ** 2
                if int((ids == image_token_id).sum()) != int(img["image_grid_thw"].prod(-1).sum() // merge):
                    acc["skipped"] += 1
                    continue
                extra = {
                    "pixel_values": img["pixel_values"].to(device, torch.bfloat16),
                    "image_grid_thw": img["image_grid_thw"].to(device),
                    "mm_token_type_ids": (ids == image_token_id).long(),
                }
            hidden = backbone(input_ids=ids, use_cache=False, **extra).last_hidden_state[0]
            loss = hidden.new_zeros((), dtype=torch.float32)
            adv = float(p["advantage"])
            for (s, e), old in zip(p["spans"], p["old_logprobs"]):
                new_lp = chunk_logprobs(hidden[s - 1 : e - 1], lm_head, ids[0, s:e])
                old_lp = torch.tensor(old, device=device, dtype=torch.float32)
                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1 - args.eps_low, 1 + args.eps_high)
                loss = loss - torch.minimum(ratio * adv, clipped * adv).sum()
                with torch.no_grad():
                    acc["tokens"] += e - s
                    acc["clipped"] += int(((ratio - clipped).abs() > 1e-6).sum())
                    acc["ratio_sum"] += float(ratio.sum())
                    acc["kl_sum"] += float((old_lp - new_lp).sum())
            (loss / n_tok).backward()
            acc["loss"] += float(loss) / float(n_tok)
            del hidden, loss
        # Sum gradients across GPUs (loss is already normalised by global tokens).
        flat = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        for g in flat:
            dist.all_reduce(g)
        for p, g in zip(params, flat):
            p.grad = g
        summed = torch.tensor(
            [acc["loss"], acc["tokens"], acc["clipped"], acc["ratio_sum"], acc["kl_sum"], acc["skipped"]], device=device
        )
        dist.all_reduce(summed)
        loss_v, tok, clip_n, ratio_s, kl_s, skipped = summed.tolist()
        # KL early stop (same decision on every rank: stats are all-reduced).
        early_stop = kl_s / max(1, tok) > args.kl_stop
        if early_stop:
            opt.zero_grad(set_to_none=True)
            gnorm = 0.0
        else:
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
            opt.step()
            opt.zero_grad(set_to_none=True)
        rec = {
            "step": step,
            "loss": loss_v,
            "tokens": tok,
            "clip_frac": clip_n / max(1, tok),
            "mean_ratio": ratio_s / max(1, tok),
            "approx_kl": kl_s / max(1, tok),
            "grad_norm": gnorm,
            "skipped": skipped,
            "elapsed_s": time.time() - t0,
            "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "early_stop": early_stop,
        }
        stats_all.append(rec)
        if rank == 0:
            print(json.dumps(rec), flush=True)
        if early_stop:
            if rank == 0:
                print(f"KL early stop at step {step}: approx_kl {rec['approx_kl']:.4f} > {args.kl_stop}", flush=True)
            break

    if rank == 0:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        (out / "train_stats.json").write_text(json.dumps(stats_all, indent=1))
        print(f"saved adapter to {out}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
