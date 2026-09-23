"""RL orchestrator: rollouts on both GPUs -> level-segment advantages -> LoRA update
on both GPUs (vLLM asleep) -> hot-load the new adapter -> repeat; periodic evals.

    python -m arc3rl.loop baseline            # evaluate the base model on every game
    python -m arc3rl.loop train               # run / resume the training loop
    python -m arc3rl.loop eval --adapter DIR  # evaluate one adapter

State lives in $CKPT_ROOT/state.json so the loop resumes after interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from arc3rl.data import build_packs, select_packs
from arc3rl.reward import RewardConfig, assign_advantages, load_episodes, summarize
from arc3rl.rollout import RolloutConfig, all_games, run_rollouts

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "/cache/nvme0/Qwen/runs"))
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", "/data/checkpoints/qwen38-arc3-rl"))
RESULTS = REPO / "results"
SERVERS = ["http://127.0.0.1:1234", "http://127.0.0.1:1235"]
BASE_MODEL = os.environ.get("SERVED_NAME", "Qwen3.8-27B")
SPLIT_FILE = REPO / "configs/split.json"


@dataclass
class LoopConfig:
    passes_per_train_game: int = 8
    eval_passes: int = 4
    eval_every: int = 3
    iterations: int = 12
    train_token_budget: int = 6_000_000  # total packed-sequence tokens trained per iteration
    lr: float = 2e-5
    packs_per_step: int = 4
    reward: RewardConfig = field(default_factory=RewardConfig)


# ---------------------------------------------------------------- servers
def _post(url: str, path: str, **kw) -> requests.Response:
    r = requests.post(url + path, timeout=600, **kw)
    r.raise_for_status()
    return r


def servers_sleep() -> None:
    for s in SERVERS:
        _post(s, "/sleep", params={"level": "1"})
    time.sleep(5)


def servers_wake() -> None:
    for s in SERVERS:
        _post(s, "/wake_up")
    for s in SERVERS:
        while requests.get(s + "/is_sleeping", timeout=30).json().get("is_sleeping"):
            time.sleep(2)


def load_adapter(name: str, path: Path, previous: str | None) -> None:
    for s in SERVERS:
        _post(s, "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(path)})
        if previous:
            try:
                _post(s, "/v1/unload_lora_adapter", json={"lora_name": previous})
            except requests.HTTPError:
                pass


def ensure_adapter_loaded(name: str | None, path: Path | None) -> None:
    if not name:
        return
    for s in SERVERS:
        models = [m["id"] for m in requests.get(s + "/v1/models", timeout=30).json()["data"]]
        if name not in models:
            _post(s, "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(path)})


# ---------------------------------------------------------------- evaluation
def rollout_cfg(model: str) -> RolloutConfig:
    return RolloutConfig(model=model, servers=[s + "/v1" for s in SERVERS])


def evaluate(tag: str, model: str, games: list[str], passes: int, reward: RewardConfig) -> dict:
    run_dir = RUNS_ROOT / "eval" / tag
    if not (run_dir / "rollout_done.json").exists():
        run_rollouts(rollout_cfg(model), games, passes, run_dir)
    eps = load_episodes(sorted(run_dir.glob("server*")))
    result = {"tag": tag, "model": model, "passes": passes, **summarize(eps, reward)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval_{tag}.json").write_text(json.dumps(result, indent=1))
    return result


def load_split() -> dict:
    return json.loads(SPLIT_FILE.read_text())


# ---------------------------------------------------------------- training
def train_adapter(packs_path: Path, out: Path, init: Path | None, cfg: LoopConfig, log_path: Path) -> None:
    env = dict(os.environ)
    env.update({"LD_LIBRARY_PATH": "", "PYTORCH_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "0,1"})
    cmd = [
        str(Path(os.environ["TRAIN_VENV"]) / "bin/torchrun"), "--nproc_per_node", "2", "--master-port", "29512",
        "-m", "arc3rl.train", "--packs", str(packs_path), "--out", str(out),
        "--lr", str(cfg.lr), "--packs-per-step", str(cfg.packs_per_step),
    ]
    if init is not None:
        cmd += ["--init", str(init)]
    with open(log_path, "w") as log:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def train_loop(cfg: LoopConfig) -> None:
    split = load_split()
    state_path = CKPT_ROOT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"iteration": 0, "adapter": None, "name": None}
    history_path = RESULTS / "train_history.jsonl"
    RESULTS.mkdir(exist_ok=True)

    while state["iteration"] < cfg.iterations:
        it = state["iteration"] + 1
        model = state["name"] or BASE_MODEL
        ensure_adapter_loaded(state["name"], Path(state["adapter"]) if state["adapter"] else None)
        it_dir = RUNS_ROOT / "train" / f"iter{it:03d}"
        t0 = time.time()

        # 1. on-policy rollouts on the training games
        if not (it_dir / "rollout_done.json").exists():
            run_rollouts(rollout_cfg(model), split["train"], cfg.passes_per_train_game, it_dir)
        t_roll = time.time() - t0
        eps = load_episodes(sorted(it_dir.glob("server*")), with_records=True)
        groups = assign_advantages(eps, cfg.reward)
        packs = select_packs(build_packs(eps), cfg.train_token_budget, seed=it)
        roll_summary = summarize(eps, cfg.reward)
        packs_path = it_dir / "packs.jsonl"
        with open(packs_path, "w") as f:
            for p in packs:
                f.write(json.dumps(p.to_json()) + "\n")
        n_signal = sum(1 for segs in groups.values() if len(segs) > 1 and any(abs(s.advantage) > 1e-6 for s in segs))
        del eps

        # 2. LoRA update with both GPUs while vLLM sleeps
        out = CKPT_ROOT / f"iter{it:03d}"
        t1 = time.time()
        if packs:
            servers_sleep()
            try:
                train_adapter(packs_path, out, Path(state["adapter"]) if state["adapter"] else None, cfg, it_dir / "train.log")
            finally:
                servers_wake()
            name = f"policy-iter{it:03d}"
            load_adapter(name, out, state["name"])
            state.update({"adapter": str(out), "name": name})
        t_train = time.time() - t1

        rec = {
            "iteration": it,
            "policy_used_for_rollouts": model,
            "rollout": roll_summary["overall"],
            "rollout_per_game": roll_summary["per_game"],
            "groups_with_signal": n_signal,
            "packs_trained": len(packs),
            "packed_tokens": sum(len(p.input_ids) for p in packs),
            "train_stats": json.loads((out / "train_stats.json").read_text()) if packs else [],
            "seconds": {"rollout": t_roll, "train": t_train},
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        state["iteration"] = it
        state_path.write_text(json.dumps(state, indent=1))
        shutil.rmtree(it_dir / "server0" / "src", ignore_errors=True)

        if it % cfg.eval_every == 0 and state["name"]:
            evaluate(f"iter{it:03d}", state["name"], split["train"] + split["heldout"], cfg.eval_passes, cfg.reward)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "train", "eval"])
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--iterations", type=int, default=LoopConfig.iterations)
    args = ap.parse_args()
    cfg = LoopConfig(iterations=args.iterations)
    if args.cmd == "baseline":
        print(json.dumps(evaluate("baseline", BASE_MODEL, all_games(), cfg.eval_passes, cfg.reward)["overall"], indent=1))
    elif args.cmd == "train":
        train_loop(cfg)
    else:
        split = load_split()
        name = "eval-" + Path(args.adapter).name
        ensure_adapter_loaded(name, Path(args.adapter))
        evaluate(args.tag or Path(args.adapter).name, name, split["train"] + split["heldout"], cfg.eval_passes, cfg.reward)


if __name__ == "__main__":
    main()
