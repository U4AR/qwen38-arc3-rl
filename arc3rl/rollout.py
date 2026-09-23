"""Run Duck episodes against the local vLLM servers and collect trajectories.

One `inference-taaf-run` process per server; each game's passes are split
evenly across servers so both GPUs stay busy and every game gets the same
policy snapshot.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DUCK = REPO / "vendor/duck-harness/ARC3-Inference"
ENVS_DIR = Path(os.environ.get("ENVS_DIR", "/data/projects/TestARC/environment_files"))
SERVE_VENV = Path(os.environ.get("SERVE_VENV", "/cache/nvme0/venvs/arc3"))


def all_games() -> list[str]:
    """Game ids like `ar25-0c556536`, from the offline environment files."""
    games = []
    for game_dir in sorted(p for p in ENVS_DIR.iterdir() if p.is_dir()):
        for version in sorted(p for p in game_dir.iterdir() if p.is_dir()):
            games.append(f"{game_dir.name}-{version.name}")
    return games


@dataclass
class RolloutConfig:
    model: str  # served model name or LoRA adapter name
    servers: list[str]  # base urls, e.g. http://127.0.0.1:1234/v1
    minutes_per_game: float = 120.0  # safety cap only; the token budget is the real limit
    max_actions: int | None = 400
    max_generated_tokens: int = 150_000  # per episode, load-independent (Kaggle Duck: 132 min/game)
    concurrent_per_server: int = 12  # more thrashes the KV cache (hybrid-attention state is large)
    video_tool: bool = True
    grid_image_upscale: int = 4  # 64x64 board -> 256x256 image every turn (Duck configs/inference.json)
    temperature: float = 0.6
    max_output_tokens: int = 0  # 0 = rest of the context window, as in the Duck
    context_window: int = 32768  # Duck analyzer history budget; vLLM window is 65536 (Duck Kaggle default)


def _env(cfg: RolloutConfig, base_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "LOCAL_ANALYZER_BASE_URL": base_url,
            "LOCAL_ANALYZER_PROVIDER": "vllm",
            "LOCAL_ANALYZER_MODEL_ID": cfg.model,
            "LOCAL_ANALYZER_CONTEXT_WINDOW": str(cfg.context_window),
            "LOCAL_ANALYZER_MAX_OUTPUT": str(cfg.max_output_tokens),
            "LOCAL_ANALYZER_TIMEOUT": "900",
            "LOCAL_ANALYZER_TEMPERATURE": str(cfg.temperature),
            "LOCAL_ANALYZER_TOP_P": "0.95",
            "LOCAL_ANALYZER_TOP_K": "20",
            "LOCAL_ANALYZER_TOOL_STEPS": "0",
            "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
            "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
            "LOCAL_ANALYZER_YIELD_SECONDS": "60",
            "LOCAL_ANALYZER_ENABLE_THINKING": "true",
            "LOCAL_ANALYZER_API_KEY": "EMPTY",
            # As in the Duck: the current board as an image on every turn.
            "MULTIMODAL_CONTEXT": "current_grid",
            "MULTIMODAL_UPSCALE": str(cfg.grid_image_upscale),
            "VIDEO_TOOL": "1" if cfg.video_tool else "0",
            "RL_TRAJECTORY_LOG": "1",
            "EPISODE_MAX_GENERATED_TOKENS": str(cfg.max_generated_tokens),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _launch(cfg: RolloutConfig, base_url: str, games: list[str], passes: int, out_dir: Path) -> subprocess.Popen:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(SERVE_VENV / "bin/inference-taaf-run"),
        "--game", ",".join(games),
        "--environments-dir", str(ENVS_DIR),
        "--experiment-dir", str(out_dir),
        "--n-passes", str(passes),
        "--concurrent-jobs", str(cfg.concurrent_per_server),
        "--max-runtime-minutes", str(cfg.minutes_per_game),
        "--agent", "inference",
        "--model", "local",
        "--analyzer-timeout", "900",
        "--deployment-target", "inline",
        "--deployment-wait",
    ]
    if cfg.max_actions:
        cmd += ["--max-actions", str(cfg.max_actions)]
    log = open(out_dir / "runner.log", "w")
    return subprocess.Popen(cmd, cwd=DUCK, env=_env(cfg, base_url), stdout=log, stderr=subprocess.STDOUT)


def run_rollouts(cfg: RolloutConfig, games: list[str], passes: int, run_dir: Path) -> list[Path]:
    """Play `passes` episodes of every game; returns one experiment dir per server."""
    n = len(cfg.servers)
    per_server = [passes // n + (1 if i < passes % n else 0) for i in range(n)]
    procs, dirs = [], []
    for i, (url, p) in enumerate(zip(cfg.servers, per_server)):
        if p == 0:
            continue
        d = run_dir / f"server{i}"
        procs.append(_launch(cfg, url, games, p, d))
        dirs.append(d)
    (run_dir / "rollout_config.json").write_text(
        json.dumps({"games": games, "passes": passes, **cfg.__dict__}, indent=2)
    )
    started = time.time()
    for proc in procs:
        proc.wait()
    (run_dir / "rollout_done.json").write_text(
        json.dumps({"seconds": time.time() - started, "returncodes": [p.returncode for p in procs]})
    )
    return dirs
