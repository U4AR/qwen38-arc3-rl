"""Episode parsing, reward, and level-segment advantages.

Reward is assigned per *level segment*: the requests the model issued while a
given level was on screen. A segment earns

    r = solved * (1 + alpha * max(0, 1 - tokens / token_ref))

so solving a level dominates, and among solves, fewer generated tokens earn
more. Advantages are computed GRPO-style within the group of all segments of
the same (game, level) in one iteration -- every such segment starts from the
same level layout, so they are directly comparable.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterator

_PASS_RE = re.compile(r"_p(\d+)\.html$")


@dataclass
class RewardConfig:
    alpha: float = 0.5  # weight of the token-efficiency bonus relative to solving
    token_ref: float = 40_000.0  # generated tokens at which the bonus reaches zero


@dataclass
class Segment:
    level: int
    solved: bool
    tokens: int  # generated (completion) tokens spent on this level
    records: list[dict] = field(default_factory=list)
    reward: float = 0.0
    advantage: float = 0.0


@dataclass
class Episode:
    game_id: str
    pass_index: int
    source: Path
    number_of_levels: int
    levels_completed: int
    final_score: float
    actions: int
    state: str
    rl_log: Path | None
    generated_tokens: int = 0
    segments: dict[int, Segment] = field(default_factory=dict)
    video_calls: int = 0


def iter_records(path: Path) -> Iterator[dict]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        return  # truncated tail from a killed process
    except (EOFError, OSError):
        return


def load_episodes(experiment_dirs: list[Path], *, with_records: bool = False) -> list[Episode]:
    episodes: list[Episode] = []
    for exp in experiment_dirs:
        bench = next(iter(sorted(exp.rglob("benchmark.json"))), None)
        if bench is None:
            continue
        data = json.loads(bench.read_text())
        artifacts = bench.parent / "artifacts"
        for run in data.get("game_runs", []):
            m = _PASS_RE.search(str(run.get("solver_analysis_html") or ""))
            if m is None:
                continue
            pass_index = int(m.group(1))
            stem = f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', run['game_id'])}_p{pass_index}"
            rl_log = artifacts / f"{stem}_rl.jsonl.gz"
            ep = Episode(
                game_id=run["game_id"],
                pass_index=pass_index,
                source=exp,
                number_of_levels=int(run.get("number_of_levels") or 0),
                levels_completed=int(run.get("levels_completed") or 0),
                final_score=float(run.get("final_score") or 0.0),
                actions=len(run.get("history") or []),
                state=str(run.get("state")),
                rl_log=rl_log if rl_log.exists() else None,
            )
            if ep.rl_log is not None:
                for rec in iter_records(ep.rl_log):
                    level = int(rec.get("level") or 1)
                    seg = ep.segments.setdefault(
                        level, Segment(level=level, solved=ep.levels_completed >= level, tokens=0)
                    )
                    n = len(rec.get("token_ids") or [])
                    seg.tokens += n
                    ep.generated_tokens += n
                    if rec.get("images"):
                        ep.video_calls += 1
                    if with_records:
                        seg.records.append(rec)
            episodes.append(ep)
    return episodes


def segment_reward(seg: Segment, cfg: RewardConfig) -> float:
    if not seg.solved:
        return 0.0
    return 1.0 + cfg.alpha * max(0.0, 1.0 - seg.tokens / cfg.token_ref)


def assign_advantages(episodes: list[Episode], cfg: RewardConfig) -> dict[tuple[str, int], list[Segment]]:
    groups: dict[tuple[str, int], list[Segment]] = defaultdict(list)
    for ep in episodes:
        for seg in ep.segments.values():
            seg.reward = segment_reward(seg, cfg)
            groups[(ep.game_id, seg.level)].append(seg)
    for segs in groups.values():
        baseline = mean(s.reward for s in segs)
        for s in segs:
            # Dr.GRPO-style: centre but do not divide by the group std.
            s.advantage = s.reward - baseline if len(segs) > 1 else 0.0
    return groups


def episode_reward(ep: Episode, cfg: RewardConfig) -> float:
    return sum(segment_reward(s, cfg) for s in ep.segments.values())


def summarize(episodes: list[Episode], cfg: RewardConfig) -> dict:
    """Aggregate metrics, overall and per game."""
    per_game: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        per_game[ep.game_id].append(ep)

    def agg(eps: list[Episode]) -> dict:
        solved_levels = sum(e.levels_completed for e in eps)
        solved_tokens = sum(s.tokens for e in eps for s in e.segments.values() if s.solved)
        return {
            "episodes": len(eps),
            "levels_per_episode": solved_levels / max(1, len(eps)),
            "frac_episodes_ge1_level": sum(e.levels_completed > 0 for e in eps) / max(1, len(eps)),
            "mean_score": mean(e.final_score for e in eps) if eps else 0.0,
            "reward_per_episode": mean(episode_reward(e, cfg) for e in eps) if eps else 0.0,
            "gen_tokens_per_episode": mean(e.generated_tokens for e in eps) if eps else 0.0,
            "gen_tokens_per_solved_level": solved_tokens / solved_levels if solved_levels else None,
            "actions_per_episode": mean(e.actions for e in eps) if eps else 0.0,
            "video_calls_per_episode": mean(e.video_calls for e in eps) if eps else 0.0,
        }

    return {
        "overall": agg(episodes),
        "per_game": {g: agg(eps) for g, eps in sorted(per_game.items())},
    }
