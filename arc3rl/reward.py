"""Episode parsing, reward, and level-segment advantages.

Reward is assigned per *level segment*: the requests the model issued while a
given level was on screen. A solved level k earns

    r = k * (solve_bonus + min(1.15, (human_actions / agent_actions)^2) + alpha * token_eff)

mirroring the ARC-AGI-3 score (RHAE): level k has weight k, and each level is
scored by action efficiency against the human baseline, with the action count
reset at every level (as in the game). `solve_bonus` keeps a slow solve worth
more than no solve; `token_eff = max(0, 1 - tokens / token_ref)` rewards
solving with fewer generated tokens. Unsolved levels earn 0. Advantages are computed GRPO-style within the group of all segments of
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
    solve_bonus: float = 0.5  # any solve beats no solve, however many actions it took
    action_efficiency: bool = True  # min(1.15, (human/agent actions)^2), as in RHAE
    alpha: float = 0.25  # weight of the generated-token efficiency bonus
    token_ref: float = 60_000.0  # generated tokens at which the token bonus reaches zero
    level_weighting: bool = True  # weight level k by k, like the official RHAE score


@dataclass
class Segment:
    level: int
    solved: bool
    tokens: int  # generated (completion) tokens spent on this level
    actions: int = 0  # game actions spent on this level (reset per level, as in the game)
    human_actions: int = 0  # human baseline for this level
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
    actions_per_level: list[int] = field(default_factory=list)
    base_actions_per_level: list[int] = field(default_factory=list)


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


def _load_images(rl_log: Path) -> dict[str, str]:
    path = rl_log.with_name(rl_log.name.replace("_rl.jsonl.gz", "_images.jsonl.gz"))
    return {r["h"]: r["url"] for r in iter_records(path)} if path.exists() else {}


def load_episodes(experiment_dirs: list[Path], *, with_records: bool = False) -> list[Episode]:
    episodes: list[Episode] = []
    for exp in experiment_dirs:
        bench = next(iter(sorted(exp.rglob("benchmark.json"))), None)
        if bench is None:
            continue
        data = json.loads(bench.read_text())
        artifacts = bench.parent / "artifacts"
        for run in data.get("game_runs", []):
            if run.get("final_score") is None or run.get("state") == "playing":
                continue  # unfinished (e.g. run was stopped); no outcome to score
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
                actions_per_level=list(run.get("actions_per_level") or []),
                base_actions_per_level=list(run.get("base_actions_per_level") or []),
            )
            if ep.rl_log is not None:
                images = _load_images(ep.rl_log) if with_records else {}
                for rec in iter_records(ep.rl_log):
                    level = int(rec.get("level") or 1)
                    seg = ep.segments.get(level)
                    if seg is None:
                        idx = level - 1
                        seg = ep.segments[level] = Segment(
                            level=level,
                            solved=ep.levels_completed >= level,
                            tokens=0,
                            actions=ep.actions_per_level[idx] if idx < len(ep.actions_per_level) else 0,
                            human_actions=ep.base_actions_per_level[idx] if idx < len(ep.base_actions_per_level) else 0,
                        )
                    n = len(rec.get("token_ids") or [])
                    seg.tokens += n
                    ep.generated_tokens += n
                    ep.video_calls += int(rec.get("video_replays") or 0)
                    if with_records:
                        # Resolve image hashes (older logs stored data URLs inline).
                        rec["images"] = [images.get(x, x) for x in rec.get("images") or []]
                        seg.records.append(rec)
            episodes.append(ep)
    return episodes


def segment_reward(seg: Segment, cfg: RewardConfig) -> float:
    if not seg.solved:
        return 0.0
    weight = float(seg.level) if cfg.level_weighting else 1.0
    token_eff = max(0.0, 1.0 - seg.tokens / cfg.token_ref)
    if cfg.action_efficiency and seg.actions > 0 and seg.human_actions > 0:
        action_eff = min(1.15, (seg.human_actions / seg.actions) ** 2)
    else:
        action_eff = 1.0 if not cfg.action_efficiency else 0.0
    return weight * (cfg.solve_bonus + action_eff + cfg.alpha * token_eff)


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
