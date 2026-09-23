"""Turn rollout records into packed policy-gradient training samples.

Consecutive requests within one Duck turn usually extend each other exactly
(the next prompt is the previous prompt + the sampled completion + a tool
result), so they are packed into one sequence with several completion spans:
one forward/backward pass then trains every span.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from arc3rl.reward import Episode


@dataclass
class Pack:
    input_ids: list[int]
    spans: list[tuple[int, int]] = field(default_factory=list)  # [start, end) of sampled tokens
    old_logprobs: list[list[float]] = field(default_factory=list)
    advantage: float = 0.0
    images: list[str] = field(default_factory=list)
    game_id: str = ""
    level: int = 0

    def to_json(self) -> dict:
        return self.__dict__


def _packs_for_records(records: list[dict], advantage: float, game_id: str, level: int) -> list[Pack]:
    packs: list[Pack] = []
    current: Pack | None = None
    for rec in sorted(records, key=lambda r: r["i"]):
        prompt, completion = rec["prompt_token_ids"], rec["token_ids"]
        logprobs = rec.get("logprobs") or []
        if len(logprobs) != len(completion):
            continue  # cannot compute a ratio without the sampling logprobs
        if current is None or prompt[: len(current.input_ids)] != current.input_ids:
            current = Pack(input_ids=[], advantage=advantage, game_id=game_id, level=level)
            packs.append(current)
        start = len(prompt)
        current.input_ids = list(prompt) + list(completion)
        current.spans.append((start, start + len(completion)))
        current.old_logprobs.append(list(logprobs))
        current.images = list(rec.get("images") or [])
    return packs


def build_packs(episodes: list[Episode]) -> list[Pack]:
    packs: list[Pack] = []
    for ep in episodes:
        for seg in ep.segments.values():
            if abs(seg.advantage) < 1e-6 or not seg.records:
                continue
            packs.extend(_packs_for_records(seg.records, seg.advantage, ep.game_id, seg.level))
    return packs


def select_packs(packs: list[Pack], token_budget: int, seed: int) -> list[Pack]:
    """Sample packs without replacement, weighted by |advantage|, up to a
    budget of total sequence tokens (the dominant cost of a training pass)."""
    rng = random.Random(seed)
    keyed = sorted(packs, key=lambda p: rng.random() ** (1.0 / max(1e-6, abs(p.advantage))), reverse=True)
    chosen, used = [], 0
    for p in keyed:
        if used + len(p.input_ids) > token_budget:
            continue
        chosen.append(p)
        used += len(p.input_ids)
    return chosen
