"""Choose the train / held-out game split from the baseline evaluations.

GRPO only learns from (game, level) groups whose rollouts disagree, so games
the starting policy never makes progress on give no gradient. Games with any
baseline success are sorted by mean levels and dealt alternately to train and
held-out (balanced difficulty) until `n_train` are in train; everything else is
held out. The rule only reads baseline results, never held-out outcomes after
training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def choose(n_train: int = 8) -> dict:
    evals = [json.loads((REPO / f"results/eval_{t}.json").read_text()) for t in ("baseline", "baseline-terse")]
    games = sorted(evals[0]["per_game"])
    mean_levels = {g: sum(e["per_game"][g]["levels_per_episode"] for e in evals) / len(evals) for g in games}
    solvable = sorted((g for g in games if mean_levels[g] > 0), key=lambda g: (-mean_levels[g], g))
    train, heldout = [], []
    for i, g in enumerate(solvable):
        (train if i % 2 == 0 and len(train) < n_train else heldout).append(g)
    heldout += [g for g in games if g not in train and g not in heldout]
    return {"train": sorted(train), "heldout": sorted(heldout), "rule": __doc__.strip(), "mean_levels_baselines": mean_levels}


if __name__ == "__main__":
    split = choose(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    out = REPO / "configs/split.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(split, indent=1))
    print(json.dumps({k: split[k] for k in ("train", "heldout")}, indent=1))
