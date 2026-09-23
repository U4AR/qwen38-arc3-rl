"""Write results/REPORT.md from results/eval_*.json and results/train_history.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _subset(per_game: dict, games: list[str]) -> dict:
    rows = [per_game[g] for g in games if g in per_game]
    if not rows:
        return {}
    eps = sum(r["episodes"] for r in rows)
    w = lambda k: sum(r[k] * r["episodes"] for r in rows) / eps  # noqa: E731
    solved = sum(r["levels_per_episode"] * r["episodes"] for r in rows)
    tok_solved = sum((r["gen_tokens_per_solved_level"] or 0) * r["levels_per_episode"] * r["episodes"] for r in rows)
    return {
        "episodes": eps,
        "levels_per_episode": w("levels_per_episode"),
        "frac_episodes_ge1_level": w("frac_episodes_ge1_level"),
        "reward_per_episode": w("reward_per_episode"),
        "mean_score": w("mean_score"),
        "gen_tokens_per_episode": w("gen_tokens_per_episode"),
        "gen_tokens_per_solved_level": tok_solved / solved if solved else None,
        "video_calls_per_episode": w("video_calls_per_episode"),
    }


def _fmt(v, digits=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.{digits}f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


COLS = [
    ("levels_per_episode", "levels / episode"),
    ("frac_episodes_ge1_level", "episodes with ≥1 level"),
    ("reward_per_episode", "reward / episode"),
    ("mean_score", "ARC score"),
    ("gen_tokens_per_solved_level", "gen tokens / solved level"),
    ("gen_tokens_per_episode", "gen tokens / episode"),
    ("video_calls_per_episode", "watch_video calls / episode"),
]


def main() -> None:
    split = json.loads((REPO / "configs/split.json").read_text()) if (REPO / "configs/split.json").exists() else None
    evals = {p.stem.removeprefix("eval_"): json.loads(p.read_text()) for p in sorted(RESULTS.glob("eval_*.json"))}
    order = sorted(evals, key=lambda t: (t != "baseline", t))
    lines = ["# Results", ""]
    if split:
        lines += [f"Train games ({len(split['train'])}): {', '.join(split['train'])}", "",
                  f"Held-out games ({len(split['heldout'])}): {', '.join(split['heldout'])}", ""]
    subsets = [("All games", None)]
    if split:
        subsets = [("Train games", split["train"]), ("Held-out games", split["heldout"])] + subsets
    for title, games in subsets:
        lines += [f"## {title}", "", "| policy | " + " | ".join(c[1] for c in COLS) + " |",
                  "|---" * (len(COLS) + 1) + "|"]
        for tag in order:
            e = evals[tag]
            agg = e["overall"] if games is None else _subset(e["per_game"], games)
            if agg:
                lines.append(f"| {tag} ({agg['episodes']} ep) | " + " | ".join(_fmt(agg.get(k)) for k, _ in COLS) + " |")
        lines.append("")

    lines += ["## Per game: levels / episode", "", "| game | split | " + " | ".join(order) + " |", "|---" * (len(order) + 2) + "|"]
    games_all = sorted({g for e in evals.values() for g in e["per_game"]})
    for g in games_all:
        where = "train" if split and g in split["train"] else "held-out" if split else ""
        lines.append(f"| {g} | {where} | " + " | ".join(_fmt(evals[t]["per_game"].get(g, {}).get("levels_per_episode")) for t in order) + " |")
    lines.append("")

    hist_path = RESULTS / "train_history.jsonl"
    if hist_path.exists():
        hist = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()]
        lines += ["## Training iterations (on-policy rollouts on train games)", "",
                  "| iter | policy | levels/ep | reward/ep | gen tok/solved level | gen tok/ep | groups w/ signal | packs | mean clip frac | approx KL | rollout min | train min |",
                  "|---" * 12 + "|"]
        for h in hist:
            r, ts = h["rollout"], h["train_stats"]
            lines.append(
                f"| {h['iteration']} | {h['policy_used_for_rollouts']} | {_fmt(r['levels_per_episode'])} | {_fmt(r['reward_per_episode'])} | "
                f"{_fmt(r['gen_tokens_per_solved_level'])} | {_fmt(r['gen_tokens_per_episode'])} | {h['groups_with_signal']} | {h['packs_trained']} | "
                f"{_fmt(mean(s['clip_frac'] for s in ts), 3) if ts else '–'} | {_fmt(mean(s['approx_kl'] for s in ts), 4) if ts else '–'} | "
                f"{h['seconds']['rollout'] / 60:.0f} | {h['seconds']['train'] / 60:.0f} |"
            )
        lines.append("")
    (RESULTS / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
