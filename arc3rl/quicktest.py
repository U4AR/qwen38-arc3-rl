"""Token-matched comparison of rollout sets: levels and RHAE reached within B generated tokens.

    python -m arc3rl.quicktest --budget 40000 terse=/path/iter001 trained=/path/quick

Each episode is truncated at B generated tokens: a level counts only if it was
completed before the episode had generated B tokens. Its RHAE uses the
official per-level formula with that level's action count.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

from arc3rl.reward import iter_records

_PASS_RE = re.compile(r"_p(\d+)\.html$")


def rhae(levels_done: int, actions: list[int], human: list[int], n_levels: int) -> float:
    total = weights = max_w = 0.0
    for i in range(n_levels):
        w = i + 1
        weights += w
        score = min(115.0, (human[i] / actions[i]) ** 2 * 100) if i < levels_done and actions[i] > 0 else 0.0
        if score > 0:
            max_w += w
        total += score * w
    return min(total / weights, max_w / weights * 100) if weights else 0.0


def episodes(run_dir: Path, budget: int) -> list[dict]:
    out = []
    for bench in sorted(run_dir.glob("server*/benchmark.json")):
        for run in json.loads(bench.read_text())["game_runs"]:
            m = _PASS_RE.search(str(run.get("solver_analysis_html") or ""))
            if m is None or run.get("final_score") is None:
                continue
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", run["game_id"]) + f"_p{m.group(1)}"
            log = bench.parent / "artifacts" / f"{stem}_rl.jsonl.gz"
            if not log.exists():
                continue
            spent, done = 0, 0
            for rec in iter_records(log):
                if spent > budget:
                    break
                done = max(done, (rec.get("level") or 1) - 1)  # level on screen when this request started
                spent += len(rec.get("token_ids") or [])
            done = min(done, int(run.get("levels_completed") or 0))
            if spent <= budget:  # episode ended within budget: count its final outcome
                done = int(run.get("levels_completed") or 0)
            out.append({
                "game": run["game_id"],
                "levels": done,
                "rhae": rhae(done, run["actions_per_level"], run["base_actions_per_level"], run["number_of_levels"]),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40_000)
    ap.add_argument("runs", nargs="+", help="name=run_dir")
    args = ap.parse_args()
    rows = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        rows[name] = episodes(Path(path), args.budget)
    games = sorted({e["game"] for eps in rows.values() for e in eps})
    print(f"within {args.budget:,} generated tokens per episode")
    print(f"{'':10s}" + "".join(f"{n:>24s}" for n in rows))
    for g in games + ["ALL"]:
        cells = []
        for eps in rows.values():
            sel = [e for e in eps if g == "ALL" or e["game"] == g]
            cells.append(f"{mean(e['levels'] for e in sel):.2f} lv  {mean(e['rhae'] for e in sel):5.2f} RHAE ({len(sel)})" if sel else "-")
        print(f"{g[:10]:10s}" + "".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
