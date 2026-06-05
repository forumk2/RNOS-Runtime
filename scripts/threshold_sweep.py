"""Threshold sensitivity sweep — RNOS mode only.

Grid over (allow_max, degrade_max) with allow_max < degrade_max:
  allow_max   ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}
  degrade_max ∈ {4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5}

For each valid pair (allow_max < degrade_max), runs all
  seeds × scenarios × personas
and computes:
  false_refusal_rate      = P(refused | scenario ∈ {recovering, stable})
  missed_containment_rate = P(not refused by step 10 | scenario = cascade)

Output: results/sweep_<tag>.jsonl  (one row per threshold pair + aggregate metrics)

Usage
-----
    python scripts/threshold_sweep.py --seeds 30 --tag full
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_harness import _run_single

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ALLOW_MAX_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
DEGRADE_MAX_GRID = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]

SCENARIO_NAMES = ["cascade", "flaky", "recovering", "stable"]
PERSONA_NAMES = ["adversarial", "cautious", "mixed"]

_FALSE_REFUSAL_SCENARIOS = {"recovering", "stable"}
_CONTAINMENT_SCENARIO = "cascade"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _run_pair(args: tuple) -> dict[str, Any]:
    """Run one (allow_max, degrade_max) grid point across all seeds×scenarios×personas."""
    allow_max: float = args[0]
    degrade_max: float = args[1]
    num_seeds: int = args[2]
    max_steps: int = args[3]
    tag: str = args[4]

    rows: list[dict[str, Any]] = []
    for seed in range(num_seeds):
        for scenario in SCENARIO_NAMES:
            for persona in PERSONA_NAMES:
                rec = _run_single(
                    seed,
                    "rnos",
                    scenario,
                    persona,
                    max_steps=max_steps,
                    dry_run=True,
                    allow_max=allow_max,
                    degrade_max=degrade_max,
                    tag=tag,
                )
                rows.append(rec)

    # Aggregate metrics for this grid point
    refused_in_false_scenarios = sum(
        1 for r in rows
        if r["scenario"] in _FALSE_REFUSAL_SCENARIOS and r["final_state"] in ("refused", "degrade_exhausted")
    )
    total_false_scenarios = sum(1 for r in rows if r["scenario"] in _FALSE_REFUSAL_SCENARIOS)

    cascade_rows = [r for r in rows if r["scenario"] == _CONTAINMENT_SCENARIO]
    # "missed containment" = cascade run where RNOS had NOT refused by step 10
    missed_containment = sum(
        1 for r in cascade_rows
        if r["final_state"] not in ("refused", "degrade_exhausted")
        or (r["first_intervention_step"] is not None and r["first_intervention_step"] > 10)
        or r["first_intervention_step"] is None
    )

    false_refusal_rate = refused_in_false_scenarios / max(total_false_scenarios, 1)
    missed_containment_rate = missed_containment / max(len(cascade_rows), 1)

    return {
        "allow_max": allow_max,
        "degrade_max": degrade_max,
        "false_refusal_rate": round(false_refusal_rate, 4),
        "missed_containment_rate": round(missed_containment_rate, 4),
        "n_runs": len(rows),
        "refused_in_false_scenarios": refused_in_false_scenarios,
        "total_false_scenarios": total_false_scenarios,
        "missed_containment": missed_containment,
        "total_cascade": len(cascade_rows),
        "tag": tag,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Sweep entry point
# ---------------------------------------------------------------------------


def run_sweep(
    *,
    num_seeds: int = 30,
    max_steps: int = 20,
    tag: str = "",
    workers: int | None = None,
) -> Path:
    """Run the full grid and write per-pair rows to JSONL.

    Returns the path to the output file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"sweep_{tag}.jsonl" if tag else RESULTS_DIR / "sweep_default.jsonl"
    out_path.write_text("", encoding="utf-8")

    # Build valid pairs (allow_max < degrade_max only)
    pairs: list[tuple[float, float]] = [
        (a, d)
        for a in ALLOW_MAX_GRID
        for d in DEGRADE_MAX_GRID
        if a < d
    ]

    work = [(a, d, num_seeds, max_steps, tag) for a, d in pairs]
    total = len(work)
    print(f"Threshold sweep: {total} grid points  seeds={num_seeds}  tag={tag!r}")
    t0 = time.monotonic()

    n_workers = workers if workers is not None else min(multiprocessing.cpu_count(), 8)
    completed = 0

    with open(out_path, "a", encoding="utf-8") as fh:
        if n_workers <= 1:
            for args in work:
                rec = _run_pair(args)
                fh.write(json.dumps(rec) + "\n")
                completed += 1
                elapsed = time.monotonic() - t0
                print(f"  [{completed}/{total}] allow={rec['allow_max']} degrade={rec['degrade_max']}  "
                      f"frr={rec['false_refusal_rate']:.3f}  mcr={rec['missed_containment_rate']:.3f}  "
                      f"({elapsed:.1f}s)")
        else:
            with multiprocessing.Pool(processes=n_workers) as pool:
                for rec in pool.imap_unordered(_run_pair, work):
                    fh.write(json.dumps(rec) + "\n")
                    completed += 1
                    elapsed = time.monotonic() - t0
                    print(f"  [{completed}/{total}] allow={rec['allow_max']} degrade={rec['degrade_max']}  "
                          f"frr={rec['false_refusal_rate']:.3f}  mcr={rec['missed_containment_rate']:.3f}  "
                          f"({elapsed:.1f}s)")

    elapsed = time.monotonic() - t0
    print(f"Done: {completed} rows -> {out_path}  ({elapsed:.1f}s)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="RNOS threshold sensitivity sweep.")
    parser.add_argument("--seeds", type=int, default=30, metavar="N")
    parser.add_argument("--max-steps", type=int, default=20, metavar="N")
    parser.add_argument("--tag", type=str, default="", metavar="TEXT")
    parser.add_argument("--workers", type=int, default=None, metavar="N")
    args = parser.parse_args()

    out = run_sweep(
        num_seeds=args.seeds,
        max_steps=args.max_steps,
        tag=args.tag,
        workers=args.workers,
    )
    print(f"Sweep results: {out}")


if __name__ == "__main__":
    main()
