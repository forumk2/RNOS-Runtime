"""Exp-6b ARM 2, Phase A driver: generate the wrongness grid and write artifacts.

    python -m experiments.exp6b.arm2.generate_grid

Writes results/exp6b_arm2/ traces + manifest and docs/exp6b_arm2_grid.md.
STOPs (exit code 1, doc says STOP) if >5% of nonzero-delta traces fail the
coherent-confident-wrong manipulation check after regeneration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.exp6b.arm2.gridgen import (
    CONTROL_CELL,
    DEGRADE_THRESHOLD,
    DELTAS,
    EPSILON,
    MANIFEST_PATH,
    N_SEEDS,
    N_STEPS,
    ONSETS,
    RESULTS_DIR,
    RNOS_TRACE_PATH,
    SHAPES,
    all_cells,
    generate_cell,
    write_trace_files,
)

DOC_PATH = _REPO_ROOT / "docs" / "exp6b_arm2_grid.md"
STOP_FAILURE_FRACTION = 0.05


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RNOS_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RNOS_TRACE_PATH.write_text("", encoding="utf-8")

    cells = all_cells()
    manifest_cells: list[dict] = []
    n_nonzero_total = 0
    n_nonzero_failed = 0

    print(f"Exp-6b ARM 2 Phase A: {len(cells)} cells x {N_SEEDS} seeds")

    for spec in cells:
        traces = generate_cell(spec)
        files = [write_trace_files(t) for t in traces]

        n_pass = sum(1 for t in traces if t.check_ok)
        n_fail = len(traces) - n_pass
        regenerated = [t.seed for t in traces if t.gen_seed != t.seed]
        if spec["delta"] > 0.0:
            n_nonzero_total += len(traces)
            n_nonzero_failed += n_fail

        cell_entry = {
            **spec,
            "n_traces": len(traces),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "regenerated_seeds": regenerated,
            "failed_seeds": {
                t.seed: t.check_failures for t in traces if not t.check_ok
            },
            "mean_terminal_distance": round(mean(t.terminal_distance for t in traces), 4),
            "max_entropy_observed": round(max(t.max_entropy for t in traces), 4),
            "files": files,
        }
        manifest_cells.append(cell_entry)
        flag = "" if n_fail == 0 else f"  ** {n_fail} FAILED -> quarantine **"
        print(f"  {spec['cell']:18s} pass={n_pass}/{len(traces)}  "
              f"term_dist={cell_entry['mean_terminal_distance']:8.3f}  "
              f"max_ent={cell_entry['max_entropy_observed']:.3f}{flag}")

    failure_fraction = (n_nonzero_failed / n_nonzero_total) if n_nonzero_total else 0.0
    stop = failure_fraction > STOP_FAILURE_FRACTION

    manifest = {
        "experiment": "6b_arm2_phaseA",
        "grid": {
            "deltas_nonzero": DELTAS,
            "onsets": ONSETS,
            "shapes": SHAPES,
            "control_cell": CONTROL_CELL,
            "n_seeds": N_SEEDS,
            "n_steps": N_STEPS,
            "epsilon": EPSILON,
            "degrade_threshold": DEGRADE_THRESHOLD,
        },
        "n_cells": len(cells),
        "n_traces_total": sum(c["n_traces"] for c in manifest_cells),
        "nonzero_delta_traces": n_nonzero_total,
        "nonzero_delta_failures": n_nonzero_failed,
        "failure_fraction": round(failure_fraction, 4),
        "stop_rule_triggered": stop,
        "cells": manifest_cells,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest -> {MANIFEST_PATH.relative_to(_REPO_ROOT)}")

    doc = _build_doc(manifest)
    DOC_PATH.write_text(doc, encoding="utf-8")
    print(f"Grid doc -> {DOC_PATH.relative_to(_REPO_ROOT)}")

    if stop:
        print(f"\nSTOP: {failure_fraction:.1%} of nonzero-delta traces failed "
              f"the coherence check (> {STOP_FAILURE_FRACTION:.0%}).")
        sys.exit(1)
    print(f"\nPhase A gate PASSED: failure fraction {failure_fraction:.1%} "
          f"<= {STOP_FAILURE_FRACTION:.0%}.")


def _build_doc(manifest: dict) -> str:
    g = manifest["grid"]
    lines = [
        "# Exp-6b ARM 2 — Phase A: Wrongness Grid",
        "",
        "**Date:** 2026-06-11  "
        f"**Status:** {'STOP — manipulation check failed' if manifest['stop_rule_triggered'] else 'PASS — gate cleared for Phase B'}",
        "",
        "## Grid definition",
        "",
        "Wrongness is parameterized over the unmodified accumulator env "
        "(`tools/accumulator_env.py`) by a planner wrapper in "
        "`experiments/exp6b/arm2/gridgen.py`. The post-onset effective target is "
        "`T_CORRECT - delta`; tool names still alternate `step_toward`/`add` and "
        "wrongness lives only in tool arguments, exactly as in exp-6.",
        "",
        f"- **delta** (offset magnitude): 0 (control) + {g['deltas_nonzero']} — "
        "delta=20 reproduces exp-6's original `T_WRONG = -10`; delta=1 is just past "
        f"the correctness tolerance EPSILON={g['epsilon']}",
        f"- **onset**: steps {g['onsets']} (step 1 | ~1/3 | ~2/3 of {g['n_steps']} steps)",
        "- **shape**: `sudden` (full offset at onset) | `gradual` (linear ramp from "
        "0 at onset to full delta at the final step)",
        f"- **seeds**: {g['n_seeds']} per cell; delta=0 collapses onset/shape and is "
        "generated once per seed (cell `d0`)",
        "- **per-seed variation**: step magnitude ~ U[1.5, 2.5], latency 20±3 ms, "
        "drawn per (seed, step). Exp-6 variant A was deterministic across seeds; "
        "this makes seeds distinct traces while keeping the entropy floor (~0.51) "
        "far below DEGRADE=3.0.",
        "",
        "Traces are emitted in the exp-6 CSV schema (identical columns, written by "
        "the imported exp-6 `_write_csv`). The schema has no tool-name/argument "
        "columns, so each trace also gets a `.calls.jsonl` sidecar (step, tool, "
        "kwargs, v_before, v_after) for the Phase C judge harness. The CSV itself "
        "is unchanged so existing exp-6 tooling reads it.",
        "",
        "## Manipulation check (exp-6 coherent-confident-wrong definition)",
        "",
        "Every nonzero-delta trace must satisfy, over the WHOLE run:",
        "",
        "1. wrong terminal state: `distance_to_correct(final) > 0.5`",
        "2. zero tool failures (every step executed, success=True)",
        "3. flat `consecutive_failures` (0 throughout)",
        "4. existing detectors silent: entropy < 3.0 (default DEGRADE, the stricter "
        "of exp-6's two policy sets) throughout; RNOS decision ALLOW throughout; "
        "AdaptiveCircuitBreaker never blocks; coherence regime `resonant` "
        "throughout; lambda_proxy ≥ 0.45 throughout",
        "",
        "delta=0 control traces must satisfy 2–4 and be CORRECT "
        "(terminal distance ≤ 0.5).",
        "",
        "Failed traces are regenerated with a displaced seed (≤3 attempts), then "
        "quarantined and flagged here — never silently included.",
        "",
        "## Cell table",
        "",
        "| Cell | delta | onset | shape | Traces | Pass | Fail | Regen seeds | Mean terminal dist | Max entropy |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in manifest["cells"]:
        onset = c["onset"] if c["onset"] is not None else "—"
        shape = c["shape"] if c["shape"] is not None else "—"
        regen = ", ".join(map(str, c["regenerated_seeds"])) or "none"
        lines.append(
            f"| {c['cell']} | {c['delta']:g} | {onset} | {shape} "
            f"| {c['n_traces']} | {c['n_pass']} | {c['n_fail']} | {regen} "
            f"| {c['mean_terminal_distance']:.3f} | {c['max_entropy_observed']:.3f} |"
        )

    lines += [
        "",
        "## Check results",
        "",
        f"- Total traces: **{manifest['n_traces_total']}** "
        f"({manifest['n_cells']} cells)",
        f"- Nonzero-delta traces: {manifest['nonzero_delta_traces']}, "
        f"failures after regeneration: **{manifest['nonzero_delta_failures']}** "
        f"({manifest['failure_fraction']:.1%})",
        f"- STOP rule (>5% nonzero-delta failures): "
        f"**{'TRIGGERED' if manifest['stop_rule_triggered'] else 'not triggered'}**",
        "",
        "Notes:",
        "",
        "- For delta=20 with onset=17, the run ends before v reaches the full "
        "wrong target (≈9 post-onset steps × ≈2 units/step < 20 units); the trace "
        "is still terminally wrong by a wide margin, and realized terminal "
        "distance per cell is recorded above.",
        "- `distance_to_correct` is logged to CSV only (exp-6 oracle-independence "
        "convention) and is withheld from all judge prompts in Phases B–D.",
        "",
        "## Files",
        "",
        "- Traces: `results/exp6b_arm2/traces/exp6b2_{cell}_seed{NN}.csv` + "
        "`.calls.jsonl`",
        "- Manifest: `results/exp6b_arm2/grid_manifest.json`",
        "- Generator: `experiments/exp6b/arm2/gridgen.py`, "
        "`experiments/exp6b/arm2/generate_grid.py` (no core code modified)",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
