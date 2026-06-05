"""Evaluation report generator.

Reads results/eval_<tag>.jsonl (from eval_harness.py) and optionally
results/sweep_<tag>.jsonl (from threshold_sweep.py) and produces:

  results/eval_report_<tag>.md   — markdown summary tables
  results/boxplot_steps_<tag>.png
  results/boxplot_failures_<tag>.png
  results/entropy_progression_<tag>.png
  results/threshold_heatmap_<tag>.png   (requires sweep file)

Usage
-----
    python scripts/eval_report.py --tag full
    python scripts/eval_report.py --tag full --no-sweep   # skip heatmap
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SCENARIOS = ["cascade", "flaky", "recovering", "stable"]
MODES = ["rnos", "circuit_breaker", "baseline"]
PERSONAS = ["adversarial", "cautious", "mixed"]

_FALSE_REFUSAL_SCENARIOS = {"recovering", "stable"}
_CONTAINMENT_SCENARIO = "cascade"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Per-run derived metrics
# ---------------------------------------------------------------------------


def _enrich(row: dict[str, Any]) -> dict[str, Any]:
    executions = max(row.get("tool_executions", 0), 1)
    failures = row.get("tool_failures", 0)
    row["per_execution_failure_rate"] = failures / executions
    refused = row.get("final_state") in ("refused", "degrade_exhausted")
    row["refused_bool"] = refused
    # first_intervention within 10 steps for cascade containment
    fis = row.get("first_intervention_step")
    row["contained_by_10"] = (
        row.get("scenario") == _CONTAINMENT_SCENARIO
        and refused
        and fis is not None
        and fis <= 10
    )
    return row


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(vals, p))


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _scenario_table(rows: list[dict[str, Any]], scenario: str, mode: str, persona: str) -> dict[str, Any]:
    subset = [r for r in rows if r["scenario"] == scenario and r["mode"] == mode and r["persona"] == persona]
    if not subset:
        return {}

    total_steps_vals = [r.get("total_steps", 0) for r in subset]
    tool_failures_vals = [r.get("tool_failures", 0) for r in subset]
    frate_vals = [r.get("per_execution_failure_rate", 0.0) for r in subset]

    refused_count = sum(1 for r in subset if r.get("refused_bool"))
    false_refusal_rate = refused_count / len(subset) if scenario in _FALSE_REFUSAL_SCENARIOS else None

    cascade_rows = [r for r in subset if r.get("scenario") == _CONTAINMENT_SCENARIO]
    missed_containment = sum(1 for r in cascade_rows if not r.get("contained_by_10")) if cascade_rows else None
    missed_containment_rate = missed_containment / len(cascade_rows) if cascade_rows else None

    result = {
        "n": len(subset),
        "total_steps": _stats(total_steps_vals),
        "tool_failures": _stats(tool_failures_vals),
        "per_execution_failure_rate": _stats(frate_vals),
    }
    if false_refusal_rate is not None:
        result["false_refusal_rate"] = false_refusal_rate
    if missed_containment_rate is not None:
        result["missed_containment_rate"] = missed_containment_rate
    return result


def _compute_savings(rows: list[dict[str, Any]], scenario: str, persona: str) -> list[float]:
    """compute_saved = 1 - rnos_steps / baseline_steps, per seed."""
    rnos_by_seed = {
        r["seed"]: r["total_steps"]
        for r in rows
        if r["scenario"] == scenario and r["mode"] == "rnos" and r["persona"] == persona
    }
    baseline_by_seed = {
        r["seed"]: r["total_steps"]
        for r in rows
        if r["scenario"] == scenario and r["mode"] == "baseline" and r["persona"] == persona
    }
    savings = []
    for seed in rnos_by_seed:
        if seed in baseline_by_seed and baseline_by_seed[seed] > 0:
            savings.append(1.0 - rnos_by_seed[seed] / baseline_by_seed[seed])
    return savings


def _find_gate_breaks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identify RNOS false refusals and missed containments."""
    breaks = []
    for r in rows:
        if r.get("mode") != "rnos":
            continue
        scenario = r.get("scenario")
        if scenario in _FALSE_REFUSAL_SCENARIOS and r.get("refused_bool"):
            breaks.append({
                "type": "false_refusal",
                "scenario": scenario,
                "persona": r.get("persona"),
                "seed": r.get("seed"),
                "final_state": r.get("final_state"),
                "first_intervention_step": r.get("first_intervention_step"),
            })
        elif scenario == _CONTAINMENT_SCENARIO and not r.get("contained_by_10"):
            breaks.append({
                "type": "missed_containment",
                "scenario": scenario,
                "persona": r.get("persona"),
                "seed": r.get("seed"),
                "final_state": r.get("final_state"),
                "first_intervention_step": r.get("first_intervention_step"),
            })
    return breaks


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):  # nan check
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _stats_row(label: str, s: dict[str, float]) -> str:
    return (
        f"| {label} | {_fmt(s['mean'])} | {_fmt(s['std'])} "
        f"| {_fmt(s['p10'])} | {_fmt(s['p50'])} | {_fmt(s['p90'])} |"
    )


def _render_markdown(
    rows: list[dict[str, Any]],
    tag: str,
    breaks: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# RNOS Evaluation Report — tag: `{tag}`\n")
    lines.append(f"*{len(rows)} runs across {len(set(r['seed'] for r in rows))} seeds, "
                 f"{len(set(r['scenario'] for r in rows))} scenarios, "
                 f"{len(set(r['mode'] for r in rows))} modes, "
                 f"{len(set(r['persona'] for r in rows))} personas.*\n")

    for scenario in SCENARIOS:
        lines.append(f"\n## Scenario: `{scenario}`\n")
        for mode in MODES:
            lines.append(f"\n### Mode: `{mode}`\n")
            lines.append("| Persona | Metric | mean | std | p10 | p50 | p90 |")
            lines.append("|---------|--------|------|-----|-----|-----|-----|")
            for persona in PERSONAS:
                t = _scenario_table(rows, scenario, mode, persona)
                if not t:
                    continue
                n = t.get("n", 0)
                lines.append(_stats_row(f"{persona} (n={n}) total_steps", t["total_steps"]))
                lines.append(_stats_row(f"{persona} tool_failures", t["tool_failures"]))
                lines.append(_stats_row(f"{persona} failure_rate/exec", t["per_execution_failure_rate"]))
                if "false_refusal_rate" in t:
                    lines.append(f"| {persona} | false_refusal_rate | {_fmt(t['false_refusal_rate'])} | — | — | — | — |")
                if "missed_containment_rate" in t:
                    lines.append(f"| {persona} | missed_containment_rate | {_fmt(t['missed_containment_rate'])} | — | — | — | — |")

            # Compute savings (RNOS only)
            if mode == "rnos":
                for persona in PERSONAS:
                    savings = _compute_savings(rows, scenario, persona)
                    if savings:
                        s = _stats(savings)
                        lines.append(_stats_row(f"{persona} compute_saved", s))

    # --- Where the gate breaks ---
    lines.append("\n## Where the gate breaks\n")
    false_refusals = [b for b in breaks if b["type"] == "false_refusal"]
    missed = [b for b in breaks if b["type"] == "missed_containment"]

    lines.append(f"**False refusals** (RNOS refused on recovering/stable): {len(false_refusals)}\n")
    if false_refusals:
        lines.append("| scenario | persona | seed | final_state | first_intervention_step |")
        lines.append("|----------|---------|------|-------------|------------------------|")
        for b in sorted(false_refusals, key=lambda x: (x["scenario"], x["persona"], x["seed"])):
            lines.append(
                f"| {b['scenario']} | {b['persona']} | {b['seed']} "
                f"| {b['final_state']} | {b['first_intervention_step']} |"
            )

    lines.append(f"\n**Missed containments** (RNOS did not refuse cascade by step 10): {len(missed)}\n")
    if missed:
        lines.append("| scenario | persona | seed | final_state | first_intervention_step |")
        lines.append("|----------|---------|------|-------------|------------------------|")
        for b in sorted(missed, key=lambda x: (x["persona"], x["seed"])):
            lines.append(
                f"| {b['scenario']} | {b['persona']} | {b['seed']} "
                f"| {b['final_state']} | {b['first_intervention_step']} |"
            )

    lines.append("\n---\n*Generated by scripts/eval_report.py*\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _make_boxplots(rows: list[dict[str, Any]], tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping charts")
        return

    # Box plots: total_steps and tool_failures per scenario, faceted by mode
    for metric, label in [("total_steps", "Total Steps"), ("tool_failures", "Tool Failures")]:
        fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(14, 5), sharey=False)
        fig.suptitle(f"{label} by Mode — all personas, seeds 0-{max(r['seed'] for r in rows)}")

        for ax, scenario in zip(axes, SCENARIOS):
            data_by_mode = {}
            for mode in MODES:
                vals = [
                    r.get(metric, 0)
                    for r in rows
                    if r["scenario"] == scenario and r["mode"] == mode
                ]
                data_by_mode[mode] = vals

            ax.boxplot(
                [data_by_mode[m] for m in MODES],
                tick_labels=MODES,
                patch_artist=True,
            )
            ax.set_title(scenario)
            ax.set_ylabel(label if ax == axes[0] else "")
            ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        slug = metric.replace("_", "")
        out = RESULTS_DIR / f"boxplot_{slug}_{tag}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Chart: {out}")


def _make_entropy_progression(rows: list[dict[str, Any]], tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rnos_rows = [r for r in rows if r["mode"] == "rnos"]
    if not rnos_rows:
        return

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(14, 4), sharey=True)
    fig.suptitle("Mean ± 1σ entropy progression (RNOS mode, all personas)")

    for ax, scenario in zip(axes, SCENARIOS):
        subset = [r for r in rnos_rows if r["scenario"] == scenario]
        if not subset:
            continue

        # Determine max trace length across seeds
        max_len = max(len(r.get("entropy_trace", [])) for r in subset)
        if max_len == 0:
            continue

        # Pad shorter traces with the last value (RNOS stopped early)
        padded = []
        for r in subset:
            trace = r.get("entropy_trace", [])
            if len(trace) < max_len:
                pad_val = trace[-1] if trace else 0.0
                trace = trace + [pad_val] * (max_len - len(trace))
            padded.append(trace[:max_len])

        arr = np.array(padded, dtype=float)
        steps = np.arange(1, max_len + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        ax.plot(steps, mean, label="mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.3, label="±1σ")
        ax.axhline(3.0, color="orange", linestyle="--", linewidth=0.8, label="degrade (3.0)")
        ax.axhline(6.0, color="red", linestyle="--", linewidth=0.8, label="refuse (6.0)")
        ax.set_title(scenario)
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy" if ax == axes[0] else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    plt.tight_layout()
    out = RESULTS_DIR / f"entropy_progression_{tag}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Chart: {out}")


def _make_threshold_heatmap(sweep_rows: list[dict[str, Any]], tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not sweep_rows:
        return

    allow_vals = sorted({r["allow_max"] for r in sweep_rows})
    degrade_vals = sorted({r["degrade_max"] for r in sweep_rows})

    frr_grid = np.full((len(allow_vals), len(degrade_vals)), np.nan)
    mcr_grid = np.full((len(allow_vals), len(degrade_vals)), np.nan)

    for r in sweep_rows:
        ai = allow_vals.index(r["allow_max"])
        di = degrade_vals.index(r["degrade_max"])
        frr_grid[ai, di] = r["false_refusal_rate"]
        mcr_grid[ai, di] = r["missed_containment_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Threshold sensitivity (RNOS) — FRR vs MCR")

    for ax, grid, title, cmap in [
        (axes[0], frr_grid, "False Refusal Rate\n(recovering + stable)", "YlOrRd"),
        (axes[1], mcr_grid, "Missed Containment Rate\n(cascade, step ≤10)", "YlOrRd"),
    ]:
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(degrade_vals)))
        ax.set_xticklabels([str(v) for v in degrade_vals])
        ax.set_yticks(range(len(allow_vals)))
        ax.set_yticklabels([str(v) for v in allow_vals])
        ax.set_xlabel("degrade_max (refuse threshold)")
        ax.set_ylabel("allow_max (degrade threshold)")
        ax.set_title(title)

        # Annotate default (3.0, 6.0)
        if 3.0 in allow_vals and 6.0 in degrade_vals:
            ai = allow_vals.index(3.0)
            di = degrade_vals.index(6.0)
            ax.add_patch(plt.Rectangle((di - 0.5, ai - 0.5), 1, 1, fill=False, edgecolor="blue", lw=2))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = RESULTS_DIR / f"threshold_heatmap_{tag}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Chart: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RNOS evaluation report and charts.")
    parser.add_argument("--tag", type=str, default="", metavar="TEXT", help="Tag matching eval_harness output.")
    parser.add_argument("--no-sweep", action="store_true", help="Skip threshold heatmap even if sweep file exists.")
    args = parser.parse_args()

    tag = args.tag
    eval_path = RESULTS_DIR / f"eval_{tag}.jsonl" if tag else RESULTS_DIR / "eval_default.jsonl"
    sweep_path = RESULTS_DIR / f"sweep_{tag}.jsonl" if tag else RESULTS_DIR / "sweep_default.jsonl"

    if not eval_path.exists():
        print(f"Error: {eval_path} not found.  Run eval_harness.py first.")
        sys.exit(1)

    print(f"Loading {eval_path} …")
    rows = [_enrich(r) for r in _load_jsonl(eval_path)]
    print(f"  {len(rows)} rows loaded")

    breaks = _find_gate_breaks(rows)

    md = _render_markdown(rows, tag, breaks)
    md_path = RESULTS_DIR / f"eval_report_{tag}.md" if tag else RESULTS_DIR / "eval_report_default.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Report: {md_path}")

    _make_boxplots(rows, tag)
    _make_entropy_progression(rows, tag)

    if not args.no_sweep and sweep_path.exists():
        print(f"Loading {sweep_path} …")
        sweep_rows = _load_jsonl(sweep_path)
        print(f"  {len(sweep_rows)} sweep rows")
        _make_threshold_heatmap(sweep_rows, tag)
    elif not args.no_sweep:
        print(f"Sweep file {sweep_path} not found — skipping heatmap. Run threshold_sweep.py first.")

    print("Done.")


if __name__ == "__main__":
    main()
