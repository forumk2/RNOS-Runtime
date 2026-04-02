"""Generate a Markdown comparison report and PNG chart from collected run data.

Reads ``results/runs.jsonl`` and writes:

* ``results/report.md``  — Markdown comparison table + run history
* ``results/comparison_chart.png``  — Grouped bar chart (skipped with ``--no-chart``)

Usage::

    python scripts/generate_report.py
    python scripts/generate_report.py --tag "session-2026-04-01"
    python scripts/generate_report.py --seed 4
    python scripts/generate_report.py --no-chart
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RUNS_PATH = RESULTS_DIR / "runs.jsonl"
REPORT_PATH = RESULTS_DIR / "report.md"
CHART_PATH = RESULTS_DIR / "comparison_chart.png"

_MODE_ORDER = ("rnos", "circuit_breaker", "baseline")
_MODE_LABELS = {
    "rnos": "RNOS",
    "circuit_breaker": "Circuit Breaker",
    "baseline": "Baseline",
}

# Chart colours per mode
_COLOURS = {
    "rnos": "#2ecc71",
    "circuit_breaker": "#f39c12",
    "baseline": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_runs(path: Path = RUNS_PATH) -> list[dict[str, Any]]:
    """Load all run records from *path*.

    Args:
        path: Path to the JSONL file produced by ``run_agent.py``.

    Returns:
        List of run dicts, in file order (oldest first).

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No results file found at {path}. "
            "Run the agent first: python scripts/run_agent.py"
        )
    runs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed line {lineno}: {exc}")
    return runs


def filter_runs(
    runs: list[dict[str, Any]],
    *,
    tag: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return the subset of *runs* matching *tag* and/or *seed*.

    Args:
        runs: All run records.
        tag: If given, keep only records where ``run["tag"] == tag``.
        seed: If given, keep only records where ``run["seed"] == seed``.

    Returns:
        Filtered list; original order preserved.
    """
    result = runs
    if tag is not None:
        result = [r for r in result if r.get("tag") == tag]
    if seed is not None:
        result = [r for r in result if r.get("seed") == seed]
    return result


def latest_per_mode(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the most-recent run for each mode.

    Args:
        runs: Run records, may be unordered.

    Returns:
        Dict mapping mode name → latest run dict.  Missing modes are absent.
    """
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        mode = run.get("mode", "")
        ts = run.get("timestamp", "")
        if mode not in latest or ts > latest[mode].get("timestamp", ""):
            latest[mode] = run
    return latest


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _blocked(run: dict[str, Any]) -> int:
    """Return the number of non-executing steps for *run*.

    For circuit_breaker runs this is ``total_blocked_steps``.
    For RNOS/baseline it is inferred as ``total_loop_steps - total_steps_executed``.
    """
    if run.get("mode") == "circuit_breaker":
        return int(run.get("total_blocked_steps", 0))
    return max(
        0,
        int(run.get("total_loop_steps", 0)) - int(run.get("total_steps_executed", 0)),
    )


def _successes(run: dict[str, Any]) -> int:
    """Return the number of successful tool executions."""
    executed = int(run.get("total_steps_executed", 0))
    failures = int(run.get("total_tool_failures", 0))
    return max(0, executed - failures)


def _wasted_planner_calls(run: dict[str, Any]) -> int:
    """Return the number of planner calls that did not produce a success.

    Calculated as ``total_loop_steps - successes``.
    """
    return max(0, int(run.get("total_loop_steps", 0)) - _successes(run))


def _pct_reduction(baseline_val: float, mode_val: float) -> str:
    """Return a formatted reduction percentage string.

    Positive means *mode_val* is lower than *baseline_val* (an improvement).
    Returns ``"N/A"`` if *baseline_val* is zero.
    """
    if baseline_val == 0:
        return "N/A"
    pct = (baseline_val - mode_val) / baseline_val * 100
    return f"{pct:+.1f}%"


def _fmt(value: object, missing: str = "no data") -> str:
    """Return *value* as a string, or *missing* if None/absent."""
    if value is None:
        return missing
    return str(value)


def _first_intervention(run: dict[str, Any]) -> str:
    """Human-readable string for the first intervention event."""
    step = run.get("first_intervention_step")
    kind = run.get("first_intervention_type")
    if step is None:
        return "N/A"
    return f"step {step} ({kind})"


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def generate_chart(
    mode_data: dict[str, dict[str, Any]],
    output_path: Path = CHART_PATH,
) -> None:
    """Render a grouped bar chart comparing the three modes.

    Args:
        mode_data: Dict mapping mode name → latest run dict.
        output_path: Destination PNG path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend, works without a display
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib not installed — skipping chart generation.")
        return

    categories = ["Total Steps", "Tool Executions", "Tool Failures", "Planner Calls (wasted)"]
    modes_present = [m for m in _MODE_ORDER if m in mode_data]

    if not modes_present:
        print("WARNING: no run data available — skipping chart generation.")
        return

    # Build value arrays per mode (None → 0 with annotation)
    values: dict[str, list[float]] = {}
    for mode in modes_present:
        run = mode_data[mode]
        values[mode] = [
            float(run.get("total_loop_steps", 0)),
            float(run.get("total_steps_executed", 0)),
            float(run.get("total_tool_failures", 0)),
            float(_wasted_planner_calls(run)),
        ]

    x = np.arange(len(categories))
    n_modes = len(modes_present)
    bar_width = 0.25
    offsets = np.linspace(-(n_modes - 1) / 2, (n_modes - 1) / 2, n_modes) * bar_width

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for offset, mode in zip(offsets, modes_present):
        bars = ax.bar(
            x + offset,
            values[mode],
            width=bar_width,
            label=_MODE_LABELS[mode],
            color=_COLOURS[mode],
            zorder=3,
        )
        # Value labels on top of each bar
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.15,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=9,
                zorder=4,
            )

    # --- titles & labels ---
    # Pull metadata from any available run for subtitle
    sample_run = next(iter(mode_data.values()))
    seed = sample_run.get("seed", "?")
    max_steps = sample_run.get("max_steps", "?")
    persona = sample_run.get("persona", "?")
    dry_run_label = "dry-run" if sample_run.get("dry_run") else "live"

    ax.set_title(
        f"RNOS-Runtime: Three-Way Comparison (seed={seed})",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        1.01,
        f"max_steps={max_steps}, persona={persona}, {dry_run_label}",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Grid on y-axis only
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _comparison_table(mode_data: dict[str, dict[str, Any]]) -> str:
    """Return the Latest Comparison markdown table."""
    header = (
        "| Metric | RNOS | Circuit Breaker | Baseline |\n"
        "|---|---|---|---|\n"
    )

    def col(mode: str, key: str, default: str = "no data") -> str:
        if mode not in mode_data:
            return "no data"
        return _fmt(mode_data[mode].get(key), default)

    def col_fn(mode: str, fn: Any, default: str = "no data") -> str:
        if mode not in mode_data:
            return "no data"
        return _fmt(fn(mode_data[mode]), default)

    rows = [
        f"| Mode | {col('rnos','mode')} | {col('circuit_breaker','mode')} | {col('baseline','mode')} |",
        f"| Total Steps | {col('rnos','total_loop_steps')} | {col('circuit_breaker','total_loop_steps')} | {col('baseline','total_loop_steps')} |",
        f"| Tool Executions | {col('rnos','total_steps_executed')} | {col('circuit_breaker','total_steps_executed')} | {col('baseline','total_steps_executed')} |",
        f"| Tool Failures | {col('rnos','total_tool_failures')} | {col('circuit_breaker','total_tool_failures')} | {col('baseline','total_tool_failures')} |",
        f"| Blocked/Refused Steps | {col_fn('rnos',_blocked)} | {col_fn('circuit_breaker',_blocked)} | {col_fn('baseline',_blocked)} |",
        f"| Duration (seconds) | {col('rnos','duration_seconds')} | {col('circuit_breaker','duration_seconds')} | {col('baseline','duration_seconds')} |",
        f"| Planner Compute (ms) | {col('rnos','planner_latency_total_ms')} | {col('circuit_breaker','planner_latency_total_ms')} | {col('baseline','planner_latency_total_ms')} |",
        f"| First Intervention | {col_fn('rnos',_first_intervention)} | {col_fn('circuit_breaker',_first_intervention)} | N/A |",
        f"| Final State | {col('rnos','final_state')} | {col('circuit_breaker','final_state')} | {col('baseline','final_state')} |",
        f"| Seed | {col('rnos','seed')} | {col('circuit_breaker','seed')} | {col('baseline','seed')} |",
        f"| Persona | {col('rnos','persona')} | {col('circuit_breaker','persona')} | {col('baseline','persona')} |",
        f"| Dry Run | {col('rnos','dry_run')} | {col('circuit_breaker','dry_run')} | {col('baseline','dry_run')} |",
    ]
    return header + "\n".join(rows)


def _reduction_table(mode_data: dict[str, dict[str, Any]]) -> str:
    """Return the Reduction vs Baseline markdown table."""
    header = (
        "| Metric | RNOS | Circuit Breaker |\n"
        "|---|---|---|\n"
    )

    if "baseline" not in mode_data:
        return header + "| (baseline data missing — cannot compute reductions) | N/A | N/A |"

    base = mode_data["baseline"]
    b_steps = float(base.get("total_loop_steps", 0))
    b_exec = float(base.get("total_steps_executed", 0))
    b_fail = float(base.get("total_tool_failures", 0))
    b_plan = float(base.get("planner_latency_total_ms", 0))

    def row(label: str, key: str, base_val: float) -> str:
        rnos_val = float(mode_data["rnos"].get(key, 0)) if "rnos" in mode_data else None
        cb_val = float(mode_data["circuit_breaker"].get(key, 0)) if "circuit_breaker" in mode_data else None
        r_rnos = _pct_reduction(base_val, rnos_val) if rnos_val is not None else "no data"
        r_cb = _pct_reduction(base_val, cb_val) if cb_val is not None else "no data"
        return f"| {label} | {r_rnos} | {r_cb} |"

    rows = [
        row("Step Reduction", "total_loop_steps", b_steps),
        row("Execution Reduction", "total_steps_executed", b_exec),
        row("Failure Reduction", "total_tool_failures", b_fail),
        row("Planner Compute Saved", "planner_latency_total_ms", b_plan),
    ]
    return header + "\n".join(rows)


def _key_findings(mode_data: dict[str, dict[str, Any]]) -> str:
    """Return dynamically generated Key Findings bullet list."""
    bullets: list[str] = []

    base = mode_data.get("baseline")
    rnos = mode_data.get("rnos")
    cb = mode_data.get("circuit_breaker")

    if base and rnos:
        b_steps = float(base.get("total_loop_steps", 0))
        b_fail = float(base.get("total_tool_failures", 0))
        r_steps = float(rnos.get("total_loop_steps", 0))
        r_fail = float(rnos.get("total_tool_failures", 0))
        if b_steps > 0:
            pct_steps = (b_steps - r_steps) / b_steps * 100
            bullets.append(
                f"RNOS reduced total steps by {pct_steps:.1f}% compared to baseline "
                f"({int(r_steps)} vs {int(b_steps)})."
            )
        if b_fail > 0:
            pct_fail = (b_fail - r_fail) / b_fail * 100
            bullets.append(
                f"RNOS reduced tool failures by {pct_fail:.1f}% compared to baseline "
                f"({int(r_fail)} vs {int(b_fail)})."
            )
        elif b_fail == 0:
            bullets.append("Baseline had zero tool failures in this run — no failure reduction to report.")

    if base and cb:
        b_exec = float(base.get("total_steps_executed", 0))
        b_steps = float(base.get("total_loop_steps", 0))
        cb_exec = float(cb.get("total_steps_executed", 0))
        cb_steps = float(cb.get("total_loop_steps", 0))
        if b_exec > 0:
            pct_exec = (b_exec - cb_exec) / b_exec * 100
            bullets.append(
                f"Circuit breaker reduced tool executions by {pct_exec:.1f}% "
                f"({int(cb_exec)} vs {int(b_exec)})."
            )
        if b_steps > 0:
            pct_steps_cb = (b_steps - cb_steps) / b_steps * 100
            bullets.append(
                f"Circuit breaker reduced total steps by {pct_steps_cb:.1f}% "
                f"({int(cb_steps)} vs {int(b_steps)})."
            )

    if rnos:
        fi_step = rnos.get("first_intervention_step")
        fi_type = rnos.get("first_intervention_type")
        if fi_step is not None:
            bullets.append(
                f"RNOS first intervened at step {fi_step} with a '{fi_type}' decision."
            )

    if cb:
        cb_blocked = int(cb.get("total_blocked_steps", 0))
        if cb_blocked > 0:
            bullets.append(
                f"Circuit breaker blocked {cb_blocked} step(s) during cooldown windows."
            )

    if not bullets:
        bullets.append("Insufficient data across modes to generate findings.")

    return "\n".join(f"- {b}" for b in bullets)


def _run_history_table(runs: list[dict[str, Any]], limit: int = 20) -> str:
    """Return a markdown table of the last *limit* runs, newest first."""
    recent = sorted(runs, key=lambda r: r.get("timestamp", ""), reverse=True)[:limit]
    if not recent:
        return "_No runs recorded yet._"

    header = (
        "| Timestamp | Mode | Seed | Steps | Executions | Failures | Tag | Dry Run |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in recent:
        ts = r.get("timestamp", "")[:19].replace("T", " ")
        rows.append(
            f"| {ts} | {r.get('mode','?')} | {r.get('seed','?')} "
            f"| {r.get('total_loop_steps','?')} | {r.get('total_steps_executed','?')} "
            f"| {r.get('total_tool_failures','?')} | {r.get('tag','')} | {r.get('dry_run','?')} |"
        )
    return header + "\n".join(rows)


# ---------------------------------------------------------------------------
# Top-level report builder
# ---------------------------------------------------------------------------


def build_report(
    runs: list[dict[str, Any]],
    *,
    tag: str | None,
    seed: int | None,
    generate_chart_flag: bool,
) -> str:
    """Build and write the Markdown report (and optionally the chart).

    Args:
        runs: All run records (unfiltered).
        tag: Tag filter for the latest-comparison section.
        seed: Seed filter used when no tag is given.
        generate_chart_flag: When True, also write the PNG chart.

    Returns:
        The rendered Markdown string.
    """
    # Filter for the latest-comparison section
    filtered = filter_runs(runs, tag=tag, seed=seed)
    mode_data = latest_per_mode(filtered)

    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    if tag:
        filter_label = f"tag={tag}"
    elif seed is not None:
        filter_label = f"seed={seed}"
    else:
        filter_label = "all runs"

    comparison_table = _comparison_table(mode_data)
    reduction_table = _reduction_table(mode_data)
    findings = _key_findings(mode_data)
    history_table = _run_history_table(runs)

    if generate_chart_flag and mode_data:
        generate_chart(mode_data)
        chart_line = "\n![Comparison Chart](comparison_chart.png)\n"
    else:
        chart_line = ""

    report = f"""# RNOS-Runtime Comparison Report

Generated: {now}
Filter: {filter_label}

## Latest Comparison

{comparison_table}

## Reduction vs Baseline

{reduction_table}

## Key Findings

{findings}
{chart_line}
## Run History

{history_table}
"""
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the report generator."""
    parser = argparse.ArgumentParser(
        description="Generate a comparison report from results/runs.jsonl.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        metavar="TEXT",
        help="Filter the latest-comparison section to runs with this tag.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Filter the latest-comparison section to runs with this seed.",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip PNG chart generation.",
    )
    args = parser.parse_args()

    try:
        runs = load_runs()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    report_md = build_report(
        runs,
        tag=args.tag,
        seed=args.seed,
        generate_chart_flag=not args.no_chart,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    if not args.no_chart:
        if CHART_PATH.exists():
            print(f"Chart written to {CHART_PATH}")


if __name__ == "__main__":
    main()
