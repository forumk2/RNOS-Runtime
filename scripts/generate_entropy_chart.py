"""Generate an entropy and trust progression chart from an RNOS trace.

Reads ``logs/rnos_trace.jsonl`` (or a custom path) and writes
``docs/entropy_progression.png``.

Only ``assessment`` stage records are used — these carry ``entropy``,
``trust``, and ``decision`` values. Execution-step records and outcome
records are ignored.

Usage::

    python scripts/generate_entropy_chart.py
    python scripts/generate_entropy_chart.py --trace logs/rnos_trace.jsonl
    python scripts/generate_entropy_chart.py --output docs/entropy_progression.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = REPO_ROOT / "logs" / "rnos_trace.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "entropy_progression.png"

# Policy thresholds — must match rnos/policy.py defaults
_DEGRADE_THRESHOLD = 3.0
_REFUSE_THRESHOLD = 6.0

# Decision colours for scatter markers
_DECISION_COLOURS: dict[str, str] = {
    "allow": "#2ecc71",    # green
    "degrade": "#f39c12",  # amber
    "refuse": "#e74c3c",   # red
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_assessments(trace_path: Path) -> list[dict[str, Any]]:
    """Load assessment records from a JSONL trace file.

    Args:
        trace_path: Path to the JSONL file written by ``RNOSRuntime``.

    Returns:
        List of assessment dicts in file order, each containing at minimum
        ``step`` (inferred from index + 1), ``entropy``, ``trust``, and
        ``decision``.

    Raises:
        FileNotFoundError: If the trace file does not exist.
        ValueError: If the trace contains no assessment records.
    """
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {trace_path}\n"
            "Run the RNOS agent first: python scripts/run_agent.py --dry-run"
        )

    assessments: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed line {lineno}: {exc}")
                continue
            if record.get("stage") == "assessment":
                # Step number: prefer metadata["step"], fall back to sequential index
                step = record.get("metadata", {}).get("step", len(assessments) + 1)
                assessments.append({
                    "step": int(step),
                    "entropy": float(record.get("entropy", 0.0)),
                    "trust": float(record.get("trust", 0.0)),
                    "decision": str(record.get("decision", "allow")),
                })

    if not assessments:
        raise ValueError(
            f"No assessment records found in {trace_path}. "
            "Re-run the RNOS agent (not circuit-breaker or baseline mode) to populate the trace."
        )

    assessments.sort(key=lambda r: r["step"])
    return assessments


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def generate_chart(
    assessments: list[dict[str, Any]],
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    """Render the entropy and trust progression chart.

    Args:
        assessments: Sorted list of assessment dicts from :func:`load_assessments`.
        output_path: Destination PNG path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("ERROR: matplotlib is required. Install it with: pip install matplotlib")
        raise

    steps = [r["step"] for r in assessments]
    entropies = [r["entropy"] for r in assessments]
    trusts = [r["trust"] for r in assessments]
    decisions = [r["decision"] for r in assessments]

    n_steps = len(steps)
    refused_step: int | None = next(
        (r["step"] for r in assessments if r["decision"] == "refuse"), None
    )

    # Determine first/last step for shading extent
    x_min = min(steps) - 0.4
    x_max = max(steps) + 0.4
    entropy_max = max(max(entropies) * 1.15, _REFUSE_THRESHOLD * 1.15, 7.5)

    # -----------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    # --- shaded regions (entropy zones) ------------------------------------
    ax1.axhspan(0, _DEGRADE_THRESHOLD, xmin=0, xmax=1,
                color="#2ecc71", alpha=0.08, zorder=0)
    ax1.axhspan(_DEGRADE_THRESHOLD, _REFUSE_THRESHOLD, xmin=0, xmax=1,
                color="#f39c12", alpha=0.10, zorder=0)
    ax1.axhspan(_REFUSE_THRESHOLD, entropy_max, xmin=0, xmax=1,
                color="#e74c3c", alpha=0.10, zorder=0)

    # --- threshold lines ---------------------------------------------------
    ax1.axhline(_DEGRADE_THRESHOLD, color="#f39c12", linestyle="--",
                linewidth=1.2, alpha=0.8, zorder=1, label=f"DEGRADE threshold ({_DEGRADE_THRESHOLD})")
    ax1.axhline(_REFUSE_THRESHOLD, color="#e74c3c", linestyle="--",
                linewidth=1.2, alpha=0.8, zorder=1, label=f"REFUSE threshold ({_REFUSE_THRESHOLD})")

    # --- entropy line -------------------------------------------------------
    ax1.plot(steps, entropies, color="#2ecc71", linewidth=2.0,
             zorder=3, label="Entropy")

    # Scatter markers coloured by decision
    marker_colours = [_DECISION_COLOURS.get(d, "#888888") for d in decisions]
    ax1.scatter(steps, entropies, c=marker_colours, s=80,
                zorder=4, edgecolors="white", linewidths=0.8)

    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel("Entropy", fontsize=11, color="#2ecc71")
    ax1.tick_params(axis="y", labelcolor="#2ecc71")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, entropy_max)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.spines["top"].set_visible(False)

    # Light horizontal grid (entropy axis)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(False)

    # --- trust line (right axis) -------------------------------------------
    ax2 = ax1.twinx()
    ax2.plot(steps, trusts, color="#3498db", linewidth=2.0,
             linestyle="-", zorder=3, label="Trust")
    ax2.scatter(steps, trusts, c="#3498db", s=60,
                zorder=4, edgecolors="white", linewidths=0.8)
    ax2.set_ylabel("Trust", fontsize=11, color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")
    ax2.set_ylim(0, 1.15)
    ax2.spines["top"].set_visible(False)

    # --- titles -----------------------------------------------------------
    seed_guess = "?"  # We don't store seed in the trace; use placeholder
    subtitle_refused = f"refused at step {refused_step}" if refused_step else "no refusal"
    ax1.set_title(
        "RNOS Entropy & Trust Progression",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax1.text(
        0.5,
        1.01,
        f"{n_steps} steps, {subtitle_refused}",
        ha="center",
        va="bottom",
        transform=ax1.transAxes,
        fontsize=9,
        color="#555555",
    )

    # --- combined legend --------------------------------------------------
    decision_patches = [
        mpatches.Patch(color="#2ecc71", label="Decision: ALLOW"),
        mpatches.Patch(color="#f39c12", label="Decision: DEGRADE"),
        mpatches.Patch(color="#e74c3c", label="Decision: REFUSE"),
    ]
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2 + decision_patches,
        labels1 + labels2 + [p.get_label() for p in decision_patches],
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Entropy chart written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the entropy progression chart generator."""
    parser = argparse.ArgumentParser(
        description="Generate entropy/trust progression chart from an RNOS trace.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE,
        metavar="PATH",
        help=f"Path to the JSONL trace file (default: {DEFAULT_TRACE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Output PNG path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    try:
        assessments = load_assessments(args.trace)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    print(f"Loaded {len(assessments)} assessment records from {args.trace}")
    generate_chart(assessments, output_path=args.output)


if __name__ == "__main__":
    main()
