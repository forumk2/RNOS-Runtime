"""Run all three execution modes in sequence and generate a comparison report.

Usage::

    python scripts/run_comparison.py --max-steps 20 --seed 4 --tag "session-2026-04-01"
    python scripts/run_comparison.py --max-steps 20 --seed 4 --dry-run --tag "verify"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT = REPO_ROOT / "scripts" / "run_agent.py"
GENERATE_REPORT = REPO_ROOT / "scripts" / "generate_report.py"

_BANNER_INNER_WIDTH = 34  # characters between the box walls

_MODES: list[tuple[str, list[str]]] = [
    ("RNOS Mode", []),
    ("Circuit Breaker Mode", ["--circuit-breaker"]),
    ("Baseline Mode", ["--no-rnos"]),
]


def _banner(title: str) -> str:
    """Return a three-line box banner for *title*.

    Example output::

        ╔══════════════════════════════════╗
        ║  Circuit Breaker Mode            ║
        ╚══════════════════════════════════╝
    """
    inner = f"  {title:<{_BANNER_INNER_WIDTH - 2}}"
    bar = "═" * _BANNER_INNER_WIDTH
    return f"\n╔{bar}╗\n║{inner}║\n╚{bar}╝"


def _run_mode(
    mode_name: str,
    extra_flags: list[str],
    common_flags: list[str],
) -> int:
    """Run a single mode as a subprocess, streaming output to the terminal.

    Args:
        mode_name: Human-readable label printed in the banner.
        extra_flags: Mode-specific flags (e.g. ``["--circuit-breaker"]``).
        common_flags: Flags shared across all modes.

    Returns:
        The subprocess return code.
    """
    print(_banner(mode_name))
    cmd = [sys.executable, str(RUN_AGENT)] + common_flags + extra_flags
    result = subprocess.run(cmd)
    print()  # blank line after each run
    return result.returncode


def _build_common_flags(args: argparse.Namespace) -> list[str]:
    """Translate parsed CLI args into flags for ``run_agent.py``.

    Args:
        args: Parsed argument namespace from this script's parser.

    Returns:
        List of string tokens to pass to the subprocess.
    """
    flags: list[str] = [
        "--max-steps", str(args.max_steps),
        "--seed", str(args.seed),
    ]
    if args.tag:
        flags += ["--tag", args.tag]
    if args.dry_run:
        flags.append("--dry-run")
    if args.persona:
        flags += ["--persona", args.persona]
    return flags


def main() -> None:
    """Entry point: run all three modes then generate the comparison report."""
    parser = argparse.ArgumentParser(
        description="Run RNOS, circuit breaker, and baseline modes then generate a report.",
    )
    parser.add_argument("--max-steps", type=int, default=20, metavar="N",
                        help="Maximum loop steps per run (default: 20).")
    parser.add_argument("--seed", type=int, default=4, metavar="N",
                        help="Random seed forwarded to all runs (default: 4).")
    parser.add_argument("--tag", type=str, default="", metavar="TEXT",
                        help="Free-text label stored with each run for later filtering.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use the stub planner; no LM Studio required.")
    parser.add_argument("--persona",
                        choices=["adversarial", "cautious", "mixed"],
                        default="adversarial",
                        help="Planner persona forwarded to all runs (default: adversarial).")
    args = parser.parse_args()

    common_flags = _build_common_flags(args)

    exit_codes: list[int] = []
    for mode_name, extra_flags in _MODES:
        rc = _run_mode(mode_name, extra_flags, common_flags)
        exit_codes.append(rc)

    if any(rc != 0 for rc in exit_codes):
        print("WARNING: one or more runs exited with a non-zero return code.")

    # --- generate report -----------------------------------------------------
    report_cmd = [sys.executable, str(GENERATE_REPORT)]
    if args.tag:
        report_cmd += ["--tag", args.tag]
    else:
        report_cmd += ["--seed", str(args.seed)]

    print("Generating comparison report…")
    subprocess.run(report_cmd)

    results_dir = REPO_ROOT / "results"
    print(f"\nReport generated: {results_dir / 'report.md'}")
    print(f"Chart generated:  {results_dir / 'comparison_chart.png'}")


if __name__ == "__main__":
    main()
