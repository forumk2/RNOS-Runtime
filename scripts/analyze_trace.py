"""Analyze execution-step traces with Runtime Coherence Metrics v0.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnos.coherence import compute_runtime_coherence, format_runtime_coherence_report

DEFAULT_TRACE_PATH = Path(__file__).resolve().parents[1] / "logs" / "rnos_trace.jsonl"


def _load_execution_steps(trace_path: Path) -> list[dict[str, object]]:
    """Load canonical execution_step events from a JSONL trace."""
    execution_steps: list[dict[str, object]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("stage") == "execution_step":
                execution_steps.append(event)
    return execution_steps


def main() -> None:
    """Entry point for coherence analysis over a stored trace."""
    parser = argparse.ArgumentParser(
        description="Compute Runtime Coherence Metrics v0.1 for an execution trace.",
    )
    parser.add_argument(
        "trace_path",
        nargs="?",
        type=Path,
        default=DEFAULT_TRACE_PATH,
        help="Path to a JSONL trace containing execution_step events.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the structured report as JSON.",
    )
    args = parser.parse_args()

    execution_steps = _load_execution_steps(args.trace_path)
    if not execution_steps:
        print(f"No execution_step events found in {args.trace_path}")
        sys.exit(1)

    report = compute_runtime_coherence(execution_steps)
    print(format_runtime_coherence_report(report))

    if args.json:
        print("\n=== Coherence JSON ===")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
