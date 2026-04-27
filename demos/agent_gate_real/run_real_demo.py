"""CLI entrypoint for the real RNOS Agent Gate repository loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_runtime.real_loop.real_runner import (
    default_real_scenarios,
    format_real_results,
    run_real_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real RNOS Agent Gate repository loop.")
    parser.add_argument(
        "--agent",
        choices=("mock", "lm"),
        default="mock",
        help="Planner implementation to use.",
    )
    parser.add_argument("--live", action="store_true", help="Publish events to the local RNOS live server.")
    args = parser.parse_args()

    comparisons = run_real_benchmark(default_real_scenarios(), agent_kind=args.agent, live=args.live)
    print("RNOS Agent Gate Real Repository Benchmark")
    print("=========================================")
    print()
    print(format_real_results(comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
