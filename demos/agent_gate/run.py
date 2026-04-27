"""CLI entrypoint for the RNOS Agent Gate benchmark demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_runtime.runner import (
    default_agent_gate_scenarios,
    format_agent_gate_results,
    run_agent_gate_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RNOS Agent Gate benchmark demo.")
    parser.add_argument("--live", action="store_true", help="Publish events to the local RNOS live server.")
    args = parser.parse_args()

    scenarios = default_agent_gate_scenarios()
    comparisons = run_agent_gate_benchmark(scenarios, live=args.live)
    print("RNOS Agent Gate Benchmark")
    print("=========================")
    print()
    print(format_agent_gate_results(comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
