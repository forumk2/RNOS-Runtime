"""CLI entrypoint for the RNOS Agent Gate benchmark demo."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_runtime.runner import format_agent_gate_results, run_agent_gate_benchmark
from demos.agent_gate.scenarios import destructive_command, drift_scenario, failure_loop


def main() -> int:
    scenarios = [
        failure_loop.create_scenario(),
        destructive_command.create_scenario(),
        drift_scenario.create_scenario(),
    ]
    comparisons = run_agent_gate_benchmark(scenarios)
    print("RNOS Agent Gate Benchmark")
    print("=========================")
    print()
    print(format_agent_gate_results(comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
