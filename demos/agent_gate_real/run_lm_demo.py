"""Run the real RNOS Agent Gate loop with an LM Studio planner."""

from __future__ import annotations

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
    comparisons = run_real_benchmark(
        default_real_scenarios(),
        agent_kind="lm",
        live=True,
    )
    print("RNOS Agent Gate LM Studio Repository Benchmark")
    print("==============================================")
    print()
    print(format_real_results(comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
