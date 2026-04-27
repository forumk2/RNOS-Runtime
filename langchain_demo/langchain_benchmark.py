import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_runner import run_langchain_naive, run_langchain_rnos
from utils import DemoMetrics


SCENARIOS = [
    ("syntax_error_fix", "fix broken python code with syntax errors"),
    ("terrain_failure", "build a terrain system"),
    ("invalid_python_repair", "write invalid python and fix it"),
]


def _reduction(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return max(0.0, (before - after) / before)


def _print_metrics(label: str, metrics: DemoMetrics) -> None:
    print(f"{label}:")
    print(f"attempts={metrics.attempts}")
    print(f"failures={metrics.failures}")
    print(f"refusals={metrics.refusals}")
    print(f"wasted_attempts={metrics.wasted_attempts}")
    print(f"duration={metrics.duration:.4f}s")


def _print_result(
    scenario_name: str,
    naive: DemoMetrics,
    rnos: DemoMetrics,
) -> None:
    attempt_reduction = _reduction(naive.attempts, rnos.attempts)
    wasted_reduction = _reduction(naive.wasted_attempts, rnos.wasted_attempts)

    print("-" * 50)
    print(f"SCENARIO: {scenario_name}")
    print()
    _print_metrics("LANGCHAIN NAIVE", naive)
    print()
    _print_metrics("LANGCHAIN + RNOS", rnos)
    print()
    print("RESULT:")
    print(f"attempt_reduction={attempt_reduction:.0%}")
    print(f"wasted_reduction={wasted_reduction:.0%}")


def main() -> None:
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        for scenario_name, task in SCENARIOS:
            naive = run_langchain_naive(task)
            rnos = run_langchain_rnos(task)
            _print_result(scenario_name, naive, rnos)
    finally:
        logging.disable(previous_disable_level)


if __name__ == "__main__":
    main()
