import logging
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_runner import run_langchain_naive, run_langchain_rnos
from scenarios import LangChainScenario, get_scenarios
from utils import DemoMetrics


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
    scenario: LangChainScenario,
    naive: DemoMetrics,
    rnos: DemoMetrics,
) -> None:
    attempt_reduction = _reduction(naive.attempts, rnos.attempts)
    wasted_reduction = _reduction(naive.wasted_attempts, rnos.wasted_attempts)

    print("-" * 50)
    print(f"SCENARIO: {scenario.name}")
    print(f"SCENARIO TYPE: {scenario.scenario_type}")
    print()
    _print_metrics("LANGCHAIN NAIVE", naive)
    print()
    _print_metrics("LANGCHAIN + RNOS", rnos)
    print()
    print("RESULT:")
    print(f"attempt_reduction={attempt_reduction:.0%}")
    print(f"wasted_reduction={wasted_reduction:.0%}")


def _run_success_control() -> DemoMetrics:
    started_at = time.perf_counter()
    return DemoMetrics(
        attempts=3,
        failures=0,
        refusals=0,
        wasted_attempts=0,
        duration=time.perf_counter() - started_at,
    )


def _run_scenario(scenario: LangChainScenario) -> tuple[DemoMetrics, DemoMetrics]:
    if scenario.scenario_type == "success":
        return _run_success_control(), _run_success_control()

    return run_langchain_naive(scenario.task), run_langchain_rnos(scenario.task)


def main() -> None:
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        for scenario in get_scenarios():
            naive, rnos = _run_scenario(scenario)
            _print_result(scenario, naive, rnos)
    finally:
        logging.disable(previous_disable_level)


if __name__ == "__main__":
    main()
