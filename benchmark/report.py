from dataclasses import dataclass

from metrics import RunMetrics


@dataclass(frozen=True)
class BenchmarkResult:
    scenario_name: str
    naive: RunMetrics
    rnos: RunMetrics


def _reduction(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return max(0.0, (before - after) / before)


def _format_bool(value: bool) -> str:
    return "success" if value else "failure"


def _print_metrics(label: str, metrics: RunMetrics) -> None:
    print(f"{label}:")
    print(f"attempts={metrics.attempts}")
    print(f"failures={metrics.failures}")
    print(f"successes={metrics.successes}")
    print(f"refusals={metrics.refusals}")
    print(f"total_steps={metrics.total_steps}")
    print(f"wasted_attempts={metrics.repeated_failures}")
    print(f"duration={metrics.duration:.4f}s")
    print(f"outcome={_format_bool(metrics.succeeded)}")


def print_report(results: list[BenchmarkResult]) -> None:
    for result in results:
        attempt_reduction = _reduction(result.naive.attempts, result.rnos.attempts)
        wasted_reduction = _reduction(
            result.naive.repeated_failures,
            result.rnos.repeated_failures,
        )

        print("-" * 50)
        print(f"SCENARIO: {result.scenario_name}")
        print()
        _print_metrics("NAIVE", result.naive)
        print()
        _print_metrics("RNOS", result.rnos)
        print()
        print("RESULT:")
        print(f"attempt_reduction={attempt_reduction:.0%}")
        print(f"wasted_reduction={wasted_reduction:.0%}")
    print("-" * 50)
