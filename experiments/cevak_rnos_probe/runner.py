"""Experiment runner: 1000 seeds × 5 scenarios.

Usage:
    python -m experiments.cevak_rnos_probe.runner
    python experiments/cevak_rnos_probe/runner.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

from .cevak_eval import evaluate as cevak_evaluate
from .metrics import ScenarioStats, rnos_applicable, cevak_applicable
from .rnos_eval import evaluate as rnos_evaluate
from .scenario_generator import ALL_GENERATORS

N_SEEDS = 1000
N_STEPS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    if d == 0:
        return "N/A"
    return f"{100 * n / d:.1f}%"


def _acc_str(acc: float | None) -> str:
    if acc is None:
        return "not applicable"
    return f"{acc * 100:.1f}%"


def _print_scenario(stats: ScenarioStats) -> None:
    n = stats.n_runs
    agreed = stats.agreement_count
    disagreed = n - agreed
    print(f"\nScenario: {stats.scenario:<22}  |  Ground truth: {stats.ground_truth}  |  Runs: {n}")
    print(f"  Agreement: {_pct(agreed, n):<6}  |  Disagreement: {_pct(disagreed, n)}")

    rnos_acc = stats.rnos_accuracy()
    if rnos_acc is None:
        print(f"  RNOS  Accuracy: not applicable (not designed to detect {stats.ground_truth})")
    else:
        print(f"  RNOS  Accuracy: {rnos_acc * 100:.1f}%  ({stats.rnos_correct_count}/{stats.n_rnos_applicable} correct)")

    cevak_acc = stats.cevak_accuracy()
    if cevak_acc is None:
        print(f"  CEVAK Accuracy: not applicable (not designed to detect {stats.ground_truth})")
    else:
        evasion_note = ""
        if stats.ground_truth == "EVASION":
            evasion_note = f"  (ADE fired on {stats.cevak_evasion_count}/{n} traces)"
        print(f"  CEVAK Accuracy: {cevak_acc * 100:.1f}%  ({stats.cevak_correct_count}/{stats.n_cevak_applicable} correct){evasion_note}")

    print("  Action pair breakdown (RNOS:CEVAK):")
    for pair, count in sorted(stats.action_pair_counts.items(), key=lambda x: -x[1]):
        print(f"    {pair:<22}  {count:>5}  ({_pct(count, n)})")


def _print_aggregate(all_stats: list[ScenarioStats]) -> None:
    print("\n" + "=" * 72)
    print("AGGREGATE SUMMARY")
    print("=" * 72)
    header = f"{'Scenario':<24}  {'GT':<8}  {'RNOS Acc':>10}  {'CEVAK Acc':>10}  {'Agreement':>10}"
    print(header)
    print("-" * 72)
    for s in all_stats:
        rnos_a = s.rnos_accuracy()
        cevak_a = s.cevak_accuracy()
        rnos_str = f"{rnos_a*100:.1f}%" if rnos_a is not None else "N/A"
        cevak_str = f"{cevak_a*100:.1f}%" if cevak_a is not None else "N/A"
        agree_str = f"{s.agreement_rate()*100:.1f}%"
        print(f"{s.scenario:<24}  {s.ground_truth:<8}  {rnos_str:>10}  {cevak_str:>10}  {agree_str:>10}")
    print("=" * 72)


def _generate_conclusion(all_stats: list[ScenarioStats]) -> None:
    print("\nCONCLUSION")
    print("-" * 72)
    lines: list[str] = []
    both_failed: list[str] = []

    for s in all_stats:
        ra = s.rnos_accuracy()
        ca = s.cevak_accuracy()

        if s.ground_truth == "FAILURE":
            both_failed.append(s.scenario)
            continue

        if ra is not None and ca is not None:
            # Both applicable
            if ra > ca:
                lines.append(
                    f"{s.scenario} ({s.ground_truth}): RNOS higher accuracy "
                    f"({ra*100:.1f}% vs {ca*100:.1f}%)."
                )
            elif ca > ra:
                lines.append(
                    f"{s.scenario} ({s.ground_truth}): CEVAK higher accuracy "
                    f"({ca*100:.1f}% vs {ra*100:.1f}%)."
                )
            else:
                lines.append(
                    f"{s.scenario} ({s.ground_truth}): RNOS and CEVAK tied "
                    f"({ra*100:.1f}%)."
                )
        elif ra is not None:
            lines.append(
                f"{s.scenario} ({s.ground_truth}): RNOS achieved {ra*100:.1f}%; "
                f"CEVAK not applicable."
            )
        elif ca is not None:
            lines.append(
                f"{s.scenario} ({s.ground_truth}): CEVAK achieved {ca*100:.1f}%; "
                f"RNOS not applicable."
            )

    for line in lines:
        print(f"  {line}")

    if both_failed:
        for sc in both_failed:
            print(
                f"  {sc} (FAILURE): both evaluators produced 0% accuracy by "
                f"construction — no observable signal in the fields."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(n_seeds: int = N_SEEDS, n_steps: int = N_STEPS) -> list[ScenarioStats]:
    all_stats: list[ScenarioStats] = []

    for generator in ALL_GENERATORS:
        # Peek at seed=0 to get scenario/ground_truth labels
        sample = generator(0, n_steps)
        stats = ScenarioStats(
            scenario=sample.scenario,
            ground_truth=sample.ground_truth,
        )

        t0 = time.time()
        print(f"Running {sample.scenario} ({n_seeds} seeds)...", end="", flush=True)

        for seed in range(n_seeds):
            trace = generator(seed, n_steps)
            rnos_result = rnos_evaluate(trace)
            cevak_result = cevak_evaluate(trace)
            stats.record(rnos_result, cevak_result, trace.ground_truth)

        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s)")

        _print_scenario(stats)
        all_stats.append(stats)

    _print_aggregate(all_stats)
    _generate_conclusion(all_stats)
    return all_stats


if __name__ == "__main__":
    run_experiment()
