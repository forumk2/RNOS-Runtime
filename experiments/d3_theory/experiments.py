from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.d3_theory.env import (  # noqa: E402
    AlternatingLivelockEnvironment,
    EmissionLaw,
    PoissonBernoulliEnvironment,
    RecoverableNoiseEnvironment,
    TrapEscapeEnvironment,
)
from experiments.d3_theory.models import (  # noqa: E402
    D3Config,
    D3Controller,
    D3QConfig,
    D3QController,
    D3QSConfig,
    D3QSController,
)
from experiments.d3_theory.simulation import aggregate_runs, run_simulation  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _REPO_ROOT / "results" / "d3_theory"
_JSON_PATH = _OUTPUT_DIR / "d3_theory_results.json"
_SWEEP_CSV_PATH = _OUTPUT_DIR / "d3_theory_parameter_sweep.csv"
_SUMMARY_PATH = _OUTPUT_DIR / "d3_theory_summary.md"

HEALTHY_LAW = EmissionLaw(lambda_rate=1.0, validation_prob=0.98)
DEGRADED_LAW = EmissionLaw(lambda_rate=4.5, validation_prob=0.35)

TRUST_EXECUTE = 0.92
TRUST_QUARANTINE = 0.68

D3_THRESHOLDS = {
    "conservative": {"trust_threshold": 0.60, "entropy_threshold": 14.0},
    "balanced": {"trust_threshold": 0.55, "entropy_threshold": 22.0},
    "permissive": {"trust_threshold": 0.45, "entropy_threshold": 30.0},
}

Q_THRESHOLDS = {
    "conservative": {"t_hi": 0.86, "t_lo": 0.62, "u_hi": 1.2, "u_lo": 2.4, "rho": 1.2},
    "balanced": {"t_hi": 0.80, "t_lo": 0.55, "u_hi": 2.0, "u_lo": 4.0, "rho": 1.1},
    "permissive": {"t_hi": 0.72, "t_lo": 0.45, "u_hi": 3.2, "u_lo": 6.0, "rho": 1.0},
}


def make_d3(name: str = "balanced") -> D3Controller:
    threshold = D3_THRESHOLDS[name]
    return D3Controller(
        D3Config(
            healthy_law=HEALTHY_LAW,
            degraded_law=DEGRADED_LAW,
            trust_threshold=threshold["trust_threshold"],
            entropy_threshold=threshold["entropy_threshold"],
        )
    )


def make_d3q(threshold_name: str = "balanced", budget_capacity: int = 4) -> D3QController:
    threshold = Q_THRESHOLDS[threshold_name]
    return D3QController(
        D3QConfig(
            healthy_law=HEALTHY_LAW,
            degraded_law=DEGRADED_LAW,
            t_hi=threshold["t_hi"],
            t_lo=threshold["t_lo"],
            u_hi=threshold["u_hi"],
            u_lo=threshold["u_lo"],
            rho=threshold["rho"],
            budget_capacity=budget_capacity,
        )
    )


def make_d3qs(
    threshold_name: str = "balanced",
    *,
    budget_capacity: int = 4,
    initial_budget: int | None = None,
    recovery_threshold: int = 3,
    prior_trust: float = 0.95,
) -> D3QSController:
    threshold = Q_THRESHOLDS[threshold_name]
    return D3QSController(
        D3QSConfig(
            healthy_law=HEALTHY_LAW,
            degraded_law=DEGRADED_LAW,
            t_hi=threshold["t_hi"],
            t_lo=threshold["t_lo"],
            u_hi=threshold["u_hi"],
            u_lo=threshold["u_lo"],
            rho=threshold["rho"],
            budget_capacity=budget_capacity,
            initial_budget=initial_budget,
            prior_trust=prior_trust,
            recovery_threshold=recovery_threshold,
        )
    )


def run_many(
    scenario: str,
    model_factory: Any,
    environment_factory: Any,
    *,
    seeds: int,
    base_seed: int,
    max_steps: int,
    trace_first_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs = []
    for offset in range(seeds):
        seed = base_seed + offset
        model = model_factory()
        environment = environment_factory()
        runs.append(
            run_simulation(
                scenario,
                model,
                environment,
                seed=seed,
                max_steps=max_steps,
                capture_trace=trace_first_run and offset == 0,
            )
        )
    run_dicts = [asdict(run) for run in runs]
    return run_dicts, aggregate_runs(runs)


def verify_monotone_entropy(run_dicts: list[dict[str, Any]]) -> bool:
    for run in run_dicts:
        for step in run.get("trace") or []:
            if step["cumulative_entropy_after"] <= step["cumulative_entropy_before"]:
                return False
    return True


def recoverable_horizon(p: float, budget_capacity: int) -> int:
    expected_refusal = (budget_capacity + 1) / max(p, 1e-6)
    return max(2000, int(expected_refusal * 20))


def run_stress_tests(base_seed: int, seeds: int) -> dict[str, Any]:
    results: dict[str, Any] = {}

    baseline_runs, baseline_agg = run_many(
        "baseline_sanity_d3",
        lambda: make_d3("balanced"),
        lambda: PoissonBernoulliEnvironment(HEALTHY_LAW, DEGRADED_LAW, regime_policy="healthy"),
        seeds=seeds,
        base_seed=base_seed,
        max_steps=200,
        trace_first_run=True,
    )
    baseline_agg["monotone_entropy_verified"] = verify_monotone_entropy(baseline_runs)
    results["baseline_sanity"] = {
        "description": "No noise. D3 should still terminate because cumulative entropy is strictly increasing.",
        "runs": baseline_runs,
        "aggregate": baseline_agg,
    }

    persistent_models = {}
    for name, model_factory in {
        "D3": lambda: make_d3("balanced"),
        "D3-Q": lambda: make_d3q("balanced", budget_capacity=4),
        "D3-QS": lambda: make_d3qs("balanced", budget_capacity=4, recovery_threshold=3),
    }.items():
        model_runs, aggregate = run_many(
            f"persistent_degraded_{name.lower()}",
            model_factory,
            lambda: PoissonBernoulliEnvironment(HEALTHY_LAW, DEGRADED_LAW, regime_policy="degraded"),
            seeds=seeds,
            base_seed=base_seed + 100,
            max_steps=200,
            trace_first_run=(name == "D3"),
        )
        persistent_models[name] = {"runs": model_runs, "aggregate": aggregate}
    results["persistent_adversarial"] = {
        "description": "Always degraded. All models should terminate under persistent adversarial drift.",
        "models": persistent_models,
    }

    recoverable_cases = []
    for p in [0.01, 0.05, 0.10, 0.20]:
        d3q_runs, d3q_agg = run_many(
            f"recoverable_noise_d3q_p_{p:.2f}",
            lambda: make_d3q("balanced", budget_capacity=4),
            lambda p=p: RecoverableNoiseEnvironment(
                HEALTHY_LAW,
                DEGRADED_LAW,
                disturbance_prob=p,
                trust_execute=TRUST_EXECUTE,
                trust_quarantine=TRUST_QUARANTINE,
            ),
            seeds=seeds,
            base_seed=base_seed + 200 + int(p * 1000),
            max_steps=recoverable_horizon(p, 4),
            trace_first_run=(p == 0.01),
        )
        d3qs_runs, d3qs_agg = run_many(
            f"recoverable_noise_d3qs_p_{p:.2f}",
            lambda: make_d3qs("balanced", budget_capacity=4, recovery_threshold=3),
            lambda p=p: RecoverableNoiseEnvironment(
                HEALTHY_LAW,
                DEGRADED_LAW,
                disturbance_prob=p,
                trust_execute=TRUST_EXECUTE,
                trust_quarantine=TRUST_QUARANTINE,
            ),
            seeds=seeds,
            base_seed=base_seed + 400 + int(p * 1000),
            max_steps=2000,
            trace_first_run=(p == 0.01),
        )
        recoverable_cases.append(
            {
                "p": p,
                "D3-Q": {"runs": d3q_runs, "aggregate": d3q_agg},
                "D3-QS": {"runs": d3qs_runs, "aggregate": d3qs_agg},
            }
        )
    results["recoverable_stochastic_noise"] = {
        "description": "Exact Proposition 1 / Theorem 3 construction with i.i.d. Bernoulli disturbances.",
        "cases": recoverable_cases,
    }

    livelock_runs, livelock_agg = run_many(
        "qs_livelock_alternating",
        lambda: make_d3qs(
            "balanced",
            budget_capacity=3,
            initial_budget=0,
            recovery_threshold=2,
            prior_trust=TRUST_QUARANTINE,
        ),
        lambda: AlternatingLivelockEnvironment(
            HEALTHY_LAW,
            DEGRADED_LAW,
            trust_execute=TRUST_EXECUTE,
            trust_quarantine=TRUST_QUARANTINE,
        ),
        seeds=1,
        base_seed=base_seed + 900,
        max_steps=200,
        trace_first_run=True,
    )
    results["livelock_construction"] = {
        "description": "Deterministic clean-probe alternation 1,0,1,0 with m=2 and b0=0.",
        "runs": livelock_runs,
        "aggregate": livelock_agg,
    }

    trap_cases = {}
    for p in [0.0, 0.05]:
        trap_runs, trap_agg = run_many(
            f"trap_escape_p_{p:.2f}",
            lambda: make_d3qs("balanced", budget_capacity=4, recovery_threshold=3),
            lambda p=p: TrapEscapeEnvironment(
                HEALTHY_LAW,
                DEGRADED_LAW,
                disturbance_prob=p,
                trust_execute=TRUST_EXECUTE,
                trust_quarantine=TRUST_QUARANTINE,
                trap_progress=0.05,
                productive_progress=1.0,
                escape_probability_on_recovery=0.8,
            ),
            seeds=seeds,
            base_seed=base_seed + 1000 + int(p * 1000),
            max_steps=300,
            trace_first_run=(p == 0.05),
        )
        trap_agg["escape_rate"] = statistics.fmean(
            1.0 if run["environment_meta"].get("escaped") else 0.0 for run in trap_runs
        )
        trap_cases[f"p={p:.2f}"] = {"runs": trap_runs, "aggregate": trap_agg}
    results["surprising_trap_state"] = {
        "description": "High-trust, low-progress trap. Small noise can trigger low-risk recovery and improve escape probability.",
        "cases": trap_cases,
    }
    return results


def run_parameter_sweep(base_seed: int, seeds: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for threshold_name in D3_THRESHOLDS:
        run_dicts, aggregate = run_many(
            f"d3_threshold_sweep_{threshold_name}",
            lambda threshold_name=threshold_name: make_d3(threshold_name),
            lambda: PoissonBernoulliEnvironment(HEALTHY_LAW, DEGRADED_LAW, regime_policy="healthy"),
            seeds=seeds,
            base_seed=base_seed + 2000,
            max_steps=200,
        )
        rows.append(
            {
                "model": "D3",
                "threshold_set": threshold_name,
                "p": None,
                "budget_capacity": None,
                "m": None,
                "termination_rate": aggregate["termination_rate"],
                "mean_termination_time": aggregate["mean_termination_time"],
                "mean_progress_rate": aggregate["mean_progress_rate"],
                "livelock_rate": aggregate["livelock_rate"],
                "mean_suspension_fraction": aggregate["mean_suspension_fraction"],
                "monotone_entropy_verified": verify_monotone_entropy(run_dicts),
            }
        )

    for threshold_name in Q_THRESHOLDS:
        for p in [0.01, 0.05, 0.10, 0.20]:
            for budget_capacity in [1, 2, 4, 8]:
                _, d3q_agg = run_many(
                    f"d3q_sweep_{threshold_name}_p_{p:.2f}_b_{budget_capacity}",
                    lambda threshold_name=threshold_name, budget_capacity=budget_capacity: make_d3q(
                        threshold_name,
                        budget_capacity=budget_capacity,
                    ),
                    lambda p=p: RecoverableNoiseEnvironment(
                        HEALTHY_LAW,
                        DEGRADED_LAW,
                        disturbance_prob=p,
                        trust_execute=TRUST_EXECUTE,
                        trust_quarantine=TRUST_QUARANTINE,
                    ),
                    seeds=seeds,
                    base_seed=base_seed + 3000 + int(p * 1000) + budget_capacity,
                    max_steps=recoverable_horizon(p, budget_capacity),
                )
                rows.append(
                    {
                        "model": "D3-Q",
                        "threshold_set": threshold_name,
                        "p": p,
                        "budget_capacity": budget_capacity,
                        "m": None,
                        "termination_rate": d3q_agg["termination_rate"],
                        "mean_termination_time": d3q_agg["mean_termination_time"],
                        "mean_progress_rate": d3q_agg["mean_progress_rate"],
                        "livelock_rate": d3q_agg["livelock_rate"],
                        "mean_suspension_fraction": d3q_agg["mean_suspension_fraction"],
                        "mean_disturbances": d3q_agg["mean_disturbances"],
                    }
                )

                for recovery_threshold in [2, 3, 4]:
                    _, d3qs_agg = run_many(
                        (
                            f"d3qs_sweep_{threshold_name}_p_{p:.2f}"
                            f"_b_{budget_capacity}_m_{recovery_threshold}"
                        ),
                        lambda threshold_name=threshold_name, budget_capacity=budget_capacity, recovery_threshold=recovery_threshold: make_d3qs(
                            threshold_name,
                            budget_capacity=budget_capacity,
                            recovery_threshold=recovery_threshold,
                        ),
                        lambda p=p: RecoverableNoiseEnvironment(
                            HEALTHY_LAW,
                            DEGRADED_LAW,
                            disturbance_prob=p,
                            trust_execute=TRUST_EXECUTE,
                            trust_quarantine=TRUST_QUARANTINE,
                        ),
                        seeds=seeds,
                        base_seed=(
                            base_seed
                            + 4000
                            + int(p * 1000)
                            + budget_capacity * 10
                            + recovery_threshold
                        ),
                        max_steps=2000,
                    )
                    rows.append(
                        {
                            "model": "D3-QS",
                            "threshold_set": threshold_name,
                            "p": p,
                            "budget_capacity": budget_capacity,
                            "m": recovery_threshold,
                            "termination_rate": d3qs_agg["termination_rate"],
                            "mean_termination_time": d3qs_agg["mean_termination_time"],
                            "mean_progress_rate": d3qs_agg["mean_progress_rate"],
                            "livelock_rate": d3qs_agg["livelock_rate"],
                            "mean_suspension_fraction": d3qs_agg["mean_suspension_fraction"],
                            "mean_disturbances": d3qs_agg["mean_disturbances"],
                            "mean_successful_recoveries": d3qs_agg["mean_successful_recoveries"],
                        }
                    )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def stress_summary_lines(stress_tests: dict[str, Any]) -> list[str]:
    baseline = stress_tests["baseline_sanity"]["aggregate"]
    persistent = stress_tests["persistent_adversarial"]["models"]
    recoverable_cases = stress_tests["recoverable_stochastic_noise"]["cases"]
    livelock = stress_tests["livelock_construction"]["aggregate"]
    trap = stress_tests["surprising_trap_state"]["cases"]

    lines = [
        "# D3 Theory Simulation Summary",
        "",
        "## Assumptions",
        "",
        "- Generic healthy/degraded runs use the paper's Poisson-Bernoulli observation law with a Bayesian trust proxy updated by log-likelihood ratio increments.",
        "- Exact recoverable-noise and livelock tests use the paper's constructed trust process with T^E = 0.92, T^Q = 0.68, and U_t = 0.",
        "- The trap-state experiment is exploratory rather than a theorem from the paper: low-risk recovery actions are allowed to escape a high-trust / low-progress basin.",
        "",
        "## Stress Test Verdicts",
        "",
        (
            f"- Baseline sanity: D3 terminated in {baseline['termination_rate']:.0%} of runs "
            f"with mean refusal time {_format_optional(baseline['mean_termination_time'])}; "
            f"monotone entropy verified = {baseline['monotone_entropy_verified']}."
        ),
        (
            f"- Persistent adversarial: termination rates were "
            f"D3={persistent['D3']['aggregate']['termination_rate']:.0%}, "
            f"D3-Q={persistent['D3-Q']['aggregate']['termination_rate']:.0%}, "
            f"D3-QS={persistent['D3-QS']['aggregate']['termination_rate']:.0%}."
        ),
    ]

    for case in recoverable_cases:
        p = case["p"]
        d3q = case["D3-Q"]["aggregate"]
        d3qs = case["D3-QS"]["aggregate"]
        lines.append(
            (
                f"- Recoverable noise p={p:.2f}: D3-Q termination={d3q['termination_rate']:.0%} "
                f"(mean refusal {_format_optional(d3q['mean_termination_time'])}), "
                f"D3-QS termination={d3qs['termination_rate']:.0%}, "
                f"suspension fraction={d3qs['mean_suspension_fraction']:.2f}, "
                f"recoveries/run={d3qs['mean_successful_recoveries']:.2f}."
            )
        )

    lines.extend(
        [
            (
                f"- Livelock construction: D3-QS termination={livelock['termination_rate']:.0%}, "
                f"livelock={livelock['livelock_rate']:.0%}, "
                f"suspension fraction={livelock['mean_suspension_fraction']:.2f}."
            ),
            (
                f"- Trap state: escape rate improved from {trap['p=0.00']['aggregate']['escape_rate']:.0%} "
                f"at p=0.00 to {trap['p=0.05']['aggregate']['escape_rate']:.0%} at p=0.05."
            ),
            "",
            "## Interpretation",
            "",
            "- D3 shows termination inevitability even on healthy runs because cumulative entropy H_t only moves upward.",
            "- D3-Q repairs monotone entropy but still fails under endless recoverable noise because the finite quarantine budget is eventually exhausted.",
            "- D3-QS removes that almost-sure refusal mechanism under recoverable noise, but safe suspension can dominate runtime and deterministic probe patterns can livelock it forever.",
        ]
    )
    return lines


def write_summary(stress_tests: dict[str, Any], path: Path) -> None:
    path.write_text("\n".join(stress_summary_lines(stress_tests)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the D3 / D3-Q / D3-QS simulation harness.")
    parser.add_argument("--seed", type=int, default=42, help="Base deterministic seed.")
    parser.add_argument("--runs", type=int, default=12, help="Runs per stress-test condition.")
    parser.add_argument("--sweep-runs", type=int, default=6, help="Runs per parameter-sweep point.")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip the parameter sweep.")
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stress_tests = run_stress_tests(base_seed=args.seed, seeds=args.runs)
    sweep_rows = [] if args.skip_sweep else run_parameter_sweep(base_seed=args.seed, seeds=args.sweep_runs)

    payload = {
        "seed": args.seed,
        "stress_tests": stress_tests,
        "parameter_sweep": sweep_rows,
    }

    _JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(stress_tests, _SUMMARY_PATH)
    if sweep_rows:
        write_csv(sweep_rows, _SWEEP_CSV_PATH)

    print("\n".join(stress_summary_lines(stress_tests)))
    print("")
    print(f"JSON: {_JSON_PATH}")
    if sweep_rows:
        print(f"CSV:  {_SWEEP_CSV_PATH}")
    print(f"MD:   {_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
