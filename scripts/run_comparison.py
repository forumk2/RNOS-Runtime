"""Run all four execution modes across all three personas and multiple scenarios.

Usage::

    # Dry-run (stub planner, no LM Studio) — UnstableAPI + configurable scenarios:
    python scripts/run_comparison.py --dry-run --tag "dry-verify"

    # Live run against LM Studio (all personas, all scenarios):
    python scripts/run_comparison.py --max-steps 20 --seed 4 --tag "live-2026-06-04"

    # Live run, single persona only:
    python scripts/run_comparison.py --persona adversarial --tag "live-adv"

    # Suppress the configurable-API scenario block:
    python scripts/run_comparison.py --dry-run --no-scenario-block

Report columns
--------------
  Scenario         — which failure geometry was run
  Persona          — LLM persona (n/a for configurable scenarios)
  Mode             — RNOS / CircuitBreaker / Hybrid / Baseline
  Loop steps       — total planner invocations
  Exec steps       — actual tool calls made
  Failures         — tool call failures recorded
  Intervention @   — first step where control acted (DEGRADE/REFUSE/BLOCK)
  Tokens saved     — (baseline_loop_steps − mode_loop_steps) × avg_tokens_per_step
  Cost saved ($)   — tokens_saved × cost_per_1k_tokens / 1000

Worst-case reporting: for each mode the row with the MOST failures is shown.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT = REPO_ROOT / "scripts" / "run_agent.py"
GENERATE_REPORT = REPO_ROOT / "scripts" / "generate_report.py"

_BANNER_INNER_WIDTH = 46

_MODES_LLM: list[tuple[str, list[str]]] = [
    ("Baseline (no control)", ["--no-rnos"]),
    ("RNOS Mode",             []),
    ("Circuit Breaker Mode",  ["--circuit-breaker"]),
    ("Hybrid Mode",           ["--hybrid"]),
]

_ALL_PERSONAS = ["adversarial", "cautious", "mixed"]

# ---------------------------------------------------------------------------
# Configurable-API scenario runner (no LLM required)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(REPO_ROOT))

# Late imports so the script is importable without the full package installed.
def _import_scenario_deps() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    import logging as _log
    _log.getLogger("rnos.runtime").setLevel(logging.WARNING)

    from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
    from baselines.circuit_breaker import CircuitBreaker
    from experiments.configurable_api import make_fanout_cascade
    from experiments.experiment_5_hybrid.scenarios import (
        make_cascading_burst,
        make_distributed_low_rate,
    )
    from experiments.experiment_2 import EXP2_POLICY
    from rnos.hybrid import HybridController
    from rnos.runtime import RNOSRuntime
    from rnos.types import ActionRecord, PolicyDecision
    return (
        AdaptiveCircuitBreaker, CircuitBreaker,
        make_cascading_burst, make_distributed_low_rate, make_fanout_cascade,
        EXP2_POLICY,
        (HybridController, RNOSRuntime, ActionRecord, PolicyDecision),
    )


@dataclass
class ScenarioRow:
    scenario: str
    persona: str
    mode: str
    loop_steps: int
    exec_steps: int
    failures: int
    intervention_step: int | None
    tokens_saved: int = 0
    cost_saved: float = 0.0


def _run_one_mode(
    mode_label: str,
    api: Any,
    max_steps: int,
    *,
    AdaptiveCircuitBreaker: Any,
    HybridController: Any,
    RNOSRuntime: Any,
    ActionRecord: Any,
    PolicyDecision: Any,
    EXP2_POLICY: Any,
    trace_path: Any,
) -> tuple[int, int, int, int | None]:
    """Run a single mode; return (loop_steps, exec_steps, failures, first_intervention)."""
    api.reset()

    if mode_label == "RNOS":
        runtime = RNOSRuntime(trace_path=trace_path, policy_config=EXP2_POLICY)
    elif mode_label == "CircuitBreaker":
        cbk = AdaptiveCircuitBreaker(
            window_size=10, initial_failure_rate=0.60, initial_cooldown_steps=3,
        )
    elif mode_label == "Hybrid":
        rt = RNOSRuntime(trace_path=trace_path, policy_config=EXP2_POLICY)
        acb = AdaptiveCircuitBreaker(
            window_size=10, initial_failure_rate=0.60, initial_cooldown_steps=3,
        )
        hctrl = HybridController(rt, acb)

    steps_executed = 0
    tool_failures = 0
    retry_count = 0
    prev_latency: float | None = None
    first_intervention: int | None = None
    loop_steps = 0

    for step in range(1, max_steps + 1):
        loop_steps = step
        action = ActionRecord(
            tool_name="configurable_api",
            depth=0,
            retry_count=retry_count,
            latency_ms=prev_latency,
            cumulative_calls=steps_executed,
        )

        if mode_label == "Baseline":
            pass  # always execute

        elif mode_label == "RNOS":
            assessment = runtime.evaluate(action)
            if assessment.decision == PolicyDecision.REFUSE:
                if first_intervention is None:
                    first_intervention = step
                break
            if assessment.decision == PolicyDecision.DEGRADE and first_intervention is None:
                first_intervention = step

        elif mode_label == "CircuitBreaker":
            cbk.tick()
            allowed, cb_reason = cbk.should_execute()
            if not allowed:
                if first_intervention is None:
                    first_intervention = step
                if cb_reason == "permanently_open":
                    break
                continue
            if cb_reason == "half_open_probe" and first_intervention is None:
                first_intervention = step

        elif mode_label == "Hybrid":
            hctrl.tick()
            hdec = hctrl.evaluate(action)
            if hdec.decision == "REFUSE":
                if first_intervention is None:
                    first_intervention = step
                break
            if hdec.decision == "DEGRADE" and first_intervention is None:
                first_intervention = step

        outcome = api.call()
        steps_executed += 1
        if not outcome.success:
            tool_failures += 1
            retry_count += 1
        else:
            retry_count = 0
        action.latency_ms = outcome.latency_ms
        prev_latency = outcome.latency_ms

        if mode_label == "RNOS":
            runtime.record_outcome(action, success=outcome.success)
        elif mode_label == "CircuitBreaker":
            cbk.record_result(success=outcome.success)
        elif mode_label == "Hybrid":
            hctrl.record_outcome(action, success=outcome.success)

    return loop_steps, steps_executed, tool_failures, first_intervention


def _run_configurable_scenario(
    scenario_name: str,
    api: Any,
    max_steps: int,
    avg_tokens: int,
    cost_per_1k: float,
    *,
    AdaptiveCircuitBreaker: Any,
    CircuitBreaker: Any,
    HybridController: Any,
    RNOSRuntime: Any,
    ActionRecord: Any,
    PolicyDecision: Any,
    EXP2_POLICY: Any,
) -> list[ScenarioRow]:
    """Run one ConfigurableAPI scenario under all four modes, return rows."""
    trace_path = REPO_ROOT / "logs" / f"comp_{scenario_name}_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[ScenarioRow] = []
    baseline_loop_steps = max_steps

    for mode_label in ("Baseline", "RNOS", "CircuitBreaker", "Hybrid"):
        loop_steps, exec_steps, failures, first_intervention = _run_one_mode(
            mode_label=mode_label,
            api=api,
            max_steps=max_steps,
            AdaptiveCircuitBreaker=AdaptiveCircuitBreaker,
            HybridController=HybridController,
            RNOSRuntime=RNOSRuntime,
            ActionRecord=ActionRecord,
            PolicyDecision=PolicyDecision,
            EXP2_POLICY=EXP2_POLICY,
            trace_path=trace_path,
        )

        if mode_label == "Baseline":
            baseline_loop_steps = loop_steps

        tokens_saved = max(0, baseline_loop_steps - loop_steps) * avg_tokens
        cost_saved = tokens_saved * cost_per_1k / 1000.0

        rows.append(ScenarioRow(
            scenario=scenario_name,
            persona="n/a",
            mode=mode_label,
            loop_steps=loop_steps,
            exec_steps=exec_steps,
            failures=failures,
            intervention_step=first_intervention,
            tokens_saved=tokens_saved,
            cost_saved=round(cost_saved, 4),
        ))

    return rows


# ---------------------------------------------------------------------------
# LLM-planner runner (subprocess via run_agent.py)
# ---------------------------------------------------------------------------


def _banner(title: str) -> str:
    inner = f"  {title:<{_BANNER_INNER_WIDTH - 2}}"
    bar = "-" * _BANNER_INNER_WIDTH
    return f"\n+{bar}+\n|{inner}|\n+{bar}+"


def _run_llm_mode(
    mode_name: str,
    extra_flags: list[str],
    common_flags: list[str],
) -> int:
    print(_banner(mode_name))
    cmd = [sys.executable, str(RUN_AGENT)] + common_flags + extra_flags
    result = subprocess.run(cmd)
    print()
    return result.returncode


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_scenario_table(rows: list[ScenarioRow]) -> None:
    """Print a formatted comparison table for configurable-API scenarios."""
    # Worst-case per (scenario, mode): row with most failures
    worst: dict[tuple[str, str], ScenarioRow] = {}
    for row in rows:
        key = (row.scenario, row.mode)
        if key not in worst or row.failures > worst[key].failures:
            worst[key] = row

    col_w = [20, 6, 16, 6, 6, 6, 10, 10, 10]
    headers = [
        "Scenario", "Pers.", "Mode",
        "Loop", "Exec", "Fail",
        "Interv.@", "TokSaved", "Cost$(saved)",
    ]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header_row = "|" + "|".join(
        f" {h:<{w - 1}}" for h, w in zip(headers, col_w)
    ) + "|"

    print("\n" + sep)
    print(header_row)
    print(sep)

    # Order: Baseline first, then by failures descending within mode
    order = {"Baseline": 0, "RNOS": 1, "CircuitBreaker": 2, "Hybrid": 3}
    sorted_rows = sorted(
        worst.values(),
        key=lambda r: (r.scenario, order.get(r.mode, 9), -r.failures),
    )

    prev_scenario = None
    for row in sorted_rows:
        if row.scenario != prev_scenario and prev_scenario is not None:
            print(sep)
        prev_scenario = row.scenario
        interv = str(row.intervention_step) if row.intervention_step else "—"
        cells = [
            row.scenario, row.persona, row.mode,
            str(row.loop_steps), str(row.exec_steps), str(row.failures),
            interv, str(row.tokens_saved), f"{row.cost_saved:.4f}",
        ]
        print("|" + "|".join(f" {c:<{w - 1}}" for c, w in zip(cells, col_w)) + "|")

    print(sep)
    print(
        "\nTokens saved and cost saved are relative to the Baseline (no-control) run.\n"
        "Worst-case row shown per (scenario, mode) pair.\n"
        "avg_tokens_per_step and cost_per_1k_tokens are CLI-configurable."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run RNOS, Circuit Breaker, Hybrid, and Baseline across all personas "
            "and multiple failure scenarios, then report planner invocations and "
            "tokens/$ saved."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=20, metavar="N")
    parser.add_argument("--seed", type=int, default=4, metavar="N")
    parser.add_argument("--tag", type=str, default="", metavar="TEXT")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stub planner; no LM Studio required.")
    parser.add_argument(
        "--persona",
        choices=["adversarial", "cautious", "mixed"],
        default=None,
        help="Single persona to run. Omit to run all three.",
    )
    parser.add_argument(
        "--no-scenario-block",
        action="store_true",
        help="Skip the configurable-API scenario comparison block.",
    )
    parser.add_argument(
        "--avg-tokens-per-step",
        type=int,
        default=500,
        metavar="N",
        help="Estimated tokens per planner invocation for cost calculation (default: 500).",
    )
    parser.add_argument(
        "--cost-per-1k-tokens",
        type=float,
        default=0.010,
        metavar="F",
        help="USD cost per 1000 tokens (default: 0.010).",
    )
    args = parser.parse_args()

    personas = [args.persona] if args.persona else _ALL_PERSONAS

    # -----------------------------------------------------------------------
    # Section 1: LLM-planner runs (UnstableAPI, all personas × all modes)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  UnstableAPI — LLM-planner runs")
    print("=" * 50)

    all_exit_codes: list[int] = []
    for persona in personas:
        print(f"\n{'=' * 50}")
        print(f"  Persona: {persona}")
        print(f"{'=' * 50}")

        common_flags: list[str] = [
            "--max-steps", str(args.max_steps),
            "--seed", str(args.seed),
            "--persona", persona,
        ]
        if args.tag:
            common_flags += ["--tag", f"{args.tag}-{persona}"]
        if args.dry_run:
            common_flags.append("--dry-run")

        for mode_name, extra_flags in _MODES_LLM:
            rc = _run_llm_mode(mode_name, extra_flags, common_flags)
            all_exit_codes.append(rc)

    if any(rc != 0 for rc in all_exit_codes):
        print("\nWARNING: one or more LLM runs exited with a non-zero return code.")

    # -----------------------------------------------------------------------
    # Section 2: Generate the JSONL-based comparison report
    # -----------------------------------------------------------------------
    report_cmd = [sys.executable, str(GENERATE_REPORT)]
    if args.tag:
        report_cmd += ["--tag", args.tag]
    else:
        report_cmd += ["--seed", str(args.seed)]
    print("\nGenerating comparison report…")
    subprocess.run(report_cmd)
    results_dir = REPO_ROOT / "results"
    print(f"Report generated: {results_dir / 'report.md'}")
    print(f"Chart generated:  {results_dir / 'comparison_chart.png'}")

    # -----------------------------------------------------------------------
    # Section 3: Configurable-API scenario block (deterministic, no LLM)
    # -----------------------------------------------------------------------
    if args.no_scenario_block:
        return

    print("\n" + "=" * 50)
    print("  Configurable-API scenarios (deterministic)")
    print("  Scenarios: distributed_low_rate | cascading_burst | fanout_cascade")
    print("  Modes:     Baseline | RNOS | CircuitBreaker | Hybrid (coupled)")
    print("=" * 50)

    try:
        (
            AdaptiveCircuitBreaker, CircuitBreaker,
            make_cascading_burst, make_distributed_low_rate, make_fanout_cascade,
            EXP2_POLICY,
            (HybridController, RNOSRuntime, ActionRecord, PolicyDecision),
        ) = _import_scenario_deps()
    except ImportError as exc:
        print(f"\nSkipping scenario block (import error): {exc}")
        return

    all_rows: list[ScenarioRow] = []
    scenario_factories = [
        ("distributed_low_rate", make_distributed_low_rate),
        ("cascading_burst",      make_cascading_burst),
        ("fanout_cascade",       make_fanout_cascade),
    ]

    for scenario_name, factory in scenario_factories:
        api = factory(seed=args.seed)
        print(f"\n  Running scenario: {scenario_name} …")
        rows = _run_configurable_scenario(
            scenario_name=scenario_name,
            api=api,
            max_steps=args.max_steps,
            avg_tokens=args.avg_tokens_per_step,
            cost_per_1k=args.cost_per_1k_tokens,
            AdaptiveCircuitBreaker=AdaptiveCircuitBreaker,
            CircuitBreaker=CircuitBreaker,
            HybridController=HybridController,
            RNOSRuntime=RNOSRuntime,
            ActionRecord=ActionRecord,
            PolicyDecision=PolicyDecision,
            EXP2_POLICY=EXP2_POLICY,
        )
        all_rows.extend(rows)
        for row in rows:
            interv = f"step {row.intervention_step}" if row.intervention_step else "never"
            print(
                f"    {row.mode:<16}  loops={row.loop_steps:>3}  "
                f"exec={row.exec_steps:>3}  fail={row.failures:>3}  "
                f"interv@{interv:<10}  "
                f"tok_saved={row.tokens_saved:>6}  "
                f"cost_saved=${row.cost_saved:.4f}"
            )

    print()
    _print_scenario_table(all_rows)


if __name__ == "__main__":
    main()
