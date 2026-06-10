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
import math
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
def _import_scenario_deps() -> tuple[Any, ...]:
    import logging as _log
    _log.getLogger("rnos.runtime").setLevel(logging.WARNING)

    from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
    from baselines.circuit_breaker import CircuitBreaker
    from experiments.configurable_api import (
        make_benign,
        make_fanout_cascade,
        make_recoverable_burst,
    )
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
        make_recoverable_burst, make_benign,
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
    task_completed: bool = False
    tokens_saved: int = 0
    cost_saved: float = 0.0
    net_value: float = 0.0
    scenario_type: str = "failure"  # "failure" | "recoverable" | "benign"


@dataclass
class AggregatedRow:
    """Per-(scenario, mode) summary over N seeds."""
    scenario: str
    mode: str
    scenario_type: str   # "failure" | "recoverable" | "benign"
    n_seeds: int
    loop_steps_mean: float
    exec_steps_mean: float
    failures_mean: float
    survival_rate: float         # fraction of seeds where task_completed=True
    survival_ci_lo: float
    survival_ci_hi: float
    false_positive_rate: float   # fraction where baseline completed but mode didn't
    fpr_ci_lo: float
    fpr_ci_hi: float
    net_value_mean: float
    net_value_ci_half: float     # half-width of 95% CI
    tokens_saved_mean: float
    cost_saved_mean: float


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Return (mean, half-width of 95% CI) for a list of values."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, z * math.sqrt(var) / math.sqrt(n)


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
) -> tuple[int, int, int, int | None, bool]:
    """Run a single mode; return (loop_steps, exec_steps, failures, first_intervention, task_completed)."""
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

    task_completed = (
        getattr(api, "productive_goal", 0) > 0
        and getattr(api, "productive_successes", 0) >= getattr(api, "productive_goal", 0)
    )
    return loop_steps, steps_executed, tool_failures, first_intervention, task_completed


def _run_configurable_scenario(
    scenario_name: str,
    api: Any,
    max_steps: int,
    avg_tokens: int,
    cost_per_1k: float,
    task_value: float = 0.0,
    scenario_type: str = "failure",
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
    baseline_completed = False
    cost_per_step = avg_tokens * cost_per_1k / 1000.0
    task_value_tokens = int(task_value / (cost_per_1k / 1000.0)) if cost_per_1k > 0 else 0

    for mode_label in ("Baseline", "RNOS", "CircuitBreaker", "Hybrid"):
        loop_steps, exec_steps, failures, first_intervention, task_completed = _run_one_mode(
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
            baseline_completed = task_completed

        raw_saved = (baseline_loop_steps - loop_steps) * avg_tokens
        if baseline_completed and not task_completed:
            # False positive: net the forfeited task value against the step savings
            raw_saved -= task_value_tokens
        tokens_saved = raw_saved
        cost_saved = tokens_saved * cost_per_1k / 1000.0
        net_value = (task_value if task_completed else 0.0) - cost_per_step * loop_steps

        rows.append(ScenarioRow(
            scenario=scenario_name,
            persona="n/a",
            mode=mode_label,
            loop_steps=loop_steps,
            exec_steps=exec_steps,
            failures=failures,
            intervention_step=first_intervention,
            task_completed=task_completed,
            tokens_saved=tokens_saved,
            cost_saved=round(cost_saved, 4),
            net_value=round(net_value, 4),
            scenario_type=scenario_type,
        ))

    return rows


# ---------------------------------------------------------------------------
# Multi-seed aggregated runner
# ---------------------------------------------------------------------------


def _run_scenario_multi_seed(
    scenario_name: str,
    factory: Any,
    seeds: list[int],
    scenario_type: str,
    max_steps: int,
    task_value: float,
    avg_tokens: int,
    cost_per_1k: float,
    *,
    AdaptiveCircuitBreaker: Any,
    HybridController: Any,
    RNOSRuntime: Any,
    ActionRecord: Any,
    PolicyDecision: Any,
    EXP2_POLICY: Any,
) -> list[AggregatedRow]:
    """Run factory(seed) for each seed; aggregate survival, FPR, net_value."""
    cost_per_step = avg_tokens * cost_per_1k / 1000.0
    task_value_tokens = int(task_value / (cost_per_1k / 1000.0)) if cost_per_1k > 0 else 0
    trace_path = REPO_ROOT / "logs" / f"comp_{scenario_name}_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    modes = ["Baseline", "RNOS", "CircuitBreaker", "Hybrid"]
    # Indexed as [mode][seed_index] = (loop, exec, fail, interv, completed)
    per_mode: dict[str, list[tuple[int, int, int, int | None, bool]]] = {m: [] for m in modes}

    for seed in seeds:
        api = factory(seed=seed)
        for mode in modes:
            result = _run_one_mode(
                mode_label=mode,
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
            per_mode[mode].append(result)

    baseline = per_mode["Baseline"]
    baseline_loops = [r[0] for r in baseline]
    baseline_done = [r[4] for r in baseline]

    rows: list[AggregatedRow] = []
    for mode in modes:
        rs = per_mode[mode]
        n = len(seeds)
        loops = [r[0] for r in rs]
        execs = [r[1] for r in rs]
        fails = [r[2] for r in rs]
        done = [r[4] for r in rs]

        net_values = [
            (task_value if tc else 0.0) - cost_per_step * ls
            for tc, ls in zip(done, loops)
        ]
        tokens_saved_list: list[float] = []
        for i in range(n):
            raw = (baseline_loops[i] - loops[i]) * avg_tokens
            if baseline_done[i] and not done[i]:
                raw -= task_value_tokens
            tokens_saved_list.append(raw)

        n_done = sum(done)
        surv = n_done / n if n > 0 else 0.0
        surv_ci = _wilson_ci(n_done, n)

        if scenario_type in ("recoverable", "benign"):
            n_fp = sum(1 for bd, md in zip(baseline_done, done) if bd and not md)
            fpr = n_fp / n if n > 0 else 0.0
            fpr_ci = _wilson_ci(n_fp, n)
        else:
            fpr, fpr_ci = 0.0, (0.0, 0.0)

        nv_mean, nv_ci = _mean_ci(net_values)
        ts_mean = sum(tokens_saved_list) / n if n > 0 else 0.0
        cs_mean = ts_mean * cost_per_1k / 1000.0

        rows.append(AggregatedRow(
            scenario=scenario_name,
            mode=mode,
            scenario_type=scenario_type,
            n_seeds=n,
            loop_steps_mean=round(sum(loops) / n, 1),
            exec_steps_mean=round(sum(execs) / n, 1),
            failures_mean=round(sum(fails) / n, 1),
            survival_rate=round(surv, 3),
            survival_ci_lo=round(surv_ci[0], 3),
            survival_ci_hi=round(surv_ci[1], 3),
            false_positive_rate=round(fpr, 3),
            fpr_ci_lo=round(fpr_ci[0], 3),
            fpr_ci_hi=round(fpr_ci[1], 3),
            net_value_mean=round(nv_mean, 4),
            net_value_ci_half=round(nv_ci, 4),
            tokens_saved_mean=round(ts_mean, 1),
            cost_saved_mean=round(cs_mean, 4),
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
# Hardening table and dominance report
# ---------------------------------------------------------------------------


def _fmt_ci_pct(lo: float, hi: float) -> str:
    return f"[{lo * 100:.0f},{hi * 100:.0f}]"


def _print_hardening_table(rows: list[AggregatedRow], task_value: float) -> None:
    """Print the baseline-hardening results table (failure + recoverable + benign)."""
    failure_rows = [r for r in rows if r.scenario_type == "failure"]
    recovery_rows = [r for r in rows if r.scenario_type in ("recoverable", "benign")]

    order = {"Baseline": 0, "RNOS": 1, "CircuitBreaker": 2, "Hybrid": 3}

    def _print_section(title: str, section_rows: list[AggregatedRow], columns: str) -> None:
        print(f"\n  {title}")
        if columns == "failure":
            hdr = f"  {'Scenario':<20} {'Mode':<16} {'N':>4} {'Loop':>6} {'Fail':>6} {'NetVal':>9}  {'CI±':>7}  {'TokSaved':>10}"
            sep = "  " + "-" * (len(hdr) - 2)
            print(sep)
            print(hdr)
            print(sep)
            prev = ""
            for r in sorted(section_rows, key=lambda x: (x.scenario, order.get(x.mode, 9))):
                if r.scenario != prev and prev:
                    print()
                prev = r.scenario
                print(
                    f"  {r.scenario:<20} {r.mode:<16} {r.n_seeds:>4} "
                    f"{r.loop_steps_mean:>6.1f} {r.failures_mean:>6.1f} "
                    f"{r.net_value_mean:>+9.4f}  {r.net_value_ci_half:>7.4f}  "
                    f"{r.tokens_saved_mean:>10.0f}"
                )
            print(sep)
        else:
            hdr = (
                f"  {'Scenario':<20} {'Mode':<16} {'N':>4} "
                f"{'Survival%':>11} {'CI':>9} "
                f"{'FPR%':>7} {'CI':>9} "
                f"{'NetVal':>9}  {'CI±':>7}"
            )
            sep = "  " + "-" * (len(hdr) - 2)
            print(sep)
            print(hdr)
            print(sep)
            prev = ""
            for r in sorted(section_rows, key=lambda x: (x.scenario, order.get(x.mode, 9))):
                if r.scenario != prev and prev:
                    print()
                prev = r.scenario
                surv_pct = f"{r.survival_rate * 100:.0f}%"
                surv_ci_s = _fmt_ci_pct(r.survival_ci_lo, r.survival_ci_hi)
                fpr_pct = f"{r.false_positive_rate * 100:.0f}%" if r.scenario_type != "failure" else "n/a"
                fpr_ci_s = _fmt_ci_pct(r.fpr_ci_lo, r.fpr_ci_hi) if r.scenario_type != "failure" else "n/a"
                print(
                    f"  {r.scenario:<20} {r.mode:<16} {r.n_seeds:>4} "
                    f"{surv_pct:>11} {surv_ci_s:>9} "
                    f"{fpr_pct:>7} {fpr_ci_s:>9} "
                    f"{r.net_value_mean:>+9.4f}  {r.net_value_ci_half:>7.4f}"
                )
            print(sep)

    print("\n" + "=" * 70)
    print("  Baseline Hardening: full suite results")
    print(f"  task_value V={task_value:.2f} USD   cost_per_step c=avg_tokens*cost_per_1k/1000")
    print("=" * 70)

    if failure_rows:
        _print_section("Failure scenarios (no goal; lower net_value = less waste)", failure_rows, "failure")
    if recovery_rows:
        _print_section("Recoverable/benign scenarios (goal reachable; survival matters)", recovery_rows, "recovery")

    print("\n  Note: tokens_saved on recoverable/benign is netted against forfeited V on false positives.")
    print(f"  V/c sensitivity crossover: breakeven if goal_step = V / c.  Sweep with --task-value.")


def _dominance_verdict(rows: list[AggregatedRow]) -> list[str]:
    """Return lines summarising whether Hybrid weakly dominates RNOS and CB."""
    by_sc: dict[str, dict[str, AggregatedRow]] = {}
    for r in rows:
        by_sc.setdefault(r.scenario, {})[r.mode] = r

    lines: list[str] = []
    strict_wins_vs_rnos: list[str] = []
    strict_wins_vs_cb: list[str] = []
    loses_vs_rnos: list[str] = []
    loses_vs_cb: list[str] = []
    fp_scenarios: list[str] = []

    for sc, modes in sorted(by_sc.items()):
        hyb = modes.get("Hybrid")
        rnos = modes.get("RNOS")
        cbk = modes.get("CircuitBreaker")
        if not (hyb and rnos and cbk):
            continue
        h_nv, r_nv, c_nv = hyb.net_value_mean, rnos.net_value_mean, cbk.net_value_mean
        if h_nv > r_nv + 1e-9:
            strict_wins_vs_rnos.append(sc)
        elif h_nv < r_nv - 1e-9:
            loses_vs_rnos.append(sc)
        if h_nv > c_nv + 1e-9:
            strict_wins_vs_cb.append(sc)
        elif h_nv < c_nv - 1e-9:
            loses_vs_cb.append(sc)
        if hyb.scenario_type in ("recoverable", "benign") and hyb.false_positive_rate > 0:
            fp_scenarios.append(f"{sc} FPR={hyb.false_positive_rate * 100:.0f}%")
        sc_type = hyb.scenario_type
        lines.append(
            f"  {sc:<22}  hyb={h_nv:+.4f}  rnos={r_nv:+.4f}  cb={c_nv:+.4f}"
            + (f"  [surv={hyb.survival_rate * 100:.0f}% fpr={hyb.false_positive_rate * 100:.0f}%]"
               if sc_type in ("recoverable", "benign") else "")
        )

    summary: list[str] = []
    summary.append("")
    summary.append("=== DOMINANCE VERDICT ===")
    summary.extend(lines)
    summary.append("")
    if loses_vs_rnos:
        summary.append(f"  Hybrid LOSES to RNOS on: {', '.join(loses_vs_rnos)}")
    elif strict_wins_vs_rnos:
        summary.append(f"  Hybrid STRICTLY dominates RNOS on: {', '.join(strict_wins_vs_rnos)}")
    else:
        summary.append("  Hybrid matches RNOS on all failure/recovery scenarios (no strict win, no loss).")
    if loses_vs_cb:
        summary.append(f"  Hybrid LOSES to CB on: {', '.join(loses_vs_cb)}")
    elif strict_wins_vs_cb:
        summary.append(f"  Hybrid STRICTLY dominates CB on: {', '.join(strict_wins_vs_cb)}")
    else:
        summary.append("  Hybrid matches CB on all scenarios (no strict win, no loss).")
    if fp_scenarios:
        summary.append("")
        summary.append("  FALSE POSITIVE NOTE (speed/safety frontier, not a bug):")
        for fp in fp_scenarios:
            summary.append(f"    {fp} — early window is indistinguishable from cascading failure.")
        summary.append("    A controller cannot distinguish recoverable_burst from cascading_burst")
        summary.append("    in the shared opening window; refusal is unavoidable if it catches the burst.")
    summary.append("")
    return summary


def _write_hardening_report(
    rows: list[AggregatedRow],
    task_value: float,
    cost_per_1k: float,
    avg_tokens: int,
    n_seeds: int,
    output_path: Path,
) -> None:
    """Write docs/baseline_hardening_results.md with measured results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_lines = _dominance_verdict(rows)

    by_sc: dict[str, dict[str, AggregatedRow]] = {}
    for r in rows:
        by_sc.setdefault(r.scenario, {})[r.mode] = r

    order = {"Baseline": 0, "RNOS": 1, "CircuitBreaker": 2, "Hybrid": 3}
    lines: list[str] = []
    lines.append("# Baseline Hardening Results")
    lines.append("")
    lines.append(f"Run date: 2026-06-04  |  N={n_seeds} seeds  |  V={task_value:.2f} USD  "
                 f"|  cost={cost_per_1k:.3f}/1k tokens  |  avg_tokens={avg_tokens}")
    lines.append("")
    lines.append("## Full suite — net_value (μ ± 95% CI)")
    lines.append("")
    lines.append("| Scenario | Type | Mode | N | Loop(μ) | Fail(μ) | Survival% [CI] | FPR% [CI] | NetVal(μ) | CI± |")
    lines.append("|----------|------|------|---|---------|---------|----------------|-----------|-----------|-----|")
    for sc, modes in sorted(by_sc.items()):
        for mode_name in sorted(modes.keys(), key=lambda m: order.get(m, 9)):
            r = modes[mode_name]
            surv = f"{r.survival_rate * 100:.0f}% {_fmt_ci_pct(r.survival_ci_lo, r.survival_ci_hi)}"
            fpr = (
                f"{r.false_positive_rate * 100:.0f}% {_fmt_ci_pct(r.fpr_ci_lo, r.fpr_ci_hi)}"
                if r.scenario_type in ("recoverable", "benign")
                else "n/a"
            )
            lines.append(
                f"| {r.scenario} | {r.scenario_type} | {mode_name} | {r.n_seeds} | "
                f"{r.loop_steps_mean:.1f} | {r.failures_mean:.1f} | {surv} | {fpr} | "
                f"{r.net_value_mean:+.4f} | ±{r.net_value_ci_half:.4f} |"
            )

    lines.append("")
    lines.append("## Dominance verdict")
    lines.append("")
    for vl in verdict_lines:
        lines.append(vl)

    lines.append("")
    lines.append("## V/c sensitivity")
    lines.append("")
    lines.append(f"V={task_value:.2f}, c=avg_tokens*cost_per_1k/1000 = {avg_tokens}*{cost_per_1k}/{1000}")
    c = avg_tokens * cost_per_1k / 1000.0
    lines.append(f"Completing a recoverable run is net-positive when goal_step < V/c = {task_value/c:.1f} steps.")
    lines.append("Rerun with `--task-value X` to sweep this ratio without changing thresholds.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nHardening report written: {output_path}")


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
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        metavar="N",
        help="Number of random seeds for multi-seed scenario runs (default: 20).",
    )
    parser.add_argument(
        "--task-value",
        type=float,
        default=1.0,
        metavar="V",
        help="Task value V in USD for net_value calculation (default: 1.0).",
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

    print("\n" + "=" * 70)
    print("  Configurable-API scenarios (Baseline Hardening)")
    print("  Failure:     distributed_low_rate | cascading_burst | fanout_cascade")
    print("  Recoverable: recoverable_burst")
    print("  Benign:      benign")
    print(f"  Seeds: {args.n_seeds}   task_value V={args.task_value:.2f}   max_steps={args.max_steps}")
    print("=" * 70)

    try:
        (
            AdaptiveCircuitBreaker, CircuitBreaker,
            make_cascading_burst, make_distributed_low_rate, make_fanout_cascade,
            make_recoverable_burst, make_benign,
            EXP2_POLICY,
            (HybridController, RNOSRuntime, ActionRecord, PolicyDecision),
        ) = _import_scenario_deps()
    except ImportError as exc:
        print(f"\nSkipping scenario block (import error): {exc}")
        return

    seeds = list(range(args.n_seeds))
    scenario_factories = [
        ("distributed_low_rate", make_distributed_low_rate, "failure"),
        ("cascading_burst",      make_cascading_burst,      "failure"),
        ("fanout_cascade",       make_fanout_cascade,       "failure"),
        ("recoverable_burst",    make_recoverable_burst,    "recoverable"),
        ("benign",               make_benign,               "benign"),
    ]

    shared_kwargs = dict(
        AdaptiveCircuitBreaker=AdaptiveCircuitBreaker,
        HybridController=HybridController,
        RNOSRuntime=RNOSRuntime,
        ActionRecord=ActionRecord,
        PolicyDecision=PolicyDecision,
        EXP2_POLICY=EXP2_POLICY,
    )

    all_agg: list[AggregatedRow] = []
    for scenario_name, factory, sc_type in scenario_factories:
        print(f"\n  Running: {scenario_name} ({sc_type}, {args.n_seeds} seeds) …", flush=True)
        agg_rows = _run_scenario_multi_seed(
            scenario_name=scenario_name,
            factory=factory,
            seeds=seeds,
            scenario_type=sc_type,
            max_steps=args.max_steps,
            task_value=args.task_value,
            avg_tokens=args.avg_tokens_per_step,
            cost_per_1k=args.cost_per_1k_tokens,
            **shared_kwargs,
        )
        all_agg.extend(agg_rows)
        for r in agg_rows:
            surv_str = f"surv={r.survival_rate * 100:.0f}%" if r.scenario_type != "failure" else ""
            fpr_str = f"fpr={r.false_positive_rate * 100:.0f}%" if r.scenario_type in ("recoverable", "benign") else ""
            print(
                f"    {r.mode:<16}  loops={r.loop_steps_mean:>5.1f}  "
                f"fail={r.failures_mean:>5.1f}  "
                f"netval={r.net_value_mean:>+8.4f}  "
                f"tok_saved={r.tokens_saved_mean:>8.0f}  "
                + (f"{surv_str:<10} {fpr_str}" if (surv_str or fpr_str) else "")
            )

    print()
    _print_hardening_table(all_agg, args.task_value)

    for line in _dominance_verdict(all_agg):
        print(line)

    report_path = REPO_ROOT / "docs" / "baseline_hardening_results.md"
    _write_hardening_report(
        rows=all_agg,
        task_value=args.task_value,
        cost_per_1k=args.cost_per_1k_tokens,
        avg_tokens=args.avg_tokens_per_step,
        n_seeds=args.n_seeds,
        output_path=report_path,
    )


if __name__ == "__main__":
    main()
