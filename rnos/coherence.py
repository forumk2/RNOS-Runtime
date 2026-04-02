"""Runtime Coherence Metrics v0.1 over execution traces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_PHASE_PRESSURE = {
    "stable": 0.0,
    "unstable": 0.5,
    "collapse": 1.0,
}
_REGIMES = ("resonant", "critical", "collapse")


def compute_runtime_coherence(step_trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute Runtime Coherence Metrics v0.1 over a per-step trace."""
    if not step_trace:
        empty_summary = {
            "avg_r": 0.0,
            "avg_H": 0.0,
            "avg_Lambda": 0.0,
            "regime_percentages": {regime: 0.0 for regime in _REGIMES},
            "first_step_lambda_below_0_20": None,
            "longest_zero_r_run": 0,
        }
        return {
            "table": [],
            "summary": empty_summary,
            "interpretation": [
                "coherent failure: not observed in this trace",
                "desynchronized loop: not observed in this trace",
            ],
            "r_series": [],
            "H_series": [],
            "Lambda_series": [],
        }

    max_failures = max(float(step["consecutive_failures"]) for step in step_trace)
    max_latency = max(float(step["planner_latency_ms"]) for step in step_trace)
    failure_denominator = max_failures if max_failures > 0.0 else 1.0
    latency_denominator = max_latency if max_latency > 0.0 else 1.0

    table: list[dict[str, Any]] = []
    regime_counts = {regime: 0 for regime in _REGIMES}
    r_series: list[float] = []
    h_series: list[float] = []
    lambda_series: list[float] = []

    for raw_step in step_trace:
        step = int(raw_step["step"])
        phase = str(raw_step["phase"]).lower()
        decision = str(raw_step["decision"]).upper()
        tool_result = str(raw_step["tool_result"]).upper()
        planner_emitted_tool_call = bool(raw_step.get("planner_emitted_tool_call", True))

        if phase not in _PHASE_PRESSURE:
            raise ValueError(f"Unsupported phase for step {step}: {phase!r}")

        s_pe = 1 if decision == "EXECUTE" else 0
        if decision == "EXECUTE":
            s_pg = 1
        elif decision in {"BLOCKED", "STOPPED"} and planner_emitted_tool_call:
            s_pg = 0
        else:
            s_pg = 1
        s_pt = 1 if tool_result in {"SUCCESS", "FAILURE"} else 0
        s_et = 1 if decision == "EXECUTE" else 0
        r_t = (s_pe + s_pg + s_pt + s_et) / 4.0

        f_t = float(raw_step["consecutive_failures"]) / failure_denominator
        c_t = _PHASE_PRESSURE[phase]
        b_t = 1.0 if decision in {"BLOCKED", "STOPPED"} else 0.0
        l_t = float(raw_step["planner_latency_ms"]) / latency_denominator
        h_t = 0.35 * f_t + 0.20 * c_t + 0.25 * b_t + 0.20 * l_t
        lambda_t = r_t / (1.0 + h_t)

        if lambda_t > 0.45:
            regime = "resonant"
        elif lambda_t >= 0.20:
            regime = "critical"
        else:
            regime = "collapse"

        row = {
            "step": step,
            "phase": phase,
            "decision": decision,
            "tool_result": tool_result,
            "consecutive_failures": int(raw_step["consecutive_failures"]),
            "r_t": r_t,
            "H_t": h_t,
            "Lambda_t": lambda_t,
            "regime": regime,
        }
        table.append(row)
        regime_counts[regime] += 1
        r_series.append(r_t)
        h_series.append(h_t)
        lambda_series.append(lambda_t)

    count = len(table)
    summary = {
        "avg_r": sum(r_series) / count,
        "avg_H": sum(h_series) / count,
        "avg_Lambda": sum(lambda_series) / count,
        "regime_percentages": {
            regime: (regime_counts[regime] / count) * 100.0 for regime in _REGIMES
        },
        "first_step_lambda_below_0_20": next(
            (row["step"] for row in table if row["Lambda_t"] < 0.20),
            None,
        ),
        "longest_zero_r_run": _longest_zero_r_run(table),
    }

    return {
        "table": table,
        "summary": summary,
        "interpretation": _build_interpretation(table),
        "r_series": r_series,
        "H_series": h_series,
        "Lambda_series": lambda_series,
    }


def format_runtime_coherence_report(report: Mapping[str, Any]) -> str:
    """Render a human-readable coherence report."""
    lines = ["step | r_t | H_t | Lambda_t | regime"]
    for row in report["table"]:
        lines.append(
            f"{row['step']} | {row['r_t']:.3f} | {row['H_t']:.3f} | "
            f"{row['Lambda_t']:.3f} | {row['regime']}"
        )

    summary = report["summary"]
    regime_percentages = summary["regime_percentages"]

    lines.extend(
        [
            "",
            "Summary",
            f"avg_r={summary['avg_r']:.3f}",
            f"avg_H={summary['avg_H']:.3f}",
            f"avg_Lambda={summary['avg_Lambda']:.3f}",
            (
                "% steps in each regime: "
                f"resonant={regime_percentages['resonant']:.1f}% "
                f"critical={regime_percentages['critical']:.1f}% "
                f"collapse={regime_percentages['collapse']:.1f}%"
            ),
            (
                "first step where Lambda_t < 0.20="
                f"{summary['first_step_lambda_below_0_20']}"
            ),
            (
                "longest consecutive run where r_t == 0="
                f"{summary['longest_zero_r_run']}"
            ),
            "",
            "Interpretation",
        ]
    )
    lines.extend(f"- {line}" for line in report["interpretation"])
    return "\n".join(lines)


def _build_interpretation(table: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return strict, trace-bound interpretation lines."""
    coherent_run = _find_coherent_failure_run(table)
    if coherent_run is None:
        coherent_line = "coherent failure: not observed in this trace"
    else:
        start_step, end_step, start_h, end_h = coherent_run
        coherent_line = (
            f"coherent failure: enters at step {end_step}; high r_t persists "
            f"while H_t rises {start_h:.3f}->{end_h:.3f}"
        )

    zero_run = _find_first_zero_r_run(table)
    if zero_run is None:
        desync_line = "desynchronized loop: not observed in this trace"
    else:
        start_step, end_step = zero_run
        if start_step == end_step:
            desync_line = f"desynchronized loop: enters at step {start_step} with r_t = 0.000"
        else:
            desync_line = f"desynchronized loop: steps {start_step}-{end_step} hold r_t = 0.000"

    return [coherent_line, desync_line]


def _find_coherent_failure_run(
    table: Sequence[Mapping[str, Any]],
) -> tuple[int, int, float, float] | None:
    """Find the first consecutive run with high r_t and strictly rising H_t."""
    best_run: tuple[int, int, float, float] | None = None
    start_index: int | None = None

    for index, row in enumerate(table):
        if row["r_t"] < 0.75:
            start_index = None
            continue

        if start_index is None:
            start_index = index
            continue

        previous = table[index - 1]
        is_consecutive = row["step"] == previous["step"] + 1
        h_is_rising = row["H_t"] > previous["H_t"]
        failures_are_rising = row["consecutive_failures"] > previous["consecutive_failures"]
        non_stable_phase = row["phase"] != "stable" or previous["phase"] != "stable"
        if (
            not is_consecutive
            or previous["r_t"] < 0.75
            or not h_is_rising
            or not failures_are_rising
            or not non_stable_phase
        ):
            start_index = index
            continue

        run_start = table[start_index]
        best_run = (
            int(run_start["step"]),
            int(row["step"]),
            float(run_start["H_t"]),
            float(row["H_t"]),
        )
        break

    return best_run


def _find_first_zero_r_run(table: Sequence[Mapping[str, Any]]) -> tuple[int, int] | None:
    """Return the first contiguous run where r_t is exactly zero."""
    start_step: int | None = None
    end_step: int | None = None
    previous_step: int | None = None

    for row in table:
        if abs(float(row["r_t"])) > 1e-12:
            if start_step is not None:
                break
            previous_step = int(row["step"])
            continue

        current_step = int(row["step"])
        if start_step is None:
            start_step = current_step
            end_step = current_step
        elif previous_step is not None and current_step == previous_step + 1:
            end_step = current_step
        else:
            break

        previous_step = current_step

    if start_step is None or end_step is None:
        return None
    return start_step, end_step


def _longest_zero_r_run(table: Sequence[Mapping[str, Any]]) -> int:
    """Return the longest consecutive run where r_t is exactly zero."""
    longest = 0
    current = 0
    previous_step: int | None = None

    for row in table:
        current_step = int(row["step"])
        if abs(float(row["r_t"])) <= 1e-12:
            if previous_step is not None and current_step == previous_step + 1:
                current += 1
            else:
                current = 1
            longest = max(longest, current)
        else:
            current = 0
        previous_step = current_step

    return longest


__all__ = ["compute_runtime_coherence", "format_runtime_coherence_report"]
