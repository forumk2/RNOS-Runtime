"""Run a local RNOS intervention scenario against the unstable API tool."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord, PolicyDecision
from tools.unstable_api import UnstableAPITool

TRACE_PATH = Path(__file__).resolve().parents[1] / "logs" / "rnos_trace.jsonl"


def run_scenario(
    *,
    name: str,
    failure_rate: float,
    max_steps: int,
    seed: int | None,
    stop_on_success: bool,
) -> dict[str, object]:
    """Execute repeated unstable tool calls until success, refusal, or max steps."""

    if seed is not None:
        random.seed(seed)

    rnos = RNOSRuntime()
    tool = UnstableAPITool(failure_rate=failure_rate)

    retry_count = 0
    refused = False
    degrade_budget: int | None = None
    final_entropy = 0.0
    final_trust = 0.0
    steps_executed = 0

    print(f"\n=== Scenario: {name} ===")
    print(
        f"failure_rate={failure_rate:.2f} seed={seed if seed is not None else 'system'} "
        f"max_steps={max_steps}"
    )

    for step in range(1, max_steps + 1):
        metadata: dict[str, object] = {"scenario": name}
        if degrade_budget is not None:
            metadata["remaining_degraded_retries"] = degrade_budget

        action = ActionRecord(
            tool_name="unstable_api",
            payload={"resource": "/status"},
            depth=step - 1,
            retry_count=retry_count,
            metadata=metadata,
        )
        assessment = rnos.evaluate(action)
        final_entropy = assessment.entropy
        final_trust = assessment.trust

        print(
            f"[step {step:02d}] depth={action.depth} entropy={assessment.entropy:.3f} "
            f"trust={assessment.trust:.3f} decision={assessment.decision.value.upper()} "
            f"retry_count={retry_count}"
        )

        if assessment.decision is PolicyDecision.REFUSE:
            refused = True
            print("           tool_result=SKIPPED (RNOS refused execution)")
            break

        if assessment.decision is PolicyDecision.DEGRADE:
            degrade_budget = int(assessment.constraints.get("max_additional_steps", 1))
            action.payload["_rnos_constraints"] = assessment.constraints
            print(
                "           degraded_mode=True "
                f"constraints={json.dumps(assessment.constraints, sort_keys=True)}"
            )

        result = tool.run(**action.payload)
        steps_executed += 1
        rnos.record_outcome(action, success=result.ok)

        print(
            "           tool_result="
            f"{'SUCCESS' if result.ok else 'FAILURE'} ({result.message})"
        )
        print(f"           result_data={json.dumps(result.data, sort_keys=True)}")

        if result.ok:
            retry_count = 0
            if stop_on_success:
                print("           stopping_after_success=True")
                break
            continue

        retry_count += 1

        if assessment.decision is PolicyDecision.DEGRADE and degrade_budget is not None:
            degrade_budget = max(0, degrade_budget - 1)
            print(f"           remaining_degraded_retries={degrade_budget}")

    summary = {
        "scenario": name,
        "total_steps_executed": steps_executed,
        "refused": refused,
        "final_entropy": final_entropy,
        "final_trust": final_trust,
    }

    print("\nSummary")
    print(f"  total_steps_executed={steps_executed}")
    print(f"  refused={refused}")
    print(f"  final_entropy={final_entropy:.3f}")
    print(f"  final_trust={final_trust:.3f}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local RNOS unstable API scenarios.")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--failure-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--include-secondary-scenario",
        action="store_true",
        help="Also run a lower-failure scenario to compare RNOS behavior.",
    )
    args = parser.parse_args()

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text("", encoding="utf-8")

    summaries = [
        run_scenario(
            name="failing_loop",
            failure_rate=args.failure_rate,
            max_steps=args.max_steps,
            seed=args.seed,
            stop_on_success=True,
        )
    ]

    if args.include_secondary_scenario:
        summaries.append(
            run_scenario(
                name="intermittent_loop",
                failure_rate=0.45,
                max_steps=args.max_steps,
                seed=2,
                stop_on_success=False,
            )
        )

    print("\n=== All Scenarios ===")
    print(json.dumps(summaries, indent=2))
    print(f"Trace log written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
