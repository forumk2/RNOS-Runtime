"""Run an LM Studio-driven RNOS intervention scenario."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.parser import parse_action
from agent.planner import Planner
from rnos.runtime import RNOSRuntime
from rnos.types import PolicyDecision
from tools.unstable_api import UnstableAPI, UnstableAPITool

TRACE_PATH = Path(__file__).resolve().parents[1] / "logs" / "rnos_trace.jsonl"


def run_agent(*, max_steps: int, seed: int) -> dict[str, object]:
    """Execute an RNOS-controlled tool loop driven by LM Studio."""

    random.seed(seed)
    planner = Planner()
    rnos = RNOSRuntime()
    unstable_api = UnstableAPI()
    tool = UnstableAPITool(api=unstable_api)
    history: list[dict[str, object]] = []
    retry_count = 0
    refused = False
    degrade_remaining: int | None = None
    final_entropy = 0.0
    final_trust = 0.0
    steps_executed = 0

    print("=== LM Studio RNOS Loop ===")
    print(f"seed={seed} max_steps={max_steps}")

    for step in range(1, max_steps + 1):
        had_degrade_budget = degrade_remaining is not None
        llm_output = planner.get_next_action(history)
        action = parse_action(llm_output)
        action.depth = step - 1
        action.retry_count = retry_count
        action.payload = {"resource": "/status"}
        action.metadata["step"] = step

        assessment = rnos.evaluate(action)
        final_entropy = assessment.entropy
        final_trust = assessment.trust

        print(
            f"[step {step:02d}] llm_output={llm_output!r} depth={action.depth} "
            f"entropy={assessment.entropy:.3f} "
            f"trust={assessment.trust:.3f} decision={assessment.decision.value.upper()} "
            f"retry_count={retry_count}"
        )

        if assessment.decision is PolicyDecision.REFUSE:
            refused = True
            print("           stop=RNOS refused execution")
            break

        if degrade_remaining == 0:
            print("           stop=DEGRADE budget exhausted")
            break

        if action.tool_name != "unstable_api":
            print("           tool_result=SKIPPED (planner requested unknown tool)")
            rnos.record_outcome(action, success=False)
            history.append(
                {
                    "step": step,
                    "llm_output": llm_output,
                    "tool": action.tool_name,
                    "result": "unknown_tool",
                }
            )
            steps_executed += 1
            break

        if assessment.decision is PolicyDecision.DEGRADE:
            degrade_remaining = int(assessment.constraints.get("max_additional_steps", 1))
            action.payload["_rnos_constraints"] = assessment.constraints
            print(
                "           degraded_mode=True "
                f"constraints={json.dumps(assessment.constraints, sort_keys=True)}"
            )

        result = tool.run(**action.payload)
        steps_executed += 1
        rnos.record_outcome(action, success=result.success)

        print(
            "           tool_result="
            f"{'SUCCESS' if result.success else 'FAILURE'} ({result.message})"
        )
        print(
            f"           phase={result.result_data.get('phase')} "
            f"call_count={result.result_data.get('call_count')} "
            f"failure_streak={result.result_data.get('failure_streak')}"
        )
        print(f"           result_data={json.dumps(result.result_data, sort_keys=True)}")

        history.append(
            {
                "step": step,
                "llm_output": llm_output,
                "tool": action.tool_name,
                "decision": assessment.decision.value,
                "ok": result.success,
                "phase": result.result_data.get("phase"),
                "retry_count": retry_count,
            }
        )

        retry_count = 0 if result.success else retry_count + 1

        if had_degrade_budget:
            degrade_remaining = max(0, degrade_remaining - 1)
            print(f"           remaining_degraded_retries={degrade_remaining}")

    summary = {
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
    parser = argparse.ArgumentParser(description="Run the LM Studio RNOS agent loop.")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text("", encoding="utf-8")

    summary = run_agent(max_steps=args.max_steps, seed=args.seed)

    print("\n=== Summary JSON ===")
    print(json.dumps(summary, indent=2))
    print(f"Trace log written to {TRACE_PATH}")


if __name__ == "__main__":
    main()
