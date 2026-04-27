import time
from dataclasses import replace

from agent_runtime import rnos_adapter
from agent_runtime.cevak import compute_cevak

from langchain_adapter import convert_to_execution_result
from utils import DemoMetrics, collect_metrics, create_langchain_agent


MAX_ATTEMPTS = 10


def _invoke_agent(agent, task: str):
    try:
        return agent.invoke({"messages": [{"role": "user", "content": task}]})
    except Exception as exc:
        return exc


def run_langchain_naive(task: str, max_attempts: int = MAX_ATTEMPTS) -> DemoMetrics:
    agent = create_langchain_agent()
    history = []
    started_at = time.perf_counter()

    for index in range(max_attempts):
        output = _invoke_agent(agent, task)
        previous_result = history[-1] if history else None
        result = convert_to_execution_result(
            output,
            index,
            previous_result=previous_result,
        )
        result = replace(result, cevak=compute_cevak(result, history))
        history.append(result)
        print(f"[LANGCHAIN NAIVE] step={index + 1} success={result.success}")

    return collect_metrics(history, refusals=0, started_at=started_at)


def run_langchain_rnos(task: str, max_attempts: int = MAX_ATTEMPTS) -> DemoMetrics:
    agent = create_langchain_agent()
    history = []
    refusals = 0
    started_at = time.perf_counter()

    for index in range(max_attempts):
        output = _invoke_agent(agent, task)
        previous_result = history[-1] if history else None
        result = convert_to_execution_result(
            output,
            index,
            previous_result=previous_result,
        )
        result = replace(result, cevak=compute_cevak(result, history))
        history.append(result)

        decision = rnos_adapter.evaluate_state(history)
        print(f"[LANGCHAIN + RNOS] step={index + 1} success={result.success}")
        print("[RNOS DECISION]")
        print(f"action={decision.action}")
        print(f"reason={decision.reason}")

        if decision.action == "refuse":
            refusals += 1
            print("RNOS stopped execution")
            break

    return collect_metrics(history, refusals=refusals, started_at=started_at)
