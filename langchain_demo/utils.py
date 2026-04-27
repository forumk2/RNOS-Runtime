import time
from dataclasses import dataclass

from agent_runtime.types import ExecutionResult


@dataclass(frozen=True)
class DemoMetrics:
    attempts: int
    failures: int
    refusals: int
    wasted_attempts: int
    duration: float


class DeterministicLangChainAgent:
    def __init__(self) -> None:
        self._attempt = 0

    def invoke(self, payload: dict) -> dict:
        self._attempt += 1
        task = payload["messages"][-1]["content"]
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"attempt={self._attempt}; task={task}; "
                        "generated invalid python: print(run("
                    ),
                }
            ],
            "error": "SyntaxError: '(' was never closed",
        }


def create_langchain_agent():
    try:
        from langchain.agents import create_agent

        return create_agent(
            model="openai:gpt-5.4",
            tools=[],
            system_prompt="You are a coding assistant",
        )
    except Exception:
        return DeterministicLangChainAgent()


def count_wasted_attempts(history: list[ExecutionResult]) -> int:
    return sum(
        1
        for result in history
        if (result.ast_similarity_to_previous or 0.0) > 0.85
        and (result.ast_progress_score or 0.0) < 0.1
    )


def collect_metrics(
    history: list[ExecutionResult],
    refusals: int,
    started_at: float,
) -> DemoMetrics:
    return DemoMetrics(
        attempts=len(history),
        failures=sum(1 for result in history if not result.success),
        refusals=refusals,
        wasted_attempts=count_wasted_attempts(history),
        duration=time.perf_counter() - started_at,
    )
