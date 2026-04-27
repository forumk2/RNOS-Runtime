from dataclasses import dataclass

from agent_runtime.types import Task


@dataclass(frozen=True)
class Scenario:
    name: str
    task: Task


def terrain_failure() -> Task:
    return Task(prompt="build a terrain system")


def simple_success() -> Task:
    return Task(prompt="create a hello world python file")


def stuck_loop() -> Task:
    return Task(prompt="fix broken python code with repeated syntax error")


def get_scenarios() -> list[Scenario]:
    return [
        Scenario(name="terrain_failure", task=terrain_failure()),
        Scenario(name="simple_success", task=simple_success()),
        Scenario(name="stuck_loop", task=stuck_loop()),
    ]
