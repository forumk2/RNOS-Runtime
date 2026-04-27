from dataclasses import dataclass


@dataclass(frozen=True)
class LangChainScenario:
    name: str
    task: str
    scenario_type: str


def syntax_error_fix() -> LangChainScenario:
    return LangChainScenario(
        name="syntax_error_fix",
        task="fix broken python code with syntax errors",
        scenario_type="failure_loop",
    )


def terrain_failure() -> LangChainScenario:
    return LangChainScenario(
        name="terrain_failure",
        task="build a terrain system",
        scenario_type="failure_loop",
    )


def invalid_python_repair() -> LangChainScenario:
    return LangChainScenario(
        name="invalid_python_repair",
        task="write invalid python and fix it",
        scenario_type="failure_loop",
    )


def tool_loop_spam() -> LangChainScenario:
    return LangChainScenario(
        name="tool_loop_spam",
        task="Search for the weather in Atlantis and summarize it.",
        scenario_type="failure_loop",
    )


def hallucinated_research() -> LangChainScenario:
    return LangChainScenario(
        name="hallucinated_research",
        task="Find peer-reviewed papers on quantum dragons and summarize them.",
        scenario_type="hallucination",
    )


def planning_drift() -> LangChainScenario:
    return LangChainScenario(
        name="planning_drift",
        task="Plan and implement a file parser for a custom format.",
        scenario_type="planning_drift",
    )


def simple_success() -> LangChainScenario:
    return LangChainScenario(
        name="simple_success",
        task="Write a Python function that returns the square of a number.",
        scenario_type="success",
    )


def get_scenarios() -> list[LangChainScenario]:
    return [
        syntax_error_fix(),
        terrain_failure(),
        invalid_python_repair(),
        tool_loop_spam(),
        hallucinated_research(),
        planning_drift(),
        simple_success(),
    ]
