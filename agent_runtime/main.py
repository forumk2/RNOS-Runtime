import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent_runtime.runner import run_task
    from agent_runtime.types import Task
    from agent_runtime.utils import configure_logging
else:
    from .runner import run_task
    from .types import Task
    from .utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an RNOS-governed agent task.")
    parser.add_argument("prompt", help="Development task prompt for the agent runtime.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    run_task(Task(prompt=args.prompt))


if __name__ == "__main__":
    main()
