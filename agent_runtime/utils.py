import logging
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR / "workspace"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_workspace() -> Path:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def cleanup_workspace() -> None:
    workspace = ensure_workspace()
    for generated_file in workspace.glob("step_*.py"):
        generated_file.unlink()


def step_number(step: str) -> int:
    match = re.match(r"^\s*(\d+)[\.\)]\s+", step)
    if not match:
        return 0
    return int(match.group(1))
