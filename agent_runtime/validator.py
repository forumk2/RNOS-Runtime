import logging
from pathlib import Path

from .ast_diff import flatten_ast
from .ast_similarity import ast_fingerprint
from .types import ExecutionResult
from .utils import ensure_workspace


logger = logging.getLogger(__name__)


def _python_files(workspace: Path) -> list[Path]:
    return sorted(workspace.glob("step_*.py"))


def validate() -> ExecutionResult:
    workspace = ensure_workspace()
    files = _python_files(workspace)

    if not files:
        return ExecutionResult(
            success=False,
            output="workspace validation failed",
            error=f"no generated files found in {workspace}",
        )

    latest = files[-1]
    fingerprint = None
    tokens = None
    try:
        source = latest.read_text(encoding="utf-8")
        fingerprint = ast_fingerprint(source)
        tokens = flatten_ast(source)
        compile(source, str(latest), "exec")
    except (OSError, SyntaxError) as exc:
        logger.warning("validator.syntax_failed", extra={"path": str(latest)})
        return ExecutionResult(
            success=False,
            output=f"validated {latest}",
            error=str(exc),
            artifact_path=str(latest),
            ast_fingerprint=fingerprint,
            ast_tokens=tokens,
            failure_type=type(exc).__name__,
        )

    logger.info("validator.syntax_ok", extra={"path": str(latest)})
    return ExecutionResult(
        success=True,
        output=f"validated {latest}",
        artifact_path=str(latest),
        ast_fingerprint=fingerprint,
        ast_tokens=tokens,
    )
