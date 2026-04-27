import logging
from pathlib import Path

from .ast_change_vector import extract_features
from .ast_diff import flatten_ast
from .ast_similarity import ast_fingerprint
from .types import ExecutionResult
from .utils import ensure_workspace
from .tool_executor import ToolExecutionResult


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
    features = None
    try:
        source = latest.read_text(encoding="utf-8")
        fingerprint = ast_fingerprint(source)
        tokens = flatten_ast(source)
        features = extract_features(source)
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
            ast_features=features,
            failure_type=type(exc).__name__,
        )

    logger.info("validator.syntax_ok", extra={"path": str(latest)})
    return ExecutionResult(
        success=True,
        output=f"validated {latest}",
        artifact_path=str(latest),
        ast_fingerprint=fingerprint,
        ast_tokens=tokens,
        ast_features=features,
    )


class GateValidator:
    """Validation simulator for deterministic Agent Gate runs."""

    def validate(
        self,
        result: ToolExecutionResult,
        recent_errors: list[str] | None = None,
    ) -> dict[str, object]:
        recent_errors = recent_errors or []

        if result.destructive:
            return {
                "success": False,
                "error": "destructive action rejected by validation",
                "confidence": 0.05,
                "partial_success": False,
            }

        if result.success:
            return {
                "success": True,
                "error": "",
                "confidence": 0.95,
                "partial_success": bool(result.metadata.get("partial_success", False)),
            }

        repeated = result.error in recent_errors[-3:]
        confidence = 0.2 if repeated else 0.35
        error = result.error
        if repeated:
            error = f"repeated failure: {error}"

        return {
            "success": False,
            "error": error,
            "confidence": confidence,
            "partial_success": bool(result.metadata.get("partial_success", False)),
        }
