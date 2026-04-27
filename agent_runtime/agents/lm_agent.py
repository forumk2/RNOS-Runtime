"""LM Studio backed planner for the RNOS real repository loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Protocol


class PlanLike(Protocol):
    action: str
    description: str
    target: str
    diff: str


@dataclass(frozen=True)
class LMAgentConfig:
    base_url: str = field(default_factory=lambda: os.getenv("RNOS_LM_BASE_URL", "http://127.0.0.1:1234/v1"))
    model: str = field(default_factory=lambda: os.getenv("RNOS_LM_MODEL", "local-model"))
    api_key: str = field(default_factory=lambda: os.getenv("RNOS_LM_API_KEY", "lm-studio"))
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout: float = field(default_factory=lambda: float(os.getenv("RNOS_LM_TIMEOUT", "60")))


class LMAgent:
    """OpenAI-compatible LM Studio agent that returns JSON-only plans."""

    allowed_actions = {"read_file", "edit_file", "run_tests", "finish"}

    def __init__(
        self,
        scenario: Any,
        *,
        repo_root: Path,
        plan_type: type[PlanLike],
        config: LMAgentConfig | None = None,
    ) -> None:
        self.scenario = scenario
        self.repo_root = repo_root.resolve()
        self.plan_type = plan_type
        self.config = config or LMAgentConfig()
        self._client = self._build_client()

    def plan(self, state: Any) -> PlanLike | None:
        payload = self._request_plan(state)
        action = str(payload.get("action", "")).strip()
        if action not in self.allowed_actions:
            raise ValueError(f"LM agent returned unsupported action: {action!r}")
        if action == "finish":
            return None

        target = str(payload.get("target", "")).strip()
        if action in {"read_file", "edit_file"} and not target:
            raise ValueError(f"LM agent action {action} requires a target")
        _validate_relative_target(target)
        if action == "edit_file" and not str(payload.get("diff", "")).lstrip().startswith("--- "):
            raise ValueError("LM agent edit_file action requires a unified diff")

        return self.plan_type(
            action=action,
            description=str(payload.get("description", action)).strip() or action,
            target=target,
            diff=str(payload.get("diff", "")),
        )

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("LMAgent requires the openai package") from exc

        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    def _request_plan(self, state: Any) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt(state)},
                {"role": "user", "content": json.dumps(self._state_payload(state), sort_keys=True)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rnos_plan",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["read_file", "edit_file", "run_tests", "finish"],
                            },
                            "target": {"type": "string"},
                            "description": {"type": "string"},
                            "diff": {"type": "string"},
                        },
                        "required": ["action", "target", "description", "diff"],
                    },
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LM agent returned empty content")
        return _parse_json_object(content)

    def _system_prompt(self, state: Any) -> str:
        prompt = (
            "You are an RNOS-controlled coding agent. Return JSON only. "
            "Never include markdown, prose, shell commands, or comments outside JSON. "
            "Allowed actions are read_file, edit_file, run_tests, finish. "
            "For edit_file, provide a unified diff in the diff field. "
            "Only target files inside sandbox_repo. Do not use absolute paths or path traversal. "
            "Schema: {\"action\": string, \"target\": string, \"description\": string, \"diff\": string}."
        )
        feedback = "\n\n".join(str(item) for item in list(getattr(state, "recovery_feedback", []))[-2:] if item)
        if feedback:
            prompt = (
                f"{prompt}\n\nRNOS RECOVERY FEEDBACK:\n{feedback}\n\n"
                "You are in recovery mode. Use the available_files payload as your file context. "
                "Do not choose read_file again for the same target after a failed test. "
                "Prefer edit_file with the smallest valid unified diff; choose run_tests only after a patch was applied."
            )
        return prompt

    def _state_payload(self, state: Any) -> dict[str, Any]:
        return {
            "objective": getattr(self.scenario, "description", ""),
            "scenario": getattr(self.scenario, "name", "unknown"),
            "attempts": getattr(state, "attempts", 0),
            "retry_count": getattr(state, "retry_count", 0),
            "validation_failures": getattr(state, "validation_failures", 0),
            "recovery_attempts": getattr(state, "recovery_attempts", 0),
            "files_modified": sorted(getattr(state, "files_modified", set())),
            "available_files": _readable_files(self.repo_root),
            "recent_targets": list(getattr(state, "targets", []))[-5:],
            "recent_plans": list(getattr(state, "plan_texts", []))[-5:],
            "recovery_guidance": list(getattr(state, "recovery_guidance", []))[-5:],
            "recovery_feedback": list(getattr(state, "recovery_feedback", []))[-3:],
            "latest_validation_output": str(getattr(state, "latest_validation_output", ""))[-2000:],
        }


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise ValueError("LM agent must return a single JSON object")
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LM agent JSON response must be an object")
    return parsed


def _validate_relative_target(target: str) -> None:
    if not target:
        return
    path = Path(target)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"LM agent target escapes sandbox: {target}")


def _readable_files(repo_root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(repo_root).as_posix()
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        try:
            files[rel] = data.decode("utf-8")[:4000]
        except UnicodeDecodeError:
            continue
    return files


Agent = LMAgent
