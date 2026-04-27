"""Subprocess test runner for sandbox repositories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys


@dataclass(frozen=True)
class TestResult:
    success: bool
    output: str
    failures: int

    def as_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "output": self.output,
            "failures": self.failures,
        }


class TestRunner:
    """Runs tests in a sandbox repo without invoking a shell."""

    def __init__(self, repo_root: Path, timeout_seconds: int = 10) -> None:
        self.repo_root = repo_root.resolve()
        self.timeout_seconds = timeout_seconds
        if self.repo_root.name != "sandbox_repo":
            raise ValueError("tests may only run inside ./sandbox_repo")

    def run(self) -> TestResult:
        self._clear_pycache()
        command = [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            ".",
            "-p",
            "test_*.py",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return TestResult(success=False, output=output + "\nTEST TIMEOUT", failures=1)

        output = completed.stdout + completed.stderr
        return TestResult(
            success=completed.returncode == 0,
            output=output.strip(),
            failures=_count_failures(output, completed.returncode),
        )

    def _clear_pycache(self) -> None:
        for path in self.repo_root.rglob("__pycache__"):
            if path.is_dir() and self.repo_root in path.resolve().parents:
                shutil.rmtree(path)


def _count_failures(output: str, return_code: int) -> int:
    if return_code == 0:
        return 0
    markers = output.count("FAIL:") + output.count("ERROR:")
    return max(markers, 1)
