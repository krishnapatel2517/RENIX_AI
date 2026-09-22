"""
RENIX Test Runner
=================

Runs automated tests for RENIX projects.

Responsibilities:
    - Discover Python tests
    - Run pytest when available
    - Fall back to unittest
    - Capture stdout/stderr
    - Enforce timeouts
    - Parse basic test results
    - Return structured results
    - Support single files, directories, and test names

This module executes tests but does not modify source code.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class TestCaseResult:
    """Result of an individual test case."""

    name: str
    status: str
    duration_seconds: float | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
        }


@dataclass
class TestRunResult:
    """Complete result of a test execution."""

    success: bool
    command: list[str]
    framework: str
    return_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    tests_run: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    test_cases: list[TestCaseResult] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "command": self.command,
            "framework": self.framework,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "tests_run": self.tests_run,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "test_cases": [
                case.to_dict()
                for case in self.test_cases
            ],
        }


@dataclass
class TestDiscoveryResult:
    """Result of test discovery."""

    files: list[str] = field(
        default_factory=list
    )
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "count": self.count,
        }


class TestRunner:
    """Automated test runner for RENIX."""

    DEFAULT_TIMEOUT = 300.0

    PYTEST_FILE_PATTERN = re.compile(
        r"^test_.*\.py$|.*_test\.py$"
    )

    PYTEST_SUMMARY_PATTERN = re.compile(
        r"(?P<count>\d+)\s+"
        r"(?P<status>passed|failed|skipped|error|errors?)"
    )

    UNITTEST_SUMMARY_PATTERN = re.compile(
        r"Ran\s+(\d+)\s+tests?"
    )

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_output_size: int = 4 * 1024 * 1024,
        prefer_pytest: bool = True,
    ) -> None:

        self.project_root = (
            Path(project_root)
            .expanduser()
            .resolve()
            if project_root
            else Path.cwd().resolve()
        )

        self.timeout = timeout
        self.max_output_size = max_output_size
        self.prefer_pytest = prefer_pytest

    # =========================================================
    # TEST DISCOVERY
    # =========================================================

    def discover(
        self,
        target: str | Path | None = None,
    ) -> TestDiscoveryResult:

        root = (
            self.project_root
            if target is None
            else Path(target)
            .expanduser()
            .resolve()
        )

        if not root.exists():
            raise FileNotFoundError(
                f"Test target does not exist: {root}"
            )

        if root.is_file():

            if self._is_test_file(root):
                return TestDiscoveryResult(
                    files=[str(root)],
                    count=1,
                )

            return TestDiscoveryResult()

        files: list[str] = []

        ignored = {
            ".git",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            "node_modules",
            ".idea",
            ".vscode",
            "build",
            "dist",
        }

        for path in root.rglob("*.py"):

            if any(
                part in ignored
                for part in path.parts
            ):
                continue

            if self._is_test_file(path):
                files.append(
                    str(path)
                )

        files.sort()

        return TestDiscoveryResult(
            files=files,
            count=len(files),
        )

    def _is_test_file(
        self,
        path: Path,
    ) -> bool:

        return bool(
            self.PYTEST_FILE_PATTERN.match(
                path.name
            )
        )

    # =========================================================
    # RUN TESTS
    # =========================================================

    def run(
        self,
        target: str | Path | None = None,
        *,
        test_name: str | None = None,
        arguments: Sequence[str] | None = None,
        timeout: float | None = None,
        framework: str | None = None,
    ) -> TestRunResult:

        selected_framework = (
            framework
            or self.detect_framework()
        )

        if selected_framework == "pytest":

            return self.run_pytest(
                target=target,
                test_name=test_name,
                arguments=arguments,
                timeout=timeout,
            )

        if selected_framework == "unittest":

            return self.run_unittest(
                target=target,
                test_name=test_name,
                timeout=timeout,
            )

        raise ValueError(
            f"Unsupported test framework: "
            f"{selected_framework}"
        )

    # =========================================================
    # FRAMEWORK DETECTION
    # =========================================================

    def detect_framework(self) -> str:

        pytest_available = (
            importlib.util.find_spec(
                "pytest"
            )
            is not None
        )

        if (
            self.prefer_pytest
            and pytest_available
        ):
            return "pytest"

        return "unittest"

    # =========================================================
    # PYTEST
    # =========================================================

    def run_pytest(
        self,
        target: str | Path | None = None,
        *,
        test_name: str | None = None,
        arguments: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> TestRunResult:

        target_path = (
            self.project_root
            if target is None
            else Path(target)
            .expanduser()
            .resolve()
        )

        command = [
            sys.executable,
            "-m",
            "pytest",
        ]

        if target_path != self.project_root:
            command.append(
                str(target_path)
            )

        else:
            command.append(
                str(self.project_root)
            )

        if test_name:
            command.extend(
                [
                    "-k",
                    test_name,
                ]
            )

        command.extend(
            [
                "-r",
                "a",
                "--tb=short",
            ]
        )

        if arguments:
            command.extend(
                str(argument)
                for argument in arguments
            )

        return self._execute(
            command,
            framework="pytest",
            timeout=timeout,
        )

    # =========================================================
    # UNITTEST
    # =========================================================

    def run_unittest(
        self,
        target: str | Path | None = None,
        *,
        test_name: str | None = None,
        timeout: float | None = None,
    ) -> TestRunResult:

        target_path = (
            self.project_root
            if target is None
            else Path(target)
            .expanduser()
            .resolve()
        )

        command = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
        ]

        if target_path.is_dir():

            command.extend(
                [
                    "-s",
                    str(target_path),
                ]
            )

        elif target_path.is_file():

            module = self._file_to_module(
                target_path
            )

            command.append(
                module
            )

        if test_name:
            command.append(
                test_name
            )

        return self._execute(
            command,
            framework="unittest",
            timeout=timeout,
        )

    # =========================================================
    # COMMAND EXECUTION
    # =========================================================

    def _execute(
        self,
        command: list[str],
        *,
        framework: str,
        timeout: float | None = None,
    ) -> TestRunResult:

        effective_timeout = (
            self.timeout
            if timeout is None
            else timeout
        )

        if effective_timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        logger.info(
            "Running tests: %s",
            command,
        )

        start = time.perf_counter()

        try:

            process = subprocess.run(
                command,
                cwd=str(
                    self.project_root
                ),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )

            duration = (
                time.perf_counter()
                - start
            )

            stdout = self._limit_output(
                process.stdout
            )

            stderr = self._limit_output(
                process.stderr
            )

            result = self._parse_result(
                framework=framework,
                command=command,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
            )

            return result

        except subprocess.TimeoutExpired as exc:

            duration = (
                time.perf_counter()
                - start
            )

            stdout = self._convert_output(
                exc.stdout
            )

            stderr = self._convert_output(
                exc.stderr
            )

            return TestRunResult(
                success=False,
                command=command,
                framework=framework,
                return_code=None,
                stdout=self._limit_output(
                    stdout
                ),
                stderr=self._limit_output(
                    stderr
                ),
                duration_seconds=duration,
                timed_out=True,
            )

        except FileNotFoundError as exc:

            return TestRunResult(
                success=False,
                command=command,
                framework=framework,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=(
                    time.perf_counter()
                    - start
                ),
                errors=1,
            )

        except Exception as exc:

            logger.exception(
                "Test runner failed."
            )

            return TestRunResult(
                success=False,
                command=command,
                framework=framework,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=(
                    time.perf_counter()
                    - start
                ),
                errors=1,
            )

    # =========================================================
    # RESULT PARSING
    # =========================================================

    def _parse_result(
        self,
        *,
        framework: str,
        command: list[str],
        return_code: int,
        stdout: str,
        stderr: str,
        duration: float,
    ) -> TestRunResult:

        combined = (
            stdout
            + "\n"
            + stderr
        )

        if framework == "pytest":
            counts = self._parse_pytest_summary(
                combined
            )

        else:
            counts = self._parse_unittest_summary(
                combined
            )

        passed = counts["passed"]
        failed = counts["failed"]
        skipped = counts["skipped"]
        errors = counts["errors"]

        tests_run = counts["tests_run"]

        success = (
            return_code == 0
            and failed == 0
            and errors == 0
        )

        test_cases = (
            self._parse_test_cases(
                combined,
                framework,
            )
        )

        return TestRunResult(
            success=success,
            command=command,
            framework=framework,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            tests_run=tests_run,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            test_cases=test_cases,
        )

    def _parse_pytest_summary(
        self,
        output: str,
    ) -> dict[str, int]:

        counts = {
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        }

        matches = list(
            self.PYTEST_SUMMARY_PATTERN.finditer(
                output
            )
        )

        for match in matches:

            count = int(
                match.group(
                    "count"
                )
            )

            status = (
                match.group(
                    "status"
                )
                .lower()
            )

            if status == "passed":
                counts["passed"] += count

            elif status == "failed":
                counts["failed"] += count

            elif status == "skipped":
                counts["skipped"] += count

            elif status in {
                "error",
                "errors",
            }:
                counts["errors"] += count

        counts["tests_run"] = (
            counts["passed"]
            + counts["failed"]
            + counts["skipped"]
            + counts["errors"]
        )

        return counts

    def _parse_unittest_summary(
        self,
        output: str,
    ) -> dict[str, int]:

        counts = {
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        }

        match = self.UNITTEST_SUMMARY_PATTERN.search(
            output
        )

        if match:
            counts["tests_run"] = int(
                match.group(1)
            )

        failed_match = re.search(
            r"failures=(\d+)",
            output,
            re.IGNORECASE,
        )

        error_match = re.search(
            r"errors=(\d+)",
            output,
            re.IGNORECASE,
        )

        skipped_match = re.search(
            r"skipped=(\d+)",
            output,
            re.IGNORECASE,
        )

        if failed_match:
            counts["failed"] = int(
                failed_match.group(1)
            )

        if error_match:
            counts["errors"] = int(
                error_match.group(1)
            )

        if skipped_match:
            counts["skipped"] = int(
                skipped_match.group(1)
            )

        counts["passed"] = max(
            0,
            counts["tests_run"]
            - counts["failed"]
            - counts["errors"]
            - counts["skipped"],
        )

        return counts

    # =========================================================
    # TEST CASE PARSING
    # =========================================================

    def _parse_test_cases(
        self,
        output: str,
        framework: str,
    ) -> list[TestCaseResult]:

        cases: list[TestCaseResult] = []

        if framework == "pytest":

            pattern = re.compile(
                r"^(?P<name>.+?)\s+"
                r"(?P<status>PASSED|FAILED|SKIPPED|ERROR)"
                r"(?:\s+\[(?P<duration>[0-9.]+)s\])?$",
                re.MULTILINE,
            )

            for match in pattern.finditer(
                output
            ):

                duration = match.group(
                    "duration"
                )

                cases.append(
                    TestCaseResult(
                        name=match.group(
                            "name"
                        ).strip(),
                        status=match.group(
                            "status"
                        ).lower(),
                        duration_seconds=(
                            float(duration)
                            if duration
                            else None
                        ),
                    )
                )

        return cases

    # =========================================================
    # MODULE CONVERSION
    # =========================================================

    def _file_to_module(
        self,
        file_path: Path,
    ) -> str:

        try:

            relative = file_path.relative_to(
                self.project_root
            )

        except ValueError:

            return file_path.stem

        parts = list(
            relative.with_suffix(
                ""
            ).parts
        )

        if parts and parts[-1] == "__init__":
            parts.pop()

        return ".".join(
            parts
        )

    # =========================================================
    # OUTPUT
    # =========================================================

    def _limit_output(
        self,
        output: str | bytes | None,
    ) -> str:

        text = self._convert_output(
            output
        )

        if len(text) <= self.max_output_size:
            return text

        return (
            text[:self.max_output_size]
            + "\n\n"
            "[RENIX] Test output truncated."
        )

    @staticmethod
    def _convert_output(
        output: str | bytes | None,
    ) -> str:

        if output is None:
            return ""

        if isinstance(
            output,
            bytes,
        ):
            return output.decode(
                "utf-8",
                errors="replace",
            )

        return str(output)


__all__ = [
    "TestRunner",
    "TestRunResult",
    "TestCaseResult",
    "TestDiscoveryResult",
]


