"""
RENIX Build Manager
===================

Handles project build operations for RENIX.

Responsibilities:
    - Detect project type
    - Run configured build commands
    - Build Python projects
    - Build/package Python applications
    - Capture build output
    - Enforce build timeouts
    - Return structured build results
    - Validate build prerequisites

This module does not deploy applications.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class BuildIssue:
    """A warning or error produced during a build."""

    severity: str
    message: str
    file_path: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
        }


@dataclass
class BuildResult:
    """Result of a build operation."""

    success: bool
    project_type: str
    command: list[str]
    return_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    artifacts: list[str] = field(
        default_factory=list
    )
    issues: list[BuildIssue] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "project_type": self.project_type,
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "artifacts": self.artifacts,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


@dataclass
class ProjectInfo:
    """Detected information about a project."""

    root: str
    project_type: str
    entry_point: str | None = None
    build_system: str | None = None
    package_name: str | None = None
    has_requirements: bool = False
    has_pyproject: bool = False
    has_setup_py: bool = False
    has_package_json: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "project_type": self.project_type,
            "entry_point": self.entry_point,
            "build_system": self.build_system,
            "package_name": self.package_name,
            "has_requirements": self.has_requirements,
            "has_pyproject": self.has_pyproject,
            "has_setup_py": self.has_setup_py,
            "has_package_json": self.has_package_json,
        }


class BuildManager:
    """RENIX project build engine."""

    DEFAULT_TIMEOUT = 600.0

    def __init__(
        self,
        *,
        project_root: str | os.PathLike[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_output_size: int = 4 * 1024 * 1024,
    ) -> None:

        self.project_root = (
            Path(project_root)
            .expanduser()
            .resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )

        self.timeout = timeout
        self.max_output_size = max_output_size

    # =========================================================
    # PROJECT DETECTION
    # =========================================================

    def inspect_project(
        self,
        project_root: str | os.PathLike[str] | None = None,
    ) -> ProjectInfo:

        root = self._validate_directory(
            project_root
        )

        pyproject = root / "pyproject.toml"
        setup_py = root / "setup.py"
        requirements = root / "requirements.txt"
        package_json = root / "package.json"

        entry_point = self._find_python_entry_point(
            root
        )

        if pyproject.exists():
            project_type = "python"
            build_system = "pyproject"

        elif setup_py.exists():
            project_type = "python"
            build_system = "setuptools"

        elif package_json.exists():
            project_type = "javascript"
            build_system = "npm"

        elif entry_point:
            project_type = "python"
            build_system = "python"

        else:
            project_type = "unknown"
            build_system = None

        package_name = self._detect_package_name(
            root
        )

        return ProjectInfo(
            root=str(root),
            project_type=project_type,
            entry_point=entry_point,
            build_system=build_system,
            package_name=package_name,
            has_requirements=requirements.exists(),
            has_pyproject=pyproject.exists(),
            has_setup_py=setup_py.exists(),
            has_package_json=package_json.exists(),
        )

    def _find_python_entry_point(
        self,
        root: Path,
    ) -> str | None:

        preferred = [
            "main.py",
            "app.py",
            "run.py",
            "start.py",
        ]

        for filename in preferred:

            candidate = root / filename

            if candidate.is_file():
                return str(candidate)

        return None

    def _detect_package_name(
        self,
        root: Path,
    ) -> str | None:

        candidates = [
            root / "pyproject.toml",
            root / "setup.py",
        ]

        for path in candidates:

            if not path.exists():
                continue

            try:

                text = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

            except OSError:
                continue

            for line in text.splitlines():

                stripped = line.strip()

                if (
                    stripped.startswith(
                        "name ="
                    )
                    or stripped.startswith(
                        "name="
                    )
                ):

                    value = (
                        stripped.split(
                            "=",
                            1
                        )[1]
                        .strip()
                        .strip("\"'")
                    )

                    if value:
                        return value

        return None

    # =========================================================
    # MAIN BUILD API
    # =========================================================

    def build(
        self,
        project_root: str | os.PathLike[str] | None = None,
        *,
        command: Sequence[str] | None = None,
        timeout: float | None = None,
        clean: bool = False,
    ) -> BuildResult:

        root = self._validate_directory(
            project_root
        )

        info = self.inspect_project(
            root
        )

        if clean:
            self.clean(
                root
            )

        if command is not None:

            return self.run_command(
                list(command),
                project_type=info.project_type,
                project_root=root,
                timeout=timeout,
            )

        if info.project_type == "python":

            return self.build_python(
                root,
                timeout=timeout,
            )

        if info.project_type == "javascript":

            return self.build_javascript(
                root,
                timeout=timeout,
            )

        return BuildResult(
            success=False,
            project_type="unknown",
            command=[],
            return_code=None,
            stdout="",
            stderr=(
                "RENIX could not determine the "
                "project build system."
            ),
            duration_seconds=0.0,
            issues=[
                BuildIssue(
                    severity="error",
                    message=(
                        "No supported build configuration "
                        "was detected."
                    ),
                )
            ],
        )

    # =========================================================
    # PYTHON BUILD
    # =========================================================

    def build_python(
        self,
        project_root: str | os.PathLike[str] | None = None,
        *,
        timeout: float | None = None,
    ) -> BuildResult:

        root = self._validate_directory(
            project_root
        )

        info = self.inspect_project(
            root
        )

        # Prefer standard Python build
        # when pyproject.toml exists.
        if info.has_pyproject:

            build_module = (
                importlib.util.find_spec(
                    "build"
                )
                is not None
            )

            if build_module:

                command = [
                    sys.executable,
                    "-m",
                    "build",
                ]

            else:

                command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "--no-deps",
                ]

        elif info.has_setup_py:

            command = [
                sys.executable,
                "setup.py",
                "build",
            ]

        elif info.entry_point:

            command = [
                sys.executable,
                "-m",
                "py_compile",
                info.entry_point,
            ]

        else:

            return BuildResult(
                success=False,
                project_type="python",
                command=[],
                return_code=None,
                stdout="",
                stderr=(
                    "No Python build configuration found."
                ),
                duration_seconds=0.0,
            )

        return self.run_command(
            command,
            project_type="python",
            project_root=root,
            timeout=timeout,
        )

    # =========================================================
    # JAVASCRIPT BUILD
    # =========================================================

    def build_javascript(
        self,
        project_root: str | os.PathLike[str] | None = None,
        *,
        timeout: float | None = None,
    ) -> BuildResult:

        root = self._validate_directory(
            project_root
        )

        package_json = (
            root / "package.json"
        )

        if not package_json.exists():

            return BuildResult(
                success=False,
                project_type="javascript",
                command=[],
                return_code=None,
                stdout="",
                stderr=(
                    "package.json was not found."
                ),
                duration_seconds=0.0,
            )

        command = [
            "npm",
            "run",
            "build",
        ]

        return self.run_command(
            command,
            project_type="javascript",
            project_root=root,
            timeout=timeout,
        )

    # =========================================================
    # COMMAND EXECUTION
    # =========================================================

    def run_command(
        self,
        command: Sequence[str],
        *,
        project_type: str = "custom",
        project_root: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
        environment: dict[str, str] | None = None,
    ) -> BuildResult:

        if not command:
            raise ValueError(
                "Build command cannot be empty."
            )

        root = self._validate_directory(
            project_root
        )

        effective_timeout = (
            self.timeout
            if timeout is None
            else timeout
        )

        if effective_timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        normalized_command = [
            str(item)
            for item in command
        ]

        env = os.environ.copy()

        if environment:
            env.update(
                {
                    str(key): str(value)
                    for key, value in environment.items()
                }
            )

        logger.info(
            "Starting build: %s",
            normalized_command,
        )

        start = time.perf_counter()

        try:

            process = subprocess.run(
                normalized_command,
                cwd=str(root),
                env=env,
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

            issues = self._parse_issues(
                stdout,
                stderr,
            )

            artifacts = self.find_artifacts(
                root
            )

            return BuildResult(
                success=(
                    process.returncode == 0
                ),
                project_type=project_type,
                command=normalized_command,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                artifacts=artifacts,
                issues=issues,
            )

        except subprocess.TimeoutExpired as exc:

            duration = (
                time.perf_counter()
                - start
            )

            return BuildResult(
                success=False,
                project_type=project_type,
                command=normalized_command,
                return_code=None,
                stdout=self._convert_output(
                    exc.stdout
                ),
                stderr=self._convert_output(
                    exc.stderr
                ),
                duration_seconds=duration,
                timed_out=True,
                issues=[
                    BuildIssue(
                        severity="error",
                        message=(
                            f"Build exceeded timeout "
                            f"of {effective_timeout} seconds."
                        ),
                    )
                ],
            )

        except FileNotFoundError as exc:

            return BuildResult(
                success=False,
                project_type=project_type,
                command=normalized_command,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=(
                    time.perf_counter()
                    - start
                ),
                issues=[
                    BuildIssue(
                        severity="error",
                        message=(
                            "Build executable was not found: "
                            f"{normalized_command[0]}"
                        ),
                    )
                ],
            )

        except PermissionError as exc:

            return BuildResult(
                success=False,
                project_type=project_type,
                command=normalized_command,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=(
                    time.perf_counter()
                    - start
                ),
                issues=[
                    BuildIssue(
                        severity="error",
                        message=str(exc),
                    )
                ],
            )

        except Exception as exc:

            logger.exception(
                "Unexpected build failure."
            )

            return BuildResult(
                success=False,
                project_type=project_type,
                command=normalized_command,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=(
                    time.perf_counter()
                    - start
                ),
                issues=[
                    BuildIssue(
                        severity="error",
                        message=str(exc),
                    )
                ],
            )

    # =========================================================
    # CLEAN
    # =========================================================

    def clean(
        self,
        project_root: str | os.PathLike[str] | None = None,
    ) -> list[str]:

        root = self._validate_directory(
            project_root
        )

        removed: list[str] = []

        directories = [
            "build",
            "dist",
            ".pytest_cache",
        ]

        for directory_name in directories:

            directory = (
                root / directory_name
            )

            if not directory.exists():
                continue

            self._remove_directory(
                directory
            )

            removed.append(
                str(directory)
            )

        for path in root.rglob(
            "__pycache__"
        ):

            if path.is_dir():

                self._remove_directory(
                    path
                )

                removed.append(
                    str(path)
                )

        for path in root.rglob(
            "*.pyc"
        ):

            if path.is_file():

                try:
                    path.unlink()
                    removed.append(
                        str(path)
                    )
                except OSError:
                    logger.warning(
                        "Unable to remove %s",
                        path,
                    )

        return removed

    def _remove_directory(
        self,
        path: Path,
    ) -> None:

        import shutil

        try:
            shutil.rmtree(
                path
            )
        except OSError as exc:
            logger.warning(
                "Unable to remove %s: %s",
                path,
                exc,
            )

    # =========================================================
    # ARTIFACT DETECTION
    # =========================================================

    def find_artifacts(
        self,
        project_root: str | os.PathLike[str] | None = None,
    ) -> list[str]:

        root = self._validate_directory(
            project_root
        )

        artifact_directories = {
            "build",
            "dist",
        }

        artifacts: list[str] = []

        for directory_name in artifact_directories:

            directory = (
                root / directory_name
            )

            if not directory.is_dir():
                continue

            for path in directory.rglob(
                "*"
            ):

                if path.is_file():
                    artifacts.append(
                        str(path)
                    )

        return sorted(
            artifacts
        )

    # =========================================================
    # ISSUE PARSING
    # =========================================================

    def _parse_issues(
        self,
        stdout: str,
        stderr: str,
    ) -> list[BuildIssue]:

        issues: list[BuildIssue] = []

        combined = (
            stdout
            + "\n"
            + stderr
        )

        for line in combined.splitlines():

            stripped = line.strip()

            if not stripped:
                continue

            lower = stripped.lower()

            if (
                "error" in lower
                or "failed" in lower
                or "failure" in lower
            ):

                issues.append(
                    BuildIssue(
                        severity="error",
                        message=stripped,
                    )
                )

            elif (
                "warning" in lower
                or "deprecated" in lower
            ):

                issues.append(
                    BuildIssue(
                        severity="warning",
                        message=stripped,
                    )
                )

        return self._deduplicate_issues(
            issues
        )

    @staticmethod
    def _deduplicate_issues(
        issues: list[BuildIssue],
    ) -> list[BuildIssue]:

        result: list[BuildIssue] = []
        seen: set[str] = set()

        for issue in issues:

            key = (
                f"{issue.severity}:"
                f"{issue.message}"
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(issue)

        return result

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_directory(
        self,
        path: str | os.PathLike[str] | None,
    ) -> Path:

        target = (
            self.project_root
            if path is None
            else Path(path)
            .expanduser()
            .resolve()
        )

        if not target.exists():
            raise FileNotFoundError(
                f"Project directory does not exist: {target}"
            )

        if not target.is_dir():
            raise NotADirectoryError(
                f"Expected project directory: {target}"
            )

        return target

    # =========================================================
    # OUTPUT HELPERS
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
            "[RENIX] Build output truncated."
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
    "BuildManager",
    "BuildResult",
    "BuildIssue",
    "ProjectInfo",
]


