"""
RENIX Dependency Manager
========================

Manages project dependencies for RENIX coding workflows.

Responsibilities:
    - Detect dependency files
    - Read Python requirements
    - Check installed packages
    - Install dependencies
    - Uninstall dependencies
    - Upgrade dependencies
    - Generate requirements files
    - Validate missing dependencies
    - Run dependency commands safely

This module does not automatically execute arbitrary dependency
commands supplied by an external source without explicit approval.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Dependency:
    """Represents a project dependency."""

    name: str
    version_specifier: str | None = None
    source: str | None = None
    optional: bool = False

    @property
    def requirement(self) -> str:
        """Return a pip-compatible requirement string."""

        if self.version_specifier:
            return (
                f"{self.name}"
                f"{self.version_specifier}"
            )

        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version_specifier": self.version_specifier,
            "source": self.source,
            "optional": self.optional,
            "requirement": self.requirement,
        }


@dataclass
class InstalledPackage:
    """Information about an installed package."""

    name: str
    version: str
    location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "location": self.location,
        }


@dataclass
class DependencyIssue:
    """Dependency validation issue."""

    package: str
    issue_type: str
    message: str
    required_version: str | None = None
    installed_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "issue_type": self.issue_type,
            "message": self.message,
            "required_version": self.required_version,
            "installed_version": self.installed_version,
        }


@dataclass
class DependencyOperationResult:
    """Result of a dependency operation."""

    success: bool
    operation: str
    command: list[str]
    return_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    dependencies: list[Dependency] = field(
        default_factory=list
    )
    issues: list[DependencyIssue] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "dependencies": [
                dependency.to_dict()
                for dependency in self.dependencies
            ],
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


class DependencyManager:
    """RENIX dependency management engine."""

    DEFAULT_TIMEOUT = 300.0

    REQUIREMENT_FILES = (
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "requirements-prod.txt",
    )

    PROJECT_FILES = (
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
    )

    SAFE_PACKAGE_NAME = re.compile(
        r"^[A-Za-z0-9_.-]+$"
    )

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
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
    # DEPENDENCY FILE DISCOVERY
    # =========================================================

    def find_dependency_files(
        self,
        project_root: str | Path | None = None,
    ) -> list[Path]:

        root = self._validate_root(
            project_root
        )

        files: list[Path] = []

        for filename in self.REQUIREMENT_FILES:

            path = root / filename

            if path.is_file():
                files.append(path)

        for filename in self.PROJECT_FILES:

            path = root / filename

            if path.is_file():
                files.append(path)

        return files

    def find_primary_requirements_file(
        self,
        project_root: str | Path | None = None,
    ) -> Path | None:

        root = self._validate_root(
            project_root
        )

        primary = (
            root / "requirements.txt"
        )

        if primary.is_file():
            return primary

        return None

    # =========================================================
    # PARSING
    # =========================================================

    def parse_requirements(
        self,
        requirements_file: str | Path,
    ) -> list[Dependency]:

        path = Path(
            requirements_file
        ).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Requirements file not found: {path}"
            )

        dependencies: list[Dependency] = []

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith(
                (
                    "-r ",
                    "--requirement ",
                    "-c ",
                    "--constraint ",
                    "--index-url ",
                    "--extra-index-url ",
                    "--find-links ",
                    "--trusted-host ",
                )
            ):
                continue

            line = self._remove_inline_comment(
                line
            )

            dependency = (
                self._parse_requirement_line(
                    line,
                    source=str(path),
                )
            )

            if dependency:
                dependencies.append(
                    dependency
                )

        return dependencies

    def _parse_requirement_line(
        self,
        line: str,
        *,
        source: str | None = None,
    ) -> Dependency | None:

        line = line.strip()

        if not line:
            return None

        optional = False

        if line.endswith(
            " # optional"
        ):
            optional = True
            line = line[:-11].strip()

        # Editable/local installs.
        if line.startswith(
            "-e "
        ):
            return Dependency(
                name=line[3:].strip(),
                source=source,
                optional=optional,
            )

        if line.startswith(
            "git+"
        ):
            return Dependency(
                name=line,
                source=source,
                optional=optional,
            )

        # PEP 508 package specifications.
        match = re.match(
            r"^([A-Za-z0-9_.-]+)"
            r"(\[[^\]]+\])?"
            r"((?:===|==|!=|~=|>=|<=|>|<).*)?$",
            line,
        )

        if not match:
            return Dependency(
                name=line,
                source=source,
                optional=optional,
            )

        name = match.group(1)
        extras = match.group(2) or ""
        version = match.group(3)

        return Dependency(
            name=name + extras,
            version_specifier=version,
            source=source,
            optional=optional,
        )

    @staticmethod
    def _remove_inline_comment(
        line: str,
    ) -> str:

        if " #" in line:
            return line.split(
                " #",
                1,
            )[0].strip()

        return line

    # =========================================================
    # INSTALLED PACKAGES
    # =========================================================

    def list_installed(
        self,
    ) -> list[InstalledPackage]:

        packages: list[InstalledPackage] = []

        for distribution in (
            importlib.metadata.distributions()
        ):

            try:
                name = (
                    distribution.metadata.get(
                        "Name"
                    )
                    or distribution.name
                )

                version = (
                    distribution.version
                )

                location = (
                    str(
                        distribution.locate_file("")
                    )
                    if distribution
                    else None
                )

                packages.append(
                    InstalledPackage(
                        name=name,
                        version=version,
                        location=location,
                    )
                )

            except Exception:
                logger.debug(
                    "Unable to inspect package.",
                    exc_info=True,
                )

        packages.sort(
            key=lambda item: item.name.lower()
        )

        return packages

    def get_installed_version(
        self,
        package_name: str,
    ) -> str | None:

        normalized = self._normalize_name(
            package_name
        )

        for distribution in (
            importlib.metadata.distributions()
        ):

            try:

                name = (
                    distribution.metadata.get(
                        "Name"
                    )
                    or distribution.name
                )

                if (
                    self._normalize_name(name)
                    == normalized
                ):
                    return distribution.version

            except Exception:
                continue

        return None

    def is_installed(
        self,
        package_name: str,
    ) -> bool:

        return (
            self.get_installed_version(
                package_name
            )
            is not None
        )

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:

        return re.sub(
            r"[-_.]+",
            "-",
            name.split("[", 1)[0]
            .strip()
            .lower(),
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    def validate(
        self,
        dependencies: Iterable[Dependency],
    ) -> list[DependencyIssue]:

        issues: list[DependencyIssue] = []

        for dependency in dependencies:

            package_name = (
                dependency.name
                .split("[", 1)[0]
            )

            installed = (
                self.get_installed_version(
                    package_name
                )
            )

            if installed is None:

                issues.append(
                    DependencyIssue(
                        package=package_name,
                        issue_type="missing",
                        message=(
                            f"{package_name} is not installed."
                        ),
                        required_version=(
                            dependency.version_specifier
                        ),
                    )
                )

                continue

            version_specifier = (
                dependency.version_specifier
            )

            if version_specifier:

                if not self._version_matches(
                    installed,
                    version_specifier,
                ):

                    issues.append(
                        DependencyIssue(
                            package=package_name,
                            issue_type="version",
                            message=(
                                f"Installed version "
                                f"{installed} does not satisfy "
                                f"{version_specifier}."
                            ),
                            required_version=(
                                version_specifier
                            ),
                            installed_version=installed,
                        )
                    )

        return issues

    def validate_file(
        self,
        requirements_file: str | Path,
    ) -> list[DependencyIssue]:

        dependencies = (
            self.parse_requirements(
                requirements_file
            )
        )

        return self.validate(
            dependencies
        )

    # =========================================================
    # INSTALL
    # =========================================================

    def install(
        self,
        dependencies: Sequence[str | Dependency] | None = None,
        *,
        requirements_file: str | Path | None = None,
        upgrade: bool = False,
        timeout: float | None = None,
    ) -> DependencyOperationResult:

        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
        ]

        parsed_dependencies: list[
            Dependency
        ] = []

        if requirements_file is not None:

            path = Path(
                requirements_file
            ).expanduser().resolve()

            if not path.is_file():
                raise FileNotFoundError(
                    f"Requirements file not found: {path}"
                )

            command.extend(
                [
                    "-r",
                    str(path),
                ]
            )

            parsed_dependencies = (
                self.parse_requirements(
                    path
                )
            )

        elif dependencies:

            parsed_dependencies = (
                self._normalize_dependencies(
                    dependencies
                )
            )

            command.extend(
                dependency.requirement
                for dependency
                in parsed_dependencies
            )

        else:

            primary = (
                self.find_primary_requirements_file()
            )

            if primary is not None:

                command.extend(
                    [
                        "-r",
                        str(primary),
                    ]
                )

                parsed_dependencies = (
                    self.parse_requirements(
                        primary
                    )
                )

            else:

                return DependencyOperationResult(
                    success=False,
                    operation="install",
                    command=command,
                    return_code=None,
                    stdout="",
                    stderr=(
                        "No dependencies or "
                        "requirements.txt were supplied."
                    ),
                    duration_seconds=0.0,
                )

        if upgrade:
            command.insert(
                4,
                "--upgrade",
            )

        return self._execute(
            command,
            operation="install",
            dependencies=parsed_dependencies,
            timeout=timeout,
        )

    # =========================================================
    # UNINSTALL
    # =========================================================

    def uninstall(
        self,
        dependencies: Sequence[str],
        *,
        yes: bool = True,
        timeout: float | None = None,
    ) -> DependencyOperationResult:

        normalized = [
            self._validate_package_name(
                package
            )
            for package in dependencies
        ]

        command = [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
        ]

        if yes:
            command.append("-y")

        command.extend(
            normalized
        )

        parsed = [
            Dependency(name=name)
            for name in normalized
        ]

        return self._execute(
            command,
            operation="uninstall",
            dependencies=parsed,
            timeout=timeout,
        )

    # =========================================================
    # UPGRADE
    # =========================================================

    def upgrade(
        self,
        dependencies: Sequence[str] | None = None,
        *,
        timeout: float | None = None,
    ) -> DependencyOperationResult:

        if dependencies is None:
            dependencies = []

            installed = (
                self.list_installed()
            )

            dependencies.extend(
                package.name
                for package in installed
            )

        normalized = [
            self._validate_package_name(
                package
            )
            for package in dependencies
        ]

        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
        ]

        command.extend(
            normalized
        )

        parsed = [
            Dependency(name=name)
            for name in normalized
        ]

        return self._execute(
            command,
            operation="upgrade",
            dependencies=parsed,
            timeout=timeout,
        )

    # =========================================================
    # REQUIREMENTS GENERATION
    # =========================================================

    def generate_requirements(
        self,
        output_file: str | Path | None = None,
        *,
        freeze: bool = True,
    ) -> Path:

        output = (
            Path(output_file)
            if output_file
            else (
                self.project_root
                / "requirements.txt"
            )
        )

        output = (
            output.expanduser()
            .resolve()
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if freeze:

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "freeze",
                ],
                cwd=str(
                    self.project_root
                ),
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:

                raise RuntimeError(
                    "Unable to generate requirements.txt: "
                    + result.stderr.strip()
                )

            content = result.stdout

        else:

            packages = (
                self.list_installed()
            )

            content = "\n".join(
                package.name
                for package in packages
            )

            if content:
                content += "\n"

        output.write_text(
            content,
            encoding="utf-8",
        )

        return output

    # =========================================================
    # PIP INFORMATION
    # =========================================================

    def check_pip(
        self,
    ) -> bool:

        try:

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--version",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            return result.returncode == 0

        except (
            OSError,
            subprocess.SubprocessError,
        ):
            return False

    def pip_version(
        self,
    ) -> str | None:

        try:

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--version",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if result.returncode != 0:
                return None

            return result.stdout.strip()

        except (
            OSError,
            subprocess.SubprocessError,
        ):
            return None

    # =========================================================
    # COMMAND EXECUTION
    # =========================================================

    def _execute(
        self,
        command: list[str],
        *,
        operation: str,
        dependencies: list[Dependency],
        timeout: float | None,
    ) -> DependencyOperationResult:

        effective_timeout = (
            self.timeout
            if timeout is None
            else timeout
        )

        if effective_timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        start = __import__(
            "time"
        ).perf_counter()

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
                __import__(
                    "time"
                ).perf_counter()
                - start
            )

            return DependencyOperationResult(
                success=(
                    process.returncode == 0
                ),
                operation=operation,
                command=command,
                return_code=process.returncode,
                stdout=self._limit_output(
                    process.stdout
                ),
                stderr=self._limit_output(
                    process.stderr
                ),
                duration_seconds=duration,
                dependencies=dependencies,
                issues=self._parse_issues(
                    process.stdout,
                    process.stderr,
                    dependencies,
                ),
            )

        except subprocess.TimeoutExpired as exc:

            duration = (
                __import__(
                    "time"
                ).perf_counter()
                - start
            )

            return DependencyOperationResult(
                success=False,
                operation=operation,
                command=command,
                return_code=None,
                stdout=self._convert_output(
                    exc.stdout
                ),
                stderr=self._convert_output(
                    exc.stderr
                ),
                duration_seconds=duration,
                dependencies=dependencies,
                issues=[
                    DependencyIssue(
                        package="",
                        issue_type="timeout",
                        message=(
                            f"{operation} operation exceeded "
                            f"{effective_timeout} seconds."
                        ),
                    )
                ],
            )

        except FileNotFoundError as exc:

            return DependencyOperationResult(
                success=False,
                operation=operation,
                command=command,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=(
                    __import__(
                        "time"
                    ).perf_counter()
                    - start
                ),
                dependencies=dependencies,
                issues=[
                    DependencyIssue(
                        package="",
                        issue_type="executable",
                        message=(
                            "Python/pip executable "
                            "could not be started."
                        ),
                    )
                ],
            )

    # =========================================================
    # ISSUE PARSING
    # =========================================================

    def _parse_issues(
        self,
        stdout: str,
        stderr: str,
        dependencies: Sequence[Dependency],
    ) -> list[DependencyIssue]:

        issues: list[DependencyIssue] = []

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
                "no matching distribution" in lower
                or "could not find a version" in lower
            ):

                issues.append(
                    DependencyIssue(
                        package="",
                        issue_type="unavailable",
                        message=stripped,
                    )
                )

            elif (
                "requires" in lower
                and "but" in lower
            ):

                issues.append(
                    DependencyIssue(
                        package="",
                        issue_type="conflict",
                        message=stripped,
                    )
                )

            elif (
                "error:" in lower
                or "error " in lower
            ):

                issues.append(
                    DependencyIssue(
                        package="",
                        issue_type="error",
                        message=stripped,
                    )
                )

        return issues

    # =========================================================
    # HELPERS
    # =========================================================

    def _normalize_dependencies(
        self,
        dependencies: Sequence[
            str | Dependency
        ],
    ) -> list[Dependency]:

        result: list[Dependency] = []

        for dependency in dependencies:

            if isinstance(
                dependency,
                Dependency,
            ):

                result.append(
                    dependency
                )
                continue

            requirement = (
                self._parse_requirement_line(
                    str(dependency)
                )
            )

            if requirement is None:
                continue

            result.append(
                requirement
            )

        return result

    def _validate_package_name(
        self,
        package_name: str,
    ) -> str:

        value = package_name.strip()

        if not value:
            raise ValueError(
                "Package name cannot be empty."
            )

        # Reject command-line options and shell-like
        # values. This manager accepts package names,
        # not arbitrary pip arguments.
        if value.startswith("-"):
            raise ValueError(
                f"Invalid package name: {value}"
            )

        base_name = value.split(
            "[",
            1,
        )[0]

        if not self.SAFE_PACKAGE_NAME.match(
            base_name
        ):
            raise ValueError(
                f"Invalid package name: {value}"
            )

        return value

    def _version_matches(
        self,
        installed: str,
        specifier: str,
    ) -> bool:

        try:

            from packaging.specifiers import (
                SpecifierSet,
            )

            from packaging.version import (
                Version,
            )

            return Version(
                installed
            ) in SpecifierSet(
                specifier
            )

        except Exception:

            # If packaging is unavailable,
            # perform a conservative exact check
            # for == specifications.
            if specifier.startswith("=="):

                return (
                    installed
                    == specifier[2:].strip()
                )

            logger.warning(
                "Unable to evaluate version "
                "specifier %s.",
                specifier,
            )

            return True

    def _validate_root(
        self,
        root: str | Path | None,
    ) -> Path:

        path = (
            self.project_root
            if root is None
            else Path(root)
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Project directory does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"Expected project directory: {path}"
            )

        return path

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
            "[RENIX] Dependency output truncated."
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
    "DependencyManager",
    "Dependency",
    "InstalledPackage",
    "DependencyIssue",
    "DependencyOperationResult",
]


