"""
RENIX Debugger
==============

Debugging engine for RENIX.

Responsibilities:
    - Run Python programs safely
    - Capture stdout/stderr
    - Capture exit codes
    - Detect Python syntax errors
    - Extract traceback information
    - Identify likely error locations
    - Provide structured debugging results
    - Run commands with timeout protection

This module does not automatically modify source code.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class DebugError:
    """Structured representation of a debugging error."""

    error_type: str
    message: str
    file_path: str | None = None
    line_number: int | None = None
    column: int | None = None
    traceback_text: str | None = None
    source_line: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "traceback": self.traceback_text,
            "source_line": self.source_line,
        }


@dataclass
class DebugResult:
    """Result returned by a debugging operation."""

    success: bool
    command: list[str]
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    errors: list[DebugError] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "errors": [
                error.to_dict()
                for error in self.errors
            ],
        }


@dataclass
class SyntaxResult:
    """Result of a syntax check."""

    valid: bool
    error: DebugError | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error": (
                self.error.to_dict()
                if self.error
                else None
            ),
        }


class Debugger:
    """RENIX debugging and execution engine."""

    TRACEBACK_PATTERN = re.compile(
        r'File "(.+?)", line (\d+), in (.+)'
    )

    ERROR_PATTERN = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Interrupt)):\s*(.*)$",
        re.MULTILINE,
    )

    def __init__(
        self,
        *,
        allowed_root: str | os.PathLike[str] | None = None,
        timeout: float = 30.0,
        max_output_size: int = 2 * 1024 * 1024,
        python_executable: str | None = None,
    ) -> None:

        self.allowed_root = (
            Path(allowed_root)
            .expanduser()
            .resolve()
            if allowed_root is not None
            else None
        )

        self.timeout = timeout
        self.max_output_size = max_output_size
        self.python_executable = (
            python_executable
            or sys.executable
        )

    # =========================================================
    # PYTHON SYNTAX CHECK
    # =========================================================

    def check_syntax(
        self,
        file_path: str | os.PathLike[str],
    ) -> SyntaxResult:

        path = self._validate_file(
            file_path
        )

        if path.suffix.lower() not in {
            ".py",
            ".pyw",
        }:
            return SyntaxResult(
                valid=True
            )

        try:

            source = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            ast.parse(
                source,
                filename=str(path),
            )

            return SyntaxResult(
                valid=True
            )

        except SyntaxError as exc:

            source_line = None

            if exc.text:
                source_line = (
                    exc.text.strip()
                )

            error = DebugError(
                error_type="SyntaxError",
                message=exc.msg,
                file_path=str(path),
                line_number=exc.lineno,
                column=exc.offset,
                source_line=source_line,
            )

            return SyntaxResult(
                valid=False,
                error=error,
            )

        except Exception as exc:

            return SyntaxResult(
                valid=False,
                error=DebugError(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    file_path=str(path),
                ),
            )

    # =========================================================
    # RUN PYTHON
    # =========================================================

    def run_python(
        self,
        file_path: str | os.PathLike[str],
        *,
        arguments: Sequence[str] | None = None,
        timeout: float | None = None,
        working_directory: str | os.PathLike[str] | None = None,
        environment: dict[str, str] | None = None,
    ) -> DebugResult:

        path = self._validate_file(
            file_path
        )

        syntax = self.check_syntax(
            path
        )

        if not syntax.valid:

            return DebugResult(
                success=False,
                command=[
                    self.python_executable,
                    str(path),
                ],
                stderr=(
                    syntax.error.message
                    if syntax.error
                    else "Syntax error."
                ),
                errors=[
                    syntax.error
                ]
                if syntax.error
                else [],
            )

        command = [
            self.python_executable,
            str(path),
        ]

        if arguments:
            command.extend(
                str(argument)
                for argument in arguments
            )

        return self.run_command(
            command,
            timeout=timeout,
            working_directory=(
                working_directory
                or path.parent
            ),
            environment=environment,
        )

    # =========================================================
    # RUN COMMAND
    # =========================================================

    def run_command(
        self,
        command: Sequence[str],
        *,
        timeout: float | None = None,
        working_directory: str | os.PathLike[str] | None = None,
        environment: dict[str, str] | None = None,
    ) -> DebugResult:

        if not command:
            raise ValueError(
                "command cannot be empty."
            )

        normalized_command = [
            str(item)
            for item in command
        ]

        cwd = None

        if working_directory is not None:

            cwd_path = (
                Path(working_directory)
                .expanduser()
                .resolve()
            )

            if self.allowed_root is not None:
                self._ensure_inside_root(
                    cwd_path
                )

            if not cwd_path.exists():
                raise FileNotFoundError(
                    f"Working directory does not exist: {cwd_path}"
                )

            if not cwd_path.is_dir():
                raise NotADirectoryError(
                    str(cwd_path)
                )

            cwd = str(cwd_path)

        env = os.environ.copy()

        if environment:
            env.update(
                {
                    str(key): str(value)
                    for key, value in environment.items()
                }
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

        logger.info(
            "Running debug command: %s",
            normalized_command,
        )

        start = __import__(
            "time"
        ).perf_counter()

        try:

            process = subprocess.run(
                normalized_command,
                cwd=cwd,
                env=env,
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

            stdout = self._limit_output(
                process.stdout
            )

            stderr = self._limit_output(
                process.stderr
            )

            errors = self.parse_errors(
                stderr
            )

            success = (
                process.returncode == 0
            )

            return DebugResult(
                success=success,
                command=normalized_command,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                errors=errors,
            )

        except subprocess.TimeoutExpired as exc:

            duration = (
                __import__(
                    "time"
                ).perf_counter()
                - start
            )

            stdout = self._convert_output(
                exc.stdout
            )

            stderr = self._convert_output(
                exc.stderr
            )

            error = DebugError(
                error_type="TimeoutError",
                message=(
                    f"Process exceeded timeout "
                    f"of {effective_timeout} seconds."
                ),
                traceback_text=stderr,
            )

            return DebugResult(
                success=False,
                command=normalized_command,
                return_code=None,
                stdout=self._limit_output(
                    stdout
                ),
                stderr=self._limit_output(
                    stderr
                ),
                duration_seconds=duration,
                timed_out=True,
                errors=[error],
            )

        except FileNotFoundError as exc:

            return DebugResult(
                success=False,
                command=normalized_command,
                stderr=str(exc),
                errors=[
                    DebugError(
                        error_type="FileNotFoundError",
                        message=str(exc),
                    )
                ],
            )

        except PermissionError as exc:

            return DebugResult(
                success=False,
                command=normalized_command,
                stderr=str(exc),
                errors=[
                    DebugError(
                        error_type="PermissionError",
                        message=str(exc),
                    )
                ],
            )

        except Exception as exc:

            logger.exception(
                "Unexpected debugger failure."
            )

            return DebugResult(
                success=False,
                command=normalized_command,
                stderr=str(exc),
                errors=[
                    DebugError(
                        error_type=type(exc).__name__,
                        message=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                ],
            )

    # =========================================================
    # ERROR PARSING
    # =========================================================

    def parse_errors(
        self,
        stderr: str,
    ) -> list[DebugError]:

        if not stderr:
            return []

        errors: list[DebugError] = []

        traceback_matches = list(
            self.TRACEBACK_PATTERN.finditer(
                stderr
            )
        )

        final_error = None

        matches = list(
            self.ERROR_PATTERN.finditer(
                stderr
            )
        )

        if matches:
            final_error = matches[-1]

        if traceback_matches:

            last_frame = traceback_matches[-1]

            file_path = last_frame.group(
                1
            )

            line_number = int(
                last_frame.group(
                    2
                )
            )

            error_type = (
                final_error.group(1)
                if final_error
                else "RuntimeError"
            )

            message = (
                final_error.group(2)
                if final_error
                else stderr.strip().splitlines()[-1]
            )

            source_line = self._get_source_line(
                file_path,
                line_number,
            )

            errors.append(
                DebugError(
                    error_type=error_type,
                    message=message,
                    file_path=file_path,
                    line_number=line_number,
                    traceback_text=stderr,
                    source_line=source_line,
                )
            )

            return errors

        if final_error:

            errors.append(
                DebugError(
                    error_type=final_error.group(1),
                    message=final_error.group(2),
                    traceback_text=stderr,
                )
            )

            return errors

        errors.append(
            DebugError(
                error_type="UnknownError",
                message=stderr.strip(),
                traceback_text=stderr,
            )
        )

        return errors

    # =========================================================
    # SOURCE LINE
    # =========================================================

    def _get_source_line(
        self,
        file_path: str,
        line_number: int,
    ) -> str | None:

        try:

            path = Path(
                file_path
            ).expanduser().resolve()

            if not path.exists():
                return None

            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

            if 1 <= line_number <= len(lines):
                return lines[
                    line_number - 1
                ].strip()

        except Exception:

            logger.debug(
                "Unable to read source line.",
                exc_info=True,
            )

        return None

    # =========================================================
    # DEBUGGING HELPERS
    # =========================================================

    def inspect_python(
        self,
        file_path: str | os.PathLike[str],
    ) -> dict[str, Any]:

        path = self._validate_file(
            file_path
        )

        if path.suffix.lower() not in {
            ".py",
            ".pyw",
        }:
            raise ValueError(
                "inspect_python requires a Python file."
            )

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        tree = ast.parse(
            source,
            filename=str(path),
        )

        functions = []
        classes = []
        imports = []

        for node in ast.walk(
            tree
        ):

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):

                functions.append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "async": isinstance(
                            node,
                            ast.AsyncFunctionDef,
                        ),
                    }
                )

            elif isinstance(
                node,
                ast.ClassDef,
            ):

                classes.append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                    }
                )

            elif isinstance(
                node,
                ast.Import,
            ):

                imports.extend(
                    alias.name
                    for alias in node.names
                )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):

                if node.module:
                    imports.append(
                        node.module
                    )

        return {
            "path": str(path),
            "functions": functions,
            "classes": classes,
            "imports": sorted(
                set(imports)
            ),
        }

    # =========================================================
    # OUTPUT CONTROL
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
            text[
                :self.max_output_size
            ]
            + "\n\n"
            "[RENIX] Output truncated."
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

    # =========================================================
    # PATH SAFETY
    # =========================================================

    def _validate_file(
        self,
        file_path: str | os.PathLike[str],
    ) -> Path:

        path = (
            Path(file_path)
            .expanduser()
            .resolve()
        )

        if self.allowed_root is not None:
            self._ensure_inside_root(
                path
            )

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        if not path.is_file():
            raise IsADirectoryError(
                f"Expected a file: {path}"
            )

        return path

    def _ensure_inside_root(
        self,
        path: Path,
    ) -> None:

        if self.allowed_root is None:
            return

        try:

            path.relative_to(
                self.allowed_root
            )

        except ValueError as exc:

            raise PermissionError(
                "Path is outside the allowed RENIX workspace."
            ) from exc


__all__ = [
    "Debugger",
    "DebugError",
    "DebugResult",
    "SyntaxResult",
]


