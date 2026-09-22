"""
RENIX Terminal Manager
======================

Provides a controlled terminal interface for RENIX's coding system.

Responsibilities:
    - Execute terminal commands
    - Capture stdout and stderr
    - Support working directories
    - Support environment variables
    - Enforce execution timeouts
    - Track command history
    - Detect command failures
    - Provide structured execution results

Security:
    - Commands are executed without shell=True by default.
    - Arbitrary shell pipelines/redirection require explicit shell mode.
    - Destructive commands can be blocked through a configurable policy.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Structured result returned after command execution."""

    command: list[str]
    success: bool
    return_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    blocked: bool = False
    working_directory: str | None = None

    @property
    def output(self) -> str:
        """Return stdout, falling back to stderr when necessary."""

        if self.stdout.strip():
            return self.stdout

        return self.stderr

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "success": self.success,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "blocked": self.blocked,
            "working_directory": self.working_directory,
        }


@dataclass
class CommandRecord:
    """Historical record of an executed command."""

    timestamp: float
    result: CommandResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "result": self.result.to_dict(),
        }


@dataclass
class TerminalPolicy:
    """Security policy for terminal execution."""

    allow_shell: bool = False
    block_destructive_commands: bool = True
    allow_sudo: bool = False
    allow_shutdown: bool = False
    allow_reboot: bool = False
    max_command_length: int = 4096

    blocked_commands: set[str] = field(
        default_factory=lambda: {
            "format",
            "fdisk",
            "mkfs",
            "diskpart",
        }
    )

    destructive_tokens: tuple[str, ...] = (
        "rm -rf /",
        "rm -rf /*",
        "del /f /s /q",
        "format c:",
        "format c",
        "shutdown",
        "reboot",
        "poweroff",
    )


class TerminalManager:
    """Controlled terminal execution engine for RENIX."""

    DEFAULT_TIMEOUT = 120.0
    MAX_HISTORY = 500

    def __init__(
        self,
        *,
        working_directory: str | os.PathLike[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_output_size: int = 4 * 1024 * 1024,
        policy: TerminalPolicy | None = None,
    ) -> None:

        self.working_directory = (
            Path(working_directory)
            .expanduser()
            .resolve()
            if working_directory is not None
            else Path.cwd().resolve()
        )

        self.timeout = timeout
        self.max_output_size = max_output_size
        self.policy = (
            policy
            if policy is not None
            else TerminalPolicy()
        )

        self.history: list[CommandRecord] = []

    # =========================================================
    # PUBLIC EXECUTION API
    # =========================================================

    def execute(
        self,
        command: str | Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
        environment: Mapping[str, str] | None = None,
        shell: bool = False,
        check: bool = False,
    ) -> CommandResult:
        """
        Execute a terminal command.

        Args:
            command:
                Command string or argument sequence.
            cwd:
                Optional working directory.
            timeout:
                Maximum execution time.
            environment:
                Additional environment variables.
            shell:
                Whether to execute through the system shell.
            check:
                Raise RuntimeError when the command fails.
        """

        normalized = self._normalize_command(
            command,
            shell=shell,
        )

        command_display = (
            normalized
            if isinstance(normalized, list)
            else [normalized]
        )

        validation_error = self._validate_command(
            normalized,
            shell=shell,
        )

        if validation_error:

            result = CommandResult(
                command=command_display,
                success=False,
                return_code=None,
                stdout="",
                stderr=validation_error,
                duration_seconds=0.0,
                blocked=True,
                working_directory=str(
                    cwd
                    or self.working_directory
                ),
            )

            self._record(result)

            if check:
                raise RuntimeError(
                    validation_error
                )

            return result

        target_directory = self._resolve_cwd(
            cwd
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

        env = os.environ.copy()

        if environment:
            env.update(
                {
                    str(key): str(value)
                    for key, value in environment.items()
                }
            )

        start = time.perf_counter()

        logger.info(
            "Executing terminal command: %s",
            self._display_command(
                normalized
            ),
        )

        try:

            process = subprocess.run(
                normalized,
                cwd=str(target_directory),
                env=env,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                shell=shell,
                check=False,
            )

            duration = (
                time.perf_counter()
                - start
            )

            result = CommandResult(
                command=command_display,
                success=(
                    process.returncode == 0
                ),
                return_code=process.returncode,
                stdout=self._limit_output(
                    process.stdout
                ),
                stderr=self._limit_output(
                    process.stderr
                ),
                duration_seconds=duration,
                working_directory=str(
                    target_directory
                ),
            )

            self._record(result)

            if (
                check
                and not result.success
            ):
                raise RuntimeError(
                    self._format_failure(
                        result
                    )
                )

            return result

        except subprocess.TimeoutExpired as exc:

            duration = (
                time.perf_counter()
                - start
            )

            result = CommandResult(
                command=command_display,
                success=False,
                return_code=None,
                stdout=self._limit_output(
                    self._convert_output(
                        exc.stdout
                    )
                ),
                stderr=self._limit_output(
                    self._convert_output(
                        exc.stderr
                    )
                ),
                duration_seconds=duration,
                timed_out=True,
                working_directory=str(
                    target_directory
                ),
            )

            self._record(result)

            if check:
                raise RuntimeError(
                    "Terminal command timed out."
                )

            return result

        except FileNotFoundError as exc:

            duration = (
                time.perf_counter()
                - start
            )

            result = CommandResult(
                command=command_display,
                success=False,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=duration,
                working_directory=str(
                    target_directory
                ),
            )

            self._record(result)

            if check:
                raise RuntimeError(
                    str(exc)
                )

            return result

        except PermissionError as exc:

            duration = (
                time.perf_counter()
                - start
            )

            result = CommandResult(
                command=command_display,
                success=False,
                return_code=None,
                stdout="",
                stderr=str(exc),
                duration_seconds=duration,
                working_directory=str(
                    target_directory
                ),
            )

            self._record(result)

            if check:
                raise RuntimeError(
                    str(exc)
                )

            return result

    # =========================================================
    # STRING COMMAND HELPER
    # =========================================================

    def execute_string(
        self,
        command: str,
        *,
        cwd: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
        environment: Mapping[str, str] | None = None,
        shell: bool = False,
        check: bool = False,
    ) -> CommandResult:
        """Execute a command provided as a string."""

        return self.execute(
            command,
            cwd=cwd,
            timeout=timeout,
            environment=environment,
            shell=shell,
            check=check,
        )

    # =========================================================
    # PYTHON COMMANDS
    # =========================================================

    def python(
        self,
        arguments: Sequence[str] | None = None,
        *,
        script: str | Path | None = None,
        timeout: float | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> CommandResult:
        """Execute Python using the current interpreter."""

        command = [
            os.sys.executable,
        ]

        if script is not None:
            command.append(
                str(
                    Path(script)
                    .expanduser()
                    .resolve()
                )
            )

        if arguments:
            command.extend(
                str(argument)
                for argument in arguments
            )

        return self.execute(
            command,
            cwd=cwd,
            timeout=timeout,
        )

    def python_module(
        self,
        module: str,
        arguments: Sequence[str] | None = None,
        *,
        timeout: float | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> CommandResult:
        """Execute a Python module."""

        command = [
            os.sys.executable,
            "-m",
            module,
        ]

        if arguments:
            command.extend(
                str(argument)
                for argument in arguments
            )

        return self.execute(
            command,
            cwd=cwd,
            timeout=timeout,
        )

    # =========================================================
    # PIP
    # =========================================================

    def pip(
        self,
        arguments: Sequence[str],
        *,
        timeout: float | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> CommandResult:
        """Execute pip through the active Python interpreter."""

        command = [
            os.sys.executable,
            "-m",
            "pip",
        ]

        command.extend(
            str(argument)
            for argument in arguments
        )

        return self.execute(
            command,
            cwd=cwd,
            timeout=timeout,
        )

    # =========================================================
    # GIT
    # =========================================================

    def git(
        self,
        arguments: Sequence[str],
        *,
        timeout: float | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> CommandResult:
        """Execute a Git command."""

        command = [
            "git",
        ]

        command.extend(
            str(argument)
            for argument in arguments
        )

        return self.execute(
            command,
            cwd=cwd,
            timeout=timeout,
        )

    # =========================================================
    # COMMAND CHECKING
    # =========================================================

    def command_exists(
        self,
        command: str,
    ) -> bool:
        """Check whether an executable is available."""

        import shutil

        return (
            shutil.which(command)
            is not None
        )

    def version(
        self,
        command: str,
        *,
        version_argument: str = "--version",
    ) -> CommandResult:
        """Get a command's version."""

        return self.execute(
            [
                command,
                version_argument,
            ]
        )

    # =========================================================
    # WORKING DIRECTORY
    # =========================================================

    def set_working_directory(
        self,
        directory: str | os.PathLike[str],
    ) -> Path:
        """Change the default working directory."""

        path = (
            Path(directory)
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {path}"
            )

        self.working_directory = path

        return path

    def get_working_directory(self) -> Path:
        """Return the current working directory."""

        return self.working_directory

    def _resolve_cwd(
        self,
        cwd: str | os.PathLike[str] | None,
    ) -> Path:

        if cwd is None:
            path = self.working_directory

        else:
            path = (
                Path(cwd)
                .expanduser()
                .resolve()
            )

        if not path.exists():
            raise FileNotFoundError(
                f"Working directory does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"Working directory is not a directory: {path}"
            )

        return path

    # =========================================================
    # HISTORY
    # =========================================================

    def _record(
        self,
        result: CommandResult,
    ) -> None:

        self.history.append(
            CommandRecord(
                timestamp=time.time(),
                result=result,
            )
        )

        if len(self.history) > self.MAX_HISTORY:

            self.history = (
                self.history[-self.MAX_HISTORY :]
            )

    def get_history(
        self,
        limit: int | None = None,
    ) -> list[CommandRecord]:

        if limit is None:
            return list(self.history)

        if limit <= 0:
            return []

        return list(
            self.history[-limit:]
        )

    def clear_history(self) -> None:
        """Clear terminal command history."""

        self.history.clear()

    # =========================================================
    # SECURITY
    # =========================================================

    def _validate_command(
        self,
        command: str | list[str],
        *,
        shell: bool,
    ) -> str | None:

        if isinstance(
            command,
            str,
        ):
            command_text = command.strip()

        else:
            command_text = self._display_command(
                command
            ).strip()

        if not command_text:
            return (
                "Command cannot be empty."
            )

        if (
            len(command_text)
            > self.policy.max_command_length
        ):
            return (
                "Command exceeds the maximum "
                "allowed command length."
            )

        if shell and not self.policy.allow_shell:
            return (
                "Shell execution is disabled "
                "by the RENIX terminal policy."
            )

        lowered = command_text.lower()

        if (
            not self.policy.allow_sudo
            and self._contains_sudo(
                lowered
            )
        ):
            return (
                "Administrative sudo commands "
                "are disabled by policy."
            )

        if (
            not self.policy.allow_shutdown
            and self._contains_shutdown(
                lowered
            )
        ):
            return (
                "Shutdown commands are disabled "
                "by RENIX policy."
            )

        if (
            not self.policy.allow_reboot
            and self._contains_reboot(
                lowered
            )
        ):
            return (
                "Reboot commands are disabled "
                "by RENIX policy."
            )

        if (
            self.policy.block_destructive_commands
            and self._is_destructive(
                lowered
            )
        ):
            return (
                "Potentially destructive command "
                "blocked by RENIX safety policy."
            )

        return None

    def _is_destructive(
        self,
        command: str,
    ) -> bool:

        for token in (
            self.policy.destructive_tokens
        ):
            if token in command:
                return True

        try:

            parts = shlex.split(
                command,
                posix=False,
            )

        except ValueError:

            parts = command.split()

        if not parts:
            return False

        executable = (
            Path(parts[0]).name.lower()
        )

        return (
            executable
            in {
                command.lower()
                for command
                in self.policy.blocked_commands
            }
        )

    @staticmethod
    def _contains_sudo(
        command: str,
    ) -> bool:

        return (
            command.startswith("sudo ")
            or " sudo " in command
        )

    @staticmethod
    def _contains_shutdown(
        command: str,
    ) -> bool:

        return any(
            token in command
            for token in (
                "shutdown ",
                "shutdown.exe",
            )
        )

    @staticmethod
    def _contains_reboot(
        command: str,
    ) -> bool:

        return any(
            token in command
            for token in (
                "reboot",
                "restart-computer",
            )
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _normalize_command(
        command: str | Sequence[str],
        *,
        shell: bool,
    ) -> str | list[str]:

        if isinstance(
            command,
            str,
        ):

            if shell:
                return command

            return shlex.split(
                command,
                posix=os.name != "nt",
            )

        return [
            str(item)
            for item in command
        ]

    @staticmethod
    def _display_command(
        command: str | Sequence[str],
    ) -> str:

        if isinstance(
            command,
            str,
        ):
            return command

        return " ".join(
            shlex.quote(
                str(item)
            )
            for item in command
        )

    def _format_failure(
        self,
        result: CommandResult,
    ) -> str:

        output = result.output.strip()

        if output:
            return (
                f"Command failed with exit code "
                f"{result.return_code}: {output}"
            )

        return (
            f"Command failed with exit code "
            f"{result.return_code}."
        )

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
            "[RENIX] Terminal output truncated."
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
    "TerminalManager",
    "TerminalPolicy",
    "CommandResult",
    "CommandRecord",
]


