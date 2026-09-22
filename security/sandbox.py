"""
RENIX Security Sandbox
======================

Provides a controlled execution environment for potentially
unsafe commands, scripts, and tasks.

Features:

* Sandbox sessions
* Command allow/block lists
* Working directory isolation
* Execution timeouts
* Environment variable filtering
* Resource limits
* Audit history
* Safe subprocess execution

Important:
This module is a safety wrapper, not a perfect security boundary.
For executing truly untrusted code, use OS-level containers,
virtual machines, or dedicated sandboxing infrastructure.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RENIX.Sandbox")

class Sandbox:
    """
    Controlled execution environment for RENIX.

    Example:

        sandbox = Sandbox()

        result = sandbox.execute(
            ["python", "--version"]
        )
    """

    DEFAULT_ALLOWED_COMMANDS = {
        "python",
        "python3",
        "pip",
        "pip3",
        "git",
        "node",
        "npm",
        "npx",
    }

    DEFAULT_BLOCKED_COMMANDS = {
        "rm",
        "rmdir",
        "del",
        "format",
        "shutdown",
        "reboot",
        "poweroff",
        "taskkill",
        "reg",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize sandbox.
        """

        self.config = config or {}

        self.enabled = bool(
            self.config.get(
                "enabled",
                True,
            )
        )

        self.default_timeout = int(
            self.config.get(
                "default_timeout",
                30,
            )
        )

        self.max_output_length = int(
            self.config.get(
                "max_output_length",
                20000,
            )
        )

        self.keep_workspaces = bool(
            self.config.get(
                "keep_workspaces",
                False,
            )
        )

        self.allowed_commands = set(
            self.config.get(
                "allowed_commands",
                self.DEFAULT_ALLOWED_COMMANDS,
            )
        )

        self.blocked_commands = set(
            self.config.get(
                "blocked_commands",
                self.DEFAULT_BLOCKED_COMMANDS,
            )
        )

        self.sessions: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.history: List[
            Dict[str, Any]
        ] = []

        self.max_history = int(
            self.config.get(
                "max_history",
                500,
            )
        )

        logger.info(
            "Sandbox initialized."
        )

    # ============================================================
    # SESSION MANAGEMENT
    # ============================================================

    def create_session(
        self,
        name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a new sandbox session.
        """

        session_id = str(
            uuid.uuid4()
        )

        workspace = Path(
            tempfile.mkdtemp(
                prefix="renix_sandbox_"
            )
        )

        session = {
            "session_id": session_id,
            "name": (
                name
                or f"sandbox-{session_id[:8]}"
            ),
            "workspace": str(
                workspace
            ),
            "created_at": (
                datetime.now().isoformat()
            ),
            "timeout": (
                timeout
                or self.default_timeout
            ),
            "active": True,
            "executions": 0,
        }

        self.sessions[
            session_id
        ] = session

        logger.info(
            "Sandbox session created: %s",
            session_id,
        )

        return session.copy()

    def get_session(
        self,
        session_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Get sandbox session information.
        """

        session = self.sessions.get(
            session_id
        )

        if session is None:
            return None

        return session.copy()

    def close_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Close and clean up a sandbox session.
        """

        session = self.sessions.get(
            session_id
        )

        if session is None:
            return False

        workspace = Path(
            session["workspace"]
        )

        session["active"] = False

        if (
            workspace.exists()
            and not self.keep_workspaces
        ):

            try:

                shutil.rmtree(
                    workspace,
                    ignore_errors=True,
                )

            except Exception as error:

                logger.error(
                    "Failed to remove sandbox "
                    "workspace: %s",
                    error,
                )

        del self.sessions[
            session_id
        ]

        logger.info(
            "Sandbox session closed: %s",
            session_id,
        )

        return True

    # ============================================================
    # COMMAND VALIDATION
    # ============================================================

    def _get_command_name(
        self,
        command: List[str],
    ) -> str:
        """
        Extract command executable name.
        """

        if not command:
            return ""

        executable = Path(
            str(command[0])
        ).name.lower()

        return executable

    def is_command_allowed(
        self,
        command: List[str],
    ) -> Dict[str, Any]:
        """
        Check whether a command is allowed
        inside the sandbox.
        """

        if not command:

            return {
                "allowed": False,
                "reason": (
                    "Empty command."
                ),
            }

        command_name = (
            self._get_command_name(
                command
            )
        )

        if command_name in (
            self.blocked_commands
        ):

            return {
                "allowed": False,
                "reason": (
                    f"Blocked command: "
                    f"{command_name}"
                ),
            }

        if (
            self.allowed_commands
            and command_name
            not in self.allowed_commands
        ):

            return {
                "allowed": False,
                "reason": (
                    f"Command not in allowlist: "
                    f"{command_name}"
                ),
            }

        return {
            "allowed": True,
            "reason": (
                "Command allowed."
            ),
        }

    # ============================================================
    # ENVIRONMENT
    # ============================================================

    def _build_environment(
        self,
        environment: Optional[
            Dict[str, str]
        ] = None,
    ) -> Dict[str, str]:
        """
        Create restricted environment variables.
        """

        safe_keys = {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
        }

        sandbox_environment = {
            key: value
            for key, value
            in os.environ.items()
            if key in safe_keys
        }

        if environment:

            for key, value in (
                environment.items()
            ):

                if isinstance(
                    key,
                    str,
                ) and isinstance(
                    value,
                    str,
                ):

                    sandbox_environment[
                        key
                    ] = value

        return sandbox_environment

    # ============================================================
    # OUTPUT LIMITING
    # ============================================================

    def _limit_output(
        self,
        output: Optional[str],
    ) -> str:
        """
        Limit stored output size.
        """

        if not output:
            return ""

        if len(output) <= (
            self.max_output_length
        ):

            return output

        return (
            output[
                :self.max_output_length
            ]
            + "\n...[OUTPUT TRUNCATED]"
        )

    # ============================================================
    # EXECUTION
    # ============================================================

    def execute(
        self,
        command: List[str],
        session_id: Optional[
            str
        ] = None,
        timeout: Optional[
            int
        ] = None,
        environment: Optional[
            Dict[str, str]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Execute a command inside a sandbox workspace.

        Commands are executed without shell=True.
        """

        result = {
            "success": False,
            "command": command,
            "session_id": session_id,
            "return_code": None,
            "stdout": "",
            "stderr": "",
            "error": None,
            "timed_out": False,
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        if not self.enabled:

            result["error"] = (
                "Sandbox is disabled."
            )

            self._record_history(
                result
            )

            return result

        validation = (
            self.is_command_allowed(
                command
            )
        )

        if not validation["allowed"]:

            result["error"] = (
                validation["reason"]
            )

            self._record_history(
                result
            )

            logger.warning(
                "Sandbox blocked command: %s",
                command,
            )

            return result

        session = None

        if session_id:

            session = self.sessions.get(
                session_id
            )

            if session is None:

                result["error"] = (
                    "Sandbox session not found."
                )

                self._record_history(
                    result
                )

                return result

            if not session.get(
                "active",
                False,
            ):

                result["error"] = (
                    "Sandbox session is inactive."
                )

                self._record_history(
                    result
                )

                return result

        # Create temporary session if none provided.
        temporary_session = False

        if session is None:

            session = self.create_session()

            session_id = (
                session["session_id"]
            )

            result["session_id"] = (
                session_id
            )

            temporary_session = True

        workspace = (
            session["workspace"]
        )

        execution_timeout = (
            timeout
            or session.get(
                "timeout",
                self.default_timeout,
            )
        )

        try:

            process = subprocess.run(
                command,
                cwd=workspace,
                env=self._build_environment(
                    environment
                ),
                capture_output=True,
                text=True,
                timeout=execution_timeout,
                shell=False,
            )

            result["return_code"] = (
                process.returncode
            )

            result["stdout"] = (
                self._limit_output(
                    process.stdout
                )
            )

            result["stderr"] = (
                self._limit_output(
                    process.stderr
                )
            )

            result["success"] = (
                process.returncode == 0
            )

        except subprocess.TimeoutExpired as error:

            result["timed_out"] = True

            result["error"] = (
                f"Execution timed out after "
                f"{execution_timeout} seconds."
            )

            result["stdout"] = (
                self._limit_output(
                    error.stdout
                    if isinstance(
                        error.stdout,
                        str,
                    )
                    else ""
                )
            )

            result["stderr"] = (
                self._limit_output(
                    error.stderr
                    if isinstance(
                        error.stderr,
                        str,
                    )
                    else ""
                )
            )

            logger.warning(
                "Sandbox command timed out: %s",
                command,
            )

        except FileNotFoundError:

            result["error"] = (
                "Command executable not found."
            )

        except PermissionError:

            result["error"] = (
                "Permission denied while "
                "executing command."
            )

        except Exception as error:

            result["error"] = str(
                error
            )

            logger.exception(
                "Sandbox execution error."
            )

        session["executions"] += 1

        result["finished_at"] = (
            datetime.now().isoformat()
        )

        self._record_history(
            result
        )

        # Automatically clean temporary session.
        if temporary_session:

            self.close_session(
                session_id
            )

        return result

    # ============================================================
    # FILE OPERATIONS
    # ============================================================

    def write_file(
        self,
        session_id: str,
        relative_path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Write a file inside sandbox workspace.

        Prevents path traversal outside
        the sandbox directory.
        """

        result = {
            "success": False,
            "path": relative_path,
            "error": None,
        }

        session = self.sessions.get(
            session_id
        )

        if session is None:

            result["error"] = (
                "Sandbox session not found."
            )

            return result

        workspace = Path(
            session["workspace"]
        ).resolve()

        target = (
            workspace
            / relative_path
        ).resolve()

        # Prevent directory traversal.
        if (
            workspace not in target.parents
            and target != workspace
        ):

            result["error"] = (
                "Path escapes sandbox workspace."
            )

            return result

        try:

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            target.write_text(
                content,
                encoding=encoding,
            )

            result["success"] = True
            result["path"] = str(
                target
            )

        except Exception as error:

            result["error"] = str(
                error
            )

        return result

    def read_file(
        self,
        session_id: str,
        relative_path: str,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Read a file inside sandbox workspace.
        """

        result = {
            "success": False,
            "content": None,
            "path": relative_path,
            "error": None,
        }

        session = self.sessions.get(
            session_id
        )

        if session is None:

            result["error"] = (
                "Sandbox session not found."
            )

            return result

        workspace = Path(
            session["workspace"]
        ).resolve()

        target = (
            workspace
            / relative_path
        ).resolve()

        if (
            workspace not in target.parents
            and target != workspace
        ):

            result["error"] = (
                "Path escapes sandbox workspace."
            )

            return result

        if not target.exists():

            result["error"] = (
                "File does not exist."
            )

            return result

        try:

            result["content"] = (
                target.read_text(
                    encoding=encoding
                )
            )

            result["success"] = True

        except Exception as error:

            result["error"] = str(
                error
            )

        return result

    def list_files(
        self,
        session_id: str,
    ) -> List[str]:
        """
        List files inside sandbox workspace.
        """

        session = self.sessions.get(
            session_id
        )

        if session is None:
            return []

        workspace = Path(
            session["workspace"]
        )

        if not workspace.exists():
            return []

        files = []

        for path in workspace.rglob(
            "*"
        ):

            if path.is_file():

                files.append(
                    str(
                        path.relative_to(
                            workspace
                        )
                    )
                )

        return files

    # ============================================================
    # HISTORY
    # ============================================================

    def _record_history(
        self,
        result: Dict[str, Any],
    ) -> None:
        """
        Store execution history.
        """

        self.history.append(
            result.copy()
        )

        if len(self.history) > (
            self.max_history
        ):

            overflow = (
                len(self.history)
                - self.max_history
            )

            self.history = (
                self.history[
                    overflow:
                ]
            )

    def get_history(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return sandbox execution history.
        """

        return self.history[
            -max(1, limit):
        ]

    def clear_history(
        self,
    ) -> None:
        """
        Clear sandbox history.
        """

        self.history.clear()

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return sandbox status.
        """

        return {
            "enabled": self.enabled,
            "active_sessions": len(
                self.sessions
            ),
            "history_entries": len(
                self.history
            ),
            "default_timeout": (
                self.default_timeout
            ),
            "allowed_commands": sorted(
                self.allowed_commands
            ),
            "blocked_commands": sorted(
                self.blocked_commands
            ),
        }

    # ============================================================
    # SHUTDOWN
    # ============================================================

    def shutdown(
        self,
    ) -> None:
        """
        Close all sandbox sessions.
        """

        logger.info(
            "Shutting down Sandbox..."
        )

        for session_id in list(
            self.sessions.keys()
        ):

            self.close_session(
                session_id
            )

        self.clear_history()

        logger.info(
            "Sandbox shutdown complete."
        )
