"""
RENIX - System Agent

Handles operating-system-level tasks for RENIX.

Responsibilities:
- System information
- Application launching
- Application closing
- System commands
- Power actions
- Basic environment control
- System status
- Delegation to computer/system services

This agent does not contain the main orchestration logic.
The orchestrator should decide when this agent is appropriate.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SystemResult:
    """Standard result returned by the system agent."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data or {},
            "error": self.error,
        }


class SystemAgent:
    """
    RENIX system-level agent.

    Provides a controlled interface for common operating-system
    operations without exposing raw subprocess execution directly
    to higher-level components.
    """

    name = "system_agent"
    description = "Controls and monitors the operating system."

    def __init__(self) -> None:
        self.platform = platform.system()
        self.running = True

    # ================================================================
    # MAIN ENTRY POINT
    # ================================================================

    def execute(
        self,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SystemResult:
        """
        Execute a supported system action.

        Args:
            action: Name of the requested action.
            parameters: Optional action parameters.

        Returns:
            SystemResult
        """

        parameters = parameters or {}

        if not action:
            return SystemResult(
                success=False,
                message="No system action was provided.",
                error="missing_action",
            )

        action = action.strip().lower()

        handlers = {
            "system_info": self.get_system_info,
            "get_system_info": self.get_system_info,
            "status": self.get_system_info,
            "platform": self.get_system_info,

            "open": lambda: self.open_application(
                parameters.get("application")
                or parameters.get("app")
                or parameters.get("path")
            ),

            "open_application": lambda: self.open_application(
                parameters.get("application")
                or parameters.get("app")
                or parameters.get("path")
            ),

            "launch": lambda: self.open_application(
                parameters.get("application")
                or parameters.get("app")
                or parameters.get("path")
            ),

            "close": lambda: self.close_application(
                parameters.get("application")
                or parameters.get("app")
                or parameters.get("process")
            ),

            "close_application": lambda: self.close_application(
                parameters.get("application")
                or parameters.get("app")
                or parameters.get("process")
            ),

            "sleep": self.sleep_system,
            "shutdown": self.shutdown,
            "restart": self.restart,
            "reboot": self.restart,
            "lock": self.lock_system,

            "environment": self.get_environment,
            "get_environment": self.get_environment,

            "python": self.get_python_info,
            "python_info": self.get_python_info,
        }

        handler = handlers.get(action)

        if handler is None:
            return SystemResult(
                success=False,
                message=f"Unsupported system action: {action}",
                error="unsupported_action",
            )

        try:
            result = handler()

            if isinstance(result, SystemResult):
                return result

            return SystemResult(
                success=True,
                message="System action completed.",
                data={"result": result},
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message="The system action could not be completed.",
                error=str(exc),
            )

    # ================================================================
    # SYSTEM INFORMATION
    # ================================================================

    def get_system_info(self) -> SystemResult:
        """Return basic operating-system information."""

        try:
            info = {
                "os": platform.system(),
                "os_release": platform.release(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
            }

            return SystemResult(
                success=True,
                message="System information retrieved.",
                data=info,
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message="Unable to retrieve system information.",
                error=str(exc),
            )

    # ================================================================
    # APPLICATION CONTROL
    # ================================================================

    def open_application(
        self,
        application: Optional[str],
    ) -> SystemResult:
        """
        Open an application, executable, file, or URL.

        The operating system decides how the target is opened.
        """

        if not application:
            return SystemResult(
                success=False,
                message="No application or path was provided.",
                error="missing_application",
            )

        application = str(application).strip()

        try:
            if self.platform == "Windows":
                os.startfile(application)  # type: ignore[attr-defined]

            elif self.platform == "Darwin":
                subprocess.Popen(
                    ["open", application],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            elif self.platform == "Linux":
                subprocess.Popen(
                    ["xdg-open", application],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            else:
                return SystemResult(
                    success=False,
                    message=f"Unsupported operating system: {self.platform}",
                    error="unsupported_platform",
                )

            return SystemResult(
                success=True,
                message=f"Opened {application}.",
                data={
                    "application": application,
                },
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message=f"Unable to open {application}.",
                error=str(exc),
            )

    def close_application(
        self,
        application: Optional[str],
    ) -> SystemResult:
        """
        Close an application by process name.

        This method intentionally uses a controlled process-name
        operation rather than accepting arbitrary shell commands.
        """

        if not application:
            return SystemResult(
                success=False,
                message="No application or process name was provided.",
                error="missing_application",
            )

        application = str(application).strip()

        try:
            if self.platform == "Windows":
                command = [
                    "taskkill",
                    "/IM",
                    application,
                    "/T",
                    "/F",
                ]

            elif self.platform in {"Linux", "Darwin"}:
                command = [
                    "pkill",
                    "-f",
                    application,
                ]

            else:
                return SystemResult(
                    success=False,
                    message=f"Unsupported operating system: {self.platform}",
                    error="unsupported_platform",
                )

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )

            if completed.returncode != 0:
                error_text = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or "Process could not be closed."
                )

                return SystemResult(
                    success=False,
                    message=f"Unable to close {application}.",
                    error=error_text,
                )

            return SystemResult(
                success=True,
                message=f"Closed {application}.",
                data={
                    "application": application,
                },
            )

        except subprocess.TimeoutExpired:
            return SystemResult(
                success=False,
                message=f"Timed out while closing {application}.",
                error="timeout",
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message=f"Unable to close {application}.",
                error=str(exc),
            )

    # ================================================================
    # POWER MANAGEMENT
    # ================================================================

    def shutdown(self) -> SystemResult:
        """
        Shut down the computer.

        The actual shutdown command is intentionally isolated here
        so security/confirmation layers can be added above the agent.
        """

        try:
            if self.platform == "Windows":
                command = ["shutdown", "/s", "/t", "0"]

            elif self.platform in {"Linux", "Darwin"}:
                command = ["shutdown", "-h", "now"]

            else:
                return SystemResult(
                    success=False,
                    message="Shutdown is not supported on this platform.",
                    error="unsupported_platform",
                )

            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )

            return SystemResult(
                success=True,
                message="System shutdown initiated.",
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message="Unable to initiate shutdown.",
                error=str(exc),
            )

    def restart(self) -> SystemResult:
        """Restart the computer."""

        try:
            if self.platform == "Windows":
                command = ["shutdown", "/r", "/t", "0"]

            elif self.platform in {"Linux", "Darwin"}:
                command = ["shutdown", "-r", "now"]

            else:
                return SystemResult(
                    success=False,
                    message="Restart is not supported on this platform.",
                    error="unsupported_platform",
                )

            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )

            return SystemResult(
                success=True,
                message="System restart initiated.",
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message="Unable to initiate restart.",
                error=str(exc),
            )

    def sleep_system(self) -> SystemResult:
        """Put the system into sleep mode."""

        try:
            if self.platform == "Windows":
                # Windows sleep through PowerShell.
                command = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "[System.Windows.Forms.Application]::SetSuspendState("
                    "'Suspend', $false, $false)",
                ]

            elif self.platform == "Linux":
                command = ["systemctl", "suspend"]

            elif self.platform == "Darwin":
                command = ["pmset", "sleepnow"]

            else:
                return SystemResult(
                    success=False,
                    message="Sleep is not supported on this platform.",
                    error="unsupported_platform",
                )

            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )

            return SystemResult(
                success=True,
                message="System sleep initiated.",
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message="Unable to put the system to sleep.",
                error=str(exc),
            )

    def lock_system(self) -> SystemResult:
        """Lock the current user session."""

        try:
            if self.platform == "Windows":
                command = [
                    "rundll32.exe",
                    "user32.dll,LockWorkStation",
                ]

            elif self.platform == "Linux":
                command = ["loginctl", "lock-session"]

            elif self.platform == "Darwin":
                command = [
                    "/System/Library/CoreServices/Menu Extras/User.menu/"
                    "Contents/Resources/CGSession",
                    "-suspend",
                ]

            else:
                return SystemResult(
                    success=False,
                    message="Locking is not supported on this platform.",
                    error="unsupported_platform",
                )

            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )

            return SystemResult(
                success=True,
                message="System lock initiated.",
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message="Unable to lock the system.",
                error=str(exc),
            )

    # ================================================================
    # ENVIRONMENT
    # ================================================================

    def get_environment(self) -> SystemResult:
        """Return selected environment information."""

        try:
            important_keys = [
                "PATH",
                "TEMP",
                "TMP",
                "USERNAME",
                "USER",
                "HOME",
                "USERPROFILE",
                "COMPUTERNAME",
                "SHELL",
            ]

            environment = {
                key: os.environ.get(key)
                for key in important_keys
                if os.environ.get(key) is not None
            }

            return SystemResult(
                success=True,
                message="Environment information retrieved.",
                data=environment,
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message="Unable to retrieve environment information.",
                error=str(exc),
            )

    # ================================================================
    # PYTHON INFORMATION
    # ================================================================

    def get_python_info(self) -> SystemResult:
        """Return information about the Python runtime."""

        try:
            info = {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
                "prefix": sys.prefix,
                "base_prefix": sys.base_prefix,
                "path_count": len(sys.path),
            }

            return SystemResult(
                success=True,
                message="Python information retrieved.",
                data=info,
            )

        except Exception as exc:
            return SystemResult(
                success=False,
                message="Unable to retrieve Python information.",
                error=str(exc),
            )

    # ================================================================
    # CAPABILITIES
    # ================================================================

    def get_capabilities(self) -> List[str]:
        """Return actions supported by this agent."""

        return [
            "system_info",
            "get_system_info",
            "status",
            "open",
            "open_application",
            "launch",
            "close",
            "close_application",
            "sleep",
            "shutdown",
            "restart",
            "reboot",
            "lock",
            "environment",
            "get_environment",
            "python",
            "python_info",
        ]

    def health_check(self) -> Dict[str, Any]:
        """Return a basic health report for the agent."""

        return {
            "agent": self.name,
            "status": "healthy" if self.running else "stopped",
            "platform": self.platform,
            "capabilities": len(self.get_capabilities()),
        }

    def stop(self) -> None:
        """Stop the agent."""

        self.running = False

    def start(self) -> None:
        """Start the agent."""

        self.running = True


# ====================================================================
# DEFAULT AGENT INSTANCE
# ====================================================================

system_agent = SystemAgent()


__all__ = [
    "SystemResult",
    "SystemAgent",
    "system_agent",
]


