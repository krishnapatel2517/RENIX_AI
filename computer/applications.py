"""
RENIX Application Controller
============================

Launch, inspect, focus, and close Windows applications.

Features:
- Launch applications
- Open files with their default application
- Open URLs
- Launch through shell commands
- Find running applications
- Focus application windows
- Close applications
- Start applications without blocking RENIX
- Track launched processes

Designed primarily for Windows 11.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ApplicationInfo:
    """Information about an application launched by RENIX."""

    name: str
    command: str
    pid: Optional[int] = None
    started_at: float = field(
        default_factory=time.time
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "pid": self.pid,
            "started_at": self.started_at,
        }


class ApplicationController:
    """
    High-level application manager for RENIX.

    Examples:

        apps.launch("notepad")
        apps.launch("calculator")
        apps.open_file("C:/Users/User/Desktop/test.txt")
        apps.open_url("https://example.com")
    """

    DEFAULT_APPLICATIONS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "paint": "mspaint.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "cmd": "cmd.exe",
        "command prompt": "cmd.exe",
        "powershell": "powershell.exe",
        "terminal": "wt.exe",
        "windows terminal": "wt.exe",
        "task manager": "taskmgr.exe",
        "control panel": "control.exe",
        "settings": "ms-settings:",
        "snipping tool": "snippingtool.exe",
        "wordpad": "write.exe",
    }

    def __init__(
        self,
        enabled: bool = True,
        shell: bool = False,
    ) -> None:

        self.enabled = enabled
        self.shell = shell

        self.launched: dict[int, ApplicationInfo] = {}

        self._window_manager: Optional[Any] = None

        logger.info(
            "RENIX application controller initialized."
        )

    # ==========================================================
    # STATE
    # ==========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX application control is disabled."
            )

    # ==========================================================
    # WINDOW MANAGER
    # ==========================================================

    def _get_window_manager(self) -> Any:

        if self._window_manager is None:

            try:
                from .windows import WindowManager

                self._window_manager = (
                    WindowManager()
                )

            except Exception as exc:

                logger.warning(
                    "Window manager unavailable: %s",
                    exc,
                )

                self._window_manager = False

        if self._window_manager is False:
            return None

        return self._window_manager

    # ==========================================================
    # APPLICATION RESOLUTION
    # ==========================================================

    def resolve_application(
        self,
        name: str,
    ) -> str:

        if not isinstance(name, str):
            raise TypeError(
                "Application name must be a string."
            )

        name = name.strip()

        if not name:
            raise ValueError(
                "Application name cannot be empty."
            )

        normalized = name.lower()

        if normalized in self.DEFAULT_APPLICATIONS:
            return self.DEFAULT_APPLICATIONS[
                normalized
            ]

        return name

    # ==========================================================
    # LAUNCH
    # ==========================================================

    def launch(
        self,
        application: str,
        args: Optional[list[str]] = None,
        cwd: Optional[str] = None,
        wait: bool = False,
    ) -> ApplicationInfo:

        self._check()

        executable = self.resolve_application(
            application
        )

        args = args or []

        command_parts = [
            executable,
            *[
                str(argument)
                for argument in args
            ],
        ]

        command_display = " ".join(
            shlex.quote(part)
            for part in command_parts
        )

        try:

            if executable.startswith(
                "ms-settings:"
            ):

                os.startfile(
                    executable
                )

                info = ApplicationInfo(
                    name=application,
                    command=command_display,
                    pid=None,
                )

                return info

            process = subprocess.Popen(
                command_parts,
                cwd=cwd,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            info = ApplicationInfo(
                name=application,
                command=command_display,
                pid=process.pid,
            )

            self.launched[
                process.pid
            ] = info

            if wait:
                process.wait()

            logger.info(
                "Launched application: %s",
                application,
            )

            return info

        except FileNotFoundError as exc:

            raise FileNotFoundError(
                f"Application not found: "
                f"{application}"
            ) from exc

        except Exception as exc:

            logger.exception(
                "Failed to launch application: %s",
                application,
            )

            raise RuntimeError(
                f"Failed to launch "
                f"{application}: {exc}"
            ) from exc

    # ==========================================================
    # LAUNCH COMMAND
    # ==========================================================

    def launch_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        wait: bool = False,
    ) -> Optional[ApplicationInfo]:

        self._check()

        if not command.strip():
            raise ValueError(
                "Command cannot be empty."
            )

        try:

            parts = shlex.split(
                command,
                posix=False,
            )

            if not parts:
                return None

            return self.launch(
                parts[0],
                args=parts[1:],
                cwd=cwd,
                wait=wait,
            )

        except Exception as exc:

            logger.warning(
                "Structured launch failed: %s",
                exc,
            )

            if not self.shell:
                raise

            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=True,
            )

            info = ApplicationInfo(
                name=parts[0]
                if parts
                else command,
                command=command,
                pid=process.pid,
            )

            self.launched[
                process.pid
            ] = info

            return info

    # ==========================================================
    # OPEN FILE
    # ==========================================================

    def open_file(
        self,
        path: str,
    ) -> bool:

        self._check()

        file_path = Path(path).expanduser()

        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

        try:

            os.startfile(
                str(file_path)
            )

            logger.info(
                "Opened file: %s",
                file_path,
            )

            return True

        except Exception as exc:

            raise RuntimeError(
                f"Unable to open file: "
                f"{file_path}"
            ) from exc

    # ==========================================================
    # OPEN FOLDER
    # ==========================================================

    def open_folder(
        self,
        path: str,
    ) -> bool:

        self._check()

        folder = Path(path).expanduser()

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder does not exist: {folder}"
            )

        if not folder.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {folder}"
            )

        try:

            subprocess.Popen(
                [
                    "explorer.exe",
                    str(folder),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception as exc:

            raise RuntimeError(
                f"Unable to open folder: {folder}"
            ) from exc

    # ==========================================================
    # OPEN URL
    # ==========================================================

    def open_url(
        self,
        url: str,
    ) -> bool:

        self._check()

        url = str(url).strip()

        if not url:
            raise ValueError(
                "URL cannot be empty."
            )

        if not (
            url.startswith("http://")
            or url.startswith("https://")
            or url.startswith("mailto:")
        ):
            url = "https://" + url

        return bool(
            webbrowser.open(url)
        )

    # ==========================================================
    # COMMON APPLICATIONS
    # ==========================================================

    def open_notepad(self) -> ApplicationInfo:
        return self.launch("notepad")

    def open_calculator(self) -> ApplicationInfo:
        return self.launch("calculator")

    def open_paint(self) -> ApplicationInfo:
        return self.launch("paint")

    def open_explorer(self) -> ApplicationInfo:
        return self.launch("explorer")

    def open_terminal(self) -> ApplicationInfo:
        return self.launch("terminal")

    def open_powershell(self) -> ApplicationInfo:
        return self.launch("powershell")

    def open_task_manager(self) -> ApplicationInfo:
        return self.launch("task manager")

    def open_settings(self) -> ApplicationInfo:
        return self.launch("settings")

    # ==========================================================
    # FOCUS APPLICATION
    # ==========================================================

    def focus(
        self,
        window_title: str,
        exact: bool = False,
    ) -> dict[str, Any]:

        self._check()

        manager = self._get_window_manager()

        if manager is None:
            raise RuntimeError(
                "Window manager unavailable."
            )

        return manager.focus(
            window_title,
            exact=exact,
        )

    # ==========================================================
    # CLOSE APPLICATION
    # ==========================================================

    def close(
        self,
        window_title: str,
        exact: bool = False,
    ) -> bool:

        self._check()

        manager = self._get_window_manager()

        if manager is None:
            raise RuntimeError(
                "Window manager unavailable."
            )

        return manager.close(
            window_title,
            exact=exact,
        )

    # ==========================================================
    # FIND APPLICATION
    # ==========================================================

    def is_running(
        self,
        window_title: str,
        exact: bool = False,
    ) -> bool:

        manager = self._get_window_manager()

        if manager is None:
            return False

        return manager.exists(
            window_title,
            exact=exact,
        )

    def find_windows(
        self,
        query: str,
        exact: bool = False,
    ) -> list[dict[str, Any]]:

        manager = self._get_window_manager()

        if manager is None:
            return []

        return [
            window
            for window in manager.list_windows()
            if (
                query.lower()
                in window["title"].lower()
                if not exact
                else
                window["title"].lower()
                == query.lower()
            )
        ]

    # ==========================================================
    # PROCESS MANAGEMENT
    # ==========================================================

    def is_process_alive(
        self,
        pid: int,
    ) -> bool:

        try:

            import psutil

            return psutil.pid_exists(
                int(pid)
            )

        except ImportError:

            return False

        except Exception:

            return False

    def get_process(
        self,
        pid: int,
    ) -> Optional[ApplicationInfo]:

        return self.launched.get(
            int(pid)
        )

    def get_launched_applications(
        self,
    ) -> list[dict[str, Any]]:

        active = []

        for pid, info in list(
            self.launched.items()
        ):

            if self.is_process_alive(pid):

                active.append(
                    info.to_dict()
                )

            else:

                self.launched.pop(
                    pid,
                    None,
                )

        return active

    # ==========================================================
    # TERMINATE PROCESS
    # ==========================================================

    def terminate(
        self,
        pid: int,
        force: bool = False,
    ) -> bool:

        self._check()

        pid = int(pid)

        try:

            import psutil

            process = psutil.Process(pid)

            if force:
                process.kill()
            else:
                process.terminate()

            self.launched.pop(
                pid,
                None,
            )

            return True

        except ImportError as exc:

            raise RuntimeError(
                "psutil is required for "
                "process termination."
            ) from exc

        except Exception as exc:

            raise RuntimeError(
                f"Unable to terminate PID {pid}: "
                f"{exc}"
            ) from exc

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "tracked_processes": len(
                self.launched
            ),
            "running": len(
                self.get_launched_applications()
            ),
        }

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX application controller shut down."
        )


ApplicationManager = ApplicationController


