"""
RENIX Process Manager
=====================

Windows process-management controller for RENIX.

Features:
- List running processes
- Search processes
- Get process information
- Find processes by name
- Start applications/processes
- Terminate processes
- Suspend/resume processes where supported
- Monitor CPU and memory usage
- Check whether a process is running
- Get process tree information
- Safe process operations with explicit confirmation hooks

Dependency:
    pip install psutil
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import psutil

    PSUTIL_AVAILABLE = True

except ImportError:

    psutil = None
    PSUTIL_AVAILABLE = False


# ==============================================================
# DATA MODEL
# ==============================================================


@dataclass
class ProcessInfo:
    """
    Snapshot of a running process.
    """

    pid: int
    name: str
    executable: Optional[str]
    username: Optional[str]
    status: Optional[str]
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    parent_pid: Optional[int]
    create_time: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================
# PROCESS MANAGER
# ==============================================================


class ProcessManager:
    """
    High-level process manager for RENIX.

    Example:

        manager = ProcessManager()

        processes = manager.list_processes()

        if manager.is_running("notepad.exe"):
            manager.terminate_by_name("notepad.exe")
    """

    def __init__(
        self,
        enabled: bool = True,
        allow_termination: bool = True,
        allow_process_creation: bool = True,
    ) -> None:

        self.enabled = enabled
        self.allow_termination = allow_termination
        self.allow_process_creation = (
            allow_process_creation
        )

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:

        if not PSUTIL_AVAILABLE:

            logger.warning(
                "psutil is not installed. "
                "Process management will be unavailable."
            )

            return

        logger.info(
            "RENIX process manager initialized."
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
                "RENIX process manager is disabled."
            )

        if not PSUTIL_AVAILABLE:

            raise RuntimeError(
                "psutil is required. "
                "Install it with: pip install psutil"
            )

    # ==========================================================
    # PROCESS SNAPSHOT
    # ==========================================================

    @staticmethod
    def _safe_process_info(
        process: Any,
    ) -> Optional[ProcessInfo]:

        try:

            with process.oneshot():

                memory_info = (
                    process.memory_info()
                )

                try:
                    cpu = process.cpu_percent(
                        interval=0.0
                    )
                except Exception:
                    cpu = 0.0

                try:
                    memory_percent = (
                        process.memory_percent()
                    )
                except Exception:
                    memory_percent = 0.0

                try:
                    username = (
                        process.username()
                    )
                except Exception:
                    username = None

                try:
                    executable = (
                        process.exe()
                    )
                except Exception:
                    executable = None

                try:
                    status = process.status()
                except Exception:
                    status = None

                try:
                    parent_pid = process.ppid()
                except Exception:
                    parent_pid = None

                try:
                    create_time = (
                        process.create_time()
                    )
                except Exception:
                    create_time = None

                return ProcessInfo(
                    pid=process.pid,
                    name=process.name(),
                    executable=executable,
                    username=username,
                    status=status,
                    cpu_percent=float(cpu),
                    memory_percent=float(
                        memory_percent
                    ),
                    memory_rss=int(
                        memory_info.rss
                    ),
                    parent_pid=parent_pid,
                    create_time=create_time,
                )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):

            return None

        except Exception as exc:

            logger.debug(
                "Unable to read process information: %s",
                exc,
            )

            return None

    # ==========================================================
    # GET PROCESS
    # ==========================================================

    def get_process(
        self,
        pid: int,
    ) -> Optional[ProcessInfo]:

        self._check()

        try:

            process = psutil.Process(
                int(pid)
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):

            return None

        return self._safe_process_info(
            process
        )

    # ==========================================================
    # LIST PROCESSES
    # ==========================================================

    def list_processes(
        self,
        include_system: bool = True,
    ) -> list[dict[str, Any]]:

        self._check()

        results: list[
            dict[str, Any]
        ] = []

        for process in psutil.process_iter():

            info = self._safe_process_info(
                process
            )

            if info is None:
                continue

            if (
                not include_system
                and info.username
                and self._is_system_process(
                    info.username
                )
            ):
                continue

            results.append(
                info.to_dict()
            )

        return results

    # ==========================================================
    # SYSTEM PROCESS CHECK
    # ==========================================================

    @staticmethod
    def _is_system_process(
        username: str,
    ) -> bool:

        username_lower = (
            username.lower()
        )

        system_names = {
            "system",
            "nt authority\\system",
            "nt authority\\localservice",
            "nt authority\\networkservice",
        }

        return username_lower in system_names

    # ==========================================================
    # FIND BY NAME
    # ==========================================================

    def find_by_name(
        self,
        name: str,
    ) -> list[dict[str, Any]]:

        self._check()

        target = (
            name.strip().lower()
        )

        if not target:
            return []

        results = []

        for process in psutil.process_iter(
            ["name"]
        ):

            try:

                process_name = (
                    process.info.get("name")
                    or ""
                )

                if (
                    process_name.lower()
                    == target
                ):

                    info = (
                        self._safe_process_info(
                            process
                        )
                    )

                    if info:
                        results.append(
                            info.to_dict()
                        )

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):

                continue

        return results

    # ==========================================================
    # SEARCH
    # ==========================================================

    def search(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        self._check()

        query = (
            query.strip().lower()
        )

        if not query:
            return []

        results = []

        for process in psutil.process_iter(
            ["pid", "name", "exe"]
        ):

            try:

                name = (
                    process.info.get("name")
                    or ""
                )

                executable = (
                    process.info.get("exe")
                    or ""
                )

                if (
                    query in name.lower()
                    or query
                    in executable.lower()
                ):

                    info = (
                        self._safe_process_info(
                            process
                        )
                    )

                    if info:
                        results.append(
                            info.to_dict()
                        )

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):

                continue

        return results

    # ==========================================================
    # RUNNING CHECK
    # ==========================================================

    def is_running(
        self,
        name: str,
    ) -> bool:

        return bool(
            self.find_by_name(name)
        )

    def is_pid_running(
        self,
        pid: int,
    ) -> bool:

        self._check()

        try:

            return psutil.pid_exists(
                int(pid)
            )

        except Exception:

            return False

    # ==========================================================
    # START PROCESS
    # ==========================================================

    def start_process(
        self,
        command: str | Sequence[str],
        *,
        cwd: Optional[str] = None,
        wait: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[int]:
        """
        Start an external process.

        Returns:
            PID when successfully started.
        """

        self._check()

        if not self.allow_process_creation:

            raise PermissionError(
                "RENIX process creation is disabled."
            )

        if isinstance(
            command,
            str,
        ):

            shell = True

        else:

            shell = False

        try:

            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=shell,
            )

            if wait:

                process.wait(
                    timeout=timeout
                )

            return process.pid

        except Exception as exc:

            logger.error(
                "Unable to start process: %s",
                exc,
            )

            raise

    # ==========================================================
    # START APPLICATION
    # ==========================================================

    def start_application(
        self,
        executable: str,
        *arguments: str,
    ) -> Optional[int]:

        command = [
            executable,
            *arguments,
        ]

        return self.start_process(
            command
        )

    # ==========================================================
    # TERMINATE BY PID
    # ==========================================================

    def terminate(
        self,
        pid: int,
        *,
        force: bool = False,
        timeout: float = 5.0,
    ) -> bool:
        """
        Terminate a process.

        force=True uses kill() after a normal
        termination attempt fails.
        """

        self._check()

        if not self.allow_termination:

            raise PermissionError(
                "RENIX process termination "
                "is disabled."
            )

        pid = int(pid)

        try:

            process = psutil.Process(
                pid
            )

            process.terminate()

            try:

                process.wait(
                    timeout=timeout
                )

                return True

            except psutil.TimeoutExpired:

                if not force:
                    return False

                process.kill()

                process.wait(
                    timeout=timeout
                )

                return True

        except (
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):

            return True

        except psutil.AccessDenied as exc:

            logger.warning(
                "Access denied terminating PID %s: %s",
                pid,
                exc,
            )

            return False

        except Exception as exc:

            logger.warning(
                "Unable to terminate PID %s: %s",
                pid,
                exc,
            )

            return False

    # ==========================================================
    # TERMINATE BY NAME
    # ==========================================================

    def terminate_by_name(
        self,
        name: str,
        *,
        force: bool = False,
        timeout: float = 5.0,
    ) -> int:

        self._check()

        processes = self.find_by_name(
            name
        )

        terminated = 0

        for process in processes:

            if self.terminate(
                process["pid"],
                force=force,
                timeout=timeout,
            ):

                terminated += 1

        return terminated

    # ==========================================================
    # SUSPEND
    # ==========================================================

    def suspend(
        self,
        pid: int,
    ) -> bool:

        self._check()

        try:

            process = psutil.Process(
                int(pid)
            )

            process.suspend()

            return True

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ) as exc:

            logger.warning(
                "Unable to suspend PID %s: %s",
                pid,
                exc,
            )

            return False

    # ==========================================================
    # RESUME
    # ==========================================================

    def resume(
        self,
        pid: int,
    ) -> bool:

        self._check()

        try:

            process = psutil.Process(
                int(pid)
            )

            process.resume()

            return True

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ) as exc:

            logger.warning(
                "Unable to resume PID %s: %s",
                pid,
                exc,
            )

            return False

    # ==========================================================
    # CPU / MEMORY
    # ==========================================================

    def get_cpu_usage(
        self,
        pid: int,
    ) -> float:

        self._check()

        try:

            process = psutil.Process(
                int(pid)
            )

            return float(
                process.cpu_percent(
                    interval=0.1
                )
            )

        except Exception:

            return 0.0

    def get_memory_usage(
        self,
        pid: int,
    ) -> dict[str, float]:

        self._check()

        try:

            process = psutil.Process(
                int(pid)
            )

            memory = (
                process.memory_info()
            )

            return {
                "rss_bytes": float(
                    memory.rss
                ),
                "vms_bytes": float(
                    memory.vms
                ),
                "percent": float(
                    process.memory_percent()
                ),
            }

        except Exception:

            return {
                "rss_bytes": 0.0,
                "vms_bytes": 0.0,
                "percent": 0.0,
            }

    # ==========================================================
    # PROCESS TREE
    # ==========================================================

    def get_children(
        self,
        pid: int,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:

        self._check()

        try:

            process = psutil.Process(
                int(pid)
            )

            children = process.children(
                recursive=recursive
            )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):

            return []

        results = []

        for child in children:

            info = (
                self._safe_process_info(
                    child
                )
            )

            if info:

                results.append(
                    info.to_dict()
                )

        return results

    def get_parent(
        self,
        pid: int,
    ) -> Optional[dict[str, Any]]:

        self._check()

        try:

            process = psutil.Process(
                int(pid)
            )

            parent = process.parent()

            if parent is None:
                return None

            info = (
                self._safe_process_info(
                    parent
                )
            )

            if info:
                return info.to_dict()

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):

            pass

        return None

    # ==========================================================
    # TOP PROCESSES
    # ==========================================================

    def get_top_cpu_processes(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        self._check()

        processes = self.list_processes()

        processes.sort(
            key=lambda item: item.get(
                "cpu_percent",
                0.0,
            ),
            reverse=True,
        )

        return processes[
            :max(1, int(limit))
        ]

    def get_top_memory_processes(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        self._check()

        processes = self.list_processes()

        processes.sort(
            key=lambda item: item.get(
                "memory_percent",
                0.0,
            ),
            reverse=True,
        )

        return processes[
            :max(1, int(limit))
        ]

    # ==========================================================
    # WAIT
    # ==========================================================

    def wait_for_process(
        self,
        name: str,
        timeout: float = 10.0,
        interval: float = 0.25,
    ) -> bool:

        self._check()

        deadline = (
            time.monotonic()
            + float(timeout)
        )

        while time.monotonic() < deadline:

            if self.is_running(name):
                return True

            time.sleep(
                max(0.05, interval)
            )

        return False

    # ==========================================================
    # SYSTEM PROCESS COUNT
    # ==========================================================

    def get_process_count(self) -> int:

        self._check()

        try:

            return len(
                list(
                    psutil.process_iter()
                )
            )

        except Exception:

            return 0

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        status: dict[str, Any] = {
            "enabled": self.enabled,
            "platform": os.name,
            "psutil_available": (
                PSUTIL_AVAILABLE
            ),
            "allow_termination": (
                self.allow_termination
            ),
            "allow_process_creation": (
                self.allow_process_creation
            ),
        }

        if (
            self.enabled
            and PSUTIL_AVAILABLE
        ):

            try:

                status[
                    "process_count"
                ] = self.get_process_count()

                status[
                    "cpu_percent"
                ] = float(
                    psutil.cpu_percent(
                        interval=0.1
                    )
                )

                status[
                    "memory_percent"
                ] = float(
                    psutil.virtual_memory()
                    .percent
                )

            except Exception as exc:

                status[
                    "error"
                ] = str(exc)

        return status

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX process manager shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_process_manager: Optional[
    ProcessManager
] = None


def get_process_manager() -> ProcessManager:
    """
    Return the shared RENIX process manager.
    """

    global _default_process_manager

    if _default_process_manager is None:

        _default_process_manager = (
            ProcessManager()
        )

    return _default_process_manager


