"""
RENIX Process Monitor
=====================

Provides process monitoring and analysis for RENIX.

Features:
- Running process information
- CPU usage per process
- Memory usage per process
- Process searching
- Top CPU-consuming processes
- Top memory-consuming processes
- Process statistics
- Process history
- Suspicious/high-resource process detection

Dependency:
    pip install psutil
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


class ProcessMonitor:
    """
    Process monitoring component for RENIX.

    Example:
        monitor = ProcessMonitor()

        stats = monitor.get_stats()

        top_processes = monitor.get_top_cpu_processes()
    """

    def __init__(
        self,
        *,
        history_size: int = 120,
        process_limit: int = 200,
    ) -> None:

        self.history_size = max(
            1,
            int(history_size),
        )

        self.process_limit = max(
            1,
            int(process_limit),
        )

        self.history: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.history_size
        )

        self.last_stats: (
            dict[str, Any] | None
        ) = None

        self.last_update_time: (
            float | None
        ) = None

    # ============================================================
    # MAIN API
    # ============================================================

    def get_stats(
        self,
        *,
        store: bool = True,
        include_processes: bool = False,
    ) -> dict[str, Any]:
        """
        Collect process statistics.
        """

        if psutil is None:

            return {
                "available": False,
                "error": (
                    "psutil is not installed."
                ),
            }

        try:

            processes = (
                self._get_processes()
            )

            summary = (
                self._create_summary(
                    processes
                )
            )

            stats: dict[str, Any] = {
                "available": True,
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                **summary,
            }

            if include_processes:

                stats["processes"] = (
                    processes
                )

            self.last_stats = stats

            self.last_update_time = (
                time.time()
            )

            if store:

                self.history.append(
                    stats
                )

            return stats

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # PROCESS COLLECTION
    # ============================================================

    def _get_processes(
        self,
    ) -> list[dict[str, Any]]:
        """
        Collect information about running processes.
        """

        if psutil is None:
            return []

        processes: list[
            dict[str, Any]
        ] = []

        try:

            for process in psutil.process_iter(
                attrs=[
                    "pid",
                    "name",
                    "username",
                    "status",
                    "create_time",
                    "memory_info",
                    "memory_percent",
                    "cpu_percent",
                    "exe",
                    "cmdline",
                    "num_threads",
                ]
            ):

                if (
                    len(processes)
                    >= self.process_limit
                ):
                    break

                try:

                    info = process.info

                    memory_info = (
                        info.get(
                            "memory_info"
                        )
                    )

                    memory_rss = 0

                    if memory_info is not None:

                        memory_rss = int(
                            memory_info.rss
                        )

                    create_time = (
                        info.get(
                            "create_time"
                        )
                    )

                    process_data = {
                        "pid": (
                            info.get(
                                "pid"
                            )
                        ),
                        "name": (
                            info.get(
                                "name"
                            )
                            or "Unknown"
                        ),
                        "username": (
                            info.get(
                                "username"
                            )
                        ),
                        "status": (
                            info.get(
                                "status"
                            )
                        ),
                        "cpu_percent": (
                            float(
                                info.get(
                                    "cpu_percent"
                                )
                                or 0.0
                            )
                        ),
                        "memory_percent": (
                            round(
                                float(
                                    info.get(
                                        "memory_percent"
                                    )
                                    or 0.0
                                ),
                                2,
                            )
                        ),
                        "memory_bytes": (
                            memory_rss
                        ),
                        "memory_mb": (
                            round(
                                memory_rss
                                / 1024
                                / 1024,
                                2,
                            )
                        ),
                        "threads": (
                            info.get(
                                "num_threads"
                            )
                        ),
                        "executable": (
                            info.get(
                                "exe"
                            )
                        ),
                        "command": (
                            " ".join(
                                info.get(
                                    "cmdline"
                                )
                                or []
                            )
                        ),
                        "created_at": (
                            self._format_timestamp(
                                create_time
                            )
                        ),
                    }

                    processes.append(
                        process_data
                    )

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue

                except Exception:
                    continue

        except Exception:
            pass

        return processes

    # ============================================================
    # PROCESS LIST
    # ============================================================

    def get_processes(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return running processes.
        """

        processes = (
            self._get_processes()
        )

        if limit is None:
            return processes

        limit = max(
            0,
            int(limit),
        )

        return processes[:limit]

    # ============================================================
    # PROCESS SEARCH
    # ============================================================

    def find_processes(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Search processes by name, executable,
        or command line.
        """

        if not query:
            return []

        query = query.lower().strip()

        results = []

        for process in (
            self._get_processes()
        ):

            name = str(
                process.get(
                    "name",
                    ""
                )
            ).lower()

            executable = str(
                process.get(
                    "executable",
                    ""
                )
                or ""
            ).lower()

            command = str(
                process.get(
                    "command",
                    ""
                )
                or ""
            ).lower()

            if (
                query in name
                or query in executable
                or query in command
            ):

                results.append(
                    process
                )

        return results

    def get_process(
        self,
        pid: int,
    ) -> dict[str, Any] | None:
        """
        Return detailed information
        about a specific process.
        """

        if psutil is None:
            return None

        try:

            process = psutil.Process(
                int(pid)
            )

            memory_info = (
                process.memory_info()
            )

            return {
                "pid": process.pid,
                "name": process.name(),
                "username": (
                    process.username()
                ),
                "status": process.status(),
                "cpu_percent": (
                    process.cpu_percent(
                        interval=0.1
                    )
                ),
                "memory_percent": (
                    round(
                        process.memory_percent(),
                        2,
                    )
                ),
                "memory_bytes": (
                    int(memory_info.rss)
                ),
                "memory_mb": (
                    round(
                        memory_info.rss
                        / 1024
                        / 1024,
                        2,
                    )
                ),
                "threads": (
                    process.num_threads()
                ),
                "executable": (
                    process.exe()
                ),
                "command": (
                    " ".join(
                        process.cmdline()
                    )
                ),
                "created_at": (
                    self._format_timestamp(
                        process.create_time()
                    )
                ),
            }

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):

            return None

        except Exception:
            return None

    # ============================================================
    # TOP CPU PROCESSES
    # ============================================================

    def get_top_cpu_processes(
        self,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return processes using the most CPU.
        """

        processes = (
            self._get_processes()
        )

        processes.sort(
            key=lambda item: float(
                item.get(
                    "cpu_percent",
                    0.0,
                )
            ),
            reverse=True,
        )

        return processes[
            :max(
                0,
                int(limit),
            )
        ]

    # ============================================================
    # TOP MEMORY PROCESSES
    # ============================================================

    def get_top_memory_processes(
        self,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return processes using the most RAM.
        """

        processes = (
            self._get_processes()
        )

        processes.sort(
            key=lambda item: float(
                item.get(
                    "memory_percent",
                    0.0,
                )
            ),
            reverse=True,
        )

        return processes[
            :max(
                0,
                int(limit),
            )
        ]

    # ============================================================
    # PROCESS SUMMARY
    # ============================================================

    @staticmethod
    def _create_summary(
        processes: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Create overall process statistics.
        """

        total_cpu = 0.0
        total_memory_percent = 0.0
        total_memory_bytes = 0

        total_threads = 0

        statuses: dict[
            str,
            int,
        ] = {}

        for process in processes:

            total_cpu += float(
                process.get(
                    "cpu_percent",
                    0.0,
                )
                or 0.0
            )

            total_memory_percent += float(
                process.get(
                    "memory_percent",
                    0.0,
                )
                or 0.0
            )

            total_memory_bytes += int(
                process.get(
                    "memory_bytes",
                    0,
                )
                or 0
            )

            threads = process.get(
                "threads"
            )

            if isinstance(
                threads,
                int,
            ):

                total_threads += threads

            status = str(
                process.get(
                    "status",
                    "unknown",
                )
            )

            statuses[status] = (
                statuses.get(
                    status,
                    0,
                )
                + 1
            )

        return {
            "process_count": len(
                processes
            ),
            "total_cpu_percent": (
                round(
                    total_cpu,
                    2,
                )
            ),
            "total_memory_percent": (
                round(
                    total_memory_percent,
                    2,
                )
            ),
            "total_memory_bytes": (
                total_memory_bytes
            ),
            "total_memory_mb": (
                round(
                    total_memory_bytes
                    / 1024
                    / 1024,
                    2,
                )
            ),
            "total_threads": (
                total_threads
            ),
            "status_counts": (
                statuses
            ),
        }

    # ============================================================
    # HIGH RESOURCE PROCESSES
    # ============================================================

    def get_high_resource_processes(
        self,
        *,
        cpu_threshold: float = 50.0,
        memory_threshold: float = 10.0,
    ) -> list[dict[str, Any]]:
        """
        Return processes consuming unusually
        high CPU or memory.
        """

        results = []

        for process in (
            self._get_processes()
        ):

            cpu = float(
                process.get(
                    "cpu_percent",
                    0.0,
                )
            )

            memory = float(
                process.get(
                    "memory_percent",
                    0.0,
                )
            )

            if (
                cpu >= cpu_threshold
                or memory >= memory_threshold
            ):

                process_copy = (
                    process.copy()
                )

                process_copy[
                    "high_cpu"
                ] = (
                    cpu >= cpu_threshold
                )

                process_copy[
                    "high_memory"
                ] = (
                    memory >= memory_threshold
                )

                results.append(
                    process_copy
                )

        return results

    # ============================================================
    # PROCESS TERMINATION
    # ============================================================

    def terminate_process(
        self,
        pid: int,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Terminate a process.

        IMPORTANT:
        RENIX should normally ask for user
        confirmation before calling this method.
        """

        if psutil is None:

            return {
                "success": False,
                "error": (
                    "psutil is not installed."
                ),
            }

        try:

            process = psutil.Process(
                int(pid)
            )

            name = process.name()

            if force:

                process.kill()

                action = "killed"

            else:

                process.terminate()

                action = "terminated"

            return {
                "success": True,
                "pid": pid,
                "name": name,
                "action": action,
            }

        except psutil.NoSuchProcess:

            return {
                "success": False,
                "error": (
                    "Process does not exist."
                ),
            }

        except psutil.AccessDenied:

            return {
                "success": False,
                "error": (
                    "Permission denied."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> dict[str, Any]:
        """
        Return simplified process system status.
        """

        stats = self.get_stats(
            store=False
        )

        if not stats.get(
            "available"
        ):

            return {
                "available": False,
                "status": "unknown",
            }

        process_count = int(
            stats.get(
                "process_count",
                0,
            )
        )

        if process_count >= 500:

            status = "high"

        elif process_count >= 300:

            status = "moderate"

        else:

            status = "normal"

        return {
            "available": True,
            "status": status,
            "process_count": (
                process_count
            ),
            "healthy": (
                status == "normal"
            ),
        }

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(
        self,
    ) -> dict[str, Any]:
        """
        Analyze running processes.
        """

        stats = self.get_stats()

        if not stats.get(
            "available"
        ):

            return {
                "available": False,
                "status": "unknown",
                "issues": [],
                "recommendations": [],
            }

        high_resource = (
            self.get_high_resource_processes()
        )

        issues: list[str] = []

        recommendations: list[
            str
        ] = []

        if high_resource:

            issues.append(
                f"{len(high_resource)} "
                "high-resource process(es) detected."
            )

            recommendations.append(
                "Review high CPU and memory "
                "consuming applications."
            )

        process_count = int(
            stats.get(
                "process_count",
                0,
            )
        )

        if process_count >= 500:

            issues.append(
                "Very high number of running "
                "processes detected."
            )

            recommendations.append(
                "Check unnecessary background "
                "applications."
            )

            status = "high"

        elif high_resource:

            status = "warning"

        else:

            status = "normal"

        return {
            "available": True,
            "status": status,
            "process_count": (
                process_count
            ),
            "high_resource_processes": (
                high_resource
            ),
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # HISTORY
    # ============================================================

    def get_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:

        history = list(
            self.history
        )

        if limit is None:
            return history

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:
            return []

        return history[-limit:]

    def get_process_count_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return process count history.
        """

        history = self.get_history(
            limit=limit
        )

        return [
            {
                "timestamp": item.get(
                    "timestamp"
                ),
                "process_count": item.get(
                    "process_count"
                ),
                "total_memory_percent": (
                    item.get(
                        "total_memory_percent"
                    )
                ),
            }
            for item in history
        ]

    def clear_history(
        self,
    ) -> None:

        self.history.clear()

    # ============================================================
    # UTILITY
    # ============================================================

    @staticmethod
    def _format_timestamp(
        timestamp: Any,
    ) -> str | None:
        """
        Convert Unix timestamp to ISO format.
        """

        if not isinstance(
            timestamp,
            (int, float),
        ):
            return None

        try:

            return (
                datetime.fromtimestamp(
                    timestamp
                ).isoformat(
                    timespec="seconds"
                )
            )

        except Exception:
            return None

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
    ) -> None:
        """
        Reset process monitor state.
        """

        self.history.clear()

        self.last_stats = None

        self.last_update_time = None


__all__ = [
    "ProcessMonitor",
]


