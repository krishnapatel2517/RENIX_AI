"""
RENIX Disk Monitor
==================

Provides detailed disk and storage monitoring.

Features:
- All disk partitions
- Total / used / free storage
- Disk usage percentage
- Disk I/O statistics
- Per-disk monitoring
- History tracking
- Health analysis
- Low storage warnings

Dependency:
    pip install psutil
"""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


class DiskMonitor:
    """
    Disk monitoring component for RENIX.

    Example:
        monitor = DiskMonitor()

        stats = monitor.get_stats()

        print(stats["partitions"])
    """

    def __init__(
        self,
        *,
        history_size: int = 120,
    ) -> None:

        self.history_size = max(
            1,
            int(history_size),
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
    ) -> dict[str, Any]:
        """
        Collect complete disk statistics.
        """

        if psutil is None:

            return {
                "available": False,
                "error": (
                    "psutil is not installed."
                ),
            }

        try:

            partitions = (
                self._get_partitions()
            )

            io_stats = (
                self._get_io_stats()
            )

            summary = (
                self._create_summary(
                    partitions
                )
            )

            stats = {
                "available": True,
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "partitions": partitions,
                "io": io_stats,
                **summary,
            }

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
    # PARTITIONS
    # ============================================================

    def _get_partitions(
        self,
    ) -> list[dict[str, Any]]:
        """
        Collect information about all disk partitions.
        """

        if psutil is None:
            return []

        partitions: list[
            dict[str, Any]
        ] = []

        try:

            disk_partitions = (
                psutil.disk_partitions(
                    all=False
                )
            )

            seen_mountpoints = set()

            for partition in (
                disk_partitions
            ):

                mountpoint = (
                    partition.mountpoint
                )

                if (
                    mountpoint
                    in seen_mountpoints
                ):
                    continue

                seen_mountpoints.add(
                    mountpoint
                )

                try:

                    usage = (
                        psutil.disk_usage(
                            mountpoint
                        )
                    )

                    partitions.append(
                        self._format_partition(
                            partition,
                            usage,
                        )
                    )

                except (
                    PermissionError,
                    OSError,
                ):
                    continue

        except Exception:
            pass

        return partitions

    @staticmethod
    def _format_partition(
        partition: Any,
        usage: Any,
    ) -> dict[str, Any]:
        """
        Format partition and storage information.
        """

        total = int(
            usage.total
        )

        used = int(
            usage.used
        )

        free = int(
            usage.free
        )

        percent = float(
            usage.percent
        )

        return {
            "device": (
                partition.device
            ),
            "mountpoint": (
                partition.mountpoint
            ),
            "filesystem": (
                partition.fstype
            ),
            "options": (
                partition.opts
            ),

            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,

            "percent": percent,

            "total_mb": (
                round(
                    total
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "used_mb": (
                round(
                    used
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "free_mb": (
                round(
                    free
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "total_gb": (
                round(
                    total
                    / 1024
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "used_gb": (
                round(
                    used
                    / 1024
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "free_gb": (
                round(
                    free
                    / 1024
                    / 1024
                    / 1024,
                    2,
                )
            ),
        }

    # ============================================================
    # DISK I/O
    # ============================================================

    def _get_io_stats(
        self,
    ) -> dict[str, Any]:

        if psutil is None:
            return {}

        try:

            io = (
                psutil.disk_io_counters()
            )

            if io is None:
                return {}

            return {
                "read_count": int(
                    io.read_count
                ),
                "write_count": int(
                    io.write_count
                ),
                "read_bytes": int(
                    io.read_bytes
                ),
                "write_bytes": int(
                    io.write_bytes
                ),
                "read_mb": (
                    round(
                        io.read_bytes
                        / 1024
                        / 1024,
                        2,
                    )
                ),
                "write_mb": (
                    round(
                        io.write_bytes
                        / 1024
                        / 1024,
                        2,
                    )
                ),
                "read_time_ms": int(
                    getattr(
                        io,
                        "read_time",
                        0,
                    )
                ),
                "write_time_ms": int(
                    getattr(
                        io,
                        "write_time",
                        0,
                    )
                ),
            }

        except Exception:
            return {}

    def get_io_stats(
        self,
    ) -> dict[str, Any]:
        """
        Return disk input/output statistics.
        """

        return self._get_io_stats()

    # ============================================================
    # SUMMARY
    # ============================================================

    @staticmethod
    def _create_summary(
        partitions: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Create total storage summary.
        """

        total_bytes = 0
        used_bytes = 0
        free_bytes = 0

        usage_values = []

        for partition in partitions:

            total_bytes += int(
                partition.get(
                    "total_bytes",
                    0,
                )
            )

            used_bytes += int(
                partition.get(
                    "used_bytes",
                    0,
                )
            )

            free_bytes += int(
                partition.get(
                    "free_bytes",
                    0,
                )
            )

            percent = partition.get(
                "percent"
            )

            if isinstance(
                percent,
                (int, float),
            ):

                usage_values.append(
                    float(percent)
                )

        overall_percent = (
            round(
                used_bytes
                / total_bytes
                * 100,
                2,
            )
            if total_bytes > 0
            else 0.0
        )

        return {
            "partition_count": len(
                partitions
            ),

            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "free_bytes": free_bytes,

            "usage_percent": (
                overall_percent
            ),

            "total_gb": (
                round(
                    total_bytes
                    / 1024
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "used_gb": (
                round(
                    used_bytes
                    / 1024
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "free_gb": (
                round(
                    free_bytes
                    / 1024
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "highest_partition_usage": (
                max(
                    usage_values
                )
                if usage_values
                else 0.0
            ),
        }

    # ============================================================
    # PARTITION ACCESS
    # ============================================================

    def get_partition(
        self,
        mountpoint: str,
    ) -> dict[str, Any] | None:
        """
        Return information about a specific partition.
        """

        stats = self.get_stats(
            store=False
        )

        partitions = stats.get(
            "partitions",
            [],
        )

        for partition in partitions:

            if (
                partition.get(
                    "mountpoint"
                )
                == mountpoint
            ):

                return partition

        return None

    def get_partition_usage(
        self,
        mountpoint: str,
    ) -> float | None:
        """
        Return usage percentage for a partition.
        """

        partition = (
            self.get_partition(
                mountpoint
            )
        )

        if partition is None:
            return None

        percent = partition.get(
            "percent"
        )

        if isinstance(
            percent,
            (int, float),
        ):

            return float(
                percent
            )

        return None

    # ============================================================
    # QUICK METRICS
    # ============================================================

    def get_usage(
        self,
    ) -> float:
        """
        Return overall disk usage percentage.
        """

        stats = self.get_stats(
            store=False
        )

        return float(
            stats.get(
                "usage_percent",
                0.0,
            )
        )

    def get_total_space(
        self,
    ) -> int:

        stats = self.get_stats(
            store=False
        )

        return int(
            stats.get(
                "total_bytes",
                0,
            )
        )

    def get_used_space(
        self,
    ) -> int:

        stats = self.get_stats(
            store=False
        )

        return int(
            stats.get(
                "used_bytes",
                0,
            )
        )

    def get_free_space(
        self,
    ) -> int:

        stats = self.get_stats(
            store=False
        )

        return int(
            stats.get(
                "free_bytes",
                0,
            )
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> dict[str, Any]:
        """
        Return simplified disk health status.
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

        highest_usage = float(
            stats.get(
                "highest_partition_usage",
                0.0,
            )
        )

        if highest_usage >= 95:

            status = "critical"

        elif highest_usage >= 90:

            status = "high"

        elif highest_usage >= 80:

            status = "moderate"

        else:

            status = "normal"

        return {
            "available": True,
            "status": status,
            "usage_percent": (
                stats.get(
                    "usage_percent"
                )
            ),
            "highest_partition_usage": (
                highest_usage
            ),
            "healthy": (
                highest_usage < 90
            ),
        }

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(
        self,
    ) -> dict[str, Any]:
        """
        Analyze disk health and storage usage.
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

        issues: list[str] = []

        recommendations: list[
            str
        ] = []

        critical_partitions = []
        high_partitions = []

        for partition in stats.get(
            "partitions",
            [],
        ):

            usage = float(
                partition.get(
                    "percent",
                    0.0,
                )
            )

            mountpoint = partition.get(
                "mountpoint",
                "Unknown",
            )

            if usage >= 95:

                critical_partitions.append(
                    mountpoint
                )

            elif usage >= 90:

                high_partitions.append(
                    mountpoint
                )

        if critical_partitions:

            issues.append(
                "Critical disk usage detected "
                f"on: {', '.join(critical_partitions)}"
            )

            recommendations.append(
                "Free disk space immediately."
            )

        if high_partitions:

            issues.append(
                "High disk usage detected "
                f"on: {', '.join(high_partitions)}"
            )

            recommendations.append(
                "Remove unnecessary files or "
                "move large files to another drive."
            )

        if critical_partitions:

            status = "critical"

        elif high_partitions:

            status = "high"

        elif (
            stats.get(
                "highest_partition_usage",
                0,
            )
            >= 80
        ):

            status = "moderate"

        else:

            status = "normal"

        return {
            "available": True,
            "status": status,
            "current_usage": (
                stats.get(
                    "usage_percent"
                )
            ),
            "free_space_gb": (
                stats.get(
                    "free_gb"
                )
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
        """
        Return disk monitoring history.
        """

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

    def get_usage_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return simplified disk usage history.
        """

        history = self.get_history(
            limit=limit
        )

        return [
            {
                "timestamp": item.get(
                    "timestamp"
                ),
                "usage_percent": item.get(
                    "usage_percent"
                ),
                "free_gb": item.get(
                    "free_gb"
                ),
                "highest_partition_usage": (
                    item.get(
                        "highest_partition_usage"
                    )
                ),
            }
            for item in history
        ]

    def get_average_usage(
        self,
        *,
        samples: int | None = None,
    ) -> float:
        """
        Calculate average disk usage from history.
        """

        history = self.get_history(
            limit=samples
        )

        values = [
            float(
                item.get(
                    "usage_percent",
                    0,
                )
            )
            for item in history
            if isinstance(
                item.get(
                    "usage_percent"
                ),
                (int, float),
            )
        ]

        if not values:
            return 0.0

        return round(
            sum(values)
            / len(values),
            2,
        )

    def clear_history(
        self,
    ) -> None:
        """
        Clear disk monitoring history.
        """

        self.history.clear()

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
    ) -> None:
        """
        Reset disk monitor state.
        """

        self.history.clear()

        self.last_stats = None

        self.last_update_time = None


__all__ = [
    "DiskMonitor",
]


