"""
RENIX CPU Monitor
=================

Provides detailed CPU monitoring including:
- Overall CPU usage
- Per-core usage
- Physical and logical core counts
- CPU frequency
- Load averages where supported
- CPU times
- Basic history tracking
"""

from __future__ import annotations

import platform
import time
from collections import deque
from datetime import datetime
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


class CPUMonitor:
    """
    Detailed CPU monitoring component for RENIX.

    Example:
        monitor = CPUMonitor()

        stats = monitor.get_stats()

        print(stats["percent"])
        print(stats["per_core"])
    """

    def __init__(
        self,
        *,
        history_size: int = 120,
        sample_interval: float = 0.1,
    ) -> None:

        self.history_size = max(
            1,
            int(history_size),
        )

        self.sample_interval = max(
            0.0,
            float(sample_interval),
        )

        self.history = deque(
            maxlen=self.history_size
        )

        self.last_stats: dict[
            str,
            Any,
        ] | None = None

        self.last_update_time: (
            float | None
        ) = None

    # ============================================================
    # MAIN API
    # ============================================================

    def get_stats(
        self,
        *,
        interval: float | None = None,
        store: bool = True,
    ) -> dict[str, Any]:
        """
        Collect complete CPU statistics.

        Args:
            interval:
                Sampling interval for CPU percentage.
                If None, uses the monitor default.

            store:
                Whether to store the result in history.
        """

        if psutil is None:
            return {
                "available": False,
                "error": (
                    "psutil is not installed."
                ),
            }

        try:

            sample_interval = (
                self.sample_interval
                if interval is None
                else max(
                    0.0,
                    float(interval),
                )
            )

            overall_percent = (
                psutil.cpu_percent(
                    interval=sample_interval
                )
            )

            per_core = (
                psutil.cpu_percent(
                    interval=None,
                    percpu=True,
                )
            )

            frequency = (
                self._get_frequency()
            )

            cpu_times = (
                self._get_cpu_times()
            )

            load_average = (
                self._get_load_average()
            )

            stats = {
                "available": True,
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "percent": overall_percent,
                "per_core": per_core,
                "physical_cores": (
                    psutil.cpu_count(
                        logical=False
                    )
                ),
                "logical_processors": (
                    psutil.cpu_count(
                        logical=True
                    )
                ),
                "frequency": frequency,
                "load_average": load_average,
                "cpu_times": cpu_times,
                "processor": (
                    platform.processor()
                ),
                "architecture": (
                    platform.machine()
                ),
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
    # CPU USAGE
    # ============================================================

    def get_usage(
        self,
        *,
        interval: float | None = None,
    ) -> float:

        """
        Return overall CPU usage percentage.
        """

        if psutil is None:
            return 0.0

        try:

            sample_interval = (
                self.sample_interval
                if interval is None
                else max(
                    0.0,
                    float(interval),
                )
            )

            return float(
                psutil.cpu_percent(
                    interval=sample_interval
                )
            )

        except Exception:
            return 0.0

    def get_per_core_usage(
        self,
        *,
        interval: float | None = None,
    ) -> list[float]:

        """
        Return CPU usage for each logical processor.
        """

        if psutil is None:
            return []

        try:

            sample_interval = (
                self.sample_interval
                if interval is None
                else max(
                    0.0,
                    float(interval),
                )
            )

            return list(
                psutil.cpu_percent(
                    interval=sample_interval,
                    percpu=True,
                )
            )

        except Exception:
            return []

    # ============================================================
    # CPU CORES
    # ============================================================

    @staticmethod
    def get_physical_cores(
    ) -> int | None:

        """
        Return the number of physical CPU cores.
        """

        if psutil is None:
            return None

        try:

            return psutil.cpu_count(
                logical=False
            )

        except Exception:
            return None

    @staticmethod
    def get_logical_processors(
    ) -> int | None:

        """
        Return the number of logical processors.
        """

        if psutil is None:
            return None

        try:

            return psutil.cpu_count(
                logical=True
            )

        except Exception:
            return None

    # ============================================================
    # CPU FREQUENCY
    # ============================================================

    def get_frequency(
        self,
    ) -> dict[str, float | None]:

        """
        Return CPU frequency information.
        """

        return self._get_frequency()

    @staticmethod
    def _get_frequency(
    ) -> dict[str, float | None]:

        if psutil is None:
            return {
                "current_mhz": None,
                "min_mhz": None,
                "max_mhz": None,
            }

        try:

            frequency = psutil.cpu_freq()

            if frequency is None:
                return {
                    "current_mhz": None,
                    "min_mhz": None,
                    "max_mhz": None,
                }

            return {
                "current_mhz": (
                    float(
                        frequency.current
                    )
                    if frequency.current
                    is not None
                    else None
                ),
                "min_mhz": (
                    float(
                        frequency.min
                    )
                    if frequency.min
                    is not None
                    else None
                ),
                "max_mhz": (
                    float(
                        frequency.max
                    )
                    if frequency.max
                    is not None
                    else None
                ),
            }

        except Exception:

            return {
                "current_mhz": None,
                "min_mhz": None,
                "max_mhz": None,
            }

    # ============================================================
    # CPU TIMES
    # ============================================================

    @staticmethod
    def _get_cpu_times(
    ) -> dict[str, float]:

        if psutil is None:
            return {}

        try:

            times = psutil.cpu_times()

            return {
                field: float(
                    getattr(
                        times,
                        field,
                    )
                )
                for field in times._fields
            }

        except Exception:
            return {}

    def get_cpu_times(
        self,
    ) -> dict[str, float]:

        """
        Return CPU time statistics.
        """

        return self._get_cpu_times()

    # ============================================================
    # LOAD AVERAGE
    # ============================================================

    @staticmethod
    def _get_load_average(
    ) -> dict[str, float | None]:

        try:

            if hasattr(
                psutil,
                "getloadavg",
            ):

                load_1,
                load_5,
                load_15 = (
                    psutil.getloadavg()
                )

                return {
                    "1_min": float(
                        load_1
                    ),
                    "5_min": float(
                        load_5
                    ),
                    "15_min": float(
                        load_15
                    ),
                }

        except Exception:
            pass

        return {
            "1_min": None,
            "5_min": None,
            "15_min": None,
        }

    def get_load_average(
        self,
    ) -> dict[str, float | None]:

        """
        Return system load averages when supported.
        """

        return self._get_load_average()

    # ============================================================
    # CPU STATUS
    # ============================================================

    def get_status(
        self,
    ) -> dict[str, Any]:

        """
        Return a simplified CPU health status.
        """

        usage = self.get_usage()

        if usage >= 95:
            status = "critical"

        elif usage >= 85:
            status = "high"

        elif usage >= 65:
            status = "moderate"

        else:
            status = "normal"

        return {
            "status": status,
            "usage_percent": usage,
            "healthy": usage < 85,
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
        Return collected CPU monitoring history.
        """

        items = list(
            self.history
        )

        if limit is None:
            return items

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:
            return []

        return items[-limit:]

    def get_usage_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:

        """
        Return simplified CPU usage history.
        """

        history = self.get_history(
            limit=limit
        )

        return [
            {
                "timestamp": item.get(
                    "timestamp"
                ),
                "percent": item.get(
                    "percent"
                ),
            }
            for item in history
        ]

    def clear_history(
        self,
    ) -> None:

        """
        Clear stored CPU monitoring history.
        """

        self.history.clear()

    # ============================================================
    # ANALYSIS
    # ============================================================

    def get_average_usage(
        self,
        *,
        samples: int | None = None,
    ) -> float:

        """
        Calculate average CPU usage from history.
        """

        history = self.get_history(
            limit=samples
        )

        values = [
            item.get(
                "percent",
                0,
            )
            for item in history
            if isinstance(
                item.get(
                    "percent"
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

    def get_peak_usage(
        self,
        *,
        samples: int | None = None,
    ) -> float:

        """
        Return highest CPU usage from history.
        """

        history = self.get_history(
            limit=samples
        )

        values = [
            item.get(
                "percent",
                0,
            )
            for item in history
            if isinstance(
                item.get(
                    "percent"
                ),
                (int, float),
            )
        ]

        return (
            max(values)
            if values
            else 0.0
        )

    def analyze(
        self,
    ) -> dict[str, Any]:

        """
        Analyze current CPU performance.
        """

        stats = self.get_stats()

        if not stats.get(
            "available"
        ):
            return {
                "available": False,
                "status": "unknown",
                "issues": [],
            }

        usage = float(
            stats.get(
                "percent",
                0.0,
            )
        )

        issues: list[str] = []
        recommendations: list[
            str
        ] = []

        if usage >= 95:

            issues.append(
                "Critical CPU usage detected."
            )

            recommendations.append(
                "Close unnecessary "
                "high-resource applications."
            )

        elif usage >= 85:

            issues.append(
                "CPU usage is very high."
            )

            recommendations.append(
                "Check running processes "
                "for heavy applications."
            )

        elif usage >= 70:

            issues.append(
                "CPU usage is moderately high."
            )

        average_usage = (
            self.get_average_usage()
        )

        peak_usage = (
            self.get_peak_usage()
        )

        return {
            "available": True,
            "status": (
                "critical"
                if usage >= 95
                else "high"
                if usage >= 85
                else "moderate"
                if usage >= 70
                else "normal"
            ),
            "current_usage": usage,
            "average_usage": average_usage,
            "peak_usage": peak_usage,
            "issues": issues,
            "recommendations": recommendations,
        }

    # ============================================================
    # UTILITIES
    # ============================================================

    def reset(
        self,
    ) -> None:

        """
        Reset monitoring state.
        """

        self.history.clear()
        self.last_stats = None
        self.last_update_time = None


__all__ = [
    "CPUMonitor",
]


