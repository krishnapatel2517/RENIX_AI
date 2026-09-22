"""
RENIX RAM Monitor
=================

Provides detailed RAM and virtual memory monitoring.

Features:
- Total RAM
- Used RAM
- Available RAM
- RAM percentage
- Swap memory
- Memory history
- Health analysis
- High memory usage alerts

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


class RAMMonitor:
    """
    RAM monitoring component for RENIX.

    Example:
        monitor = RAMMonitor()

        stats = monitor.get_stats()

        print(stats["memory"]["percent"])
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
        Collect complete RAM statistics.
        """

        if psutil is None:

            return {
                "available": False,
                "error": (
                    "psutil is not installed."
                ),
            }

        try:

            virtual_memory = (
                psutil.virtual_memory()
            )

            swap_memory = (
                psutil.swap_memory()
            )

            stats = {
                "available": True,
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "memory": (
                    self._format_virtual_memory(
                        virtual_memory
                    )
                ),
                "swap": (
                    self._format_swap_memory(
                        swap_memory
                    )
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
    # MEMORY FORMATTING
    # ============================================================

    @staticmethod
    def _format_virtual_memory(
        memory: Any,
    ) -> dict[str, Any]:
        """
        Format psutil virtual memory data.
        """

        total = int(
            memory.total
        )

        used = int(
            memory.used
        )

        available = int(
            memory.available
        )

        free = int(
            memory.free
        )

        percent = float(
            memory.percent
        )

        return {
            "total_bytes": total,
            "used_bytes": used,
            "available_bytes": available,
            "free_bytes": free,
            "percent": percent,

            "total_mb": (
                round(
                    total / 1024 / 1024,
                    2,
                )
            ),

            "used_mb": (
                round(
                    used / 1024 / 1024,
                    2,
                )
            ),

            "available_mb": (
                round(
                    available
                    / 1024
                    / 1024,
                    2,
                )
            ),

            "free_mb": (
                round(
                    free / 1024 / 1024,
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

            "available_gb": (
                round(
                    available
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

    @staticmethod
    def _format_swap_memory(
        swap: Any,
    ) -> dict[str, Any]:
        """
        Format swap memory information.
        """

        total = int(
            swap.total
        )

        used = int(
            swap.used
        )

        free = int(
            swap.free
        )

        percent = float(
            swap.percent
        )

        return {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "percent": percent,

            "total_mb": (
                round(
                    total / 1024 / 1024,
                    2,
                )
            ),

            "used_mb": (
                round(
                    used / 1024 / 1024,
                    2,
                )
            ),

            "free_mb": (
                round(
                    free / 1024 / 1024,
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
    # QUICK MEMORY ACCESS
    # ============================================================

    def get_usage(
        self,
    ) -> float:
        """
        Return RAM usage percentage.
        """

        if psutil is None:
            return 0.0

        try:

            return float(
                psutil.virtual_memory().percent
            )

        except Exception:
            return 0.0

    def get_total_memory(
        self,
    ) -> int:
        """
        Return total RAM in bytes.
        """

        if psutil is None:
            return 0

        try:

            return int(
                psutil.virtual_memory().total
            )

        except Exception:
            return 0

    def get_used_memory(
        self,
    ) -> int:
        """
        Return used RAM in bytes.
        """

        if psutil is None:
            return 0

        try:

            return int(
                psutil.virtual_memory().used
            )

        except Exception:
            return 0

    def get_available_memory(
        self,
    ) -> int:
        """
        Return available RAM in bytes.
        """

        if psutil is None:
            return 0

        try:

            return int(
                psutil.virtual_memory().available
            )

        except Exception:
            return 0

    def get_free_memory(
        self,
    ) -> int:
        """
        Return free RAM in bytes.
        """

        if psutil is None:
            return 0

        try:

            return int(
                psutil.virtual_memory().free
            )

        except Exception:
            return 0

    # ============================================================
    # SWAP MEMORY
    # ============================================================

    def get_swap_stats(
        self,
    ) -> dict[str, Any]:
        """
        Return swap memory statistics.
        """

        if psutil is None:
            return {}

        try:

            return (
                self._format_swap_memory(
                    psutil.swap_memory()
                )
            )

        except Exception:
            return {}

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> dict[str, Any]:
        """
        Return simplified RAM health status.
        """

        usage = self.get_usage()

        available = (
            self.get_available_memory()
        )

        if usage >= 95:

            status = "critical"

        elif usage >= 85:

            status = "high"

        elif usage >= 70:

            status = "moderate"

        else:

            status = "normal"

        return {
            "status": status,
            "usage_percent": usage,
            "available_bytes": available,
            "healthy": usage < 85,
        }

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(
        self,
    ) -> dict[str, Any]:
        """
        Analyze current memory usage.
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

        memory = stats.get(
            "memory",
            {},
        )

        swap = stats.get(
            "swap",
            {},
        )

        usage = float(
            memory.get(
                "percent",
                0.0,
            )
        )

        swap_usage = float(
            swap.get(
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
                "Critical RAM usage detected."
            )

            recommendations.append(
                "Close unnecessary applications "
                "immediately."
            )

        elif usage >= 85:

            issues.append(
                "RAM usage is very high."
            )

            recommendations.append(
                "Check running applications "
                "for excessive memory usage."
            )

        elif usage >= 70:

            issues.append(
                "RAM usage is moderately high."
            )

        if swap_usage >= 80:

            issues.append(
                "High swap memory usage detected."
            )

            recommendations.append(
                "Consider closing memory-heavy "
                "applications."
            )

        if usage >= 95:

            status = "critical"

        elif usage >= 85:

            status = "high"

        elif usage >= 70:

            status = "moderate"

        else:

            status = "normal"

        return {
            "available": True,
            "status": status,
            "current_usage": usage,
            "swap_usage": swap_usage,
            "average_usage": (
                self.get_average_usage()
            ),
            "peak_usage": (
                self.get_peak_usage()
            ),
            "issues": issues,
            "recommendations": recommendations,
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
        Return RAM monitoring history.
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
        Return simplified RAM usage history.
        """

        history = self.get_history(
            limit=limit
        )

        result = []

        for item in history:

            memory = item.get(
                "memory",
                {},
            )

            swap = item.get(
                "swap",
                {},
            )

            result.append(
                {
                    "timestamp": item.get(
                        "timestamp"
                    ),
                    "memory_percent": (
                        memory.get(
                            "percent"
                        )
                        if isinstance(
                            memory,
                            dict,
                        )
                        else None
                    ),
                    "swap_percent": (
                        swap.get(
                            "percent"
                        )
                        if isinstance(
                            swap,
                            dict,
                        )
                        else None
                    ),
                }
            )

        return result

    def clear_history(
        self,
    ) -> None:
        """
        Clear stored RAM history.
        """

        self.history.clear()

    # ============================================================
    # STATISTICS
    # ============================================================

    def get_average_usage(
        self,
        *,
        samples: int | None = None,
    ) -> float:
        """
        Calculate average RAM usage.
        """

        history = self.get_history(
            limit=samples
        )

        values = []

        for item in history:

            memory = item.get(
                "memory",
                {},
            )

            if isinstance(
                memory,
                dict,
            ):

                percent = memory.get(
                    "percent"
                )

                if isinstance(
                    percent,
                    (int, float),
                ):

                    values.append(
                        float(percent)
                    )

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
        Return highest recorded RAM usage.
        """

        history = self.get_history(
            limit=samples
        )

        values = []

        for item in history:

            memory = item.get(
                "memory",
                {},
            )

            if isinstance(
                memory,
                dict,
            ):

                percent = memory.get(
                    "percent"
                )

                if isinstance(
                    percent,
                    (int, float),
                ):

                    values.append(
                        float(percent)
                    )

        if not values:
            return 0.0

        return max(
            values
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
    ) -> None:
        """
        Reset RAM monitor state.
        """

        self.history.clear()

        self.last_stats = None

        self.last_update_time = None


__all__ = [
    "RAMMonitor",
]


