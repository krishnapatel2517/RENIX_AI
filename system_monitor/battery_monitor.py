"""
RENIX Battery Monitor
=====================

Provides battery and power monitoring.

Features:
- Battery percentage
- Charging status
- Remaining time estimation
- Power plug status
- Battery health status
- History tracking
- Low battery warnings

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


class BatteryMonitor:
    """
    Battery monitoring component for RENIX.

    Example:
        monitor = BatteryMonitor()

        stats = monitor.get_stats()

        print(stats["percent"])
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
        Collect complete battery statistics.
        """

        if psutil is None:

            return {
                "available": False,
                "error": (
                    "psutil is not installed."
                ),
            }

        try:

            battery = psutil.sensors_battery()

            if battery is None:

                return {
                    "available": False,
                    "has_battery": False,
                    "message": (
                        "No battery detected."
                    ),
                    "timestamp": (
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    ),
                }

            percent = float(
                battery.percent
            )

            plugged = bool(
                battery.power_plugged
            )

            seconds_left = (
                self._normalize_seconds(
                    battery.secsleft
                )
            )

            stats = {
                "available": True,
                "has_battery": True,
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "percent": percent,
                "power_plugged": plugged,
                "charging": plugged,
                "seconds_left": (
                    seconds_left
                ),
                "time_remaining": (
                    self._format_time(
                        seconds_left
                    )
                ),
                "status": (
                    self._get_battery_status(
                        percent,
                        plugged,
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
    # BATTERY STATUS
    # ============================================================

    @staticmethod
    def _get_battery_status(
        percent: float,
        plugged: bool,
    ) -> str:
        """
        Determine battery status.
        """

        if plugged:

            if percent >= 100:
                return "fully_charged"

            return "charging"

        if percent <= 5:
            return "critical"

        if percent <= 15:
            return "very_low"

        if percent <= 30:
            return "low"

        if percent <= 60:
            return "medium"

        return "good"

    # ============================================================
    # TIME HANDLING
    # ============================================================

    @staticmethod
    def _normalize_seconds(
        seconds: Any,
    ) -> int | None:
        """
        Normalize battery remaining seconds.

        psutil may return special negative constants
        when remaining time cannot be determined.
        """

        if not isinstance(
            seconds,
            (int, float),
        ):
            return None

        seconds = int(
            seconds
        )

        if seconds < 0:
            return None

        return seconds

    @staticmethod
    def _format_time(
        seconds: int | None,
    ) -> str | None:
        """
        Convert seconds into readable time.
        """

        if seconds is None:
            return None

        hours = (
            seconds // 3600
        )

        minutes = (
            seconds % 3600
        ) // 60

        remaining_seconds = (
            seconds % 60
        )

        if hours > 0:

            return (
                f"{hours}h "
                f"{minutes}m"
            )

        if minutes > 0:

            return (
                f"{minutes}m "
                f"{remaining_seconds}s"
            )

        return (
            f"{remaining_seconds}s"
        )

    # ============================================================
    # QUICK METRICS
    # ============================================================

    def get_percentage(
        self,
    ) -> float | None:
        """
        Return battery percentage.
        """

        stats = self.get_stats(
            store=False
        )

        if not stats.get(
            "has_battery",
            False,
        ):
            return None

        percent = stats.get(
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

    def is_charging(
        self,
    ) -> bool | None:
        """
        Return whether battery is charging.
        """

        stats = self.get_stats(
            store=False
        )

        if not stats.get(
            "has_battery",
            False,
        ):
            return None

        return bool(
            stats.get(
                "charging",
                False,
            )
        )

    def is_plugged(
        self,
    ) -> bool | None:
        """
        Return whether power adapter is connected.
        """

        stats = self.get_stats(
            store=False
        )

        if not stats.get(
            "has_battery",
            False,
        ):
            return None

        return bool(
            stats.get(
                "power_plugged",
                False,
            )
        )

    def get_remaining_time(
        self,
    ) -> int | None:
        """
        Return remaining battery time in seconds.
        """

        stats = self.get_stats(
            store=False
        )

        return stats.get(
            "seconds_left"
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> dict[str, Any]:
        """
        Return simplified battery health status.
        """

        stats = self.get_stats(
            store=False
        )

        if not stats.get(
            "has_battery",
            False,
        ):

            return {
                "available": False,
                "has_battery": False,
                "status": "not_available",
            }

        percent = float(
            stats.get(
                "percent",
                0.0,
            )
        )

        plugged = bool(
            stats.get(
                "power_plugged",
                False,
            )
        )

        if plugged:

            status = "charging"

        elif percent <= 5:

            status = "critical"

        elif percent <= 15:

            status = "very_low"

        elif percent <= 30:

            status = "low"

        else:

            status = "normal"

        return {
            "available": True,
            "has_battery": True,
            "status": status,
            "percent": percent,
            "power_plugged": plugged,
            "charging": plugged,
            "time_remaining": (
                stats.get(
                    "time_remaining"
                )
            ),
        }

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(
        self,
    ) -> dict[str, Any]:
        """
        Analyze current battery condition.
        """

        stats = self.get_stats()

        if not stats.get(
            "has_battery",
            False,
        ):

            return {
                "available": False,
                "has_battery": False,
                "status": "not_available",
                "issues": [],
                "recommendations": [],
            }

        percent = float(
            stats.get(
                "percent",
                0.0,
            )
        )

        plugged = bool(
            stats.get(
                "power_plugged",
                False,
            )
        )

        issues: list[str] = []

        recommendations: list[
            str
        ] = []

        if not plugged:

            if percent <= 5:

                issues.append(
                    "Battery level is critical."
                )

                recommendations.append(
                    "Connect the charger immediately."
                )

                status = "critical"

            elif percent <= 15:

                issues.append(
                    "Battery level is very low."
                )

                recommendations.append(
                    "Connect the charger soon."
                )

                status = "very_low"

            elif percent <= 30:

                issues.append(
                    "Battery level is low."
                )

                recommendations.append(
                    "Consider charging the device."
                )

                status = "low"

            else:

                status = "normal"

        else:

            status = "charging"

            if percent >= 100:

                status = "fully_charged"

        return {
            "available": True,
            "has_battery": True,
            "status": status,
            "percent": percent,
            "power_plugged": plugged,
            "time_remaining": (
                stats.get(
                    "time_remaining"
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
        Return battery monitoring history.
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

    def get_percentage_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return simplified battery percentage history.
        """

        history = self.get_history(
            limit=limit
        )

        result = []

        for item in history:

            result.append(
                {
                    "timestamp": item.get(
                        "timestamp"
                    ),
                    "percent": item.get(
                        "percent"
                    ),
                    "charging": item.get(
                        "charging"
                    ),
                    "status": item.get(
                        "status"
                    ),
                }
            )

        return result

    def get_average_percentage(
        self,
        *,
        samples: int | None = None,
    ) -> float:
        """
        Calculate average recorded battery percentage.
        """

        history = self.get_history(
            limit=samples
        )

        values = []

        for item in history:

            percent = item.get(
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

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear battery monitoring history.
        """

        self.history.clear()

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
    ) -> None:
        """
        Reset battery monitor state.
        """

        self.history.clear()

        self.last_stats = None

        self.last_update_time = None


__all__ = [
    "BatteryMonitor",
]


