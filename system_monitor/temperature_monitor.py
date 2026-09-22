"""
RENIX Temperature Monitor
=========================

Provides hardware temperature monitoring.

Features:
- CPU temperature detection
- GPU temperature detection when supported
- Multiple temperature sensors
- Temperature history
- High temperature alerts
- Health analysis

Dependency:
    pip install psutil

Note:
    Temperature sensor availability depends on
    the operating system and hardware.
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


class TemperatureMonitor:
    """
    Hardware temperature monitoring component
    for RENIX.

    Example:
        monitor = TemperatureMonitor()

        stats = monitor.get_stats()

        print(stats["sensors"])
    """

    def __init__(
        self,
        *,
        history_size: int = 120,
        warning_temperature: float = 80.0,
        critical_temperature: float = 95.0,
    ) -> None:

        self.history_size = max(
            1,
            int(history_size),
        )

        self.warning_temperature = float(
            warning_temperature
        )

        self.critical_temperature = float(
            critical_temperature
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
        Collect all available temperature data.
        """

        if psutil is None:

            return {
                "available": False,
                "error": (
                    "psutil is not installed."
                ),
            }

        try:

            sensors = (
                self._get_temperature_sensors()
            )

            summary = (
                self._create_summary(
                    sensors
                )
            )

            stats = {
                "available": bool(
                    sensors
                ),
                "timestamp": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "sensors": sensors,
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
    # SENSOR COLLECTION
    # ============================================================

    def _get_temperature_sensors(
        self,
    ) -> list[dict[str, Any]]:
        """
        Collect available temperature sensors.

        psutil.sensors_temperatures() is not
        supported on every operating system.
        """

        if psutil is None:
            return []

        sensors: list[
            dict[str, Any]
        ] = []

        if not hasattr(
            psutil,
            "sensors_temperatures",
        ):
            return sensors

        try:

            temperature_data = (
                psutil.sensors_temperatures()
            )

            if not temperature_data:
                return sensors

            for group_name, entries in (
                temperature_data.items()
            ):

                for index, entry in enumerate(
                    entries
                ):

                    current = getattr(
                        entry,
                        "current",
                        None,
                    )

                    high = getattr(
                        entry,
                        "high",
                        None,
                    )

                    critical = getattr(
                        entry,
                        "critical",
                        None,
                    )

                    label = getattr(
                        entry,
                        "label",
                        ""
                    )

                    sensor = {
                        "group": group_name,
                        "label": (
                            label
                            or f"Sensor {index + 1}"
                        ),
                        "current": (
                            float(current)
                            if current is not None
                            else None
                        ),
                        "high": (
                            float(high)
                            if high is not None
                            else None
                        ),
                        "critical": (
                            float(critical)
                            if critical is not None
                            else None
                        ),
                    }

                    sensor["status"] = (
                        self._get_sensor_status(
                            sensor
                        )
                    )

                    sensors.append(
                        sensor
                    )

        except (
            AttributeError,
            OSError,
            PermissionError,
        ):
            return []

        except Exception:
            return []

        return sensors

    # ============================================================
    # SENSOR STATUS
    # ============================================================

    def _get_sensor_status(
        self,
        sensor: dict[str, Any],
    ) -> str:
        """
        Determine health status for one sensor.
        """

        current = sensor.get(
            "current"
        )

        if current is None:
            return "unknown"

        current = float(
            current
        )

        critical = sensor.get(
            "critical"
        )

        high = sensor.get(
            "high"
        )

        if (
            critical is not None
            and current >= float(critical)
        ):
            return "critical"

        if current >= self.critical_temperature:
            return "critical"

        if (
            high is not None
            and current >= float(high)
        ):
            return "high"

        if current >= self.warning_temperature:
            return "high"

        if current >= (
            self.warning_temperature - 10
        ):
            return "moderate"

        return "normal"

    # ============================================================
    # SUMMARY
    # ============================================================

    def _create_summary(
        self,
        sensors: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Create summary from all sensors.
        """

        temperatures = [
            float(sensor["current"])
            for sensor in sensors
            if sensor.get(
                "current"
            ) is not None
        ]

        if not temperatures:

            return {
                "sensor_count": len(
                    sensors
                ),
                "average_temperature": None,
                "highest_temperature": None,
                "lowest_temperature": None,
                "status": "unavailable",
            }

        highest = max(
            temperatures
        )

        lowest = min(
            temperatures
        )

        average = (
            sum(temperatures)
            / len(temperatures)
        )

        if highest >= (
            self.critical_temperature
        ):

            status = "critical"

        elif highest >= (
            self.warning_temperature
        ):

            status = "high"

        elif highest >= (
            self.warning_temperature - 10
        ):

            status = "moderate"

        else:

            status = "normal"

        return {
            "sensor_count": len(
                sensors
            ),
            "average_temperature": (
                round(
                    average,
                    2,
                )
            ),
            "highest_temperature": (
                round(
                    highest,
                    2,
                )
            ),
            "lowest_temperature": (
                round(
                    lowest,
                    2,
                )
            ),
            "status": status,
        }

    # ============================================================
    # QUICK ACCESS
    # ============================================================

    def get_highest_temperature(
        self,
    ) -> float | None:
        """
        Return highest detected temperature.
        """

        stats = self.get_stats(
            store=False
        )

        value = stats.get(
            "highest_temperature"
        )

        if isinstance(
            value,
            (int, float),
        ):
            return float(
                value
            )

        return None

    def get_average_temperature(
        self,
    ) -> float | None:
        """
        Return average detected temperature.
        """

        stats = self.get_stats(
            store=False
        )

        value = stats.get(
            "average_temperature"
        )

        if isinstance(
            value,
            (int, float),
        ):
            return float(
                value
            )

        return None

    def get_sensor(
        self,
        label: str,
    ) -> dict[str, Any] | None:
        """
        Find a sensor by label.
        """

        if not label:
            return None

        query = label.lower().strip()

        stats = self.get_stats(
            store=False
        )

        for sensor in stats.get(
            "sensors",
            [],
        ):

            sensor_label = str(
                sensor.get(
                    "label",
                    "",
                )
            ).lower()

            group = str(
                sensor.get(
                    "group",
                    "",
                )
            ).lower()

            if (
                query in sensor_label
                or query in group
            ):

                return sensor

        return None

    def get_sensors_by_group(
        self,
        group: str,
    ) -> list[dict[str, Any]]:
        """
        Return sensors belonging to a group.
        """

        if not group:
            return []

        query = group.lower().strip()

        stats = self.get_stats(
            store=False
        )

        return [
            sensor
            for sensor in stats.get(
                "sensors",
                [],
            )
            if query
            in str(
                sensor.get(
                    "group",
                    "",
                )
            ).lower()
        ]

    # ============================================================
    # CPU / GPU HELPERS
    # ============================================================

    def get_cpu_temperature(
        self,
    ) -> float | None:
        """
        Attempt to find CPU temperature.

        Sensor naming varies across platforms,
        so this uses common CPU-related keywords.
        """

        keywords = [
            "cpu",
            "core",
            "package",
            "k10temp",
            "coretemp",
        ]

        temperatures = []

        stats = self.get_stats(
            store=False
        )

        for sensor in stats.get(
            "sensors",
            [],
        ):

            text = (
                f"{sensor.get('group', '')} "
                f"{sensor.get('label', '')}"
            ).lower()

            if any(
                keyword in text
                for keyword in keywords
            ):

                current = sensor.get(
                    "current"
                )

                if isinstance(
                    current,
                    (int, float),
                ):

                    temperatures.append(
                        float(current)
                    )

        if not temperatures:
            return None

        return round(
            max(temperatures),
            2,
        )

    def get_gpu_temperature(
        self,
    ) -> float | None:
        """
        Attempt to find GPU temperature.
        """

        keywords = [
            "gpu",
            "graphics",
            "nvidia",
            "amdgpu",
        ]

        temperatures = []

        stats = self.get_stats(
            store=False
        )

        for sensor in stats.get(
            "sensors",
            [],
        ):

            text = (
                f"{sensor.get('group', '')} "
                f"{sensor.get('label', '')}"
            ).lower()

            if any(
                keyword in text
                for keyword in keywords
            ):

                current = sensor.get(
                    "current"
                )

                if isinstance(
                    current,
                    (int, float),
                ):

                    temperatures.append(
                        float(current)
                    )

        if not temperatures:
            return None

        return round(
            max(temperatures),
            2,
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> dict[str, Any]:
        """
        Return simplified temperature health status.
        """

        stats = self.get_stats(
            store=False
        )

        if not stats.get(
            "available"
        ):

            return {
                "available": False,
                "status": "unavailable",
                "message": (
                    "Temperature sensors are not "
                    "available on this system."
                ),
            }

        return {
            "available": True,
            "status": (
                stats.get(
                    "status"
                )
            ),
            "highest_temperature": (
                stats.get(
                    "highest_temperature"
                )
            ),
            "average_temperature": (
                stats.get(
                    "average_temperature"
                )
            ),
            "healthy": (
                stats.get(
                    "status"
                )
                == "normal"
            ),
        }

    # ============================================================
    # ANALYSIS
    # ============================================================

    def analyze(
        self,
    ) -> dict[str, Any]:
        """
        Analyze system temperature health.
        """

        stats = self.get_stats()

        if not stats.get(
            "available"
        ):

            return {
                "available": False,
                "status": "unavailable",
                "issues": [],
                "recommendations": [
                    (
                        "Hardware temperature sensors "
                        "are not available through "
                        "the current system interface."
                    )
                ],
            }

        issues: list[str] = []

        recommendations: list[
            str
        ] = []

        critical_sensors = []
        high_sensors = []

        for sensor in stats.get(
            "sensors",
            [],
        ):

            status = sensor.get(
                "status"
            )

            label = (
                f"{sensor.get('group')} - "
                f"{sensor.get('label')}"
            )

            if status == "critical":

                critical_sensors.append(
                    label
                )

            elif status == "high":

                high_sensors.append(
                    label
                )

        if critical_sensors:

            issues.append(
                "Critical temperatures detected: "
                + ", ".join(
                    critical_sensors
                )
            )

            recommendations.append(
                "Reduce system load immediately "
                "and check cooling."
            )

        if high_sensors:

            issues.append(
                "High temperatures detected: "
                + ", ".join(
                    high_sensors
                )
            )

            recommendations.append(
                "Check airflow, fans, and "
                "background workloads."
            )

        status = stats.get(
            "status",
            "unknown",
        )

        return {
            "available": True,
            "status": status,
            "highest_temperature": (
                stats.get(
                    "highest_temperature"
                )
            ),
            "average_temperature": (
                stats.get(
                    "average_temperature"
                )
            ),
            "cpu_temperature": (
                self.get_cpu_temperature()
            ),
            "gpu_temperature": (
                self.get_gpu_temperature()
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
        Return temperature monitoring history.
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

    def get_temperature_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return simplified temperature history.
        """

        history = self.get_history(
            limit=limit
        )

        return [
            {
                "timestamp": item.get(
                    "timestamp"
                ),
                "average_temperature": (
                    item.get(
                        "average_temperature"
                    )
                ),
                "highest_temperature": (
                    item.get(
                        "highest_temperature"
                    )
                ),
                "status": item.get(
                    "status"
                ),
            }
            for item in history
        ]

    def get_peak_temperature(
        self,
        *,
        samples: int | None = None,
    ) -> float | None:
        """
        Return highest recorded temperature.
        """

        history = self.get_history(
            limit=samples
        )

        values = [
            float(
                item.get(
                    "highest_temperature"
                )
            )
            for item in history
            if isinstance(
                item.get(
                    "highest_temperature"
                ),
                (int, float),
            )
        ]

        if not values:
            return None

        return round(
            max(values),
            2,
        )

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear temperature history.
        """

        self.history.clear()

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
    ) -> None:
        """
        Reset temperature monitor state.
        """

        self.history.clear()

        self.last_stats = None

        self.last_update_time = None


__all__ = [
    "TemperatureMonitor",
]


