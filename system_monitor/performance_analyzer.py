"""
RENIX Performance Analyzer
==========================

Analyzes system performance using data from:
- CPU
- GPU
- RAM
- Disk
- Network
- Battery
- Temperature
- Processes

Produces:
- Overall performance score
- Bottleneck detection
- Health status
- Performance recommendations
- Historical trends

This module can work independently or receive data
from the other RENIX system monitor modules.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from statistics import mean
from typing import Any


class PerformanceAnalyzer:
    """
    RENIX system performance analyzer.

    Example:
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze({
            "cpu": {"percent": 45},
            "ram": {"percent": 62},
            "disk": {"percent": 20},
        })
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

        self.last_report: (
            dict[str, Any] | None
        ) = None

    # ============================================================
    # MAIN ANALYSIS
    # ============================================================

    def analyze(
        self,
        metrics: dict[str, Any],
        *,
        store: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze complete system performance.

        Expected structure:

        {
            "cpu": {...},
            "gpu": {...},
            "ram": {...},
            "disk": {...},
            "network": {...},
            "battery": {...},
            "temperature": {...},
            "processes": {...},
        }
        """

        metrics = metrics or {}

        component_reports = {
            "cpu": self._analyze_cpu(
                metrics.get("cpu", {})
            ),
            "gpu": self._analyze_gpu(
                metrics.get("gpu", {})
            ),
            "ram": self._analyze_ram(
                metrics.get("ram", {})
            ),
            "disk": self._analyze_disk(
                metrics.get("disk", {})
            ),
            "network": self._analyze_network(
                metrics.get("network", {})
            ),
            "battery": self._analyze_battery(
                metrics.get("battery", {})
            ),
            "temperature": (
                self._analyze_temperature(
                    metrics.get(
                        "temperature",
                        {},
                    )
                )
            ),
            "processes": (
                self._analyze_processes(
                    metrics.get(
                        "processes",
                        {},
                    )
                )
            ),
        }

        available_reports = [
            report
            for report in (
                component_reports.values()
            )
            if report.get("available")
        ]

        scores = [
            float(
                report.get(
                    "score",
                    100,
                )
            )
            for report in available_reports
        ]

        if scores:

            overall_score = round(
                sum(scores)
                / len(scores),
                2,
            )

        else:

            overall_score = 100.0

        bottlenecks = (
            self._detect_bottlenecks(
                component_reports
            )
        )

        issues = []

        recommendations = []

        for report in (
            component_reports.values()
        ):

            issues.extend(
                report.get(
                    "issues",
                    [],
                )
            )

            recommendations.extend(
                report.get(
                    "recommendations",
                    [],
                )
            )

        if bottlenecks:

            for bottleneck in bottlenecks:

                name = bottleneck[
                    "component"
                ]

                severity = bottleneck[
                    "severity"
                ]

                issues.append(
                    f"{name.upper()} may be a "
                    f"{severity} bottleneck."
                )

        status = (
            self._calculate_status(
                overall_score
            )
        )

        report = {
            "timestamp": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "overall_score": (
                overall_score
            ),
            "status": status,
            "healthy": (
                status in {
                    "excellent",
                    "good",
                }
            ),
            "components": (
                component_reports
            ),
            "bottlenecks": (
                bottlenecks
            ),
            "issues": (
                self._unique_items(
                    issues
                )
            ),
            "recommendations": (
                self._unique_items(
                    recommendations
                )
            ),
        }

        self.last_report = report

        if store:

            self.history.append(
                report
            )

        return report

    # ============================================================
    # CPU
    # ============================================================

    def _analyze_cpu(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if not data:
            return self._unavailable_report()

        percent = self._extract_number(
            data,
            [
                "percent",
                "usage_percent",
                "cpu_percent",
            ],
        )

        if percent is None:
            return self._unavailable_report()

        score = self._usage_score(
            percent
        )

        issues = []
        recommendations = []

        if percent >= 95:

            issues.append(
                "CPU usage is critically high."
            )

            recommendations.append(
                "Close CPU-intensive applications."
            )

        elif percent >= 80:

            issues.append(
                "CPU usage is high."
            )

            recommendations.append(
                "Check background processes."
            )

        return {
            "available": True,
            "score": score,
            "usage_percent": percent,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # GPU
    # ============================================================

    def _analyze_gpu(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if not data:
            return self._unavailable_report()

        percent = self._extract_number(
            data,
            [
                "percent",
                "usage_percent",
                "gpu_percent",
            ],
        )

        if percent is None:
            return self._unavailable_report()

        score = self._usage_score(
            percent
        )

        issues = []
        recommendations = []

        if percent >= 95:

            issues.append(
                "GPU usage is critically high."
            )

            recommendations.append(
                "Reduce graphics-intensive "
                "workloads."
            )

        elif percent >= 85:

            issues.append(
                "GPU usage is high."
            )

            recommendations.append(
                "Check GPU-intensive "
                "applications."
            )

        return {
            "available": True,
            "score": score,
            "usage_percent": percent,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # RAM
    # ============================================================

    def _analyze_ram(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if not data:
            return self._unavailable_report()

        percent = self._extract_number(
            data,
            [
                "percent",
                "usage_percent",
                "memory_percent",
            ],
        )

        if percent is None:
            return self._unavailable_report()

        score = self._usage_score(
            percent
        )

        issues = []
        recommendations = []

        if percent >= 95:

            issues.append(
                "RAM usage is critically high."
            )

            recommendations.append(
                "Close unnecessary "
                "applications."
            )

        elif percent >= 85:

            issues.append(
                "RAM usage is high."
            )

            recommendations.append(
                "Reduce background memory "
                "usage."
            )

        return {
            "available": True,
            "score": score,
            "usage_percent": percent,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # DISK
    # ============================================================

    def _analyze_disk(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if not data:
            return self._unavailable_report()

        percent = self._extract_number(
            data,
            [
                "percent",
                "usage_percent",
                "disk_percent",
            ],
        )

        if percent is None:
            return self._unavailable_report()

        score = self._usage_score(
            percent
        )

        issues = []
        recommendations = []

        if percent >= 95:

            issues.append(
                "Disk space is critically low."
            )

            recommendations.append(
                "Free disk space immediately."
            )

        elif percent >= 85:

            issues.append(
                "Disk usage is high."
            )

            recommendations.append(
                "Clean unnecessary files."
            )

        return {
            "available": True,
            "score": score,
            "usage_percent": percent,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # NETWORK
    # ============================================================

    def _analyze_network(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if not data:
            return self._unavailable_report()

        issues = []
        recommendations = []

        connected = data.get(
            "connected"
        )

        if connected is False:

            issues.append(
                "Network connection is unavailable."
            )

            recommendations.append(
                "Check network connectivity."
            )

            return {
                "available": True,
                "score": 20.0,
                "connected": False,
                "issues": issues,
                "recommendations": (
                    recommendations
                ),
            }

        score = 100.0

        latency = self._extract_number(
            data,
            [
                "latency",
                "latency_ms",
                "ping",
            ],
        )

        if latency is not None:

            if latency >= 300:

                score -= 50

                issues.append(
                    "Network latency is very high."
                )

            elif latency >= 150:

                score -= 30

                issues.append(
                    "Network latency is high."
                )

            elif latency >= 80:

                score -= 10

        return {
            "available": True,
            "score": max(
                0.0,
                score,
            ),
            "connected": (
                connected
                if connected is not None
                else True
            ),
            "latency": latency,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # BATTERY
    # ============================================================

    def _analyze_battery(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if not data:
            return self._unavailable_report()

        has_battery = data.get(
            "has_battery",
            True,
        )

        if not has_battery:
            return self._unavailable_report()

        percent = self._extract_number(
            data,
            [
                "percent",
                "battery_percent",
            ],
        )

        if percent is None:
            return self._unavailable_report()

        score = 100.0

        issues = []
        recommendations = []

        if percent <= 5:

            score = 10.0

            issues.append(
                "Battery level is critical."
            )

            recommendations.append(
                "Connect the charger immediately."
            )

        elif percent <= 15:

            score = 30.0

            issues.append(
                "Battery level is very low."
            )

            recommendations.append(
                "Charge the device soon."
            )

        elif percent <= 30:

            score = 60.0

            issues.append(
                "Battery level is low."
            )

        return {
            "available": True,
            "score": score,
            "percent": percent,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # TEMPERATURE
    # ============================================================

    def _analyze_temperature(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if not data:
            return self._unavailable_report()

        temperature = self._extract_number(
            data,
            [
                "highest_temperature",
                "temperature",
                "average_temperature",
            ],
        )

        if temperature is None:
            return self._unavailable_report()

        score = 100.0

        issues = []
        recommendations = []

        if temperature >= 95:

            score = 10.0

            issues.append(
                "System temperature is critical."
            )

            recommendations.append(
                "Reduce workload and check "
                "system cooling immediately."
            )

        elif temperature >= 85:

            score = 40.0

            issues.append(
                "System temperature is high."
            )

            recommendations.append(
                "Improve airflow and reduce "
                "heavy workloads."
            )

        elif temperature >= 75:

            score = 70.0

            issues.append(
                "System temperature is elevated."
            )

        return {
            "available": True,
            "score": score,
            "temperature": temperature,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # PROCESSES
    # ============================================================

    def _analyze_processes(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        if not data:
            return self._unavailable_report()

        count = self._extract_number(
            data,
            [
                "process_count",
                "count",
            ],
        )

        if count is None:
            return self._unavailable_report()

        score = 100.0

        issues = []
        recommendations = []

        if count >= 500:

            score = 50.0

            issues.append(
                "Very high number of processes."
            )

            recommendations.append(
                "Check unnecessary background "
                "applications."
            )

        elif count >= 350:

            score = 75.0

            issues.append(
                "High number of running processes."
            )

        return {
            "available": True,
            "score": score,
            "process_count": count,
            "issues": issues,
            "recommendations": (
                recommendations
            ),
        }

    # ============================================================
    # BOTTLENECK DETECTION
    # ============================================================

    def _detect_bottlenecks(
        self,
        reports: dict[
            str,
            dict[str, Any],
        ],
    ) -> list[dict[str, Any]]:
        """
        Detect components that may be causing
        performance bottlenecks.
        """

        bottlenecks = []

        for name, report in reports.items():

            if not report.get(
                "available"
            ):
                continue

            score = float(
                report.get(
                    "score",
                    100.0,
                )
            )

            if score <= 30:

                severity = "critical"

            elif score <= 60:

                severity = "high"

            elif score <= 80:

                severity = "moderate"

            else:

                continue

            bottlenecks.append(
                {
                    "component": name,
                    "score": score,
                    "severity": severity,
                }
            )

        bottlenecks.sort(
            key=lambda item: (
                item["score"]
            )
        )

        return bottlenecks

    # ============================================================
    # TREND ANALYSIS
    # ============================================================

    def get_trend(
        self,
        *,
        samples: int | None = None,
    ) -> dict[str, Any]:
        """
        Analyze overall performance trend.
        """

        history = list(
            self.history
        )

        if samples is not None:

            samples = max(
                1,
                int(samples),
            )

            history = history[
                -samples:
            ]

        if len(history) < 2:

            return {
                "trend": "unknown",
                "message": (
                    "Not enough performance "
                    "history available."
                ),
            }

        scores = [
            float(
                item.get(
                    "overall_score",
                    0.0,
                )
            )
            for item in history
        ]

        midpoint = (
            len(scores) // 2
        )

        first_half = scores[
            :midpoint
        ]

        second_half = scores[
            midpoint:
        ]

        first_average = mean(
            first_half
        )

        second_average = mean(
            second_half
        )

        difference = round(
            second_average
            - first_average,
            2,
        )

        if difference >= 5:

            trend = "improving"

        elif difference <= -5:

            trend = "declining"

        else:

            trend = "stable"

        return {
            "trend": trend,
            "difference": difference,
            "average_score": round(
                mean(scores),
                2,
            ),
            "samples": len(
                scores
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
        Return performance history.
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

    def get_score_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return simplified score history.
        """

        history = self.get_history(
            limit=limit
        )

        return [
            {
                "timestamp": item.get(
                    "timestamp"
                ),
                "overall_score": (
                    item.get(
                        "overall_score"
                    )
                ),
                "status": item.get(
                    "status"
                ),
            }
            for item in history
        ]

    # ============================================================
    # STATUS
    # ============================================================

    @staticmethod
    def _calculate_status(
        score: float,
    ) -> str:

        if score >= 90:
            return "excellent"

        if score >= 75:
            return "good"

        if score >= 60:
            return "moderate"

        if score >= 40:
            return "poor"

        return "critical"

    @staticmethod
    def _usage_score(
        usage: float,
    ) -> float:
        """
        Convert usage percentage into a health score.

        Lower usage = better performance score.
        """

        usage = max(
            0.0,
            min(
                100.0,
                float(usage),
            ),
        )

        if usage <= 50:
            return 100.0

        if usage <= 70:
            return 90.0

        if usage <= 80:
            return 75.0

        if usage <= 90:
            return 55.0

        if usage <= 95:
            return 30.0

        return 10.0

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _extract_number(
        data: dict[str, Any],
        keys: list[str],
    ) -> float | None:

        for key in keys:

            value = data.get(
                key
            )

            if isinstance(
                value,
                (int, float),
            ):

                return float(
                    value
                )

        return None

    @staticmethod
    def _unavailable_report(
    ) -> dict[str, Any]:

        return {
            "available": False,
            "score": 100.0,
            "issues": [],
            "recommendations": [],
        }

    @staticmethod
    def _unique_items(
        items: list[str],
    ) -> list[str]:

        seen = set()

        result = []

        for item in items:

            if item not in seen:

                seen.add(
                    item
                )

                result.append(
                    item
                )

        return result

    # ============================================================
    # HISTORY MANAGEMENT
    # ============================================================

    def clear_history(
        self,
    ) -> None:

        self.history.clear()

    def reset(
        self,
    ) -> None:

        self.history.clear()

        self.last_report = None


__all__ = [
    "PerformanceAnalyzer",
]


