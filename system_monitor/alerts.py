"""
RENIX System Monitor Alerts
===========================

Handles system monitoring alerts.

Features:
- CPU alerts
- RAM alerts
- Disk alerts
- Temperature alerts
- Battery alerts
- Network alerts
- Custom alert rules
- Alert cooldowns
- Alert history
- Priority levels
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Any


class SystemAlerts:
    """
    RENIX system monitoring alert manager.

    Example:
        alerts = SystemAlerts()

        triggered = alerts.check({
            "cpu": {"percent": 95},
            "ram": {"percent": 88},
            "temperature": {
                "highest_temperature": 91
            }
        })

        for alert in triggered:
            print(alert["message"])
    """

    PRIORITY_LEVELS = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    def __init__(
        self,
        *,
        history_size: int = 500,
        cooldown_seconds: float = 60.0,
    ) -> None:

        self.history_size = max(
            1,
            int(history_size),
        )

        self.cooldown_seconds = max(
            0.0,
            float(cooldown_seconds),
        )

        self.alert_history: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.history_size
        )

        self.last_triggered: dict[
            str,
            float
        ] = {}

        self.rules = (
            self._create_default_rules()
        )

        self.enabled = True

    # ============================================================
    # DEFAULT RULES
    # ============================================================

    def _create_default_rules(
        self,
    ) -> dict[str, dict[str, Any]]:

        return {
            "cpu_high": {
                "enabled": True,
                "component": "cpu",
                "metric": "percent",
                "operator": ">=",
                "threshold": 85.0,
                "priority": "high",
                "message": (
                    "CPU usage is high."
                ),
            },

            "cpu_critical": {
                "enabled": True,
                "component": "cpu",
                "metric": "percent",
                "operator": ">=",
                "threshold": 95.0,
                "priority": "critical",
                "message": (
                    "CPU usage is critically high."
                ),
            },

            "ram_high": {
                "enabled": True,
                "component": "ram",
                "metric": "percent",
                "operator": ">=",
                "threshold": 85.0,
                "priority": "high",
                "message": (
                    "RAM usage is high."
                ),
            },

            "ram_critical": {
                "enabled": True,
                "component": "ram",
                "metric": "percent",
                "operator": ">=",
                "threshold": 95.0,
                "priority": "critical",
                "message": (
                    "RAM usage is critically high."
                ),
            },

            "disk_high": {
                "enabled": True,
                "component": "disk",
                "metric": "percent",
                "operator": ">=",
                "threshold": 85.0,
                "priority": "high",
                "message": (
                    "Disk usage is high."
                ),
            },

            "disk_critical": {
                "enabled": True,
                "component": "disk",
                "metric": "percent",
                "operator": ">=",
                "threshold": 95.0,
                "priority": "critical",
                "message": (
                    "Disk space is critically low."
                ),
            },

            "temperature_high": {
                "enabled": True,
                "component": "temperature",
                "metric": (
                    "highest_temperature"
                ),
                "operator": ">=",
                "threshold": 80.0,
                "priority": "high",
                "message": (
                    "System temperature is high."
                ),
            },

            "temperature_critical": {
                "enabled": True,
                "component": "temperature",
                "metric": (
                    "highest_temperature"
                ),
                "operator": ">=",
                "threshold": 95.0,
                "priority": "critical",
                "message": (
                    "System temperature is critical."
                ),
            },

            "battery_low": {
                "enabled": True,
                "component": "battery",
                "metric": "percent",
                "operator": "<=",
                "threshold": 20.0,
                "priority": "medium",
                "message": (
                    "Battery level is low."
                ),
            },

            "battery_critical": {
                "enabled": True,
                "component": "battery",
                "metric": "percent",
                "operator": "<=",
                "threshold": 5.0,
                "priority": "critical",
                "message": (
                    "Battery level is critical."
                ),
            },

            "network_disconnected": {
                "enabled": True,
                "component": "network",
                "metric": "connected",
                "operator": "==",
                "threshold": False,
                "priority": "medium",
                "message": (
                    "Network connection lost."
                ),
            },
        }

    # ============================================================
    # MAIN CHECK
    # ============================================================

    def check(
        self,
        metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Check metrics against all enabled rules.

        Returns newly triggered alerts.
        """

        if not self.enabled:
            return []

        metrics = metrics or {}

        triggered_alerts = []

        for rule_name, rule in (
            self.rules.items()
        ):

            if not rule.get(
                "enabled",
                True,
            ):
                continue

            alert = self._evaluate_rule(
                rule_name,
                rule,
                metrics,
            )

            if alert is None:
                continue

            if not self._can_trigger(
                rule_name
            ):
                continue

            self._record_trigger(
                rule_name
            )

            self.alert_history.append(
                alert
            )

            triggered_alerts.append(
                alert
            )

        return self._remove_duplicate_alerts(
            triggered_alerts
        )

    # ============================================================
    # RULE EVALUATION
    # ============================================================

    def _evaluate_rule(
        self,
        rule_name: str,
        rule: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any] | None:

        component = rule.get(
            "component"
        )

        metric = rule.get(
            "metric"
        )

        if not component or not metric:
            return None

        component_data = metrics.get(
            component,
            {},
        )

        if not isinstance(
            component_data,
            dict,
        ):
            return None

        if metric not in component_data:
            return None

        value = component_data.get(
            metric
        )

        threshold = rule.get(
            "threshold"
        )

        operator = rule.get(
            "operator"
        )

        if not self._compare(
            value,
            threshold,
            operator,
        ):
            return None

        priority = (
            rule.get(
                "priority",
                "medium",
            )
        )

        return {
            "id": self._create_alert_id(
                rule_name
            ),
            "rule": rule_name,
            "timestamp": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "component": component,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "operator": operator,
            "priority": priority,
            "priority_value": (
                self.PRIORITY_LEVELS.get(
                    priority,
                    2,
                )
            ),
            "message": rule.get(
                "message",
                f"Alert triggered: {rule_name}",
            ),
        }

    # ============================================================
    # COMPARISON
    # ============================================================

    @staticmethod
    def _compare(
        value: Any,
        threshold: Any,
        operator: str,
    ) -> bool:

        try:

            if operator == ">":
                return value > threshold

            if operator == ">=":
                return value >= threshold

            if operator == "<":
                return value < threshold

            if operator == "<=":
                return value <= threshold

            if operator == "==":
                return value == threshold

            if operator == "!=":
                return value != threshold

        except (
            TypeError,
            ValueError,
        ):
            return False

        return False

    # ============================================================
    # COOLDOWN
    # ============================================================

    def _can_trigger(
        self,
        rule_name: str,
    ) -> bool:

        last_time = (
            self.last_triggered.get(
                rule_name
            )
        )

        if last_time is None:
            return True

        elapsed = (
            time.time()
            - last_time
        )

        return (
            elapsed >= self.cooldown_seconds
        )

    def _record_trigger(
        self,
        rule_name: str,
    ) -> None:

        self.last_triggered[
            rule_name
        ] = time.time()

    def reset_cooldowns(
        self,
    ) -> None:
        """
        Reset all alert cooldown timers.
        """

        self.last_triggered.clear()

    def reset_rule_cooldown(
        self,
        rule_name: str,
    ) -> bool:
        """
        Reset cooldown for one rule.
        """

        if rule_name not in (
            self.last_triggered
        ):
            return False

        del self.last_triggered[
            rule_name
        ]

        return True

    # ============================================================
    # RULE MANAGEMENT
    # ============================================================

    def add_rule(
        self,
        name: str,
        *,
        component: str,
        metric: str,
        operator: str,
        threshold: Any,
        priority: str = "medium",
        message: str | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Add a custom alert rule.
        """

        if not name:
            raise ValueError(
                "Rule name cannot be empty."
            )

        if operator not in {
            ">",
            ">=",
            "<",
            "<=",
            "==",
            "!=",
        }:
            raise ValueError(
                f"Unsupported operator: {operator}"
            )

        if priority not in (
            self.PRIORITY_LEVELS
        ):
            raise ValueError(
                f"Invalid priority: {priority}"
            )

        self.rules[name] = {
            "enabled": bool(enabled),
            "component": component,
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "priority": priority,
            "message": (
                message
                or f"Alert triggered: {name}"
            ),
        }

    def remove_rule(
        self,
        name: str,
    ) -> bool:
        """
        Remove an alert rule.
        """

        if name not in self.rules:
            return False

        del self.rules[name]

        self.last_triggered.pop(
            name,
            None,
        )

        return True

    def enable_rule(
        self,
        name: str,
    ) -> bool:

        rule = self.rules.get(
            name
        )

        if rule is None:
            return False

        rule["enabled"] = True

        return True

    def disable_rule(
        self,
        name: str,
    ) -> bool:

        rule = self.rules.get(
            name
        )

        if rule is None:
            return False

        rule["enabled"] = False

        return True

    def update_rule(
        self,
        name: str,
        **updates: Any,
    ) -> bool:
        """
        Update properties of an existing rule.
        """

        rule = self.rules.get(
            name
        )

        if rule is None:
            return False

        allowed_fields = {
            "enabled",
            "component",
            "metric",
            "operator",
            "threshold",
            "priority",
            "message",
        }

        for key, value in (
            updates.items()
        ):

            if key not in allowed_fields:
                continue

            if (
                key == "operator"
                and value
                not in {
                    ">",
                    ">=",
                    "<",
                    "<=",
                    "==",
                    "!=",
                }
            ):
                continue

            if (
                key == "priority"
                and value
                not in self.PRIORITY_LEVELS
            ):
                continue

            rule[key] = value

        return True

    # ============================================================
    # ALERT HISTORY
    # ============================================================

    def get_history(
        self,
        *,
        limit: int | None = None,
        priority: str | None = None,
        component: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get alert history with optional filters.
        """

        alerts = list(
            self.alert_history
        )

        if priority:

            alerts = [
                alert
                for alert in alerts
                if alert.get(
                    "priority"
                )
                == priority
            ]

        if component:

            alerts = [
                alert
                for alert in alerts
                if alert.get(
                    "component"
                )
                == component
            ]

        if limit is not None:

            limit = max(
                0,
                int(limit),
            )

            if limit == 0:
                return []

            alerts = alerts[
                -limit:
            ]

        return alerts

    def get_recent_alerts(
        self,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        return self.get_history(
            limit=limit
        )

    def get_critical_alerts(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:

        return self.get_history(
            limit=limit,
            priority="critical",
        )

    def clear_history(
        self,
    ) -> None:

        self.alert_history.clear()

    # ============================================================
    # MANUAL ALERT
    # ============================================================

    def create_alert(
        self,
        *,
        component: str,
        message: str,
        priority: str = "medium",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create and store a manual alert.
        """

        if priority not in (
            self.PRIORITY_LEVELS
        ):
            priority = "medium"

        alert = {
            "id": self._create_alert_id(
                "manual"
            ),
            "rule": "manual",
            "timestamp": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "component": component,
            "metric": None,
            "value": None,
            "threshold": None,
            "operator": None,
            "priority": priority,
            "priority_value": (
                self.PRIORITY_LEVELS[
                    priority
                ]
            ),
            "message": message,
            "data": data or {},
        }

        self.alert_history.append(
            alert
        )

        return alert

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_summary(
        self,
    ) -> dict[str, Any]:
        """
        Return alert system summary.
        """

        alerts = list(
            self.alert_history
        )

        priority_counts = {
            priority: 0
            for priority in (
                self.PRIORITY_LEVELS
            )
        }

        component_counts: dict[
            str,
            int
        ] = {}

        for alert in alerts:

            priority = alert.get(
                "priority"
            )

            if priority in priority_counts:

                priority_counts[
                    priority
                ] += 1

            component = str(
                alert.get(
                    "component",
                    "unknown",
                )
            )

            component_counts[
                component
            ] = (
                component_counts.get(
                    component,
                    0,
                )
                + 1
            )

        return {
            "enabled": self.enabled,
            "total_rules": len(
                self.rules
            ),
            "enabled_rules": sum(
                1
                for rule in (
                    self.rules.values()
                )
                if rule.get(
                    "enabled",
                    True,
                )
            ),
            "history_count": len(
                alerts
            ),
            "priority_counts": (
                priority_counts
            ),
            "component_counts": (
                component_counts
            ),
            "cooldown_seconds": (
                self.cooldown_seconds
            ),
        }

    # ============================================================
    # ALERT SYSTEM CONTROL
    # ============================================================

    def enable(
        self,
    ) -> None:

        self.enabled = True

    def disable(
        self,
    ) -> None:

        self.enabled = False

    def reset(
        self,
    ) -> None:
        """
        Reset alert manager state.

        Custom rules are removed and default
        rules are restored.
        """

        self.alert_history.clear()

        self.last_triggered.clear()

        self.rules = (
            self._create_default_rules()
        )

        self.enabled = True

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _create_alert_id(
        prefix: str,
    ) -> str:

        timestamp = int(
            time.time() * 1000
        )

        return (
            f"{prefix}_{timestamp}"
        )

    @staticmethod
    def _remove_duplicate_alerts(
        alerts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        seen = set()

        unique = []

        for alert in alerts:

            key = (
                alert.get("rule"),
                alert.get("component"),
                alert.get("metric"),
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique.append(
                alert
            )

        unique.sort(
            key=lambda alert: (
                alert.get(
                    "priority_value",
                    0,
                )
            ),
            reverse=True,
        )

        return unique


__all__ = [
    "SystemAlerts",
    "AlertManager",
]


AlertManager = SystemAlerts


