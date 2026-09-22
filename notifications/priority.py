"""
RENIX Notifications - Priority
===============================

Defines notification priorities and utilities used by
the RENIX notification system.

Priority order:

    LOW < NORMAL < HIGH < CRITICAL
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class NotificationPriority(IntEnum):
    """Priority levels for RENIX notifications."""

    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40

    @property
    def label(self) -> str:
        """Return a human-readable priority label."""

        return self.name.capitalize()

    @property
    def urgency(self) -> str:
        """Return a simplified urgency classification."""

        if self >= NotificationPriority.CRITICAL:
            return "critical"

        if self >= NotificationPriority.HIGH:
            return "high"

        if self >= NotificationPriority.NORMAL:
            return "normal"

        return "low"

    @classmethod
    def from_value(
        cls,
        value: Any,
        default: "NotificationPriority" = None,
    ) -> "NotificationPriority":
        """
        Convert a string, integer, or existing priority
        into NotificationPriority.
        """

        if isinstance(value, cls):
            return value

        if value is None:
            return (
                default
                if default is not None
                else cls.NORMAL
            )

        if isinstance(value, str):
            normalized = (
                value.strip()
                .upper()
                .replace("-", "_")
                .replace(" ", "_")
            )

            aliases = {
                "INFO": cls.NORMAL,
                "INFORMATION": cls.NORMAL,
                "NOTICE": cls.NORMAL,
                "WARN": cls.HIGH,
                "WARNING": cls.HIGH,
                "URGENT": cls.CRITICAL,
                "EMERGENCY": cls.CRITICAL,
                "ERROR": cls.CRITICAL,
                "FATAL": cls.CRITICAL,
                "MINOR": cls.LOW,
            }

            if normalized in aliases:
                return aliases[normalized]

            try:
                return cls[normalized]
            except KeyError:
                pass

            try:
                value = int(value)
            except ValueError:
                pass

        if isinstance(value, int):
            members = list(cls)

            # Exact priority value.
            for priority in members:
                if value == priority.value:
                    return priority

            # Numeric ranges allow flexible external values.
            if value >= cls.CRITICAL.value:
                return cls.CRITICAL

            if value >= cls.HIGH.value:
                return cls.HIGH

            if value >= cls.NORMAL.value:
                return cls.NORMAL

            return cls.LOW

        if default is not None:
            return default

        raise ValueError(
            f"Invalid notification priority: {value!r}"
        )


# ============================================================
# CONVERSION HELPERS
# ============================================================


def get_priority_value(
    priority: NotificationPriority | str | int,
) -> int:
    """Return the numeric value of a notification priority."""

    return NotificationPriority.from_value(
        priority
    ).value


def get_priority_name(
    priority: NotificationPriority | str | int,
) -> str:
    """Return the enum name of a notification priority."""

    return NotificationPriority.from_value(
        priority
    ).name


def get_priority_label(
    priority: NotificationPriority | str | int,
) -> str:
    """Return a human-readable priority label."""

    return NotificationPriority.from_value(
        priority
    ).label


def get_priority_urgency(
    priority: NotificationPriority | str | int,
) -> str:
    """Return the urgency classification."""

    return NotificationPriority.from_value(
        priority
    ).urgency


# ============================================================
# COMPARISON HELPERS
# ============================================================


def is_higher_priority(
    first: NotificationPriority | str | int,
    second: NotificationPriority | str | int,
) -> bool:
    """Return True if first has higher priority than second."""

    return (
        get_priority_value(first)
        > get_priority_value(second)
    )


def is_lower_priority(
    first: NotificationPriority | str | int,
    second: NotificationPriority | str | int,
) -> bool:
    """Return True if first has lower priority than second."""

    return (
        get_priority_value(first)
        < get_priority_value(second)
    )


def is_at_least(
    priority: NotificationPriority | str | int,
    minimum: NotificationPriority | str | int,
) -> bool:
    """Check whether priority meets a minimum level."""

    return (
        get_priority_value(priority)
        >= get_priority_value(minimum)
    )


def is_critical(
    priority: NotificationPriority | str | int,
) -> bool:
    """Return True if priority is critical."""

    return (
        NotificationPriority.from_value(priority)
        == NotificationPriority.CRITICAL
    )


def is_high_or_above(
    priority: NotificationPriority | str | int,
) -> bool:
    """Return True if priority is HIGH or CRITICAL."""

    return is_at_least(
        priority,
        NotificationPriority.HIGH,
    )


# ============================================================
# SORTING
# ============================================================


def sort_priorities(
    priorities: list[
        NotificationPriority | str | int
    ],
    *,
    descending: bool = True,
) -> list[NotificationPriority]:
    """
    Sort notification priorities by urgency.
    """

    converted = [
        NotificationPriority.from_value(
            priority
        )
        for priority in priorities
    ]

    return sorted(
        converted,
        key=lambda item: item.value,
        reverse=descending,
    )


# ============================================================
# SERIALIZATION
# ============================================================


def priority_to_dict(
    priority: NotificationPriority | str | int,
) -> dict[str, Any]:
    """Convert a priority into a serializable dictionary."""

    normalized = NotificationPriority.from_value(
        priority
    )

    return {
        "name": normalized.name,
        "value": normalized.value,
        "label": normalized.label,
        "urgency": normalized.urgency,
    }


def priority_from_dict(
    data: dict[str, Any],
) -> NotificationPriority:
    """Create a priority from serialized data."""

    if not isinstance(data, dict):
        raise TypeError(
            "Priority data must be a dictionary."
        )

    if "name" in data:
        return NotificationPriority.from_value(
            data["name"]
        )

    if "value" in data:
        return NotificationPriority.from_value(
            data["value"]
        )

    raise ValueError(
        "Priority dictionary must contain "
        "'name' or 'value'."
    )


# ============================================================
# DISPLAY
# ============================================================


def priority_description(
    priority: NotificationPriority | str | int,
) -> str:
    """Return a human-readable description."""

    normalized = NotificationPriority.from_value(
        priority
    )

    descriptions = {
        NotificationPriority.LOW:
            "Low-importance notification.",
        NotificationPriority.NORMAL:
            "Standard notification requiring normal attention.",
        NotificationPriority.HIGH:
            "Important notification requiring prompt attention.",
        NotificationPriority.CRITICAL:
            "Critical notification requiring immediate attention.",
    }

    return descriptions[normalized]


def all_priorities() -> list[NotificationPriority]:
    """Return all available notification priorities."""

    return list(NotificationPriority)


__all__ = [
    "NotificationPriority",
    "get_priority_value",
    "get_priority_name",
    "get_priority_label",
    "get_priority_urgency",
    "is_higher_priority",
    "is_lower_priority",
    "is_at_least",
    "is_critical",
    "is_high_or_above",
    "sort_priorities",
    "priority_to_dict",
    "priority_from_dict",
    "priority_description",
    "all_priorities",
]


