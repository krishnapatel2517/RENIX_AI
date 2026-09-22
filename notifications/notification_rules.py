"""
RENIX Notifications - Notification Rules
=========================================

Rule engine for deciding whether RENIX notifications should
be allowed, suppressed, modified, or routed.

Examples:
    - Suppress low-priority notifications during study mode.
    - Allow critical notifications during Do Not Disturb.
    - Route cricket notifications to a specific category.
    - Limit repeated notifications.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .notification_manager import Notification
from .priority import (
    NotificationPriority,
    get_priority_value,
)


RuleCondition = Callable[[Notification], bool]
RuleAction = Callable[[Notification], Any]


@dataclass
class NotificationRule:
    """Represents one notification-processing rule."""

    name: str
    condition: RuleCondition

    action: RuleAction | None = None

    enabled: bool = True
    priority: int = 0
    stop_processing: bool = False

    description: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def matches(
        self,
        notification: Notification,
    ) -> bool:
        """Return True when this rule matches."""

        if not self.enabled:
            return False

        try:
            return bool(
                self.condition(notification)
            )
        except Exception:
            return False

    def execute(
        self,
        notification: Notification,
    ) -> Any:
        """Execute this rule's action."""

        if self.action is None:
            return None

        return self.action(notification)


@dataclass
class RuleResult:
    """Result of processing one notification."""

    allowed: bool = True

    notification: Notification | None = None

    matched_rules: list[str] = field(
        default_factory=list
    )

    executed_rules: list[str] = field(
        default_factory=list
    )

    blocked_by: str | None = None

    errors: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""

        return {
            "allowed": self.allowed,
            "notification": (
                self.notification.to_dict()
                if self.notification
                else None
            ),
            "matched_rules": list(
                self.matched_rules
            ),
            "executed_rules": list(
                self.executed_rules
            ),
            "blocked_by": self.blocked_by,
            "errors": list(self.errors),
        }


class NotificationRuleEngine:
    """
    Processes RENIX notifications through configurable rules.

    Rules are evaluated from highest rule priority to lowest.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled

        self._rules: dict[
            str,
            NotificationRule,
        ] = {}

        self._last_results: list[
            RuleResult
        ] = []

        self._max_results = 100

    # ========================================================
    # RULE MANAGEMENT
    # ========================================================

    def add_rule(
        self,
        rule: NotificationRule,
    ) -> None:
        """Add or replace a notification rule."""

        if not isinstance(
            rule,
            NotificationRule,
        ):
            raise TypeError(
                "rule must be a NotificationRule."
            )

        self._rules[rule.name] = rule

    def remove_rule(
        self,
        name: str,
    ) -> bool:
        """Remove a rule."""

        return (
            self._rules.pop(
                name,
                None,
            )
            is not None
        )

    def get_rule(
        self,
        name: str,
    ) -> NotificationRule | None:
        """Return a rule by name."""

        return self._rules.get(name)

    def enable_rule(
        self,
        name: str,
    ) -> bool:
        """Enable a rule."""

        rule = self.get_rule(name)

        if rule is None:
            return False

        rule.enabled = True
        return True

    def disable_rule(
        self,
        name: str,
    ) -> bool:
        """Disable a rule."""

        rule = self.get_rule(name)

        if rule is None:
            return False

        rule.enabled = False
        return True

    def clear_rules(self) -> None:
        """Remove all rules."""

        self._rules.clear()

    def rules(
        self,
        *,
        enabled_only: bool = False,
    ) -> list[NotificationRule]:
        """Return registered rules sorted by priority."""

        rules = list(
            self._rules.values()
        )

        if enabled_only:
            rules = [
                rule
                for rule in rules
                if rule.enabled
            ]

        return sorted(
            rules,
            key=lambda rule: rule.priority,
            reverse=True,
        )

    # ========================================================
    # PROCESSING
    # ========================================================

    def process(
        self,
        notification: Notification,
    ) -> RuleResult:
        """
        Process a notification through all matching rules.
        """

        if not isinstance(
            notification,
            Notification,
        ):
            raise TypeError(
                "notification must be a Notification."
            )

        result = RuleResult(
            notification=notification
        )

        if not self.enabled:
            self._store_result(result)
            return result

        for rule in self.rules(
            enabled_only=True
        ):
            if not rule.matches(
                notification
            ):
                continue

            result.matched_rules.append(
                rule.name
            )

            try:
                action_result = (
                    rule.execute(
                        notification
                    )
                )

                result.executed_rules.append(
                    rule.name
                )

                self._apply_action_result(
                    result,
                    action_result,
                    rule,
                )

            except Exception as exc:
                result.errors.append(
                    f"{rule.name}: {exc}"
                )

            if rule.stop_processing:
                break

        self._store_result(result)

        return result

    def process_many(
        self,
        notifications: Iterable[Notification],
    ) -> list[RuleResult]:
        """Process multiple notifications."""

        return [
            self.process(notification)
            for notification in notifications
        ]

    # ========================================================
    # ACTION RESULTS
    # ========================================================

    @staticmethod
    def _apply_action_result(
        result: RuleResult,
        action_result: Any,
        rule: NotificationRule,
    ) -> None:
        """
        Interpret the value returned by a rule action.

        Supported action results:

            None / True
                Continue.

            False
                Block notification.

            dict
                Supports:
                allowed
                title
                message
                category
                source
                priority
                data
        """

        if action_result is None:
            return

        if action_result is True:
            return

        if action_result is False:
            result.allowed = False
            result.blocked_by = rule.name
            return

        if not isinstance(
            action_result,
            Mapping,
        ):
            return

        if "allowed" in action_result:
            allowed = bool(
                action_result["allowed"]
            )

            if not allowed:
                result.allowed = False
                result.blocked_by = rule.name

        notification = result.notification

        if notification is None:
            return

        if "title" in action_result:
            notification.title = str(
                action_result["title"]
            )

        if "message" in action_result:
            notification.message = str(
                action_result["message"]
            )

        if "category" in action_result:
            notification.category = str(
                action_result["category"]
            )

        if "source" in action_result:
            notification.source = str(
                action_result["source"]
            )

        if "priority" in action_result:
            notification.priority = (
                NotificationPriority.from_value(
                    action_result["priority"]
                )
            )

        if "data" in action_result:
            extra_data = action_result["data"]

            if isinstance(
                extra_data,
                Mapping,
            ):
                notification.data.update(
                    extra_data
                )

    # ========================================================
    # COMMON RULE BUILDERS
    # ========================================================

    def add_minimum_priority_rule(
        self,
        name: str,
        minimum: NotificationPriority | str | int,
        *,
        priority: int = 0,
    ) -> NotificationRule:
        """
        Block notifications below a minimum priority.
        """

        minimum_value = get_priority_value(
            minimum
        )

        def condition(
            notification: Notification,
        ) -> bool:
            return (
                get_priority_value(
                    notification.priority
                )
                < minimum_value
            )

        def action(
            notification: Notification,
        ) -> bool:
            return False

        rule = NotificationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            stop_processing=True,
            description=(
                "Blocks notifications below "
                f"{minimum_value}."
            ),
        )

        self.add_rule(rule)

        return rule

    def add_category_block_rule(
        self,
        name: str,
        categories: Iterable[str],
        *,
        priority: int = 0,
    ) -> NotificationRule:
        """Block notifications belonging to categories."""

        category_set = {
            str(category).lower()
            for category in categories
        }

        def condition(
            notification: Notification,
        ) -> bool:
            return (
                notification.category.lower()
                in category_set
            )

        def action(
            notification: Notification,
        ) -> bool:
            return False

        rule = NotificationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            stop_processing=True,
            description=(
                "Blocks selected notification categories."
            ),
            metadata={
                "categories": list(
                    category_set
                )
            },
        )

        self.add_rule(rule)

        return rule

    def add_source_rule(
        self,
        name: str,
        sources: Iterable[str],
        *,
        priority: int = 0,
    ) -> NotificationRule:
        """Create a rule matching notification sources."""

        source_set = {
            str(source).lower()
            for source in sources
        }

        def condition(
            notification: Notification,
        ) -> bool:
            return (
                notification.source.lower()
                in source_set
            )

        rule = NotificationRule(
            name=name,
            condition=condition,
            priority=priority,
            description=(
                "Matches selected notification sources."
            ),
            metadata={
                "sources": list(
                    source_set
                )
            },
        )

        self.add_rule(rule)

        return rule

    def add_keyword_rule(
        self,
        name: str,
        keywords: Iterable[str],
        *,
        priority: int = 0,
        case_sensitive: bool = False,
    ) -> NotificationRule:
        """Create a rule matching title/message keywords."""

        if case_sensitive:
            keyword_set = {
                str(keyword)
                for keyword in keywords
            }
        else:
            keyword_set = {
                str(keyword).lower()
                for keyword in keywords
            }

        def condition(
            notification: Notification,
        ) -> bool:
            text = (
                notification.title
                + " "
                + notification.message
            )

            if not case_sensitive:
                text = text.lower()

            return any(
                keyword in text
                for keyword in keyword_set
            )

        rule = NotificationRule(
            name=name,
            condition=condition,
            priority=priority,
            description=(
                "Matches keywords in notification text."
            ),
            metadata={
                "keywords": list(
                    keyword_set
                )
            },
        )

        self.add_rule(rule)

        return rule

    # ========================================================
    # BUILT-IN MODES
    # ========================================================

    def add_do_not_disturb_rule(
        self,
        name: str = "do_not_disturb",
        *,
        allow_critical: bool = True,
        priority: int = 100,
    ) -> NotificationRule:
        """
        Add a Do Not Disturb rule.

        By default only CRITICAL notifications pass.
        """

        def condition(
            notification: Notification,
        ) -> bool:
            if allow_critical:
                return (
                    notification.priority
                    != NotificationPriority.CRITICAL
                )

            return True

        def action(
            notification: Notification,
        ) -> bool:
            return False

        rule = NotificationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            stop_processing=True,
            description=(
                "Suppresses notifications while "
                "Do Not Disturb is active."
            ),
        )

        self.add_rule(rule)

        return rule

    def add_study_mode_rule(
        self,
        name: str = "study_mode",
        *,
        minimum_priority: NotificationPriority = (
            NotificationPriority.HIGH
        ),
        priority: int = 110,
    ) -> NotificationRule:
        """
        Add a study-mode rule.

        Low and normal notifications are suppressed.
        """

        minimum_value = get_priority_value(
            minimum_priority
        )

        def condition(
            notification: Notification,
        ) -> bool:
            return (
                get_priority_value(
                    notification.priority
                )
                < minimum_value
            )

        def action(
            notification: Notification,
        ) -> bool:
            return False

        rule = NotificationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            stop_processing=True,
            description=(
                "Suppresses low-priority notifications "
                "during study mode."
            ),
        )

        self.add_rule(rule)

        return rule

    # ========================================================
    # TRANSFORMATION RULES
    # ========================================================

    def add_transform_rule(
        self,
        name: str,
        condition: RuleCondition,
        transform: RuleAction,
        *,
        priority: int = 0,
    ) -> NotificationRule:
        """
        Add a rule that modifies notifications.

        The transform may return:

            None
            True
            False
            dict of notification fields
        """

        rule = NotificationRule(
            name=name,
            condition=condition,
            action=transform,
            priority=priority,
            description=(
                "Transforms matching notifications."
            ),
        )

        self.add_rule(rule)

        return rule

    # ========================================================
    # RATE LIMITING
    # ========================================================

    def add_rate_limit_rule(
        self,
        name: str,
        *,
        max_count: int,
        window_seconds: float,
        key: Callable[
            [Notification],
            str,
        ] | None = None,
        priority: int = 90,
    ) -> NotificationRule:
        """
        Add a simple in-memory rate-limiting rule.

        Example:
            maximum 3 notifications of the same category
            within 30 seconds.
        """

        if max_count < 1:
            raise ValueError(
                "max_count must be >= 1."
            )

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds must be > 0."
            )

        timestamps: dict[
            str,
            list[float],
        ] = {}

        make_key = key or (
            lambda notification:
                notification.category
        )

        def condition(
            notification: Notification,
        ) -> bool:
            now = time.monotonic()
            notification_key = str(
                make_key(notification)
            )

            history = timestamps.setdefault(
                notification_key,
                [],
            )

            cutoff = (
                now - window_seconds
            )

            history[:] = [
                timestamp
                for timestamp in history
                if timestamp >= cutoff
            ]

            history.append(now)

            return len(history) > max_count

        def action(
            notification: Notification,
        ) -> bool:
            return False

        rule = NotificationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            stop_processing=True,
            description=(
                "Suppresses excessive repeated "
                "notifications."
            ),
            metadata={
                "max_count": max_count,
                "window_seconds": window_seconds,
            },
        )

        self.add_rule(rule)

        return rule

    # ========================================================
    # RESULTS
    # ========================================================

    def _store_result(
        self,
        result: RuleResult,
    ) -> None:
        self._last_results.append(
            result
        )

        if len(
            self._last_results
        ) > self._max_results:
            self._last_results.pop(
                0
            )

    def last_results(
        self,
        limit: int = 20,
    ) -> list[RuleResult]:
        """Return recent rule-processing results."""

        if limit < 0:
            raise ValueError(
                "limit must be >= 0."
            )

        return self._last_results[
            -limit:
        ]

    # ========================================================
    # CONTROL
    # ========================================================

    def enable(self) -> None:
        """Enable the rule engine."""

        self.enabled = True

    def disable(self) -> None:
        """Disable the rule engine."""

        self.enabled = False

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict[str, Any]:
        """Return rule engine status."""

        return {
            "enabled": self.enabled,
            "rule_count": len(
                self._rules
            ),
            "enabled_rules": len(
                self.rules(
                    enabled_only=True
                )
            ),
            "stored_results": len(
                self._last_results
            ),
            "rules": [
                {
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "priority": rule.priority,
                    "stop_processing": (
                        rule.stop_processing
                    ),
                }
                for rule in self.rules()
            ],
        }

    def reset(self) -> None:
        """Reset rules and processing history."""

        self._rules.clear()
        self._last_results.clear()

    def __len__(self) -> int:
        """Return the number of registered rules."""

        return len(self._rules)

    def __repr__(self) -> str:
        return (
            "NotificationRuleEngine("
            f"enabled={self.enabled}, "
            f"rules={len(self._rules)})"
        )


__all__ = [
    "NotificationRule",
    "RuleResult",
    "NotificationRuleEngine",
]


