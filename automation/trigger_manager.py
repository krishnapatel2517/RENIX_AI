"""
RENIX Trigger Manager
=====================

Manages automation triggers.

Supported trigger categories:

    - event
    - time
    - schedule
    - condition
    - webhook
    - manual
    - startup
    - shutdown
    - custom

The TriggerManager does not execute workflows itself.
It notifies registered callbacks when a trigger fires.
"""

from __future__ import annotations

import logging
import threading
import uuid

from datetime import datetime, time as dt_time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TriggerManager:
    """Central manager for RENIX automation triggers."""

    def __init__(
        self,
        *,
        event_bus: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:

        self.event_bus = event_bus

        self.context: dict[str, Any] = dict(
            context or {}
        )

        self._triggers: dict[
            str,
            dict[str, Any],
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def start(self) -> None:
        """Start trigger management."""

        with self._lock:

            if self._running:
                return

            self._running = True

        self._emit(
            "trigger_manager.started",
            {},
        )

        logger.info(
            "RENIX Trigger Manager started."
        )

    def stop(self) -> None:
        """Stop trigger management."""

        with self._lock:

            if not self._running:
                return

            self._running = False

        self._emit(
            "trigger_manager.stopped",
            {},
        )

        logger.info(
            "RENIX Trigger Manager stopped."
        )

    def is_running(self) -> bool:
        return self._running

    # ========================================================
    # REGISTRATION
    # ========================================================

    def register(
        self,
        trigger: dict[str, Any],
    ) -> str:
        """
        Register a trigger.

        Example:

        {
            "name": "morning_trigger",
            "type": "time",
            "time": "07:00",
            "workflow_id": "morning_routine"
        }
        """

        normalized = self.validate(
            trigger
        )

        trigger_id = normalized["id"]

        with self._lock:
            self._triggers[
                trigger_id
            ] = normalized

        self._emit(
            "trigger.registered",
            normalized,
        )

        return trigger_id

    def unregister(
        self,
        trigger_id: str,
    ) -> bool:
        """Remove a trigger."""

        with self._lock:

            trigger = self._triggers.pop(
                trigger_id,
                None,
            )

        if trigger is None:
            return False

        self._emit(
            "trigger.unregistered",
            trigger,
        )

        return True

    def get(
        self,
        trigger_id: str,
    ) -> dict[str, Any] | None:
        """Return trigger by ID."""

        with self._lock:

            trigger = self._triggers.get(
                trigger_id
            )

            if trigger is None:
                return None

            return dict(
                trigger
            )

    def list(
        self,
        *,
        enabled_only: bool = False,
        trigger_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List registered triggers."""

        with self._lock:
            triggers = list(
                self._triggers.values()
            )

        if enabled_only:
            triggers = [
                trigger
                for trigger in triggers
                if trigger.get(
                    "enabled",
                    True,
                )
            ]

        if trigger_type is not None:
            trigger_type = (
                trigger_type
                .strip()
                .lower()
            )

            triggers = [
                trigger
                for trigger in triggers
                if trigger.get(
                    "type"
                ) == trigger_type
            ]

        return [
            dict(trigger)
            for trigger in triggers
        ]

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        trigger: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and normalize a trigger."""

        if not isinstance(
            trigger,
            dict,
        ):
            raise TypeError(
                "trigger must be a dictionary."
            )

        normalized = dict(
            trigger
        )

        trigger_id = str(
            normalized.get(
                "id"
            ) or uuid.uuid4()
        ).strip()

        name = str(
            normalized.get(
                "name",
                "",
            )
        ).strip()

        trigger_type = str(
            normalized.get(
                "type",
                "",
            )
        ).strip().lower()

        if not name:
            raise ValueError(
                "Trigger name is required."
            )

        if not trigger_type:
            raise ValueError(
                "Trigger type is required."
            )

        supported = {
            "event",
            "time",
            "schedule",
            "condition",
            "webhook",
            "manual",
            "startup",
            "shutdown",
            "custom",
        }

        if trigger_type not in supported:
            raise ValueError(
                f"Unsupported trigger type: {trigger_type}"
            )

        normalized["id"] = trigger_id
        normalized["name"] = name
        normalized["type"] = trigger_type

        normalized.setdefault(
            "enabled",
            True,
        )

        normalized.setdefault(
            "workflow_id",
            None,
        )

        normalized.setdefault(
            "metadata",
            {},
        )

        if not isinstance(
            normalized["metadata"],
            dict,
        ):
            raise TypeError(
                "trigger.metadata must be a dictionary."
            )

        if trigger_type == "time":
            self._validate_time_trigger(
                normalized
            )

        if trigger_type == "event":
            self._validate_event_trigger(
                normalized
            )

        if trigger_type == "webhook":
            self._validate_webhook_trigger(
                normalized
            )

        return normalized

    def _validate_time_trigger(
        self,
        trigger: dict[str, Any],
    ) -> None:

        value = trigger.get(
            "time"
        )

        if value is None:
            raise ValueError(
                "Time trigger requires 'time'."
            )

        self._parse_time(
            value
        )

    def _validate_event_trigger(
        self,
        trigger: dict[str, Any],
    ) -> None:

        event = str(
            trigger.get(
                "event",
                ""
            )
        ).strip()

        if not event:
            raise ValueError(
                "Event trigger requires 'event'."
            )

    def _validate_webhook_trigger(
        self,
        trigger: dict[str, Any],
    ) -> None:

        path = str(
            trigger.get(
                "path",
                ""
            )
        ).strip()

        if not path:
            raise ValueError(
                "Webhook trigger requires 'path'."
            )

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(
        self,
        trigger_id: str,
    ) -> bool:
        return self._set_enabled(
            trigger_id,
            True,
        )

    def disable(
        self,
        trigger_id: str,
    ) -> bool:
        return self._set_enabled(
            trigger_id,
            False,
        )

    def _set_enabled(
        self,
        trigger_id: str,
        enabled: bool,
    ) -> bool:

        with self._lock:

            trigger = self._triggers.get(
                trigger_id
            )

            if trigger is None:
                return False

            trigger["enabled"] = enabled

        self._emit(
            "trigger.updated",
            {
                "id": trigger_id,
                "enabled": enabled,
            },
        )

        return True

    # ========================================================
    # FIRING
    # ========================================================

    def fire(
        self,
        trigger_id: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Fire a specific trigger.

        Returns callback results.
        """

        trigger = self.get(
            trigger_id
        )

        if trigger is None:
            raise KeyError(
                f"Trigger not found: {trigger_id}"
            )

        if not trigger.get(
            "enabled",
            True,
        ):
            return []

        event_payload = {
            "trigger_id": trigger_id,
            "trigger": trigger,
            "payload": dict(
                payload or {}
            ),
            "timestamp": self._timestamp(),
        }

        self._emit(
            "trigger.fired",
            event_payload,
        )

        results: list[Any] = []

        callbacks = list(
            self._callbacks.get(
                trigger_id,
                [],
            )
        )

        callbacks += list(
            self._callbacks.get(
                "*",
                [],
            )
        )

        for callback in callbacks:

            try:

                result = callback(
                    event_payload
                )

                results.append(
                    result
                )

            except Exception:

                logger.exception(
                    "Trigger callback failed: %s",
                    trigger_id,
                )

        return results

    def fire_event(
        self,
        event: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Fire all triggers listening for an event.
        """

        event = event.strip()

        if not event:
            raise ValueError(
                "event cannot be empty."
            )

        results: list[Any] = []

        for trigger in self.list(
            enabled_only=True,
            trigger_type="event",
        ):

            if trigger.get(
                "event"
            ) != event:
                continue

            results.extend(
                self.fire(
                    trigger["id"],
                    payload=payload,
                )
            )

        return results

    def fire_manual(
        self,
        trigger_id: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Manually fire a trigger."""

        trigger = self.get(
            trigger_id
        )

        if trigger is None:
            raise KeyError(
                f"Trigger not found: {trigger_id}"
            )

        if trigger.get(
            "type"
        ) != "manual":
            raise ValueError(
                "Specified trigger is not a manual trigger."
            )

        return self.fire(
            trigger_id,
            payload=payload,
        )

    def fire_webhook(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Fire triggers matching a webhook path."""

        path = path.strip()

        if not path:
            raise ValueError(
                "Webhook path cannot be empty."
            )

        results: list[Any] = []

        for trigger in self.list(
            enabled_only=True,
            trigger_type="webhook",
        ):

            if trigger.get(
                "path"
            ) != path:
                continue

            results.extend(
                self.fire(
                    trigger["id"],
                    payload=payload,
                )
            )

        return results

    # ========================================================
    # TIME TRIGGERS
    # ========================================================

    def check_time_triggers(
        self,
        current_time: datetime | None = None,
    ) -> list[Any]:
        """
        Check registered time triggers.

        Intended to be called by a scheduler.
        """

        now = current_time or datetime.now()

        results: list[Any] = []

        for trigger in self.list(
            enabled_only=True,
            trigger_type="time",
        ):

            target = self._parse_time(
                trigger["time"]
            )

            if (
                now.hour == target.hour
                and now.minute == target.minute
            ):
                results.extend(
                    self.fire(
                        trigger["id"],
                        payload={
                            "current_time": now.isoformat()
                        },
                    )
                )

        return results

    # ========================================================
    # CALLBACKS
    # ========================================================

    def subscribe(
        self,
        trigger_id: str,
        callback: Callable[..., Any],
    ) -> None:
        """
        Subscribe to a trigger.

        Use "*" to receive every fired trigger.
        """

        if not trigger_id.strip():
            raise ValueError(
                "trigger_id cannot be empty."
            )

        if not callable(
            callback
        ):
            raise TypeError(
                "callback must be callable."
            )

        self._callbacks.setdefault(
            trigger_id,
            [],
        ).append(
            callback
        )

    def unsubscribe(
        self,
        trigger_id: str,
        callback: Callable[..., Any],
    ) -> bool:

        callbacks = self._callbacks.get(
            trigger_id,
            [],
        )

        if callback not in callbacks:
            return False

        callbacks.remove(
            callback
        )

        return True

    # ========================================================
    # CONTEXT
    # ========================================================

    def set_context(
        self,
        key: str,
        value: Any,
    ) -> None:

        if not key.strip():
            raise ValueError(
                "Context key cannot be empty."
            )

        self.context[key] = value

    def update_context(
        self,
        values: dict[str, Any],
    ) -> None:

        if not isinstance(
            values,
            dict,
        ):
            raise TypeError(
                "values must be a dictionary."
            )

        self.context.update(
            values
        )

    def get_context(
        self,
    ) -> dict[str, Any]:
        return dict(
            self.context
        )

    # ========================================================
    # EVENTS
    # ========================================================

    def _emit(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> None:

        if self.event_bus is not None:

            publish = getattr(
                self.event_bus,
                "publish",
                None,
            )

            if callable(
                publish
            ):

                try:
                    publish(
                        event,
                        payload,
                    )
                except Exception:
                    logger.exception(
                        "Failed to publish trigger event."
                    )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _parse_time(
        value: Any,
    ) -> dt_time:

        if isinstance(
            value,
            dt_time,
        ):
            return value

        value = str(
            value
        ).strip()

        formats = (
            "%H:%M",
            "%H:%M:%S",
            "%I:%M %p",
            "%I:%M:%S %p",
        )

        for fmt in formats:

            try:
                return datetime.strptime(
                    value,
                    fmt,
                ).time()

            except ValueError:
                continue

        raise ValueError(
            f"Invalid time format: {value}"
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(
            timespec="seconds"
        )


__all__ = [
    "TriggerManager",
]


