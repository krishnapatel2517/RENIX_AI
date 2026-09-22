"""
RENIX Notifications - Notification Manager
==========================================

Central notification manager for RENIX.

Responsibilities:
- Create notifications
- Queue notifications
- Track notification history
- Handle priorities
- Dismiss/read notifications
- Dispatch notifications to registered handlers
- Prevent notification spam
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Iterable, Mapping

from .priority import (
    NotificationPriority,
    get_priority_value,
)


NotificationHandler = Callable[
    ["Notification"],
    Any,
]


@dataclass
class Notification:
    """Represents one RENIX notification."""

    title: str
    message: str

    priority: NotificationPriority = (
        NotificationPriority.NORMAL
    )

    category: str = "general"

    notification_id: str = field(
        default_factory=lambda: str(
            uuid.uuid4()
        )
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    read: bool = False
    dismissed: bool = False

    source: str = "RENIX"

    data: dict[str, Any] = field(
        default_factory=dict
    )

    expires_at: str | None = None

    def mark_read(self) -> None:
        """Mark notification as read."""

        self.read = True

    def dismiss(self) -> None:
        """Dismiss the notification."""

        self.dismissed = True

    def is_active(self) -> bool:
        """Return whether notification is still active."""

        return not self.dismissed

    def to_dict(self) -> dict[str, Any]:
        """Convert notification to a dictionary."""

        result = asdict(self)

        result["priority"] = (
            self.priority.name
        )

        return result


class NotificationManager:
    """
    Central notification manager.

    The manager maintains:

    1. Pending notification queue
    2. Notification history
    3. Registered handlers
    4. Spam/rate-limit state
    """

    def __init__(
        self,
        *,
        max_history: int = 500,
        max_queue: int = 100,
        default_source: str = "RENIX",
    ) -> None:

        if max_history < 1:
            raise ValueError(
                "max_history must be >= 1"
            )

        if max_queue < 1:
            raise ValueError(
                "max_queue must be >= 1"
            )

        self.max_history = max_history
        self.max_queue = max_queue
        self.default_source = default_source

        self._queue: Deque[
            Notification
        ] = deque(
            maxlen=max_queue
        )

        self._history: Deque[
            Notification
        ] = deque(
            maxlen=max_history
        )

        self._handlers: dict[
            str,
            NotificationHandler,
        ] = {}

        self._lock = threading.RLock()

        self._suppression_keys: set[str] = set()

        self._enabled = True

    # ========================================================
    # STATE
    # ========================================================

    @property
    def enabled(self) -> bool:
        """Return whether notifications are enabled."""

        return self._enabled

    def enable(self) -> None:
        """Enable notifications."""

        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        """Disable notifications."""

        with self._lock:
            self._enabled = False

    # ========================================================
    # CREATE
    # ========================================================

    def create(
        self,
        title: str,
        message: str,
        *,
        priority: NotificationPriority = (
            NotificationPriority.NORMAL
        ),
        category: str = "general",
        source: str | None = None,
        data: Mapping[str, Any] | None = None,
        expires_at: str | None = None,
        enqueue: bool = True,
    ) -> Notification:
        """
        Create a notification.

        If enqueue=True, it is added to the notification queue.
        """

        notification = Notification(
            title=str(title),
            message=str(message),
            priority=priority,
            category=str(category),
            source=(
                source
                or self.default_source
            ),
            data=dict(data or {}),
            expires_at=expires_at,
        )

        with self._lock:
            self._history.append(
                notification
            )

            if (
                enqueue
                and self._enabled
                and not self._is_suppressed(
                    notification
                )
            ):
                self._queue.append(
                    notification
                )

        return notification

    # ========================================================
    # SHORTCUTS
    # ========================================================

    def info(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> Notification:
        """Create an informational notification."""

        return self.create(
            title,
            message,
            priority=NotificationPriority.NORMAL,
            **kwargs,
        )

    def success(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> Notification:
        """Create a success notification."""

        return self.create(
            title,
            message,
            priority=NotificationPriority.LOW,
            **kwargs,
        )

    def warning(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> Notification:
        """Create a warning notification."""

        return self.create(
            title,
            message,
            priority=NotificationPriority.HIGH,
            **kwargs,
        )

    def error(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> Notification:
        """Create an error notification."""

        return self.create(
            title,
            message,
            priority=NotificationPriority.CRITICAL,
            **kwargs,
        )

    # ========================================================
    # QUEUE
    # ========================================================

    def pending(
        self,
    ) -> list[Notification]:
        """Return all pending notifications."""

        with self._lock:
            return list(self._queue)

    def pending_count(self) -> int:
        """Return pending notification count."""

        with self._lock:
            return len(self._queue)

    def pop_next(
        self,
    ) -> Notification | None:
        """
        Remove and return the highest-priority
        pending notification.
        """

        with self._lock:
            if not self._queue:
                return None

            notifications = list(
                self._queue
            )

            notifications.sort(
                key=lambda item:
                    get_priority_value(
                        item.priority
                    ),
                reverse=True,
            )

            selected = notifications[0]

            self._queue.clear()

            for notification in notifications:
                if notification is not selected:
                    self._queue.append(
                        notification
                    )

            return selected

    def clear_queue(self) -> int:
        """Clear all pending notifications."""

        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    # ========================================================
    # DISPATCH
    # ========================================================

    def dispatch(
        self,
        notification: Notification | None = None,
    ) -> bool:
        """
        Dispatch a notification.

        If no notification is supplied, the next
        highest-priority pending notification is used.
        """

        if notification is None:
            notification = self.pop_next()

        if notification is None:
            return False

        if notification.dismissed:
            return False

        handlers = self._get_handlers(
            notification.category
        )

        handled = False

        for handler in handlers:
            try:
                result = handler(
                    notification
                )

                if result is not False:
                    handled = True

            except Exception:
                # A failing notification handler should
                # not stop other handlers.
                continue

        return handled

    def dispatch_all(self) -> int:
        """Dispatch all pending notifications."""

        count = 0

        while True:
            notification = self.pop_next()

            if notification is None:
                break

            if self.dispatch(
                notification
            ):
                count += 1

        return count

    # ========================================================
    # HANDLERS
    # ========================================================

    def register_handler(
        self,
        name: str,
        handler: NotificationHandler,
        *,
        category: str = "*",
    ) -> None:
        """
        Register a notification handler.

        category="*" means the handler receives
        all notification categories.
        """

        key = (
            f"{category}:{name}"
        )

        with self._lock:
            self._handlers[key] = handler

    def unregister_handler(
        self,
        name: str,
        *,
        category: str = "*",
    ) -> bool:
        """Remove a registered notification handler."""

        key = (
            f"{category}:{name}"
        )

        with self._lock:
            return (
                self._handlers.pop(
                    key,
                    None,
                )
                is not None
            )

    def _get_handlers(
        self,
        category: str,
    ) -> list[NotificationHandler]:

        with self._lock:
            handlers = []

            for key, handler in (
                self._handlers.items()
            ):
                registered_category = (
                    key.split(
                        ":",
                        1,
                    )[0]
                )

                if registered_category in {
                    "*",
                    category,
                }:
                    handlers.append(handler)

            return handlers

    # ========================================================
    # HISTORY
    # ========================================================

    def history(
        self,
        *,
        limit: int | None = None,
        category: str | None = None,
        unread_only: bool = False,
    ) -> list[Notification]:
        """Return notification history."""

        with self._lock:
            notifications = list(
                self._history
            )

        if category is not None:
            notifications = [
                item
                for item in notifications
                if item.category == category
            ]

        if unread_only:
            notifications = [
                item
                for item in notifications
                if not item.read
            ]

        notifications.reverse()

        if limit is not None:
            if limit < 0:
                raise ValueError(
                    "limit must be >= 0"
                )

            notifications = notifications[
                :limit
            ]

        return notifications

    def get(
        self,
        notification_id: str,
    ) -> Notification | None:
        """Find a notification by ID."""

        with self._lock:
            for notification in self._history:
                if (
                    notification.notification_id
                    == notification_id
                ):
                    return notification

        return None

    # ========================================================
    # READ / DISMISS
    # ========================================================

    def mark_read(
        self,
        notification_id: str,
    ) -> bool:
        """Mark a notification as read."""

        notification = self.get(
            notification_id
        )

        if notification is None:
            return False

        notification.mark_read()
        return True

    def mark_all_read(
        self,
    ) -> int:
        """Mark all notifications as read."""

        count = 0

        with self._lock:
            for notification in self._history:
                if not notification.read:
                    notification.mark_read()
                    count += 1

        return count

    def dismiss(
        self,
        notification_id: str,
    ) -> bool:
        """Dismiss a notification."""

        notification = self.get(
            notification_id
        )

        if notification is None:
            return False

        notification.dismiss()

        with self._lock:
            self._queue = deque(
                (
                    item
                    for item in self._queue
                    if item.notification_id
                    != notification_id
                ),
                maxlen=self.max_queue,
            )

        return True

    def dismiss_all(
        self,
    ) -> int:
        """Dismiss all active notifications."""

        count = 0

        with self._lock:
            for notification in self._history:
                if not notification.dismissed:
                    notification.dismiss()
                    count += 1

            self._queue.clear()

        return count

    # ========================================================
    # SUPPRESSION
    # ========================================================

    def suppress(
        self,
        key: str,
    ) -> None:
        """Suppress notifications using a suppression key."""

        with self._lock:
            self._suppression_keys.add(
                str(key)
            )

    def unsuppress(
        self,
        key: str,
    ) -> None:
        """Remove a suppression key."""

        with self._lock:
            self._suppression_keys.discard(
                str(key)
            )

    def clear_suppression(self) -> None:
        """Remove all suppression keys."""

        with self._lock:
            self._suppression_keys.clear()

    def _is_suppressed(
        self,
        notification: Notification,
    ) -> bool:

        possible_keys = {
            notification.category,
            f"{notification.category}:{notification.title}",
        }

        return bool(
            possible_keys
            & self._suppression_keys
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
    ) -> list[Notification]:
        """Search notification history."""

        query = str(query).strip().lower()

        if not query:
            return []

        results: list[Notification] = []

        with self._lock:
            notifications = list(
                self._history
            )

        for notification in reversed(
            notifications
        ):
            searchable = " ".join(
                [
                    notification.title,
                    notification.message,
                    notification.category,
                    notification.source,
                ]
            ).lower()

            if query in searchable:
                results.append(
                    notification
                )

            if len(results) >= limit:
                break

        return results

    # ========================================================
    # COUNTS
    # ========================================================

    def unread_count(self) -> int:
        """Return number of unread notifications."""

        with self._lock:
            return sum(
                not notification.read
                and not notification.dismissed
                for notification in self._history
            )

    def count_by_priority(
        self,
    ) -> dict[str, int]:
        """Count notifications by priority."""

        counts: dict[str, int] = {}

        with self._lock:
            for notification in self._history:
                name = notification.priority.name

                counts[name] = (
                    counts.get(name, 0)
                    + 1
                )

        return counts

    def count_by_category(
        self,
    ) -> dict[str, int]:
        """Count notifications by category."""

        counts: dict[str, int] = {}

        with self._lock:
            for notification in self._history:
                counts[
                    notification.category
                ] = (
                    counts.get(
                        notification.category,
                        0,
                    )
                    + 1
                )

        return counts

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def export_history(
        self,
    ) -> list[dict[str, Any]]:
        """Export notification history."""

        with self._lock:
            return [
                notification.to_dict()
                for notification in self._history
            ]

    def import_history(
        self,
        items: Iterable[Mapping[str, Any]],
    ) -> int:
        """
        Import notification history.

        Existing notifications are not cleared.
        """

        count = 0

        with self._lock:
            for item in items:
                try:
                    priority_value = item.get(
                        "priority",
                        "NORMAL",
                    )

                    if isinstance(
                        priority_value,
                        NotificationPriority,
                    ):
                        priority = priority_value
                    else:
                        priority = (
                            NotificationPriority[
                                str(
                                    priority_value
                                ).upper()
                            ]
                        )

                    notification = Notification(
                        title=str(
                            item.get(
                                "title",
                                "",
                            )
                        ),
                        message=str(
                            item.get(
                                "message",
                                "",
                            )
                        ),
                        priority=priority,
                        category=str(
                            item.get(
                                "category",
                                "general",
                            )
                        ),
                        notification_id=str(
                            item.get(
                                "notification_id",
                                uuid.uuid4(),
                            )
                        ),
                        created_at=str(
                            item.get(
                                "created_at",
                                datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            )
                        ),
                        read=bool(
                            item.get(
                                "read",
                                False,
                            )
                        ),
                        dismissed=bool(
                            item.get(
                                "dismissed",
                                False,
                            )
                        ),
                        source=str(
                            item.get(
                                "source",
                                self.default_source,
                            )
                        ),
                        data=dict(
                            item.get(
                                "data",
                                {},
                            )
                        ),
                        expires_at=item.get(
                            "expires_at"
                        ),
                    )

                    self._history.append(
                        notification
                    )

                    count += 1

                except Exception:
                    continue

        return count

    # ========================================================
    # RESET
    # ========================================================

    def clear_history(self) -> int:
        """Clear notification history."""

        with self._lock:
            count = len(self._history)
            self._history.clear()
            return count

    def reset(self) -> None:
        """Reset the notification manager."""

        with self._lock:
            self._queue.clear()
            self._history.clear()
            self._suppression_keys.clear()

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict[str, Any]:
        """Return notification manager status."""

        with self._lock:
            return {
                "enabled": self._enabled,
                "pending": len(
                    self._queue
                ),
                "history": len(
                    self._history
                ),
                "unread": self.unread_count(),
                "handlers": len(
                    self._handlers
                ),
                "suppressed_categories": len(
                    self._suppression_keys
                ),
                "max_history": self.max_history,
                "max_queue": self.max_queue,
            }

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __len__(self) -> int:
        """Return number of stored notifications."""

        with self._lock:
            return len(self._history)

    def __repr__(self) -> str:
        return (
            "NotificationManager("
            f"enabled={self._enabled}, "
            f"pending={len(self._queue)}, "
            f"history={len(self._history)})"
        )


