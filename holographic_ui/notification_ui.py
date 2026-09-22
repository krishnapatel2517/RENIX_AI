"""
RENIX Holographic UI
Notification UI
=================

Holographic notification system for RENIX.

Supports:
- Info / success / warning / error / system notifications
- Priority levels
- Auto-expiration
- Persistent notifications
- Progress notifications
- Action buttons/callbacks
- Queue management
- Notification history
- Read/unread state
- Positioning
- Renderer-independent snapshots

The renderer can consume `get_snapshot()` and draw
the actual holographic notification cards.
"""

from __future__ import annotations

import threading
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# ENUMS
# ============================================================================


class NotificationType(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SYSTEM = "system"
    AI = "ai"
    TASK = "task"
    SECURITY = "security"


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationState(str, Enum):
    QUEUED = "queued"
    VISIBLE = "visible"
    READ = "read"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class NotificationPosition(str, Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


# ============================================================================
# ACTION
# ============================================================================


@dataclass
class NotificationAction:
    id: str
    label: str
    callback: Optional[
        Callable[..., Any]
    ] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# NOTIFICATION
# ============================================================================


@dataclass
class Notification:
    id: str

    title: str

    message: str

    notification_type: NotificationType = (
        NotificationType.INFO
    )

    priority: NotificationPriority = (
        NotificationPriority.NORMAL
    )

    state: NotificationState = (
        NotificationState.QUEUED
    )

    created_at: float = field(
        default_factory=time.time
    )

    visible_at: Optional[float] = None

    expires_at: Optional[float] = None

    duration: Optional[float] = 5.0

    persistent: bool = False

    progress: Optional[float] = None

    icon: Optional[str] = None

    source: Optional[str] = None

    position: NotificationPosition = (
        NotificationPosition.TOP_RIGHT
    )

    actions: list[
        NotificationAction
    ] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    read: bool = False

    dismissible: bool = True

    sound: Optional[str] = None

    glow: float = 1.0

    def activate(self) -> None:

        now = time.time()

        self.state = (
            NotificationState.VISIBLE
        )

        self.visible_at = now

        if (
            not self.persistent
            and self.duration is not None
            and self.duration > 0
        ):

            self.expires_at = (
                now + self.duration
            )

    def mark_read(self) -> None:

        self.read = True

        if self.state == NotificationState.VISIBLE:

            self.state = (
                NotificationState.READ
            )

    def dismiss(self) -> None:

        self.state = (
            NotificationState.DISMISSED
        )

    def expired(self) -> bool:

        if self.persistent:
            return False

        if self.expires_at is None:
            return False

        return time.time() >= self.expires_at


# ============================================================================
# NOTIFICATION UI
# ============================================================================


class NotificationUI:

    def __init__(
        self,
        *,
        max_visible: int = 5,
        max_history: int = 200,
        default_duration: float = 5.0,
        default_position: NotificationPosition = (
            NotificationPosition.TOP_RIGHT
        ),
    ) -> None:

        self.max_visible = max(
            1,
            int(max_visible),
        )

        self.max_history = max(
            1,
            int(max_history),
        )

        self.default_duration = max(
            0.0,
            float(default_duration),
        )

        self.default_position = (
            default_position
        )

        self._notifications: dict[
            str,
            Notification,
        ] = {}

        self._queue: list[str] = []

        self._history: list[str] = []

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False

        self._last_update = time.monotonic()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:
                return

            self._running = True

            self._last_update = (
                time.monotonic()
            )

        self._emit(
            "started",
            self,
        )

    def stop(self) -> None:

        with self._lock:

            self._running = False

        self._emit(
            "stopped",
            self,
        )

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        delta_time: Optional[float] = None,
    ) -> None:

        if not self._running:
            return

        now = time.monotonic()

        if delta_time is None:

            delta_time = (
                now - self._last_update
            )

        self._last_update = now

        self._process_expiration()

        self._activate_queued_notifications()

        self._emit(
            "updated",
            delta_time,
        )

    # ========================================================================
    # CREATE
    # ========================================================================

    def notify(
        self,
        title: str,
        message: str,
        *,
        notification_type: NotificationType = (
            NotificationType.INFO
        ),
        priority: NotificationPriority = (
            NotificationPriority.NORMAL
        ),
        duration: Optional[float] = None,
        persistent: bool = False,
        progress: Optional[float] = None,
        icon: Optional[str] = None,
        source: Optional[str] = None,
        position: Optional[
            NotificationPosition
        ] = None,
        actions: Optional[
            list[NotificationAction]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        dismissible: bool = True,
        sound: Optional[str] = None,
        glow: float = 1.0,
    ) -> str:

        if not title:

            raise ValueError(
                "Notification title cannot be empty"
            )

        if duration is None:

            duration = (
                self.default_duration
            )

        notification = Notification(
            id=uuid.uuid4().hex,
            title=title,
            message=message,
            notification_type=(
                notification_type
            ),
            priority=priority,
            duration=duration,
            persistent=persistent,
            progress=self._normalize_progress(
                progress
            ),
            icon=icon,
            source=source,
            position=(
                position
                or self.default_position
            ),
            actions=list(
                actions or []
            ),
            metadata=dict(
                metadata or {}
            ),
            dismissible=dismissible,
            sound=sound,
            glow=max(
                0.0,
                float(glow),
            ),
        )

        with self._lock:

            self._notifications[
                notification.id
            ] = notification

            self._queue.append(
                notification.id
            )

            self._history.append(
                notification.id
            )

            self._trim_history()

        self._emit(
            "notification_created",
            notification,
        )

        if self._running:

            self._activate_queued_notifications()

        return notification.id

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    def info(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> str:

        return self.notify(
            title,
            message,
            notification_type=(
                NotificationType.INFO
            ),
            **kwargs,
        )

    def success(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> str:

        return self.notify(
            title,
            message,
            notification_type=(
                NotificationType.SUCCESS
            ),
            **kwargs,
        )

    def warning(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> str:

        return self.notify(
            title,
            message,
            notification_type=(
                NotificationType.WARNING
            ),
            priority=(
                kwargs.pop(
                    "priority",
                    NotificationPriority.HIGH,
                )
            ),
            **kwargs,
        )

    def error(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> str:

        return self.notify(
            title,
            message,
            notification_type=(
                NotificationType.ERROR
            ),
            priority=(
                kwargs.pop(
                    "priority",
                    NotificationPriority.HIGH,
                )
            ),
            **kwargs,
        )

    def system(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> str:

        return self.notify(
            title,
            message,
            notification_type=(
                NotificationType.SYSTEM
            ),
            **kwargs,
        )

    def ai(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> str:

        return self.notify(
            title,
            message,
            notification_type=(
                NotificationType.AI
            ),
            **kwargs,
        )

    def task(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> str:

        return self.notify(
            title,
            message,
            notification_type=(
                NotificationType.TASK
            ),
            **kwargs,
        )

    def security(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> str:

        return self.notify(
            title,
            message,
            notification_type=(
                NotificationType.SECURITY
            ),
            priority=(
                kwargs.pop(
                    "priority",
                    NotificationPriority.CRITICAL,
                )
            ),
            **kwargs,
        )

    # ========================================================================
    # QUEUE
    # ========================================================================

    def _activate_queued_notifications(
        self,
    ) -> None:

        with self._lock:

            visible = self._visible_notifications()

            available = (
                self.max_visible
                - len(visible)
            )

            if available <= 0:
                return

            queued = [
                self._notifications[
                    notification_id
                ]
                for notification_id
                in self._queue
                if notification_id
                in self._notifications
                and self._notifications[
                    notification_id
                ].state
                == NotificationState.QUEUED
            ]

            queued.sort(
                key=self._priority_value,
                reverse=True,
            )

            for notification in queued[
                :available
            ]:

                notification.activate()

                if notification.id in self._queue:

                    self._queue.remove(
                        notification.id
                    )

                self._emit(
                    "notification_shown",
                    notification,
                )

    def _visible_notifications(
        self,
    ) -> list[Notification]:

        return [
            notification
            for notification
            in self._notifications.values()
            if notification.state
            in (
                NotificationState.VISIBLE,
                NotificationState.READ,
            )
        ]

    # ========================================================================
    # EXPIRATION
    # ========================================================================

    def _process_expiration(
        self,
    ) -> None:

        expired: list[
            Notification
        ] = []

        with self._lock:

            for notification in (
                self._notifications.values()
            ):

                if (
                    notification.state
                    not in (
                        NotificationState.VISIBLE,
                        NotificationState.READ,
                    )
                ):
                    continue

                if notification.expired():

                    notification.state = (
                        NotificationState.EXPIRED
                    )

                    expired.append(
                        notification
                    )

        for notification in expired:

            self._emit(
                "notification_expired",
                notification,
            )

        if expired:

            self._activate_queued_notifications()

    # ========================================================================
    # GET
    # ========================================================================

    def get(
        self,
        notification_id: str,
    ) -> Optional[Notification]:

        with self._lock:

            return self._notifications.get(
                notification_id
            )

    def visible(
        self,
    ) -> list[Notification]:

        with self._lock:

            return list(
                self._visible_notifications()
            )

    def queued(
        self,
    ) -> list[Notification]:

        with self._lock:

            result: list[
                Notification
            ] = []

            for notification_id in (
                self._queue
            ):

                notification = (
                    self._notifications.get(
                        notification_id
                    )
                )

                if notification:

                    result.append(
                        notification
                    )

            return result

    def history(
        self,
    ) -> list[Notification]:

        with self._lock:

            result: list[
                Notification
            ] = []

            for notification_id in (
                self._history
            ):

                notification = (
                    self._notifications.get(
                        notification_id
                    )
                )

                if notification:

                    result.append(
                        notification
                    )

            return result

    # ========================================================================
    # READ / DISMISS
    # ========================================================================

    def mark_read(
        self,
        notification_id: str,
    ) -> bool:

        notification = self.get(
            notification_id
        )

        if notification is None:
            return False

        notification.mark_read()

        self._emit(
            "notification_read",
            notification,
        )

        return True

    def dismiss(
        self,
        notification_id: str,
    ) -> bool:

        notification = self.get(
            notification_id
        )

        if notification is None:
            return False

        if not notification.dismissible:

            return False

        notification.dismiss()

        with self._lock:

            if notification_id in self._queue:

                self._queue.remove(
                    notification_id
                )

        self._emit(
            "notification_dismissed",
            notification,
        )

        self._activate_queued_notifications()

        return True

    def dismiss_all(
        self,
    ) -> int:

        dismissed = 0

        with self._lock:

            notifications = list(
                self._notifications.values()
            )

        for notification in notifications:

            if (
                notification.dismissible
                and notification.state
                in (
                    NotificationState.QUEUED,
                    NotificationState.VISIBLE,
                    NotificationState.READ,
                )
            ):

                notification.dismiss()

                dismissed += 1

        with self._lock:

            self._queue.clear()

        self._emit(
            "all_dismissed",
            dismissed,
        )

        return dismissed

    # ========================================================================
    # PROGRESS
    # ========================================================================

    def set_progress(
        self,
        notification_id: str,
        progress: Optional[float],
    ) -> bool:

        notification = self.get(
            notification_id
        )

        if notification is None:
            return False

        notification.progress = (
            self._normalize_progress(
                progress
            )
        )

        self._emit(
            "progress_changed",
            notification,
        )

        return True

    def complete_progress(
        self,
        notification_id: str,
        *,
        success: bool = True,
        message: Optional[str] = None,
    ) -> bool:

        notification = self.get(
            notification_id
        )

        if notification is None:
            return False

        notification.progress = 1.0

        if message is not None:

            notification.message = message

        notification.notification_type = (
            NotificationType.SUCCESS
            if success
            else NotificationType.ERROR
        )

        self._emit(
            "progress_completed",
            notification,
        )

        return True

    # ========================================================================
    # ACTIONS
    # ========================================================================

    def add_action(
        self,
        notification_id: str,
        label: str,
        callback: Optional[
            Callable[..., Any]
        ] = None,
        *,
        action_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[str]:

        notification = self.get(
            notification_id
        )

        if notification is None:
            return None

        action = NotificationAction(
            id=(
                action_id
                or uuid.uuid4().hex
            ),
            label=label,
            callback=callback,
            metadata=dict(
                metadata or {}
            ),
        )

        notification.actions.append(
            action
        )

        self._emit(
            "action_added",
            notification,
            action,
        )

        return action.id

    def trigger_action(
        self,
        notification_id: str,
        action_id: str,
    ) -> Any:

        notification = self.get(
            notification_id
        )

        if notification is None:
            return None

        action = next(
            (
                action
                for action
                in notification.actions
                if action.id == action_id
            ),
            None,
        )

        if action is None:
            return None

        self._emit(
            "action_triggered",
            notification,
            action,
        )

        if action.callback is None:
            return None

        return action.callback(
            notification,
            action,
        )

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def get_snapshot(
        self,
    ) -> list[dict[str, Any]]:

        notifications = self.visible()

        notifications.sort(
            key=lambda notification: (
                self._priority_value(
                    notification
                ),
                notification.created_at,
            ),
            reverse=True,
        )

        return [
            self._serialize(
                notification
            )
            for notification in notifications
        ]

    def get_history_snapshot(
        self,
    ) -> list[dict[str, Any]]:

        return [
            self._serialize(
                notification
            )
            for notification in self.history()
        ]

    def _serialize(
        self,
        notification: Notification,
    ) -> dict[str, Any]:

        return {
            "id": notification.id,
            "title": notification.title,
            "message": notification.message,
            "type": (
                notification.notification_type.value
            ),
            "priority": (
                notification.priority.value
            ),
            "state": (
                notification.state.value
            ),
            "created_at": notification.created_at,
            "visible_at": notification.visible_at,
            "expires_at": notification.expires_at,
            "duration": notification.duration,
            "persistent": notification.persistent,
            "progress": notification.progress,
            "icon": notification.icon,
            "source": notification.source,
            "position": (
                notification.position.value
            ),
            "read": notification.read,
            "dismissible": notification.dismissible,
            "sound": notification.sound,
            "glow": notification.glow,
            "actions": [
                {
                    "id": action.id,
                    "label": action.label,
                    "metadata": dict(
                        action.metadata
                    ),
                }
                for action in notification.actions
            ],
            "metadata": dict(
                notification.metadata
            ),
        }

    # ========================================================================
    # HISTORY
    # ========================================================================

    def clear_history(
        self,
    ) -> None:

        with self._lock:

            active_ids = {
                notification.id
                for notification
                in self._notifications.values()
                if notification.state
                in (
                    NotificationState.QUEUED,
                    NotificationState.VISIBLE,
                    NotificationState.READ,
                )
            }

            self._history = [
                notification_id
                for notification_id
                in self._history
                if notification_id
                in active_ids
            ]

            self._notifications = {
                notification_id: notification
                for notification_id, notification
                in self._notifications.items()
                if notification_id
                in active_ids
            }

    def _trim_history(
        self,
    ) -> None:

        if len(self._history) <= self.max_history:

            return

        remove_count = (
            len(self._history)
            - self.max_history
        )

        old_ids = self._history[
            :remove_count
        ]

        self._history = self._history[
            remove_count:
        ]

        for notification_id in old_ids:

            notification = (
                self._notifications.get(
                    notification_id
                )
            )

            if notification is None:
                continue

            if notification.state in (
                NotificationState.QUEUED,
                NotificationState.VISIBLE,
                NotificationState.READ,
            ):

                continue

            self._notifications.pop(
                notification_id,
                None,
            )

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _normalize_progress(
        progress: Optional[float],
    ) -> Optional[float]:

        if progress is None:
            return None

        return max(
            0.0,
            min(
                1.0,
                float(progress),
            ),
        )

    @staticmethod
    def _priority_value(
        notification: Notification,
    ) -> int:

        values = {
            NotificationPriority.LOW: 0,
            NotificationPriority.NORMAL: 1,
            NotificationPriority.HIGH: 2,
            NotificationPriority.CRITICAL: 3,
        }

        return values.get(
            notification.priority,
            1,
        )

    # ========================================================================
    # EVENTS
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not event:
            raise ValueError(
                "Event cannot be empty"
            )

        if not callable(callback):
            raise TypeError(
                "Callback must be callable"
            )

        with self._lock:

            self._callbacks.setdefault(
                event,
                [],
            ).append(callback)

    def off(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        with self._lock:

            callbacks = self._callbacks.get(
                event,
                [],
            )

            if callback in callbacks:

                callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks.get(
                    event,
                    [],
                )
            )

        for callback in callbacks:

            try:

                callback(
                    *args,
                    **kwargs,
                )

            except Exception:
                # UI callbacks should never crash
                # the notification engine.
                pass

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(self) -> dict[str, Any]:

        with self._lock:

            visible = len(
                self._visible_notifications()
            )

            queued = len(
                self._queue
            )

            unread = sum(
                1
                for notification
                in self._notifications.values()
                if not notification.read
                and notification.state
                in (
                    NotificationState.VISIBLE,
                    NotificationState.READ,
                )
            )

            return {
                "running": self._running,
                "visible": visible,
                "queued": queued,
                "unread": unread,
                "history": len(
                    self._history
                ),
                "max_visible": (
                    self.max_visible
                ),
                "max_history": (
                    self.max_history
                ),
            }


# ============================================================================
# FACTORY
# ============================================================================


def create_notification_ui() -> NotificationUI:

    ui = NotificationUI(
        max_visible=5,
        max_history=200,
        default_duration=5.0,
        default_position=(
            NotificationPosition.TOP_RIGHT
        ),
    )

    ui.start()

    return ui


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "NotificationType",
    "NotificationPriority",
    "NotificationState",
    "NotificationPosition",
    "NotificationAction",
    "Notification",
    "NotificationUI",
    "create_notification_ui",
]


