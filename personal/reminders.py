"""
RENIX Personal Reminders

Manages one-time and recurring reminders, due reminders,
snoozing, completion, cancellation, priorities, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable
import uuid


@dataclass
class Reminder:
    """Represents a RENIX reminder."""

    title: str
    remind_at: datetime
    description: str = ""
    priority: str = "normal"
    reminder_id: str | None = None
    completed: bool = False
    cancelled: bool = False
    recurring: bool = False
    recurrence_minutes: int | None = None
    snoozed_until: datetime | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.reminder_id:
            self.reminder_id = (
                f"REM-{uuid.uuid4().hex[:12].upper()}"
            )

        self.priority = (
            self.priority or "normal"
        ).lower()

        if self.recurrence_minutes is not None:
            self.recurrence_minutes = max(
                1,
                int(self.recurrence_minutes),
            )

    def is_due(
        self,
        now: datetime | None = None,
    ) -> bool:

        now = now or datetime.now()

        if self.completed or self.cancelled:
            return False

        target = (
            self.snoozed_until
            or self.remind_at
        )

        return target <= now

    def to_dict(self) -> dict[str, Any]:

        return {
            "reminder_id": self.reminder_id,
            "title": self.title,
            "remind_at": self.remind_at.isoformat(),
            "description": self.description,
            "priority": self.priority,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "recurring": self.recurring,
            "recurrence_minutes": (
                self.recurrence_minutes
            ),
            "snoozed_until": (
                self.snoozed_until.isoformat()
                if self.snoozed_until
                else None
            ),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


class ReminderManager:
    """Central reminder manager for RENIX."""

    VALID_PRIORITIES = {
        "low",
        "normal",
        "high",
        "urgent",
    }

    def __init__(self) -> None:

        self.reminders: dict[
            str,
            Reminder,
        ] = {}

    # ============================================================
    # CREATE
    # ============================================================

    def create_reminder(
        self,
        title: str,
        remind_at: datetime,
        *,
        description: str = "",
        priority: str = "normal",
        reminder_id: str | None = None,
        recurring: bool = False,
        recurrence_minutes: int | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Reminder:

        title = (
            title or ""
        ).strip()

        if not title:
            raise ValueError(
                "Reminder title cannot be empty."
            )

        if not isinstance(
            remind_at,
            datetime,
        ):
            raise TypeError(
                "remind_at must be a datetime."
            )

        priority = (
            priority or "normal"
        ).lower()

        if priority not in self.VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority: {priority}"
            )

        reminder = Reminder(
            title=title,
            remind_at=remind_at,
            description=description,
            priority=priority,
            reminder_id=reminder_id,
            recurring=recurring,
            recurrence_minutes=recurrence_minutes,
            tags=list(tags or []),
            metadata=dict(
                metadata or {}
            ),
        )

        if reminder.reminder_id in self.reminders:
            raise ValueError(
                "Reminder ID already exists."
            )

        self.reminders[
            reminder.reminder_id
        ] = reminder

        return reminder

    def add_reminder(
        self,
        reminder: Reminder,
    ) -> str:

        if not isinstance(
            reminder,
            Reminder,
        ):
            raise TypeError(
                "reminder must be a Reminder."
            )

        if reminder.reminder_id in self.reminders:
            raise ValueError(
                "Reminder ID already exists."
            )

        self.reminders[
            reminder.reminder_id
        ] = reminder

        return reminder.reminder_id

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def get(
        self,
        reminder_id: str,
    ) -> Reminder | None:

        return self.reminders.get(
            reminder_id
        )

    def all(
        self,
        *,
        include_completed: bool = True,
        include_cancelled: bool = False,
    ) -> list[Reminder]:

        reminders = list(
            self.reminders.values()
        )

        if not include_completed:
            reminders = [
                item
                for item in reminders
                if not item.completed
            ]

        if not include_cancelled:
            reminders = [
                item
                for item in reminders
                if not item.cancelled
            ]

        return sorted(
            reminders,
            key=lambda item: item.remind_at,
        )

    def upcoming(
        self,
        *,
        hours: float = 24,
        now: datetime | None = None,
    ) -> list[Reminder]:

        now = now or datetime.now()

        end = now + timedelta(
            hours=hours
        )

        return sorted(
            [
                item
                for item in self.reminders.values()
                if not item.completed
                and not item.cancelled
                and item.remind_at >= now
                and item.remind_at <= end
            ],
            key=lambda item: item.remind_at,
        )

    def due(
        self,
        now: datetime | None = None,
    ) -> list[Reminder]:

        now = now or datetime.now()

        return sorted(
            [
                item
                for item in self.reminders.values()
                if item.is_due(now)
            ],
            key=lambda item: (
                self._priority_value(
                    item.priority
                ),
                item.remind_at,
            ),
            reverse=True,
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
    ) -> list[Reminder]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        results = []

        for reminder in self.reminders.values():

            searchable = " ".join(
                [
                    reminder.title,
                    reminder.description,
                    reminder.priority,
                    " ".join(reminder.tags),
                ]
            ).casefold()

            if query in searchable:
                results.append(reminder)

        return sorted(
            results,
            key=lambda item: item.remind_at,
        )

    # ============================================================
    # UPDATE
    # ============================================================

    def update(
        self,
        reminder_id: str,
        **changes: Any,
    ) -> bool:

        reminder = self.get(
            reminder_id
        )

        if reminder is None:
            return False

        allowed = {
            "title",
            "remind_at",
            "description",
            "priority",
            "recurring",
            "recurrence_minutes",
            "tags",
            "metadata",
        }

        for key, value in changes.items():

            if key not in allowed:
                continue

            if key == "priority":

                value = str(
                    value
                ).lower()

                if value not in self.VALID_PRIORITIES:
                    raise ValueError(
                        f"Invalid priority: {value}"
                    )

            if key == "title":

                value = str(
                    value
                ).strip()

                if not value:
                    raise ValueError(
                        "Reminder title cannot be empty."
                    )

            if key == "remind_at":

                if not isinstance(
                    value,
                    datetime,
                ):
                    raise TypeError(
                        "remind_at must be a datetime."
                    )

            if key == "recurrence_minutes":

                if value is not None:
                    value = max(
                        1,
                        int(value),
                    )

            setattr(
                reminder,
                key,
                value,
            )

        return True

    # ============================================================
    # COMPLETE / CANCEL
    # ============================================================

    def complete(
        self,
        reminder_id: str,
    ) -> bool:

        reminder = self.get(
            reminder_id
        )

        if reminder is None:
            return False

        if reminder.recurring:

            return self._advance_recurring(
                reminder
            )

        reminder.completed = True
        reminder.snoozed_until = None

        return True

    def cancel(
        self,
        reminder_id: str,
    ) -> bool:

        reminder = self.get(
            reminder_id
        )

        if reminder is None:
            return False

        reminder.cancelled = True
        reminder.snoozed_until = None

        return True

    def restore(
        self,
        reminder_id: str,
    ) -> bool:

        reminder = self.get(
            reminder_id
        )

        if reminder is None:
            return False

        reminder.cancelled = False
        reminder.completed = False

        return True

    # ============================================================
    # SNOOZE
    # ============================================================

    def snooze(
        self,
        reminder_id: str,
        minutes: int = 10,
    ) -> bool:

        reminder = self.get(
            reminder_id
        )

        if reminder is None:
            return False

        if reminder.completed or reminder.cancelled:
            return False

        minutes = max(
            1,
            int(minutes),
        )

        reminder.snoozed_until = (
            datetime.now()
            + timedelta(
                minutes=minutes
            )
        )

        return True

    def snooze_until(
        self,
        reminder_id: str,
        target: datetime,
    ) -> bool:

        reminder = self.get(
            reminder_id
        )

        if reminder is None:
            return False

        if reminder.completed or reminder.cancelled:
            return False

        if not isinstance(
            target,
            datetime,
        ):
            raise TypeError(
                "target must be a datetime."
            )

        reminder.snoozed_until = target

        return True

    # ============================================================
    # DELETE
    # ============================================================

    def delete(
        self,
        reminder_id: str,
    ) -> bool:

        if reminder_id not in self.reminders:
            return False

        del self.reminders[
            reminder_id
        ]

        return True

    def clear_completed(self) -> int:

        ids = [
            reminder_id
            for reminder_id, reminder
            in self.reminders.items()
            if reminder.completed
        ]

        for reminder_id in ids:
            del self.reminders[
                reminder_id
            ]

        return len(ids)

    # ============================================================
    # NATURAL COMMANDS
    # ============================================================

    def handle_command(
        self,
        command: str,
    ) -> dict[str, Any]:

        """
        Handle simple reminder commands.

        Supported examples:

            "show reminders"
            "show due reminders"
            "find homework reminder"
            "cancel REM-123"
        """

        command = (
            command or ""
        ).strip()

        if not command:

            return {
                "success": False,
                "message": "Empty reminder command.",
            }

        lowered = command.casefold()

        if (
            "due reminder"
            in lowered
            or lowered == "due"
        ):

            return {
                "success": True,
                "reminders": [
                    item.to_dict()
                    for item in self.due()
                ],
            }

        if (
            "show reminder"
            in lowered
            or lowered == "reminders"
        ):

            return {
                "success": True,
                "reminders": [
                    item.to_dict()
                    for item in self.all(
                        include_completed=False
                    )
                ],
            }

        if lowered.startswith(
            "find reminder "
        ):

            query = command[
                len("find reminder "):
            ]

            return {
                "success": True,
                "reminders": [
                    item.to_dict()
                    for item in self.search(
                        query
                    )
                ],
            }

        if lowered.startswith(
            "cancel "
        ):

            reminder_id = command[
                len("cancel "):
            ].strip()

            return {
                "success": self.cancel(
                    reminder_id
                ),
                "reminder_id": reminder_id,
            }

        if lowered.startswith(
            "complete "
        ):

            reminder_id = command[
                len("complete "):
            ].strip()

            return {
                "success": self.complete(
                    reminder_id
                ),
                "reminder_id": reminder_id,
            }

        return {
            "success": False,
            "message": (
                "Unknown reminder command."
            ),
        }

    # ============================================================
    # RECURRING REMINDERS
    # ============================================================

    def process_due_recurring(
        self,
        now: datetime | None = None,
    ) -> list[Reminder]:

        now = now or datetime.now()

        processed = []

        for reminder in list(
            self.reminders.values()
        ):

            if not reminder.recurring:
                continue

            if reminder.cancelled:
                continue

            if reminder.is_due(now):

                self._advance_recurring(
                    reminder
                )

                processed.append(
                    reminder
                )

        return processed

    def _advance_recurring(
        self,
        reminder: Reminder,
    ) -> bool:

        if (
            not reminder.recurring
            or not reminder.recurrence_minutes
        ):

            reminder.completed = True
            return True

        reminder.remind_at = (
            reminder.remind_at
            + timedelta(
                minutes=reminder.recurrence_minutes
            )
        )

        reminder.snoozed_until = None
        reminder.completed = False

        return True

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(self) -> dict[str, Any]:

        reminders = list(
            self.reminders.values()
        )

        return {
            "total": len(reminders),
            "active": sum(
                not item.completed
                and not item.cancelled
                for item in reminders
            ),
            "completed": sum(
                item.completed
                for item in reminders
            ),
            "cancelled": sum(
                item.cancelled
                for item in reminders
            ),
            "recurring": sum(
                item.recurring
                for item in reminders
            ),
            "due": len(
                self.due()
            ),
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "reminders": [
                item.to_dict()
                for item in self.reminders.values()
            ]
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "Reminder state must be a dictionary."
            )

        self.reminders.clear()

        raw_reminders = data.get(
            "reminders",
            [],
        )

        if not isinstance(
            raw_reminders,
            list,
        ):
            return

        for raw in raw_reminders:

            if not isinstance(
                raw,
                dict,
            ):
                continue

            try:

                remind_at = datetime.fromisoformat(
                    raw["remind_at"]
                )

                snoozed_raw = raw.get(
                    "snoozed_until"
                )

                snoozed_until = (
                    datetime.fromisoformat(
                        snoozed_raw
                    )
                    if snoozed_raw
                    else None
                )

                reminder = Reminder(
                    title=str(
                        raw.get(
                            "title",
                            "Reminder",
                        )
                    ),
                    remind_at=remind_at,
                    description=str(
                        raw.get(
                            "description",
                            "",
                        )
                    ),
                    priority=str(
                        raw.get(
                            "priority",
                            "normal",
                        )
                    ),
                    reminder_id=raw.get(
                        "reminder_id"
                    ),
                    completed=bool(
                        raw.get(
                            "completed",
                            False,
                        )
                    ),
                    cancelled=bool(
                        raw.get(
                            "cancelled",
                            False,
                        )
                    ),
                    recurring=bool(
                        raw.get(
                            "recurring",
                            False,
                        )
                    ),
                    recurrence_minutes=raw.get(
                        "recurrence_minutes"
                    ),
                    snoozed_until=snoozed_until,
                    tags=list(
                        raw.get(
                            "tags",
                            [],
                        )
                    ),
                    metadata=dict(
                        raw.get(
                            "metadata",
                            {},
                        )
                    ),
                )

                self.reminders[
                    reminder.reminder_id
                ] = reminder

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue


__all__ = [
    "Reminder",
    "ReminderManager",
]


