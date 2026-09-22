"""
RENIX Personal Calendar Manager

Handles calendar events, schedules, event searching,
upcoming events, recurring events, and calendar state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable
import uuid


@dataclass
class CalendarEvent:
    """Represents a calendar event."""

    title: str
    start_time: datetime
    end_time: datetime | None = None
    description: str = ""
    location: str = ""
    category: str = ""
    event_id: str | None = None
    all_day: bool = False
    recurring: bool = False
    recurrence_rule: str | None = None
    reminder_minutes: int | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = (
                f"EVENT-{uuid.uuid4().hex[:12].upper()}"
            )

        if (
            self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError(
                "Event end time cannot be before start time."
            )

    @property
    def duration(self) -> float:
        """Return event duration in seconds."""

        if self.end_time is None:
            return 0.0

        return (
            self.end_time - self.start_time
        ).total_seconds()

    def overlaps(
        self,
        start: datetime,
        end: datetime,
    ) -> bool:
        """Check whether the event overlaps a time range."""

        event_end = (
            self.end_time
            or self.start_time
        )

        return (
            self.start_time < end
            and event_end > start
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": (
                self.end_time.isoformat()
                if self.end_time
                else None
            ),
            "description": self.description,
            "location": self.location,
            "category": self.category,
            "all_day": self.all_day,
            "recurring": self.recurring,
            "recurrence_rule": self.recurrence_rule,
            "reminder_minutes": self.reminder_minutes,
            "duration": self.duration,
            "metadata": dict(self.metadata),
        }


class CalendarManager:
    """
    RENIX calendar management system.

    This class manages calendar data locally. External calendar
    providers such as Google Calendar or Outlook can later be
    connected through the integrations layer.
    """

    def __init__(self) -> None:

        self.events: dict[
            str,
            CalendarEvent,
        ] = {}

        self.default_calendar = "RENIX"

    # ============================================================
    # EVENT CREATION
    # ============================================================

    def create_event(
        self,
        title: str,
        start_time: datetime,
        *,
        end_time: datetime | None = None,
        description: str = "",
        location: str = "",
        category: str = "",
        event_id: str | None = None,
        all_day: bool = False,
        recurring: bool = False,
        recurrence_rule: str | None = None,
        reminder_minutes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CalendarEvent:

        title = (
            title or ""
        ).strip()

        if not title:
            raise ValueError(
                "Calendar event title cannot be empty."
            )

        if not isinstance(
            start_time,
            datetime,
        ):
            raise TypeError(
                "start_time must be a datetime."
            )

        event = CalendarEvent(
            title=title,
            start_time=start_time,
            end_time=end_time,
            description=description,
            location=location,
            category=category,
            event_id=event_id,
            all_day=all_day,
            recurring=recurring,
            recurrence_rule=recurrence_rule,
            reminder_minutes=reminder_minutes,
            metadata=dict(
                metadata or {}
            ),
        )

        if event.event_id in self.events:
            raise ValueError(
                f"Event '{event.event_id}' already exists."
            )

        self.events[
            event.event_id
        ] = event

        return event

    def add_event(
        self,
        event: CalendarEvent,
    ) -> str:

        if not isinstance(
            event,
            CalendarEvent,
        ):
            raise TypeError(
                "event must be a CalendarEvent."
            )

        if event.event_id in self.events:
            raise ValueError(
                f"Event '{event.event_id}' already exists."
            )

        self.events[
            event.event_id
        ] = event

        return event.event_id

    def add_events(
        self,
        events: Iterable[CalendarEvent],
    ) -> list[str]:

        return [
            self.add_event(event)
            for event in events
        ]

    # ============================================================
    # EVENT RETRIEVAL
    # ============================================================

    def get_event(
        self,
        event_id: str,
    ) -> CalendarEvent | None:

        return self.events.get(
            event_id
        )

    def get_events(
        self,
    ) -> list[dict[str, Any]]:

        return [
            event.to_dict()
            for event in sorted(
                self.events.values(),
                key=lambda item: item.start_time,
            )
        ]

    def get_events_between(
        self,
        start: datetime,
        end: datetime,
    ) -> list[CalendarEvent]:

        if end < start:
            raise ValueError(
                "end cannot be before start."
            )

        return sorted(
            [
                event
                for event in self.events.values()
                if event.overlaps(
                    start,
                    end,
                )
            ],
            key=lambda event: event.start_time,
        )

    # ============================================================
    # TODAY / UPCOMING
    # ============================================================

    def get_today_events(
        self,
        now: datetime | None = None,
    ) -> list[CalendarEvent]:

        now = now or datetime.now()

        start = datetime(
            now.year,
            now.month,
            now.day,
        )

        end = start + timedelta(
            days=1
        )

        return self.get_events_between(
            start,
            end,
        )

    def get_upcoming_events(
        self,
        *,
        days: int = 7,
        now: datetime | None = None,
    ) -> list[CalendarEvent]:

        now = now or datetime.now()

        if days < 0:
            raise ValueError(
                "days cannot be negative."
            )

        end = now + timedelta(
            days=days
        )

        return sorted(
            [
                event
                for event in self.events.values()
                if event.start_time >= now
                and event.start_time <= end
            ],
            key=lambda event: event.start_time,
        )

    def get_next_event(
        self,
        now: datetime | None = None,
    ) -> CalendarEvent | None:

        now = now or datetime.now()

        upcoming = [
            event
            for event in self.events.values()
            if event.start_time >= now
        ]

        if not upcoming:
            return None

        return min(
            upcoming,
            key=lambda event: event.start_time,
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
    ) -> list[CalendarEvent]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        results = []

        for event in self.events.values():

            searchable = " ".join(
                [
                    event.title,
                    event.description,
                    event.location,
                    event.category,
                ]
            ).casefold()

            if query in searchable:
                results.append(event)

        return sorted(
            results,
            key=lambda event: event.start_time,
        )

    def search_category(
        self,
        category: str,
    ) -> list[CalendarEvent]:

        category = (
            category or ""
        ).strip().casefold()

        return sorted(
            [
                event
                for event in self.events.values()
                if category
                in event.category.casefold()
            ],
            key=lambda event: event.start_time,
        )

    # ============================================================
    # UPDATE
    # ============================================================

    def update_event(
        self,
        event_id: str,
        **changes: Any,
    ) -> bool:

        event = self.get_event(
            event_id
        )

        if event is None:
            return False

        allowed_fields = {
            "title",
            "start_time",
            "end_time",
            "description",
            "location",
            "category",
            "all_day",
            "recurring",
            "recurrence_rule",
            "reminder_minutes",
            "metadata",
        }

        for key, value in changes.items():

            if key not in allowed_fields:
                continue

            if key == "title":

                value = (
                    str(value)
                    .strip()
                )

                if not value:
                    continue

            setattr(
                event,
                key,
                value,
            )

        if (
            event.end_time is not None
            and event.end_time < event.start_time
        ):
            raise ValueError(
                "Event end time cannot be before start time."
            )

        return True

    def reschedule_event(
        self,
        event_id: str,
        start_time: datetime,
        *,
        end_time: datetime | None = None,
    ) -> bool:

        return self.update_event(
            event_id,
            start_time=start_time,
            end_time=end_time,
        )

    # ============================================================
    # DELETE
    # ============================================================

    def delete_event(
        self,
        event_id: str,
    ) -> bool:

        if event_id not in self.events:
            return False

        self.events.pop(
            event_id
        )

        return True

    def clear_events(self) -> None:

        self.events.clear()

    # ============================================================
    # CONFLICT DETECTION
    # ============================================================

    def check_conflicts(
        self,
        start: datetime,
        end: datetime,
        *,
        exclude_event_id: str | None = None,
    ) -> list[CalendarEvent]:

        return sorted(
            [
                event
                for event in self.events.values()
                if event.event_id
                != exclude_event_id
                and event.overlaps(
                    start,
                    end,
                )
            ],
            key=lambda event: event.start_time,
        )

    def has_conflict(
        self,
        start: datetime,
        end: datetime,
        *,
        exclude_event_id: str | None = None,
    ) -> bool:

        return bool(
            self.check_conflicts(
                start,
                end,
                exclude_event_id=exclude_event_id,
            )
        )

    # ============================================================
    # REMINDER INFORMATION
    # ============================================================

    def get_due_reminders(
        self,
        now: datetime | None = None,
    ) -> list[CalendarEvent]:

        now = now or datetime.now()

        due = []

        for event in self.events.values():

            if event.reminder_minutes is None:
                continue

            reminder_time = (
                event.start_time
                - timedelta(
                    minutes=event.reminder_minutes
                )
            )

            if (
                reminder_time <= now
                and event.start_time >= now
            ):
                due.append(event)

        return sorted(
            due,
            key=lambda event: event.start_time,
        )

    # ============================================================
    # DAILY SUMMARY
    # ============================================================

    def daily_summary(
        self,
        date: datetime | None = None,
    ) -> dict[str, Any]:

        date = date or datetime.now()

        start = datetime(
            date.year,
            date.month,
            date.day,
        )

        end = start + timedelta(
            days=1
        )

        events = self.get_events_between(
            start,
            end,
        )

        total_duration = sum(
            event.duration
            for event in events
        )

        return {
            "date": start.date().isoformat(),
            "event_count": len(events),
            "total_duration_seconds": (
                total_duration
            ),
            "events": [
                event.to_dict()
                for event in events
            ],
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "default_calendar": (
                self.default_calendar
            ),
            "events": self.get_events(),
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
                "Calendar state must be a dictionary."
            )

        self.events.clear()

        self.default_calendar = str(
            data.get(
                "default_calendar",
                "RENIX",
            )
        )

        raw_events = data.get(
            "events",
            [],
        )

        if not isinstance(
            raw_events,
            list,
        ):
            return

        for raw in raw_events:

            if not isinstance(
                raw,
                dict,
            ):
                continue

            try:

                start_time = datetime.fromisoformat(
                    raw["start_time"]
                )

                end_raw = raw.get(
                    "end_time"
                )

                end_time = (
                    datetime.fromisoformat(
                        end_raw
                    )
                    if end_raw
                    else None
                )

                event = CalendarEvent(
                    title=str(
                        raw.get(
                            "title",
                            "Untitled Event",
                        )
                    ),
                    start_time=start_time,
                    end_time=end_time,
                    description=str(
                        raw.get(
                            "description",
                            "",
                        )
                    ),
                    location=str(
                        raw.get(
                            "location",
                            "",
                        )
                    ),
                    category=str(
                        raw.get(
                            "category",
                            "",
                        )
                    ),
                    event_id=raw.get(
                        "event_id"
                    ),
                    all_day=bool(
                        raw.get(
                            "all_day",
                            False,
                        )
                    ),
                    recurring=bool(
                        raw.get(
                            "recurring",
                            False,
                        )
                    ),
                    recurrence_rule=raw.get(
                        "recurrence_rule"
                    ),
                    reminder_minutes=raw.get(
                        "reminder_minutes"
                    ),
                    metadata=dict(
                        raw.get(
                            "metadata",
                            {},
                        )
                    ),
                )

                self.events[
                    event.event_id
                ] = event

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue


__all__ = [
    "CalendarEvent",
    "CalendarManager",
]


