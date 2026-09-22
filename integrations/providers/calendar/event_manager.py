"""
RENIX Calendar Event Manager
============================

High-level event management utilities built on top of
CalendarService.

Responsibilities:
    - Create and update events
    - Quick event creation
    - Find events
    - Cancel events
    - Reschedule events
    - Detect conflicts
    - Find free time
    - Handle recurring event helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from .calendar_service import (
    CalendarEvent,
    CalendarService,
    FreeBusyPeriod,
)


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class TimeSlot:
    """Represents a free time slot."""

    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


@dataclass
class Conflict:
    """Represents a calendar conflict."""

    event: CalendarEvent
    overlap_start: datetime
    overlap_end: datetime

    @property
    def duration(self) -> timedelta:
        return (
            self.overlap_end
            - self.overlap_start
        )


# ============================================================
# EVENT MANAGER
# ============================================================


class EventManager:
    """
    High-level event management layer.

    Example:

        manager = EventManager(calendar_service)

        event = manager.quick_create(
            calendar_id="primary",
            title="Study",
            start=start,
            duration_minutes=60,
        )
    """

    def __init__(
        self,
        calendar_service: CalendarService,
    ) -> None:

        if calendar_service is None:
            raise ValueError(
                "calendar_service is required."
            )

        self.calendar = calendar_service

    # ========================================================
    # QUICK CREATE
    # ========================================================

    def quick_create(
        self,
        calendar_id: str,
        title: str,
        *,
        start: datetime,
        duration_minutes: int = 60,
        description: str = "",
        location: str | None = None,
        attendees: Iterable[Any] | None = None,
        reminders: Iterable[Any] | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> CalendarEvent:
        """
        Create an event using a start time and duration.
        """

        if duration_minutes <= 0:
            raise ValueError(
                "duration_minutes must be greater than zero."
            )

        end = start + timedelta(
            minutes=duration_minutes
        )

        return self.calendar.create_event(
            calendar_id=calendar_id,
            title=title,
            start=start,
            end=end,
            description=description,
            location=location,
            attendees=attendees,
            reminders=reminders,
            provider=provider,
            **kwargs,
        )

    # ========================================================
    # RESCHEDULE
    # ========================================================

    def reschedule(
        self,
        event_id: str,
        *,
        new_start: datetime,
        new_end: datetime | None = None,
        duration_minutes: int | None = None,
        calendar_id: str | None = None,
        provider: str | None = None,
        check_conflicts: bool = True,
        **kwargs: Any,
    ) -> CalendarEvent:
        """
        Reschedule an existing event.

        Either provide new_end or duration_minutes.
        """

        if new_end is None:

            if duration_minutes is None:

                existing = self.calendar.get_event(
                    event_id,
                    calendar_id=calendar_id,
                    provider=provider,
                )

                if (
                    existing.start is None
                    or existing.end is None
                ):
                    raise ValueError(
                        "Existing event has no usable duration."
                    )

                duration = (
                    existing.end
                    - existing.start
                )

                new_end = (
                    new_start
                    + duration
                )

            else:

                if duration_minutes <= 0:
                    raise ValueError(
                        "duration_minutes must be greater than zero."
                    )

                new_end = (
                    new_start
                    + timedelta(
                        minutes=duration_minutes
                    )
                )

        if new_end <= new_start:
            raise ValueError(
                "new_end must be after new_start."
            )

        if check_conflicts:

            conflicts = self.find_conflicts(
                start=new_start,
                end=new_end,
                calendar_id=calendar_id,
                provider=provider,
                exclude_event_id=event_id,
            )

            if conflicts:
                raise ValueError(
                    "The new event time conflicts with "
                    "another calendar event."
                )

        return self.calendar.update_event(
            event_id,
            calendar_id=calendar_id,
            start=new_start,
            end=new_end,
            provider=provider,
            **kwargs,
        )

    # ========================================================
    # CANCEL
    # ========================================================

    def cancel(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        provider: str | None = None,
    ) -> bool:
        """Cancel/delete an event."""

        return self.calendar.delete_event(
            event_id,
            calendar_id=calendar_id,
            provider=provider,
        )

    # ========================================================
    # FIND
    # ========================================================

    def find(
        self,
        query: str,
        *,
        calendar_id: str | None = None,
        provider: str | None = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[CalendarEvent]:
        """Search for events."""

        return self.calendar.search_events(
            query,
            calendar_id=calendar_id,
            provider=provider,
            limit=limit,
            **kwargs,
        )

    # ========================================================
    # CONFLICT DETECTION
    # ========================================================

    def find_conflicts(
        self,
        *,
        start: datetime,
        end: datetime,
        calendar_id: str | None = None,
        provider: str | None = None,
        exclude_event_id: str | None = None,
    ) -> list[Conflict]:
        """
        Find events overlapping a requested time range.
        """

        if end <= start:
            raise ValueError(
                "end must be after start."
            )

        events = self.calendar.list_events(
            calendar_id=calendar_id,
            start=start,
            end=end,
            provider=provider,
            limit=1000,
        )

        conflicts: list[Conflict] = []

        for event in events:

            if (
                exclude_event_id
                and event.id
                == exclude_event_id
            ):
                continue

            if (
                event.start is None
                or event.end is None
            ):
                continue

            overlap_start = max(
                start,
                event.start,
            )

            overlap_end = min(
                end,
                event.end,
            )

            if overlap_start < overlap_end:

                conflicts.append(
                    Conflict(
                        event=event,
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                    )
                )

        conflicts.sort(
            key=lambda item: item.overlap_start
        )

        return conflicts

    # ========================================================
    # IS FREE
    # ========================================================

    def is_free(
        self,
        *,
        start: datetime,
        end: datetime,
        calendar_id: str | None = None,
        provider: str | None = None,
        exclude_event_id: str | None = None,
    ) -> bool:
        """Return True when a time range has no conflicts."""

        return not self.find_conflicts(
            start=start,
            end=end,
            calendar_id=calendar_id,
            provider=provider,
            exclude_event_id=exclude_event_id,
        )

    # ========================================================
    # FIND FREE SLOTS
    # ========================================================

    def find_free_slots(
        self,
        *,
        start: datetime,
        end: datetime,
        duration_minutes: int,
        calendar_id: str | None = None,
        provider: str | None = None,
        working_start_hour: int = 0,
        working_end_hour: int = 24,
    ) -> list[TimeSlot]:
        """
        Find available slots inside a time range.

        This operates on calendar events and returns gaps
        large enough for the requested duration.
        """

        if end <= start:
            raise ValueError(
                "end must be after start."
            )

        if duration_minutes <= 0:
            raise ValueError(
                "duration_minutes must be greater than zero."
            )

        if not (
            0 <= working_start_hour <= 23
        ):
            raise ValueError(
                "working_start_hour must be between 0 and 23."
            )

        if not (
            1 <= working_end_hour <= 24
        ):
            raise ValueError(
                "working_end_hour must be between 1 and 24."
            )

        if working_start_hour >= working_end_hour:
            raise ValueError(
                "working_start_hour must be before "
                "working_end_hour."
            )

        events = self.calendar.list_events(
            calendar_id=calendar_id,
            start=start,
            end=end,
            provider=provider,
            limit=1000,
        )

        busy_periods = []

        for event in events:

            if (
                event.start is None
                or event.end is None
            ):
                continue

            busy_start = max(
                start,
                event.start,
            )

            busy_end = min(
                end,
                event.end,
            )

            if busy_start < busy_end:
                busy_periods.append(
                    (
                        busy_start,
                        busy_end,
                    )
                )

        merged = self._merge_periods(
            busy_periods
        )

        slots: list[TimeSlot] = []

        day = start.date()

        while day <= end.date():

            day_start = datetime(
                day.year,
                day.month,
                day.day,
                working_start_hour,
                tzinfo=start.tzinfo,
            )

            if working_end_hour == 24:

                day_end = (
                    day_start
                    + timedelta(
                        hours=24
                        - working_start_hour
                    )
                )

            else:

                day_end = datetime(
                    day.year,
                    day.month,
                    day.day,
                    working_end_hour,
                    tzinfo=start.tzinfo,
                )

            window_start = max(
                start,
                day_start,
            )

            window_end = min(
                end,
                day_end,
            )

            if window_start < window_end:

                cursor = window_start

                for busy_start, busy_end in merged:

                    if busy_end <= cursor:
                        continue

                    if busy_start >= window_end:
                        break

                    free_end = min(
                        busy_start,
                        window_end,
                    )

                    if (
                        free_end
                        - cursor
                        >= timedelta(
                            minutes=duration_minutes
                        )
                    ):
                        slots.append(
                            TimeSlot(
                                start=cursor,
                                end=free_end,
                            )
                        )

                    cursor = max(
                        cursor,
                        busy_end,
                    )

                    if cursor >= window_end:
                        break

                if (
                    window_end
                    - cursor
                    >= timedelta(
                        minutes=duration_minutes
                    )
                ):
                    slots.append(
                        TimeSlot(
                            start=cursor,
                            end=window_end,
                        )
                    )

            day += timedelta(
                days=1
            )

        return slots

    # ========================================================
    # NEXT AVAILABLE
    # ========================================================

    def next_available(
        self,
        *,
        start: datetime,
        duration_minutes: int,
        search_days: int = 7,
        calendar_id: str | None = None,
        provider: str | None = None,
        working_start_hour: int = 8,
        working_end_hour: int = 22,
    ) -> TimeSlot | None:
        """
        Find the next available slot.
        """

        end = start + timedelta(
            days=max(
                1,
                search_days,
            )
        )

        slots = self.find_free_slots(
            start=start,
            end=end,
            duration_minutes=duration_minutes,
            calendar_id=calendar_id,
            provider=provider,
            working_start_hour=working_start_hour,
            working_end_hour=working_end_hour,
        )

        if not slots:
            return None

        return slots[0]

    # ========================================================
    # TODAY
    # ========================================================

    def today(
        self,
        *,
        calendar_id: str | None = None,
        provider: str | None = None,
    ) -> list[CalendarEvent]:
        """Return today's events."""

        return self.calendar.today(
            calendar_id=calendar_id,
            provider=provider,
        )

    # ========================================================
    # UPCOMING
    # ========================================================

    def upcoming(
        self,
        *,
        days: int = 7,
        calendar_id: str | None = None,
        provider: str | None = None,
        limit: int = 100,
    ) -> list[CalendarEvent]:
        """Return upcoming events."""

        return self.calendar.upcoming(
            days=days,
            calendar_id=calendar_id,
            provider=provider,
            limit=limit,
        )

    # ========================================================
    # PERIOD MERGING
    # ========================================================

    @staticmethod
    def _merge_periods(
        periods: list[
            tuple[
                datetime,
                datetime,
            ]
        ],
    ) -> list[
        tuple[
            datetime,
            datetime,
        ]
    ]:
        """Merge overlapping busy periods."""

        if not periods:
            return []

        sorted_periods = sorted(
            periods,
            key=lambda period: period[0],
        )

        merged = [
            sorted_periods[0]
        ]

        for current_start, current_end in (
            sorted_periods[1:]
        ):

            previous_start, previous_end = (
                merged[-1]
            )

            if current_start <= previous_end:

                merged[-1] = (
                    previous_start,
                    max(
                        previous_end,
                        current_end,
                    ),
                )

            else:

                merged.append(
                    (
                        current_start,
                        current_end,
                    )
                )

        return merged


__all__ = [
    "TimeSlot",
    "Conflict",
    "EventManager",
]


