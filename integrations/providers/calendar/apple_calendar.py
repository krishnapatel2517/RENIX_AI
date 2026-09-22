"""
RENIX Apple Calendar Provider
=============================

Apple Calendar integration layer.

This provider is designed around an injected adapter so RENIX does not
depend directly on macOS-specific frameworks. The adapter can be backed
by EventKit, AppleScript, CalDAV, or another implementation.

Expected adapter methods are intentionally lightweight:
    - list_calendars()
    - get_calendar(calendar_id)
    - create_event(...)
    - get_event(...)
    - update_event(...)
    - delete_event(...)
    - list_events(...)
    - free_busy(...)

This keeps the RENIX calendar architecture provider-agnostic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from .calendar_provider import CalendarProvider


class AppleCalendarProvider(CalendarProvider):
    """Apple Calendar provider using an injected adapter."""

    name = "apple_calendar"

    def __init__(
        self,
        *,
        adapter: Any = None,
    ) -> None:
        self.adapter = adapter

    # ========================================================
    # INTERNAL
    # ========================================================

    def _require_adapter(self) -> Any:
        """Return the configured Apple Calendar adapter."""

        if self.adapter is None:
            raise RuntimeError(
                "Apple Calendar adapter is not configured."
            )

        return self.adapter

    def _call(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a method on the configured adapter."""

        adapter = self._require_adapter()

        function = getattr(
            adapter,
            method,
            None,
        )

        if not callable(function):
            raise RuntimeError(
                f"Apple Calendar adapter does not implement "
                f"'{method}'."
            )

        return function(
            *args,
            **kwargs,
        )

    @staticmethod
    def _normalize_calendar(
        calendar: Any,
    ) -> dict[str, Any]:
        """Normalize an Apple calendar."""

        if isinstance(
            calendar,
            dict,
        ):
            data = dict(calendar)

        else:
            data = {
                "id": getattr(
                    calendar,
                    "id",
                    None,
                ),
                "name": getattr(
                    calendar,
                    "name",
                    "",
                ),
                "description": getattr(
                    calendar,
                    "description",
                    "",
                ),
                "color": getattr(
                    calendar,
                    "color",
                    None,
                ),
            }

        return {
            "id": data.get("id"),
            "name": data.get(
                "name",
                "",
            ),
            "description": data.get(
                "description",
                "",
            ),
            "color": data.get(
                "color"
            ),
            "raw": calendar,
        }

    @staticmethod
    def _normalize_event(
        event: Any,
        *,
        calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """Normalize an Apple calendar event."""

        if isinstance(
            event,
            dict,
        ):
            data = dict(event)

        else:
            data = {
                "id": getattr(
                    event,
                    "id",
                    None,
                ),
                "title": getattr(
                    event,
                    "title",
                    "",
                ),
                "description": getattr(
                    event,
                    "description",
                    "",
                ),
                "location": getattr(
                    event,
                    "location",
                    None,
                ),
                "start": getattr(
                    event,
                    "start",
                    None,
                ),
                "end": getattr(
                    event,
                    "end",
                    None,
                ),
                "calendar_id": getattr(
                    event,
                    "calendar_id",
                    calendar_id,
                ),
                "all_day": getattr(
                    event,
                    "all_day",
                    False,
                ),
                "url": getattr(
                    event,
                    "url",
                    None,
                ),
                "recurrence": getattr(
                    event,
                    "recurrence",
                    None,
                ),
                "attendees": getattr(
                    event,
                    "attendees",
                    [],
                ),
            }

        return {
            "id": data.get("id"),
            "calendar_id": (
                data.get(
                    "calendar_id"
                )
                or calendar_id
            ),
            "title": data.get(
                "title",
                data.get(
                    "summary",
                    "",
                ),
            ),
            "summary": data.get(
                "summary",
                data.get(
                    "title",
                    "",
                ),
            ),
            "description": data.get(
                "description",
                "",
            ),
            "location": data.get(
                "location"
            ),
            "start": data.get(
                "start"
            ),
            "end": data.get(
                "end"
            ),
            "timezone": data.get(
                "timezone"
            ),
            "is_all_day": data.get(
                "is_all_day",
                data.get(
                    "all_day",
                    False,
                ),
            ),
            "status": data.get(
                "status"
            ),
            "url": data.get(
                "url"
            ),
            "recurrence": data.get(
                "recurrence"
            ),
            "attendees": data.get(
                "attendees",
                [],
            ),
            "reminders": data.get(
                "reminders",
                [],
            ),
            "created": data.get(
                "created"
            ),
            "updated": data.get(
                "updated"
            ),
            "raw": event,
        }

    # ========================================================
    # CALENDARS
    # ========================================================

    def list_calendars(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available Apple calendars."""

        calendars = self._call(
            "list_calendars",
            **kwargs,
        )

        return [
            self._normalize_calendar(
                calendar
            )
            for calendar in (
                calendars or []
            )
        ]

    def get_calendar(
        self,
        calendar_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a single Apple calendar."""

        calendar = self._call(
            "get_calendar",
            calendar_id,
            **kwargs,
        )

        return self._normalize_calendar(
            calendar
        )

    # ========================================================
    # CREATE EVENT
    # ========================================================

    def create_event(
        self,
        calendar_id: str,
        title: str,
        *,
        start: datetime,
        end: datetime,
        description: str = "",
        location: str | None = None,
        attendees: Iterable[Any] | None = None,
        reminders: Iterable[Any] | None = None,
        recurrence: Iterable[Any] | None = None,
        all_day: bool = False,
        timezone: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create an Apple Calendar event."""

        if end <= start:
            raise ValueError(
                "Event end must be after event start."
            )

        event = self._call(
            "create_event",
            calendar_id=calendar_id,
            title=title,
            start=start,
            end=end,
            description=description,
            location=location,
            attendees=attendees,
            reminders=reminders,
            recurrence=recurrence,
            all_day=all_day,
            timezone=timezone,
            **kwargs,
        )

        return self._normalize_event(
            event,
            calendar_id=calendar_id,
        )

    # ========================================================
    # GET EVENT
    # ========================================================

    def get_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Retrieve an Apple Calendar event."""

        event = self._call(
            "get_event",
            event_id,
            calendar_id=calendar_id,
            **kwargs,
        )

        return self._normalize_event(
            event,
            calendar_id=calendar_id,
        )

    # ========================================================
    # UPDATE EVENT
    # ========================================================

    def update_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update an Apple Calendar event."""

        if (
            "start" in kwargs
            and "end" in kwargs
        ):
            start = kwargs["start"]
            end = kwargs["end"]

            if (
                isinstance(
                    start,
                    datetime,
                )
                and isinstance(
                    end,
                    datetime,
                )
                and end <= start
            ):
                raise ValueError(
                    "Event end must be after event start."
                )

        event = self._call(
            "update_event",
            event_id,
            calendar_id=calendar_id,
            **kwargs,
        )

        return self._normalize_event(
            event,
            calendar_id=calendar_id,
        )

    # ========================================================
    # DELETE EVENT
    # ========================================================

    def delete_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Delete an Apple Calendar event."""

        result = self._call(
            "delete_event",
            event_id,
            calendar_id=calendar_id,
            **kwargs,
        )

        if result is None:
            return True

        return bool(result)

    # ========================================================
    # LIST EVENTS
    # ========================================================

    def list_events(
        self,
        *,
        calendar_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List Apple Calendar events."""

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        if (
            start is not None
            and end is not None
            and end <= start
        ):
            raise ValueError(
                "end must be after start."
            )

        events = self._call(
            "list_events",
            calendar_id=calendar_id,
            start=start,
            end=end,
            limit=limit,
            **kwargs,
        )

        normalized = [
            self._normalize_event(
                event,
                calendar_id=calendar_id,
            )
            for event in (
                events or []
            )
        ]

        return normalized[:limit]

    # ========================================================
    # SEARCH EVENTS
    # ========================================================

    def search_events(
        self,
        query: str,
        *,
        calendar_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search Apple Calendar events."""

        if not query.strip():
            return []

        adapter = self._require_adapter()

        search_method = getattr(
            adapter,
            "search_events",
            None,
        )

        if callable(search_method):

            events = search_method(
                query,
                calendar_id=calendar_id,
                start=start,
                end=end,
                limit=limit,
                **kwargs,
            )

        else:

            events = self.list_events(
                calendar_id=calendar_id,
                start=start,
                end=end,
                limit=limit,
                **kwargs,
            )

            query_lower = query.lower()

            events = [
                event
                for event in events
                if query_lower
                in (
                    str(
                        event.get(
                            "title",
                            "",
                        )
                    )
                    + " "
                    + str(
                        event.get(
                            "description",
                            "",
                        )
                    )
                    + " "
                    + str(
                        event.get(
                            "location",
                            "",
                        )
                    )
                ).lower()
            ]

        return [
            self._normalize_event(
                event,
                calendar_id=calendar_id,
            )
            for event in (
                events or []
            )
        ][:limit]

    # ========================================================
    # FREE / BUSY
    # ========================================================

    def free_busy(
        self,
        *,
        start: datetime,
        end: datetime,
        calendar_ids: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return Apple Calendar free/busy information."""

        if end <= start:
            raise ValueError(
                "end must be after start."
            )

        return self._call(
            "free_busy",
            start=start,
            end=end,
            calendar_ids=list(
                calendar_ids
                or []
            ),
            **kwargs,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ) -> bool:
        """Check whether the Apple Calendar adapter works."""

        try:
            self.list_calendars()
            return True

        except Exception:
            return False

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
    ) -> dict[str, bool]:
        """Return supported Apple Calendar operations."""

        return {
            "list_calendars": True,
            "get_calendar": True,
            "create_event": True,
            "get_event": True,
            "update_event": True,
            "delete_event": True,
            "list_events": True,
            "search_events": True,
            "free_busy": True,
            "recurring_events": True,
            "reminders": True,
            "attendees": True,
        }


__all__ = [
    "AppleCalendarProvider",
]


