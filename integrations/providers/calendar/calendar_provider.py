"""
RENIX Calendar Provider Interface
=================================

Abstract provider contract for calendar integrations.

Concrete providers such as Google Calendar, Microsoft
Outlook, Apple Calendar, or a local calendar backend
should implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Iterable


class CalendarProvider(ABC):
    """
    Base interface for all RENIX calendar providers.
    """

    name: str = "unknown"

    # ========================================================
    # CALENDARS
    # ========================================================

    @abstractmethod
    def list_calendars(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return available calendars."""

        raise NotImplementedError

    @abstractmethod
    def get_calendar(
        self,
        calendar_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a specific calendar."""

        raise NotImplementedError

    # ========================================================
    # EVENTS
    # ========================================================

    @abstractmethod
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
        recurrence: Iterable[str] | None = None,
        all_day: bool = False,
        timezone: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a calendar event."""

        raise NotImplementedError

    @abstractmethod
    def get_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Retrieve a calendar event."""

        raise NotImplementedError

    @abstractmethod
    def update_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update a calendar event."""

        raise NotImplementedError

    @abstractmethod
    def delete_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Delete a calendar event."""

        raise NotImplementedError

    @abstractmethod
    def list_events(
        self,
        *,
        calendar_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return calendar events."""

        raise NotImplementedError

    # ========================================================
    # SEARCH
    # ========================================================

    def search_events(
        self,
        query: str,
        *,
        calendar_id: str | None = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search calendar events.

        Providers may override this for native search.
        """

        events = self.list_events(
            calendar_id=calendar_id,
            limit=limit,
            **kwargs,
        )

        query_lower = query.lower()

        results = []

        for event in events:

            title = str(
                event.get(
                    "title",
                    event.get(
                        "summary",
                        "",
                    ),
                )
            )

            description = str(
                event.get(
                    "description",
                    "",
                )
            )

            location = str(
                event.get(
                    "location",
                    "",
                )
            )

            searchable = " ".join(
                (
                    title,
                    description,
                    location,
                )
            ).lower()

            if query_lower in searchable:
                results.append(event)

        return results

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
    ) -> dict[str, Any]:
        """
        Return busy periods.

        Providers can override this with their native
        free/busy API.
        """

        events = self.list_events(
            start=start,
            end=end,
            limit=1000,
            **kwargs,
        )

        busy = []

        for event in events:

            event_start = event.get(
                "start"
            )

            event_end = event.get(
                "end"
            )

            if (
                event_start is None
                or event_end is None
            ):
                continue

            busy.append(
                {
                    "start": event_start,
                    "end": event_end,
                    "calendar_id": event.get(
                        "calendar_id"
                    ),
                }
            )

        return {
            "busy": busy,
        }

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ) -> bool:
        """
        Basic provider health check.

        Concrete providers can override this.
        """

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
        """
        Return supported calendar capabilities.
        """

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

    # ========================================================
    # METADATA
    # ========================================================

    def metadata(
        self,
    ) -> dict[str, Any]:
        """Return provider metadata."""

        return {
            "name": self.name,
            "type": "calendar",
            "capabilities": self.capabilities(),
        }


__all__ = [
    "CalendarProvider",
]


