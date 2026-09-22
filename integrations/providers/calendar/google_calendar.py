"""
RENIX Google Calendar Provider
==============================

Google Calendar implementation for the RENIX calendar layer.

This provider is intentionally isolated from the rest of RENIX so that
the application can use the common CalendarProvider interface regardless
of which calendar service is configured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from .calendar_provider import CalendarProvider


class GoogleCalendarProvider(CalendarProvider):
    """Google Calendar provider implementation."""

    name = "google_calendar"

    def __init__(
        self,
        *,
        credentials: Any = None,
        service: Any = None,
    ) -> None:
        self.credentials = credentials
        self.service = service

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _require_service(self) -> Any:
        """
        Return the Google Calendar API service.

        Authentication/client construction is intentionally kept outside
        this provider. A configured Google API service can be injected
        during application startup.
        """

        if self.service is None:
            raise RuntimeError(
                "Google Calendar API service is not configured."
            )

        return self.service

    @staticmethod
    def _event_to_dict(
        event: dict[str, Any],
        *,
        calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """Normalize a Google Calendar event."""

        start_data = event.get("start", {})
        end_data = event.get("end", {})

        return {
            "id": event.get("id"),
            "calendar_id": calendar_id,
            "title": event.get(
                "summary",
                "",
            ),
            "summary": event.get(
                "summary",
                "",
            ),
            "description": event.get(
                "description",
                "",
            ),
            "location": event.get(
                "location"
            ),
            "start": (
                start_data.get("dateTime")
                or start_data.get("date")
            ),
            "end": (
                end_data.get("dateTime")
                or end_data.get("date")
            ),
            "timezone": (
                start_data.get("timeZone")
            ),
            "status": event.get(
                "status"
            ),
            "html_link": event.get(
                "htmlLink"
            ),
            "created": event.get(
                "created"
            ),
            "updated": event.get(
                "updated"
            ),
            "recurrence": event.get(
                "recurrence",
                [],
            ),
            "attendees": event.get(
                "attendees",
                [],
            ),
            "reminders": event.get(
                "reminders",
                {},
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
        """Return calendars accessible to the authenticated account."""

        service = self._require_service()

        response = (
            service.calendarList()
            .list(
                **kwargs
            )
            .execute()
        )

        return response.get(
            "items",
            [],
        )

    def get_calendar(
        self,
        calendar_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a Google calendar."""

        service = self._require_service()

        return (
            service.calendars()
            .get(
                calendarId=calendar_id,
                **kwargs,
            )
            .execute()
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
        recurrence: Iterable[str] | None = None,
        all_day: bool = False,
        timezone: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create an event in Google Calendar."""

        service = self._require_service()

        if end <= start:
            raise ValueError(
                "Event end must be after event start."
            )

        if all_day:
            event_start = {
                "date": start.date().isoformat()
            }

            event_end = {
                "date": end.date().isoformat()
            }

        else:
            event_start = {
                "dateTime": start.isoformat()
            }

            event_end = {
                "dateTime": end.isoformat()
            }

            if timezone:
                event_start["timeZone"] = timezone
                event_end["timeZone"] = timezone

        body: dict[str, Any] = {
            "summary": title,
            "description": description,
            "start": event_start,
            "end": event_end,
        }

        if location:
            body["location"] = location

        if attendees:
            body["attendees"] = [
                self._normalize_attendee(
                    attendee
                )
                for attendee in attendees
            ]

        if reminders:
            body["reminders"] = {
                "useDefault": False,
                "overrides": list(
                    reminders
                ),
            }

        if recurrence:
            body["recurrence"] = list(
                recurrence
            )

        body.update(
            kwargs.pop(
                "body",
                {},
            )
        )

        response = (
            service.events()
            .insert(
                calendarId=calendar_id,
                body=body,
                **kwargs,
            )
            .execute()
        )

        return self._event_to_dict(
            response,
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
        """Retrieve a Google Calendar event."""

        service = self._require_service()

        calendar_id = (
            calendar_id
            or "primary"
        )

        response = (
            service.events()
            .get(
                calendarId=calendar_id,
                eventId=event_id,
                **kwargs,
            )
            .execute()
        )

        return self._event_to_dict(
            response,
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
        """Update a Google Calendar event."""

        service = self._require_service()

        calendar_id = (
            calendar_id
            or "primary"
        )

        existing = (
            service.events()
            .get(
                calendarId=calendar_id,
                eventId=event_id,
            )
            .execute()
        )

        body = dict(
            existing
        )

        title = kwargs.pop(
            "title",
            None,
        )

        if title is not None:
            body["summary"] = title

        if "description" in kwargs:
            body["description"] = kwargs.pop(
                "description"
            )

        if "location" in kwargs:
            body["location"] = kwargs.pop(
                "location"
            )

        start = kwargs.pop(
            "start",
            None,
        )

        end = kwargs.pop(
            "end",
            None,
        )

        timezone = kwargs.pop(
            "timezone",
            None,
        )

        if start is not None:

            body["start"] = self._build_datetime(
                start,
                timezone=timezone,
                existing=body.get(
                    "start",
                    {},
                ),
            )

        if end is not None:

            body["end"] = self._build_datetime(
                end,
                timezone=timezone,
                existing=body.get(
                    "end",
                    {},
                ),
            )

        if "attendees" in kwargs:

            attendees = kwargs.pop(
                "attendees"
            )

            body["attendees"] = [
                self._normalize_attendee(
                    attendee
                )
                for attendee in attendees
            ]

        if "recurrence" in kwargs:

            recurrence = kwargs.pop(
                "recurrence"
            )

            body["recurrence"] = list(
                recurrence
            )

        body.update(
            kwargs.pop(
                "body",
                {},
            )
        )

        response = (
            service.events()
            .update(
                calendarId=calendar_id,
                eventId=event_id,
                body=body,
                **kwargs,
            )
            .execute()
        )

        return self._event_to_dict(
            response,
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
        """Delete a Google Calendar event."""

        service = self._require_service()

        calendar_id = (
            calendar_id
            or "primary"
        )

        (
            service.events()
            .delete(
                calendarId=calendar_id,
                eventId=event_id,
                **kwargs,
            )
            .execute()
        )

        return True

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
        """Return events in a time range."""

        service = self._require_service()

        calendar_id = (
            calendar_id
            or "primary"
        )

        request_kwargs: dict[str, Any] = {
            "calendarId": calendar_id,
            "maxResults": max(
                1,
                min(
                    int(limit),
                    2500,
                ),
            ),
            "singleEvents": kwargs.pop(
                "singleEvents",
                True,
            ),
            "orderBy": kwargs.pop(
                "orderBy",
                "startTime",
            ),
        }

        if start is not None:
            request_kwargs["timeMin"] = (
                start.isoformat()
            )

        if end is not None:
            request_kwargs["timeMax"] = (
                end.isoformat()
            )

        request_kwargs.update(
            kwargs
        )

        response = (
            service.events()
            .list(
                **request_kwargs
            )
            .execute()
        )

        return [
            self._event_to_dict(
                event,
                calendar_id=calendar_id,
            )
            for event in response.get(
                "items",
                [],
            )
        ]

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
        """Query Google Calendar free/busy information."""

        service = self._require_service()

        if end <= start:
            raise ValueError(
                "end must be after start."
            )

        ids = list(
            calendar_ids
            or ["primary"]
        )

        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "items": [
                {
                    "id": calendar_id
                }
                for calendar_id in ids
            ],
        }

        body.update(
            kwargs.pop(
                "body",
                {},
            )
        )

        response = (
            service.freebusy()
            .query(
                body=body,
                **kwargs,
            )
            .execute()
        )

        return response.get(
            "calendars",
            {},
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _normalize_attendee(
        attendee: Any,
    ) -> dict[str, Any]:
        """Normalize an attendee into Google Calendar format."""

        if isinstance(
            attendee,
            str,
        ):
            return {
                "email": attendee
            }

        if isinstance(
            attendee,
            dict,
        ):
            return dict(
                attendee
            )

        raise TypeError(
            "Attendees must be email strings or dictionaries."
        )

    @staticmethod
    def _build_datetime(
        value: datetime,
        *,
        timezone: str | None = None,
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a Google Calendar date/time object."""

        if not isinstance(
            value,
            datetime,
        ):
            raise TypeError(
                "Calendar date/time must be a datetime."
            )

        existing = existing or {}

        result: dict[str, Any] = {
            "dateTime": value.isoformat()
        }

        tz = (
            timezone
            or existing.get(
                "timeZone"
            )
        )

        if tz:
            result["timeZone"] = tz

        return result

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ) -> bool:
        """Check whether Google Calendar is reachable."""

        try:
            self.list_calendars(
                maxResults=1
            )
            return True

        except Exception:
            return False

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def capabilities(
        self,
    ) -> dict[str, bool]:
        """Return Google Calendar capabilities."""

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
    "GoogleCalendarProvider",
]


