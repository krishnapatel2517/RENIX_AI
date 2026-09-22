"""
RENIX Outlook Calendar Provider
===============================

Microsoft Outlook / Microsoft Graph Calendar provider.

This module implements RENIX's common CalendarProvider interface
using an injected Microsoft Graph client.

Authentication and token acquisition are intentionally kept outside
this provider so the integration remains modular.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from .calendar_provider import CalendarProvider


class OutlookCalendarProvider(CalendarProvider):
    """Microsoft Outlook Calendar provider."""

    name = "outlook_calendar"

    def __init__(
        self,
        *,
        client: Any = None,
        user_id: str = "me",
    ) -> None:
        self.client = client
        self.user_id = user_id

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _require_client(self) -> Any:
        """Return the configured Microsoft Graph client."""

        if self.client is None:
            raise RuntimeError(
                "Microsoft Graph client is not configured."
            )

        return self.client

    def _user_path(self) -> str:
        """Return the Graph user path."""

        if self.user_id == "me":
            return "/me"

        return f"/users/{self.user_id}"

    @staticmethod
    def _response_json(
        response: Any,
    ) -> dict[str, Any]:
        """Normalize a Graph SDK/raw response."""

        if response is None:
            return {}

        if isinstance(
            response,
            dict,
        ):
            return response

        json_method = getattr(
            response,
            "json",
            None,
        )

        if callable(json_method):
            result = json_method()

            if isinstance(
                result,
                dict,
            ):
                return result

        body = getattr(
            response,
            "value",
            None,
        )

        if body is not None:
            return {
                "value": body
            }

        return {}

    @staticmethod
    def _normalize_calendar(
        calendar: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize Outlook calendar data."""

        return {
            "id": calendar.get("id"),
            "name": calendar.get(
                "name",
                "",
            ),
            "description": calendar.get(
                "description",
                "",
            ),
            "color": calendar.get(
                "color"
            ),
            "is_default": calendar.get(
                "isDefaultCalendar",
                False,
            ),
            "can_edit": calendar.get(
                "canEdit",
                False,
            ),
            "raw": calendar,
        }

    @staticmethod
    def _normalize_event(
        event: dict[str, Any],
        *,
        calendar_id: str | None = None,
    ) -> dict[str, Any]:
        """Normalize Outlook event data."""

        start = event.get(
            "start",
            {},
        )

        end = event.get(
            "end",
            {},
        )

        return {
            "id": event.get("id"),
            "calendar_id": calendar_id,
            "title": event.get(
                "subject",
                "",
            ),
            "summary": event.get(
                "subject",
                "",
            ),
            "description": (
                event.get(
                    "body",
                    {},
                ).get(
                    "content",
                    "",
                )
                if isinstance(
                    event.get(
                        "body",
                        {},
                    ),
                    dict,
                )
                else ""
            ),
            "location": (
                event.get(
                    "location",
                    {},
                ).get(
                    "displayName"
                )
                if isinstance(
                    event.get(
                        "location",
                        {},
                    ),
                    dict,
                )
                else event.get(
                    "location"
                )
            ),
            "start": start.get(
                "dateTime"
            ),
            "end": end.get(
                "dateTime"
            ),
            "timezone": start.get(
                "timeZone"
            ),
            "status": event.get(
                "showAs"
            ),
            "is_all_day": event.get(
                "isAllDay",
                False,
            ),
            "recurrence": event.get(
                "recurrence"
            ),
            "attendees": event.get(
                "attendees",
                [],
            ),
            "reminders": event.get(
                "reminderMinutesBeforeStart"
            ),
            "web_link": event.get(
                "webLink"
            ),
            "created": event.get(
                "createdDateTime"
            ),
            "updated": event.get(
                "lastModifiedDateTime"
            ),
            "raw": event,
        }

    @staticmethod
    def _attendee(
        attendee: Any,
    ) -> dict[str, Any]:
        """Convert an attendee into Graph format."""

        if isinstance(
            attendee,
            str,
        ):
            email = attendee

            return {
                "emailAddress": {
                    "address": email,
                    "name": email,
                },
                "type": "required",
            }

        if isinstance(
            attendee,
            dict,
        ):
            if "emailAddress" in attendee:
                return dict(attendee)

            email = attendee.get(
                "email"
            )

            if email:
                return {
                    "emailAddress": {
                        "address": email,
                        "name": attendee.get(
                            "name",
                            email,
                        ),
                    },
                    "type": attendee.get(
                        "type",
                        "required",
                    ),
                }

        raise TypeError(
            "Attendees must be email strings or dictionaries."
        )

    # ========================================================
    # GENERIC REQUEST
    # ========================================================

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a request against an injected Graph client.

        Supports clients exposing request()/get()/post()/patch()/delete().
        """

        client = self._require_client()

        method = method.upper()

        request_method = getattr(
            client,
            method.lower(),
            None,
        )

        if callable(request_method):

            kwargs: dict[str, Any] = {}

            if body is not None:
                kwargs["json"] = body

            if params is not None:
                kwargs["params"] = params

            response = request_method(
                path,
                **kwargs,
            )

            return self._response_json(
                response
            )

        request = getattr(
            client,
            "request",
            None,
        )

        if callable(request):

            response = request(
                method,
                path,
                json=body,
                params=params,
            )

            return self._response_json(
                response
            )

        raise RuntimeError(
            "Unsupported Microsoft Graph client. "
            "Provide a client exposing HTTP methods."
        )

    # ========================================================
    # CALENDARS
    # ========================================================

    def list_calendars(
        self,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return Outlook calendars."""

        params = dict(
            kwargs
        )

        response = self._request(
            "GET",
            f"{self._user_path()}/calendars",
            params=params or None,
        )

        return [
            self._normalize_calendar(
                calendar
            )
            for calendar in response.get(
                "value",
                [],
            )
        ]

    def get_calendar(
        self,
        calendar_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return one Outlook calendar."""

        response = self._request(
            "GET",
            f"{self._user_path()}/calendars/{calendar_id}",
            params=kwargs or None,
        )

        return self._normalize_calendar(
            response
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
        """Create an Outlook event."""

        if end <= start:
            raise ValueError(
                "Event end must be after event start."
            )

        body: dict[str, Any] = {
            "subject": title,
            "body": {
                "contentType": "text",
                "content": description,
            },
            "start": {
                "dateTime": start.isoformat(),
                "timeZone": timezone or "UTC",
            },
            "end": {
                "dateTime": end.isoformat(),
                "timeZone": timezone or "UTC",
            },
            "isAllDay": all_day,
        }

        if location:
            body["location"] = {
                "displayName": location
            }

        if attendees:
            body["attendees"] = [
                self._attendee(
                    attendee
                )
                for attendee in attendees
            ]

        if reminders:

            reminder_list = list(
                reminders
            )

            if reminder_list:
                first = reminder_list[0]

                if isinstance(
                    first,
                    dict,
                ):
                    minutes = first.get(
                        "minutesBeforeStart",
                        15,
                    )
                else:
                    minutes = int(
                        first
                    )

                body[
                    "reminderMinutesBeforeStart"
                ] = minutes

                body[
                    "isReminderOn"
                ] = True

        if recurrence:
            recurrence_values = list(
                recurrence
            )

            if recurrence_values:
                body[
                    "recurrence"
                ] = recurrence_values

        body.update(
            kwargs.pop(
                "body",
                {},
            )
        )

        response = self._request(
            "POST",
            f"{self._user_path()}/calendars/"
            f"{calendar_id}/events",
            body=body,
        )

        return self._normalize_event(
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
        """Retrieve an Outlook event."""

        path = (
            f"{self._user_path()}/events/{event_id}"
            if calendar_id is None
            else
            f"{self._user_path()}/calendars/"
            f"{calendar_id}/events/{event_id}"
        )

        response = self._request(
            "GET",
            path,
            params=kwargs or None,
        )

        return self._normalize_event(
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
        """Update an Outlook event."""

        body: dict[str, Any] = {}

        title = kwargs.pop(
            "title",
            None,
        )

        if title is not None:
            body["subject"] = title

        if "description" in kwargs:

            body["body"] = {
                "contentType": "text",
                "content": kwargs.pop(
                    "description"
                ),
            }

        if "location" in kwargs:

            body["location"] = {
                "displayName": kwargs.pop(
                    "location"
                )
            }

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
            "UTC",
        )

        if start is not None:

            body["start"] = {
                "dateTime": start.isoformat(),
                "timeZone": timezone,
            }

        if end is not None:

            body["end"] = {
                "dateTime": end.isoformat(),
                "timeZone": timezone,
            }

        if "attendees" in kwargs:

            body["attendees"] = [
                self._attendee(
                    attendee
                )
                for attendee in kwargs.pop(
                    "attendees"
                )
            ]

        if "reminder_minutes" in kwargs:

            body[
                "reminderMinutesBeforeStart"
            ] = int(
                kwargs.pop(
                    "reminder_minutes"
                )
            )

            body[
                "isReminderOn"
            ] = True

        body.update(
            kwargs.pop(
                "body",
                {},
            )
        )

        if not body:
            raise ValueError(
                "No event fields were supplied for update."
            )

        path = (
            f"{self._user_path()}/events/{event_id}"
            if calendar_id is None
            else
            f"{self._user_path()}/calendars/"
            f"{calendar_id}/events/{event_id}"
        )

        response = self._request(
            "PATCH",
            path,
            body=body,
        )

        return self._normalize_event(
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
        """Delete an Outlook event."""

        path = (
            f"{self._user_path()}/events/{event_id}"
            if calendar_id is None
            else
            f"{self._user_path()}/calendars/"
            f"{calendar_id}/events/{event_id}"
        )

        self._request(
            "DELETE",
            path,
            params=kwargs or None,
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
        """Return Outlook events."""

        params: dict[str, Any] = {
            "$top": max(
                1,
                min(
                    int(limit),
                    999,
                ),
            ),
            "$orderby": "start/dateTime",
        }

        if start is not None:
            params[
                "startDateTime"
            ] = start.isoformat()

        if end is not None:
            params[
                "endDateTime"
            ] = end.isoformat()

        params.update(
            kwargs
        )

        if calendar_id is None:

            path = (
                f"{self._user_path()}/events"
            )

        else:

            path = (
                f"{self._user_path()}/calendars/"
                f"{calendar_id}/events"
            )

        response = self._request(
            "GET",
            path,
            params=params,
        )

        return [
            self._normalize_event(
                event,
                calendar_id=calendar_id,
            )
            for event in response.get(
                "value",
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
        """
        Query Outlook schedule/free-busy information.
        """

        if end <= start:
            raise ValueError(
                "end must be after start."
            )

        ids = list(
            calendar_ids
            or []
        )

        body = {
            "schedules": ids or [self.user_id],
            "startTime": {
                "dateTime": start.isoformat(),
                "timeZone": "UTC",
            },
            "endTime": {
                "dateTime": end.isoformat(),
                "timeZone": "UTC",
            },
            "availabilityViewInterval": 30,
        }

        body.update(
            kwargs.pop(
                "body",
                {},
            )
        )

        response = self._request(
            "POST",
            f"{self._user_path()}/calendar/"
            "getSchedule",
            body=body,
        )

        return response

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
    ) -> bool:
        """Check whether Microsoft Graph is reachable."""

        try:
            self.list_calendars(
                **{
                    "$top": 1
                }
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
        """Return Outlook Calendar capabilities."""

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
    "OutlookCalendarProvider",
]


