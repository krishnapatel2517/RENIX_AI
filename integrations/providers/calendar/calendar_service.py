"""
RENIX Calendar Service
======================

High-level, provider-independent calendar service.

Responsibilities:
    - List calendars
    - Create events
    - Get events
    - Update events
    - Delete events
    - Search events
    - Find free/busy periods
    - Handle recurring events
    - Normalize provider responses
    - Provider fallback support

Provider adapters should expose compatible methods such as:

    list_calendars()
    create_event()
    get_event()
    update_event()
    delete_event()
    list_events()
    search_events()
    free_busy()

The service keeps provider-specific logic outside RENIX's
core calendar API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class CalendarServiceError(RuntimeError):
    """Base exception for calendar operations."""


class CalendarProviderError(CalendarServiceError):
    """Raised when a calendar provider fails."""


class CalendarConfigurationError(CalendarServiceError):
    """Raised when calendar configuration is invalid."""


class EventNotFoundError(CalendarServiceError):
    """Raised when a requested event cannot be found."""


class InvalidEventError(CalendarServiceError):
    """Raised when event data is invalid."""


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class Calendar:
    """Normalized calendar representation."""

    id: str

    name: str

    description: str = ""

    timezone: str | None = None

    color: str | None = None

    primary: bool = False

    read_only: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class CalendarEvent:
    """Normalized calendar event."""

    id: str | None = None

    calendar_id: str | None = None

    title: str = ""

    description: str = ""

    location: str | None = None

    start: datetime | None = None

    end: datetime | None = None

    all_day: bool = False

    timezone: str | None = None

    attendees: list[Any] = field(
        default_factory=list
    )

    organizer: Any = None

    recurrence: list[str] = field(
        default_factory=list
    )

    reminders: list[Any] = field(
        default_factory=list
    )

    status: str = "confirmed"

    url: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate basic event data."""

        if not self.title.strip():
            raise InvalidEventError(
                "Event title cannot be empty."
            )

        if (
            self.start is not None
            and self.end is not None
            and self.end < self.start
        ):
            raise InvalidEventError(
                "Event end time cannot be before "
                "the start time."
            )


@dataclass
class FreeBusyPeriod:
    """A busy time period."""

    start: datetime

    end: datetime

    calendar_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class FreeBusyResult:
    """Normalized free/busy response."""

    busy: list[FreeBusyPeriod] = field(
        default_factory=list
    )

    calendars: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# CALENDAR SERVICE
# ============================================================


class CalendarService:
    """
    High-level calendar service.

    Example:

        calendar = CalendarService(
            provider_registry
        )

        calendars = calendar.list_calendars()

        event = calendar.create_event(
            calendar_id="primary",
            title="Study",
            start=start_time,
            end=end_time,
        )
    """

    def __init__(
        self,
        provider_registry: Any,
        *,
        default_provider: str | None = None,
    ) -> None:

        if provider_registry is None:
            raise CalendarConfigurationError(
                "provider_registry is required."
            )

        self.registry = provider_registry

        self.default_provider = (
            default_provider
        )

    # ========================================================
    # PROVIDER
    # ========================================================

    def provider(
        self,
        name: str | None = None,
    ) -> Any:
        """
        Return the requested or active calendar provider.
        """

        provider_name = (
            name
            or self.default_provider
        )

        if provider_name:

            getter = getattr(
                self.registry,
                "get",
                None,
            )

            if not callable(getter):
                raise CalendarProviderError(
                    "Provider registry does not "
                    "support provider lookup."
                )

            try:
                provider = getter(
                    provider_name
                )
            except Exception as exc:
                raise CalendarProviderError(
                    f"Unable to load calendar provider "
                    f"'{provider_name}'."
                ) from exc

            if provider is None:
                raise CalendarProviderError(
                    f"Calendar provider "
                    f"'{provider_name}' is unavailable."
                )

            return provider

        getter = getattr(
            self.registry,
            "get_active",
            None,
        )

        if not callable(getter):
            raise CalendarProviderError(
                "No active calendar provider is available."
            )

        provider = getter()

        if provider is None:
            raise CalendarProviderError(
                "Calendar provider registry returned no "
                "active provider."
            )

        return provider

    # ========================================================
    # CALENDARS
    # ========================================================

    def list_calendars(
        self,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> list[Calendar]:
        """Return all available calendars."""

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "list_calendars",
                "get_calendars",
                "calendars",
            ),
        )

        if method is None:
            raise CalendarProviderError(
                "Calendar provider does not support "
                "calendar listing."
            )

        try:
            raw = method(**kwargs)
            return [
                self._normalize_calendar(
                    item
                )
                for item in self._as_list(raw)
            ]
        except Exception as exc:
            raise CalendarProviderError(
                f"Unable to list calendars: {exc}"
            ) from exc

    # ========================================================
    # GET CALENDAR
    # ========================================================

    def get_calendar(
        self,
        calendar_id: str,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Calendar:
        """Return one calendar."""

        if not calendar_id:
            raise ValueError(
                "calendar_id is required."
            )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "get_calendar",
                "calendar",
            ),
        )

        if method is None:
            raise CalendarProviderError(
                "Calendar provider does not support "
                "calendar lookup."
            )

        try:
            raw = method(
                calendar_id=calendar_id,
                **kwargs,
            )

            return self._normalize_calendar(
                raw
            )

        except Exception as exc:
            raise CalendarProviderError(
                f"Unable to get calendar: {exc}"
            ) from exc

    # ========================================================
    # CREATE EVENT
    # ========================================================

    def create_event(
        self,
        calendar_id: str,
        title: str,
        *,
        start: datetime | str,
        end: datetime | str,
        description: str = "",
        location: str | None = None,
        attendees: Iterable[Any] | None = None,
        reminders: Iterable[Any] | None = None,
        recurrence: Iterable[str] | None = None,
        all_day: bool = False,
        timezone_name: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> CalendarEvent:
        """Create a calendar event."""

        if not calendar_id:
            raise InvalidEventError(
                "calendar_id is required."
            )

        if not title.strip():
            raise InvalidEventError(
                "Event title cannot be empty."
            )

        start_dt = self._parse_datetime(
            start
        )

        end_dt = self._parse_datetime(
            end
        )

        event = CalendarEvent(
            calendar_id=calendar_id,
            title=title.strip(),
            description=description,
            location=location,
            start=start_dt,
            end=end_dt,
            attendees=list(
                attendees or []
            ),
            reminders=list(
                reminders or []
            ),
            recurrence=list(
                recurrence or []
            ),
            all_day=all_day,
            timezone=timezone_name,
        )

        event.validate()

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "create_event",
                "add_event",
                "insert_event",
            ),
        )

        if method is None:
            raise CalendarProviderError(
                "Calendar provider does not support "
                "event creation."
            )

        payload = self._event_payload(
            event
        )

        payload.update(kwargs)

        try:
            raw = method(
                **payload
            )

            return self._normalize_event(
                raw,
                fallback=event,
            )

        except Exception as exc:
            raise CalendarProviderError(
                f"Unable to create event: {exc}"
            ) from exc

    # ========================================================
    # GET EVENT
    # ========================================================

    def get_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> CalendarEvent:
        """Retrieve an event."""

        if not event_id:
            raise EventNotFoundError(
                "event_id is required."
            )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "get_event",
                "event",
                "fetch_event",
            ),
        )

        if method is None:
            raise CalendarProviderError(
                "Calendar provider does not support "
                "event lookup."
            )

        payload = {
            "event_id": event_id,
            **kwargs,
        }

        if calendar_id:
            payload[
                "calendar_id"
            ] = calendar_id

        try:
            raw = method(
                **payload
            )

            if raw is None:
                raise EventNotFoundError(
                    f"Event '{event_id}' was not found."
                )

            return self._normalize_event(
                raw
            )

        except EventNotFoundError:
            raise

        except Exception as exc:
            raise CalendarProviderError(
                f"Unable to get event: {exc}"
            ) from exc

    # ========================================================
    # UPDATE EVENT
    # ========================================================

    def update_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        title: str | None = None,
        description: str | None = None,
        location: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        attendees: Iterable[Any] | None = None,
        reminders: Iterable[Any] | None = None,
        recurrence: Iterable[str] | None = None,
        all_day: bool | None = None,
        timezone_name: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> CalendarEvent:
        """Update an existing event."""

        if not event_id:
            raise InvalidEventError(
                "event_id is required."
            )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "update_event",
                "edit_event",
                "patch_event",
            ),
        )

        if method is None:
            raise CalendarProviderError(
                "Calendar provider does not support "
                "event updates."
            )

        payload: dict[str, Any] = {
            "event_id": event_id,
        }

        if calendar_id is not None:
            payload[
                "calendar_id"
            ] = calendar_id

        if title is not None:
            payload[
                "title"
            ] = title

        if description is not None:
            payload[
                "description"
            ] = description

        if location is not None:
            payload[
                "location"
            ] = location

        if start is not None:
            payload[
                "start"
            ] = self._parse_datetime(
                start
            )

        if end is not None:
            payload[
                "end"
            ] = self._parse_datetime(
                end
            )

        if attendees is not None:
            payload[
                "attendees"
            ] = list(attendees)

        if reminders is not None:
            payload[
                "reminders"
            ] = list(reminders)

        if recurrence is not None:
            payload[
                "recurrence"
            ] = list(recurrence)

        if all_day is not None:
            payload[
                "all_day"
            ] = all_day

        if timezone_name is not None:
            payload[
                "timezone"
            ] = timezone_name

        payload.update(kwargs)

        try:
            raw = method(
                **payload
            )

            return self._normalize_event(
                raw
            )

        except Exception as exc:
            raise CalendarProviderError(
                f"Unable to update event: {exc}"
            ) from exc

    # ========================================================
    # DELETE EVENT
    # ========================================================

    def delete_event(
        self,
        event_id: str,
        *,
        calendar_id: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Delete an event."""

        if not event_id:
            raise InvalidEventError(
                "event_id is required."
            )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "delete_event",
                "remove_event",
                "cancel_event",
            ),
        )

        if method is None:
            raise CalendarProviderError(
                "Calendar provider does not support "
                "event deletion."
            )

        payload = {
            "event_id": event_id,
            **kwargs,
        }

        if calendar_id:
            payload[
                "calendar_id"
            ] = calendar_id

        try:
            result = method(
                **payload
            )

            if result is None:
                return True

            if isinstance(
                result,
                bool,
            ):
                return result

            return True

        except Exception as exc:
            raise CalendarProviderError(
                f"Unable to delete event: {exc}"
            ) from exc

    # ========================================================
    # LIST EVENTS
    # ========================================================

    def list_events(
        self,
        *,
        calendar_id: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        provider: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[CalendarEvent]:
        """List calendar events."""

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "list_events",
                "get_events",
                "events",
            ),
        )

        if method is None:
            raise CalendarProviderError(
                "Calendar provider does not support "
                "event listing."
            )

        payload: dict[str, Any] = {
            "limit": max(
                1,
                int(limit),
            )
        }

        if calendar_id:
            payload[
                "calendar_id"
            ] = calendar_id

        if start is not None:
            payload[
                "start"
            ] = self._parse_datetime(
                start
            )

        if end is not None:
            payload[
                "end"
            ] = self._parse_datetime(
                end
            )

        payload.update(kwargs)

        try:
            raw = method(
                **payload
            )

            return [
                self._normalize_event(
                    item
                )
                for item in self._as_list(
                    raw
                )
            ]

        except Exception as exc:
            raise CalendarProviderError(
                f"Unable to list events: {exc}"
            ) from exc

    # ========================================================
    # SEARCH EVENTS
    # ========================================================

    def search_events(
        self,
        query: str,
        *,
        calendar_id: str | None = None,
        provider: str | None = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[CalendarEvent]:
        """Search events by title, description, or provider query."""

        if not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "search_events",
                "find_events",
            ),
        )

        if method is not None:

            payload = {
                "query": query.strip(),
                "limit": max(
                    1,
                    int(limit),
                ),
                **kwargs,
            }

            if calendar_id:
                payload[
                    "calendar_id"
                ] = calendar_id

            try:
                raw = method(
                    **payload
                )

                return [
                    self._normalize_event(
                        item
                    )
                    for item in self._as_list(
                        raw
                    )
                ]

            except Exception as exc:
                raise CalendarProviderError(
                    f"Unable to search events: {exc}"
                ) from exc

        # Provider has no native search.
        # Fall back to retrieving events and filtering locally.

        events = self.list_events(
            calendar_id=calendar_id,
            provider=provider,
            limit=limit,
            **kwargs,
        )

        query_lower = query.lower()

        return [
            event
            for event in events
            if (
                query_lower
                in event.title.lower()
                or query_lower
                in event.description.lower()
                or (
                    event.location
                    and query_lower
                    in event.location.lower()
                )
            )
        ]

    # ========================================================
    # TODAY
    # ========================================================

    def today(
        self,
        *,
        calendar_id: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> list[CalendarEvent]:
        """Return today's events."""

        now = datetime.now(
            timezone.utc
        )

        start = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        end = start + timedelta(
            days=1
        )

        return self.list_events(
            calendar_id=calendar_id,
            start=start,
            end=end,
            provider=provider,
            **kwargs,
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
        **kwargs: Any,
    ) -> list[CalendarEvent]:
        """Return upcoming events."""

        now = datetime.now(
            timezone.utc
        )

        end = now + timedelta(
            days=max(
                1,
                int(days),
            )
        )

        return self.list_events(
            calendar_id=calendar_id,
            start=now,
            end=end,
            provider=provider,
            limit=limit,
            **kwargs,
        )

    # ========================================================
    # FREE / BUSY
    # ========================================================

    def free_busy(
        self,
        *,
        start: datetime | str,
        end: datetime | str,
        calendar_ids: Iterable[str] | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> FreeBusyResult:
        """Return busy periods for calendars."""

        start_dt = self._parse_datetime(
            start
        )

        end_dt = self._parse_datetime(
            end
        )

        if end_dt < start_dt:
            raise ValueError(
                "Free/busy end cannot be before start."
            )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "free_busy",
                "get_free_busy",
                "availability",
            ),
        )

        if method is None:

            # Fallback implementation using events.
            events = self.list_events(
                start=start_dt,
                end=end_dt,
                provider=provider,
                **kwargs,
            )

            periods = []

            for event in events:

                if (
                    event.start is None
                    or event.end is None
                ):
                    continue

                periods.append(
                    FreeBusyPeriod(
                        start=event.start,
                        end=event.end,
                        calendar_id=(
                            event.calendar_id
                        ),
                    )
                )

            return FreeBusyResult(
                busy=periods
            )

        payload = {
            "start": start_dt,
            "end": end_dt,
            **kwargs,
        }

        if calendar_ids is not None:
            payload[
                "calendar_ids"
            ] = list(calendar_ids)

        try:

            raw = method(
                **payload
            )

            return self._normalize_free_busy(
                raw
            )

        except Exception as exc:

            raise CalendarProviderError(
                f"Unable to get free/busy data: {exc}"
            ) from exc

    # ========================================================
    # EVENT PAYLOAD
    # ========================================================

    @staticmethod
    def _event_payload(
        event: CalendarEvent,
    ) -> dict[str, Any]:
        """Convert normalized event into provider payload."""

        return {
            "calendar_id": event.calendar_id,
            "title": event.title,
            "description": event.description,
            "location": event.location,
            "start": event.start,
            "end": event.end,
            "attendees": event.attendees,
            "reminders": event.reminders,
            "recurrence": event.recurrence,
            "all_day": event.all_day,
            "timezone": event.timezone,
        }

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_calendar(
        raw: Any,
    ) -> Calendar:

        if isinstance(
            raw,
            Calendar,
        ):
            return raw

        if not isinstance(
            raw,
            dict,
        ):
            return Calendar(
                id=str(raw),
                name=str(raw),
            )

        return Calendar(
            id=str(
                raw.get(
                    "id",
                    "",
                )
            ),
            name=str(
                raw.get(
                    "name",
                    raw.get(
                        "summary",
                        "",
                    ),
                )
            ),
            description=str(
                raw.get(
                    "description",
                    "",
                )
                or ""
            ),
            timezone=(
                raw.get(
                    "timezone"
                )
                or raw.get(
                    "timeZone"
                )
            ),
            color=(
                raw.get(
                    "color"
                )
                or raw.get(
                    "backgroundColor"
                )
            ),
            primary=bool(
                raw.get(
                    "primary",
                    False,
                )
            ),
            read_only=bool(
                raw.get(
                    "read_only",
                    raw.get(
                        "readOnly",
                        False,
                    ),
                )
            ),
            metadata=dict(
                raw.get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    @classmethod
    def _normalize_event(
        cls,
        raw: Any,
        *,
        fallback: CalendarEvent | None = None,
    ) -> CalendarEvent:

        if isinstance(
            raw,
            CalendarEvent,
        ):
            return raw

        if not isinstance(
            raw,
            dict,
        ):

            if fallback is not None:
                return fallback

            return CalendarEvent(
                id=str(raw)
            )

        event = CalendarEvent(
            id=(
                str(
                    raw.get(
                        "id"
                    )
                )
                if raw.get("id")
                is not None
                else None
            ),
            calendar_id=(
                raw.get(
                    "calendar_id"
                )
                or raw.get(
                    "calendarId"
                )
            ),
            title=str(
                raw.get(
                    "title",
                    raw.get(
                        "summary",
                        "",
                    ),
                )
                or ""
            ),
            description=str(
                raw.get(
                    "description",
                    "",
                )
                or ""
            ),
            location=raw.get(
                "location"
            ),
            start=cls._optional_datetime(
                raw.get(
                    "start"
                )
            ),
            end=cls._optional_datetime(
                raw.get(
                    "end"
                )
            ),
            all_day=bool(
                raw.get(
                    "all_day",
                    raw.get(
                        "allDay",
                        False,
                    ),
                )
            ),
            timezone=(
                raw.get(
                    "timezone"
                )
                or raw.get(
                    "timeZone"
                )
            ),
            attendees=list(
                raw.get(
                    "attendees",
                    [],
                )
                or []
            ),
            organizer=raw.get(
                "organizer"
            ),
            recurrence=list(
                raw.get(
                    "recurrence",
                    [],
                )
                or []
            ),
            reminders=list(
                raw.get(
                    "reminders",
                    [],
                )
                or []
            ),
            status=str(
                raw.get(
                    "status",
                    "confirmed",
                )
            ),
            url=(
                raw.get(
                    "url"
                )
                or raw.get(
                    "htmlLink"
                )
            ),
            metadata=dict(
                raw.get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )

        if not event.title and fallback:
            event.title = fallback.title

        return event

    @staticmethod
    def _normalize_free_busy(
        raw: Any,
    ) -> FreeBusyResult:

        if isinstance(
            raw,
            FreeBusyResult,
        ):
            return raw

        if not isinstance(
            raw,
            dict,
        ):
            return FreeBusyResult()

        raw_busy = (
            raw.get(
                "busy"
            )
            or raw.get(
                "periods"
            )
            or []
        )

        periods = []

        for item in raw_busy:

            if isinstance(
                item,
                FreeBusyPeriod,
            ):
                periods.append(item)
                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            start = CalendarService._optional_datetime(
                item.get(
                    "start"
                )
            )

            end = CalendarService._optional_datetime(
                item.get(
                    "end"
                )
            )

            if start is None or end is None:
                continue

            periods.append(
                FreeBusyPeriod(
                    start=start,
                    end=end,
                    calendar_id=(
                        item.get(
                            "calendar_id"
                        )
                    ),
                    metadata=dict(
                        item.get(
                            "metadata",
                            {},
                        )
                        or {}
                    ),
                )
            )

        return FreeBusyResult(
            busy=periods,
            calendars=dict(
                raw.get(
                    "calendars",
                    {},
                )
                or {}
            ),
            metadata=dict(
                raw.get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    # ========================================================
    # DATETIME HELPERS
    # ========================================================

    @staticmethod
    def _parse_datetime(
        value: datetime | str,
    ) -> datetime:

        if isinstance(
            value,
            datetime,
        ):

            if value.tzinfo is None:
                return value.replace(
                    tzinfo=timezone.utc
                )

            return value

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Datetime must be a datetime "
                "object or ISO string."
            )

        text = value.strip()

        if text.endswith(
            "Z"
        ):
            text = (
                text[:-1]
                + "+00:00"
            )

        try:

            parsed = datetime.fromisoformat(
                text
            )

        except ValueError as exc:

            raise ValueError(
                f"Invalid datetime: {value}"
            ) from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed

    @classmethod
    def _optional_datetime(
        cls,
        value: Any,
    ) -> datetime | None:

        if value is None:
            return None

        if isinstance(
            value,
            datetime,
        ):
            return cls._parse_datetime(
                value
            )

        if isinstance(
            value,
            dict,
        ):

            value = (
                value.get(
                    "dateTime"
                )
                or value.get(
                    "datetime"
                )
                or value.get(
                    "date"
                )
            )

        try:

            return cls._parse_datetime(
                value
            )

        except (
            ValueError,
            TypeError,
        ):

            return None

    # ========================================================
    # GENERAL HELPERS
    # ========================================================

    @staticmethod
    def _find_method(
        provider: Any,
        names: tuple[str, ...],
    ) -> Any | None:

        for name in names:

            method = getattr(
                provider,
                name,
                None,
            )

            if callable(method):
                return method

        return None

    @staticmethod
    def _as_list(
        value: Any,
    ) -> list[Any]:

        if value is None:
            return []

        if isinstance(
            value,
            list,
        ):
            return value

        if isinstance(
            value,
            tuple,
        ):
            return list(value)

        return [value]


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "Calendar",
    "CalendarEvent",
    "FreeBusyPeriod",
    "FreeBusyResult",
    "CalendarService",
    "CalendarServiceError",
    "CalendarProviderError",
    "CalendarConfigurationError",
    "EventNotFoundError",
    "InvalidEventError",
]


