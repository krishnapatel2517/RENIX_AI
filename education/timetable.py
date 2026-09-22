"""
RENIX Education — Timetable Manager
====================================

Manages RENIX's academic timetable.

Responsibilities:
    - Create, update and delete timetable entries
    - Manage recurring weekly classes/study blocks
    - Manage one-time events
    - Detect overlapping entries
    - Find today's schedule
    - Find upcoming classes/study sessions
    - Track completed timetable entries
    - Calculate daily/weekly study time
    - Export timetable data
    - Import timetable data
    - Publish timetable events to RENIX's event bus
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class TimetableEntry:
    """
    Represents one timetable block.

    entry_type:
        class
        study
        homework
        revision
        exam
        activity
        break
        custom

    recurrence:
        none
        daily
        weekdays
        weekly
    """

    id: str
    title: str
    subject: str | None
    entry_date: str
    start_time: str
    end_time: str

    entry_type: str = "study"
    room: str | None = None
    teacher: str | None = None
    notes: str = ""

    recurrence: str = "none"
    weekdays: list[int] = field(default_factory=list)

    completed: bool = False
    cancelled: bool = False

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def duration_minutes(self) -> int:
        """Return duration of the timetable block."""

        start = _parse_time(self.start_time)
        end = _parse_time(self.end_time)

        start_minutes = (
            start.hour * 60
            + start.minute
        )

        end_minutes = (
            end.hour * 60
            + end.minute
        )

        if end_minutes < start_minutes:
            end_minutes += 24 * 60

        return end_minutes - start_minutes

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_minutes"] = (
            self.duration_minutes
        )
        return data


@dataclass
class TimetableStatistics:
    """Statistics generated from timetable entries."""

    total_entries: int = 0
    completed_entries: int = 0
    cancelled_entries: int = 0

    total_minutes: int = 0
    study_minutes: int = 0
    class_minutes: int = 0
    revision_minutes: int = 0
    homework_minutes: int = 0
    exam_minutes: int = 0

    @property
    def completion_rate(self) -> float:
        if self.total_entries <= 0:
            return 0.0

        return (
            self.completed_entries
            / self.total_entries
        ) * 100

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["completion_rate"] = (
            self.completion_rate
        )
        return data


# ============================================================
# CONSTANTS
# ============================================================


VALID_ENTRY_TYPES = {
    "class",
    "study",
    "homework",
    "revision",
    "exam",
    "activity",
    "break",
    "custom",
}

VALID_RECURRENCES = {
    "none",
    "daily",
    "weekdays",
    "weekly",
}


# Python weekday:
# Monday = 0
# Tuesday = 1
# Wednesday = 2
# Thursday = 3
# Friday = 4
# Saturday = 5
# Sunday = 6

WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


# ============================================================
# TIMETABLE MANAGER
# ============================================================


class TimetableManager:
    """
    Central timetable system for RENIX Education.

    Example:

        timetable = TimetableManager()

        entry = timetable.add_entry(
            title="Mathematics",
            subject="Mathematics",
            entry_date="2026-08-24",
            start_time="18:00",
            end_time="19:00",
            entry_type="study",
        )

        todays_schedule = timetable.get_today()

    """

    def __init__(
        self,
        *,
        storage: Any | None = None,
        event_bus: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:

        self.storage = storage
        self.event_bus = event_bus

        self._clock = (
            clock
            if clock is not None
            else datetime.now
        )

        self._entries: dict[
            str,
            TimetableEntry,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "RENIX TimetableManager initialized."
        )

    # ========================================================
    # CREATE
    # ========================================================

    def add_entry(
        self,
        *,
        title: str,
        entry_date: date | datetime | str,
        start_time: time | str,
        end_time: time | str,
        subject: str | None = None,
        entry_type: str = "study",
        room: str | None = None,
        teacher: str | None = None,
        notes: str = "",
        recurrence: str = "none",
        weekdays: Iterable[int] | None = None,
        allow_overlap: bool = False,
    ) -> TimetableEntry:

        title = self._clean_text(
            title,
            "title",
        )

        normalized_date = (
            self._serialize_date(
                entry_date
            )
        )

        normalized_start = (
            self._serialize_time(
                start_time
            )
        )

        normalized_end = (
            self._serialize_time(
                end_time
            )
        )

        entry_type = (
            entry_type.strip().lower()
        )

        if entry_type not in VALID_ENTRY_TYPES:
            raise ValueError(
                f"Invalid timetable entry type: "
                f"{entry_type}"
            )

        recurrence = (
            recurrence.strip().lower()
        )

        if recurrence not in VALID_RECURRENCES:
            raise ValueError(
                f"Invalid recurrence: {recurrence}"
            )

        normalized_weekdays = (
            self._normalize_weekdays(
                weekdays
            )
        )

        if (
            recurrence == "weekly"
            and not normalized_weekdays
        ):
            normalized_weekdays = [
                datetime.strptime(
                    normalized_date,
                    "%Y-%m-%d",
                ).weekday()
            ]

        if (
            recurrence != "weekly"
            and normalized_weekdays
        ):
            raise ValueError(
                "weekdays can only be supplied "
                "for weekly recurrence."
            )

        duration = self._duration_minutes(
            normalized_start,
            normalized_end,
        )

        if duration <= 0:
            raise ValueError(
                "end_time must be later than start_time."
            )

        entry = TimetableEntry(
            id=self._new_id("tt"),
            title=title,
            subject=(
                subject.strip()
                if subject
                else None
            ),
            entry_date=normalized_date,
            start_time=normalized_start,
            end_time=normalized_end,
            entry_type=entry_type,
            room=(
                room.strip()
                if room
                else None
            ),
            teacher=(
                teacher.strip()
                if teacher
                else None
            ),
            notes=notes.strip(),
            recurrence=recurrence,
            weekdays=normalized_weekdays,
        )

        with self._lock:

            if not allow_overlap:

                conflicts = (
                    self.find_conflicts(
                        entry
                    )
                )

                if conflicts:
                    raise ValueError(
                        self._format_conflict_error(
                            conflicts
                        )
                    )

            self._entries[
                entry.id
            ] = entry

            self._persist(entry)

        self._emit(
            "education.timetable.created",
            entry.to_dict(),
        )

        return entry

    # ========================================================
    # READ
    # ========================================================

    def get_entry(
        self,
        entry_id: str,
    ) -> TimetableEntry | None:

        with self._lock:
            return self._entries.get(
                entry_id
            )

    def get_entries(
        self,
        *,
        entry_type: str | None = None,
        subject: str | None = None,
        completed: bool | None = None,
        cancelled: bool | None = None,
    ) -> list[TimetableEntry]:

        with self._lock:
            entries = list(
                self._entries.values()
            )

        if entry_type is not None:

            entry_type = (
                entry_type.strip().lower()
            )

            entries = [
                entry
                for entry in entries
                if entry.entry_type
                == entry_type
            ]

        if subject is not None:

            normalized_subject = (
                subject.strip().casefold()
            )

            entries = [
                entry
                for entry in entries
                if (
                    entry.subject
                    and entry.subject.casefold()
                    == normalized_subject
                )
            ]

        if completed is not None:

            entries = [
                entry
                for entry in entries
                if entry.completed
                == completed
            ]

        if cancelled is not None:

            entries = [
                entry
                for entry in entries
                if entry.cancelled
                == cancelled
            ]

        return sorted(
            entries,
            key=self._sort_key,
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update_entry(
        self,
        entry_id: str,
        *,
        title: str | None = None,
        entry_date: date | datetime | str | None = None,
        start_time: time | str | None = None,
        end_time: time | str | None = None,
        subject: str | None = None,
        entry_type: str | None = None,
        room: str | None = None,
        teacher: str | None = None,
        notes: str | None = None,
        recurrence: str | None = None,
        weekdays: Iterable[int] | None = None,
        allow_overlap: bool = False,
    ) -> TimetableEntry:

        with self._lock:

            entry = self._require_entry(
                entry_id
            )

            if title is not None:

                entry.title = self._clean_text(
                    title,
                    "title",
                )

            if entry_date is not None:

                entry.entry_date = (
                    self._serialize_date(
                        entry_date
                    )
                )

            if start_time is not None:

                entry.start_time = (
                    self._serialize_time(
                        start_time
                    )
                )

            if end_time is not None:

                entry.end_time = (
                    self._serialize_time(
                        end_time
                    )
                )

            if subject is not None:

                entry.subject = (
                    subject.strip()
                    or None
                )

            if entry_type is not None:

                entry_type = (
                    entry_type.strip().lower()
                )

                if (
                    entry_type
                    not in VALID_ENTRY_TYPES
                ):
                    raise ValueError(
                        f"Invalid entry type: "
                        f"{entry_type}"
                    )

                entry.entry_type = (
                    entry_type
                )

            if room is not None:

                entry.room = (
                    room.strip()
                    or None
                )

            if teacher is not None:

                entry.teacher = (
                    teacher.strip()
                    or None
                )

            if notes is not None:
                entry.notes = notes.strip()

            if recurrence is not None:

                recurrence = (
                    recurrence.strip().lower()
                )

                if (
                    recurrence
                    not in VALID_RECURRENCES
                ):
                    raise ValueError(
                        f"Invalid recurrence: "
                        f"{recurrence}"
                    )

                entry.recurrence = (
                    recurrence
                )

            if weekdays is not None:

                entry.weekdays = (
                    self._normalize_weekdays(
                        weekdays
                    )
                )

            if (
                self._duration_minutes(
                    entry.start_time,
                    entry.end_time,
                )
                <= 0
            ):
                raise ValueError(
                    "end_time must be later "
                    "than start_time."
                )

            if not allow_overlap:

                conflicts = (
                    self.find_conflicts(
                        entry,
                        exclude_id=entry.id,
                    )
                )

                if conflicts:
                    raise ValueError(
                        self._format_conflict_error(
                            conflicts
                        )
                    )

            self._persist(entry)

        self._emit(
            "education.timetable.updated",
            entry.to_dict(),
        )

        return entry

    # ========================================================
    # DELETE
    # ========================================================

    def delete_entry(
        self,
        entry_id: str,
    ) -> bool:

        with self._lock:

            entry = self._entries.pop(
                entry_id,
                None,
            )

            if entry is None:
                return False

        self._delete_persisted(
            entry_id
        )

        self._emit(
            "education.timetable.deleted",
            {
                "entry_id": entry_id,
                "entry": entry.to_dict(),
            },
        )

        return True

    # ========================================================
    # COMPLETE / CANCEL
    # ========================================================

    def complete_entry(
        self,
        entry_id: str,
    ) -> TimetableEntry:

        with self._lock:

            entry = self._require_entry(
                entry_id
            )

            if entry.cancelled:
                raise RuntimeError(
                    "Cancelled entries cannot be completed."
                )

            entry.completed = True

            self._persist(entry)

        self._emit(
            "education.timetable.completed",
            entry.to_dict(),
        )

        return entry

    def reopen_entry(
        self,
        entry_id: str,
    ) -> TimetableEntry:

        with self._lock:

            entry = self._require_entry(
                entry_id
            )

            if entry.cancelled:
                raise RuntimeError(
                    "Cancelled entries cannot be reopened."
                )

            entry.completed = False

            self._persist(entry)

        self._emit(
            "education.timetable.reopened",
            entry.to_dict(),
        )

        return entry

    def cancel_entry(
        self,
        entry_id: str,
    ) -> TimetableEntry:

        with self._lock:

            entry = self._require_entry(
                entry_id
            )

            entry.cancelled = True
            entry.completed = False

            self._persist(entry)

        self._emit(
            "education.timetable.cancelled",
            entry.to_dict(),
        )

        return entry

    # ========================================================
    # DAILY SCHEDULE
    # ========================================================

    def get_day(
        self,
        target_date: date | datetime | str,
    ) -> list[TimetableEntry]:

        target = self._parse_date(
            target_date
        )

        with self._lock:

            entries = [
                entry
                for entry in self._entries.values()
                if self._entry_occurs_on(
                    entry,
                    target,
                )
            ]

        return sorted(
            entries,
            key=self._sort_key,
        )

    def get_today(
        self,
    ) -> list[TimetableEntry]:

        return self.get_day(
            self._now().date()
        )

    def get_tomorrow(
        self,
    ) -> list[TimetableEntry]:

        return self.get_day(
            self._now().date()
            + timedelta(days=1)
        )

    # ========================================================
    # WEEKLY SCHEDULE
    # ========================================================

    def get_week(
        self,
        week_start: date | datetime | str | None = None,
    ) -> dict[str, list[TimetableEntry]]:

        if week_start is None:

            current = self._now().date()

        else:

            current = self._parse_date(
                week_start
            )

        monday = (
            current
            - timedelta(
                days=current.weekday()
            )
        )

        result: dict[
            str,
            list[TimetableEntry],
        ] = {}

        for offset in range(7):

            current_day = (
                monday
                + timedelta(days=offset)
            )

            result[
                current_day.isoformat()
            ] = self.get_day(
                current_day
            )

        return result

    # ========================================================
    # UPCOMING
    # ========================================================

    def get_upcoming(
        self,
        *,
        limit: int = 10,
        include_completed: bool = False,
        days: int = 7,
    ) -> list[TimetableEntry]:

        if limit < 0:
            raise ValueError(
                "limit cannot be negative."
            )

        if days < 0:
            raise ValueError(
                "days cannot be negative."
            )

        now = self._now()

        results: list[
            tuple[
                datetime,
                TimetableEntry,
            ]
        ] = []

        end_date = (
            now.date()
            + timedelta(days=days)
        )

        for current_date_offset in range(
            days + 1
        ):

            target_date = (
                now.date()
                + timedelta(
                    days=current_date_offset
                )
            )

            for entry in self.get_day(
                target_date
            ):

                if entry.cancelled:
                    continue

                if (
                    entry.completed
                    and not include_completed
                ):
                    continue

                start = datetime.combine(
                    target_date,
                    _parse_time(
                        entry.start_time
                    ),
                )

                if start < now:
                    continue

                if (
                    target_date
                    > end_date
                ):
                    continue

                results.append(
                    (
                        start,
                        entry,
                    )
                )

        results.sort(
            key=lambda item: item[0]
        )

        return [
            entry
            for _, entry in results[:limit]
        ]

    # ========================================================
    # CURRENT ENTRY
    # ========================================================

    def get_current(
        self,
    ) -> TimetableEntry | None:

        now = self._now()

        entries = self.get_day(
            now.date()
        )

        current_minutes = (
            now.hour * 60
            + now.minute
        )

        for entry in entries:

            if entry.cancelled:
                continue

            start = _time_to_minutes(
                entry.start_time
            )

            end = _time_to_minutes(
                entry.end_time
            )

            if (
                start
                <= current_minutes
                < end
            ):
                return entry

        return None

    def get_next(
        self,
    ) -> TimetableEntry | None:

        upcoming = self.get_upcoming(
            limit=1
        )

        return (
            upcoming[0]
            if upcoming
            else None
        )

    # ========================================================
    # CONFLICT DETECTION
    # ========================================================

    def find_conflicts(
        self,
        candidate: TimetableEntry,
        *,
        exclude_id: str | None = None,
    ) -> list[TimetableEntry]:

        conflicts: list[
            TimetableEntry
        ] = []

        candidate_dates = (
            self._candidate_dates_for_conflict(
                candidate
            )
        )

        candidate_start = (
            _time_to_minutes(
                candidate.start_time
            )
        )

        candidate_end = (
            _time_to_minutes(
                candidate.end_time
            )
        )

        with self._lock:

            entries = list(
                self._entries.values()
            )

        for existing in entries:

            if (
                exclude_id
                and existing.id
                == exclude_id
            ):
                continue

            if existing.cancelled:
                continue

            existing_dates = (
                self._candidate_dates_for_conflict(
                    existing
                )
            )

            if not (
                candidate_dates
                & existing_dates
            ):
                continue

            existing_start = (
                _time_to_minutes(
                    existing.start_time
                )
            )

            existing_end = (
                _time_to_minutes(
                    existing.end_time
                )
            )

            if self._times_overlap(
                candidate_start,
                candidate_end,
                existing_start,
                existing_end,
            ):

                conflicts.append(
                    existing
                )

        return sorted(
            conflicts,
            key=self._sort_key,
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
    ) -> list[TimetableEntry]:

        query = self._clean_text(
            query,
            "search query",
        ).casefold()

        with self._lock:

            entries = list(
                self._entries.values()
            )

        results = []

        for entry in entries:

            searchable = " ".join(
                [
                    entry.title,
                    entry.subject or "",
                    entry.entry_type,
                    entry.room or "",
                    entry.teacher or "",
                    entry.notes,
                ]
            ).casefold()

            if query in searchable:
                results.append(entry)

        return sorted(
            results,
            key=self._sort_key,
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
        *,
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
    ) -> TimetableStatistics:

        if start_date is None:

            start = self._now().date()

        else:

            start = self._parse_date(
                start_date
            )

        if end_date is None:

            end = start

        else:

            end = self._parse_date(
                end_date
            )

        if end < start:
            raise ValueError(
                "end_date cannot be before start_date."
            )

        entries: list[
            TimetableEntry
        ] = []

        current = start

        while current <= end:

            entries.extend(
                self.get_day(current)
            )

            current += timedelta(
                days=1
            )

        # Recurring entries can produce the same
        # object multiple times, which is intentional:
        # statistics represent occurrences, not definitions.

        stats = TimetableStatistics()

        stats.total_entries = len(
            entries
        )

        stats.completed_entries = sum(
            entry.completed
            for entry in entries
        )

        stats.cancelled_entries = sum(
            entry.cancelled
            for entry in entries
        )

        for entry in entries:

            minutes = (
                entry.duration_minutes
            )

            if not entry.cancelled:

                stats.total_minutes += (
                    minutes
                )

            if entry.entry_type == "study":
                stats.study_minutes += minutes

            elif entry.entry_type == "class":
                stats.class_minutes += minutes

            elif entry.entry_type == "revision":
                stats.revision_minutes += minutes

            elif entry.entry_type == "homework":
                stats.homework_minutes += minutes

            elif entry.entry_type == "exam":
                stats.exam_minutes += minutes

        return stats

    # ========================================================
    # IMPORT / EXPORT
    # ========================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "entries": [
                    entry.to_dict()
                    for entry
                    in self._entries.values()
                ]
            }

    def import_state(
        self,
        data: dict[str, Any],
        *,
        replace: bool = False,
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Timetable state must be a dictionary."
            )

        raw_entries = data.get(
            "entries",
            [],
        )

        if not isinstance(
            raw_entries,
            list,
        ):
            raise TypeError(
                "'entries' must be a list."
            )

        with self._lock:

            if replace:
                self._entries.clear()

            for raw in raw_entries:

                entry = TimetableEntry(
                    id=raw["id"],
                    title=raw["title"],
                    subject=raw.get(
                        "subject"
                    ),
                    entry_date=raw[
                        "entry_date"
                    ],
                    start_time=raw[
                        "start_time"
                    ],
                    end_time=raw[
                        "end_time"
                    ],
                    entry_type=raw.get(
                        "entry_type",
                        "study",
                    ),
                    room=raw.get(
                        "room"
                    ),
                    teacher=raw.get(
                        "teacher"
                    ),
                    notes=raw.get(
                        "notes",
                        "",
                    ),
                    recurrence=raw.get(
                        "recurrence",
                        "none",
                    ),
                    weekdays=raw.get(
                        "weekdays",
                        [],
                    ),
                    completed=raw.get(
                        "completed",
                        False,
                    ),
                    cancelled=raw.get(
                        "cancelled",
                        False,
                    ),
                    created_at=raw.get(
                        "created_at",
                        self._now().isoformat(),
                    ),
                )

                self._entries[
                    entry.id
                ] = entry

    # ========================================================
    # INTERNAL DATE LOGIC
    # ========================================================

    def _entry_occurs_on(
        self,
        entry: TimetableEntry,
        target: date,
    ) -> bool:

        entry_date = datetime.strptime(
            entry.entry_date,
            "%Y-%m-%d",
        ).date()

        if target < entry_date:
            return False

        if entry.recurrence == "none":
            return target == entry_date

        if entry.recurrence == "daily":
            return True

        if entry.recurrence == "weekdays":
            return target.weekday() < 5

        if entry.recurrence == "weekly":

            return (
                target.weekday()
                in entry.weekdays
            )

        return False

    def _candidate_dates_for_conflict(
        self,
        entry: TimetableEntry,
        *,
        horizon_days: int = 14,
    ) -> set[date]:

        start = datetime.strptime(
            entry.entry_date,
            "%Y-%m-%d",
        ).date()

        dates: set[date] = set()

        if entry.recurrence == "none":

            dates.add(start)
            return dates

        for offset in range(
            horizon_days
        ):

            target = (
                start
                + timedelta(days=offset)
            )

            if self._entry_occurs_on(
                entry,
                target,
            ):
                dates.add(target)

        return dates

    # ========================================================
    # STORAGE
    # ========================================================

    def _persist(
        self,
        entry: TimetableEntry,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_timetable_entry",
            None,
        )

        if callable(method):

            try:

                method(
                    entry.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist timetable entry."
                )

    def _delete_persisted(
        self,
        entry_id: str,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "delete_timetable_entry",
            None,
        )

        if callable(method):

            try:

                method(entry_id)

            except Exception:

                logger.exception(
                    "Failed to delete persisted "
                    "timetable entry."
                )

    # ========================================================
    # EVENT BUS
    # ========================================================

    def _emit(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:

        if self.event_bus is None:
            return

        try:

            publish = getattr(
                self.event_bus,
                "publish",
                None,
            )

            if callable(publish):

                publish(
                    event_name,
                    payload,
                )

        except Exception:

            logger.exception(
                "Failed to publish event: %s",
                event_name,
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _require_entry(
        self,
        entry_id: str,
    ) -> TimetableEntry:

        entry = self._entries.get(
            entry_id
        )

        if entry is None:
            raise KeyError(
                f"Timetable entry not found: "
                f"{entry_id}"
            )

        return entry

    @staticmethod
    def _clean_text(
        value: str,
        field_name: str,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return value

    @staticmethod
    def _new_id(
        prefix: str,
    ) -> str:

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    @staticmethod
    def _serialize_date(
        value: date | datetime | str,
    ) -> str:

        if isinstance(
            value,
            datetime,
        ):
            return value.date().isoformat()

        if isinstance(
            value,
            date,
        ):
            return value.isoformat()

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            try:

                parsed = datetime.strptime(
                    value,
                    "%Y-%m-%d",
                )

            except ValueError as exc:

                raise ValueError(
                    "Date must use YYYY-MM-DD."
                ) from exc

            return parsed.date().isoformat()

        raise TypeError(
            "entry_date must be a date, "
            "datetime, or YYYY-MM-DD string."
        )

    @staticmethod
    def _parse_date(
        value: date | datetime | str,
    ) -> date:

        normalized = (
            TimetableManager._serialize_date(
                value
            )
        )

        return datetime.strptime(
            normalized,
            "%Y-%m-%d",
        ).date()

    @staticmethod
    def _serialize_time(
        value: time | str,
    ) -> str:

        if isinstance(
            value,
            time,
        ):
            return value.strftime(
                "%H:%M"
            )

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            formats = [
                "%H:%M",
                "%H:%M:%S",
                "%I:%M %p",
                "%I:%M%p",
            ]

            for fmt in formats:

                try:

                    parsed = datetime.strptime(
                        value,
                        fmt,
                    )

                    return parsed.strftime(
                        "%H:%M"
                    )

                except ValueError:
                    continue

        raise ValueError(
            "Time must be in HH:MM, HH:MM:SS, "
            "or a valid AM/PM format."
        )

    @staticmethod
    def _normalize_weekdays(
        weekdays: Iterable[int] | None,
    ) -> list[int]:

        if weekdays is None:
            return []

        normalized = set()

        for weekday in weekdays:

            if not isinstance(
                weekday,
                int,
            ):
                raise TypeError(
                    "Weekdays must contain integers."
                )

            if weekday < 0 or weekday > 6:
                raise ValueError(
                    "Weekday must be between 0 and 6."
                )

            normalized.add(weekday)

        return sorted(normalized)

    # ========================================================
    # SORTING / TIME
    # ========================================================

    @staticmethod
    def _sort_key(
        entry: TimetableEntry,
    ) -> tuple[str, str, str]:

        return (
            entry.entry_date,
            entry.start_time,
            entry.title.casefold(),
        )

    @staticmethod
    def _duration_minutes(
        start_time: str,
        end_time: str,
    ) -> int:

        start = _time_to_minutes(
            start_time
        )

        end = _time_to_minutes(
            end_time
        )

        return end - start

    @staticmethod
    def _times_overlap(
        start_a: int,
        end_a: int,
        start_b: int,
        end_b: int,
    ) -> bool:

        return (
            start_a < end_b
            and start_b < end_a
        )

    @staticmethod
    def _format_conflict_error(
        conflicts: list[TimetableEntry],
    ) -> str:

        details = ", ".join(
            (
                f"{entry.title} "
                f"({entry.start_time}-"
                f"{entry.end_time})"
            )
            for entry in conflicts
        )

        return (
            "Timetable conflict detected: "
            + details
        )


# ============================================================
# MODULE HELPERS
# ============================================================


def _parse_time(
    value: str,
) -> time:

    return datetime.strptime(
        value,
        "%H:%M",
    ).time()


def _time_to_minutes(
    value: str,
) -> int:

    parsed = _parse_time(
        value
    )

    return (
        parsed.hour * 60
        + parsed.minute
    )


__all__ = [
    "TimetableEntry",
    "TimetableStatistics",
    "TimetableManager",
]


