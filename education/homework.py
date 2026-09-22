"""
RENIX Education — Homework Manager
===================================

Manages homework, assignments, deadlines, priorities,
completion status, subjects, reminders and statistics.

This module is designed to work with:

    RENIX Core
    RENIX Education
    RENIX Memory
    RENIX Notifications
    RENIX Database
    RENIX Automation

It does not require those systems to be present.
They can be injected when available.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

VALID_PRIORITIES = {
    "low",
    "medium",
    "high",
    "urgent",
}

VALID_STATUSES = {
    "pending",
    "in_progress",
    "completed",
    "overdue",
    "cancelled",
}

VALID_TYPES = {
    "homework",
    "assignment",
    "project",
    "worksheet",
    "practice",
    "revision",
    "reading",
    "other",
}


# Higher number = higher priority.
PRIORITY_SCORE = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "urgent": 4,
}


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class Homework:
    """
    Represents a single homework/assignment item.
    """

    id: str

    title: str
    subject: str

    due_date: str
    due_time: str | None = None

    description: str = ""

    homework_type: str = "homework"
    priority: str = "medium"
    status: str = "pending"

    estimated_minutes: int = 30

    teacher: str | None = None
    chapter: str | None = None
    topic: str | None = None

    notes: str = ""

    tags: list[str] = field(
        default_factory=list
    )

    attachments: list[str] = field(
        default_factory=list
    )

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    completed_at: str | None = None

    reminder_enabled: bool = True

    reminder_minutes_before: int = 60

    progress: int = 0

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_cancelled(self) -> bool:
        return self.status == "cancelled"

    @property
    def is_overdue(self) -> bool:

        if self.is_completed:
            return False

        if self.is_cancelled:
            return False

        deadline = self.deadline_datetime()

        return datetime.now() > deadline

    def deadline_datetime(self) -> datetime:

        due_date = datetime.strptime(
            self.due_date,
            "%Y-%m-%d",
        ).date()

        if self.due_time:

            due_time = datetime.strptime(
                self.due_time,
                "%H:%M",
            ).time()

        else:

            due_time = datetime.max.time().replace(
                microsecond=0
            )

        return datetime.combine(
            due_date,
            due_time,
        )

    def to_dict(self) -> dict[str, Any]:

        return asdict(self)


@dataclass
class HomeworkStatistics:
    """
    Statistics for homework.
    """

    total: int = 0

    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    overdue: int = 0
    cancelled: int = 0

    total_estimated_minutes: int = 0
    completed_minutes: int = 0

    completion_rate: float = 0.0

    def calculate_completion_rate(
        self,
    ) -> None:

        if self.total <= 0:

            self.completion_rate = 0.0

            return

        self.completion_rate = (
            self.completed
            / self.total
        ) * 100

    def to_dict(self) -> dict[str, Any]:

        return asdict(self)


# ============================================================
# HOMEWORK MANAGER
# ============================================================


class HomeworkManager:
    """
    Main RENIX homework management system.

    Example:

        manager = HomeworkManager()

        homework = manager.add_homework(
            title="Quadratic Equations",
            subject="Mathematics",
            due_date="2026-08-25",
            estimated_minutes=60,
            priority="high",
        )

        manager.start_homework(
            homework.id
        )

        manager.complete_homework(
            homework.id
        )
    """

    def __init__(
        self,
        *,
        storage: Any | None = None,
        event_bus: Any | None = None,
        notification_manager: Any | None = None,
        clock: Callable[
            [],
            datetime
        ] | None = None,
    ) -> None:

        self.storage = storage
        self.event_bus = event_bus
        self.notification_manager = (
            notification_manager
        )

        self._clock = (
            clock
            if clock is not None
            else datetime.now
        )

        self._homework: dict[
            str,
            Homework,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "RENIX HomeworkManager initialized."
        )

    # ========================================================
    # CREATE
    # ========================================================

    def add_homework(
        self,
        *,
        title: str,
        subject: str,
        due_date: date | datetime | str,
        description: str = "",
        due_time: str | None = None,
        homework_type: str = "homework",
        priority: str = "medium",
        estimated_minutes: int = 30,
        teacher: str | None = None,
        chapter: str | None = None,
        topic: str | None = None,
        notes: str = "",
        tags: Iterable[str] | None = None,
        attachments: Iterable[str] | None = None,
        reminder_enabled: bool = True,
        reminder_minutes_before: int = 60,
        allow_duplicate: bool = False,
    ) -> Homework:

        title = self._required_text(
            title,
            "title",
        )

        subject = self._required_text(
            subject,
            "subject",
        )

        normalized_date = (
            self._serialize_date(
                due_date
            )
        )

        normalized_time = (
            self._serialize_time(
                due_time
            )
            if due_time
            else None
        )

        priority = (
            priority.strip().lower()
        )

        if priority not in VALID_PRIORITIES:

            raise ValueError(
                f"Invalid priority: {priority}"
            )

        homework_type = (
            homework_type.strip().lower()
        )

        if (
            homework_type
            not in VALID_TYPES
        ):

            raise ValueError(
                f"Invalid homework type: "
                f"{homework_type}"
            )

        if not isinstance(
            estimated_minutes,
            int,
        ):

            raise TypeError(
                "estimated_minutes must be an integer."
            )

        if estimated_minutes <= 0:

            raise ValueError(
                "estimated_minutes must be greater than zero."
            )

        if (
            not isinstance(
                reminder_minutes_before,
                int,
            )
            or reminder_minutes_before < 0
        ):

            raise ValueError(
                "reminder_minutes_before must "
                "be a non-negative integer."
            )

        normalized_tags = (
            self._normalize_list(
                tags
            )
        )

        normalized_attachments = (
            self._normalize_list(
                attachments
            )
        )

        homework = Homework(
            id=self._new_id(),
            title=title,
            subject=subject,
            due_date=normalized_date,
            due_time=normalized_time,
            description=description.strip(),
            homework_type=homework_type,
            priority=priority,
            status="pending",
            estimated_minutes=estimated_minutes,
            teacher=(
                teacher.strip()
                if teacher
                else None
            ),
            chapter=(
                chapter.strip()
                if chapter
                else None
            ),
            topic=(
                topic.strip()
                if topic
                else None
            ),
            notes=notes.strip(),
            tags=normalized_tags,
            attachments=normalized_attachments,
            reminder_enabled=reminder_enabled,
            reminder_minutes_before=(
                reminder_minutes_before
            ),
        )

        with self._lock:

            if not allow_duplicate:

                duplicates = (
                    self.find_duplicates(
                        homework
                    )
                )

                if duplicates:

                    raise ValueError(
                        "Similar homework already exists."
                    )

            self._homework[
                homework.id
            ] = homework

            self._persist(homework)

        self._schedule_reminder(
            homework
        )

        self._emit(
            "education.homework.created",
            homework.to_dict(),
        )

        return homework

    # ========================================================
    # READ
    # ========================================================

    def get_homework(
        self,
        homework_id: str,
    ) -> Homework | None:

        with self._lock:

            return self._homework.get(
                homework_id
            )

    def get_all(
        self,
    ) -> list[Homework]:

        with self._lock:

            items = list(
                self._homework.values()
            )

        return self._sort(
            items
        )

    # ========================================================
    # FILTERING
    # ========================================================

    def get_pending(
        self,
    ) -> list[Homework]:

        return self.get_all(
        )[0:] if False else self._filter(
            status="pending"
        )

    def get_in_progress(
        self,
    ) -> list[Homework]:

        return self._filter(
            status="in_progress"
        )

    def get_completed(
        self,
    ) -> list[Homework]:

        return self._filter(
            status="completed"
        )

    def get_cancelled(
        self,
    ) -> list[Homework]:

        return self._filter(
            status="cancelled"
        )

    def get_overdue(
        self,
    ) -> list[Homework]:

        now = self._now()

        with self._lock:

            items = list(
                self._homework.values()
            )

        overdue = []

        for homework in items:

            if homework.completed:
                continue

            if homework.cancelled:
                continue

            if (
                homework.deadline_datetime()
                < now
            ):

                overdue.append(
                    homework
                )

        return self._sort(
            overdue
        )

    # ========================================================
    # SUBJECT
    # ========================================================

    def get_by_subject(
        self,
        subject: str,
    ) -> list[Homework]:

        subject = self._required_text(
            subject,
            "subject",
        ).casefold()

        with self._lock:

            items = [
                homework
                for homework
                in self._homework.values()
                if homework.subject.casefold()
                == subject
            ]

        return self._sort(
            items
        )

    # ========================================================
    # DEADLINES
    # ========================================================

    def get_due_today(
        self,
    ) -> list[Homework]:

        today = self._now().date()

        return self.get_due_on(
            today
        )

    def get_due_tomorrow(
        self,
    ) -> list[Homework]:

        tomorrow = (
            self._now().date()
            + timedelta(days=1)
        )

        return self.get_due_on(
            tomorrow
        )

    def get_due_on(
        self,
        target_date: date | datetime | str,
    ) -> list[Homework]:

        normalized = (
            self._serialize_date(
                target_date
            )
        )

        with self._lock:

            items = [
                homework
                for homework
                in self._homework.values()
                if (
                    homework.due_date
                    == normalized
                )
                and not homework.cancelled
            ]

        return self._sort(
            items
        )

    def get_due_between(
        self,
        start_date: date | datetime | str,
        end_date: date | datetime | str,
    ) -> list[Homework]:

        start = self._parse_date(
            start_date
        )

        end = self._parse_date(
            end_date
        )

        if end < start:

            raise ValueError(
                "end_date cannot be before start_date."
            )

        with self._lock:

            items = list(
                self._homework.values()
            )

        results = []

        for homework in items:

            if homework.cancelled:
                continue

            due = self._parse_date(
                homework.due_date
            )

            if start <= due <= end:

                results.append(
                    homework
                )

        return self._sort(
            results
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update_homework(
        self,
        homework_id: str,
        *,
        title: str | None = None,
        subject: str | None = None,
        due_date: date | datetime | str | None = None,
        due_time: str | None = None,
        description: str | None = None,
        homework_type: str | None = None,
        priority: str | None = None,
        estimated_minutes: int | None = None,
        teacher: str | None = None,
        chapter: str | None = None,
        topic: str | None = None,
        notes: str | None = None,
        tags: Iterable[str] | None = None,
        attachments: Iterable[str] | None = None,
        reminder_enabled: bool | None = None,
        reminder_minutes_before: int | None = None,
    ) -> Homework:

        with self._lock:

            homework = self._require(
                homework_id
            )

            if title is not None:

                homework.title = (
                    self._required_text(
                        title,
                        "title",
                    )
                )

            if subject is not None:

                homework.subject = (
                    self._required_text(
                        subject,
                        "subject",
                    )
                )

            if due_date is not None:

                homework.due_date = (
                    self._serialize_date(
                        due_date
                    )
                )

            if due_time is not None:

                homework.due_time = (
                    self._serialize_time(
                        due_time
                    )
                )

            if description is not None:

                homework.description = (
                    description.strip()
                )

            if homework_type is not None:

                homework_type = (
                    homework_type.strip().lower()
                )

                if (
                    homework_type
                    not in VALID_TYPES
                ):

                    raise ValueError(
                        f"Invalid homework type: "
                        f"{homework_type}"
                    )

                homework.homework_type = (
                    homework_type
                )

            if priority is not None:

                priority = (
                    priority.strip().lower()
                )

                if (
                    priority
                    not in VALID_PRIORITIES
                ):

                    raise ValueError(
                        f"Invalid priority: "
                        f"{priority}"
                    )

                homework.priority = priority

            if estimated_minutes is not None:

                if (
                    not isinstance(
                        estimated_minutes,
                        int,
                    )
                    or estimated_minutes <= 0
                ):

                    raise ValueError(
                        "estimated_minutes must "
                        "be greater than zero."
                    )

                homework.estimated_minutes = (
                    estimated_minutes
                )

            if teacher is not None:

                homework.teacher = (
                    teacher.strip()
                    or None
                )

            if chapter is not None:

                homework.chapter = (
                    chapter.strip()
                    or None
                )

            if topic is not None:

                homework.topic = (
                    topic.strip()
                    or None
                )

            if notes is not None:

                homework.notes = (
                    notes.strip()
                )

            if tags is not None:

                homework.tags = (
                    self._normalize_list(
                        tags
                    )
                )

            if attachments is not None:

                homework.attachments = (
                    self._normalize_list(
                        attachments
                    )
                )

            if reminder_enabled is not None:

                homework.reminder_enabled = (
                    reminder_enabled
                )

            if reminder_minutes_before is not None:

                if (
                    not isinstance(
                        reminder_minutes_before,
                        int,
                    )
                    or reminder_minutes_before < 0
                ):

                    raise ValueError(
                        "reminder_minutes_before "
                        "must be non-negative."
                    )

                homework.reminder_minutes_before = (
                    reminder_minutes_before
                )

            homework.updated_at = (
                self._now().isoformat()
            )

            self._persist(
                homework
            )

        self._schedule_reminder(
            homework
        )

        self._emit(
            "education.homework.updated",
            homework.to_dict(),
        )

        return homework

    # ========================================================
    # STATUS MANAGEMENT
    # ========================================================

    def start_homework(
        self,
        homework_id: str,
    ) -> Homework:

        with self._lock:

            homework = self._require(
                homework_id
            )

            if homework.cancelled:

                raise RuntimeError(
                    "Cancelled homework cannot be started."
                )

            if homework.completed:

                raise RuntimeError(
                    "Completed homework cannot be started."
                )

            homework.status = (
                "in_progress"
            )

            homework.progress = max(
                homework.progress,
                1,
            )

            homework.updated_at = (
                self._now().isoformat()
            )

            self._persist(
                homework
            )

        self._emit(
            "education.homework.started",
            homework.to_dict(),
        )

        return homework

    def complete_homework(
        self,
        homework_id: str,
    ) -> Homework:

        with self._lock:

            homework = self._require(
                homework_id
            )

            if homework.cancelled:

                raise RuntimeError(
                    "Cancelled homework cannot be completed."
                )

            homework.status = (
                "completed"
            )

            homework.progress = 100

            homework.completed_at = (
                self._now().isoformat()
            )

            homework.updated_at = (
                self._now().isoformat()
            )

            self._persist(
                homework
            )

        self._emit(
            "education.homework.completed",
            homework.to_dict(),
        )

        return homework

    def reopen_homework(
        self,
        homework_id: str,
    ) -> Homework:

        with self._lock:

            homework = self._require(
                homework_id
            )

            if homework.cancelled:

                raise RuntimeError(
                    "Cancelled homework cannot be reopened."
                )

            homework.status = (
                "pending"
            )

            homework.completed_at = None

            homework.progress = min(
                homework.progress,
                99,
            )

            homework.updated_at = (
                self._now().isoformat()
            )

            self._persist(
                homework
            )

        self._emit(
            "education.homework.reopened",
            homework.to_dict(),
        )

        return homework

    def cancel_homework(
        self,
        homework_id: str,
    ) -> Homework:

        with self._lock:

            homework = self._require(
                homework_id
            )

            homework.status = (
                "cancelled"
            )

            homework.progress = 0

            homework.updated_at = (
                self._now().isoformat()
            )

            self._persist(
                homework
            )

        self._emit(
            "education.homework.cancelled",
            homework.to_dict(),
        )

        return homework

    # ========================================================
    # PROGRESS
    # ========================================================

    def update_progress(
        self,
        homework_id: str,
        progress: int,
    ) -> Homework:

        if not isinstance(
            progress,
            int,
        ):

            raise TypeError(
                "progress must be an integer."
            )

        if not 0 <= progress <= 100:

            raise ValueError(
                "progress must be between 0 and 100."
            )

        with self._lock:

            homework = self._require(
                homework_id
            )

            if homework.cancelled:

                raise RuntimeError(
                    "Cancelled homework cannot "
                    "have progress updated."
                )

            homework.progress = progress

            if progress == 100:

                homework.status = (
                    "completed"
                )

                homework.completed_at = (
                    self._now().isoformat()
                )

            elif progress > 0:

                homework.status = (
                    "in_progress"
                )

            else:

                homework.status = (
                    "pending"
                )

            homework.updated_at = (
                self._now().isoformat()
            )

            self._persist(
                homework
            )

        self._emit(
            "education.homework.progress_updated",
            homework.to_dict(),
        )

        return homework

    # ========================================================
    # PRIORITY
    # ========================================================

    def get_priority_queue(
        self,
        *,
        include_completed: bool = False,
    ) -> list[Homework]:

        with self._lock:

            items = list(
                self._homework.values()
            )

        if not include_completed:

            items = [
                item
                for item in items
                if not item.completed
                and not item.cancelled
            ]

        now = self._now()

        def queue_key(
            homework: Homework,
        ) -> tuple:

            deadline = (
                homework.deadline_datetime()
            )

            overdue = (
                deadline < now
                and not homework.completed
            )

            return (
                0 if overdue else 1,
                -PRIORITY_SCORE[
                    homework.priority
                ],
                deadline,
                homework.title.casefold(),
            )

        return sorted(
            items,
            key=queue_key,
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
    ) -> list[Homework]:

        query = self._required_text(
            query,
            "query",
        ).casefold()

        with self._lock:

            items = list(
                self._homework.values()
            )

        results = []

        for homework in items:

            searchable = " ".join(
                [
                    homework.title,
                    homework.subject,
                    homework.description,
                    homework.homework_type,
                    homework.priority,
                    homework.teacher or "",
                    homework.chapter or "",
                    homework.topic or "",
                    homework.notes,
                    " ".join(
                        homework.tags
                    ),
                ]
            ).casefold()

            if query in searchable:

                results.append(
                    homework
                )

        return self._sort(
            results
        )

    # ========================================================
    # TAGS
    # ========================================================

    def add_tag(
        self,
        homework_id: str,
        tag: str,
    ) -> Homework:

        tag = self._required_text(
            tag,
            "tag",
        )

        with self._lock:

            homework = self._require(
                homework_id
            )

            existing = {
                item.casefold()
                for item in homework.tags
            }

            if tag.casefold() not in existing:

                homework.tags.append(
                    tag
                )

            homework.updated_at = (
                self._now().isoformat()
            )

            self._persist(
                homework
            )

        return homework

    def remove_tag(
        self,
        homework_id: str,
        tag: str,
    ) -> Homework:

        tag = self._required_text(
            tag,
            "tag",
        )

        with self._lock:

            homework = self._require(
                homework_id
            )

            homework.tags = [
                item
                for item in homework.tags
                if item.casefold()
                != tag.casefold()
            ]

            homework.updated_at = (
                self._now().isoformat()
            )

            self._persist(
                homework
            )

        return homework

    # ========================================================
    # DUPLICATE DETECTION
    # ========================================================

    def find_duplicates(
        self,
        candidate: Homework,
    ) -> list[Homework]:

        with self._lock:

            items = list(
                self._homework.values()
            )

        results = []

        candidate_title = (
            candidate.title.casefold()
        )

        candidate_subject = (
            candidate.subject.casefold()
        )

        for item in items:

            if item.id == candidate.id:
                continue

            if item.cancelled:
                continue

            if (
                item.title.casefold()
                == candidate_title
                and item.subject.casefold()
                == candidate_subject
                and item.due_date
                == candidate.due_date
            ):

                results.append(
                    item
                )

        return results

    # ========================================================
    # DELETE
    # ========================================================

    def delete_homework(
        self,
        homework_id: str,
    ) -> bool:

        with self._lock:

            homework = self._homework.pop(
                homework_id,
                None,
            )

            if homework is None:
                return False

        self._delete_persisted(
            homework_id
        )

        self._emit(
            "education.homework.deleted",
            {
                "homework_id": homework_id,
                "homework": homework.to_dict(),
            },
        )

        return True

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
        *,
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
    ) -> HomeworkStatistics:

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

        items = self.get_due_between(
            start,
            end,
        )

        stats = HomeworkStatistics()

        stats.total = len(
            items
        )

        for homework in items:

            status = homework.status

            if status == "pending":
                stats.pending += 1

            elif status == "in_progress":
                stats.in_progress += 1

            elif status == "completed":
                stats.completed += 1

            elif status == "overdue":
                stats.overdue += 1

            elif status == "cancelled":
                stats.cancelled += 1

            if not homework.cancelled:

                stats.total_estimated_minutes += (
                    homework.estimated_minutes
                )

            if homework.completed:

                stats.completed_minutes += (
                    homework.estimated_minutes
                )

        # Recalculate overdue dynamically.
        stats.overdue = len(
            [
                item
                for item in items
                if item.is_overdue
            ]
        )

        stats.calculate_completion_rate()

        return stats

    # ========================================================
    # TODAY SUMMARY
    # ========================================================

    def today_summary(
        self,
    ) -> dict[str, Any]:

        today = self.get_due_today()

        pending = [
            item
            for item in today
            if not item.completed
            and not item.cancelled
        ]

        completed = [
            item
            for item in today
            if item.completed
        ]

        overdue = [
            item
            for item in pending
            if item.is_overdue
        ]

        return {
            "date": self._now()
            .date()
            .isoformat(),

            "total": len(today),

            "pending": len(pending),

            "completed": len(completed),

            "overdue": len(overdue),

            "estimated_minutes": sum(
                item.estimated_minutes
                for item in pending
            ),

            "homework": [
                item.to_dict()
                for item in today
            ],
        }

    # ========================================================
    # IMPORT / EXPORT
    # ========================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "homework": [
                    item.to_dict()
                    for item
                    in self._homework.values()
                ]
            }

    def import_state(
        self,
        data: dict[str, Any],
        *,
        replace: bool = False,
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "Homework state must be a dictionary."
            )

        raw_items = data.get(
            "homework",
            [],
        )

        if not isinstance(
            raw_items,
            list,
        ):

            raise TypeError(
                "'homework' must be a list."
            )

        with self._lock:

            if replace:
                self._homework.clear()

            for raw in raw_items:

                item = Homework(
                    id=raw["id"],
                    title=raw["title"],
                    subject=raw["subject"],
                    due_date=raw["due_date"],
                    due_time=raw.get(
                        "due_time"
                    ),
                    description=raw.get(
                        "description",
                        "",
                    ),
                    homework_type=raw.get(
                        "homework_type",
                        "homework",
                    ),
                    priority=raw.get(
                        "priority",
                        "medium",
                    ),
                    status=raw.get(
                        "status",
                        "pending",
                    ),
                    estimated_minutes=raw.get(
                        "estimated_minutes",
                        30,
                    ),
                    teacher=raw.get(
                        "teacher"
                    ),
                    chapter=raw.get(
                        "chapter"
                    ),
                    topic=raw.get(
                        "topic"
                    ),
                    notes=raw.get(
                        "notes",
                        "",
                    ),
                    tags=raw.get(
                        "tags",
                        [],
                    ),
                    attachments=raw.get(
                        "attachments",
                        [],
                    ),
                    created_at=raw.get(
                        "created_at",
                        self._now().isoformat(),
                    ),
                    updated_at=raw.get(
                        "updated_at",
                        self._now().isoformat(),
                    ),
                    completed_at=raw.get(
                        "completed_at"
                    ),
                    reminder_enabled=raw.get(
                        "reminder_enabled",
                        True,
                    ),
                    reminder_minutes_before=raw.get(
                        "reminder_minutes_before",
                        60,
                    ),
                    progress=raw.get(
                        "progress",
                        0,
                    ),
                )

                self._homework[
                    item.id
                ] = item

    # ========================================================
    # REMINDERS
    # ========================================================

    def _schedule_reminder(
        self,
        homework: Homework,
    ) -> None:

        if not homework.reminder_enabled:
            return

        if homework.completed:
            return

        if homework.cancelled:
            return

        manager = (
            self.notification_manager
        )

        if manager is None:
            return

        try:

            method = getattr(
                manager,
                "schedule",
                None,
            )

            if not callable(method):
                return

            deadline = (
                homework.deadline_datetime()
            )

            reminder_at = (
                deadline
                - timedelta(
                    minutes=(
                        homework
                        .reminder_minutes_before
                    )
                )
            )

            method(
                notification_id=(
                    f"homework_"
                    f"{homework.id}"
                ),
                title=(
                    f"Homework due: "
                    f"{homework.title}"
                ),
                message=(
                    f"{homework.subject} "
                    f"homework is due "
                    f"at {deadline.strftime('%H:%M')}."
                ),
                scheduled_at=reminder_at,
                metadata={
                    "homework_id": homework.id,
                    "type": "homework",
                },
            )

        except Exception:

            logger.exception(
                "Failed to schedule homework reminder."
            )

    # ========================================================
    # STORAGE
    # ========================================================

    def _persist(
        self,
        homework: Homework,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_homework",
            None,
        )

        if not callable(method):
            return

        try:

            method(
                homework.to_dict()
            )

        except Exception:

            logger.exception(
                "Failed to persist homework."
            )

    def _delete_persisted(
        self,
        homework_id: str,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "delete_homework",
            None,
        )

        if not callable(method):
            return

        try:

            method(
                homework_id
            )

        except Exception:

            logger.exception(
                "Failed to delete persisted homework."
            )

    # ========================================================
    # EVENTS
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
    # INTERNAL HELPERS
    # ========================================================

    def _filter(
        self,
        *,
        status: str,
    ) -> list[Homework]:

        with self._lock:

            items = [
                item
                for item
                in self._homework.values()
                if item.status == status
            ]

        return self._sort(
            items
        )

    def _require(
        self,
        homework_id: str,
    ) -> Homework:

        homework = self._homework.get(
            homework_id
        )

        if homework is None:

            raise KeyError(
                f"Homework not found: "
                f"{homework_id}"
            )

        return homework

    @staticmethod
    def _required_text(
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
    def _new_id() -> str:

        return (
            "hw_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _normalize_list(
        values: Iterable[str] | None,
    ) -> list[str]:

        if values is None:
            return []

        result = []

        for value in values:

            if not isinstance(
                value,
                str,
            ):

                raise TypeError(
                    "List values must be strings."
                )

            value = value.strip()

            if value and value not in result:

                result.append(
                    value
                )

        return result

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

            try:

                parsed = datetime.strptime(
                    value.strip(),
                    "%Y-%m-%d",
                )

            except ValueError as exc:

                raise ValueError(
                    "Date must use YYYY-MM-DD."
                ) from exc

            return parsed.date().isoformat()

        raise TypeError(
            "Date must be a date, datetime, "
            "or YYYY-MM-DD string."
        )

    @staticmethod
    def _parse_date(
        value: date | datetime | str,
    ) -> date:

        normalized = (
            HomeworkManager._serialize_date(
                value
            )
        )

        return datetime.strptime(
            normalized,
            "%Y-%m-%d",
        ).date()

    @staticmethod
    def _serialize_time(
        value: str,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):

            raise TypeError(
                "Time must be a string."
            )

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
            "Invalid time. Use HH:MM or AM/PM format."
        )

    @staticmethod
    def _sort(
        items: list[Homework],
    ) -> list[Homework]:

        return sorted(
            items,
            key=lambda item: (
                item.due_date,
                item.due_time or "23:59",
                -PRIORITY_SCORE.get(
                    item.priority,
                    0,
                ),
                item.title.casefold(),
            ),
        )

    def _now(self) -> datetime:

        return self._clock()

    # ========================================================
    # STRING REPRESENTATION
    # ========================================================

    def __len__(self) -> int:

        with self._lock:
            return len(
                self._homework
            )

    def __contains__(
        self,
        homework_id: str,
    ) -> bool:

        with self._lock:
            return homework_id in self._homework


__all__ = [
    "Homework",
    "HomeworkStatistics",
    "HomeworkManager",
    "VALID_PRIORITIES",
    "VALID_STATUSES",
    "VALID_TYPES",
]


