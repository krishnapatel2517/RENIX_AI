"""
RENIX Study Manager
===================

Central academic study-management system for RENIX.

Responsibilities:
    - Create and manage study sessions
    - Track subjects and chapters
    - Track study time
    - Manage study goals
    - Calculate study progress
    - Maintain current-study state
    - Generate study summaries
    - Coordinate with timetable, homework, revision,
      quiz and progress systems

This module intentionally keeps persistent storage abstract.
The database/memory systems can be connected later through
RENIX's service registry.
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
# DATA MODELS
# ============================================================


@dataclass
class StudyTopic:
    """Represents a chapter or topic that can be studied."""

    id: str
    subject: str
    name: str
    target_minutes: int = 60
    completed_minutes: int = 0
    completed: bool = False
    priority: int = 3
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    completed_at: str | None = None

    @property
    def progress(self) -> float:
        """Return topic completion percentage."""

        if self.target_minutes <= 0:
            return 100.0 if self.completed else 0.0

        percentage = (
            self.completed_minutes
            / self.target_minutes
        ) * 100

        return min(100.0, max(0.0, percentage))

    def add_study_time(
        self,
        minutes: int,
    ) -> None:
        """Add study time to the topic."""

        if minutes <= 0:
            raise ValueError(
                "Study time must be greater than zero."
            )

        self.completed_minutes += minutes

        if (
            self.completed_minutes
            >= self.target_minutes
        ):
            self.completed = True
            self.completed_at = (
                datetime.now().isoformat()
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert topic to a serializable dictionary."""

        data = asdict(self)
        data["progress"] = self.progress
        return data


@dataclass
class StudySession:
    """Represents an individual study session."""

    id: str
    subject: str
    topic: str
    planned_minutes: int
    actual_minutes: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    status: str = "planned"
    notes: str = ""
    productivity_score: float | None = None

    @property
    def completion_percentage(self) -> float:
        """Calculate planned-vs-actual completion."""

        if self.planned_minutes <= 0:
            return 0.0

        return min(
            100.0,
            (
                self.actual_minutes
                / self.planned_minutes
            )
            * 100,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data[
            "completion_percentage"
        ] = self.completion_percentage
        return data


@dataclass
class StudyGoal:
    """Represents an academic study goal."""

    id: str
    title: str
    subject: str | None
    target_value: float
    current_value: float = 0.0
    unit: str = "minutes"
    deadline: str | None = None
    completed: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def progress(self) -> float:
        if self.target_value <= 0:
            return 100.0 if self.completed else 0.0

        return min(
            100.0,
            (
                self.current_value
                / self.target_value
            )
            * 100,
        )

    def update(
        self,
        value: float,
    ) -> None:

        if value < 0:
            raise ValueError(
                "Goal value cannot be negative."
            )

        self.current_value = value

        if (
            self.current_value
            >= self.target_value
        ):
            self.completed = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["progress"] = self.progress
        return data


@dataclass
class StudyStatistics:
    """Aggregate academic study statistics."""

    total_sessions: int = 0
    completed_sessions: int = 0
    cancelled_sessions: int = 0
    total_planned_minutes: int = 0
    total_studied_minutes: int = 0
    topics_completed: int = 0
    topics_total: int = 0
    average_session_minutes: float = 0.0
    average_productivity: float = 0.0

    @property
    def study_completion_rate(self) -> float:
        if self.total_planned_minutes <= 0:
            return 0.0

        return min(
            100.0,
            (
                self.total_studied_minutes
                / self.total_planned_minutes
            )
            * 100,
        )

    @property
    def topic_completion_rate(self) -> float:
        if self.topics_total <= 0:
            return 0.0

        return (
            self.topics_completed
            / self.topics_total
        ) * 100

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data[
            "study_completion_rate"
        ] = self.study_completion_rate
        data[
            "topic_completion_rate"
        ] = self.topic_completion_rate
        return data


# ============================================================
# STUDY MANAGER
# ============================================================


class StudyManager:
    """
    Central manager for RENIX academic activities.

    Example:

        manager = StudyManager()

        topic = manager.add_topic(
            subject="Mathematics",
            name="Quadratic Equations",
            target_minutes=90,
        )

        session = manager.create_session(
            subject="Mathematics",
            topic="Quadratic Equations",
            planned_minutes=60,
        )

        manager.start_session(session.id)

        manager.finish_session(
            session.id,
            actual_minutes=55,
        )
    """

    VALID_SESSION_STATUSES = {
        "planned",
        "running",
        "completed",
        "cancelled",
        "paused",
    }

    MAX_PRODUCTIVITY_SCORE = 10.0

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

        self._topics: dict[
            str,
            StudyTopic,
        ] = {}

        self._sessions: dict[
            str,
            StudySession,
        ] = {}

        self._goals: dict[
            str,
            StudyGoal,
        ] = {}

        self._active_session_id: str | None = None

        self._lock = threading.RLock()

        logger.info(
            "RENIX StudyManager initialized."
        )

    # ========================================================
    # TOPICS
    # ========================================================

    def add_topic(
        self,
        *,
        subject: str,
        name: str,
        target_minutes: int = 60,
        priority: int = 3,
        notes: str = "",
    ) -> StudyTopic:

        subject = self._clean_text(
            subject,
            "subject",
        )

        name = self._clean_text(
            name,
            "topic name",
        )

        if target_minutes <= 0:
            raise ValueError(
                "target_minutes must be greater than zero."
            )

        if priority < 1 or priority > 5:
            raise ValueError(
                "priority must be between 1 and 5."
            )

        topic = StudyTopic(
            id=self._new_id("topic"),
            subject=subject,
            name=name,
            target_minutes=target_minutes,
            priority=priority,
            notes=notes.strip(),
        )

        with self._lock:
            self._topics[topic.id] = topic
            self._persist_topic(topic)

        self._emit(
            "education.topic.created",
            topic.to_dict(),
        )

        return topic

    def get_topic(
        self,
        topic_id: str,
    ) -> StudyTopic | None:

        with self._lock:
            return self._topics.get(topic_id)

    def get_topics(
        self,
        *,
        subject: str | None = None,
        completed: bool | None = None,
    ) -> list[StudyTopic]:

        with self._lock:

            topics = list(
                self._topics.values()
            )

        if subject is not None:

            normalized = subject.casefold()

            topics = [
                topic
                for topic in topics
                if topic.subject.casefold()
                == normalized
            ]

        if completed is not None:

            topics = [
                topic
                for topic in topics
                if topic.completed
                == completed
            ]

        return sorted(
            topics,
            key=lambda item: (
                item.completed,
                item.priority,
                item.subject.casefold(),
                item.name.casefold(),
            ),
        )

    def complete_topic(
        self,
        topic_id: str,
    ) -> StudyTopic:

        topic = self._require_topic(
            topic_id
        )

        with self._lock:

            topic.completed = True
            topic.completed_minutes = max(
                topic.completed_minutes,
                topic.target_minutes,
            )

            topic.completed_at = (
                self._now().isoformat()
            )

            self._persist_topic(topic)

        self._emit(
            "education.topic.completed",
            topic.to_dict(),
        )

        return topic

    def add_topic_study_time(
        self,
        topic_id: str,
        minutes: int,
    ) -> StudyTopic:

        if minutes <= 0:
            raise ValueError(
                "minutes must be greater than zero."
            )

        topic = self._require_topic(
            topic_id
        )

        was_completed = topic.completed

        with self._lock:

            topic.add_study_time(
                minutes
            )

            self._persist_topic(topic)

        self._emit(
            "education.topic.time_added",
            {
                "topic": topic.to_dict(),
                "minutes": minutes,
            },
        )

        if (
            not was_completed
            and topic.completed
        ):

            self._emit(
                "education.topic.completed",
                topic.to_dict(),
            )

        return topic

    # ========================================================
    # STUDY SESSIONS
    # ========================================================

    def create_session(
        self,
        *,
        subject: str,
        topic: str,
        planned_minutes: int,
        notes: str = "",
    ) -> StudySession:

        subject = self._clean_text(
            subject,
            "subject",
        )

        topic = self._clean_text(
            topic,
            "topic",
        )

        if planned_minutes <= 0:
            raise ValueError(
                "planned_minutes must be greater than zero."
            )

        session = StudySession(
            id=self._new_id("session"),
            subject=subject,
            topic=topic,
            planned_minutes=planned_minutes,
            notes=notes.strip(),
        )

        with self._lock:

            self._sessions[
                session.id
            ] = session

            self._persist_session(
                session
            )

        self._emit(
            "education.session.created",
            session.to_dict(),
        )

        return session

    def get_session(
        self,
        session_id: str,
    ) -> StudySession | None:

        with self._lock:
            return self._sessions.get(
                session_id
            )

    def get_sessions(
        self,
        *,
        subject: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[StudySession]:

        with self._lock:

            sessions = list(
                self._sessions.values()
            )

        if subject is not None:

            normalized = subject.casefold()

            sessions = [
                session
                for session in sessions
                if session.subject.casefold()
                == normalized
            ]

        if status is not None:

            status = self._validate_status(
                status
            )

            sessions = [
                session
                for session in sessions
                if session.status
                == status
            ]

        sessions.sort(
            key=lambda item: item.started_at
            or "",
            reverse=True,
        )

        if limit is not None:

            if limit < 0:
                raise ValueError(
                    "limit cannot be negative."
                )

            sessions = sessions[
                :limit
            ]

        return sessions

    def start_session(
        self,
        session_id: str,
    ) -> StudySession:

        session = self._require_session(
            session_id
        )

        with self._lock:

            if (
                self._active_session_id
                is not None
                and self._active_session_id
                != session_id
            ):

                raise RuntimeError(
                    "Another study session is already running."
                )

            if session.status == "completed":
                raise RuntimeError(
                    "Completed sessions cannot be restarted."
                )

            if session.status == "cancelled":
                raise RuntimeError(
                    "Cancelled sessions cannot be restarted."
                )

            session.status = "running"

            session.started_at = (
                session.started_at
                or self._now().isoformat()
            )

            self._active_session_id = (
                session.id
            )

            self._persist_session(
                session
            )

        self._emit(
            "education.session.started",
            session.to_dict(),
        )

        return session

    def pause_session(
        self,
        session_id: str,
    ) -> StudySession:

        session = self._require_session(
            session_id
        )

        with self._lock:

            if session.status != "running":
                raise RuntimeError(
                    "Only a running session can be paused."
                )

            session.status = "paused"

            self._active_session_id = None

            self._persist_session(
                session
            )

        self._emit(
            "education.session.paused",
            session.to_dict(),
        )

        return session

    def resume_session(
        self,
        session_id: str,
    ) -> StudySession:

        session = self._require_session(
            session_id
        )

        with self._lock:

            if session.status != "paused":
                raise RuntimeError(
                    "Only a paused session can be resumed."
                )

            if (
                self._active_session_id
                is not None
            ):

                raise RuntimeError(
                    "Another study session is already running."
                )

            session.status = "running"

            self._active_session_id = (
                session.id
            )

            self._persist_session(
                session
            )

        self._emit(
            "education.session.resumed",
            session.to_dict(),
        )

        return session

    def finish_session(
        self,
        session_id: str,
        *,
        actual_minutes: int | None = None,
        productivity_score: float | None = None,
        notes: str | None = None,
    ) -> StudySession:

        session = self._require_session(
            session_id
        )

        with self._lock:

            if session.status not in {
                "running",
                "paused",
                "planned",
            }:

                raise RuntimeError(
                    "This session cannot be finished."
                )

            if actual_minutes is None:

                if session.started_at:

                    started = datetime.fromisoformat(
                        session.started_at
                    )

                    elapsed = (
                        self._now()
                        - started
                    )

                    actual_minutes = max(
                        0,
                        int(
                            elapsed.total_seconds()
                            / 60
                        ),
                    )

                else:

                    actual_minutes = (
                        session.planned_minutes
                    )

            if actual_minutes < 0:
                raise ValueError(
                    "actual_minutes cannot be negative."
                )

            if productivity_score is not None:

                if not (
                    0
                    <= productivity_score
                    <= self.MAX_PRODUCTIVITY_SCORE
                ):

                    raise ValueError(
                        "productivity_score must be between "
                        "0 and 10."
                    )

            session.actual_minutes = (
                actual_minutes
            )

            session.productivity_score = (
                productivity_score
            )

            if notes is not None:
                session.notes = notes.strip()

            session.status = "completed"

            session.ended_at = (
                self._now().isoformat()
            )

            if (
                self._active_session_id
                == session.id
            ):

                self._active_session_id = None

            self._persist_session(
                session
            )

        self._record_topic_time(
            subject=session.subject,
            topic=session.topic,
            minutes=session.actual_minutes,
        )

        self._emit(
            "education.session.completed",
            session.to_dict(),
        )

        return session

    def cancel_session(
        self,
        session_id: str,
    ) -> StudySession:

        session = self._require_session(
            session_id
        )

        with self._lock:

            if session.status == "completed":
                raise RuntimeError(
                    "Completed sessions cannot be cancelled."
                )

            session.status = "cancelled"

            if (
                self._active_session_id
                == session.id
            ):

                self._active_session_id = None

            self._persist_session(
                session
            )

        self._emit(
            "education.session.cancelled",
            session.to_dict(),
        )

        return session

    def get_active_session(
        self,
    ) -> StudySession | None:

        if (
            self._active_session_id
            is None
        ):
            return None

        return self.get_session(
            self._active_session_id
        )

    # ========================================================
    # GOALS
    # ========================================================

    def create_goal(
        self,
        *,
        title: str,
        target_value: float,
        subject: str | None = None,
        unit: str = "minutes",
        deadline: date | datetime | str | None = None,
    ) -> StudyGoal:

        title = self._clean_text(
            title,
            "goal title",
        )

        if target_value <= 0:
            raise ValueError(
                "target_value must be greater than zero."
            )

        deadline_string = (
            self._serialize_date(
                deadline
            )
            if deadline is not None
            else None
        )

        goal = StudyGoal(
            id=self._new_id("goal"),
            title=title,
            subject=(
                subject.strip()
                if subject
                else None
            ),
            target_value=target_value,
            unit=unit.strip(),
            deadline=deadline_string,
        )

        with self._lock:

            self._goals[
                goal.id
            ] = goal

            self._persist_goal(
                goal
            )

        self._emit(
            "education.goal.created",
            goal.to_dict(),
        )

        return goal

    def update_goal(
        self,
        goal_id: str,
        value: float,
    ) -> StudyGoal:

        if value < 0:
            raise ValueError(
                "Goal value cannot be negative."
            )

        with self._lock:

            goal = self._require_goal(
                goal_id
            )

            goal.update(value)

            self._persist_goal(
                goal
            )

        self._emit(
            "education.goal.updated",
            goal.to_dict(),
        )

        if goal.completed:

            self._emit(
                "education.goal.completed",
                goal.to_dict(),
            )

        return goal

    def get_goals(
        self,
        *,
        subject: str | None = None,
        completed: bool | None = None,
    ) -> list[StudyGoal]:

        with self._lock:

            goals = list(
                self._goals.values()
            )

        if subject is not None:

            normalized = subject.casefold()

            goals = [
                goal
                for goal in goals
                if (
                    goal.subject
                    and goal.subject.casefold()
                    == normalized
                )
            ]

        if completed is not None:

            goals = [
                goal
                for goal in goals
                if goal.completed
                == completed
            ]

        return goals

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
    ) -> StudyStatistics:

        with self._lock:

            sessions = list(
                self._sessions.values()
            )

            topics = list(
                self._topics.values()
            )

        total_sessions = len(
            sessions
        )

        completed_sessions = sum(
            session.status
            == "completed"
            for session in sessions
        )

        cancelled_sessions = sum(
            session.status
            == "cancelled"
            for session in sessions
        )

        total_planned = sum(
            session.planned_minutes
            for session in sessions
        )

        total_studied = sum(
            session.actual_minutes
            for session in sessions
        )

        completed_topics = sum(
            topic.completed
            for topic in topics
        )

        productivity_values = [
            session.productivity_score
            for session in sessions
            if session.productivity_score
            is not None
        ]

        average_productivity = (
            sum(productivity_values)
            / len(productivity_values)
            if productivity_values
            else 0.0
        )

        average_session = (
            total_studied
            / completed_sessions
            if completed_sessions
            else 0.0
        )

        return StudyStatistics(
            total_sessions=total_sessions,
            completed_sessions=completed_sessions,
            cancelled_sessions=cancelled_sessions,
            total_planned_minutes=total_planned,
            total_studied_minutes=total_studied,
            topics_completed=completed_topics,
            topics_total=len(topics),
            average_session_minutes=round(
                average_session,
                2,
            ),
            average_productivity=round(
                average_productivity,
                2,
            ),
        )

    def subject_statistics(
        self,
        subject: str,
    ) -> dict[str, Any]:

        normalized = self._clean_text(
            subject,
            "subject",
        ).casefold()

        sessions = [
            session
            for session in self._sessions.values()
            if session.subject.casefold()
            == normalized
        ]

        topics = [
            topic
            for topic in self._topics.values()
            if topic.subject.casefold()
            == normalized
        ]

        planned = sum(
            session.planned_minutes
            for session in sessions
        )

        studied = sum(
            session.actual_minutes
            for session in sessions
        )

        return {
            "subject": subject,
            "sessions": len(sessions),
            "completed_sessions": sum(
                session.status
                == "completed"
                for session in sessions
            ),
            "planned_minutes": planned,
            "studied_minutes": studied,
            "topics": len(topics),
            "completed_topics": sum(
                topic.completed
                for topic in topics
            ),
            "average_productivity": self._average(
                [
                    session.productivity_score
                    for session in sessions
                    if session.productivity_score
                    is not None
                ]
            ),
        }

    # ========================================================
    # DAILY SUMMARY
    # ========================================================

    def daily_summary(
        self,
        target_date: date | None = None,
    ) -> dict[str, Any]:

        target = (
            target_date
            if target_date is not None
            else self._now().date()
        )

        sessions = []

        for session in self._sessions.values():

            timestamp = (
                session.ended_at
                or session.started_at
            )

            if not timestamp:
                continue

            try:

                session_date = (
                    datetime.fromisoformat(
                        timestamp
                    ).date()
                )

            except ValueError:
                continue

            if session_date == target:
                sessions.append(
                    session
                )

        studied_minutes = sum(
            session.actual_minutes
            for session in sessions
        )

        completed = sum(
            session.status
            == "completed"
            for session in sessions
        )

        return {
            "date": target.isoformat(),
            "sessions": len(sessions),
            "completed_sessions": completed,
            "studied_minutes": studied_minutes,
            "subjects": sorted(
                {
                    session.subject
                    for session in sessions
                }
            ),
            "sessions_detail": [
                session.to_dict()
                for session in sessions
            ],
        }

    # ========================================================
    # SEARCH
    # ========================================================

    def search_topics(
        self,
        query: str,
    ) -> list[StudyTopic]:

        query = self._clean_text(
            query,
            "search query",
        ).casefold()

        return [
            topic
            for topic in self._topics.values()
            if (
                query in topic.name.casefold()
                or query
                in topic.subject.casefold()
                or query
                in topic.notes.casefold()
            )
        ]

    def search_sessions(
        self,
        query: str,
    ) -> list[StudySession]:

        query = self._clean_text(
            query,
            "search query",
        ).casefold()

        return [
            session
            for session in self._sessions.values()
            if (
                query
                in session.subject.casefold()
                or query
                in session.topic.casefold()
                or query
                in session.notes.casefold()
            )
        ]

    # ========================================================
    # DATA EXPORT
    # ========================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "topics": [
                    topic.to_dict()
                    for topic
                    in self._topics.values()
                ],
                "sessions": [
                    session.to_dict()
                    for session
                    in self._sessions.values()
                ],
                "goals": [
                    goal.to_dict()
                    for goal
                    in self._goals.values()
                ],
                "active_session_id": (
                    self._active_session_id
                ),
            }

    def import_state(
        self,
        data: dict[str, Any],
        *,
        replace: bool = False,
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Study state must be a dictionary."
            )

        with self._lock:

            if replace:
                self._topics.clear()
                self._sessions.clear()
                self._goals.clear()
                self._active_session_id = None

            for raw_topic in data.get(
                "topics",
                [],
            ):

                topic = StudyTopic(
                    id=raw_topic["id"],
                    subject=raw_topic["subject"],
                    name=raw_topic["name"],
                    target_minutes=raw_topic.get(
                        "target_minutes",
                        60,
                    ),
                    completed_minutes=raw_topic.get(
                        "completed_minutes",
                        0,
                    ),
                    completed=raw_topic.get(
                        "completed",
                        False,
                    ),
                    priority=raw_topic.get(
                        "priority",
                        3,
                    ),
                    notes=raw_topic.get(
                        "notes",
                        "",
                    ),
                    created_at=raw_topic.get(
                        "created_at",
                        self._now().isoformat(),
                    ),
                    completed_at=raw_topic.get(
                        "completed_at"
                    ),
                )

                self._topics[
                    topic.id
                ] = topic

            for raw_session in data.get(
                "sessions",
                [],
            ):

                session = StudySession(
                    id=raw_session["id"],
                    subject=raw_session["subject"],
                    topic=raw_session["topic"],
                    planned_minutes=raw_session[
                        "planned_minutes"
                    ],
                    actual_minutes=raw_session.get(
                        "actual_minutes",
                        0,
                    ),
                    started_at=raw_session.get(
                        "started_at"
                    ),
                    ended_at=raw_session.get(
                        "ended_at"
                    ),
                    status=raw_session.get(
                        "status",
                        "planned",
                    ),
                    notes=raw_session.get(
                        "notes",
                        "",
                    ),
                    productivity_score=raw_session.get(
                        "productivity_score"
                    ),
                )

                self._sessions[
                    session.id
                ] = session

            for raw_goal in data.get(
                "goals",
                [],
            ):

                goal = StudyGoal(
                    id=raw_goal["id"],
                    title=raw_goal["title"],
                    subject=raw_goal.get(
                        "subject"
                    ),
                    target_value=raw_goal[
                        "target_value"
                    ],
                    current_value=raw_goal.get(
                        "current_value",
                        0.0,
                    ),
                    unit=raw_goal.get(
                        "unit",
                        "minutes",
                    ),
                    deadline=raw_goal.get(
                        "deadline"
                    ),
                    completed=raw_goal.get(
                        "completed",
                        False,
                    ),
                    created_at=raw_goal.get(
                        "created_at",
                        self._now().isoformat(),
                    ),
                )

                self._goals[
                    goal.id
                ] = goal

            active_id = data.get(
                "active_session_id"
            )

            if (
                active_id
                in self._sessions
            ):

                self._active_session_id = (
                    active_id
                )

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    def _require_topic(
        self,
        topic_id: str,
    ) -> StudyTopic:

        topic = self.get_topic(
            topic_id
        )

        if topic is None:
            raise KeyError(
                f"Study topic not found: {topic_id}"
            )

        return topic

    def _require_session(
        self,
        session_id: str,
    ) -> StudySession:

        session = self.get_session(
            session_id
        )

        if session is None:
            raise KeyError(
                f"Study session not found: {session_id}"
            )

        return session

    def _require_goal(
        self,
        goal_id: str,
    ) -> StudyGoal:

        goal = self._goals.get(
            goal_id
        )

        if goal is None:
            raise KeyError(
                f"Study goal not found: {goal_id}"
            )

        return goal

    def _record_topic_time(
        self,
        *,
        subject: str,
        topic: str,
        minutes: int,
    ) -> None:

        if minutes <= 0:
            return

        normalized_subject = (
            subject.casefold()
        )

        normalized_topic = (
            topic.casefold()
        )

        matching_topics = [
            item
            for item in self._topics.values()
            if (
                item.subject.casefold()
                == normalized_subject
                and item.name.casefold()
                == normalized_topic
            )
        ]

        if matching_topics:

            self.add_topic_study_time(
                matching_topics[0].id,
                minutes,
            )

    def _persist_topic(
        self,
        topic: StudyTopic,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_study_topic",
            None,
        )

        if callable(method):

            try:
                method(
                    topic.to_dict()
                )
            except Exception:
                logger.exception(
                    "Unable to persist study topic."
                )

    def _persist_session(
        self,
        session: StudySession,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_study_session",
            None,
        )

        if callable(method):

            try:
                method(
                    session.to_dict()
                )
            except Exception:
                logger.exception(
                    "Unable to persist study session."
                )

    def _persist_goal(
        self,
        goal: StudyGoal,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_study_goal",
            None,
        )

        if callable(method):

            try:
                method(
                    goal.to_dict()
                )
            except Exception:
                logger.exception(
                    "Unable to persist study goal."
                )

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
                "Unable to publish education event: %s",
                event_name,
            )

    def _now(self) -> datetime:
        return self._clock()

    @staticmethod
    def _new_id(
        prefix: str,
    ) -> str:

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

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

        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return cleaned

    @classmethod
    def _validate_status(
        cls,
        status: str,
    ) -> str:

        status = status.strip().lower()

        if status not in cls.VALID_SESSION_STATUSES:
            raise ValueError(
                f"Invalid session status: {status}"
            )

        return status

    @staticmethod
    def _serialize_date(
        value: date | datetime | str,
    ) -> str:

        if isinstance(
            value,
            datetime,
        ):

            return value.isoformat()

        if isinstance(
            value,
            date,
        ):

            return value.isoformat()

        if isinstance(
            value,
            str,
        ):

            cleaned = value.strip()

            if not cleaned:
                raise ValueError(
                    "Deadline cannot be empty."
                )

            return cleaned

        raise TypeError(
            "Deadline must be a date, datetime, or string."
        )

    @staticmethod
    def _average(
        values: Iterable[
            float | int
        ],
    ) -> float:

        values = list(values)

        if not values:
            return 0.0

        return round(
            sum(values)
            / len(values),
            2,
        )


__all__ = [
    "StudyManager",
    "StudyTopic",
    "StudySession",
    "StudyGoal",
    "StudyStatistics",
]


