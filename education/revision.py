"""
RENIX Education — Revision Manager

Handles:
- Revision topics
- Revision sessions
- Spaced-repetition scheduling
- Difficulty tracking
- Confidence tracking
- Review history
- Subject/chapter/topic organization
- Revision recommendations
- Progress statistics
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

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}

VALID_STATUSES = {
    "new",
    "learning",
    "review",
    "mastered",
    "paused",
}

VALID_SESSION_RESULTS = {
    "again",
    "hard",
    "good",
    "easy",
}

DEFAULT_INTERVALS = {
    "again": 1,
    "hard": 2,
    "good": 4,
    "easy": 7,
}


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class RevisionTopic:
    """
    Represents something the student needs to revise.
    """

    id: str

    title: str
    subject: str

    chapter: str | None = None
    topic: str | None = None

    description: str = ""

    difficulty: str = "medium"
    status: str = "new"

    confidence: int = 0
    mastery: int = 0

    estimated_minutes: int = 30

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    last_reviewed_at: str | None = None
    next_review_at: str | None = None

    review_count: int = 0
    successful_reviews: int = 0

    current_streak: int = 0
    best_streak: int = 0

    interval_days: int = 0

    ease_factor: float = 2.5

    tags: list[str] = field(
        default_factory=list
    )

    notes: str = ""

    @property
    def is_due(self) -> bool:
        """
        Returns whether this topic is currently due.
        """

        if self.status in {
            "mastered",
            "paused",
        }:
            return False

        if not self.next_review_at:
            return True

        try:
            next_review = datetime.fromisoformat(
                self.next_review_at
            )

            return datetime.now() >= next_review

        except ValueError:
            return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RevisionSession:
    """
    Represents one completed revision session.
    """

    id: str

    topic_id: str

    started_at: str
    completed_at: str

    duration_minutes: int

    result: str

    confidence_before: int
    confidence_after: int

    mastery_before: int
    mastery_after: int

    notes: str = ""

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RevisionStatistics:
    """
    Revision statistics.
    """

    total_topics: int = 0

    new_topics: int = 0
    learning_topics: int = 0
    review_topics: int = 0
    mastered_topics: int = 0
    paused_topics: int = 0

    due_topics: int = 0

    total_reviews: int = 0
    successful_reviews: int = 0

    total_minutes: int = 0

    average_confidence: float = 0.0
    average_mastery: float = 0.0

    success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# REVISION MANAGER
# ============================================================


class RevisionManager:
    """
    Main RENIX revision engine.

    Example:

        manager = RevisionManager()

        topic = manager.add_topic(
            title="Quadratic Equations",
            subject="Mathematics",
            chapter="Algebra",
            difficulty="hard",
        )

        manager.start_review(topic.id)

        manager.finish_review(
            topic.id,
            result="good",
            confidence=80,
            mastery=75,
            duration_minutes=40,
        )
    """

    def __init__(
        self,
        *,
        storage: Any | None = None,
        event_bus: Any | None = None,
        notification_manager: Any | None = None,
        clock: Callable[[], datetime] | None = None,
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

        self._topics: dict[
            str,
            RevisionTopic,
        ] = {}

        self._sessions: dict[
            str,
            RevisionSession,
        ] = {}

        self._active_sessions: dict[
            str,
            datetime,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "RENIX RevisionManager initialized."
        )

    # ========================================================
    # TOPIC CREATION
    # ========================================================

    def add_topic(
        self,
        *,
        title: str,
        subject: str,
        chapter: str | None = None,
        topic: str | None = None,
        description: str = "",
        difficulty: str = "medium",
        estimated_minutes: int = 30,
        tags: Iterable[str] | None = None,
        notes: str = "",
        confidence: int = 0,
        mastery: int = 0,
    ) -> RevisionTopic:

        title = self._required_text(
            title,
            "title",
        )

        subject = self._required_text(
            subject,
            "subject",
        )

        difficulty = difficulty.lower().strip()

        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty: {difficulty}"
            )

        if (
            not isinstance(
                estimated_minutes,
                int,
            )
            or estimated_minutes <= 0
        ):
            raise ValueError(
                "estimated_minutes must be greater than zero."
            )

        confidence = self._validate_percentage(
            confidence,
            "confidence",
        )

        mastery = self._validate_percentage(
            mastery,
            "mastery",
        )

        status = self._status_from_mastery(
            mastery
        )

        topic_obj = RevisionTopic(
            id=self._new_id(),
            title=title,
            subject=subject,
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
            description=description.strip(),
            difficulty=difficulty,
            status=status,
            confidence=confidence,
            mastery=mastery,
            estimated_minutes=estimated_minutes,
            tags=self._normalize_list(tags),
            notes=notes.strip(),
        )

        with self._lock:

            self._topics[
                topic_obj.id
            ] = topic_obj

            self._persist_topic(
                topic_obj
            )

        self._emit(
            "education.revision.topic_created",
            topic_obj.to_dict(),
        )

        return topic_obj

    # ========================================================
    # TOPIC READ
    # ========================================================

    def get_topic(
        self,
        topic_id: str,
    ) -> RevisionTopic | None:

        with self._lock:
            return self._topics.get(
                topic_id
            )

    def get_all_topics(
        self,
    ) -> list[RevisionTopic]:

        with self._lock:
            return list(
                self._topics.values()
            )

    # ========================================================
    # FILTERS
    # ========================================================

    def get_due_topics(
        self,
    ) -> list[RevisionTopic]:

        with self._lock:

            topics = [
                topic
                for topic
                in self._topics.values()
                if topic.is_due
            ]

        return self._sort_for_revision(
            topics
        )

    def get_new_topics(
        self,
    ) -> list[RevisionTopic]:

        return self._get_by_status(
            "new"
        )

    def get_learning_topics(
        self,
    ) -> list[RevisionTopic]:

        return self._get_by_status(
            "learning"
        )

    def get_mastered_topics(
        self,
    ) -> list[RevisionTopic]:

        return self._get_by_status(
            "mastered"
        )

    def get_paused_topics(
        self,
    ) -> list[RevisionTopic]:

        return self._get_by_status(
            "paused"
        )

    def get_by_subject(
        self,
        subject: str,
    ) -> list[RevisionTopic]:

        subject = self._required_text(
            subject,
            "subject",
        ).casefold()

        with self._lock:

            topics = [
                item
                for item
                in self._topics.values()
                if item.subject.casefold()
                == subject
            ]

        return self._sort_for_revision(
            topics
        )

    def get_by_chapter(
        self,
        chapter: str,
    ) -> list[RevisionTopic]:

        chapter = self._required_text(
            chapter,
            "chapter",
        ).casefold()

        with self._lock:

            topics = [
                item
                for item
                in self._topics.values()
                if item.chapter
                and item.chapter.casefold()
                == chapter
            ]

        return self._sort_for_revision(
            topics
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
    ) -> list[RevisionTopic]:

        query = self._required_text(
            query,
            "query",
        ).casefold()

        with self._lock:

            topics = list(
                self._topics.values()
            )

        results = []

        for topic in topics:

            searchable = " ".join(
                [
                    topic.title,
                    topic.subject,
                    topic.chapter or "",
                    topic.topic or "",
                    topic.description,
                    topic.difficulty,
                    topic.status,
                    topic.notes,
                    " ".join(topic.tags),
                ]
            ).casefold()

            if query in searchable:
                results.append(topic)

        return self._sort_for_revision(
            results
        )

    # ========================================================
    # REVISION SESSION
    # ========================================================

    def start_review(
        self,
        topic_id: str,
    ) -> datetime:

        with self._lock:

            topic = self._require_topic(
                topic_id
            )

            if topic.status == "paused":

                raise RuntimeError(
                    "Paused topics cannot be reviewed."
                )

            start_time = self._now()

            self._active_sessions[
                topic_id
            ] = start_time

            if topic.status == "new":

                topic.status = "learning"

                topic.updated_at = (
                    start_time.isoformat()
                )

                self._persist_topic(
                    topic
                )

        self._emit(
            "education.revision.started",
            {
                "topic_id": topic_id,
                "started_at": start_time.isoformat(),
            },
        )

        return start_time

    def finish_review(
        self,
        topic_id: str,
        *,
        result: str,
        confidence: int,
        mastery: int | None = None,
        duration_minutes: int | None = None,
        notes: str = "",
    ) -> RevisionSession:

        result = result.lower().strip()

        if result not in VALID_SESSION_RESULTS:

            raise ValueError(
                f"Invalid review result: {result}"
            )

        confidence = self._validate_percentage(
            confidence,
            "confidence",
        )

        if mastery is None:
            mastery = confidence

        mastery = self._validate_percentage(
            mastery,
            "mastery",
        )

        with self._lock:

            topic = self._require_topic(
                topic_id
            )

            started_at = (
                self._active_sessions.pop(
                    topic_id,
                    None,
                )
            )

            if started_at is None:
                started_at = self._now()

            completed_at = self._now()

            if duration_minutes is None:

                duration = (
                    completed_at
                    - started_at
                )

                duration_minutes = max(
                    1,
                    int(
                        duration.total_seconds()
                        / 60
                    ),
                )

            if (
                not isinstance(
                    duration_minutes,
                    int,
                )
                or duration_minutes <= 0
            ):

                raise ValueError(
                    "duration_minutes must be "
                    "greater than zero."
                )

            old_confidence = (
                topic.confidence
            )

            old_mastery = (
                topic.mastery
            )

            self._update_spaced_repetition(
                topic,
                result,
                confidence,
                mastery,
            )

            session = RevisionSession(
                id=self._new_session_id(),
                topic_id=topic_id,
                started_at=started_at.isoformat(),
                completed_at=completed_at.isoformat(),
                duration_minutes=duration_minutes,
                result=result,
                confidence_before=old_confidence,
                confidence_after=confidence,
                mastery_before=old_mastery,
                mastery_after=mastery,
                notes=notes.strip(),
            )

            self._sessions[
                session.id
            ] = session

            topic.last_reviewed_at = (
                completed_at.isoformat()
            )

            topic.review_count += 1

            if result in {
                "good",
                "easy",
            }:

                topic.successful_reviews += 1

            topic.updated_at = (
                completed_at.isoformat()
            )

            self._persist_topic(
                topic
            )

            self._persist_session(
                session
            )

        self._schedule_next_review(
            topic
        )

        self._emit(
            "education.revision.completed",
            session.to_dict(),
        )

        return session

    # ========================================================
    # SPACED REPETITION
    # ========================================================

    def _update_spaced_repetition(
        self,
        topic: RevisionTopic,
        result: str,
        confidence: int,
        mastery: int,
    ) -> None:

        if result == "again":

            topic.interval_days = 1

            topic.ease_factor = max(
                1.3,
                topic.ease_factor - 0.20,
            )

            topic.current_streak = 0

        elif result == "hard":

            topic.interval_days = max(
                1,
                int(
                    max(
                        1,
                        topic.interval_days,
                    )
                    * 1.2
                ),
            )

            topic.ease_factor = max(
                1.3,
                topic.ease_factor - 0.15,
            )

            topic.current_streak += 1

        elif result == "good":

            if topic.interval_days <= 0:

                topic.interval_days = 4

            else:

                topic.interval_days = max(
                    2,
                    int(
                        topic.interval_days
                        * topic.ease_factor
                    ),
                )

            topic.current_streak += 1

        elif result == "easy":

            if topic.interval_days <= 0:

                topic.interval_days = 7

            else:

                topic.interval_days = max(
                    4,
                    int(
                        topic.interval_days
                        * topic.ease_factor
                        * 1.3
                    ),
                )

            topic.ease_factor += 0.10

            topic.current_streak += 1

        topic.best_streak = max(
            topic.best_streak,
            topic.current_streak,
        )

        topic.confidence = confidence

        topic.mastery = mastery

        topic.status = (
            self._status_from_mastery(
                mastery
            )
        )

        if mastery >= 90 and result == "easy":

            topic.status = "mastered"

    def _schedule_next_review(
        self,
        topic: RevisionTopic,
    ) -> None:

        if topic.status == "mastered":

            topic.next_review_at = (
                self._future_timestamp(
                    max(
                        topic.interval_days,
                        14,
                    )
                )
            )

        elif topic.status == "paused":

            topic.next_review_at = None

        else:

            topic.next_review_at = (
                self._future_timestamp(
                    max(
                        topic.interval_days,
                        1,
                    )
                )
            )

        self._persist_topic(
            topic
        )

        self._schedule_notification(
            topic
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def recommend(
        self,
        *,
        limit: int = 5,
        subject: str | None = None,
    ) -> list[RevisionTopic]:

        if limit <= 0:
            return []

        with self._lock:

            topics = list(
                self._topics.values()
            )

        if subject:

            subject = subject.casefold()

            topics = [
                item
                for item in topics
                if item.subject.casefold()
                == subject
            ]

        topics = [
            item
            for item in topics
            if item.status != "paused"
        ]

        now = self._now()

        def score(
            topic: RevisionTopic,
        ) -> tuple:

            due_score = 0

            if not topic.next_review_at:

                due_score = 100

            else:

                try:

                    next_review = datetime.fromisoformat(
                        topic.next_review_at
                    )

                    if next_review <= now:

                        days_late = (
                            now - next_review
                        ).days

                        due_score = (
                            100
                            + min(
                                days_late,
                                30,
                            )
                        )

                except ValueError:

                    due_score = 100

            difficulty_score = {
                "hard": 3,
                "medium": 2,
                "easy": 1,
            }.get(
                topic.difficulty,
                1,
            )

            weakness_score = (
                100
                - topic.mastery
            )

            return (
                -due_score,
                -weakness_score,
                -difficulty_score,
                topic.title.casefold(),
            )

        topics.sort(
            key=score
        )

        return topics[:limit]

    # ========================================================
    # PAUSE / RESUME
    # ========================================================

    def pause_topic(
        self,
        topic_id: str,
    ) -> RevisionTopic:

        with self._lock:

            topic = self._require_topic(
                topic_id
            )

            topic.status = "paused"

            topic.next_review_at = None

            topic.updated_at = (
                self._now().isoformat()
            )

            self._persist_topic(
                topic
            )

        self._emit(
            "education.revision.paused",
            topic.to_dict(),
        )

        return topic

    def resume_topic(
        self,
        topic_id: str,
    ) -> RevisionTopic:

        with self._lock:

            topic = self._require_topic(
                topic_id
            )

            topic.status = (
                self._status_from_mastery(
                    topic.mastery
                )
            )

            topic.next_review_at = (
                self._now().isoformat()
            )

            topic.updated_at = (
                self._now().isoformat()
            )

            self._persist_topic(
                topic
            )

        self._emit(
            "education.revision.resumed",
            topic.to_dict(),
        )

        return topic

    # ========================================================
    # RESET
    # ========================================================

    def reset_topic(
        self,
        topic_id: str,
    ) -> RevisionTopic:

        with self._lock:

            topic = self._require_topic(
                topic_id
            )

            topic.status = "new"

            topic.confidence = 0

            topic.mastery = 0

            topic.review_count = 0

            topic.successful_reviews = 0

            topic.current_streak = 0

            topic.interval_days = 0

            topic.ease_factor = 2.5

            topic.last_reviewed_at = None

            topic.next_review_at = None

            topic.updated_at = (
                self._now().isoformat()
            )

            self._persist_topic(
                topic
            )

        self._emit(
            "education.revision.reset",
            topic.to_dict(),
        )

        return topic

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
    ) -> RevisionStatistics:

        with self._lock:

            topics = list(
                self._topics.values()
            )

            sessions = list(
                self._sessions.values()
            )

        stats = RevisionStatistics()

        stats.total_topics = len(
            topics
        )

        for topic in topics:

            if topic.status == "new":
                stats.new_topics += 1

            elif topic.status == "learning":
                stats.learning_topics += 1

            elif topic.status == "review":
                stats.review_topics += 1

            elif topic.status == "mastered":
                stats.mastered_topics += 1

            elif topic.status == "paused":
                stats.paused_topics += 1

            if topic.is_due:
                stats.due_topics += 1

            stats.total_minutes += (
                topic.review_count
                * topic.estimated_minutes
            )

        stats.total_reviews = len(
            sessions
        )

        stats.successful_reviews = len(
            [
                session
                for session in sessions
                if session.result
                in {
                    "good",
                    "easy",
                }
            ]
        )

        if stats.total_reviews:

            stats.success_rate = (
                stats.successful_reviews
                / stats.total_reviews
            ) * 100

        if topics:

            stats.average_confidence = (
                sum(
                    item.confidence
                    for item in topics
                )
                / len(topics)
            )

            stats.average_mastery = (
                sum(
                    item.mastery
                    for item in topics
                )
                / len(topics)
            )

        return stats

    # ========================================================
    # SESSION HISTORY
    # ========================================================

    def get_sessions(
        self,
        topic_id: str | None = None,
    ) -> list[RevisionSession]:

        with self._lock:

            sessions = list(
                self._sessions.values()
            )

        if topic_id is not None:

            sessions = [
                session
                for session in sessions
                if session.topic_id
                == topic_id
            ]

        return sorted(
            sessions,
            key=lambda session: (
                session.completed_at
            ),
            reverse=True,
        )

    # ========================================================
    # DELETE
    # ========================================================

    def delete_topic(
        self,
        topic_id: str,
    ) -> bool:

        with self._lock:

            topic = self._topics.pop(
                topic_id,
                None,
            )

            if topic is None:
                return False

            self._active_sessions.pop(
                topic_id,
                None,
            )

            session_ids = [
                session_id
                for session_id, session
                in self._sessions.items()
                if session.topic_id
                == topic_id
            ]

            for session_id in session_ids:

                self._sessions.pop(
                    session_id,
                    None,
                )

        self._delete_persisted_topic(
            topic_id
        )

        self._emit(
            "education.revision.topic_deleted",
            {
                "topic_id": topic_id,
            },
        )

        return True

    # ========================================================
    # IMPORT / EXPORT
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
                "Revision state must be a dictionary."
            )

        raw_topics = data.get(
            "topics",
            [],
        )

        raw_sessions = data.get(
            "sessions",
            [],
        )

        if replace:

            with self._lock:

                self._topics.clear()

                self._sessions.clear()

        with self._lock:

            for raw in raw_topics:

                topic = RevisionTopic(
                    **raw
                )

                self._topics[
                    topic.id
                ] = topic

            for raw in raw_sessions:

                session = RevisionSession(
                    **raw
                )

                self._sessions[
                    session.id
                ] = session

    # ========================================================
    # STORAGE
    # ========================================================

    def _persist_topic(
        self,
        topic: RevisionTopic,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_revision_topic",
            None,
        )

        if callable(method):

            try:

                method(
                    topic.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to save revision topic."
                )

    def _persist_session(
        self,
        session: RevisionSession,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_revision_session",
            None,
        )

        if callable(method):

            try:

                method(
                    session.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to save revision session."
                )

    def _delete_persisted_topic(
        self,
        topic_id: str,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "delete_revision_topic",
            None,
        )

        if callable(method):

            try:

                method(
                    topic_id
                )

            except Exception:

                logger.exception(
                    "Failed to delete revision topic."
                )

    # ========================================================
    # NOTIFICATIONS
    # ========================================================

    def _schedule_notification(
        self,
        topic: RevisionTopic,
    ) -> None:

        if (
            self.notification_manager
            is None
        ):
            return

        if not topic.next_review_at:
            return

        try:

            schedule = getattr(
                self.notification_manager,
                "schedule",
                None,
            )

            if not callable(schedule):
                return

            scheduled_at = datetime.fromisoformat(
                topic.next_review_at
            )

            schedule(
                notification_id=(
                    f"revision_{topic.id}"
                ),
                title=(
                    f"Revision: {topic.title}"
                ),
                message=(
                    f"It's time to revise "
                    f"{topic.subject}."
                ),
                scheduled_at=scheduled_at,
                metadata={
                    "topic_id": topic.id,
                    "type": "revision",
                },
            )

        except Exception:

            logger.exception(
                "Failed to schedule revision notification."
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
                "Failed to emit event: %s",
                event_name,
            )

    # ========================================================
    # HELPERS
    # ========================================================

    def _get_by_status(
        self,
        status: str,
    ) -> list[RevisionTopic]:

        with self._lock:

            topics = [
                topic
                for topic
                in self._topics.values()
                if topic.status == status
            ]

        return self._sort_for_revision(
            topics
        )

    @staticmethod
    def _sort_for_revision(
        topics: list[RevisionTopic],
    ) -> list[RevisionTopic]:

        def sort_key(
            topic: RevisionTopic,
        ) -> tuple:

            next_review = (
                topic.next_review_at
                or ""
            )

            difficulty = {
                "hard": 0,
                "medium": 1,
                "easy": 2,
            }.get(
                topic.difficulty,
                1,
            )

            return (
                0 if topic.is_due else 1,
                next_review,
                difficulty,
                topic.mastery,
                topic.title.casefold(),
            )

        return sorted(
            topics,
            key=sort_key,
        )

    def _require_topic(
        self,
        topic_id: str,
    ) -> RevisionTopic:

        topic = self._topics.get(
            topic_id
        )

        if topic is None:

            raise KeyError(
                f"Revision topic not found: "
                f"{topic_id}"
            )

        return topic

    def _now(self) -> datetime:
        return self._clock()

    def _future_timestamp(
        self,
        days: int,
    ) -> str:

        return (
            self._now()
            + timedelta(days=days)
        ).isoformat()

    @staticmethod
    def _new_id() -> str:

        return (
            "rev_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _new_session_id() -> str:

        return (
            "revs_"
            + uuid.uuid4().hex
        )

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
    def _validate_percentage(
        value: int,
        field_name: str,
    ) -> int:

        if not isinstance(
            value,
            int,
        ):

            raise TypeError(
                f"{field_name} must be an integer."
            )

        if not 0 <= value <= 100:

            raise ValueError(
                f"{field_name} must be between 0 and 100."
            )

        return value

    @staticmethod
    def _status_from_mastery(
        mastery: int,
    ) -> str:

        if mastery >= 90:
            return "mastered"

        if mastery >= 40:
            return "review"

        if mastery > 0:
            return "learning"

        return "new"

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
                result.append(value)

        return result

    def __len__(self) -> int:

        with self._lock:
            return len(
                self._topics
            )


__all__ = [
    "RevisionTopic",
    "RevisionSession",
    "RevisionStatistics",
    "RevisionManager",
    "VALID_DIFFICULTIES",
    "VALID_STATUSES",
    "VALID_SESSION_RESULTS",
]


