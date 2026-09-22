"""
RENIX Education — Progress Tracker

Tracks a student's academic progress across:
- Subjects
- Chapters/topics
- Homework
- Revision
- Quizzes
- Exams
- Practice sessions
- Accuracy
- Completion
- Study time
- Weak/strong areas

Designed to work independently from the UI and database layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class TopicProgress:
    subject: str
    topic: str

    questions_attempted: int = 0
    questions_correct: int = 0

    study_minutes: float = 0.0

    revision_count: int = 0

    mastery: float = 0.0

    last_studied: str | None = None

    notes: list[str] = field(default_factory=list)

    def accuracy(self) -> float:
        if self.questions_attempted <= 0:
            return 0.0

        return (
            self.questions_correct
            / self.questions_attempted
            * 100
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "topic": self.topic,
            "questions_attempted": self.questions_attempted,
            "questions_correct": self.questions_correct,
            "study_minutes": self.study_minutes,
            "revision_count": self.revision_count,
            "mastery": self.mastery,
            "accuracy": self.accuracy(),
            "last_studied": self.last_studied,
            "notes": list(self.notes),
        }


@dataclass
class SubjectProgress:
    subject: str

    total_topics: int = 0
    completed_topics: int = 0

    study_minutes: float = 0.0

    questions_attempted: int = 0
    questions_correct: int = 0

    exams_attempted: int = 0
    exams_completed: int = 0

    mastery: float = 0.0

    def accuracy(self) -> float:
        if self.questions_attempted <= 0:
            return 0.0

        return (
            self.questions_correct
            / self.questions_attempted
            * 100
        )

    def completion(self) -> float:
        if self.total_topics <= 0:
            return 0.0

        return (
            self.completed_topics
            / self.total_topics
            * 100
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "total_topics": self.total_topics,
            "completed_topics": self.completed_topics,
            "completion": self.completion(),
            "study_minutes": self.study_minutes,
            "questions_attempted": self.questions_attempted,
            "questions_correct": self.questions_correct,
            "accuracy": self.accuracy(),
            "exams_attempted": self.exams_attempted,
            "exams_completed": self.exams_completed,
            "mastery": self.mastery,
        }


@dataclass
class StudySession:
    subject: str
    topic: str
    minutes: float
    started_at: str
    ended_at: str | None = None

    questions_attempted: int = 0
    questions_correct: int = 0

    session_type: str = "study"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "topic": self.topic,
            "minutes": self.minutes,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "questions_attempted": self.questions_attempted,
            "questions_correct": self.questions_correct,
            "session_type": self.session_type,
        }


# ============================================================
# PROGRESS TRACKER
# ============================================================


class ProgressTracker:
    """
    Central academic progress tracker for RENIX.

    Example:

        tracker = ProgressTracker()

        tracker.register_topic(
            "Maths",
            "Quadratic Equations"
        )

        tracker.record_study(
            "Maths",
            "Quadratic Equations",
            45
        )

        tracker.record_question_result(
            "Maths",
            "Quadratic Equations",
            correct=True
        )

        tracker.mark_topic_complete(
            "Maths",
            "Quadratic Equations"
        )

        print(
            tracker.get_subject_progress("Maths")
        )
    """

    def __init__(
        self,
        *,
        storage: Any | None = None,
    ) -> None:

        self.storage = storage

        self.subjects: dict[
            str,
            SubjectProgress,
        ] = {}

        self.topics: dict[
            str,
            TopicProgress,
        ] = {}

        self.sessions: list[
            StudySession
        ] = []

        self.achievements: list[
            dict[str, Any]
        ] = []

    # ========================================================
    # SUBJECT MANAGEMENT
    # ========================================================

    def register_subject(
        self,
        subject: str,
    ) -> SubjectProgress:

        subject = self._clean_name(
            subject
        )

        if not subject:
            raise ValueError(
                "Subject cannot be empty."
            )

        if subject not in self.subjects:

            self.subjects[subject] = (
                SubjectProgress(
                    subject=subject
                )
            )

        self._save()

        return self.subjects[subject]

    def register_subjects(
        self,
        subjects: Iterable[str],
    ) -> None:

        for subject in subjects:
            self.register_subject(
                subject
            )

    # ========================================================
    # TOPIC MANAGEMENT
    # ========================================================

    def register_topic(
        self,
        subject: str,
        topic: str,
    ) -> TopicProgress:

        subject = self._clean_name(
            subject
        )

        topic = self._clean_name(
            topic
        )

        if not subject:
            raise ValueError(
                "Subject cannot be empty."
            )

        if not topic:
            raise ValueError(
                "Topic cannot be empty."
            )

        self.register_subject(
            subject
        )

        key = self._topic_key(
            subject,
            topic,
        )

        if key not in self.topics:

            self.topics[key] = (
                TopicProgress(
                    subject=subject,
                    topic=topic,
                )
            )

            self.subjects[
                subject
            ].total_topics += 1

        self._save()

        return self.topics[key]

    def register_topics(
        self,
        subject: str,
        topics: Iterable[str],
    ) -> None:

        for topic in topics:

            self.register_topic(
                subject,
                topic,
            )

    # ========================================================
    # STUDY TIME
    # ========================================================

    def record_study(
        self,
        subject: str,
        topic: str,
        minutes: float,
        *,
        session_type: str = "study",
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> StudySession:

        if minutes < 0:
            raise ValueError(
                "Study minutes cannot be negative."
            )

        progress = self.register_topic(
            subject,
            topic,
        )

        timestamp = (
            started_at
            or self._now()
        )

        progress.study_minutes += (
            float(minutes)
        )

        progress.last_studied = timestamp

        self.subjects[
            progress.subject
        ].study_minutes += float(
            minutes
        )

        session = StudySession(
            subject=progress.subject,
            topic=progress.topic,
            minutes=float(minutes),
            started_at=timestamp,
            ended_at=ended_at,
            session_type=session_type,
        )

        self.sessions.append(
            session
        )

        self._recalculate_topic_mastery(
            progress
        )

        self._recalculate_subject_mastery(
            progress.subject
        )

        self._check_achievements()

        self._save()

        return session

    # ========================================================
    # QUESTION RESULTS
    # ========================================================

    def record_question_result(
        self,
        subject: str,
        topic: str,
        *,
        correct: bool,
        count: int = 1,
    ) -> TopicProgress:

        if count <= 0:
            raise ValueError(
                "count must be greater than zero."
            )

        progress = self.register_topic(
            subject,
            topic,
        )

        progress.questions_attempted += (
            count
        )

        if correct:
            progress.questions_correct += (
                count
            )

        subject_progress = self.subjects[
            progress.subject
        ]

        subject_progress.questions_attempted += (
            count
        )

        if correct:
            subject_progress.questions_correct += (
                count
            )

        self._recalculate_topic_mastery(
            progress
        )

        self._recalculate_subject_mastery(
            progress.subject
        )

        self._check_achievements()

        self._save()

        return progress

    def record_batch_results(
        self,
        subject: str,
        topic: str,
        *,
        attempted: int,
        correct: int,
    ) -> TopicProgress:

        if attempted < 0:
            raise ValueError(
                "attempted cannot be negative."
            )

        if correct < 0:
            raise ValueError(
                "correct cannot be negative."
            )

        if correct > attempted:
            raise ValueError(
                "correct cannot exceed attempted."
            )

        progress = self.register_topic(
            subject,
            topic,
        )

        progress.questions_attempted += (
            attempted
        )

        progress.questions_correct += (
            correct
        )

        subject_progress = self.subjects[
            progress.subject
        ]

        subject_progress.questions_attempted += (
            attempted
        )

        subject_progress.questions_correct += (
            correct
        )

        self._recalculate_topic_mastery(
            progress
        )

        self._recalculate_subject_mastery(
            progress.subject
        )

        self._check_achievements()

        self._save()

        return progress

    # ========================================================
    # REVISION
    # ========================================================

    def record_revision(
        self,
        subject: str,
        topic: str,
    ) -> TopicProgress:

        progress = self.register_topic(
            subject,
            topic,
        )

        progress.revision_count += 1

        progress.last_studied = (
            self._now()
        )

        self._recalculate_topic_mastery(
            progress
        )

        self._recalculate_subject_mastery(
            progress.subject
        )

        self._save()

        return progress

    # ========================================================
    # TOPIC COMPLETION
    # ========================================================

    def mark_topic_complete(
        self,
        subject: str,
        topic: str,
    ) -> TopicProgress:

        progress = self.register_topic(
            subject,
            topic,
        )

        subject_progress = self.subjects[
            progress.subject
        ]

        already_complete = (
            progress.mastery >= 1.0
        )

        progress.mastery = 1.0

        if not already_complete:

            subject_progress.completed_topics = (
                min(
                    subject_progress.total_topics,
                    subject_progress.completed_topics + 1,
                )
            )

        self._recalculate_subject_mastery(
            progress.subject
        )

        self._check_achievements()

        self._save()

        return progress

    # ========================================================
    # PROGRESS QUERIES
    # ========================================================

    def get_topic_progress(
        self,
        subject: str,
        topic: str,
    ) -> dict[str, Any] | None:

        key = self._topic_key(
            subject,
            topic,
        )

        progress = self.topics.get(
            key
        )

        if progress is None:
            return None

        return progress.to_dict()

    def get_subject_progress(
        self,
        subject: str,
    ) -> dict[str, Any] | None:

        subject = self._clean_name(
            subject
        )

        progress = self.subjects.get(
            subject
        )

        if progress is None:
            return None

        return progress.to_dict()

    def get_all_subjects(
        self,
    ) -> list[dict[str, Any]]:

        return [
            progress.to_dict()
            for progress in self.subjects.values()
        ]

    def get_all_topics(
        self,
    ) -> list[dict[str, Any]]:

        return [
            progress.to_dict()
            for progress in self.topics.values()
        ]

    # ========================================================
    # WEAK / STRONG TOPICS
    # ========================================================

    def get_weak_topics(
        self,
        *,
        limit: int = 5,
        threshold: float = 60.0,
    ) -> list[dict[str, Any]]:

        topics = [
            topic
            for topic in self.topics.values()
            if topic.questions_attempted > 0
            and topic.accuracy() < threshold
        ]

        topics.sort(
            key=lambda item: item.accuracy()
        )

        return [
            topic.to_dict()
            for topic in topics[:limit]
        ]

    def get_strong_topics(
        self,
        *,
        limit: int = 5,
        threshold: float = 80.0,
    ) -> list[dict[str, Any]]:

        topics = [
            topic
            for topic in self.topics.values()
            if topic.questions_attempted > 0
            and topic.accuracy() >= threshold
        ]

        topics.sort(
            key=lambda item: item.accuracy(),
            reverse=True,
        )

        return [
            topic.to_dict()
            for topic in topics[:limit]
        ]

    # ========================================================
    # OVERALL PROGRESS
    # ========================================================

    def get_overall_progress(
        self,
    ) -> dict[str, Any]:

        total_topics = len(
            self.topics
        )

        completed_topics = sum(
            1
            for topic in self.topics.values()
            if topic.mastery >= 1.0
        )

        total_minutes = sum(
            topic.study_minutes
            for topic in self.topics.values()
        )

        attempted = sum(
            topic.questions_attempted
            for topic in self.topics.values()
        )

        correct = sum(
            topic.questions_correct
            for topic in self.topics.values()
        )

        accuracy = (
            correct / attempted * 100
            if attempted > 0
            else 0.0
        )

        completion = (
            completed_topics
            / total_topics
            * 100
            if total_topics > 0
            else 0.0
        )

        mastery = self._overall_mastery()

        return {
            "subjects": len(
                self.subjects
            ),
            "topics": total_topics,
            "completed_topics": completed_topics,
            "completion": completion,
            "study_minutes": total_minutes,
            "questions_attempted": attempted,
            "questions_correct": correct,
            "accuracy": accuracy,
            "mastery": mastery,
            "achievements": len(
                self.achievements
            ),
        }

    # ========================================================
    # DASHBOARD
    # ========================================================

    def get_dashboard(
        self,
    ) -> dict[str, Any]:

        return {
            "overall": self.get_overall_progress(),
            "subjects": self.get_all_subjects(),
            "weak_topics": self.get_weak_topics(),
            "strong_topics": self.get_strong_topics(),
            "recent_sessions": [
                session.to_dict()
                for session in self.sessions[-10:]
            ],
            "achievements": list(
                self.achievements
            ),
        }

    # ========================================================
    # STUDY STREAK
    # ========================================================

    def get_study_streak(
        self,
        *,
        today: datetime | None = None,
    ) -> int:

        if today is None:
            today = datetime.now()

        dates = set()

        for session in self.sessions:

            try:

                date = datetime.fromisoformat(
                    session.started_at
                ).date()

                dates.add(date)

            except (
                TypeError,
                ValueError,
            ):
                continue

        if not dates:
            return 0

        current_date = today.date()

        if current_date not in dates:

            current_date -= timedelta(
                days=1
            )

        streak = 0

        while current_date in dates:

            streak += 1

            current_date -= timedelta(
                days=1
            )

        return streak

    # ========================================================
    # ANALYTICS
    # ========================================================

    def get_subject_ranking(
        self,
    ) -> list[dict[str, Any]]:

        subjects = list(
            self.subjects.values()
        )

        subjects.sort(
            key=lambda item: (
                item.mastery,
                item.accuracy(),
                item.completion(),
            ),
            reverse=True,
        )

        return [
            {
                **item.to_dict(),
                "rank": index + 1,
            }
            for index, item in enumerate(
                subjects
            )
        ]

    def get_recent_activity(
        self,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        sessions = sorted(
            self.sessions,
            key=lambda session: (
                session.started_at
            ),
            reverse=True,
        )

        return [
            session.to_dict()
            for session in sessions[:limit]
        ]

    # ========================================================
    # ACHIEVEMENTS
    # ========================================================

    def get_achievements(
        self,
    ) -> list[dict[str, Any]]:

        return list(
            self.achievements
        )

    def _check_achievements(
        self,
    ) -> None:

        overall = self.get_overall_progress()

        self._unlock(
            "first_study_session",
            (
                len(self.sessions) >= 1
            ),
            "Completed your first study session.",
        )

        self._unlock(
            "ten_questions",
            (
                overall["questions_attempted"]
                >= 10
            ),
            "Attempted 10 questions.",
        )

        self._unlock(
            "hundred_questions",
            (
                overall["questions_attempted"]
                >= 100
            ),
            "Attempted 100 questions.",
        )

        self._unlock(
            "five_hours",
            (
                overall["study_minutes"]
                >= 300
            ),
            "Completed 5 hours of study.",
        )

        self._unlock(
            "ten_hours",
            (
                overall["study_minutes"]
                >= 600
            ),
            "Completed 10 hours of study.",
        )

        self._unlock(
            "perfect_accuracy",
            (
                overall["questions_attempted"] > 0
                and overall["accuracy"] >= 100
            ),
            "Achieved 100% accuracy.",
        )

        self._unlock(
            "five_topics",
            (
                overall["completed_topics"]
                >= 5
            ),
            "Completed 5 topics.",
        )

    def _unlock(
        self,
        achievement_id: str,
        condition: bool,
        description: str,
    ) -> None:

        if not condition:
            return

        existing = {
            achievement["id"]
            for achievement in self.achievements
        }

        if achievement_id in existing:
            return

        self.achievements.append(
            {
                "id": achievement_id,
                "description": description,
                "unlocked_at": self._now(),
            }
        )

    # ========================================================
    # MASTERY
    # ========================================================

    def _recalculate_topic_mastery(
        self,
        topic: TopicProgress,
    ) -> None:

        accuracy = (
            topic.accuracy() / 100
        )

        revision_factor = min(
            topic.revision_count
            / 5,
            1.0,
        )

        study_factor = min(
            topic.study_minutes
            / 120,
            1.0,
        )

        if topic.questions_attempted == 0:

            mastery = (
                revision_factor * 0.4
                + study_factor * 0.6
            )

        else:

            mastery = (
                accuracy * 0.65
                + revision_factor * 0.20
                + study_factor * 0.15
            )

        topic.mastery = max(
            0.0,
            min(
                1.0,
                mastery,
            ),
        )

    def _recalculate_subject_mastery(
        self,
        subject: str,
    ) -> None:

        subject_progress = self.subjects.get(
            subject
        )

        if subject_progress is None:
            return

        topic_list = [
            topic
            for topic in self.topics.values()
            if topic.subject == subject
        ]

        if not topic_list:

            subject_progress.mastery = 0.0
            return

        subject_progress.mastery = (
            sum(
                topic.mastery
                for topic in topic_list
            )
            / len(topic_list)
        )

        subject_progress.completed_topics = sum(
            1
            for topic in topic_list
            if topic.mastery >= 1.0
        )

    def _overall_mastery(
        self,
    ) -> float:

        if not self.topics:
            return 0.0

        return (
            sum(
                topic.mastery
                for topic in self.topics.values()
            )
            / len(self.topics)
            * 100
        )

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "subjects": {
                name: progress.to_dict()
                for name, progress
                in self.subjects.items()
            },
            "topics": {
                key: progress.to_dict()
                for key, progress
                in self.topics.items()
            },
            "sessions": [
                session.to_dict()
                for session in self.sessions
            ],
            "achievements": list(
                self.achievements
            ),
        }

    def load_dict(
        self,
        data: Mapping[str, Any],
    ) -> None:

        self.subjects.clear()
        self.topics.clear()
        self.sessions.clear()
        self.achievements.clear()

        for name, raw in (
            data.get(
                "subjects",
                {},
            ).items()
        ):

            self.subjects[name] = (
                SubjectProgress(
                    subject=raw.get(
                        "subject",
                        name,
                    ),
                    total_topics=int(
                        raw.get(
                            "total_topics",
                            0,
                        )
                    ),
                    completed_topics=int(
                        raw.get(
                            "completed_topics",
                            0,
                        )
                    ),
                    study_minutes=float(
                        raw.get(
                            "study_minutes",
                            0,
                        )
                    ),
                    questions_attempted=int(
                        raw.get(
                            "questions_attempted",
                            0,
                        )
                    ),
                    questions_correct=int(
                        raw.get(
                            "questions_correct",
                            0,
                        )
                    ),
                    exams_attempted=int(
                        raw.get(
                            "exams_attempted",
                            0,
                        )
                    ),
                    exams_completed=int(
                        raw.get(
                            "exams_completed",
                            0,
                        )
                    ),
                    mastery=float(
                        raw.get(
                            "mastery",
                            0,
                        )
                    ),
                )
            )

        for key, raw in (
            data.get(
                "topics",
                {},
            ).items()
        ):

            self.topics[key] = (
                TopicProgress(
                    subject=raw.get(
                        "subject",
                        "",
                    ),
                    topic=raw.get(
                        "topic",
                        "",
                    ),
                    questions_attempted=int(
                        raw.get(
                            "questions_attempted",
                            0,
                        )
                    ),
                    questions_correct=int(
                        raw.get(
                            "questions_correct",
                            0,
                        )
                    ),
                    study_minutes=float(
                        raw.get(
                            "study_minutes",
                            0,
                        )
                    ),
                    revision_count=int(
                        raw.get(
                            "revision_count",
                            0,
                        )
                    ),
                    mastery=float(
                        raw.get(
                            "mastery",
                            0,
                        )
                    ),
                    last_studied=raw.get(
                        "last_studied"
                    ),
                    notes=list(
                        raw.get(
                            "notes",
                            [],
                        )
                    ),
                )
            )

        for raw in data.get(
            "sessions",
            [],
        ):

            self.sessions.append(
                StudySession(
                    subject=raw.get(
                        "subject",
                        "",
                    ),
                    topic=raw.get(
                        "topic",
                        "",
                    ),
                    minutes=float(
                        raw.get(
                            "minutes",
                            0,
                        )
                    ),
                    started_at=raw.get(
                        "started_at",
                        self._now(),
                    ),
                    ended_at=raw.get(
                        "ended_at"
                    ),
                    questions_attempted=int(
                        raw.get(
                            "questions_attempted",
                            0,
                        )
                    ),
                    questions_correct=int(
                        raw.get(
                            "questions_correct",
                            0,
                        )
                    ),
                    session_type=raw.get(
                        "session_type",
                        "study",
                    ),
                )
            )

        self.achievements.extend(
            data.get(
                "achievements",
                [],
            )
        )

    # ========================================================
    # STORAGE
    # ========================================================

    def _save(self) -> None:

        if self.storage is None:
            return

        try:

            if hasattr(
                self.storage,
                "save_progress",
            ):

                self.storage.save_progress(
                    self.to_dict()
                )

            elif callable(
                self.storage
            ):

                self.storage(
                    self.to_dict()
                )

        except Exception:
            # Progress tracking should never
            # crash the education engine because
            # persistent storage failed.
            pass

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _clean_name(
        value: Any,
    ) -> str:

        return str(
            value or ""
        ).strip()

    @staticmethod
    def _topic_key(
        subject: str,
        topic: str,
    ) -> str:

        return (
            f"{subject.strip().casefold()}"
            f"::{topic.strip().casefold()}"
        )

    @staticmethod
    def _now() -> str:

        return datetime.now().isoformat(
            timespec="seconds"
        )


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "TopicProgress",
    "SubjectProgress",
    "StudySession",
    "ProgressTracker",
]


