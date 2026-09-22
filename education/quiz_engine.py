"""
RENIX Education — Quiz Engine

Provides:
- Quiz creation
- Question management
- Multiple-choice questions
- True/false questions
- Short-answer questions
- Difficulty levels
- Subject/chapter/topic organization
- Timed quizzes
- Answer evaluation
- Score calculation
- Accuracy tracking
- Attempt history
- Weak-topic detection
- Adaptive difficulty
- Quiz recommendations
- Import/export support

This module is intentionally independent of any particular UI.
RENIX can connect it to:
    - education/study_manager.py
    - education/question_generator.py
    - education/answer_checker.py
    - education/progress_tracker.py
    - core/event_bus.py
    - memory/memory_manager.py
    - database/repositories.py
"""

from __future__ import annotations

import logging
import threading
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

QUESTION_TYPES = {
    "mcq",
    "true_false",
    "short_answer",
}

DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}

QUIZ_STATUSES = {
    "draft",
    "ready",
    "active",
    "completed",
    "cancelled",
}

ATTEMPT_STATUSES = {
    "active",
    "completed",
    "expired",
    "cancelled",
}


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class QuizQuestion:
    """
    Represents one question inside a quiz.
    """

    id: str

    question: str
    question_type: str

    correct_answer: str

    options: list[str] = field(
        default_factory=list
    )

    explanation: str = ""

    subject: str = ""
    chapter: str = ""
    topic: str = ""

    difficulty: str = "medium"

    points: float = 1.0

    tags: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Quiz:
    """
    Represents a complete quiz.
    """

    id: str

    title: str

    questions: list[str] = field(
        default_factory=list
    )

    subject: str = ""
    chapter: str = ""
    topic: str = ""

    description: str = ""

    difficulty: str = "medium"

    time_limit_minutes: int | None = None

    passing_percentage: float = 40.0

    status: str = "draft"

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    tags: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuizAnswer:
    """
    Represents an answer submitted for one question.
    """

    question_id: str

    answer: str

    is_correct: bool

    points_earned: float

    answered_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    time_spent_seconds: float | None = None

    confidence: int | None = None

    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuizAttempt:
    """
    Represents one user's attempt at a quiz.
    """

    id: str

    quiz_id: str

    started_at: str

    status: str = "active"

    completed_at: str | None = None

    answers: dict[str, QuizAnswer] = field(
        default_factory=dict
    )

    total_points: float = 0.0

    earned_points: float = 0.0

    percentage: float = 0.0

    correct_answers: int = 0

    incorrect_answers: int = 0

    skipped_answers: int = 0

    duration_seconds: float = 0.0

    passed: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        data = asdict(self)

        data["answers"] = {
            key: value.to_dict()
            for key, value in self.answers.items()
        }

        return data


@dataclass
class QuizStatistics:
    """
    Aggregated quiz statistics.
    """

    total_quizzes: int = 0

    completed_attempts: int = 0

    average_score: float = 0.0

    highest_score: float = 0.0

    lowest_score: float = 0.0

    total_questions_answered: int = 0

    total_correct_answers: int = 0

    total_incorrect_answers: int = 0

    overall_accuracy: float = 0.0

    passed_attempts: int = 0

    failed_attempts: int = 0

    total_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# QUIZ ENGINE
# ============================================================


class QuizEngine:
    """
    Main RENIX quiz engine.

    Example:

        engine = QuizEngine()

        question = engine.add_question(
            question="What is 2 + 2?",
            question_type="mcq",
            correct_answer="4",
            options=["2", "3", "4", "5"],
            subject="Mathematics",
        )

        quiz = engine.create_quiz(
            title="Math Test",
            subject="Mathematics",
        )

        engine.add_question_to_quiz(
            quiz.id,
            question.id,
        )

        attempt = engine.start_quiz(
            quiz.id
        )

        engine.submit_answer(
            attempt.id,
            question.id,
            "4",
        )

        result = engine.finish_quiz(
            attempt.id
        )
    """

    def __init__(
        self,
        *,
        storage: Any | None = None,
        event_bus: Any | None = None,
        answer_checker: Any | None = None,
    ) -> None:

        self.storage = storage
        self.event_bus = event_bus
        self.answer_checker = answer_checker

        self._questions: dict[
            str,
            QuizQuestion,
        ] = {}

        self._quizzes: dict[
            str,
            Quiz,
        ] = {}

        self._attempts: dict[
            str,
            QuizAttempt,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "RENIX QuizEngine initialized."
        )

    # ========================================================
    # QUESTION MANAGEMENT
    # ========================================================

    def add_question(
        self,
        *,
        question: str,
        question_type: str,
        correct_answer: str,
        options: Iterable[str] | None = None,
        explanation: str = "",
        subject: str = "",
        chapter: str = "",
        topic: str = "",
        difficulty: str = "medium",
        points: float = 1.0,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QuizQuestion:

        question = self._required_text(
            question,
            "question",
        )

        correct_answer = self._required_text(
            correct_answer,
            "correct_answer",
        )

        question_type = (
            question_type
            .strip()
            .lower()
        )

        if question_type not in QUESTION_TYPES:

            raise ValueError(
                f"Unsupported question type: "
                f"{question_type}"
            )

        difficulty = (
            difficulty
            .strip()
            .lower()
        )

        if difficulty not in DIFFICULTIES:

            raise ValueError(
                f"Unsupported difficulty: "
                f"{difficulty}"
            )

        if points <= 0:

            raise ValueError(
                "points must be greater than zero."
            )

        normalized_options = (
            self._normalize_strings(
                options
            )
        )

        if question_type == "mcq":

            if len(normalized_options) < 2:

                raise ValueError(
                    "MCQ questions require at least "
                    "two options."
                )

            if not self._answer_in_options(
                correct_answer,
                normalized_options,
            ):

                raise ValueError(
                    "correct_answer must match "
                    "one of the MCQ options."
                )

        elif question_type == "true_false":

            normalized_options = [
                "True",
                "False",
            ]

            if correct_answer.casefold() not in {
                "true",
                "false",
            }:

                raise ValueError(
                    "True/false questions must have "
                    "True or False as the answer."
                )

            correct_answer = (
                "True"
                if correct_answer.casefold()
                == "true"
                else "False"
            )

        question_obj = QuizQuestion(
            id=self._new_question_id(),
            question=question,
            question_type=question_type,
            correct_answer=correct_answer,
            options=normalized_options,
            explanation=explanation.strip(),
            subject=subject.strip(),
            chapter=chapter.strip(),
            topic=topic.strip(),
            difficulty=difficulty,
            points=float(points),
            tags=self._normalize_strings(
                tags
            ),
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        with self._lock:

            self._questions[
                question_obj.id
            ] = question_obj

            self._persist_question(
                question_obj
            )

        self._emit(
            "education.quiz.question_created",
            question_obj.to_dict(),
        )

        return question_obj

    def get_question(
        self,
        question_id: str,
    ) -> QuizQuestion | None:

        with self._lock:

            return self._questions.get(
                question_id
            )

    def get_questions(
        self,
    ) -> list[QuizQuestion]:

        with self._lock:

            return list(
                self._questions.values()
            )

    def delete_question(
        self,
        question_id: str,
    ) -> bool:

        with self._lock:

            question = self._questions.pop(
                question_id,
                None,
            )

            if question is None:
                return False

            for quiz in self._quizzes.values():

                if question_id in quiz.questions:

                    quiz.questions.remove(
                        question_id
                    )

                    quiz.updated_at = (
                        datetime.now().isoformat()
                    )

                    self._persist_quiz(
                        quiz
                    )

        self._emit(
            "education.quiz.question_deleted",
            {
                "question_id": question_id,
            },
        )

        return True

    # ========================================================
    # QUIZ CREATION
    # ========================================================

    def create_quiz(
        self,
        *,
        title: str,
        subject: str = "",
        chapter: str = "",
        topic: str = "",
        description: str = "",
        difficulty: str = "medium",
        time_limit_minutes: int | None = None,
        passing_percentage: float = 40.0,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Quiz:

        title = self._required_text(
            title,
            "title",
        )

        difficulty = (
            difficulty
            .strip()
            .lower()
        )

        if difficulty not in DIFFICULTIES:

            raise ValueError(
                f"Unsupported difficulty: "
                f"{difficulty}"
            )

        if (
            time_limit_minutes is not None
            and time_limit_minutes <= 0
        ):

            raise ValueError(
                "time_limit_minutes must be "
                "greater than zero."
            )

        if not 0 <= passing_percentage <= 100:

            raise ValueError(
                "passing_percentage must be "
                "between 0 and 100."
            )

        quiz = Quiz(
            id=self._new_quiz_id(),
            title=title,
            subject=subject.strip(),
            chapter=chapter.strip(),
            topic=topic.strip(),
            description=description.strip(),
            difficulty=difficulty,
            time_limit_minutes=(
                time_limit_minutes
            ),
            passing_percentage=(
                float(passing_percentage)
            ),
            tags=self._normalize_strings(
                tags
            ),
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        with self._lock:

            self._quizzes[
                quiz.id
            ] = quiz

            self._persist_quiz(
                quiz
            )

        self._emit(
            "education.quiz.created",
            quiz.to_dict(),
        )

        return quiz

    def get_quiz(
        self,
        quiz_id: str,
    ) -> Quiz | None:

        with self._lock:

            return self._quizzes.get(
                quiz_id
            )

    def get_quizzes(
        self,
    ) -> list[Quiz]:

        with self._lock:

            return list(
                self._quizzes.values()
            )

    # ========================================================
    # QUIZ QUESTION MANAGEMENT
    # ========================================================

    def add_question_to_quiz(
        self,
        quiz_id: str,
        question_id: str,
    ) -> Quiz:

        with self._lock:

            quiz = self._require_quiz(
                quiz_id
            )

            self._require_question(
                question_id
            )

            if quiz.status in {
                "active",
                "completed",
                "cancelled",
            }:

                raise RuntimeError(
                    "Questions cannot be modified "
                    "after the quiz has started."
                )

            if question_id not in quiz.questions:

                quiz.questions.append(
                    question_id
                )

            quiz.updated_at = (
                datetime.now().isoformat()
            )

            if quiz.questions:

                quiz.status = "ready"

            self._persist_quiz(
                quiz
            )

        return quiz

    def remove_question_from_quiz(
        self,
        quiz_id: str,
        question_id: str,
    ) -> Quiz:

        with self._lock:

            quiz = self._require_quiz(
                quiz_id
            )

            if quiz.status in {
                "active",
                "completed",
                "cancelled",
            }:

                raise RuntimeError(
                    "Questions cannot be modified "
                    "after the quiz has started."
                )

            if question_id in quiz.questions:

                quiz.questions.remove(
                    question_id
                )

            if not quiz.questions:

                quiz.status = "draft"

            quiz.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_quiz(
                quiz
            )

        return quiz

    def get_quiz_questions(
        self,
        quiz_id: str,
    ) -> list[QuizQuestion]:

        with self._lock:

            quiz = self._require_quiz(
                quiz_id
            )

            return [
                self._questions[question_id]
                for question_id
                in quiz.questions
                if question_id
                in self._questions
            ]

    # ========================================================
    # QUIZ LIFECYCLE
    # ========================================================

    def publish_quiz(
        self,
        quiz_id: str,
    ) -> Quiz:

        with self._lock:

            quiz = self._require_quiz(
                quiz_id
            )

            if not quiz.questions:

                raise ValueError(
                    "Cannot publish an empty quiz."
                )

            quiz.status = "ready"

            quiz.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_quiz(
                quiz
            )

        self._emit(
            "education.quiz.published",
            quiz.to_dict(),
        )

        return quiz

    def cancel_quiz(
        self,
        quiz_id: str,
    ) -> Quiz:

        with self._lock:

            quiz = self._require_quiz(
                quiz_id
            )

            if quiz.status == "completed":

                raise RuntimeError(
                    "Completed quizzes cannot "
                    "be cancelled."
                )

            quiz.status = "cancelled"

            quiz.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_quiz(
                quiz
            )

        return quiz

    # ========================================================
    # ATTEMPTS
    # ========================================================

    def start_quiz(
        self,
        quiz_id: str,
    ) -> QuizAttempt:

        with self._lock:

            quiz = self._require_quiz(
                quiz_id
            )

            if not quiz.questions:

                raise ValueError(
                    "Quiz contains no questions."
                )

            if quiz.status not in {
                "ready",
                "active",
            }:

                raise RuntimeError(
                    f"Quiz cannot be started "
                    f"while status is "
                    f"'{quiz.status}'."
                )

            started_at = datetime.now()

            attempt = QuizAttempt(
                id=self._new_attempt_id(),
                quiz_id=quiz_id,
                started_at=started_at.isoformat(),
                total_points=self._quiz_total_points(
                    quiz
                ),
            )

            self._attempts[
                attempt.id
            ] = attempt

            quiz.status = "active"

            quiz.updated_at = (
                started_at.isoformat()
            )

            self._persist_attempt(
                attempt
            )

            self._persist_quiz(
                quiz
            )

        self._emit(
            "education.quiz.started",
            attempt.to_dict(),
        )

        return attempt

    def get_attempt(
        self,
        attempt_id: str,
    ) -> QuizAttempt | None:

        with self._lock:

            return self._attempts.get(
                attempt_id
            )

    def get_attempts(
        self,
        quiz_id: str | None = None,
    ) -> list[QuizAttempt]:

        with self._lock:

            attempts = list(
                self._attempts.values()
            )

        if quiz_id:

            attempts = [
                attempt
                for attempt in attempts
                if attempt.quiz_id
                == quiz_id
            ]

        return sorted(
            attempts,
            key=lambda item: item.started_at,
            reverse=True,
        )

    # ========================================================
    # ANSWERS
    # ========================================================

    def submit_answer(
        self,
        attempt_id: str,
        question_id: str,
        answer: str,
        *,
        confidence: int | None = None,
        time_spent_seconds: float | None = None,
    ) -> QuizAnswer:

        answer = (
            answer.strip()
            if isinstance(answer, str)
            else str(answer)
        )

        with self._lock:

            attempt = self._require_attempt(
                attempt_id
            )

            if attempt.status != "active":

                raise RuntimeError(
                    "This quiz attempt is no longer active."
                )

            quiz = self._require_quiz(
                attempt.quiz_id
            )

            if question_id not in quiz.questions:

                raise ValueError(
                    "Question does not belong "
                    "to this quiz."
                )

            question = self._require_question(
                question_id
            )

            if confidence is not None:

                if not 0 <= confidence <= 100:

                    raise ValueError(
                        "confidence must be "
                        "between 0 and 100."
                    )

            if (
                time_spent_seconds is not None
                and time_spent_seconds < 0
            ):

                raise ValueError(
                    "time_spent_seconds cannot "
                    "be negative."
                )

            is_correct = (
                self._check_answer(
                    question,
                    answer,
                )
            )

            points = (
                question.points
                if is_correct
                else 0.0
            )

            feedback = (
                question.explanation
                if is_correct
                else self._incorrect_feedback(
                    question
                )
            )

            quiz_answer = QuizAnswer(
                question_id=question_id,
                answer=answer,
                is_correct=is_correct,
                points_earned=points,
                time_spent_seconds=(
                    time_spent_seconds
                ),
                confidence=confidence,
                feedback=feedback,
            )

            previous = attempt.answers.get(
                question_id
            )

            if previous is not None:

                attempt.earned_points -= (
                    previous.points_earned
                )

                if previous.is_correct:

                    attempt.correct_answers -= 1

                else:

                    attempt.incorrect_answers -= 1

            attempt.answers[
                question_id
            ] = quiz_answer

            attempt.earned_points += points

            if is_correct:

                attempt.correct_answers += 1

            else:

                attempt.incorrect_answers += 1

            attempt.percentage = (
                self._percentage(
                    attempt.earned_points,
                    attempt.total_points,
                )
            )

            self._persist_attempt(
                attempt
            )

        self._emit(
            "education.quiz.answer_submitted",
            {
                "attempt_id": attempt_id,
                "question_id": question_id,
                "answer": quiz_answer.to_dict(),
            },
        )

        return quiz_answer

    # ========================================================
    # FINISH QUIZ
    # ========================================================

    def finish_quiz(
        self,
        attempt_id: str,
    ) -> QuizAttempt:

        with self._lock:

            attempt = self._require_attempt(
                attempt_id
            )

            if attempt.status != "active":

                raise RuntimeError(
                    "Quiz attempt is already finished."
                )

            quiz = self._require_quiz(
                attempt.quiz_id
            )

            now = datetime.now()

            started = datetime.fromisoformat(
                attempt.started_at
            )

            attempt.duration_seconds = (
                now - started
            ).total_seconds()

            unanswered = (
                len(quiz.questions)
                - len(attempt.answers)
            )

            attempt.skipped_answers = max(
                0,
                unanswered,
            )

            attempt.percentage = (
                self._percentage(
                    attempt.earned_points,
                    attempt.total_points,
                )
            )

            attempt.passed = (
                attempt.percentage
                >= quiz.passing_percentage
            )

            attempt.status = "completed"

            attempt.completed_at = (
                now.isoformat()
            )

            quiz.status = "completed"

            quiz.updated_at = (
                now.isoformat()
            )

            self._persist_attempt(
                attempt
            )

            self._persist_quiz(
                quiz
            )

        self._emit(
            "education.quiz.completed",
            attempt.to_dict(),
        )

        return attempt

    # ========================================================
    # CANCEL ATTEMPT
    # ========================================================

    def cancel_attempt(
        self,
        attempt_id: str,
    ) -> QuizAttempt:

        with self._lock:

            attempt = self._require_attempt(
                attempt_id
            )

            if attempt.status != "active":

                return attempt

            attempt.status = "cancelled"

            attempt.completed_at = (
                datetime.now().isoformat()
            )

            self._persist_attempt(
                attempt
            )

        self._emit(
            "education.quiz.attempt_cancelled",
            attempt.to_dict(),
        )

        return attempt

    # ========================================================
    # TIME LIMIT
    # ========================================================

    def is_expired(
        self,
        attempt_id: str,
    ) -> bool:

        with self._lock:

            attempt = self._require_attempt(
                attempt_id
            )

            quiz = self._require_quiz(
                attempt.quiz_id
            )

            if (
                quiz.time_limit_minutes
                is None
            ):
                return False

            if attempt.status != "active":
                return True

            started = datetime.fromisoformat(
                attempt.started_at
            )

            elapsed = (
                datetime.now()
                - started
            ).total_seconds()

            return (
                elapsed
                >= quiz.time_limit_minutes
                * 60
            )

    def expire_attempt(
        self,
        attempt_id: str,
    ) -> QuizAttempt:

        with self._lock:

            attempt = self._require_attempt(
                attempt_id
            )

            if not self.is_expired(
                attempt_id
            ):

                raise RuntimeError(
                    "Attempt has not expired."
                )

            attempt.status = "expired"

            attempt.completed_at = (
                datetime.now().isoformat()
            )

            self._persist_attempt(
                attempt
            )

        self._emit(
            "education.quiz.attempt_expired",
            attempt.to_dict(),
        )

        return attempt

    # ========================================================
    # RESULT
    # ========================================================

    def get_result(
        self,
        attempt_id: str,
    ) -> dict[str, Any]:

        with self._lock:

            attempt = self._require_attempt(
                attempt_id
            )

            quiz = self._require_quiz(
                attempt.quiz_id
            )

            question_results = []

            for question_id in quiz.questions:

                question = (
                    self._questions.get(
                        question_id
                    )
                )

                if question is None:
                    continue

                answer = (
                    attempt.answers.get(
                        question_id
                    )
                )

                question_results.append(
                    {
                        "question_id": question.id,
                        "question": question.question,
                        "answer": (
                            answer.answer
                            if answer
                            else None
                        ),
                        "correct_answer": (
                            question.correct_answer
                        ),
                        "is_correct": (
                            answer.is_correct
                            if answer
                            else False
                        ),
                        "points": question.points,
                        "points_earned": (
                            answer.points_earned
                            if answer
                            else 0.0
                        ),
                        "explanation": (
                            question.explanation
                        ),
                    }
                )

            return {
                "attempt_id": attempt.id,
                "quiz_id": quiz.id,
                "quiz_title": quiz.title,
                "status": attempt.status,
                "score": attempt.earned_points,
                "total_points": attempt.total_points,
                "percentage": attempt.percentage,
                "passed": attempt.passed,
                "correct_answers": (
                    attempt.correct_answers
                ),
                "incorrect_answers": (
                    attempt.incorrect_answers
                ),
                "skipped_answers": (
                    attempt.skipped_answers
                ),
                "duration_seconds": (
                    attempt.duration_seconds
                ),
                "questions": question_results,
            }

    # ========================================================
    # SEARCH / FILTER
    # ========================================================

    def search_questions(
        self,
        query: str,
    ) -> list[QuizQuestion]:

        query = self._required_text(
            query,
            "query",
        ).casefold()

        with self._lock:

            questions = list(
                self._questions.values()
            )

        results = []

        for question in questions:

            searchable = " ".join(
                [
                    question.question,
                    question.subject,
                    question.chapter,
                    question.topic,
                    question.difficulty,
                    question.explanation,
                    " ".join(question.options),
                    " ".join(question.tags),
                ]
            ).casefold()

            if query in searchable:

                results.append(question)

        return results

    def get_questions_by_subject(
        self,
        subject: str,
    ) -> list[QuizQuestion]:

        subject = self._required_text(
            subject,
            "subject",
        ).casefold()

        with self._lock:

            return [
                question
                for question
                in self._questions.values()
                if question.subject.casefold()
                == subject
            ]

    def get_questions_by_topic(
        self,
        topic: str,
    ) -> list[QuizQuestion]:

        topic = self._required_text(
            topic,
            "topic",
        ).casefold()

        with self._lock:

            return [
                question
                for question
                in self._questions.values()
                if question.topic.casefold()
                == topic
            ]

    # ========================================================
    # WEAK TOPICS
    # ========================================================

    def get_weak_topics(
        self,
        *,
        minimum_attempts: int = 1,
    ) -> list[dict[str, Any]]:

        topic_stats: dict[
            str,
            dict[str, Any],
        ] = {}

        with self._lock:

            attempts = list(
                self._attempts.values()
            )

            questions = dict(
                self._questions
            )

        for attempt in attempts:

            for answer in attempt.answers.values():

                question = questions.get(
                    answer.question_id
                )

                if question is None:
                    continue

                topic = (
                    question.topic
                    or question.chapter
                    or question.subject
                    or "General"
                )

                if topic not in topic_stats:

                    topic_stats[topic] = {
                        "topic": topic,
                        "attempts": 0,
                        "correct": 0,
                        "incorrect": 0,
                        "accuracy": 0.0,
                    }

                stats = topic_stats[
                    topic
                ]

                stats["attempts"] += 1

                if answer.is_correct:

                    stats["correct"] += 1

                else:

                    stats["incorrect"] += 1

        results = []

        for stats in topic_stats.values():

            if (
                stats["attempts"]
                < minimum_attempts
            ):
                continue

            stats["accuracy"] = (
                stats["correct"]
                / stats["attempts"]
            ) * 100

            results.append(stats)

        results.sort(
            key=lambda item: (
                item["accuracy"],
                -item["attempts"],
            )
        )

        return results

    # ========================================================
    # ADAPTIVE DIFFICULTY
    # ========================================================

    def recommend_difficulty(
        self,
        *,
        subject: str | None = None,
        topic: str | None = None,
    ) -> str:

        questions = self._filter_questions(
            subject=subject,
            topic=topic,
        )

        if not questions:
            return "medium"

        relevant_question_ids = {
            question.id
            for question in questions
        }

        with self._lock:

            answers = [
                answer
                for attempt
                in self._attempts.values()
                for answer
                in attempt.answers.values()
                if answer.question_id
                in relevant_question_ids
            ]

        if not answers:
            return "medium"

        accuracy = (
            sum(
                answer.is_correct
                for answer in answers
            )
            / len(answers)
        ) * 100

        if accuracy >= 85:
            return "hard"

        if accuracy >= 60:
            return "medium"

        return "easy"

    # ========================================================
    # QUIZ GENERATION
    # ========================================================

    def generate_quiz(
        self,
        *,
        title: str,
        subject: str | None = None,
        chapter: str | None = None,
        topic: str | None = None,
        difficulty: str | None = None,
        count: int = 10,
        time_limit_minutes: int | None = None,
        passing_percentage: float = 40.0,
    ) -> Quiz:

        if count <= 0:

            raise ValueError(
                "count must be greater than zero."
            )

        if difficulty is None:

            difficulty = self.recommend_difficulty(
                subject=subject,
                topic=topic,
            )

        difficulty = difficulty.lower()

        if difficulty not in DIFFICULTIES:

            raise ValueError(
                f"Unsupported difficulty: "
                f"{difficulty}"
            )

        questions = self._filter_questions(
            subject=subject,
            chapter=chapter,
            topic=topic,
            difficulty=difficulty,
        )

        if len(questions) < count:

            raise ValueError(
                f"Only {len(questions)} matching "
                f"questions are available; "
                f"{count} requested."
            )

        questions = questions[:count]

        quiz = self.create_quiz(
            title=title,
            subject=subject or "",
            chapter=chapter or "",
            topic=topic or "",
            difficulty=difficulty,
            time_limit_minutes=(
                time_limit_minutes
            ),
            passing_percentage=(
                passing_percentage
            ),
        )

        for question in questions:

            self.add_question_to_quiz(
                quiz.id,
                question.id,
            )

        self.publish_quiz(
            quiz.id
        )

        return quiz

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
    ) -> QuizStatistics:

        with self._lock:

            quizzes = list(
                self._quizzes.values()
            )

            attempts = list(
                self._attempts.values()
            )

        completed = [
            attempt
            for attempt in attempts
            if attempt.status == "completed"
        ]

        stats = QuizStatistics()

        stats.total_quizzes = len(
            quizzes
        )

        stats.completed_attempts = len(
            completed
        )

        if completed:

            scores = [
                attempt.percentage
                for attempt in completed
            ]

            stats.average_score = (
                sum(scores)
                / len(scores)
            )

            stats.highest_score = max(
                scores
            )

            stats.lowest_score = min(
                scores
            )

        for attempt in completed:

            stats.total_questions_answered += (
                len(attempt.answers)
            )

            stats.total_correct_answers += (
                attempt.correct_answers
            )

            stats.total_incorrect_answers += (
                attempt.incorrect_answers
            )

            stats.total_time_seconds += (
                attempt.duration_seconds
            )

            if attempt.passed:

                stats.passed_attempts += 1

            else:

                stats.failed_attempts += 1

        if stats.total_questions_answered:

            stats.overall_accuracy = (
                stats.total_correct_answers
                / stats.total_questions_answered
            ) * 100

        return stats

    # ========================================================
    # IMPORT / EXPORT
    # ========================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "questions": [
                    question.to_dict()
                    for question
                    in self._questions.values()
                ],
                "quizzes": [
                    quiz.to_dict()
                    for quiz
                    in self._quizzes.values()
                ],
                "attempts": [
                    attempt.to_dict()
                    for attempt
                    in self._attempts.values()
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
                "Quiz state must be a dictionary."
            )

        with self._lock:

            if replace:

                self._questions.clear()
                self._quizzes.clear()
                self._attempts.clear()

            for raw in data.get(
                "questions",
                [],
            ):

                question = QuizQuestion(
                    **raw
                )

                self._questions[
                    question.id
                ] = question

            for raw in data.get(
                "quizzes",
                [],
            ):

                quiz = Quiz(
                    **raw
                )

                self._quizzes[
                    quiz.id
                ] = quiz

            for raw in data.get(
                "attempts",
                [],
            ):

                raw_answers = raw.get(
                    "answers",
                    {},
                )

                answers = {
                    key: QuizAnswer(
                        **value
                    )
                    for key, value
                    in raw_answers.items()
                }

                raw = dict(raw)

                raw["answers"] = answers

                attempt = QuizAttempt(
                    **raw
                )

                self._attempts[
                    attempt.id
                ] = attempt

    # ========================================================
    # STORAGE
    # ========================================================

    def _persist_question(
        self,
        question: QuizQuestion,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_quiz_question",
            None,
        )

        if callable(method):

            try:

                method(
                    question.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist quiz question."
                )

    def _persist_quiz(
        self,
        quiz: Quiz,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_quiz",
            None,
        )

        if callable(method):

            try:

                method(
                    quiz.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist quiz."
                )

    def _persist_attempt(
        self,
        attempt: QuizAttempt,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_quiz_attempt",
            None,
        )

        if callable(method):

            try:

                method(
                    attempt.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist quiz attempt."
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

    def _filter_questions(
        self,
        *,
        subject: str | None = None,
        chapter: str | None = None,
        topic: str | None = None,
        difficulty: str | None = None,
    ) -> list[QuizQuestion]:

        with self._lock:

            questions = list(
                self._questions.values()
            )

        if subject:

            subject = subject.casefold()

            questions = [
                question
                for question in questions
                if question.subject.casefold()
                == subject
            ]

        if chapter:

            chapter = chapter.casefold()

            questions = [
                question
                for question in questions
                if question.chapter.casefold()
                == chapter
            ]

        if topic:

            topic = topic.casefold()

            questions = [
                question
                for question in questions
                if question.topic.casefold()
                == topic
            ]

        if difficulty:

            difficulty = (
                difficulty.casefold()
            )

            questions = [
                question
                for question in questions
                if question.difficulty.casefold()
                == difficulty
            ]

        return questions

    def _check_answer(
        self,
        question: QuizQuestion,
        answer: str,
    ) -> bool:

        if self.answer_checker is not None:

            method = getattr(
                self.answer_checker,
                "check",
                None,
            )

            if callable(method):

                try:

                    result = method(
                        question=question,
                        answer=answer,
                    )

                    if isinstance(
                        result,
                        bool,
                    ):

                        return result

                except Exception:

                    logger.exception(
                        "External answer checker failed."
                    )

        if question.question_type in {
            "mcq",
            "true_false",
        }:

            return (
                self._normalize_answer(
                    answer
                )
                == self._normalize_answer(
                    question.correct_answer
                )
            )

        # Short-answer fallback.
        #
        # RENIX's dedicated answer_checker.py
        # can later provide semantic evaluation.
        return (
            self._normalize_answer(
                answer
            )
            == self._normalize_answer(
                question.correct_answer
            )
        )

    @staticmethod
    def _normalize_answer(
        answer: str,
    ) -> str:

        return " ".join(
            str(answer)
            .strip()
            .casefold()
            .split()
        )

    @staticmethod
    def _answer_in_options(
        answer: str,
        options: Sequence[str],
    ) -> bool:

        normalized = (
            QuizEngine._normalize_answer(
                answer
            )
        )

        return any(
            QuizEngine._normalize_answer(
                option
            )
            == normalized
            for option in options
        )

    @staticmethod
    def _incorrect_feedback(
        question: QuizQuestion,
    ) -> str:

        if question.explanation:

            return question.explanation

        return (
            "Incorrect. "
            f"The correct answer is "
            f"{question.correct_answer}."
        )

    @staticmethod
    def _percentage(
        earned: float,
        total: float,
    ) -> float:

        if total <= 0:
            return 0.0

        return round(
            (earned / total) * 100,
            2,
        )

    @staticmethod
    def _quiz_total_points(
        quiz: Quiz,
    ) -> float:

        return 0.0

    def _calculate_quiz_total_points(
        self,
        quiz: Quiz,
    ) -> float:

        return sum(
            self._questions[
                question_id
            ].points
            for question_id
            in quiz.questions
            if question_id
            in self._questions
        )

    def _require_question(
        self,
        question_id: str,
    ) -> QuizQuestion:

        question = self._questions.get(
            question_id
        )

        if question is None:

            raise KeyError(
                f"Question not found: "
                f"{question_id}"
            )

        return question

    def _require_quiz(
        self,
        quiz_id: str,
    ) -> Quiz:

        quiz = self._quizzes.get(
            quiz_id
        )

        if quiz is None:

            raise KeyError(
                f"Quiz not found: "
                f"{quiz_id}"
            )

        return quiz

    def _require_attempt(
        self,
        attempt_id: str,
    ) -> QuizAttempt:

        attempt = self._attempts.get(
            attempt_id
        )

        if attempt is None:

            raise KeyError(
                f"Quiz attempt not found: "
                f"{attempt_id}"
            )

        return attempt

    @staticmethod
    def _required_text(
        value: str,
        name: str,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):

            raise TypeError(
                f"{name} must be a string."
            )

        value = value.strip()

        if not value:

            raise ValueError(
                f"{name} cannot be empty."
            )

        return value

    @staticmethod
    def _normalize_strings(
        values: Iterable[str] | None,
    ) -> list[str]:

        if values is None:
            return []

        result = []

        for value in values:

            value = str(value).strip()

            if value and value not in result:

                result.append(value)

        return result

    @staticmethod
    def _new_question_id() -> str:

        return (
            "question_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _new_quiz_id() -> str:

        return (
            "quiz_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _new_attempt_id() -> str:

        return (
            "attempt_"
            + uuid.uuid4().hex
        )

    def __len__(self) -> int:

        with self._lock:
            return len(
                self._quizzes
            )


__all__ = [
    "QUESTION_TYPES",
    "DIFFICULTIES",
    "QUIZ_STATUSES",
    "ATTEMPT_STATUSES",
    "QuizQuestion",
    "Quiz",
    "QuizAnswer",
    "QuizAttempt",
    "QuizStatistics",
    "QuizEngine",
]


