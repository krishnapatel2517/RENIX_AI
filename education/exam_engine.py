"""
RENIX Education — Exam Engine

Handles:
- Exam creation
- Question banks
- Multiple question types
- Exam sessions
- Timers
- Answer submission
- Automatic evaluation
- Marks calculation
- Negative marking
- Section-wise scoring
- Question navigation
- Attempt history
- Performance analysis
- Weak-topic detection
- Exam export/import

Designed to integrate with:
    education/study_manager.py
    education/quiz_engine.py
    education/question_generator.py
    education/answer_checker.py
    education/progress_tracker.py
    memory/memory_manager.py
    database/repositories.py
    core/event_bus.py
"""

from __future__ import annotations

import logging
import random
import threading
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

QUESTION_TYPES = {
    "mcq",
    "multiple_select",
    "true_false",
    "short_answer",
    "long_answer",
    "fill_blank",
}

EXAM_STATUSES = {
    "draft",
    "ready",
    "active",
    "completed",
    "cancelled",
}

SESSION_STATUSES = {
    "active",
    "submitted",
    "expired",
    "cancelled",
}


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class ExamQuestion:
    """Represents a question inside an exam."""

    id: str

    question: str

    question_type: str

    marks: float = 1.0

    negative_marks: float = 0.0

    options: list[str] = field(
        default_factory=list
    )

    correct_answer: Any = None

    explanation: str = ""

    subject: str = ""

    chapter: str = ""

    topic: str = ""

    difficulty: str = "medium"

    tags: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Exam:
    """Represents a complete examination."""

    id: str

    title: str

    description: str = ""

    subject: str = ""

    duration_minutes: int = 60

    total_marks: float = 0.0

    passing_percentage: float = 35.0

    shuffle_questions: bool = False

    shuffle_options: bool = False

    negative_marking: bool = False

    questions: list[str] = field(
        default_factory=list
    )

    status: str = "draft"

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubmittedAnswer:
    """Stores one answer submitted during an exam."""

    question_id: str

    answer: Any

    correct: bool = False

    marks_awarded: float = 0.0

    answered_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    response_time_seconds: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExamResult:
    """Final result of an exam session."""

    session_id: str

    exam_id: str

    total_marks: float

    marks_obtained: float

    percentage: float

    passed: bool

    correct_answers: int

    wrong_answers: int

    unanswered: int

    attempted: int

    time_taken_seconds: float

    completed_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    question_results: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )

    topic_performance: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExamSession:
    """Represents a student's live exam attempt."""

    id: str

    exam_id: str

    question_order: list[str]

    started_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    submitted_at: str | None = None

    status: str = "active"

    current_question_index: int = 0

    answers: dict[str, SubmittedAnswer] = field(
        default_factory=dict
    )

    result: ExamResult | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        if self.result is not None:

            data["result"] = (
                self.result.to_dict()
            )

        return data


# ============================================================
# EXAM ENGINE
# ============================================================


class ExamEngine:
    """
    Main RENIX examination system.

    Example:

        engine = ExamEngine()

        exam = engine.create_exam(
            title="Science Unit Test",
            subject="Science",
            duration_minutes=60,
        )

        question = engine.add_question(
            exam.id,
            question="What is H2O?",
            question_type="mcq",
            options=[
                "Water",
                "Oxygen",
                "Hydrogen",
                "Carbon dioxide",
            ],
            correct_answer="Water",
            marks=1,
        )

        session = engine.start_exam(
            exam.id
        )

        engine.submit_answer(
            session.id,
            question.id,
            "Water",
        )

        result = engine.submit_exam(
            session.id
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

        self._exams: dict[
            str,
            Exam,
        ] = {}

        self._questions: dict[
            str,
            ExamQuestion,
        ] = {}

        self._sessions: dict[
            str,
            ExamSession,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "RENIX ExamEngine initialized."
        )

    # ========================================================
    # EXAM CREATION
    # ========================================================

    def create_exam(
        self,
        *,
        title: str,
        description: str = "",
        subject: str = "",
        duration_minutes: int = 60,
        passing_percentage: float = 35.0,
        shuffle_questions: bool = False,
        shuffle_options: bool = False,
        negative_marking: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Exam:

        title = self._required_text(
            title,
            "title",
        )

        if duration_minutes <= 0:

            raise ValueError(
                "duration_minutes must "
                "be greater than zero."
            )

        if not 0 <= passing_percentage <= 100:

            raise ValueError(
                "passing_percentage must "
                "be between 0 and 100."
            )

        exam = Exam(
            id=self._new_id("exam"),
            title=title,
            description=description.strip(),
            subject=subject.strip(),
            duration_minutes=duration_minutes,
            passing_percentage=passing_percentage,
            shuffle_questions=shuffle_questions,
            shuffle_options=shuffle_options,
            negative_marking=negative_marking,
            metadata=dict(metadata or {}),
        )

        with self._lock:

            self._exams[
                exam.id
            ] = exam

            self._persist_exam(
                exam
            )

        self._emit(
            "education.exam.created",
            exam.to_dict(),
        )

        return exam

    def get_exam(
        self,
        exam_id: str,
    ) -> Exam | None:

        with self._lock:

            return self._exams.get(
                exam_id
            )

    def get_exams(
        self,
        *,
        subject: str | None = None,
        status: str | None = None,
    ) -> list[Exam]:

        with self._lock:

            exams = list(
                self._exams.values()
            )

        if subject:

            subject_key = subject.casefold()

            exams = [
                exam
                for exam in exams
                if exam.subject.casefold()
                == subject_key
            ]

        if status:

            exams = [
                exam
                for exam in exams
                if exam.status == status
            ]

        return exams

    # ========================================================
    # QUESTION BANK
    # ========================================================

    def add_question(
        self,
        exam_id: str,
        *,
        question: str,
        question_type: str = "mcq",
        marks: float = 1.0,
        negative_marks: float = 0.0,
        options: Iterable[str] | None = None,
        correct_answer: Any = None,
        explanation: str = "",
        subject: str = "",
        chapter: str = "",
        topic: str = "",
        difficulty: str = "medium",
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExamQuestion:

        question = self._required_text(
            question,
            "question",
        )

        question_type = (
            question_type.strip().lower()
        )

        if question_type not in QUESTION_TYPES:

            raise ValueError(
                "Unsupported question type: "
                + question_type
            )

        if marks <= 0:

            raise ValueError(
                "marks must be greater than zero."
            )

        if negative_marks < 0:

            raise ValueError(
                "negative_marks cannot be negative."
            )

        options_list = [
            str(option).strip()
            for option in (
                options or []
            )
            if str(option).strip()
        ]

        if question_type in {
            "mcq",
            "multiple_select",
        } and len(options_list) < 2:

            raise ValueError(
                "MCQ questions require "
                "at least two options."
            )

        if (
            question_type == "true_false"
            and correct_answer is None
        ):

            raise ValueError(
                "true_false questions require "
                "a correct answer."
            )

        question_obj = ExamQuestion(
            id=self._new_id("question"),
            question=question,
            question_type=question_type,
            marks=float(marks),
            negative_marks=float(
                negative_marks
            ),
            options=options_list,
            correct_answer=correct_answer,
            explanation=explanation.strip(),
            subject=subject.strip(),
            chapter=chapter.strip(),
            topic=topic.strip(),
            difficulty=difficulty.strip().lower(),
            tags=self._normalize_strings(
                tags
            ),
            metadata=dict(metadata or {}),
        )

        with self._lock:

            exam = self._require_exam(
                exam_id
            )

            if exam.status in {
                "active",
                "completed",
            }:

                raise RuntimeError(
                    "Questions cannot be added "
                    "to an active/completed exam."
                )

            self._questions[
                question_obj.id
            ] = question_obj

            exam.questions.append(
                question_obj.id
            )

            self._recalculate_total_marks(
                exam
            )

            exam.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_question(
                question_obj
            )

            self._persist_exam(
                exam
            )

        self._emit(
            "education.exam.question_added",
            question_obj.to_dict(),
        )

        return question_obj

    def get_question(
        self,
        question_id: str,
    ) -> ExamQuestion | None:

        with self._lock:

            return self._questions.get(
                question_id
            )

    def get_exam_questions(
        self,
        exam_id: str,
    ) -> list[ExamQuestion]:

        with self._lock:

            exam = self._require_exam(
                exam_id
            )

            return [
                self._questions[qid]
                for qid in exam.questions
                if qid in self._questions
            ]

    def remove_question(
        self,
        exam_id: str,
        question_id: str,
    ) -> bool:

        with self._lock:

            exam = self._require_exam(
                exam_id
            )

            if question_id not in exam.questions:
                return False

            exam.questions.remove(
                question_id
            )

            self._recalculate_total_marks(
                exam
            )

            exam.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_exam(
                exam
            )

        self._emit(
            "education.exam.question_removed",
            {
                "exam_id": exam_id,
                "question_id": question_id,
            },
        )

        return True

    # ========================================================
    # EXAM STATE
    # ========================================================

    def prepare_exam(
        self,
        exam_id: str,
    ) -> Exam:

        with self._lock:

            exam = self._require_exam(
                exam_id
            )

            if not exam.questions:

                raise ValueError(
                    "Cannot prepare an exam "
                    "without questions."
                )

            exam.status = "ready"

            exam.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_exam(
                exam
            )

        return exam

    # ========================================================
    # START EXAM
    # ========================================================

    def start_exam(
        self,
        exam_id: str,
    ) -> ExamSession:

        with self._lock:

            exam = self._require_exam(
                exam_id
            )

            if exam.status not in {
                "ready",
                "draft",
            }:

                raise RuntimeError(
                    "Exam cannot be started "
                    f"while status is '{exam.status}'."
                )

            if not exam.questions:

                raise ValueError(
                    "Exam has no questions."
                )

            question_order = list(
                exam.questions
            )

            if exam.shuffle_questions:

                random.shuffle(
                    question_order
                )

            if exam.shuffle_options:

                for question_id in question_order:

                    question_obj = (
                        self._questions.get(
                            question_id
                        )
                    )

                    if question_obj:

                        question_obj.options = (
                            list(
                                question_obj.options
                            )
                        )

                        random.shuffle(
                            question_obj.options
                        )

            session = ExamSession(
                id=self._new_id("session"),
                exam_id=exam_id,
                question_order=question_order,
            )

            self._sessions[
                session.id
            ] = session

            exam.status = "active"

            exam.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_exam(
                exam
            )

            self._persist_session(
                session
            )

        self._emit(
            "education.exam.started",
            session.to_dict(),
        )

        return session

    # ========================================================
    # SESSION ACCESS
    # ========================================================

    def get_session(
        self,
        session_id: str,
    ) -> ExamSession | None:

        with self._lock:

            return self._sessions.get(
                session_id
            )

    def get_current_question(
        self,
        session_id: str,
    ) -> ExamQuestion | None:

        with self._lock:

            session = self._require_session(
                session_id
            )

            if not session.question_order:
                return None

            index = (
                session.current_question_index
            )

            if not 0 <= index < len(
                session.question_order
            ):

                return None

            question_id = (
                session.question_order[index]
            )

            return self._questions.get(
                question_id
            )

    def go_to_question(
        self,
        session_id: str,
        index: int,
    ) -> ExamQuestion:

        with self._lock:

            session = self._require_session(
                session_id
            )

            self._ensure_session_active(
                session
            )

            if not 0 <= index < len(
                session.question_order
            ):

                raise IndexError(
                    "Question index is out of range."
                )

            session.current_question_index = (
                index
            )

            question = self._questions.get(
                session.question_order[index]
            )

            if question is None:

                raise KeyError(
                    "Question no longer exists."
                )

            self._persist_session(
                session
            )

            return question

    def next_question(
        self,
        session_id: str,
    ) -> ExamQuestion | None:

        with self._lock:

            session = self._require_session(
                session_id
            )

            self._ensure_session_active(
                session
            )

            if (
                session.current_question_index
                >= len(session.question_order) - 1
            ):

                return None

            session.current_question_index += 1

            self._persist_session(
                session
            )

            return self.get_current_question(
                session_id
            )

    def previous_question(
        self,
        session_id: str,
    ) -> ExamQuestion | None:

        with self._lock:

            session = self._require_session(
                session_id
            )

            self._ensure_session_active(
                session
            )

            if (
                session.current_question_index
                <= 0
            ):

                return None

            session.current_question_index -= 1

            self._persist_session(
                session
            )

            return self.get_current_question(
                session_id
            )

    # ========================================================
    # ANSWER SUBMISSION
    # ========================================================

    def submit_answer(
        self,
        session_id: str,
        question_id: str,
        answer: Any,
        *,
        response_time_seconds: float | None = None,
    ) -> SubmittedAnswer:

        with self._lock:

            session = self._require_session(
                session_id
            )

            self._ensure_session_active(
                session
            )

            if question_id not in (
                session.question_order
            ):

                raise ValueError(
                    "Question does not belong "
                    "to this exam session."
                )

            question = self._questions.get(
                question_id
            )

            if question is None:

                raise KeyError(
                    "Question not found."
                )

            submitted = SubmittedAnswer(
                question_id=question_id,
                answer=answer,
                response_time_seconds=(
                    response_time_seconds
                ),
            )

            session.answers[
                question_id
            ] = submitted

            self._persist_session(
                session
            )

        self._emit(
            "education.exam.answer_submitted",
            submitted.to_dict(),
        )

        return submitted

    # ========================================================
    # NAVIGATION / PROGRESS
    # ========================================================

    def get_progress(
        self,
        session_id: str,
    ) -> dict[str, Any]:

        with self._lock:

            session = self._require_session(
                session_id
            )

            total = len(
                session.question_order
            )

            answered = len(
                session.answers
            )

            unanswered = (
                total - answered
            )

            return {
                "session_id": session.id,
                "total_questions": total,
                "answered": answered,
                "unanswered": unanswered,
                "progress_percentage": (
                    round(
                        answered / total * 100,
                        2,
                    )
                    if total
                    else 0.0
                ),
                "current_question": (
                    session.current_question_index
                    + 1
                ),
            }

    # ========================================================
    # SUBMIT EXAM
    # ========================================================

    def submit_exam(
        self,
        session_id: str,
    ) -> ExamResult:

        with self._lock:

            session = self._require_session(
                session_id
            )

            self._ensure_session_active(
                session
            )

            exam = self._require_exam(
                session.exam_id
            )

            result = self._evaluate_session(
                session,
                exam,
            )

            session.result = result

            session.status = "submitted"

            session.submitted_at = (
                datetime.now().isoformat()
            )

            exam.status = "completed"

            exam.updated_at = (
                datetime.now().isoformat()
            )

            self._persist_session(
                session
            )

            self._persist_exam(
                exam
            )

        self._emit(
            "education.exam.completed",
            result.to_dict(),
        )

        return result

    # ========================================================
    # EVALUATION
    # ========================================================

    def _evaluate_session(
        self,
        session: ExamSession,
        exam: Exam,
    ) -> ExamResult:

        total_marks = float(
            exam.total_marks
        )

        obtained = 0.0

        correct = 0

        wrong = 0

        unanswered = 0

        question_results = {}

        topic_performance = {}

        for question_id in (
            session.question_order
        ):

            question = self._questions.get(
                question_id
            )

            if question is None:
                continue

            submitted = session.answers.get(
                question_id
            )

            if submitted is None:

                unanswered += 1

                question_results[
                    question_id
                ] = {
                    "status": "unanswered",
                    "marks": 0.0,
                }

                self._update_topic(
                    topic_performance,
                    question,
                    0.0,
                    False,
                )

                continue

            is_correct = self._check_answer(
                question,
                submitted.answer,
            )

            submitted.correct = (
                is_correct
            )

            if is_correct:

                marks = question.marks

                correct += 1

            else:

                wrong += 1

                if (
                    exam.negative_marking
                    and question.negative_marks > 0
                ):

                    marks = -question.negative_marks

                else:

                    marks = 0.0

            submitted.marks_awarded = (
                marks
            )

            obtained += marks

            question_results[
                question_id
            ] = {
                "status": (
                    "correct"
                    if is_correct
                    else "wrong"
                ),
                "marks": marks,
                "correct_answer": (
                    question.correct_answer
                ),
                "submitted_answer": (
                    submitted.answer
                ),
                "explanation": (
                    question.explanation
                ),
            }

            self._update_topic(
                topic_performance,
                question,
                marks,
                is_correct,
            )

        percentage = (
            obtained / total_marks * 100
            if total_marks > 0
            else 0.0
        )

        passed = (
            percentage
            >= exam.passing_percentage
        )

        started = datetime.fromisoformat(
            session.started_at
        )

        elapsed = (
            datetime.now() - started
        ).total_seconds()

        return ExamResult(
            session_id=session.id,
            exam_id=exam.id,
            total_marks=total_marks,
            marks_obtained=round(
                obtained,
                2,
            ),
            percentage=round(
                percentage,
                2,
            ),
            passed=passed,
            correct_answers=correct,
            wrong_answers=wrong,
            unanswered=unanswered,
            attempted=(
                correct + wrong
            ),
            time_taken_seconds=round(
                elapsed,
                2,
            ),
            question_results=(
                question_results
            ),
            topic_performance=(
                topic_performance
            ),
        )

    # ========================================================
    # ANSWER CHECKING
    # ========================================================

    def _check_answer(
        self,
        question: ExamQuestion,
        answer: Any,
    ) -> bool:

        if self.answer_checker is not None:

            checker = getattr(
                self.answer_checker,
                "check",
                None,
            )

            if callable(checker):

                try:

                    result = checker(
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

        correct = (
            question.correct_answer
        )

        if question.question_type == "multiple_select":

            return self._normalize_answers(
                answer
            ) == self._normalize_answers(
                correct
            )

        if question.question_type == "true_false":

            return (
                str(answer).strip().casefold()
                == str(correct).strip().casefold()
            )

        if isinstance(
            correct,
            str,
        ) and isinstance(
            answer,
            str,
        ):

            return (
                answer.strip().casefold()
                == correct.strip().casefold()
            )

        return answer == correct

    # ========================================================
    # TOPIC ANALYSIS
    # ========================================================

    @staticmethod
    def _update_topic(
        performance: dict[str, dict[str, Any]],
        question: ExamQuestion,
        marks: float,
        correct: bool,
    ) -> None:

        topic = (
            question.topic
            or question.chapter
            or "General"
        )

        item = performance.setdefault(
            topic,
            {
                "questions": 0,
                "correct": 0,
                "marks": 0.0,
            },
        )

        item["questions"] += 1

        if correct:

            item["correct"] += 1

        item["marks"] += marks

        item["accuracy"] = round(
            item["correct"]
            / item["questions"]
            * 100,
            2,
        )

    # ========================================================
    # EXAM ANALYTICS
    # ========================================================

    def get_weak_topics(
        self,
        result: ExamResult,
        *,
        threshold: float = 60.0,
    ) -> list[dict[str, Any]]:

        weak = []

        for topic, data in (
            result.topic_performance.items()
        ):

            if data["accuracy"] < threshold:

                weak.append(
                    {
                        "topic": topic,
                        **data,
                    }
                )

        weak.sort(
            key=lambda item:
            item["accuracy"]
        )

        return weak

    def get_result_summary(
        self,
        session_id: str,
    ) -> dict[str, Any]:

        with self._lock:

            session = self._require_session(
                session_id
            )

            if session.result is None:

                raise RuntimeError(
                    "Exam has not been submitted."
                )

            result = session.result

        return {
            "session_id": result.session_id,
            "exam_id": result.exam_id,
            "marks": (
                f"{result.marks_obtained}"
                f"/{result.total_marks}"
            ),
            "percentage": (
                result.percentage
            ),
            "passed": result.passed,
            "correct": (
                result.correct_answers
            ),
            "wrong": (
                result.wrong_answers
            ),
            "unanswered": (
                result.unanswered
            ),
            "attempted": (
                result.attempted
            ),
            "time_taken_seconds": (
                result.time_taken_seconds
            ),
            "weak_topics": self.get_weak_topics(
                result
            ),
        }

    # ========================================================
    # SEARCH
    # ========================================================

    def search_questions(
        self,
        query: str,
    ) -> list[ExamQuestion]:

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
                    question.explanation,
                    " ".join(
                        question.tags
                    ),
                ]
            ).casefold()

            if query in searchable:

                results.append(question)

        return results

    # ========================================================
    # IMPORT / EXPORT
    # ========================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "exams": [
                    exam.to_dict()
                    for exam
                    in self._exams.values()
                ],
                "questions": [
                    question.to_dict()
                    for question
                    in self._questions.values()
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
                "Exam state must be a dictionary."
            )

        with self._lock:

            if replace:

                self._exams.clear()
                self._questions.clear()
                self._sessions.clear()

            for raw in data.get(
                "exams",
                [],
            ):

                exam = Exam(
                    **raw
                )

                self._exams[
                    exam.id
                ] = exam

            for raw in data.get(
                "questions",
                [],
            ):

                question = ExamQuestion(
                    **raw
                )

                self._questions[
                    question.id
                ] = question

            for raw in data.get(
                "sessions",
                [],
            ):

                answers = {}

                for qid, raw_answer in (
                    raw.get(
                        "answers",
                        {}
                    ).items()
                ):

                    answers[qid] = (
                        SubmittedAnswer(
                            **raw_answer
                        )
                    )

                result_data = raw.get(
                    "result"
                )

                result = None

                if result_data:

                    result = ExamResult(
                        **result_data
                    )

                session = ExamSession(
                    id=raw["id"],
                    exam_id=raw["exam_id"],
                    question_order=raw[
                        "question_order"
                    ],
                    started_at=raw[
                        "started_at"
                    ],
                    submitted_at=raw.get(
                        "submitted_at"
                    ),
                    status=raw.get(
                        "status",
                        "active",
                    ),
                    current_question_index=raw.get(
                        "current_question_index",
                        0,
                    ),
                    answers=answers,
                    result=result,
                    metadata=raw.get(
                        "metadata",
                        {},
                    ),
                )

                self._sessions[
                    session.id
                ] = session

    # ========================================================
    # STORAGE
    # ========================================================

    def _persist_exam(
        self,
        exam: Exam,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_exam",
            None,
        )

        if callable(method):

            try:

                method(
                    exam.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist exam."
                )

    def _persist_question(
        self,
        question: ExamQuestion,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_exam_question",
            None,
        )

        if callable(method):

            try:

                method(
                    question.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist exam question."
                )

    def _persist_session(
        self,
        session: ExamSession,
    ) -> None:

        if self.storage is None:
            return

        method = getattr(
            self.storage,
            "save_exam_session",
            None,
        )

        if callable(method):

            try:

                method(
                    session.to_dict()
                )

            except Exception:

                logger.exception(
                    "Failed to persist exam session."
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
    # HELPERS
    # ========================================================

    def _require_exam(
        self,
        exam_id: str,
    ) -> Exam:

        exam = self._exams.get(
            exam_id
        )

        if exam is None:

            raise KeyError(
                f"Exam not found: {exam_id}"
            )

        return exam

    def _require_session(
        self,
        session_id: str,
    ) -> ExamSession:

        session = self._sessions.get(
            session_id
        )

        if session is None:

            raise KeyError(
                f"Exam session not found: "
                f"{session_id}"
            )

        return session

    @staticmethod
    def _ensure_session_active(
        session: ExamSession,
    ) -> None:

        if session.status != "active":

            raise RuntimeError(
                "Exam session is no longer active."
            )

    def _recalculate_total_marks(
        self,
        exam: Exam,
    ) -> None:

        exam.total_marks = round(
            sum(
                self._questions[qid].marks
                for qid in exam.questions
                if qid in self._questions
            ),
            2,
        )

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
    def _normalize_answers(
        answer: Any,
    ) -> list[str]:

        if answer is None:

            return []

        if isinstance(
            answer,
            (list, tuple, set),
        ):

            values = answer

        else:

            values = [answer]

        return sorted(
            str(value)
            .strip()
            .casefold()
            for value in values
        )

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
    def _new_id(
        prefix: str,
    ) -> str:

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    def __len__(self) -> int:

        with self._lock:

            return len(
                self._exams
            )


__all__ = [
    "QUESTION_TYPES",
    "EXAM_STATUSES",
    "SESSION_STATUSES",
    "ExamQuestion",
    "Exam",
    "SubmittedAnswer",
    "ExamResult",
    "ExamSession",
    "ExamEngine",
]


