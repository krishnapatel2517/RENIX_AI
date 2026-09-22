"""
RENIX AI - Study Agent

Handles education and study-related tasks.

Responsibilities:
- Study planning
- Homework management
- Revision planning
- Quiz generation
- Flashcard generation
- Exam preparation
- Question generation
- Answer checking
- Progress tracking
- Subject-specific study assistance
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .base_agent import (
    AgentCapability,
    AgentPriority,
    AgentResult,
    AgentTask,
    BaseAgent,
)


class StudyAgent(BaseAgent):
    """
    RENIX education and study agent.

    Coordinates RENIX's education subsystem and provides fallback
    functionality while individual education modules are being built.
    """

    agent_name = "study_agent"

    agent_description = (
        "Helps with studying, homework, revision, quizzes, "
        "exams, questions, answers and academic progress."
    )

    agent_version = "1.0.0"

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: AgentPriority = AgentPriority.NORMAL,
        logger: Optional[logging.Logger] = None,
        event_callback=None,
        confirmation_callback=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            priority=priority,
            logger=logger,
            event_callback=event_callback,
            confirmation_callback=confirmation_callback,
            context=context,
        )

        self.study_manager = None
        self.timetable_manager = None
        self.homework_manager = None
        self.revision_manager = None
        self.quiz_engine = None
        self.flashcard_manager = None
        self.exam_engine = None
        self.question_generator = None
        self.answer_checker = None
        self.progress_tracker = None

        self._modules_loaded = False

        self._study_sessions: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._progress: Dict[
            str,
            Dict[str, Any],
        ] = {}

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def _register_default_capabilities(self) -> None:
        """Register study-related capabilities."""

        super()._register_default_capabilities()

        capabilities = [
            AgentCapability(
                name="study",
                description=(
                    "Provide study assistance and academic guidance."
                ),
            ),
            AgentCapability(
                name="study_plan",
                description=(
                    "Create a structured study plan."
                ),
            ),
            AgentCapability(
                name="timetable",
                description=(
                    "Create or manage study timetables."
                ),
            ),
            AgentCapability(
                name="homework",
                description=(
                    "Manage and organize homework."
                ),
            ),
            AgentCapability(
                name="revision",
                description=(
                    "Create revision plans and schedules."
                ),
            ),
            AgentCapability(
                name="quiz",
                description=(
                    "Generate quizzes and practice questions."
                ),
            ),
            AgentCapability(
                name="flashcards",
                description=(
                    "Create study flashcards."
                ),
            ),
            AgentCapability(
                name="exam",
                description=(
                    "Prepare for examinations."
                ),
            ),
            AgentCapability(
                name="questions",
                description=(
                    "Generate academic questions."
                ),
            ),
            AgentCapability(
                name="check_answer",
                description=(
                    "Check and evaluate answers."
                ),
            ),
            AgentCapability(
                name="progress",
                description=(
                    "Track academic progress."
                ),
            ),
        ]

        for capability in capabilities:
            self.register_capability(
                capability
            )

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def on_initialize(self) -> bool:
        """
        Initialize education modules.
        """

        self._load_education_modules()

        return True

    def _load_education_modules(self) -> None:
        """
        Load education subsystem modules.

        Missing modules are tolerated because RENIX is being constructed
        incrementally.
        """

        if self._modules_loaded:
            return

        self._modules_loaded = True

        module_map = {
            "study_manager": (
                "education.study_manager",
                "StudyManager",
            ),
            "timetable_manager": (
                "education.timetable",
                "Timetable",
            ),
            "homework_manager": (
                "education.homework",
                "HomeworkManager",
            ),
            "revision_manager": (
                "education.revision",
                "RevisionManager",
            ),
            "quiz_engine": (
                "education.quiz_engine",
                "QuizEngine",
            ),
            "flashcard_manager": (
                "education.flashcards",
                "Flashcards",
            ),
            "exam_engine": (
                "education.exam_engine",
                "ExamEngine",
            ),
            "question_generator": (
                "education.question_generator",
                "QuestionGenerator",
            ),
            "answer_checker": (
                "education.answer_checker",
                "AnswerChecker",
            ),
            "progress_tracker": (
                "education.progress_tracker",
                "ProgressTracker",
            ),
        }

        for attribute, (
            module_name,
            class_name,
        ) in module_map.items():

            try:

                module = __import__(
                    module_name,
                    fromlist=[class_name],
                )

                cls = getattr(
                    module,
                    class_name,
                )

                try:
                    instance = cls()
                except TypeError:
                    instance = cls

                setattr(
                    self,
                    attribute,
                    instance,
                )

            except Exception as exc:

                self.logger.debug(
                    "Education module %s unavailable: %s",
                    class_name,
                    exc,
                )

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    async def execute(
        self,
        task: AgentTask,
    ) -> Any:
        """
        Execute a study task.
        """

        action = self._resolve_action(
            task
        )

        handlers = {
            "study": self.study,
            "study_plan": self.create_study_plan,
            "plan": self.create_study_plan,
            "timetable": self.create_timetable,
            "homework": self.manage_homework,
            "revision": self.create_revision_plan,
            "quiz": self.generate_quiz,
            "flashcards": self.generate_flashcards,
            "flashcard": self.generate_flashcards,
            "exam": self.exam_prepare,
            "questions": self.generate_questions,
            "question": self.generate_questions,
            "check_answer": self.check_answer,
            "answer": self.check_answer,
            "progress": self.track_progress,
            "session": self.get_session,
        }

        handler = handlers.get(
            action
        )

        if handler is None:

            return self.failure_result(
                task,
                f"Unknown study action: {action}",
                message=(
                    f"I don't know how to perform "
                    f"study action '{action}'."
                ),
                started_at=task.created_at,
            )

        try:

            parameters = dict(
                task.parameters
            )

            parameters.pop(
                "action",
                None,
            )

            parameters.pop(
                "capability",
                None,
            )

            result = handler(
                **parameters
            )

            if asyncio.iscoroutine(
                result
            ):
                result = await result

            return result

        except Exception as exc:

            self.logger.exception(
                "Study action failed: %s",
                action,
            )

            return self.failure_result(
                task,
                str(exc),
                message=(
                    f"Study action '{action}' failed."
                ),
                started_at=task.created_at,
            )

    # ========================================================================
    # ACTION RESOLUTION
    # ========================================================================

    def _resolve_action(
        self,
        task: AgentTask,
    ) -> str:

        explicit_action = task.parameters.get(
            "action"
        )

        if explicit_action:

            return str(
                explicit_action
            ).strip().lower()

        instruction = (
            task.instruction
            .strip()
            .lower()
        )

        if (
            "study plan" in instruction
            or "study schedule" in instruction
            or "make a study plan" in instruction
        ):
            return "study_plan"

        if (
            "timetable" in instruction
            or "time table" in instruction
        ):
            return "timetable"

        if (
            "homework" in instruction
            or "assignment" in instruction
        ):
            return "homework"

        if (
            "revision" in instruction
            or "revise" in instruction
        ):
            return "revision"

        if (
            "quiz" in instruction
            or "test me" in instruction
        ):
            return "quiz"

        if (
            "flashcard" in instruction
            or "flash cards" in instruction
        ):
            return "flashcards"

        if (
            "exam preparation" in instruction
            or "exam prep" in instruction
            or "prepare for exam" in instruction
        ):
            return "exam"

        if (
            "check my answer" in instruction
            or "check answer" in instruction
            or "evaluate my answer" in instruction
        ):
            return "check_answer"

        if (
            "progress" in instruction
            or "marks" in instruction
            or "score" in instruction
        ):
            return "progress"

        if (
            "question" in instruction
            or "questions" in instruction
        ):
            return "questions"

        if (
            "study" in instruction
            or "teach me" in instruction
            or "explain this" in instruction
        ):
            return "study"

        return "unknown"

    # ========================================================================
    # GENERAL STUDY
    # ========================================================================

    async def study(
        self,
        topic: str,
        subject: Optional[str] = None,
        level: Optional[str] = None,
        explanation_style: str = "clear",
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Provide structured study assistance.

        The dedicated study manager is preferred when available.
        """

        if not topic or not topic.strip():

            return {
                "success": False,
                "error": (
                    "Study topic cannot be empty."
                ),
            }

        topic = topic.strip()

        if self.study_manager is not None:

            result = await self._call_module(
                self.study_manager,
                (
                    "study",
                    "teach",
                    "explain",
                    "assist",
                ),
                topic=topic,
                subject=subject,
                level=level,
                explanation_style=explanation_style,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "study",
                    "topic": topic,
                    "subject": subject,
                    "result": result,
                }

        return {
            "success": True,
            "action": "study",
            "topic": topic,
            "subject": subject,
            "level": level,
            "explanation_style": explanation_style,
            "message": (
                f"Study topic received: {topic}"
            ),
            "note": (
                "Connect the RENIX education/LLM layer "
                "for full interactive teaching."
            ),
        }

    # ========================================================================
    # STUDY PLAN
    # ========================================================================

    async def create_study_plan(
        self,
        subjects: Sequence[Any],
        available_minutes: int = 120,
        days: int = 1,
        priority_subjects: Optional[
            Sequence[str]
        ] = None,
        chapter_per_session: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Create a structured study plan.
        """

        subject_list = [
            str(subject).strip()
            for subject in (
                subjects or []
            )
            if str(subject).strip()
        ]

        if not subject_list:

            return {
                "success": False,
                "error": (
                    "At least one subject is required."
                ),
            }

        available_minutes = max(
            15,
            int(
                available_minutes
            ),
        )

        days = max(
            1,
            int(
                days
            ),
        )

        if self.study_manager is not None:

            result = await self._call_module(
                self.study_manager,
                (
                    "create_study_plan",
                    "study_plan",
                    "create_plan",
                ),
                subjects=subject_list,
                available_minutes=available_minutes,
                days=days,
                priority_subjects=priority_subjects,
                chapter_per_session=chapter_per_session,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "study_plan",
                    "plan": result,
                }

        priority = [
            str(item).strip()
            for item in (
                priority_subjects or []
            )
            if str(item).strip()
        ]

        ordered_subjects = []

        for subject in priority:

            if subject not in ordered_subjects:

                ordered_subjects.append(
                    subject
                )

        for subject in subject_list:

            if subject not in ordered_subjects:

                ordered_subjects.append(
                    subject
                )

        total_sessions = (
            days
            * max(
                1,
                len(
                    ordered_subjects
                ),
            )
        )

        minutes_per_session = max(
            15,
            available_minutes
            // max(
                1,
                len(
                    ordered_subjects
                ),
            ),
        )

        plan = []

        for day in range(
            1,
            days + 1,
        ):

            sessions = []

            for index, subject in enumerate(
                ordered_subjects
            ):

                sessions.append(
                    {
                        "subject": subject,
                        "minutes": minutes_per_session,
                        "task": (
                            "Study one chapter/topic"
                            if chapter_per_session
                            else "Study and revise"
                        ),
                    }
                )

            plan.append(
                {
                    "day": day,
                    "sessions": sessions,
                }
            )

        return {
            "success": True,
            "action": "study_plan",
            "subjects": ordered_subjects,
            "available_minutes_per_day": available_minutes,
            "days": days,
            "total_sessions": total_sessions,
            "plan": plan,
            "method": "fallback",
        }

    # ========================================================================
    # TIMETABLE
    # ========================================================================

    async def create_timetable(
        self,
        subjects: Sequence[Any],
        start_time: str = "18:00",
        end_time: str = "22:00",
        session_minutes: int = 60,
        break_minutes: int = 10,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Create a daily study timetable.
        """

        subject_list = [
            str(subject).strip()
            for subject in (
                subjects or []
            )
            if str(subject).strip()
        ]

        if not subject_list:

            return {
                "success": False,
                "error": (
                    "At least one subject is required."
                ),
            }

        if self.timetable_manager is not None:

            result = await self._call_module(
                self.timetable_manager,
                (
                    "create_timetable",
                    "create",
                    "generate",
                ),
                subjects=subject_list,
                start_time=start_time,
                end_time=end_time,
                session_minutes=session_minutes,
                break_minutes=break_minutes,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "timetable",
                    "timetable": result,
                }

        start_minutes = self._parse_time(
            start_time
        )

        end_minutes = self._parse_time(
            end_time
        )

        if end_minutes <= start_minutes:

            end_minutes += 24 * 60

        session_minutes = max(
            15,
            int(
                session_minutes
            ),
        )

        break_minutes = max(
            0,
            int(
                break_minutes
            ),
        )

        timetable = []

        current = start_minutes

        subject_index = 0

        while (
            current + session_minutes
            <= end_minutes
            and subject_index
            < len(subject_list)
        ):

            subject = subject_list[
                subject_index
                % len(subject_list)
            ]

            session_start = current
            session_end = (
                current
                + session_minutes
            )

            timetable.append(
                {
                    "subject": subject,
                    "start": self._format_time(
                        session_start
                    ),
                    "end": self._format_time(
                        session_end
                    ),
                    "minutes": session_minutes,
                }
            )

            current = (
                session_end
                + break_minutes
            )

            subject_index += 1

        return {
            "success": True,
            "action": "timetable",
            "start_time": start_time,
            "end_time": end_time,
            "sessions": timetable,
            "method": "fallback",
        }

    # ========================================================================
    # HOMEWORK
    # ========================================================================

    async def manage_homework(
        self,
        action: str = "list",
        subject: Optional[str] = None,
        task: Optional[str] = None,
        due_date: Optional[str] = None,
        homework_id: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Add, list, complete or remove homework.
        """

        action = (
            str(
                action
            )
            .strip()
            .lower()
        )

        if self.homework_manager is not None:

            result = await self._call_module(
                self.homework_manager,
                (
                    "manage",
                    action,
                    "execute",
                ),
                action=action,
                subject=subject,
                task=task,
                due_date=due_date,
                homework_id=homework_id,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "homework",
                    "result": result,
                }

        homework = self.context.setdefault(
            "homework",
            [],
        )

        if action in (
            "add",
            "create",
            "new",
        ):

            if not task:

                return {
                    "success": False,
                    "error": (
                        "Homework task cannot be empty."
                    ),
                }

            item = {
                "id": self._make_id(
                    "hw"
                ),
                "subject": subject,
                "task": task,
                "due_date": due_date,
                "completed": False,
                "created_at": self._timestamp(),
            }

            homework.append(
                item
            )

            return {
                "success": True,
                "action": "homework",
                "operation": "add",
                "homework": item,
            }

        if action in (
            "complete",
            "done",
            "finish",
        ):

            for item in homework:

                if (
                    homework_id
                    and item.get("id")
                    == homework_id
                ):

                    item["completed"] = True
                    item["completed_at"] = (
                        self._timestamp()
                    )

                    return {
                        "success": True,
                        "action": "homework",
                        "operation": "complete",
                        "homework": item,
                    }

            return {
                "success": False,
                "error": (
                    "Homework item not found."
                ),
            }

        if action in (
            "remove",
            "delete",
        ):

            for index, item in enumerate(
                homework
            ):

                if (
                    homework_id
                    and item.get("id")
                    == homework_id
                ):

                    removed = homework.pop(
                        index
                    )

                    return {
                        "success": True,
                        "action": "homework",
                        "operation": "remove",
                        "homework": removed,
                    }

            return {
                "success": False,
                "error": (
                    "Homework item not found."
                ),
            }

        return {
            "success": True,
            "action": "homework",
            "operation": "list",
            "homework": list(
                homework
            ),
            "count": len(
                homework
            ),
        }

    # ========================================================================
    # REVISION
    # ========================================================================

    async def create_revision_plan(
        self,
        subjects: Sequence[Any],
        days: int = 7,
        sessions_per_day: int = 2,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Create a revision plan.
        """

        subject_list = [
            str(subject).strip()
            for subject in (
                subjects or []
            )
            if str(subject).strip()
        ]

        if not subject_list:

            return {
                "success": False,
                "error": (
                    "At least one subject is required."
                ),
            }

        days = max(
            1,
            int(
                days
            ),
        )

        sessions_per_day = max(
            1,
            int(
                sessions_per_day
            ),
        )

        if self.revision_manager is not None:

            result = await self._call_module(
                self.revision_manager,
                (
                    "create_revision_plan",
                    "create",
                    "generate",
                ),
                subjects=subject_list,
                days=days,
                sessions_per_day=sessions_per_day,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "revision",
                    "plan": result,
                }

        plan = []

        for day in range(
            1,
            days + 1,
        ):

            sessions = []

            for session in range(
                1,
                sessions_per_day + 1,
            ):

                index = (
                    (
                        day - 1
                    )
                    * sessions_per_day
                    + session
                    - 1
                ) % len(
                    subject_list
                )

                sessions.append(
                    {
                        "session": session,
                        "subject": subject_list[
                            index
                        ],
                        "activity": (
                            "Revise chapter/topic"
                            if session == 1
                            else "Practice questions"
                        ),
                    }
                )

            plan.append(
                {
                    "day": day,
                    "sessions": sessions,
                }
            )

        return {
            "success": True,
            "action": "revision",
            "days": days,
            "sessions_per_day": sessions_per_day,
            "plan": plan,
            "method": "fallback",
        }

    # ========================================================================
    # QUIZ
    # ========================================================================

    async def generate_quiz(
        self,
        topic: str,
        subject: Optional[str] = None,
        number_of_questions: int = 10,
        difficulty: str = "medium",
        question_type: str = "mixed",
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Generate quiz questions.
        """

        if not topic or not topic.strip():

            return {
                "success": False,
                "error": (
                    "Quiz topic cannot be empty."
                ),
            }

        number_of_questions = max(
            1,
            min(
                int(
                    number_of_questions
                ),
                100,
            ),
        )

        if self.quiz_engine is not None:

            result = await self._call_module(
                self.quiz_engine,
                (
                    "generate_quiz",
                    "generate",
                    "create",
                ),
                topic=topic,
                subject=subject,
                number_of_questions=number_of_questions,
                difficulty=difficulty,
                question_type=question_type,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "quiz",
                    "topic": topic,
                    "quiz": result,
                }

        questions = []

        for index in range(
            1,
            number_of_questions + 1,
        ):

            questions.append(
                {
                    "number": index,
                    "question": (
                        f"Question {index}: "
                        f"Explain an important concept "
                        f"from {topic}."
                    ),
                    "type": question_type,
                    "difficulty": difficulty,
                    "answer": None,
                }
            )

        return {
            "success": True,
            "action": "quiz",
            "topic": topic,
            "subject": subject,
            "difficulty": difficulty,
            "question_type": question_type,
            "questions": questions,
            "method": "fallback",
            "note": (
                "Connect the question-generation/LLM "
                "layer for intelligent quiz generation."
            ),
        }

    # ========================================================================
    # FLASHCARDS
    # ========================================================================

    async def generate_flashcards(
        self,
        topic: str,
        subject: Optional[str] = None,
        number_of_cards: int = 10,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Generate flashcards.
        """

        if not topic or not topic.strip():

            return {
                "success": False,
                "error": (
                    "Flashcard topic cannot be empty."
                ),
            }

        number_of_cards = max(
            1,
            min(
                int(
                    number_of_cards
                ),
                100,
            ),
        )

        if self.flashcard_manager is not None:

            result = await self._call_module(
                self.flashcard_manager,
                (
                    "generate_flashcards",
                    "generate",
                    "create",
                ),
                topic=topic,
                subject=subject,
                number_of_cards=number_of_cards,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "flashcards",
                    "topic": topic,
                    "cards": result,
                }

        cards = []

        for index in range(
            1,
            number_of_cards + 1,
        ):

            cards.append(
                {
                    "id": index,
                    "front": (
                        f"Important concept {index} "
                        f"from {topic}"
                    ),
                    "back": (
                        "Answer not generated yet."
                    ),
                }
            )

        return {
            "success": True,
            "action": "flashcards",
            "topic": topic,
            "subject": subject,
            "cards": cards,
            "method": "fallback",
        }

    # ========================================================================
    # EXAM PREPARATION
    # ========================================================================

    async def exam_prepare(
        self,
        subjects: Sequence[Any],
        exam_date: Optional[str] = None,
        days_remaining: Optional[int] = None,
        target_score: Optional[float] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Create an exam preparation strategy.
        """

        subject_list = [
            str(subject).strip()
            for subject in (
                subjects or []
            )
            if str(subject).strip()
        ]

        if not subject_list:

            return {
                "success": False,
                "error": (
                    "At least one exam subject is required."
                ),
            }

        if self.exam_engine is not None:

            result = await self._call_module(
                self.exam_engine,
                (
                    "prepare",
                    "create_plan",
                    "exam_prepare",
                ),
                subjects=subject_list,
                exam_date=exam_date,
                days_remaining=days_remaining,
                target_score=target_score,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "exam",
                    "plan": result,
                }

        if days_remaining is None:

            days_remaining = 7

        days_remaining = max(
            1,
            int(
                days_remaining
            ),
        )

        phases = []

        if days_remaining <= 3:

            phases = [
                {
                    "phase": "Rapid Revision",
                    "days": days_remaining,
                    "focus": (
                        "High-priority chapters, formulas, "
                        "definitions and practice questions."
                    ),
                }
            ]

        else:

            first_phase_days = max(
                1,
                days_remaining // 2,
            )

            second_phase_days = max(
                1,
                days_remaining // 3,
            )

            final_phase_days = max(
                1,
                days_remaining
                - first_phase_days
                - second_phase_days,
            )

            phases = [
                {
                    "phase": "Concept Revision",
                    "days": first_phase_days,
                    "focus": (
                        "Complete concepts and weak chapters."
                    ),
                },
                {
                    "phase": "Practice",
                    "days": second_phase_days,
                    "focus": (
                        "Questions, examples and previous papers."
                    ),
                },
                {
                    "phase": "Final Revision",
                    "days": final_phase_days,
                    "focus": (
                        "Mock tests, mistakes and rapid revision."
                    ),
                },
            ]

        return {
            "success": True,
            "action": "exam",
            "subjects": subject_list,
            "exam_date": exam_date,
            "days_remaining": days_remaining,
            "target_score": target_score,
            "phases": phases,
            "method": "fallback",
        }

    # ========================================================================
    # QUESTION GENERATION
    # ========================================================================

    async def generate_questions(
        self,
        topic: str,
        subject: Optional[str] = None,
        number_of_questions: int = 10,
        difficulty: str = "medium",
        question_type: str = "mixed",
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Generate academic questions.
        """

        if not topic or not topic.strip():

            return {
                "success": False,
                "error": (
                    "Question topic cannot be empty."
                ),
            }

        number_of_questions = max(
            1,
            min(
                int(
                    number_of_questions
                ),
                100,
            ),
        )

        if self.question_generator is not None:

            result = await self._call_module(
                self.question_generator,
                (
                    "generate_questions",
                    "generate",
                    "create",
                ),
                topic=topic,
                subject=subject,
                number_of_questions=number_of_questions,
                difficulty=difficulty,
                question_type=question_type,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "questions",
                    "topic": topic,
                    "questions": result,
                }

        questions = []

        templates = [
            (
                f"Define and explain the main concept "
                f"of {topic}."
            ),
            (
                f"What are the important features "
                f"of {topic}?"
            ),
            (
                f"Explain {topic} with a suitable example."
            ),
            (
                f"Why is {topic} important?"
            ),
            (
                f"Differentiate between the major "
                f"ideas related to {topic}."
            ),
        ]

        for index in range(
            number_of_questions
        ):

            questions.append(
                {
                    "number": index + 1,
                    "question": templates[
                        index
                        % len(
                            templates
                        )
                    ],
                    "difficulty": difficulty,
                    "type": question_type,
                }
            )

        return {
            "success": True,
            "action": "questions",
            "topic": topic,
            "subject": subject,
            "questions": questions,
            "method": "fallback",
        }

    # ========================================================================
    # ANSWER CHECKING
    # ========================================================================

    async def check_answer(
        self,
        question: str,
        answer: str,
        expected_answer: Optional[str] = None,
        marks: Optional[float] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Check an answer.
        """

        if not question or not question.strip():

            return {
                "success": False,
                "error": (
                    "Question cannot be empty."
                ),
            }

        if not answer or not answer.strip():

            return {
                "success": False,
                "error": (
                    "Answer cannot be empty."
                ),
            }

        if self.answer_checker is not None:

            result = await self._call_module(
                self.answer_checker,
                (
                    "check_answer",
                    "check",
                    "evaluate",
                    "grade",
                ),
                question=question,
                answer=answer,
                expected_answer=expected_answer,
                marks=marks,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "check_answer",
                    "result": result,
                }

        if expected_answer:

            expected_tokens = set(
                self._normalize_words(
                    expected_answer
                )
            )

            answer_tokens = set(
                self._normalize_words(
                    answer
                )
            )

            if expected_tokens:

                matched = len(
                    expected_tokens.intersection(
                        answer_tokens
                    )
                )

                coverage = (
                    matched
                    / len(
                        expected_tokens
                    )
                )

            else:

                coverage = 0.0

            if coverage >= 0.8:

                verdict = "likely_correct"

            elif coverage >= 0.5:

                verdict = "partially_correct"

            else:

                verdict = "needs_improvement"

            return {
                "success": True,
                "action": "check_answer",
                "question": question,
                "answer": answer,
                "verdict": verdict,
                "coverage": round(
                    coverage,
                    3,
                ),
                "note": (
                    "Fallback keyword comparison was used. "
                    "It is not a substitute for semantic grading."
                ),
            }

        return {
            "success": True,
            "action": "check_answer",
            "question": question,
            "answer": answer,
            "verdict": "cannot_verify",
            "note": (
                "No expected answer or answer-checking "
                "engine was provided."
            ),
        }

    # ========================================================================
    # PROGRESS TRACKING
    # ========================================================================

    async def track_progress(
        self,
        subject: str,
        score: Optional[float] = None,
        total: Optional[float] = None,
        chapter: Optional[str] = None,
        status: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Track academic progress.
        """

        if not subject or not subject.strip():

            return {
                "success": False,
                "error": (
                    "Subject cannot be empty."
                ),
            }

        subject = subject.strip()

        if self.progress_tracker is not None:

            result = await self._call_module(
                self.progress_tracker,
                (
                    "track_progress",
                    "update",
                    "record",
                    "track",
                ),
                subject=subject,
                score=score,
                total=total,
                chapter=chapter,
                status=status,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "progress",
                    "result": result,
                }

        subject_data = self._progress.setdefault(
            subject,
            {
                "subject": subject,
                "attempts": [],
                "chapters": {},
            },
        )

        record = {
            "timestamp": self._timestamp(),
            "score": score,
            "total": total,
            "chapter": chapter,
            "status": status,
        }

        subject_data[
            "attempts"
        ].append(
            record
        )

        if chapter:

            subject_data[
                "chapters"
            ][chapter] = {
                "status": status,
                "latest_score": score,
                "latest_total": total,
                "updated_at": self._timestamp(),
            }

        percentage = None

        if (
            score is not None
            and total is not None
            and float(total) > 0
        ):

            percentage = (
                float(score)
                / float(total)
                * 100
            )

        return {
            "success": True,
            "action": "progress",
            "subject": subject,
            "record": record,
            "percentage": (
                round(
                    percentage,
                    2,
                )
                if percentage is not None
                else None
            ),
            "progress": subject_data,
        }

    # ========================================================================
    # SESSION
    # ========================================================================

    def create_session(
        self,
        topic: str,
        subject: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a study session.
        """

        session_id = self._make_id(
            "study"
        )

        session = {
            "session_id": session_id,
            "topic": topic,
            "subject": subject,
            "created_at": self._timestamp(),
            "messages": [],
            "completed": False,
        }

        self._study_sessions[
            session_id
        ] = session

        return session

    async def get_session(
        self,
        session_id: str,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Retrieve a study session.
        """

        session = self._study_sessions.get(
            session_id
        )

        if session is None:

            return {
                "success": False,
                "error": (
                    f"Study session not found: {session_id}"
                ),
            }

        return {
            "success": True,
            "action": "session",
            "session": session,
        }

    # ========================================================================
    # MODULE HELPERS
    # ========================================================================

    async def _call_module(
        self,
        module: Any,
        method_names: Iterable[str],
        **kwargs: Any,
    ) -> Any:
        """
        Call the first compatible method exposed by a module.
        """

        for method_name in method_names:

            method = getattr(
                module,
                method_name,
                None,
            )

            if not callable(
                method
            ):
                continue

            try:

                result = method(
                    **kwargs
                )

                if asyncio.iscoroutine(
                    result
                ):
                    result = await result

                return result

            except TypeError:

                try:

                    result = method(
                        kwargs
                    )

                    if asyncio.iscoroutine(
                        result
                    ):
                        result = await result

                    return result

                except Exception as exc:

                    self.logger.debug(
                        "Study method %s failed: %s",
                        method_name,
                        exc,
                    )

            except Exception as exc:

                self.logger.debug(
                    "Study method %s failed: %s",
                    method_name,
                    exc,
                )

        return None

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _parse_time(
        value: str,
    ) -> int:
        """
        Convert HH:MM into minutes since midnight.
        """

        match = re.match(
            r"^\s*(\d{1,2}):(\d{2})\s*$",
            str(value),
        )

        if not match:

            raise ValueError(
                f"Invalid time format: {value}"
            )

        hour = int(
            match.group(1)
        )

        minute = int(
            match.group(2)
        )

        if not (
            0 <= hour <= 23
            and 0 <= minute <= 59
        ):

            raise ValueError(
                f"Invalid time: {value}"
            )

        return (
            hour * 60
            + minute
        )

    @staticmethod
    def _format_time(
        minutes: int,
    ) -> str:

        minutes %= 24 * 60

        hour = minutes // 60
        minute = minutes % 60

        return (
            f"{hour:02d}:{minute:02d}"
        )

    @staticmethod
    def _normalize_words(
        text: str,
    ) -> List[str]:

        return re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

    @staticmethod
    def _make_id(
        prefix: str,
    ) -> str:

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%d%H%M%S%f"
            )
        )

        return (
            f"{prefix}_{timestamp}"
        )

    @staticmethod
    def _timestamp() -> str:

        return (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )


__all__ = [
    "StudyAgent",
]


