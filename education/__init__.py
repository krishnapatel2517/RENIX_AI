"""
RENIX Education System
======================

The education package provides RENIX with an integrated
learning and academic-management system.

Main capabilities include:

    - Study planning
    - Timetable management
    - Homework management
    - Revision planning
    - Quiz generation
    - Flashcards
    - Examination preparation
    - Question generation
    - Answer checking
    - Academic progress tracking
    - Subject-specific learning systems

The package is designed to be accessed through the
RENIX education manager and individual educational
components.
"""

from .study_manager import StudyManager
from .timetable import TimetableManager
from .homework import HomeworkManager
from .revision import RevisionManager
from .quiz_engine import QuizEngine
from .flashcards import FlashcardManager
from .exam_engine import ExamEngine
from .question_generator import QuestionGenerator
from .answer_checker import AnswerChecker
from .progress_tracker import ProgressTracker

__all__ = [
    "StudyManager",
    "TimetableManager",
    "HomeworkManager",
    "RevisionManager",
    "QuizEngine",
    "FlashcardManager",
    "ExamEngine",
    "QuestionGenerator",
    "AnswerChecker",
    "ProgressTracker",
]


