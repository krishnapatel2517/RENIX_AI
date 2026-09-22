"""
RENIX Education — Answer Checker

Checks student answers against generated/reference answers.

Supports:
- MCQ
- Multiple-select
- True/False
- Fill-in-the-blank
- Short answers
- Long answers
- Partial credit
- Keyword matching
- Numeric answers
- Case/whitespace normalization
- Optional AI semantic evaluation

This module is intentionally independent from the UI,
LLM provider, and database layers.
"""

from __future__ import annotations

import math
import re
import unicodedata

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Mapping


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_TYPES = {
    "mcq",
    "multiple_select",
    "true_false",
    "fill_blank",
    "short_answer",
    "long_answer",
}

DEFAULT_EXACT_THRESHOLD = 1.0
DEFAULT_SIMILARITY_THRESHOLD = 0.82
DEFAULT_PARTIAL_THRESHOLD = 0.55


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class AnswerCheckResult:
    """Result produced after checking one answer."""

    question_id: str = ""

    is_correct: bool = False

    is_partially_correct: bool = False

    score: float = 0.0

    max_score: float = 1.0

    percentage: float = 0.0

    student_answer: Any = None

    correct_answer: Any = None

    question_type: str = ""

    feedback: str = ""

    explanation: str = ""

    matched_keywords: list[str] = field(
        default_factory=list
    )

    missing_keywords: list[str] = field(
        default_factory=list
    )

    confidence: float = 1.0

    method: str = "rule_based"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "is_correct": self.is_correct,
            "is_partially_correct": (
                self.is_partially_correct
            ),
            "score": self.score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "student_answer": self.student_answer,
            "correct_answer": self.correct_answer,
            "question_type": self.question_type,
            "feedback": self.feedback,
            "explanation": self.explanation,
            "matched_keywords": self.matched_keywords,
            "missing_keywords": self.missing_keywords,
            "confidence": self.confidence,
            "method": self.method,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationConfig:
    """Configuration controlling answer evaluation."""

    similarity_threshold: float = (
        DEFAULT_SIMILARITY_THRESHOLD
    )

    partial_threshold: float = (
        DEFAULT_PARTIAL_THRESHOLD
    )

    case_sensitive: bool = False

    ignore_punctuation: bool = True

    normalize_unicode: bool = True

    trim_whitespace: bool = True

    allow_numeric_tolerance: bool = True

    numeric_tolerance: float = 1e-6

    allow_partial_credit: bool = True

    ai_fallback: bool = True

    keyword_weight: float = 0.7

    semantic_weight: float = 0.3

    def validate(self) -> None:

        if not 0 <= self.partial_threshold <= 1:

            raise ValueError(
                "partial_threshold must be between 0 and 1."
            )

        if not 0 <= self.similarity_threshold <= 1:

            raise ValueError(
                "similarity_threshold must be between 0 and 1."
            )

        if (
            self.partial_threshold
            > self.similarity_threshold
        ):

            raise ValueError(
                "partial_threshold cannot be greater "
                "than similarity_threshold."
            )

        if self.numeric_tolerance < 0:

            raise ValueError(
                "numeric_tolerance cannot be negative."
            )

        if self.keyword_weight < 0:

            raise ValueError(
                "keyword_weight cannot be negative."
            )

        if self.semantic_weight < 0:

            raise ValueError(
                "semantic_weight cannot be negative."
            )


# ============================================================
# ANSWER CHECKER
# ============================================================


class AnswerChecker:
    """
    Main RENIX answer-evaluation engine.

    Optional AI evaluator:

        checker = AnswerChecker(
            ai_evaluator=my_ai_function
        )

    The AI evaluator receives a structured prompt and may return:

        {
            "score": 0.85,
            "feedback": "...",
            "confidence": 0.91
        }

    Score is expected between 0 and 1.
    """

    def __init__(
        self,
        *,
        ai_evaluator: Callable[
            [str],
            Any,
        ]
        | None = None,
        config: EvaluationConfig | None = None,
    ) -> None:

        self.ai_evaluator = ai_evaluator

        self.config = (
            config
            if config is not None
            else EvaluationConfig()
        )

        self.config.validate()

    # ========================================================
    # PUBLIC API
    # ========================================================

    def check(
        self,
        *,
        question: Any,
        student_answer: Any,
        correct_answer: Any = None,
        question_type: str | None = None,
        max_score: float = 1.0,
        explanation: str = "",
        keywords: Iterable[str] | None = None,
        rubric: Mapping[str, Any] | None = None,
    ) -> AnswerCheckResult:

        question_id = self._get_field(
            question,
            "id",
            "",
        )

        if correct_answer is None:

            correct_answer = self._get_field(
                question,
                "correct_answer",
                None,
            )

        if question_type is None:

            question_type = self._get_field(
                question,
                "question_type",
                "",
            )

        if not explanation:

            explanation = self._get_field(
                question,
                "explanation",
                "",
            )

        if keywords is None:

            keywords = self._get_field(
                question,
                "keywords",
                [],
            )

        if rubric is None:

            rubric = self._get_field(
                question,
                "rubric",
                None,
            )

        question_type = self._normalize_type(
            question_type
        )

        if question_type not in SUPPORTED_TYPES:

            raise ValueError(
                "Unsupported question type: "
                + question_type
            )

        if max_score <= 0:

            raise ValueError(
                "max_score must be greater than zero."
            )

        keyword_list = [
            str(item)
            for item in (
                keywords or []
            )
            if str(item).strip()
        ]

        result = self._dispatch_check(
            question=question,
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type=question_type,
            max_score=float(max_score),
            explanation=str(
                explanation or ""
            ),
            keywords=keyword_list,
            rubric=rubric,
        )

        result.question_id = str(
            question_id or ""
        )

        return result

    # ========================================================
    # DISPATCH
    # ========================================================

    def _dispatch_check(
        self,
        *,
        question: Any,
        student_answer: Any,
        correct_answer: Any,
        question_type: str,
        max_score: float,
        explanation: str,
        keywords: list[str],
        rubric: Mapping[str, Any] | None,
    ) -> AnswerCheckResult:

        if question_type == "mcq":

            return self._check_mcq(
                student_answer,
                correct_answer,
                max_score,
                explanation,
            )

        if question_type == "multiple_select":

            return self._check_multiple_select(
                student_answer,
                correct_answer,
                max_score,
                explanation,
            )

        if question_type == "true_false":

            return self._check_true_false(
                student_answer,
                correct_answer,
                max_score,
                explanation,
            )

        if question_type == "fill_blank":

            return self._check_fill_blank(
                student_answer,
                correct_answer,
                max_score,
                explanation,
            )

        if question_type == "short_answer":

            return self._check_written_answer(
                question=question,
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type=question_type,
                max_score=max_score,
                explanation=explanation,
                keywords=keywords,
                rubric=rubric,
            )

        if question_type == "long_answer":

            return self._check_written_answer(
                question=question,
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type=question_type,
                max_score=max_score,
                explanation=explanation,
                keywords=keywords,
                rubric=rubric,
            )

        raise ValueError(
            "Unsupported question type."
        )

    # ========================================================
    # MCQ
    # ========================================================

    def _check_mcq(
        self,
        student_answer: Any,
        correct_answer: Any,
        max_score: float,
        explanation: str,
    ) -> AnswerCheckResult:

        student = self._normalize_scalar(
            student_answer
        )

        correct = self._normalize_scalar(
            correct_answer
        )

        is_correct = (
            student == correct
        )

        score = (
            max_score
            if is_correct
            else 0.0
        )

        feedback = (
            "Correct! Excellent."
            if is_correct
            else "Incorrect. Review the concept and try again."
        )

        return self._result(
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type="mcq",
            score=score,
            max_score=max_score,
            is_correct=is_correct,
            feedback=feedback,
            explanation=explanation,
            confidence=1.0,
            method="exact",
        )

    # ========================================================
    # MULTIPLE SELECT
    # ========================================================

    def _check_multiple_select(
        self,
        student_answer: Any,
        correct_answer: Any,
        max_score: float,
        explanation: str,
    ) -> AnswerCheckResult:

        student = self._as_normalized_set(
            student_answer
        )

        correct = self._as_normalized_set(
            correct_answer
        )

        if not correct:

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type="multiple_select",
                score=0.0,
                max_score=max_score,
                feedback="No valid correct answers were supplied.",
                explanation=explanation,
                confidence=1.0,
                method="validation",
            )

        if student == correct:

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type="multiple_select",
                score=max_score,
                max_score=max_score,
                is_correct=True,
                feedback="Correct! All required options were selected.",
                explanation=explanation,
                confidence=1.0,
                method="exact_set",
            )

        matched = student & correct

        missing = correct - student

        incorrect = student - correct

        ratio = len(matched) / len(correct)

        if incorrect:

            ratio *= max(
                0.0,
                1.0
                - (
                    len(incorrect)
                    / max(
                        len(correct),
                        1,
                    )
                ),
            )

        score = (
            max_score * ratio
            if self.config.allow_partial_credit
            else 0.0
        )

        is_partial = (
            0 < score < max_score
        )

        feedback = self._multiple_select_feedback(
            matched=matched,
            missing=missing,
            incorrect=incorrect,
        )

        return self._result(
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type="multiple_select",
            score=score,
            max_score=max_score,
            is_partially_correct=is_partial,
            feedback=feedback,
            explanation=explanation,
            matched_keywords=list(matched),
            missing_keywords=list(missing),
            confidence=1.0,
            method="set_comparison",
        )

    # ========================================================
    # TRUE / FALSE
    # ========================================================

    def _check_true_false(
        self,
        student_answer: Any,
        correct_answer: Any,
        max_score: float,
        explanation: str,
    ) -> AnswerCheckResult:

        student = self._parse_boolean(
            student_answer
        )

        correct = self._parse_boolean(
            correct_answer
        )

        if student is None:

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type="true_false",
                score=0.0,
                max_score=max_score,
                feedback=(
                    "Please answer with True or False."
                ),
                explanation=explanation,
                confidence=1.0,
                method="validation",
            )

        is_correct = (
            student == correct
        )

        return self._result(
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type="true_false",
            score=(
                max_score
                if is_correct
                else 0.0
            ),
            max_score=max_score,
            is_correct=is_correct,
            feedback=(
                "Correct!"
                if is_correct
                else "Incorrect. Review the statement carefully."
            ),
            explanation=explanation,
            confidence=1.0,
            method="boolean",
        )

    # ========================================================
    # FILL IN THE BLANK
    # ========================================================

    def _check_fill_blank(
        self,
        student_answer: Any,
        correct_answer: Any,
        max_score: float,
        explanation: str,
    ) -> AnswerCheckResult:

        student = self._normalize_scalar(
            student_answer
        )

        correct_values = (
            self._as_normalized_values(
                correct_answer
            )
        )

        if self._numeric_match(
            student,
            correct_values,
        ):

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type="fill_blank",
                score=max_score,
                max_score=max_score,
                is_correct=True,
                feedback="Correct!",
                explanation=explanation,
                confidence=1.0,
                method="numeric_or_exact",
            )

        for correct in correct_values:

            if student == correct:

                return self._result(
                    student_answer=student_answer,
                    correct_answer=correct_answer,
                    question_type="fill_blank",
                    score=max_score,
                    max_score=max_score,
                    is_correct=True,
                    feedback="Correct!",
                    explanation=explanation,
                    confidence=1.0,
                    method="exact",
                )

        best_similarity = 0.0

        for correct in correct_values:

            similarity = self._similarity(
                student,
                correct,
            )

            best_similarity = max(
                best_similarity,
                similarity,
            )

        if (
            self.config.allow_partial_credit
            and best_similarity
            >= self.config.similarity_threshold
        ):

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type="fill_blank",
                score=max_score,
                max_score=max_score,
                is_correct=True,
                feedback=(
                    "Your answer matches the expected "
                    "answer closely."
                ),
                explanation=explanation,
                confidence=best_similarity,
                method="similarity",
            )

        return self._result(
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type="fill_blank",
            score=0.0,
            max_score=max_score,
            feedback=(
                "Not quite. Check the expected term or value."
            ),
            explanation=explanation,
            confidence=best_similarity,
            method="exact_or_similarity",
        )

    # ========================================================
    # SHORT/LONG ANSWERS
    # ========================================================

    def _check_written_answer(
        self,
        *,
        question: Any,
        student_answer: Any,
        correct_answer: Any,
        question_type: str,
        max_score: float,
        explanation: str,
        keywords: list[str],
        rubric: Mapping[str, Any] | None,
    ) -> AnswerCheckResult:

        student_text = self._normalize_text(
            student_answer
        )

        correct_text = self._normalize_text(
            correct_answer
        )

        if not student_text:

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type=question_type,
                score=0.0,
                max_score=max_score,
                feedback="No answer was provided.",
                explanation=explanation,
                confidence=1.0,
                method="empty_answer",
            )

        if not correct_text:

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type=question_type,
                score=0.0,
                max_score=max_score,
                feedback=(
                    "No reference answer is available "
                    "for automatic evaluation."
                ),
                explanation=explanation,
                confidence=0.0,
                method="missing_reference",
            )

        if self._exact_match(
            student_text,
            correct_text,
        ):

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type=question_type,
                score=max_score,
                max_score=max_score,
                is_correct=True,
                feedback="Correct! Your answer matches the reference.",
                explanation=explanation,
                confidence=1.0,
                method="exact",
            )

        matched_keywords = []
        missing_keywords = []

        if keywords:

            (
                matched_keywords,
                missing_keywords,
            ) = self._keyword_analysis(
                student_text,
                keywords,
            )

            keyword_score = (
                len(matched_keywords)
                / len(keywords)
            )

        else:

            keyword_score = 0.0

        similarity = self._similarity(
            student_text,
            correct_text,
        )

        combined_score = self._combined_written_score(
            similarity=similarity,
            keyword_score=keyword_score,
            has_keywords=bool(keywords),
        )

        if (
            self.ai_evaluator is not None
            and self.config.ai_fallback
        ):

            ai_result = (
                self._try_ai_evaluation(
                    question=question,
                    student_answer=student_answer,
                    correct_answer=correct_answer,
                    question_type=question_type,
                    max_score=max_score,
                    rubric=rubric,
                )
            )

            if ai_result is not None:

                return self._result(
                    student_answer=student_answer,
                    correct_answer=correct_answer,
                    question_type=question_type,
                    score=(
                        ai_result["score"]
                        * max_score
                    ),
                    max_score=max_score,
                    is_correct=(
                        ai_result["score"]
                        >= 0.9
                    ),
                    is_partially_correct=(
                        0.0
                        < ai_result["score"]
                        < 0.9
                    ),
                    feedback=ai_result[
                        "feedback"
                    ],
                    explanation=explanation,
                    matched_keywords=(
                        matched_keywords
                    ),
                    missing_keywords=(
                        missing_keywords
                    ),
                    confidence=ai_result[
                        "confidence"
                    ],
                    method="ai",
                )

        if (
            combined_score
            >= self.config.similarity_threshold
        ):

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type=question_type,
                score=max_score,
                max_score=max_score,
                is_correct=True,
                feedback=(
                    "Correct. Your answer demonstrates "
                    "a strong match with the expected concept."
                ),
                explanation=explanation,
                matched_keywords=matched_keywords,
                missing_keywords=missing_keywords,
                confidence=combined_score,
                method="semantic_similarity",
            )

        if (
            self.config.allow_partial_credit
            and combined_score
            >= self.config.partial_threshold
        ):

            score = (
                max_score
                * combined_score
            )

            return self._result(
                student_answer=student_answer,
                correct_answer=correct_answer,
                question_type=question_type,
                score=score,
                max_score=max_score,
                is_partially_correct=True,
                feedback=self._partial_feedback(
                    missing_keywords
                ),
                explanation=explanation,
                matched_keywords=matched_keywords,
                missing_keywords=missing_keywords,
                confidence=combined_score,
                method="partial_similarity",
            )

        return self._result(
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type=question_type,
            score=0.0,
            max_score=max_score,
            feedback=self._incorrect_written_feedback(
                missing_keywords
            ),
            explanation=explanation,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            confidence=combined_score,
            method="keyword_similarity",
        )

    # ========================================================
    # KEYWORD ANALYSIS
    # ========================================================

    def _keyword_analysis(
        self,
        student_text: str,
        keywords: Iterable[str],
    ) -> tuple[
        list[str],
        list[str],
    ]:

        matched = []
        missing = []

        for keyword in keywords:

            normalized_keyword = (
                self._normalize_text(
                    keyword
                )
            )

            if not normalized_keyword:

                continue

            if self._contains_concept(
                student_text,
                normalized_keyword,
            ):

                matched.append(
                    str(keyword)
                )

            else:

                missing.append(
                    str(keyword)
                )

        return matched, missing

    def _contains_concept(
        self,
        text: str,
        keyword: str,
    ) -> bool:

        if keyword in text:

            return True

        words = keyword.split()

        if len(words) == 1:

            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            return bool(
                re.search(
                    pattern,
                    text,
                )
            )

        return self._similarity(
            text,
            keyword,
        ) >= 0.75

    # ========================================================
    # AI EVALUATION
    # ========================================================

    def _try_ai_evaluation(
        self,
        *,
        question: Any,
        student_answer: Any,
        correct_answer: Any,
        question_type: str,
        max_score: float,
        rubric: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:

        prompt = self._build_ai_prompt(
            question=question,
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type=question_type,
            max_score=max_score,
            rubric=rubric,
        )

        try:

            response = self.ai_evaluator(
                prompt
            )

            return self._parse_ai_result(
                response
            )

        except Exception:

            return None

    def _build_ai_prompt(
        self,
        *,
        question: Any,
        student_answer: Any,
        correct_answer: Any,
        question_type: str,
        max_score: float,
        rubric: Mapping[str, Any] | None,
    ) -> str:

        question_text = self._get_field(
            question,
            "question",
            "",
        )

        rubric_text = (
            str(dict(rubric))
            if rubric
            else "No special rubric."
        )

        return f"""
You are RENIX, an educational answer evaluator.

Evaluate the student's answer fairly.

QUESTION:
{question_text}

QUESTION TYPE:
{question_type}

MAXIMUM SCORE:
{max_score}

REFERENCE ANSWER:
{correct_answer}

STUDENT ANSWER:
{student_answer}

RUBRIC:
{rubric_text}

Return ONLY JSON:

{{
    "score": 0.0,
    "confidence": 0.0,
    "feedback": "..."
}}

Rules:
- score must be between 0 and 1.
- 1 means fully correct.
- 0 means completely incorrect.
- Give partial credit when appropriate.
- Do not require identical wording.
- Judge conceptual correctness.
- Do not reward irrelevant content.
- Feedback should explain how the student can improve.
""".strip()

    def _parse_ai_result(
        self,
        response: Any,
    ) -> dict[str, Any]:

        if isinstance(
            response,
            Mapping,
        ):

            data = dict(response)

        else:

            import json

            text = str(
                response
            ).strip()

            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"\s*```$",
                "",
                text,
            )

            data = json.loads(
                text
            )

        score = self._safe_float(
            data.get(
                "score",
                0,
            ),
            0.0,
        )

        confidence = self._safe_float(
            data.get(
                "confidence",
                score,
            ),
            score,
        )

        return {
            "score": max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            ),
            "confidence": max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            "feedback": str(
                data.get(
                    "feedback",
                    "Answer evaluated.",
                )
            ),
        }

    # ========================================================
    # SCORING
    # ========================================================

    def _combined_written_score(
        self,
        *,
        similarity: float,
        keyword_score: float,
        has_keywords: bool,
    ) -> float:

        if not has_keywords:

            return similarity

        score = (
            similarity
            * self.config.semantic_weight
            + keyword_score
            * self.config.keyword_weight
        )

        total_weight = (
            self.config.semantic_weight
            + self.config.keyword_weight
        )

        if total_weight <= 0:

            return similarity

        return max(
            0.0,
            min(
                1.0,
                score / total_weight,
            ),
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_text(
        self,
        value: Any,
    ) -> str:

        if value is None:

            return ""

        text = str(
            value
        )

        if self.config.normalize_unicode:

            text = unicodedata.normalize(
                "NFKC",
                text,
            )

        if not self.config.case_sensitive:

            text = text.casefold()

        if self.config.ignore_punctuation:

            text = re.sub(
                r"[^\w\s.]",
                " ",
                text,
            )

        if self.config.trim_whitespace:

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

        return text

    def _normalize_scalar(
        self,
        value: Any,
    ) -> str:

        return self._normalize_text(
            value
        )

    def _as_normalized_values(
        self,
        value: Any,
    ) -> list[str]:

        if isinstance(
            value,
            (list, tuple, set),
        ):

            return [
                self._normalize_scalar(
                    item
                )
                for item in value
            ]

        return [
            self._normalize_scalar(
                value
            )
        ]

    def _as_normalized_set(
        self,
        value: Any,
    ) -> set[str]:

        return {
            item
            for item in self._as_normalized_values(
                value
            )
            if item
        }

    # ========================================================
    # BOOLEAN
    # ========================================================

    @staticmethod
    def _parse_boolean(
        value: Any,
    ) -> bool | None:

        if isinstance(
            value,
            bool,
        ):

            return value

        if isinstance(
            value,
            (int, float),
        ):

            if value == 1:

                return True

            if value == 0:

                return False

        if not isinstance(
            value,
            str,
        ):

            return None

        normalized = (
            value.strip()
            .casefold()
        )

        if normalized in {
            "true",
            "t",
            "yes",
            "y",
            "1",
        }:

            return True

        if normalized in {
            "false",
            "f",
            "no",
            "n",
            "0",
        }:

            return False

        return None

    # ========================================================
    # NUMERIC COMPARISON
    # ========================================================

    def _numeric_match(
        self,
        student: str,
        correct_values: Sequence[str],
    ) -> bool:

        if not self.config.allow_numeric_tolerance:

            return False

        student_number = (
            self._parse_number(
                student
            )
        )

        if student_number is None:

            return False

        for correct in correct_values:

            correct_number = (
                self._parse_number(
                    correct
                )
            )

            if correct_number is None:

                continue

            if math.isclose(
                student_number,
                correct_number,
                rel_tol=(
                    self.config.numeric_tolerance
                ),
                abs_tol=(
                    self.config.numeric_tolerance
                ),
            ):

                return True

        return False

    @staticmethod
    def _parse_number(
        value: str,
    ) -> float | None:

        if not value:

            return None

        cleaned = (
            value
            .replace(",", "")
            .replace("%", "")
            .strip()
        )

        try:

            return float(
                cleaned
            )

        except ValueError:

            return None

    # ========================================================
    # SIMILARITY
    # ========================================================

    @staticmethod
    def _similarity(
        first: str,
        second: str,
    ) -> float:

        if not first or not second:

            return 0.0

        return SequenceMatcher(
            None,
            first,
            second,
        ).ratio()

    @staticmethod
    def _exact_match(
        first: str,
        second: str,
    ) -> bool:

        return (
            first.strip()
            == second.strip()
        )

    # ========================================================
    # FEEDBACK
    # ========================================================

    @staticmethod
    def _multiple_select_feedback(
        *,
        matched: set[str],
        missing: set[str],
        incorrect: set[str],
    ) -> str:

        parts = []

        if matched:

            parts.append(
                f"{len(matched)} correct option(s) selected."
            )

        if missing:

            parts.append(
                f"{len(missing)} required option(s) were missed."
            )

        if incorrect:

            parts.append(
                f"{len(incorrect)} incorrect option(s) were selected."
            )

        return " ".join(
            parts
        ) or "Review the available options."

    @staticmethod
    def _partial_feedback(
        missing_keywords: Sequence[str],
    ) -> str:

        if missing_keywords:

            return (
                "Partially correct. "
                "Your answer contains the main idea, "
                "but it is missing: "
                + ", ".join(
                    missing_keywords
                )
                + "."
            )

        return (
            "Partially correct. "
            "Add more precise details to improve the answer."
        )

    @staticmethod
    def _incorrect_written_feedback(
        missing_keywords: Sequence[str],
    ) -> str:

        if missing_keywords:

            return (
                "The answer needs improvement. "
                "Important concepts missing: "
                + ", ".join(
                    missing_keywords
                )
                + "."
            )

        return (
            "The answer does not match the expected "
            "concept closely enough. Review the topic "
            "and try again."
        )

    # ========================================================
    # RESULT CREATION
    # ========================================================

    @staticmethod
    def _result(
        *,
        student_answer: Any,
        correct_answer: Any,
        question_type: str,
        score: float,
        max_score: float,
        is_correct: bool = False,
        is_partially_correct: bool = False,
        feedback: str = "",
        explanation: str = "",
        matched_keywords: list[str] | None = None,
        missing_keywords: list[str] | None = None,
        confidence: float = 1.0,
        method: str = "rule_based",
    ) -> AnswerCheckResult:

        score = max(
            0.0,
            min(
                max_score,
                score,
            ),
        )

        percentage = (
            score / max_score * 100
            if max_score > 0
            else 0.0
        )

        if score >= max_score:

            is_correct = True

            is_partially_correct = False

        elif score > 0:

            is_correct = False

            is_partially_correct = True

        return AnswerCheckResult(
            is_correct=is_correct,
            is_partially_correct=(
                is_partially_correct
            ),
            score=score,
            max_score=max_score,
            percentage=percentage,
            student_answer=student_answer,
            correct_answer=correct_answer,
            question_type=question_type,
            feedback=feedback,
            explanation=explanation,
            matched_keywords=(
                matched_keywords or []
            ),
            missing_keywords=(
                missing_keywords or []
            ),
            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            method=method,
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _normalize_type(
        question_type: Any,
    ) -> str:

        value = str(
            question_type or ""
        ).strip().lower()

        aliases = {
            "multiple choice": "mcq",
            "multiple-choice": "mcq",
            "multiple_choice": "mcq",
            "multi_select": "multiple_select",
            "multi-select": "multiple_select",
            "true/false": "true_false",
            "true-false": "true_false",
            "fill-in-the-blank": "fill_blank",
            "fill in the blank": "fill_blank",
            "short": "short_answer",
            "long": "long_answer",
        }

        return aliases.get(
            value,
            value,
        )

    @staticmethod
    def _get_field(
        obj: Any,
        name: str,
        default: Any = None,
    ) -> Any:

        if obj is None:

            return default

        if isinstance(
            obj,
            Mapping,
        ):

            return obj.get(
                name,
                default,
            )

        return getattr(
            obj,
            name,
            default,
        )

    @staticmethod
    def _safe_float(
        value: Any,
        fallback: float,
    ) -> float:

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return float(
                fallback
            )


# ============================================================
# BATCH CHECKING
# ============================================================


def check_answers(
    checker: AnswerChecker,
    questions: Iterable[Any],
    answers: Mapping[str, Any],
    *,
    default_max_score: float = 1.0,
) -> list[AnswerCheckResult]:

    """
    Check multiple questions at once.

    `answers` should map question IDs to
    student answers.
    """

    results = []

    for question in questions:

        question_id = str(
            checker._get_field(
                question,
                "id",
                "",
            )
        )

        student_answer = answers.get(
            question_id
        )

        result = checker.check(
            question=question,
            student_answer=student_answer,
            max_score=(
                checker._get_field(
                    question,
                    "marks",
                    default_max_score,
                )
            ),
        )

        results.append(
            result
        )

    return results


# ============================================================
# SCORE SUMMARY
# ============================================================


def calculate_score_summary(
    results: Iterable[
        AnswerCheckResult
    ],
) -> dict[str, Any]:

    result_list = list(
        results
    )

    total_score = sum(
        result.score
        for result in result_list
    )

    max_score = sum(
        result.max_score
        for result in result_list
    )

    correct_count = sum(
        1
        for result in result_list
        if result.is_correct
    )

    partial_count = sum(
        1
        for result in result_list
        if result.is_partially_correct
    )

    incorrect_count = sum(
        1
        for result in result_list
        if (
            not result.is_correct
            and not result.is_partially_correct
        )
    )

    percentage = (
        total_score / max_score * 100
        if max_score > 0
        else 0.0
    )

    return {
        "total_questions": len(
            result_list
        ),
        "correct": correct_count,
        "partially_correct": partial_count,
        "incorrect": incorrect_count,
        "total_score": total_score,
        "max_score": max_score,
        "percentage": percentage,
    }


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "SUPPORTED_TYPES",
    "EvaluationConfig",
    "AnswerCheckResult",
    "AnswerChecker",
    "check_answers",
    "calculate_score_summary",
]


