"""
RENIX Education — Question Generator

Generates structured educational questions for:
- Practice
- Quizzes
- Exams
- Revision
- Homework
- Flashcards
- Topic-based learning

The generator is provider-agnostic. An AI/LLM provider can be
injected through `generator`, but RENIX also has a deterministic
local fallback so the module remains usable without an AI API.

Supported question types:
    mcq
    multiple_select
    true_false
    short_answer
    long_answer
    fill_blank
"""

from __future__ import annotations

import json
import logging
import random
import re
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
    "multiple_select",
    "true_false",
    "short_answer",
    "long_answer",
    "fill_blank",
}

DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
    "mixed",
}

QUESTION_COUNTS = {
    "practice",
    "quiz",
    "exam",
    "revision",
    "homework",
}


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class GeneratedQuestion:
    """Represents one generated educational question."""

    id: str

    question: str

    question_type: str

    correct_answer: Any = None

    options: list[str] = field(
        default_factory=list
    )

    explanation: str = ""

    subject: str = ""

    chapter: str = ""

    topic: str = ""

    difficulty: str = "medium"

    marks: float = 1.0

    negative_marks: float = 0.0

    hints: list[str] = field(
        default_factory=list
    )

    tags: list[str] = field(
        default_factory=list
    )

    source: str = "local"

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRequest:
    """Parameters used to generate questions."""

    subject: str

    topic: str

    count: int = 10

    question_type: str = "mcq"

    difficulty: str = "medium"

    exam_type: str = "practice"

    chapter: str = ""

    marks: float = 1.0

    negative_marks: float = 0.0

    include_explanations: bool = True

    include_hints: bool = True

    avoid_duplicates: bool = True

    language: str = "English"

    syllabus_context: str = ""

    extra_instructions: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# QUESTION GENERATOR
# ============================================================


class QuestionGenerator:
    """
    RENIX educational question generation engine.

    AI integration:

        generator = QuestionGenerator(
            ai_generator=my_llm_function
        )

    The callable should accept a prompt and return either:

        - JSON string
        - dict
        - list of dicts
        - GeneratedQuestion
        - list[GeneratedQuestion]

    Example:

        engine = QuestionGenerator()

        questions = engine.generate(
            subject="Science",
            topic="Light",
            count=10,
            difficulty="medium",
            question_type="mcq",
        )
    """

    def __init__(
        self,
        *,
        ai_generator: Any | None = None,
        answer_checker: Any | None = None,
        memory: Any | None = None,
        seed: int | None = None,
    ) -> None:

        self.ai_generator = ai_generator

        self.answer_checker = answer_checker

        self.memory = memory

        self.random = random.Random(seed)

        self._history: list[
            GeneratedQuestion
        ] = []

        logger.info(
            "RENIX QuestionGenerator initialized."
        )

    # ========================================================
    # MAIN GENERATION API
    # ========================================================

    def generate(
        self,
        *,
        subject: str,
        topic: str,
        count: int = 10,
        question_type: str = "mcq",
        difficulty: str = "medium",
        exam_type: str = "practice",
        chapter: str = "",
        marks: float = 1.0,
        negative_marks: float = 0.0,
        include_explanations: bool = True,
        include_hints: bool = True,
        avoid_duplicates: bool = True,
        language: str = "English",
        syllabus_context: str = "",
        extra_instructions: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[GeneratedQuestion]:

        request = self._build_request(
            subject=subject,
            topic=topic,
            count=count,
            question_type=question_type,
            difficulty=difficulty,
            exam_type=exam_type,
            chapter=chapter,
            marks=marks,
            negative_marks=negative_marks,
            include_explanations=(
                include_explanations
            ),
            include_hints=include_hints,
            avoid_duplicates=avoid_duplicates,
            language=language,
            syllabus_context=syllabus_context,
            extra_instructions=(
                extra_instructions
            ),
            metadata=metadata,
        )

        logger.info(
            "Generating %s questions: %s / %s",
            request.count,
            request.subject,
            request.topic,
        )

        questions: list[
            GeneratedQuestion
        ] = []

        if self.ai_generator is not None:

            try:

                questions = (
                    self._generate_with_ai(
                        request
                    )
                )

            except Exception:

                logger.exception(
                    "AI question generation failed. "
                    "Falling back to local generation."
                )

        if not questions:

            questions = (
                self._generate_locally(
                    request
                )
            )

        questions = self._normalize_questions(
            questions,
            request,
        )

        if request.avoid_duplicates:

            questions = (
                self._remove_duplicates(
                    questions
                )
            )

        if len(questions) < request.count:

            questions.extend(
                self._fill_missing(
                    request,
                    existing=questions,
                    required=(
                        request.count
                        - len(questions)
                    ),
                )
            )

        questions = questions[
            : request.count
        ]

        self._history.extend(
            questions
        )

        self._store_memory(
            request,
            questions,
        )

        return questions

    # ========================================================
    # SPECIALIZED GENERATORS
    # ========================================================

    def generate_quiz(
        self,
        *,
        subject: str,
        topic: str,
        count: int = 10,
        difficulty: str = "medium",
        **kwargs: Any,
    ) -> list[GeneratedQuestion]:

        return self.generate(
            subject=subject,
            topic=topic,
            count=count,
            difficulty=difficulty,
            exam_type="quiz",
            **kwargs,
        )

    def generate_exam(
        self,
        *,
        subject: str,
        topic: str,
        count: int = 20,
        difficulty: str = "mixed",
        **kwargs: Any,
    ) -> list[GeneratedQuestion]:

        return self.generate(
            subject=subject,
            topic=topic,
            count=count,
            difficulty=difficulty,
            exam_type="exam",
            **kwargs,
        )

    def generate_practice(
        self,
        *,
        subject: str,
        topic: str,
        count: int = 10,
        difficulty: str = "medium",
        **kwargs: Any,
    ) -> list[GeneratedQuestion]:

        return self.generate(
            subject=subject,
            topic=topic,
            count=count,
            difficulty=difficulty,
            exam_type="practice",
            **kwargs,
        )

    def generate_revision(
        self,
        *,
        subject: str,
        topic: str,
        count: int = 10,
        difficulty: str = "easy",
        **kwargs: Any,
    ) -> list[GeneratedQuestion]:

        return self.generate(
            subject=subject,
            topic=topic,
            count=count,
            difficulty=difficulty,
            exam_type="revision",
            **kwargs,
        )

    def generate_homework(
        self,
        *,
        subject: str,
        topic: str,
        count: int = 10,
        difficulty: str = "medium",
        **kwargs: Any,
    ) -> list[GeneratedQuestion]:

        return self.generate(
            subject=subject,
            topic=topic,
            count=count,
            difficulty=difficulty,
            exam_type="homework",
            **kwargs,
        )

    # ========================================================
    # AI GENERATION
    # ========================================================

    def _generate_with_ai(
        self,
        request: GenerationRequest,
    ) -> list[GeneratedQuestion]:

        prompt = self._build_ai_prompt(
            request
        )

        response = self._call_ai(
            prompt
        )

        raw_questions = (
            self._parse_ai_response(
                response
            )
        )

        questions = []

        for raw in raw_questions:

            question = (
                self._question_from_dict(
                    raw,
                    request,
                    source="ai",
                )
            )

            if question is not None:

                questions.append(
                    question
                )

        return questions

    def _call_ai(
        self,
        prompt: str,
    ) -> Any:

        generator = self.ai_generator

        if generator is None:

            raise RuntimeError(
                "No AI generator configured."
            )

        if callable(generator):

            return generator(prompt)

        generate = getattr(
            generator,
            "generate",
            None,
        )

        if callable(generate):

            return generate(prompt)

        complete = getattr(
            generator,
            "complete",
            None,
        )

        if callable(complete):

            return complete(prompt)

        raise TypeError(
            "ai_generator must be callable "
            "or provide generate()/complete()."
        )

    def _build_ai_prompt(
        self,
        request: GenerationRequest,
    ) -> str:

        question_type = (
            request.question_type
        )

        return f"""
You are RENIX, an expert educational
question-generation engine.

Generate exactly {request.count}
high-quality questions.

SUBJECT:
{request.subject}

CHAPTER:
{request.chapter or "Not specified"}

TOPIC:
{request.topic}

QUESTION TYPE:
{question_type}

DIFFICULTY:
{request.difficulty}

EXAM TYPE:
{request.exam_type}

LANGUAGE:
{request.language}

MARKS:
{request.marks}

NEGATIVE MARKS:
{request.negative_marks}

SYLLABUS CONTEXT:
{request.syllabus_context}

ADDITIONAL INSTRUCTIONS:
{request.extra_instructions}

Return ONLY valid JSON.

Required structure:

{{
  "questions": [
    {{
      "question": "...",
      "question_type": "{question_type}",
      "options": [],
      "correct_answer": "...",
      "explanation": "...",
      "hints": [],
      "subject": "{request.subject}",
      "chapter": "{request.chapter}",
      "topic": "{request.topic}",
      "difficulty": "{request.difficulty}",
      "marks": {request.marks},
      "negative_marks": {request.negative_marks},
      "tags": []
    }}
  ]
}}

Rules:
1. Questions must be educationally meaningful.
2. Do not invent obviously false facts.
3. MCQs must contain plausible distractors.
4. There must be exactly one correct MCQ answer.
5. Avoid duplicate questions.
6. Keep the difficulty appropriate.
7. Explanations must teach the concept.
8. Do not include markdown outside the JSON.
""".strip()

    def _parse_ai_response(
        self,
        response: Any,
    ) -> list[dict[str, Any]]:

        if isinstance(
            response,
            GeneratedQuestion,
        ):

            return [
                response.to_dict()
            ]

        if isinstance(
            response,
            list,
        ):

            return [
                item
                for item in response
                if isinstance(
                    item,
                    dict,
                )
            ]

        if isinstance(
            response,
            dict,
        ):

            questions = response.get(
                "questions"
            )

            if isinstance(
                questions,
                list,
            ):

                return [
                    item
                    for item in questions
                    if isinstance(
                        item,
                        dict,
                    )
                ]

            return [response]

        if not isinstance(
            response,
            str,
        ):

            raise TypeError(
                "Unsupported AI response type."
            )

        text = response.strip()

        text = self._strip_code_fences(
            text
        )

        try:

            parsed = json.loads(
                text
            )

        except json.JSONDecodeError:

            extracted = (
                self._extract_json(
                    text
                )
            )

            if extracted is None:

                raise ValueError(
                    "AI returned invalid JSON."
                )

            parsed = json.loads(
                extracted
            )

        return self._parse_ai_response(
            parsed
        )

    # ========================================================
    # LOCAL GENERATION
    # ========================================================

    def _generate_locally(
        self,
        request: GenerationRequest,
    ) -> list[GeneratedQuestion]:

        questions = []

        for index in range(
            request.count
        ):

            question = (
                self._generate_local_question(
                    request,
                    index,
                )
            )

            questions.append(
                question
            )

        return questions

    def _generate_local_question(
        self,
        request: GenerationRequest,
        index: int,
    ) -> GeneratedQuestion:

        question_type = (
            request.question_type
        )

        if question_type == "mcq":

            return self._local_mcq(
                request,
                index,
            )

        if question_type == "multiple_select":

            return self._local_multiple_select(
                request,
                index,
            )

        if question_type == "true_false":

            return self._local_true_false(
                request,
                index,
            )

        if question_type == "short_answer":

            return self._local_short_answer(
                request,
                index,
            )

        if question_type == "long_answer":

            return self._local_long_answer(
                request,
                index,
            )

        if question_type == "fill_blank":

            return self._local_fill_blank(
                request,
                index,
            )

        raise ValueError(
            f"Unsupported question type: "
            f"{question_type}"
        )

    # ========================================================
    # LOCAL MCQ
    # ========================================================

    def _local_mcq(
        self,
        request: GenerationRequest,
        index: int,
    ) -> GeneratedQuestion:

        topic = request.topic

        templates = [
            (
                f"Which statement best describes "
                f"{topic}?"
            ),
            (
                f"Which of the following is most "
                f"closely related to {topic}?"
            ),
            (
                f"What is an important feature "
                f"of {topic}?"
            ),
        ]

        question_text = self.random.choice(
            templates
        )

        correct = (
            f"{topic} is an important concept "
            f"in {request.subject}."
        )

        options = [
            correct,
            f"{topic} has no connection "
            f"with {request.subject}.",
            f"{topic} is only used outside "
            f"the field of {request.subject}.",
            f"{topic} is unrelated to the "
            f"concept being studied.",
        ]

        self.random.shuffle(
            options
        )

        return GeneratedQuestion(
            id=self._new_id(),
            question=question_text,
            question_type="mcq",
            correct_answer=correct,
            options=options,
            explanation=(
                f"The correct answer is the "
                f"statement describing the role "
                f"of {topic} in {request.subject}."
            ),
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            difficulty=self._resolve_difficulty(
                request.difficulty,
                index,
            ),
            marks=request.marks,
            negative_marks=(
                request.negative_marks
            ),
            hints=(
                [f"Think about {topic}."]
                if request.include_hints
                else []
            ),
            tags=[
                request.subject,
                request.topic,
            ],
            source="local",
        )

    # ========================================================
    # MULTIPLE SELECT
    # ========================================================

    def _local_multiple_select(
        self,
        request: GenerationRequest,
        index: int,
    ) -> GeneratedQuestion:

        correct_a = (
            f"{request.topic} is studied "
            f"within {request.subject}."
        )

        correct_b = (
            f"{request.topic} is an "
            f"educational concept."
        )

        options = [
            correct_a,
            correct_b,
            (
                f"{request.topic} has absolutely "
                f"no academic relevance."
            ),
            (
                f"{request.topic} cannot be "
                f"studied or explained."
            ),
        ]

        return GeneratedQuestion(
            id=self._new_id(),
            question=(
                f"Which of the following statements "
                f"about {request.topic} are correct?"
            ),
            question_type="multiple_select",
            correct_answer=[
                correct_a,
                correct_b,
            ],
            options=options,
            explanation=(
                f"The first two statements "
                f"describe {request.topic} correctly."
            ),
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            difficulty=self._resolve_difficulty(
                request.difficulty,
                index,
            ),
            marks=request.marks,
            negative_marks=(
                request.negative_marks
            ),
            hints=(
                ["Select all statements that "
                 "are supported by the concept."]
                if request.include_hints
                else []
            ),
            tags=[
                request.subject,
                request.topic,
            ],
            source="local",
        )

    # ========================================================
    # TRUE / FALSE
    # ========================================================

    def _local_true_false(
        self,
        request: GenerationRequest,
        index: int,
    ) -> GeneratedQuestion:

        correct = True

        return GeneratedQuestion(
            id=self._new_id(),
            question=(
                f"{request.topic} is a concept "
                f"that can be studied in "
                f"{request.subject}."
            ),
            question_type="true_false",
            correct_answer=correct,
            options=[
                "True",
                "False",
            ],
            explanation=(
                f"{request.topic} can be studied "
                f"as part of {request.subject}."
            ),
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            difficulty=self._resolve_difficulty(
                request.difficulty,
                index,
            ),
            marks=request.marks,
            negative_marks=(
                request.negative_marks
            ),
            hints=(
                ["Recall the definition of "
                 f"{request.topic}."]
                if request.include_hints
                else []
            ),
            tags=[
                request.subject,
                request.topic,
            ],
            source="local",
        )

    # ========================================================
    # SHORT ANSWER
    # ========================================================

    def _local_short_answer(
        self,
        request: GenerationRequest,
        index: int,
    ) -> GeneratedQuestion:

        return GeneratedQuestion(
            id=self._new_id(),
            question=(
                f"Explain {request.topic} "
                f"in your own words."
            ),
            question_type="short_answer",
            correct_answer=(
                f"A correct answer should "
                f"accurately explain {request.topic} "
                f"and its role in {request.subject}."
            ),
            explanation=(
                f"A strong answer should define "
                f"{request.topic}, explain its key "
                f"idea, and give relevant details."
            ),
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            difficulty=self._resolve_difficulty(
                request.difficulty,
                index,
            ),
            marks=request.marks,
            negative_marks=(
                request.negative_marks
            ),
            hints=(
                [
                    f"Start by defining "
                    f"{request.topic}."
                ]
                if request.include_hints
                else []
            ),
            tags=[
                request.subject,
                request.topic,
            ],
            source="local",
        )

    # ========================================================
    # LONG ANSWER
    # ========================================================

    def _local_long_answer(
        self,
        request: GenerationRequest,
        index: int,
    ) -> GeneratedQuestion:

        return GeneratedQuestion(
            id=self._new_id(),
            question=(
                f"Explain {request.topic} in "
                f"detail. Include its definition, "
                f"important features, examples, "
                f"and significance in "
                f"{request.subject}."
            ),
            question_type="long_answer",
            correct_answer=(
                f"A complete answer should cover "
                f"the definition, major concepts, "
                f"examples and significance of "
                f"{request.topic}."
            ),
            explanation=(
                "Long-answer questions should "
                "demonstrate understanding rather "
                "than simple memorization."
            ),
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            difficulty=self._resolve_difficulty(
                request.difficulty,
                index,
            ),
            marks=request.marks,
            negative_marks=(
                request.negative_marks
            ),
            hints=(
                [
                    "Define the concept first.",
                    "Explain the main points.",
                    "Add examples where relevant.",
                ]
                if request.include_hints
                else []
            ),
            tags=[
                request.subject,
                request.topic,
            ],
            source="local",
        )

    # ========================================================
    # FILL BLANK
    # ========================================================

    def _local_fill_blank(
        self,
        request: GenerationRequest,
        index: int,
    ) -> GeneratedQuestion:

        return GeneratedQuestion(
            id=self._new_id(),
            question=(
                f"Complete the statement: "
                f"{request.topic} is an important "
                f"concept in ______."
            ),
            question_type="fill_blank",
            correct_answer=(
                request.subject
            ),
            explanation=(
                f"{request.topic} is studied "
                f"within {request.subject}."
            ),
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            difficulty=self._resolve_difficulty(
                request.difficulty,
                index,
            ),
            marks=request.marks,
            negative_marks=(
                request.negative_marks
            ),
            hints=(
                [
                    "Think about the subject "
                    "where this topic belongs."
                ]
                if request.include_hints
                else []
            ),
            tags=[
                request.subject,
                request.topic,
            ],
            source="local",
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_questions(
        self,
        questions: Sequence[
            GeneratedQuestion
        ],
        request: GenerationRequest,
    ) -> list[GeneratedQuestion]:

        normalized = []

        for question in questions:

            if not isinstance(
                question,
                GeneratedQuestion,
            ):

                continue

            if not question.question.strip():

                continue

            question.subject = (
                question.subject
                or request.subject
            )

            question.chapter = (
                question.chapter
                or request.chapter
            )

            question.topic = (
                question.topic
                or request.topic
            )

            if not request.include_explanations:

                question.explanation = ""

            if not request.include_hints:

                question.hints = []

            question.question_type = (
                question.question_type
                .strip()
                .lower()
            )

            if (
                question.question_type
                not in QUESTION_TYPES
            ):

                question.question_type = (
                    request.question_type
                )

            if not question.id:

                question.id = self._new_id()

            normalized.append(
                question
            )

        return normalized

    def _question_from_dict(
        self,
        data: dict[str, Any],
        request: GenerationRequest,
        *,
        source: str,
    ) -> GeneratedQuestion | None:

        if not isinstance(
            data,
            dict,
        ):

            return None

        question_text = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        if not question_text:

            return None

        question_type = str(
            data.get(
                "question_type",
                request.question_type,
            )
        ).strip().lower()

        if question_type not in QUESTION_TYPES:

            question_type = (
                request.question_type
            )

        options = data.get(
            "options",
            [],
        )

        if not isinstance(
            options,
            list,
        ):

            options = []

        hints = data.get(
            "hints",
            [],
        )

        if not isinstance(
            hints,
            list,
        ):

            hints = []

        tags = data.get(
            "tags",
            [],
        )

        if not isinstance(
            tags,
            list,
        ):

            tags = []

        return GeneratedQuestion(
            id=str(
                data.get(
                    "id",
                    self._new_id(),
                )
            ),
            question=question_text,
            question_type=question_type,
            correct_answer=data.get(
                "correct_answer"
            ),
            options=[
                str(item)
                for item in options
            ],
            explanation=str(
                data.get(
                    "explanation",
                    "",
                )
            ),
            subject=str(
                data.get(
                    "subject",
                    request.subject,
                )
            ),
            chapter=str(
                data.get(
                    "chapter",
                    request.chapter,
                )
            ),
            topic=str(
                data.get(
                    "topic",
                    request.topic,
                )
            ),
            difficulty=self._normalize_difficulty(
                data.get(
                    "difficulty",
                    request.difficulty,
                )
            ),
            marks=self._safe_float(
                data.get(
                    "marks",
                    request.marks,
                ),
                request.marks,
            ),
            negative_marks=self._safe_float(
                data.get(
                    "negative_marks",
                    request.negative_marks,
                ),
                request.negative_marks,
            ),
            hints=[
                str(item)
                for item in hints
            ],
            tags=[
                str(item)
                for item in tags
            ],
            source=source,
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    # ========================================================
    # DUPLICATE HANDLING
    # ========================================================

    def _remove_duplicates(
        self,
        questions: Sequence[
            GeneratedQuestion
        ],
    ) -> list[GeneratedQuestion]:

        seen: set[str] = set()

        result = []

        for question in questions:

            key = self._question_key(
                question.question
            )

            if key in seen:

                continue

            seen.add(key)

            result.append(
                question
            )

        return result

    def _fill_missing(
        self,
        request: GenerationRequest,
        *,
        existing: Sequence[
            GeneratedQuestion
        ],
        required: int,
    ) -> list[GeneratedQuestion]:

        generated = []

        existing_keys = {
            self._question_key(
                question.question
            )
            for question in existing
        }

        attempts = 0

        while (
            len(generated) < required
            and attempts < required * 10
        ):

            attempts += 1

            question = (
                self._generate_local_question(
                    request,
                    attempts,
                )
            )

            key = self._question_key(
                question.question
            )

            if (
                request.avoid_duplicates
                and key in existing_keys
            ):

                continue

            existing_keys.add(key)

            generated.append(
                question
            )

        return generated

    # ========================================================
    # REQUEST VALIDATION
    # ========================================================

    def _build_request(
        self,
        *,
        subject: str,
        topic: str,
        count: int,
        question_type: str,
        difficulty: str,
        exam_type: str,
        chapter: str,
        marks: float,
        negative_marks: float,
        include_explanations: bool,
        include_hints: bool,
        avoid_duplicates: bool,
        language: str,
        syllabus_context: str,
        extra_instructions: str,
        metadata: dict[str, Any] | None,
    ) -> GenerationRequest:

        subject = self._required_text(
            subject,
            "subject",
        )

        topic = self._required_text(
            topic,
            "topic",
        )

        if not isinstance(
            count,
            int,
        ) or count <= 0:

            raise ValueError(
                "count must be a positive integer."
            )

        if count > 500:

            raise ValueError(
                "count cannot exceed 500."
            )

        question_type = (
            question_type.strip().lower()
        )

        if question_type not in QUESTION_TYPES:

            raise ValueError(
                "Unsupported question type: "
                + question_type
            )

        difficulty = (
            difficulty.strip().lower()
        )

        if difficulty not in DIFFICULTIES:

            raise ValueError(
                "Unsupported difficulty: "
                + difficulty
            )

        exam_type = (
            exam_type.strip().lower()
        )

        if exam_type not in QUESTION_COUNTS:

            raise ValueError(
                "Unsupported exam type: "
                + exam_type
            )

        if marks <= 0:

            raise ValueError(
                "marks must be greater than zero."
            )

        if negative_marks < 0:

            raise ValueError(
                "negative_marks cannot be negative."
            )

        return GenerationRequest(
            subject=subject,
            topic=topic,
            count=count,
            question_type=question_type,
            difficulty=difficulty,
            exam_type=exam_type,
            chapter=chapter.strip(),
            marks=float(marks),
            negative_marks=float(
                negative_marks
            ),
            include_explanations=(
                include_explanations
            ),
            include_hints=include_hints,
            avoid_duplicates=(
                avoid_duplicates
            ),
            language=language.strip()
            or "English",
            syllabus_context=(
                syllabus_context.strip()
            ),
            extra_instructions=(
                extra_instructions.strip()
            ),
            metadata=dict(
                metadata or {}
            ),
        )

    # ========================================================
    # DIFFICULTY
    # ========================================================

    def _resolve_difficulty(
        self,
        difficulty: str,
        index: int,
    ) -> str:

        if difficulty != "mixed":

            return difficulty

        cycle = [
            "easy",
            "medium",
            "hard",
        ]

        return cycle[
            index % len(cycle)
        ]

    @staticmethod
    def _normalize_difficulty(
        difficulty: Any,
    ) -> str:

        value = str(
            difficulty
        ).strip().lower()

        if value not in {
            "easy",
            "medium",
            "hard",
        }:

            return "medium"

        return value

    # ========================================================
    # MEMORY
    # ========================================================

    def _store_memory(
        self,
        request: GenerationRequest,
        questions: Sequence[
            GeneratedQuestion
        ],
    ) -> None:

        if self.memory is None:
            return

        payload = {
            "type": "question_generation",
            "subject": request.subject,
            "topic": request.topic,
            "count": len(questions),
            "difficulty": request.difficulty,
            "question_type": (
                request.question_type
            ),
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        try:

            remember = getattr(
                self.memory,
                "remember",
                None,
            )

            if callable(remember):

                remember(payload)

        except Exception:

            logger.exception(
                "Failed to store question "
                "generation memory."
            )

    # ========================================================
    # HISTORY
    # ========================================================

    def get_history(
        self,
        *,
        subject: str | None = None,
        topic: str | None = None,
        limit: int | None = None,
    ) -> list[GeneratedQuestion]:

        history = list(
            self._history
        )

        if subject:

            key = subject.casefold()

            history = [
                item
                for item in history
                if item.subject.casefold()
                == key
            ]

        if topic:

            key = topic.casefold()

            history = [
                item
                for item in history
                if item.topic.casefold()
                == key
            ]

        if limit is not None:

            if limit < 0:

                raise ValueError(
                    "limit cannot be negative."
                )

            history = history[
                -limit:
            ]

        return history

    def clear_history(self) -> None:

        self._history.clear()

    # ========================================================
    # UTILITY
    # ========================================================

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
    def _question_key(
        question: str,
    ) -> str:

        normalized = (
            question.strip()
            .casefold()
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"[^\w\s]",
            "",
            normalized,
        )

        return normalized

    @staticmethod
    def _new_id() -> str:

        return (
            "generated_question_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _safe_float(
        value: Any,
        fallback: float,
    ) -> float:

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return float(fallback)

    @staticmethod
    def _strip_code_fences(
        text: str,
    ) -> str:

        text = text.strip()

        if text.startswith(
            "```"
        ):

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

        return text.strip()

    @staticmethod
    def _extract_json(
        text: str,
    ) -> str | None:

        start_object = text.find("{")
        end_object = text.rfind("}")

        if (
            start_object >= 0
            and end_object > start_object
        ):

            return text[
                start_object:
                end_object + 1
            ]

        start_array = text.find("[")
        end_array = text.rfind("]")

        if (
            start_array >= 0
            and end_array > start_array
        ):

            return text[
                start_array:
                end_array + 1
            ]

        return None

    def to_json(
        self,
        questions: Sequence[
            GeneratedQuestion
        ],
        *,
        indent: int = 2,
    ) -> str:

        return json.dumps(
            [
                question.to_dict()
                for question in questions
            ],
            indent=indent,
            ensure_ascii=False,
        )


__all__ = [
    "QUESTION_TYPES",
    "DIFFICULTIES",
    "QUESTION_COUNTS",
    "GeneratedQuestion",
    "GenerationRequest",
    "QuestionGenerator",
]


