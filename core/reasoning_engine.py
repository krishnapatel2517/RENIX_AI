"""
RENIX AI
Core Reasoning Engine

Responsible for:
- Understanding a reasoning request
- Gathering contextual information
- Building reasoning inputs
- Breaking complex problems into reasoning steps
- Evaluating possible solutions
- Selecting the strongest solution
- Producing structured reasoning results
- Supporting other RENIX engines and agents

This module does not directly execute computer actions.
Execution belongs to execution_engine.py.
"""

from __future__ import annotations

import logging
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(
    "RENIX.ReasoningEngine"
)


# ============================================================================
# REASONING STEP
# ============================================================================

@dataclass
class ReasoningStep:
    """
    Represents one step in a reasoning process.
    """

    step_id: str

    description: str

    purpose: str = ""

    input_data: Any = None

    output_data: Any = None

    status: str = "pending"

    confidence: float = 1.0

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the reasoning step to a dictionary.
        """

        return {
            "step_id": self.step_id,
            "description": self.description,
            "purpose": self.purpose,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ============================================================================
# REASONING OPTION
# ============================================================================

@dataclass
class ReasoningOption:
    """
    Represents a possible solution or approach.
    """

    option_id: str

    name: str

    description: str

    score: float = 0.0

    confidence: float = 0.0

    advantages: list[str] = field(
        default_factory=list
    )

    disadvantages: list[str] = field(
        default_factory=list
    )

    risks: list[str] = field(
        default_factory=list
    )

    requirements: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the option to a dictionary.
        """

        return {
            "option_id": self.option_id,
            "name": self.name,
            "description": self.description,
            "score": self.score,
            "confidence": self.confidence,
            "advantages": list(
                self.advantages
            ),
            "disadvantages": list(
                self.disadvantages
            ),
            "risks": list(
                self.risks
            ),
            "requirements": list(
                self.requirements
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# REASONING RESULT
# ============================================================================

@dataclass
class ReasoningResult:
    """
    Complete result of a reasoning operation.
    """

    reasoning_id: str

    query: str

    answer: Any = None

    conclusion: Any = None

    confidence: float = 0.0

    status: str = "completed"

    steps: list[ReasoningStep] = field(
        default_factory=list
    )

    options: list[ReasoningOption] = field(
        default_factory=list
    )

    selected_option: Optional[str] = None

    assumptions: list[str] = field(
        default_factory=list
    )

    uncertainties: list[str] = field(
        default_factory=list
    )

    execution_required: bool = False

    execution_plan: Optional[Any] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert result to dictionary.
        """

        return {
            "reasoning_id": self.reasoning_id,
            "query": self.query,
            "answer": self.answer,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "status": self.status,
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
            "options": [
                option.to_dict()
                for option in self.options
            ],
            "selected_option": (
                self.selected_option
            ),
            "assumptions": list(
                self.assumptions
            ),
            "uncertainties": list(
                self.uncertainties
            ),
            "execution_required": (
                self.execution_required
            ),
            "execution_plan": (
                self.execution_plan
            ),
            "metadata": dict(
                self.metadata
            ),
            "created_at": self.created_at,
        }


# ============================================================================
# REASONING ENGINE
# ============================================================================

class ReasoningEngine:
    """
    Main reasoning engine for RENIX.
    """

    def __init__(
        self,
        context_engine: Any = None,
        llm_router: Any = None,
        planner: Any = None,
        verifier: Any = None,
    ) -> None:

        self.logger = logger

        self.context_engine = (
            context_engine
        )

        self.llm_router = (
            llm_router
        )

        self.planner = planner

        self.verifier = verifier

        self.initialized = False

        self.created_at = time.time()

        self.last_reasoning: Optional[
            ReasoningResult
        ] = None

        self.history: list[
            ReasoningResult
        ] = []

        self.max_history = 500

        self.reasoning_count = 0

        self.success_count = 0

        self.failure_count = 0

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the reasoning engine.
        """

        self.initialized = True

        self.logger.info(
            "RENIX Reasoning Engine initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the reasoning engine.
        """

        self.initialized = False

        self.logger.info(
            "RENIX Reasoning Engine shutdown."
        )

    # ========================================================================
    # GENERATE ID
    # ========================================================================

    @staticmethod
    def _generate_id(
        prefix: str = "reasoning",
    ) -> str:
        """
        Generate a unique reasoning ID.
        """

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    # ========================================================================
    # NORMALIZE QUERY
    # ========================================================================

    @staticmethod
    def _normalize_query(
        query: Any,
    ) -> str:
        """
        Normalize incoming reasoning query.
        """

        if query is None:

            return ""

        return str(
            query
        ).strip()

    # ========================================================================
    # BUILD CONTEXT
    # ========================================================================

    def build_context(
        self,
        query: str,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> dict[str, Any]:
        """
        Build the reasoning context.
        """

        result: dict[
            str,
            Any
        ] = {}

        if self.context_engine is not None:

            try:

                if hasattr(
                    self.context_engine,
                    "build_context_window",
                ):

                    result.update(
                        self.context_engine
                        .build_context_window()
                    )

                elif hasattr(
                    self.context_engine,
                    "get_all",
                ):

                    result[
                        "context"
                    ] = (
                        self.context_engine
                        .get_all()
                    )

            except Exception as exc:

                self.logger.warning(
                    "Unable to obtain context: %s",
                    exc,
                )

        if context:

            result.update(
                context
            )

        result[
            "query"
        ] = query

        return result

    # ========================================================================
    # CREATE STEP
    # ========================================================================

    def create_step(
        self,
        description: str,
        *,
        purpose: str = "",
        input_data: Any = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ReasoningStep:
        """
        Create a reasoning step.
        """

        return ReasoningStep(
            step_id=self._generate_id(
                "step"
            ),
            description=description,
            purpose=purpose,
            input_data=input_data,
            metadata=metadata or {},
        )

    # ========================================================================
    # BASIC QUERY ANALYSIS
    # ========================================================================

    def analyze_query(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Perform basic structural analysis of a query.
        """

        normalized = (
            self._normalize_query(
                query
            )
        )

        words = normalized.split()

        lower = normalized.lower()

        question_words = {
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
            "can",
            "should",
            "is",
            "are",
            "will",
        }

        detected_question_words = [
            word
            for word in words
            if word.lower()
            in question_words
        ]

        action_words = {
            "open",
            "close",
            "create",
            "delete",
            "move",
            "copy",
            "rename",
            "search",
            "find",
            "play",
            "stop",
            "start",
            "run",
            "build",
            "write",
            "read",
            "send",
            "download",
            "upload",
            "schedule",
            "remind",
            "calculate",
            "compare",
            "explain",
            "analyze",
            "solve",
        }

        detected_actions = [
            word
            for word in words
            if word.lower()
            in action_words
        ]

        domains = []

        domain_keywords = {
            "computer": {
                "computer",
                "desktop",
                "screen",
                "mouse",
                "keyboard",
                "window",
            },
            "files": {
                "file",
                "folder",
                "document",
                "pdf",
            },
            "browser": {
                "browser",
                "website",
                "web",
                "internet",
                "chrome",
                "search",
            },
            "coding": {
                "code",
                "coding",
                "python",
                "program",
                "script",
                "debug",
                "project",
            },
            "study": {
                "study",
                "exam",
                "homework",
                "maths",
                "science",
                "revision",
            },
            "media": {
                "music",
                "song",
                "video",
                "movie",
                "playlist",
            },
            "cricket": {
                "cricket",
                "batting",
                "bowling",
                "match",
                "runs",
                "wicket",
            },
            "automation": {
                "automate",
                "automation",
                "workflow",
                "routine",
                "schedule",
            },
        }

        for domain, keywords in (
            domain_keywords.items()
        ):

            if any(
                keyword in lower
                for keyword in keywords
            ):

                domains.append(
                    domain
                )

        complexity = "simple"

        if len(words) > 20:

            complexity = "complex"

        elif len(words) > 10:

            complexity = "moderate"

        if len(
            detected_actions
        ) > 1:

            complexity = "complex"

        return {
            "query": normalized,
            "word_count": len(words),
            "question_words": (
                detected_question_words
            ),
            "actions": detected_actions,
            "domains": domains,
            "complexity": complexity,
            "requires_reasoning": (
                complexity != "simple"
                or bool(
                    detected_actions
                )
            ),
        }

    # ========================================================================
    # CREATE REASONING PLAN
    # ========================================================================

    def create_reasoning_plan(
        self,
        query: str,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> list[ReasoningStep]:
        """
        Create a structured reasoning plan.
        """

        analysis = self.analyze_query(
            query
        )

        steps: list[
            ReasoningStep
        ] = []

        steps.append(
            self.create_step(
                "Understand the user's request.",
                purpose=(
                    "Determine exactly what "
                    "the user wants."
                ),
                input_data=query,
            )
        )

        steps.append(
            self.create_step(
                "Inspect available context.",
                purpose=(
                    "Use relevant conversation, "
                    "task, file, application, "
                    "and session information."
                ),
                input_data=context,
            )
        )

        if analysis[
            "domains"
        ]:

            steps.append(
                self.create_step(
                    "Identify relevant domains.",
                    purpose=(
                        "Determine which RENIX "
                        "capabilities may be required."
                    ),
                    input_data=analysis[
                        "domains"
                    ],
                )
            )

        if analysis[
            "actions"
        ]:

            steps.append(
                self.create_step(
                    "Determine required actions.",
                    purpose=(
                        "Identify operations that "
                        "may need execution."
                    ),
                    input_data=analysis[
                        "actions"
                    ],
                )
            )

        if analysis[
            "complexity"
        ] in {
            "moderate",
            "complex",
        }:

            steps.append(
                self.create_step(
                    "Break the problem into manageable parts.",
                    purpose=(
                        "Decompose the request into "
                        "logical sub-problems."
                    ),
                )
            )

        steps.append(
            self.create_step(
                "Evaluate possible solutions.",
                purpose=(
                    "Compare approaches and "
                    "identify the strongest one."
                ),
            )
        )

        steps.append(
            self.create_step(
                "Verify the proposed solution.",
                purpose=(
                    "Check consistency, safety, "
                    "and completeness."
                ),
            )
        )

        steps.append(
            self.create_step(
                "Produce the final reasoning result.",
                purpose=(
                    "Return the best available "
                    "answer or action plan."
                ),
            )
        )

        return steps

    # ========================================================================
    # GENERATE OPTIONS
    # ========================================================================

    def generate_options(
        self,
        query: str,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> list[ReasoningOption]:
        """
        Generate possible approaches.

        When an LLM router is available, this method can be
        expanded by the router. The fallback implementation
        provides safe structured options.
        """

        analysis = self.analyze_query(
            query
        )

        options: list[
            ReasoningOption
        ] = []

        if analysis[
            "actions"
        ]:

            options.append(
                ReasoningOption(
                    option_id=self._generate_id(
                        "option"
                    ),
                    name="direct_action",
                    description=(
                        "Perform the requested "
                        "operation directly."
                    ),
                    score=0.75,
                    confidence=0.75,
                    advantages=[
                        "Fast",
                        "Simple",
                        "Direct",
                    ],
                    disadvantages=[
                        "May require clarification "
                        "for ambiguous requests."
                    ],
                    risks=[
                        "Potentially destructive "
                        "operations require confirmation."
                    ],
                )
            )

        options.append(
            ReasoningOption(
                option_id=self._generate_id(
                    "option"
                ),
                name="clarify",
                description=(
                    "Ask for clarification when "
                    "the request cannot be resolved "
                    "reliably from available context."
                ),
                score=0.50,
                confidence=0.90,
                advantages=[
                    "Reduces ambiguity",
                    "Prevents incorrect actions",
                ],
                disadvantages=[
                    "Slower interaction",
                ],
            )
        )

        options.append(
            ReasoningOption(
                option_id=self._generate_id(
                    "option"
                ),
                name="analyze_first",
                description=(
                    "Analyze the request and "
                    "available context before acting."
                ),
                score=0.85,
                confidence=0.85,
                advantages=[
                    "More reliable",
                    "Better for complex tasks",
                    "Better contextual awareness",
                ],
                disadvantages=[
                    "Requires additional processing",
                ],
            )
        )

        return options

    # ========================================================================
    # SCORE OPTIONS
    # ========================================================================

    def score_options(
        self,
        options: list[ReasoningOption],
        *,
        query_analysis: Optional[
            dict[str, Any]
        ] = None,
    ) -> list[ReasoningOption]:
        """
        Score possible reasoning options.
        """

        analysis = (
            query_analysis
            or {}
        )

        complexity = analysis.get(
            "complexity",
            "simple",
        )

        has_actions = bool(
            analysis.get(
                "actions"
            )
        )

        for option in options:

            score = float(
                option.score
            )

            if (
                complexity == "complex"
                and option.name
                == "analyze_first"
            ):

                score += 0.10

            if (
                has_actions
                and option.name
                == "direct_action"
            ):

                score += 0.05

            option.score = max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

            option.confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        option.confidence
                    ),
                ),
            )

        options.sort(
            key=lambda option: (
                option.score
            ),
            reverse=True,
        )

        return options

    # ========================================================================
    # SELECT BEST OPTION
    # ========================================================================

    def select_best_option(
        self,
        options: list[ReasoningOption],
    ) -> Optional[
        ReasoningOption
    ]:
        """
        Select the highest-scoring option.
        """

        if not options:

            return None

        return max(
            options,
            key=lambda option: (
                option.score
            ),
        )

    # ========================================================================
    # VERIFY RESULT
    # ========================================================================

    def verify_result(
        self,
        result: ReasoningResult,
    ) -> bool:
        """
        Verify basic consistency of a reasoning result.
        """

        if not result.query:

            return False

        if not result.steps:

            return False

        if not (
            0.0
            <= result.confidence
            <= 1.0
        ):

            return False

        for step in result.steps:

            if step.status == "failed":

                return False

        return True

    # ========================================================================
    # REASON
    # ========================================================================

    async def reason(
        self,
        query: str,
        *,
        context: Optional[
            dict[str, Any]
        ] = None,
        execute: bool = False,
    ) -> ReasoningResult:
        """
        Main reasoning entry point.
        """

        normalized_query = (
            self._normalize_query(
                query
            )
        )

        reasoning_id = (
            self._generate_id()
        )

        result = ReasoningResult(
            reasoning_id=reasoning_id,
            query=normalized_query,
            status="processing",
        )

        self.reasoning_count += 1

        try:

            if not normalized_query:

                result.status = "failed"

                result.answer = (
                    "No reasoning query was provided."
                )

                result.confidence = 0.0

                self.failure_count += 1

                self._store_result(
                    result
                )

                return result

            # ---------------------------------------------------------------
            # Context
            # ---------------------------------------------------------------

            reasoning_context = (
                self.build_context(
                    normalized_query,
                    context,
                )
            )

            # ---------------------------------------------------------------
            # Query analysis
            # ---------------------------------------------------------------

            analysis = (
                self.analyze_query(
                    normalized_query
                )
            )

            result.metadata[
                "query_analysis"
            ] = analysis

            # ---------------------------------------------------------------
            # Plan
            # ---------------------------------------------------------------

            steps = (
                self.create_reasoning_plan(
                    normalized_query,
                    reasoning_context,
                )
            )

            result.steps = steps

            # ---------------------------------------------------------------
            # Mark initial steps complete
            # ---------------------------------------------------------------

            for step in result.steps:

                step.status = "completed"

                step.confidence = 0.90

            # ---------------------------------------------------------------
            # Generate options
            # ---------------------------------------------------------------

            options = (
                self.generate_options(
                    normalized_query,
                    reasoning_context,
                )
            )

            options = self.score_options(
                options,
                query_analysis=analysis,
            )

            result.options = options

            # ---------------------------------------------------------------
            # Select best option
            # ---------------------------------------------------------------

            selected = (
                self.select_best_option(
                    options
                )
            )

            if selected:

                result.selected_option = (
                    selected.option_id
                )

                result.confidence = (
                    selected.confidence
                )

            # ---------------------------------------------------------------
            # Optional external LLM reasoning
            # ---------------------------------------------------------------

            llm_answer = None

            if self.llm_router is not None:

                try:

                    llm_answer = (
                        await self._ask_llm(
                            normalized_query,
                            reasoning_context,
                            analysis,
                        )
                    )

                except Exception as exc:

                    self.logger.warning(
                        "LLM reasoning unavailable: %s",
                        exc,
                    )

            # ---------------------------------------------------------------
            # Final answer
            # ---------------------------------------------------------------

            if llm_answer is not None:

                result.answer = (
                    llm_answer
                )

                result.conclusion = (
                    llm_answer
                )

                result.confidence = max(
                    result.confidence,
                    0.85,
                )

            else:

                result.answer = (
                    self._fallback_answer(
                        normalized_query,
                        analysis,
                        selected,
                    )
                )

                result.conclusion = (
                    result.answer
                )

            # ---------------------------------------------------------------
            # Determine whether execution is required
            # ---------------------------------------------------------------

            result.execution_required = (
                execute
                and bool(
                    analysis.get(
                        "actions"
                    )
                )
            )

            result.status = "completed"

            if self.verify_result(
                result
            ):

                self.success_count += 1

            else:

                result.status = "failed"

                self.failure_count += 1

            self._store_result(
                result
            )

            return result

        except Exception as exc:

            self.logger.exception(
                "Reasoning failure: %s",
                exc,
            )

            result.status = "failed"

            result.answer = (
                "I was unable to complete "
                "the reasoning process."
            )

            result.conclusion = (
                str(exc)
            )

            result.confidence = 0.0

            self.failure_count += 1

            self._store_result(
                result
            )

            return result

    # ========================================================================
    # ASK LLM
    # ========================================================================

    async def _ask_llm(
        self,
        query: str,
        context: dict[str, Any],
        analysis: dict[str, Any],
    ) -> Any:
        """
        Ask the configured LLM router for reasoning.

        This method supports several common router interfaces
        without forcing one specific provider implementation.
        """

        router = self.llm_router

        prompt = self._build_reasoning_prompt(
            query,
            context,
            analysis,
        )

        # --------------------------------------------------------------------
        # generate()
        # --------------------------------------------------------------------

        if hasattr(
            router,
            "generate",
        ):

            response = router.generate(
                prompt
            )

            if hasattr(
                response,
                "__await__",
            ):

                response = await response

            return self._extract_llm_text(
                response
            )

        # --------------------------------------------------------------------
        # complete()
        # --------------------------------------------------------------------

        if hasattr(
            router,
            "complete",
        ):

            response = router.complete(
                prompt
            )

            if hasattr(
                response,
                "__await__",
            ):

                response = await response

            return self._extract_llm_text(
                response
            )

        # --------------------------------------------------------------------
        # chat()
        # --------------------------------------------------------------------

        if hasattr(
            router,
            "chat",
        ):

            response = router.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are RENIX's "
                            "reasoning engine."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )

            if hasattr(
                response,
                "__await__",
            ):

                response = await response

            return self._extract_llm_text(
                response
            )

        return None

    # ========================================================================
    # BUILD REASONING PROMPT
    # ========================================================================

    def _build_reasoning_prompt(
        self,
        query: str,
        context: dict[str, Any],
        analysis: dict[str, Any],
    ) -> str:
        """
        Build an LLM reasoning prompt.
        """

        return f"""
You are RENIX's reasoning engine.

Understand the user's request accurately.

USER REQUEST:
{query}

QUERY ANALYSIS:
{analysis}

AVAILABLE CONTEXT:
{context}

Requirements:
1. Understand the user's actual goal.
2. Use relevant context.
3. Do not invent unavailable information.
4. Identify ambiguity when it matters.
5. Consider the safest and most useful approach.
6. Prefer direct solutions when the request is clear.
7. Do not execute computer operations yourself.
8. Return a concise, useful answer.
""".strip()

    # ========================================================================
    # EXTRACT LLM TEXT
    # ========================================================================

    @staticmethod
    def _extract_llm_text(
        response: Any,
    ) -> Any:
        """
        Extract usable text from common LLM response objects.
        """

        if response is None:

            return None

        if isinstance(
            response,
            str,
        ):

            return response

        if isinstance(
            response,
            dict,
        ):

            for key in (
                "text",
                "content",
                "answer",
                "response",
                "output",
            ):

                if key in response:

                    return response[
                        key
                    ]

        for attribute in (
            "text",
            "content",
            "answer",
            "response",
            "output",
        ):

            if hasattr(
                response,
                attribute,
            ):

                value = getattr(
                    response,
                    attribute,
                )

                if value is not None:

                    return value

        return str(
            response
        )

    # ========================================================================
    # FALLBACK ANSWER
    # ========================================================================

    def _fallback_answer(
        self,
        query: str,
        analysis: dict[str, Any],
        selected: Optional[
            ReasoningOption
        ],
    ) -> str:
        """
        Produce a deterministic fallback answer when no LLM
        provider is available.
        """

        if selected is None:

            return (
                "I understand the request, "
                "but I need more information "
                "to determine the best approach."
            )

        if selected.name == (
            "direct_action"
        ):

            return (
                "The request appears to require "
                "a direct action. The execution "
                "engine should handle the required "
                "operation."
            )

        if selected.name == (
            "clarify"
        ):

            return (
                "The request may require "
                "additional clarification before "
                "a reliable action can be selected."
            )

        return (
            "The request has been analyzed and "
            "the strongest approach is to process "
            "the available context before taking action."
        )

    # ========================================================================
    # STORE RESULT
    # ========================================================================

    def _store_result(
        self,
        result: ReasoningResult,
    ) -> None:
        """
        Store reasoning result in history.
        """

        self.last_reasoning = result

        self.history.append(
            result
        )

        if len(
            self.history
        ) > self.max_history:

            self.history = (
                self.history[
                    -self.max_history:
                ]
            )

    # ========================================================================
    # LAST RESULT
    # ========================================================================

    def get_last_result(
        self,
    ) -> Optional[
        ReasoningResult
    ]:
        """
        Return the latest reasoning result.
        """

        return self.last_reasoning

    # ========================================================================
    # HISTORY
    # ========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[ReasoningResult]:
        """
        Return reasoning history.
        """

        history = list(
            self.history
        )

        if limit is not None:

            limit = max(
                0,
                int(limit),
            )

            history = history[
                -limit:
            ]

        return history

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear reasoning history.
        """

        self.history.clear()

        self.last_reasoning = None

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return engine statistics.
        """

        success_rate = 0.0

        if self.reasoning_count:

            success_rate = (
                self.success_count
                / self.reasoning_count
            )

        return {
            "initialized": (
                self.initialized
            ),
            "reasoning_count": (
                self.reasoning_count
            ),
            "success_count": (
                self.success_count
            ),
            "failure_count": (
                self.failure_count
            ),
            "success_rate": success_rate,
            "history_size": len(
                self.history
            ),
            "created_at": (
                self.created_at
            ),
        }


# ============================================================================
# GLOBAL ENGINE
# ============================================================================

reasoning_engine = ReasoningEngine()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def reason(
    query: str,
    *,
    context: Optional[
        dict[str, Any]
    ] = None,
    execute: bool = False,
) -> ReasoningResult:
    """
    Convenience wrapper around the global
    RENIX reasoning engine.
    """

    return await reasoning_engine.reason(
        query,
        context=context,
        execute=execute,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ReasoningStep",
    "ReasoningOption",
    "ReasoningResult",
    "ReasoningEngine",
    "reasoning_engine",
    "reason",
]


