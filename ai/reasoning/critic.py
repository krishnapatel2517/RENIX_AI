"""
RENIX AI
Reasoning Critic

Evaluates plans, solutions, decisions, answers and generated outputs
before RENIX accepts or executes them.

The critic is intentionally independent from the LLM provider.
An optional AI callback can be connected later through the existing
RENIX reasoning/LLM architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


# ============================================================================
# ENUMS
# ============================================================================


class CritiqueStatus(str, Enum):
    """Overall critique status."""

    ACCEPT = "accept"
    ACCEPT_WITH_WARNINGS = "accept_with_warnings"
    REVISE = "revise"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class CritiqueSeverity(str, Enum):
    """Severity of a discovered problem."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CritiqueCategory(str, Enum):
    """Categories used by the critic."""

    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    FEASIBILITY = "feasibility"
    SAFETY = "safety"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    EFFICIENCY = "efficiency"
    DEPENDENCY = "dependency"
    EXECUTION = "execution"
    UNKNOWN = "unknown"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class CritiqueIssue:
    """One issue found during critique."""

    message: str

    severity: CritiqueSeverity = (
        CritiqueSeverity.WARNING
    )

    category: CritiqueCategory = (
        CritiqueCategory.UNKNOWN
    )

    code: Optional[str] = None

    location: Optional[str] = None

    evidence: Any = None

    suggestion: Optional[str] = None

    metadata: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the issue to a dictionary."""

        return {
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "code": self.code,
            "location": self.location,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "metadata": dict(self.metadata),
        }


@dataclass
class CritiqueResult:
    """Complete result returned by the critic."""

    status: CritiqueStatus

    score: float = 0.0

    confidence: float = 0.0

    summary: str = ""

    strengths: List[str] = dataclass_field(
        default_factory=list
    )

    weaknesses: List[str] = dataclass_field(
        default_factory=list
    )

    issues: List[CritiqueIssue] = dataclass_field(
        default_factory=list
    )

    recommendations: List[str] = dataclass_field(
        default_factory=list
    )

    checks_performed: List[str] = dataclass_field(
        default_factory=list
    )

    metadata: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )

    @property
    def accepted(self) -> bool:
        """Whether the result can be accepted."""

        return self.status in {
            CritiqueStatus.ACCEPT,
            CritiqueStatus.ACCEPT_WITH_WARNINGS,
        }

    @property
    def needs_revision(self) -> bool:
        """Whether revision is recommended."""

        return self.status == CritiqueStatus.REVISE

    @property
    def rejected(self) -> bool:
        """Whether the result should be rejected."""

        return self.status == CritiqueStatus.REJECT

    @property
    def critical_issues(self) -> List[CritiqueIssue]:
        """Return critical issues."""

        return [
            issue
            for issue in self.issues
            if issue.severity
            == CritiqueSeverity.CRITICAL
        ]

    @property
    def errors(self) -> List[CritiqueIssue]:
        """Return error and critical issues."""

        return [
            issue
            for issue in self.issues
            if issue.severity
            in {
                CritiqueSeverity.ERROR,
                CritiqueSeverity.CRITICAL,
            }
        ]

    @property
    def warnings(self) -> List[CritiqueIssue]:
        """Return warnings."""

        return [
            issue
            for issue in self.issues
            if issue.severity
            == CritiqueSeverity.WARNING
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""

        return {
            "status": self.status.value,
            "score": self.score,
            "confidence": self.confidence,
            "summary": self.summary,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "recommendations": list(
                self.recommendations
            ),
            "checks_performed": list(
                self.checks_performed
            ),
            "metadata": dict(self.metadata),
        }


@dataclass
class CritiqueRequest:
    """Structured critique request."""

    subject: Any

    goal: Optional[str] = None

    expected: Any = None

    context: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )

    criteria: List[str] = dataclass_field(
        default_factory=list
    )

    constraints: List[str] = dataclass_field(
        default_factory=list
    )

    strict: bool = False

    metadata: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )


# ============================================================================
# CRITIC
# ============================================================================


class Critic:
    """
    RENIX reasoning critic.

    Performs deterministic quality checks and optionally calls an AI
    evaluator for deeper semantic analysis.
    """

    def __init__(
        self,
        *,
        ai_callback: Optional[
            Callable[..., Any]
        ] = None,
        strict: bool = False,
    ) -> None:

        self.ai_callback = ai_callback
        self.strict = strict

    # ========================================================================
    # GENERAL CRITIQUE
    # ========================================================================

    def critique(
        self,
        subject: Any,
        *,
        goal: Optional[str] = None,
        expected: Any = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
        criteria: Optional[
            Sequence[str]
        ] = None,
        constraints: Optional[
            Sequence[str]
        ] = None,
        strict: Optional[bool] = None,
    ) -> CritiqueResult:
        """
        Critique any RENIX-generated result.

        Checks:
        - existence
        - structure
        - relevance
        - completeness
        - consistency
        - obvious errors
        - constraints
        - goal alignment
        """

        request = CritiqueRequest(
            subject=subject,
            goal=goal,
            expected=expected,
            context=dict(
                context or {}
            ),
            criteria=list(
                criteria or []
            ),
            constraints=list(
                constraints or []
            ),
            strict=(
                self.strict
                if strict is None
                else strict
            ),
        )

        result = self._create_result()

        # --------------------------------------------------------------------
        # EXISTENCE
        # --------------------------------------------------------------------

        self._check_existence(
            subject,
            result,
        )

        if result.errors:
            return self._finalize(
                result
            )

        # --------------------------------------------------------------------
        # STRUCTURE
        # --------------------------------------------------------------------

        self._check_structure(
            subject,
            result,
        )

        # --------------------------------------------------------------------
        # GOAL
        # --------------------------------------------------------------------

        if request.goal:

            self._check_goal_alignment(
                subject,
                request.goal,
                result,
            )

        # --------------------------------------------------------------------
        # EXPECTED VALUE
        # --------------------------------------------------------------------

        if request.expected is not None:

            self._check_expected(
                subject,
                request.expected,
                result,
            )

        # --------------------------------------------------------------------
        # CRITERIA
        # --------------------------------------------------------------------

        for criterion in request.criteria:

            self._check_criterion(
                subject,
                criterion,
                result,
            )

        # --------------------------------------------------------------------
        # CONSTRAINTS
        # --------------------------------------------------------------------

        for constraint in request.constraints:

            self._check_constraint(
                subject,
                constraint,
                result,
            )

        # --------------------------------------------------------------------
        # CONSISTENCY
        # --------------------------------------------------------------------

        self._check_consistency(
            subject,
            result,
        )

        # --------------------------------------------------------------------
        # QUALITY
        # --------------------------------------------------------------------

        self._check_quality(
            subject,
            result,
        )

        # --------------------------------------------------------------------
        # STRICT CHECKS
        # --------------------------------------------------------------------

        if request.strict:

            self._strict_checks(
                subject,
                result,
            )

        # --------------------------------------------------------------------
        # ADVANCED AI CRITIQUE
        # --------------------------------------------------------------------

        if self.ai_callback:

            ai_result = self._run_ai_critic(
                request
            )

            self._merge_ai_result(
                result,
                ai_result,
            )

        return self._finalize(
            result
        )

    # ========================================================================
    # PLAN CRITIQUE
    # ========================================================================

    def critique_plan(
        self,
        plan: Any,
        *,
        goal: Optional[str] = None,
        constraints: Optional[
            Sequence[str]
        ] = None,
        strict: Optional[bool] = None,
    ) -> CritiqueResult:
        """
        Critique an execution plan.
        """

        result = self._create_result()

        tasks = self._get_value(
            plan,
            "tasks",
            [],
        )

        result.checks_performed.append(
            "plan_tasks"
        )

        if not tasks:

            result.issues.append(
                CritiqueIssue(
                    message="The plan contains no tasks.",
                    severity=CritiqueSeverity.CRITICAL,
                    category=CritiqueCategory.COMPLETENESS,
                    code="EMPTY_PLAN",
                    suggestion=(
                        "Create at least one concrete "
                        "execution task."
                    ),
                )
            )

            return self._finalize(
                result
            )

        # --------------------------------------------------------------------
        # TASK IDS
        # --------------------------------------------------------------------

        task_ids: set[str] = set()

        for task in tasks:

            task_id = str(
                self._get_value(
                    task,
                    "id",
                    "",
                )
            ).strip()

            if not task_id:

                result.issues.append(
                    CritiqueIssue(
                        message="A plan task has no ID.",
                        severity=CritiqueSeverity.ERROR,
                        category=CritiqueCategory.COMPLETENESS,
                        code="TASK_ID_MISSING",
                    )
                )

                continue

            if task_id in task_ids:

                result.issues.append(
                    CritiqueIssue(
                        message=(
                            f"Duplicate task ID: {task_id}"
                        ),
                        severity=CritiqueSeverity.ERROR,
                        category=CritiqueCategory.CONSISTENCY,
                        code="DUPLICATE_TASK_ID",
                    )
                )

            task_ids.add(task_id)

        result.checks_performed.append(
            "unique_task_ids"
        )

        # --------------------------------------------------------------------
        # TASK CONTENT
        # --------------------------------------------------------------------

        for task in tasks:

            task_id = str(
                self._get_value(
                    task,
                    "id",
                    "unknown",
                )
            )

            title = self._get_value(
                task,
                "title",
                "",
            )

            description = self._get_value(
                task,
                "description",
                "",
            )

            if not str(title).strip():

                result.issues.append(
                    CritiqueIssue(
                        message=(
                            f"Task '{task_id}' "
                            "has no title."
                        ),
                        severity=CritiqueSeverity.ERROR,
                        category=CritiqueCategory.COMPLETENESS,
                        code="TASK_TITLE_MISSING",
                        location=task_id,
                    )
                )

            if (
                strict
                if strict is not None
                else self.strict
            ):

                if not str(
                    description
                ).strip():

                    result.issues.append(
                        CritiqueIssue(
                            message=(
                                f"Task '{task_id}' "
                                "has no description."
                            ),
                            severity=CritiqueSeverity.WARNING,
                            category=CritiqueCategory.CLARITY,
                            code="TASK_DESCRIPTION_MISSING",
                            location=task_id,
                        )
                    )

        result.checks_performed.append(
            "task_content"
        )

        # --------------------------------------------------------------------
        # DEPENDENCIES
        # --------------------------------------------------------------------

        dependency_map: Dict[
            str,
            List[str],
        ] = {}

        for task in tasks:

            task_id = str(
                self._get_value(
                    task,
                    "id",
                    "",
                )
            )

            dependencies = self._get_value(
                task,
                "dependencies",
                [],
            )

            dependency_map[
                task_id
            ] = list(
                dependencies or []
            )

            for dependency in (
                dependencies or []
            ):

                if dependency not in task_ids:

                    result.issues.append(
                        CritiqueIssue(
                            message=(
                                f"Task '{task_id}' "
                                f"depends on missing "
                                f"task '{dependency}'."
                            ),
                            severity=CritiqueSeverity.ERROR,
                            category=CritiqueCategory.DEPENDENCY,
                            code="MISSING_DEPENDENCY",
                            location=task_id,
                        )
                    )

                if dependency == task_id:

                    result.issues.append(
                        CritiqueIssue(
                            message=(
                                f"Task '{task_id}' "
                                "depends on itself."
                            ),
                            severity=CritiqueSeverity.CRITICAL,
                            category=CritiqueCategory.DEPENDENCY,
                            code="SELF_DEPENDENCY",
                            location=task_id,
                        )
                    )

        result.checks_performed.append(
            "dependency_integrity"
        )

        # --------------------------------------------------------------------
        # CYCLE DETECTION
        # --------------------------------------------------------------------

        if self._has_cycle(
            dependency_map
        ):

            result.issues.append(
                CritiqueIssue(
                    message=(
                        "The execution plan contains "
                        "a dependency cycle."
                    ),
                    severity=CritiqueSeverity.CRITICAL,
                    category=CritiqueCategory.DEPENDENCY,
                    code="DEPENDENCY_CYCLE",
                    suggestion=(
                        "Remove or reorder dependencies "
                        "so the task graph is acyclic."
                    ),
                )
            )

        result.checks_performed.append(
            "cycle_detection"
        )

        # --------------------------------------------------------------------
        # GOAL
        # --------------------------------------------------------------------

        if goal:

            self._check_goal_alignment(
                plan,
                goal,
                result,
            )

        # --------------------------------------------------------------------
        # CONSTRAINTS
        # --------------------------------------------------------------------

        for constraint in (
            constraints or []
        ):

            self._check_constraint(
                plan,
                constraint,
                result,
            )

        # --------------------------------------------------------------------
        # AI
        # --------------------------------------------------------------------

        if self.ai_callback:

            ai_result = self._run_ai_critic(
                CritiqueRequest(
                    subject=plan,
                    goal=goal,
                    constraints=list(
                        constraints or []
                    ),
                    strict=(
                        self.strict
                        if strict is None
                        else strict
                    ),
                )
            )

            self._merge_ai_result(
                result,
                ai_result,
            )

        return self._finalize(
            result
        )

    # ========================================================================
    # SOLUTION CRITIQUE
    # ========================================================================

    def critique_solution(
        self,
        problem: Any,
        solution: Any,
        *,
        strict: Optional[bool] = None,
    ) -> CritiqueResult:
        """
        Critique a proposed solution against a problem.
        """

        result = self._create_result()

        if solution is None:

            result.issues.append(
                CritiqueIssue(
                    message="No solution was provided.",
                    severity=CritiqueSeverity.CRITICAL,
                    category=CritiqueCategory.COMPLETENESS,
                    code="MISSING_SOLUTION",
                )
            )

            return self._finalize(
                result
            )

        # --------------------------------------------------------------------
        # SOLUTION DESCRIPTION
        # --------------------------------------------------------------------

        description = self._get_value(
            solution,
            "description",
            "",
        )

        if not str(
            description
        ).strip():

            result.issues.append(
                CritiqueIssue(
                    message=(
                        "The solution has no "
                        "meaningful description."
                    ),
                    severity=CritiqueSeverity.ERROR,
                    category=CritiqueCategory.CLARITY,
                    code="SOLUTION_DESCRIPTION_MISSING",
                )
            )

        # --------------------------------------------------------------------
        # SOLUTION STEPS
        # --------------------------------------------------------------------

        steps = self._get_value(
            solution,
            "steps",
            [],
        )

        if not steps:

            result.issues.append(
                CritiqueIssue(
                    message=(
                        "The solution contains "
                        "no actionable steps."
                    ),
                    severity=CritiqueSeverity.WARNING,
                    category=CritiqueCategory.COMPLETENESS,
                    code="NO_SOLUTION_STEPS",
                    suggestion=(
                        "Break the solution into "
                        "concrete executable steps."
                    ),
                )
            )

        result.checks_performed.append(
            "solution_steps"
        )

        # --------------------------------------------------------------------
        # PROBLEM REQUIREMENTS
        # --------------------------------------------------------------------

        requirements = self._get_value(
            problem,
            "requirements",
            [],
        )

        for requirement in (
            requirements or []
        ):

            self._check_criterion(
                solution,
                str(requirement),
                result,
            )

        result.checks_performed.append(
            "problem_requirements"
        )

        # --------------------------------------------------------------------
        # PROBLEM CONSTRAINTS
        # --------------------------------------------------------------------

        constraints = self._get_value(
            problem,
            "constraints",
            [],
        )

        for constraint in (
            constraints or []
        ):

            self._check_constraint(
                solution,
                str(constraint),
                result,
            )

        result.checks_performed.append(
            "problem_constraints"
        )

        # --------------------------------------------------------------------
        # AI
        # --------------------------------------------------------------------

        if self.ai_callback:

            ai_result = self._run_ai_critic(
                CritiqueRequest(
                    subject=solution,
                    context={
                        "problem": problem
                    },
                    criteria=[
                        str(item)
                        for item in (
                            requirements
                            or []
                        )
                    ],
                    constraints=[
                        str(item)
                        for item in (
                            constraints
                            or []
                        )
                    ],
                    strict=(
                        self.strict
                        if strict is None
                        else strict
                    ),
                )
            )

            self._merge_ai_result(
                result,
                ai_result,
            )

        return self._finalize(
            result
        )

    # ========================================================================
    # ANSWER CRITIQUE
    # ========================================================================

    def critique_answer(
        self,
        question: str,
        answer: str,
        *,
        expected_points: Optional[
            Sequence[str]
        ] = None,
        strict: Optional[bool] = None,
    ) -> CritiqueResult:
        """
        Critique an answer to a question.
        """

        result = self._create_result()

        result.checks_performed.extend(
            [
                "question_presence",
                "answer_presence",
                "answer_relevance",
                "expected_points",
            ]
        )

        if not question.strip():

            result.issues.append(
                CritiqueIssue(
                    message="Question is empty.",
                    severity=CritiqueSeverity.ERROR,
                    category=CritiqueCategory.RELEVANCE,
                    code="EMPTY_QUESTION",
                )
            )

        if not answer.strip():

            result.issues.append(
                CritiqueIssue(
                    message="Answer is empty.",
                    severity=CritiqueSeverity.CRITICAL,
                    category=CritiqueCategory.COMPLETENESS,
                    code="EMPTY_ANSWER",
                )
            )

            return self._finalize(
                result
            )

        # Basic relevance.
        question_words = self._keywords(
            question
        )

        answer_text = answer.lower()

        if question_words:

            matches = sum(
                word in answer_text
                for word in question_words
            )

            relevance = (
                matches
                / len(question_words)
            )

            if relevance < 0.15:

                result.issues.append(
                    CritiqueIssue(
                        message=(
                            "The answer may not be "
                            "relevant to the question."
                        ),
                        severity=CritiqueSeverity.WARNING,
                        category=CritiqueCategory.RELEVANCE,
                        code="LOW_RELEVANCE",
                    )
                )

        # Expected points.
        for point in (
            expected_points or []
        ):

            if not self._criterion_satisfied(
                answer,
                point,
            ):

                result.issues.append(
                    CritiqueIssue(
                        message=(
                            f"Expected answer point "
                            f"may be missing: {point}"
                        ),
                        severity=CritiqueSeverity.WARNING,
                        category=CritiqueCategory.COMPLETENESS,
                        code="EXPECTED_POINT_MISSING",
                        evidence=point,
                    )
                )

        # AI.
        if self.ai_callback:

            ai_result = self._run_ai_critic(
                CritiqueRequest(
                    subject=answer,
                    goal=question,
                    criteria=list(
                        expected_points
                        or []
                    ),
                    strict=(
                        self.strict
                        if strict is None
                        else strict
                    ),
                    context={
                        "question": question
                    },
                )
            )

            self._merge_ai_result(
                result,
                ai_result,
            )

        return self._finalize(
            result
        )

    # ========================================================================
    # SAFETY CRITIQUE
    # ========================================================================

    def critique_safety(
        self,
        action: Any,
        *,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CritiqueResult:
        """
        Perform conservative safety checks before execution.

        This does not replace the dedicated RENIX security subsystem.
        """

        result = self._create_result()

        text = self._to_text(
            action
        ).lower()

        result.checks_performed.append(
            "dangerous_action_scan"
        )

        dangerous_patterns = [
            (
                r"\bdelete all\b",
                "Bulk deletion request detected.",
            ),
            (
                r"\bformat (the )?drive\b",
                "Drive formatting request detected.",
            ),
            (
                r"\brm\s+-rf\b",
                "Recursive force deletion command detected.",
            ),
            (
                r"\bshutdown\b",
                "System shutdown operation detected.",
            ),
            (
                r"\bdisable security\b",
                "Security-disabling operation detected.",
            ),
            (
                r"\bdisable antivirus\b",
                "Antivirus-disabling operation detected.",
            ),
            (
                r"\bdisable firewall\b",
                "Firewall-disabling operation detected.",
            ),
        ]

        for pattern, message in dangerous_patterns:

            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):

                result.issues.append(
                    CritiqueIssue(
                        message=message,
                        severity=CritiqueSeverity.WARNING,
                        category=CritiqueCategory.SAFETY,
                        code="HIGH_RISK_ACTION",
                        suggestion=(
                            "Require explicit confirmation "
                            "and security authorization "
                            "before execution."
                        ),
                    )
                )

        result.metadata[
            "context_present"
        ] = bool(context)

        return self._finalize(
            result
        )

    # ========================================================================
    # AI CRITIC
    # ========================================================================

    def _run_ai_critic(
        self,
        request: CritiqueRequest,
    ) -> Any:
        """Run the optional advanced AI critic."""

        try:

            return self.ai_callback(
                subject=request.subject,
                goal=request.goal,
                expected=request.expected,
                context=request.context,
                criteria=request.criteria,
                constraints=request.constraints,
                strict=request.strict,
            )

        except Exception as exc:

            return {
                "status": "inconclusive",
                "confidence": 0.0,
                "summary": (
                    "AI critique could not be completed."
                ),
                "issues": [
                    {
                        "message": str(exc),
                        "severity": "warning",
                        "category": "unknown",
                        "code": "AI_CRITIC_ERROR",
                    }
                ],
            }

    def _merge_ai_result(
        self,
        result: CritiqueResult,
        ai_result: Any,
    ) -> None:
        """Merge AI critique into deterministic critique."""

        if isinstance(
            ai_result,
            CritiqueResult,
        ):

            result.issues.extend(
                ai_result.issues
            )

            result.strengths.extend(
                ai_result.strengths
            )

            result.weaknesses.extend(
                ai_result.weaknesses
            )

            result.recommendations.extend(
                ai_result.recommendations
            )

            result.checks_performed.extend(
                ai_result.checks_performed
            )

            result.metadata[
                "ai_summary"
            ] = ai_result.summary

            result.metadata[
                "ai_score"
            ] = ai_result.score

            result.metadata[
                "ai_confidence"
            ] = ai_result.confidence

            return

        if not isinstance(
            ai_result,
            dict,
        ):

            result.metadata[
                "ai_result"
            ] = ai_result

            return

        result.metadata[
            "ai_result"
        ] = ai_result

        # Score.
        if "score" in ai_result:

            try:

                result.metadata[
                    "ai_score"
                ] = float(
                    ai_result[
                        "score"
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

        # Confidence.
        if "confidence" in ai_result:

            try:

                result.confidence = max(
                    result.confidence,
                    min(
                        1.0,
                        float(
                            ai_result[
                                "confidence"
                            ]
                        ),
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

        # Summary.
        summary = ai_result.get(
            "summary"
        )

        if summary:
            result.metadata[
                "ai_summary"
            ] = summary

        # Strengths.
        for strength in ai_result.get(
            "strengths",
            [],
        ):

            if strength not in result.strengths:

                result.strengths.append(
                    str(strength)
                )

        # Weaknesses.
        for weakness in ai_result.get(
            "weaknesses",
            [],
        ):

            if weakness not in result.weaknesses:

                result.weaknesses.append(
                    str(weakness)
                )

        # Recommendations.
        for recommendation in ai_result.get(
            "recommendations",
            [],
        ):

            if (
                recommendation
                not in result.recommendations
            ):

                result.recommendations.append(
                    str(recommendation)
                )

        # Issues.
        for raw_issue in ai_result.get(
            "issues",
            [],
        ):

            if isinstance(
                raw_issue,
                CritiqueIssue,
            ):

                result.issues.append(
                    raw_issue
                )

                continue

            if not isinstance(
                raw_issue,
                dict,
            ):
                continue

            severity = self._parse_severity(
                raw_issue.get(
                    "severity",
                    "warning",
                )
            )

            category = self._parse_category(
                raw_issue.get(
                    "category",
                    "unknown",
                )
            )

            result.issues.append(
                CritiqueIssue(
                    message=str(
                        raw_issue.get(
                            "message",
                            "AI critique issue.",
                        )
                    ),
                    severity=severity,
                    category=category,
                    code=raw_issue.get(
                        "code"
                    ),
                    location=raw_issue.get(
                        "location"
                    ),
                    evidence=raw_issue.get(
                        "evidence"
                    ),
                    suggestion=raw_issue.get(
                        "suggestion"
                    ),
                    metadata=dict(
                        raw_issue.get(
                            "metadata",
                            {},
                        )
                    ),
                )
            )

    # ========================================================================
    # DETERMINISTIC CHECKS
    # ========================================================================

    def _check_existence(
        self,
        subject: Any,
        result: CritiqueResult,
    ) -> None:
        """Check whether the subject exists."""

        result.checks_performed.append(
            "subject_existence"
        )

        if self._is_empty(
            subject
        ):

            result.issues.append(
                CritiqueIssue(
                    message="Nothing was provided to critique.",
                    severity=CritiqueSeverity.CRITICAL,
                    category=CritiqueCategory.COMPLETENESS,
                    code="EMPTY_SUBJECT",
                    suggestion=(
                        "Provide a generated result, "
                        "plan, answer or solution."
                    ),
                )
            )

    def _check_structure(
        self,
        subject: Any,
        result: CritiqueResult,
    ) -> None:
        """Check basic structure."""

        result.checks_performed.append(
            "basic_structure"
        )

        if isinstance(
            subject,
            dict,
        ):

            if not subject:

                result.issues.append(
                    CritiqueIssue(
                        message="The object is empty.",
                        severity=CritiqueSeverity.WARNING,
                        category=CritiqueCategory.COMPLETENESS,
                        code="EMPTY_OBJECT",
                    )
                )

        elif isinstance(
            subject,
            (list, tuple, set),
        ):

            if not subject:

                result.issues.append(
                    CritiqueIssue(
                        message="The collection is empty.",
                        severity=CritiqueSeverity.WARNING,
                        category=CritiqueCategory.COMPLETENESS,
                        code="EMPTY_COLLECTION",
                    )
                )

        elif isinstance(
            subject,
            str,
        ):

            if not subject.strip():

                result.issues.append(
                    CritiqueIssue(
                        message="Text contains no content.",
                        severity=CritiqueSeverity.ERROR,
                        category=CritiqueCategory.COMPLETENESS,
                        code="BLANK_TEXT",
                    )
                )

    def _check_goal_alignment(
        self,
        subject: Any,
        goal: str,
        result: CritiqueResult,
    ) -> None:
        """Check basic alignment between output and goal."""

        result.checks_performed.append(
            "goal_alignment"
        )

        if not goal.strip():
            return

        subject_text = self._to_text(
            subject
        ).lower()

        goal_words = self._keywords(
            goal
        )

        if not goal_words:
            return

        matches = sum(
            word in subject_text
            for word in goal_words
        )

        ratio = (
            matches
            / len(goal_words)
        )

        if ratio < 0.20:

            result.issues.append(
                CritiqueIssue(
                    message=(
                        "The result has weak visible "
                        "alignment with the requested goal."
                    ),
                    severity=CritiqueSeverity.WARNING,
                    category=CritiqueCategory.RELEVANCE,
                    code="WEAK_GOAL_ALIGNMENT",
                    evidence={
                        "goal": goal,
                        "matched_keywords": matches,
                        "goal_keywords": len(
                            goal_words
                        ),
                    },
                    suggestion=(
                        "Review whether the output "
                        "actually solves the requested goal."
                    ),
                )
            )

    def _check_expected(
        self,
        subject: Any,
        expected: Any,
        result: CritiqueResult,
    ) -> None:
        """Compare a result against an expected value."""

        result.checks_performed.append(
            "expected_value"
        )

        if subject == expected:
            return

        if (
            isinstance(subject, str)
            and isinstance(expected, str)
            and subject.strip().lower()
            == expected.strip().lower()
        ):
            return

        result.issues.append(
            CritiqueIssue(
                message=(
                    "The result does not exactly "
                    "match the expected value."
                ),
                severity=CritiqueSeverity.ERROR,
                category=CritiqueCategory.CORRECTNESS,
                code="EXPECTED_VALUE_MISMATCH",
                evidence={
                    "actual": subject,
                    "expected": expected,
                },
            )
        )

    def _check_criterion(
        self,
        subject: Any,
        criterion: str,
        result: CritiqueResult,
    ) -> None:
        """Check whether a criterion is visibly addressed."""

        result.checks_performed.append(
            f"criterion:{criterion}"
        )

        if not criterion.strip():
            return

        text = self._to_text(
            subject
        ).lower()

        criterion_lower = (
            criterion.strip().lower()
        )

        if criterion_lower in text:
            return

        words = self._keywords(
            criterion
        )

        if not words:
            return

        matches = sum(
            word in text
            for word in words
        )

        ratio = (
            matches
            / len(words)
        )

        if ratio < 0.50:

            result.issues.append(
                CritiqueIssue(
                    message=(
                        f"The criterion may not be "
                        f"adequately addressed: {criterion}"
                    ),
                    severity=CritiqueSeverity.WARNING,
                    category=CritiqueCategory.COMPLETENESS,
                    code="CRITERION_NOT_EVIDENT",
                    evidence={
                        "criterion": criterion,
                        "match_ratio": ratio,
                    },
                    suggestion=(
                        "Add an explicit part addressing "
                        "this criterion."
                    ),
                )
            )

    def _check_constraint(
        self,
        subject: Any,
        constraint: str,
        result: CritiqueResult,
    ) -> None:
        """Check obvious constraint violations."""

        result.checks_performed.append(
            f"constraint:{constraint}"
        )

        if not constraint.strip():
            return

        text = self._to_text(
            subject
        ).lower()

        normalized = (
            constraint.strip().lower()
        )

        negative_markers = [
            "must not",
            "do not",
            "don't",
            "cannot",
            "can't",
            "without",
            "avoid",
            "never",
        ]

        if not any(
            marker in normalized
            for marker in negative_markers
        ):
            return

        parts = re.split(
            r"\b(?:must not|do not|don't|cannot|can't|without|avoid|never)\b",
            normalized,
            maxsplit=1,
        )

        if len(parts) != 2:
            return

        forbidden = parts[1].strip()

        if forbidden and forbidden in text:

            result.issues.append(
                CritiqueIssue(
                    message=(
                        f"The output may violate "
                        f"constraint: {constraint}"
                    ),
                    severity=CritiqueSeverity.ERROR,
                    category=CritiqueCategory.FEASIBILITY,
                    code="CONSTRAINT_VIOLATION",
                    evidence=forbidden,
                    suggestion=(
                        "Modify the result so that "
                        "the constraint is respected."
                    ),
                )
            )

    def _check_consistency(
        self,
        subject: Any,
        result: CritiqueResult,
    ) -> None:
        """Check obvious internal contradictions."""

        result.checks_performed.append(
            "internal_consistency"
        )

        if not isinstance(
            subject,
            str,
        ):
            return

        contradictions = [
            (
                r"\btrue\b",
                r"\bfalse\b",
                "true and false",
            ),
            (
                r"\byes\b",
                r"\bno\b",
                "yes and no",
            ),
            (
                r"\benabled\b",
                r"\bdisabled\b",
                "enabled and disabled",
            ),
            (
                r"\bopen\b",
                r"\bclosed\b",
                "open and closed",
            ),
            (
                r"\bsuccessful\b",
                r"\bfailed\b",
                "successful and failed",
            ),
        ]

        for left, right, description in contradictions:

            if (
                re.search(
                    left,
                    subject,
                    re.IGNORECASE,
                )
                and re.search(
                    right,
                    subject,
                    re.IGNORECASE,
                )
            ):

                result.issues.append(
                    CritiqueIssue(
                        message=(
                            "Potential contradiction detected: "
                            f"{description}."
                        ),
                        severity=CritiqueSeverity.WARNING,
                        category=CritiqueCategory.CONSISTENCY,
                        code="POTENTIAL_CONTRADICTION",
                    )
                )

    def _check_quality(
        self,
        subject: Any,
        result: CritiqueResult,
    ) -> None:
        """Check basic output quality."""

        result.checks_performed.append(
            "output_quality"
        )

        text = self._to_text(
            subject
        )

        if not text.strip():
            return

        suspicious_markers = [
            "TODO",
            "FIXME",
            "NOT IMPLEMENTED",
            "UNKNOWN",
            "PLACEHOLDER",
        ]

        for marker in suspicious_markers:

            if marker.lower() in text.lower():

                result.issues.append(
                    CritiqueIssue(
                        message=(
                            f"Output contains the "
                            f"placeholder marker '{marker}'."
                        ),
                        severity=CritiqueSeverity.WARNING,
                        category=CritiqueCategory.COMPLETENESS,
                        code="PLACEHOLDER_OUTPUT",
                        evidence=marker,
                        suggestion=(
                            "Replace placeholder content "
                            "with a completed result."
                        ),
                    )
                )

        error_patterns = [
            r"\btraceback\b",
            r"\bpermission denied\b",
            r"\bcommand failed\b",
            r"\bexception occurred\b",
        ]

        for pattern in error_patterns:

            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):

                result.issues.append(
                    CritiqueIssue(
                        message=(
                            "Output appears to contain "
                            "an execution error."
                        ),
                        severity=CritiqueSeverity.ERROR,
                        category=CritiqueCategory.EXECUTION,
                        code="ERROR_OUTPUT_DETECTED",
                        evidence=pattern,
                    )
                )

    def _strict_checks(
        self,
        subject: Any,
        result: CritiqueResult,
    ) -> None:
        """Run stricter deterministic checks."""

        result.checks_performed.append(
            "strict_quality"
        )

        text = self._to_text(
            subject
        ).strip()

        if len(text) < 10:

            result.issues.append(
                CritiqueIssue(
                    message=(
                        "Strict critique considers the "
                        "result too short for reliable evaluation."
                    ),
                    severity=CritiqueSeverity.WARNING,
                    category=CritiqueCategory.COMPLETENESS,
                    code="VERY_SHORT_OUTPUT",
                )
            )

        repeated = self._has_excessive_repetition(
            text
        )

        if repeated:

            result.issues.append(
                CritiqueIssue(
                    message=(
                        "Output contains excessive "
                        "repetition."
                    ),
                    severity=CritiqueSeverity.WARNING,
                    category=CritiqueCategory.CLARITY,
                    code="EXCESSIVE_REPETITION",
                )
            )

    # ========================================================================
    # FINALIZATION
    # ========================================================================

    def _create_result(
        self,
    ) -> CritiqueResult:
        """Create a blank result."""

        return CritiqueResult(
            status=CritiqueStatus.INCONCLUSIVE
        )

    def _finalize(
        self,
        result: CritiqueResult,
    ) -> CritiqueResult:
        """Calculate score, confidence and final status."""

        critical_count = len(
            result.critical_issues
        )

        error_count = len(
            result.errors
        )

        warning_count = len(
            result.warnings
        )

        # --------------------------------------------------------------------
        # SCORE
        # --------------------------------------------------------------------

        penalty = (
            critical_count * 0.45
            + error_count * 0.25
            + warning_count * 0.06
        )

        result.score = max(
            0.0,
            min(
                1.0,
                1.0 - penalty,
            ),
        )

        # --------------------------------------------------------------------
        # STATUS
        # --------------------------------------------------------------------

        if critical_count:

            result.status = (
                CritiqueStatus.REJECT
            )

        elif error_count >= 2:

            result.status = (
                CritiqueStatus.REJECT
            )

        elif error_count:

            result.status = (
                CritiqueStatus.REVISE
            )

        elif warning_count >= 3:

            result.status = (
                CritiqueStatus.ACCEPT_WITH_WARNINGS
            )

        elif warning_count:

            result.status = (
                CritiqueStatus.ACCEPT_WITH_WARNINGS
            )

        else:

            result.status = (
                CritiqueStatus.ACCEPT
            )

        # --------------------------------------------------------------------
        # STRENGTHS
        # --------------------------------------------------------------------

        if not result.issues:

            result.strengths.append(
                "No deterministic quality problems were detected."
            )

        else:

            if (
                "goal_alignment"
                in result.checks_performed
                and not any(
                    issue.code
                    == "WEAK_GOAL_ALIGNMENT"
                    for issue in result.issues
                )
            ):

                result.strengths.append(
                    "The result appears aligned with the requested goal."
                )

            if (
                "internal_consistency"
                in result.checks_performed
                and not any(
                    issue.category
                    == CritiqueCategory.CONSISTENCY
                    for issue in result.issues
                )
            ):

                result.strengths.append(
                    "No obvious internal contradictions were detected."
                )

            if (
                "dependency_integrity"
                in result.checks_performed
                and not any(
                    issue.category
                    == CritiqueCategory.DEPENDENCY
                    for issue in result.issues
                )
            ):

                result.strengths.append(
                    "Task dependencies appear structurally valid."
                )

        # --------------------------------------------------------------------
        # WEAKNESSES
        # --------------------------------------------------------------------

        for issue in result.issues:

            weakness = issue.message

            if weakness not in result.weaknesses:

                result.weaknesses.append(
                    weakness
                )

            if issue.suggestion:

                if (
                    issue.suggestion
                    not in result.recommendations
                ):

                    result.recommendations.append(
                        issue.suggestion
                    )

        # --------------------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------------------

        result.confidence = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    critical_count * 0.30
                    + error_count * 0.15
                    + warning_count * 0.03
                ),
            ),
        )

        # --------------------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------------------

        if result.status == CritiqueStatus.ACCEPT:

            result.summary = (
                "The result passed the available critique checks."
            )

        elif (
            result.status
            == CritiqueStatus.ACCEPT_WITH_WARNINGS
        ):

            result.summary = (
                "The result is generally acceptable, "
                "but warnings should be reviewed."
            )

        elif result.status == CritiqueStatus.REVISE:

            result.summary = (
                "The result should be revised "
                "before being accepted."
            )

        elif result.status == CritiqueStatus.REJECT:

            result.summary = (
                "The result should not be accepted "
                "in its current form."
            )

        else:

            result.summary = (
                "The critique result is inconclusive."
            )

        return result

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _get_value(
        obj: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        """Get a value from a dictionary or object."""

        if obj is None:
            return default

        if isinstance(
            obj,
            dict,
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
    def _is_empty(
        value: Any,
    ) -> bool:
        """Check whether a value is empty."""

        if value is None:
            return True

        if isinstance(
            value,
            str,
        ):
            return not value.strip()

        try:
            return len(value) == 0
        except (
            TypeError,
            AttributeError,
        ):
            return False

    @staticmethod
    def _to_text(
        value: Any,
    ) -> str:
        """Convert arbitrary data into searchable text."""

        if value is None:
            return ""

        if isinstance(
            value,
            str,
        ):
            return value

        if isinstance(
            value,
            dict,
        ):

            parts = []

            for key, item in value.items():

                parts.append(
                    Critic._to_text(
                        key
                    )
                )

                parts.append(
                    Critic._to_text(
                        item
                    )
                )

            return " ".join(
                parts
            )

        if isinstance(
            value,
            (list, tuple, set),
        ):

            return " ".join(
                Critic._to_text(
                    item
                )
                for item in value
            )

        return str(value)

    @staticmethod
    def _keywords(
        text: str,
    ) -> List[str]:
        """Extract useful keywords."""

        stop_words = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "into",
            "have",
            "has",
            "will",
            "should",
            "could",
            "would",
            "your",
            "you",
            "are",
            "was",
            "were",
            "can",
            "about",
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
            "does",
            "not",
        }

        words = re.findall(
            r"[A-Za-z0-9_]+",
            text.lower(),
        )

        return [
            word
            for word in words
            if len(word) >= 4
            and word not in stop_words
        ]

    @staticmethod
    def _criterion_satisfied(
        subject: Any,
        criterion: str,
    ) -> bool:
        """Determine whether a criterion is visibly satisfied."""

        text = Critic._to_text(
            subject
        ).lower()

        criterion_lower = (
            criterion.strip().lower()
        )

        if not criterion_lower:
            return True

        if criterion_lower in text:
            return True

        words = Critic._keywords(
            criterion
        )

        if not words:
            return True

        matches = sum(
            word in text
            for word in words
        )

        return (
            matches / len(words)
            >= 0.50
        )

    @staticmethod
    def _has_cycle(
        dependency_map: Dict[
            str,
            List[str],
        ],
    ) -> bool:
        """Detect cycles in a dependency graph."""

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(
            node: str,
        ) -> bool:

            if node in visiting:
                return True

            if node in visited:
                return False

            visiting.add(node)

            for dependency in dependency_map.get(
                node,
                [],
            ):

                if dependency in dependency_map:

                    if visit(
                        dependency
                    ):
                        return True

            visiting.remove(node)

            visited.add(node)

            return False

        for node in dependency_map:

            if visit(node):
                return True

        return False

    @staticmethod
    def _has_excessive_repetition(
        text: str,
    ) -> bool:
        """Detect obvious repeated phrases."""

        words = re.findall(
            r"\b\w+\b",
            text.lower(),
        )

        if len(words) < 20:
            return False

        counts: Dict[str, int] = {}

        for word in words:

            counts[word] = (
                counts.get(
                    word,
                    0,
                )
                + 1
            )

        highest = max(
            counts.values()
        )

        return (
            highest
            / len(words)
            > 0.25
        )

    @staticmethod
    def _parse_severity(
        value: Any,
    ) -> CritiqueSeverity:
        """Convert an arbitrary severity value."""

        if isinstance(
            value,
            CritiqueSeverity,
        ):
            return value

        try:

            return CritiqueSeverity(
                str(value).lower()
            )

        except ValueError:

            return CritiqueSeverity.WARNING

    @staticmethod
    def _parse_category(
        value: Any,
    ) -> CritiqueCategory:
        """Convert an arbitrary category value."""

        if isinstance(
            value,
            CritiqueCategory,
        ):
            return value

        try:

            return CritiqueCategory(
                str(value).lower()
            )

        except ValueError:

            return CritiqueCategory.UNKNOWN


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def critique(
    subject: Any,
    *,
    goal: Optional[str] = None,
    expected: Any = None,
    criteria: Optional[
        Sequence[str]
    ] = None,
    constraints: Optional[
        Sequence[str]
    ] = None,
    strict: bool = False,
) -> CritiqueResult:
    """Convenience function for general critique."""

    critic = Critic(
        strict=strict
    )

    return critic.critique(
        subject,
        goal=goal,
        expected=expected,
        criteria=criteria,
        constraints=constraints,
        strict=strict,
    )


def critique_plan(
    plan: Any,
    *,
    goal: Optional[str] = None,
    constraints: Optional[
        Sequence[str]
    ] = None,
    strict: bool = False,
) -> CritiqueResult:
    """Convenience function for plan critique."""

    critic = Critic(
        strict=strict
    )

    return critic.critique_plan(
        plan,
        goal=goal,
        constraints=constraints,
        strict=strict,
    )


def critique_solution(
    problem: Any,
    solution: Any,
    *,
    strict: bool = False,
) -> CritiqueResult:
    """Convenience function for solution critique."""

    critic = Critic(
        strict=strict
    )

    return critic.critique_solution(
        problem,
        solution,
        strict=strict,
    )


def critique_answer(
    question: str,
    answer: str,
    *,
    expected_points: Optional[
        Sequence[str]
    ] = None,
    strict: bool = False,
) -> CritiqueResult:
    """Convenience function for answer critique."""

    critic = Critic(
        strict=strict
    )

    return critic.critique_answer(
        question,
        answer,
        expected_points=expected_points,
        strict=strict,
    )


def critique_safety(
    action: Any,
    *,
    context: Optional[
        Dict[str, Any]
    ] = None,
) -> CritiqueResult:
    """Convenience function for safety critique."""

    critic = Critic()

    return critic.critique_safety(
        action,
        context=context,
    )


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "CritiqueStatus",
    "CritiqueSeverity",
    "CritiqueCategory",
    "CritiqueIssue",
    "CritiqueResult",
    "CritiqueRequest",
    "Critic",
    "ReasoningCritic",
    "critique",
    "critique_plan",
    "critique_solution",
    "critique_answer",
    "critique_safety",
]


ReasoningCritic = Critic
