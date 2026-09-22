"""
RENIX AI
Reasoning Verifier

Validates reasoning, plans, tasks, solutions, outputs and claims
before RENIX treats them as successful.

This module is intentionally independent from the LLM provider.
An external AI verifier can be connected through callbacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


# ============================================================================
# ENUMS
# ============================================================================


class VerificationStatus(str, Enum):
    """Overall verification status."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"


class VerificationLevel(str, Enum):
    """Verification strictness."""

    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


class IssueSeverity(str, Enum):
    """Severity of a verification issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class VerificationIssue:
    """One issue discovered during verification."""

    message: str

    severity: IssueSeverity = IssueSeverity.WARNING

    code: Optional[str] = None

    field: Optional[str] = None

    evidence: Any = None

    suggestion: Optional[str] = None

    metadata: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the issue."""

        return {
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "field": self.field,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "metadata": dict(self.metadata),
        }


@dataclass
class VerificationResult:
    """Result of a verification operation."""

    status: VerificationStatus

    confidence: float = 0.0

    summary: str = ""

    issues: List[VerificationIssue] = dataclass_field(
        default_factory=list
    )

    checks_performed: List[str] = dataclass_field(
        default_factory=list
    )

    evidence: List[Any] = dataclass_field(
        default_factory=list
    )

    metadata: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def passed(self) -> bool:
        """Whether verification passed."""

        return self.status == VerificationStatus.PASSED

    @property
    def failed(self) -> bool:
        """Whether verification failed."""

        return self.status == VerificationStatus.FAILED

    @property
    def warnings(self) -> List[VerificationIssue]:
        """Return warning-level issues."""

        return [
            issue
            for issue in self.issues
            if issue.severity
            == IssueSeverity.WARNING
        ]

    @property
    def errors(self) -> List[VerificationIssue]:
        """Return error-level issues."""

        return [
            issue
            for issue in self.issues
            if issue.severity
            in {
                IssueSeverity.ERROR,
                IssueSeverity.CRITICAL,
            }
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize verification result."""

        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "checks_performed": list(
                self.checks_performed
            ),
            "evidence": list(
                self.evidence
            ),
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class VerificationRequest:
    """Structured verification request."""

    subject: Any

    expected: Any = None

    criteria: List[str] = dataclass_field(
        default_factory=list
    )

    context: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )

    level: VerificationLevel = (
        VerificationLevel.STANDARD
    )

    metadata: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )


# ============================================================================
# VERIFIER
# ============================================================================


class Verifier:
    """
    RENIX verification engine.

    Performs deterministic checks and optionally delegates advanced
    verification to a callback connected to RENIX's reasoning/LLM layer.
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

        self.default_level = (
            VerificationLevel.STRICT
            if strict
            else VerificationLevel.STANDARD
        )

    # ========================================================================
    # GENERAL VERIFICATION
    # ========================================================================

    def verify(
        self,
        subject: Any,
        *,
        expected: Any = None,
        criteria: Optional[
            Sequence[str]
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
        level: Optional[
            VerificationLevel
        ] = None,
    ) -> VerificationResult:
        """
        Verify a general subject.
        """

        request = VerificationRequest(
            subject=subject,
            expected=expected,
            criteria=list(
                criteria or []
            ),
            context=dict(
                context or {}
            ),
            level=(
                level
                or self.default_level
            ),
        )

        result = self._base_result()

        # --------------------------------------------------------------
        # EMPTY SUBJECT
        # --------------------------------------------------------------

        if self._is_empty(subject):

            result.issues.append(
                VerificationIssue(
                    message=(
                        "The subject being verified "
                        "is empty."
                    ),
                    severity=IssueSeverity.ERROR,
                    code="EMPTY_SUBJECT",
                )
            )

            result.checks_performed.append(
                "non_empty_subject"
            )

            return self._finalize(
                result
            )

        result.checks_performed.append(
            "non_empty_subject"
        )

        # --------------------------------------------------------------
        # EXPECTED VALUE
        # --------------------------------------------------------------

        if expected is not None:

            result.checks_performed.append(
                "expected_value"
            )

            if not self._matches_expected(
                subject,
                expected,
            ):
                result.issues.append(
                    VerificationIssue(
                        message=(
                            "The result does not "
                            "match the expected value."
                        ),
                        severity=IssueSeverity.ERROR,
                        code="EXPECTED_MISMATCH",
                        evidence={
                            "actual": subject,
                            "expected": expected,
                        },
                    )
                )

        # --------------------------------------------------------------
        # CRITERIA
        # --------------------------------------------------------------

        for criterion in request.criteria:

            result.checks_performed.append(
                f"criterion:{criterion}"
            )

            if not self._criterion_satisfied(
                subject,
                criterion,
            ):
                result.issues.append(
                    VerificationIssue(
                        message=(
                            f"Criterion not satisfied: "
                            f"{criterion}"
                        ),
                        severity=IssueSeverity.ERROR,
                        code="CRITERION_FAILED",
                        field=criterion,
                    )
                )

        # --------------------------------------------------------------
        # STRUCTURE
        # --------------------------------------------------------------

        self._verify_structure(
            subject,
            result,
        )

        # --------------------------------------------------------------
        # ADVANCED AI VERIFICATION
        # --------------------------------------------------------------

        if self.ai_callback:

            ai_result = self._run_ai_verification(
                request
            )

            self._merge_ai_result(
                result,
                ai_result,
            )

        # --------------------------------------------------------------
        # STRICT MODE
        # --------------------------------------------------------------

        if request.level == VerificationLevel.STRICT:

            self._run_strict_checks(
                subject,
                result,
            )

        return self._finalize(
            result
        )

    # ========================================================================
    # SOLUTION VERIFICATION
    # ========================================================================

    def verify_solution(
        self,
        problem: Any,
        solution: Any,
        *,
        context: Optional[
            Dict[str, Any]
        ] = None,
        level: Optional[
            VerificationLevel
        ] = None,
    ) -> VerificationResult:
        """
        Verify a proposed solution against a problem.
        """

        result = self._base_result()

        result.checks_performed.extend(
            [
                "solution_exists",
                "solution_description",
                "solution_steps",
                "problem_requirements",
                "problem_constraints",
            ]
        )

        if solution is None:

            result.issues.append(
                VerificationIssue(
                    message="Solution is missing.",
                    severity=IssueSeverity.CRITICAL,
                    code="MISSING_SOLUTION",
                )
            )

            return self._finalize(
                result
            )

        description = self._get_value(
            solution,
            "description",
            "",
        )

        if not str(description).strip():

            result.issues.append(
                VerificationIssue(
                    message=(
                        "Solution does not contain "
                        "a usable description."
                    ),
                    severity=IssueSeverity.ERROR,
                    code="EMPTY_SOLUTION_DESCRIPTION",
                )
            )

        steps = self._get_value(
            solution,
            "steps",
            [],
        )

        if not steps:

            result.issues.append(
                VerificationIssue(
                    message=(
                        "Solution does not contain "
                        "execution steps."
                    ),
                    severity=IssueSeverity.WARNING,
                    code="NO_SOLUTION_STEPS",
                )
            )

        requirements = self._get_value(
            problem,
            "requirements",
            [],
        )

        for requirement in requirements:

            if not self._criterion_satisfied(
                solution,
                requirement,
            ):
                result.issues.append(
                    VerificationIssue(
                        message=(
                            f"Requirement may not be "
                            f"satisfied: {requirement}"
                        ),
                        severity=IssueSeverity.WARNING,
                        code="REQUIREMENT_UNVERIFIED",
                        field=requirement,
                    )
                )

        constraints = self._get_value(
            problem,
            "constraints",
            [],
        )

        for constraint in constraints:

            if self._constraint_violated(
                solution,
                constraint,
            ):
                result.issues.append(
                    VerificationIssue(
                        message=(
                            f"Solution may violate "
                            f"constraint: {constraint}"
                        ),
                        severity=IssueSeverity.ERROR,
                        code="CONSTRAINT_VIOLATION",
                        field=constraint,
                    )
                )

        if self.ai_callback:

            ai_result = self._run_ai_verification(
                VerificationRequest(
                    subject=solution,
                    expected=None,
                    criteria=list(
                        requirements
                    ),
                    context={
                        "problem": problem,
                        **(
                            context
                            or {}
                        ),
                    },
                    level=(
                        level
                        or self.default_level
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
    # TASK VERIFICATION
    # ========================================================================

    def verify_task(
        self,
        task: Any,
        *,
        graph: Any = None,
        level: Optional[
            VerificationLevel
        ] = None,
    ) -> VerificationResult:
        """
        Verify a decomposed task before execution.
        """

        result = self._base_result()

        result.checks_performed.extend(
            [
                "task_id",
                "task_title",
                "task_status",
                "dependencies",
            ]
        )

        task_id = self._get_value(
            task,
            "id",
            None,
        )

        if not task_id:

            result.issues.append(
                VerificationIssue(
                    message="Task has no ID.",
                    severity=IssueSeverity.ERROR,
                    code="TASK_ID_MISSING",
                )
            )

        title = self._get_value(
            task,
            "title",
            "",
        )

        if not str(title).strip():

            result.issues.append(
                VerificationIssue(
                    message="Task has no title.",
                    severity=IssueSeverity.ERROR,
                    code="TASK_TITLE_MISSING",
                )
            )

        dependencies = self._get_value(
            task,
            "dependencies",
            [],
        )

        if graph is not None:

            for dependency_id in dependencies:

                dependency = self._get_graph_task(
                    graph,
                    dependency_id,
                )

                if dependency is None:

                    result.issues.append(
                        VerificationIssue(
                            message=(
                                "Task references a "
                                f"missing dependency: "
                                f"{dependency_id}"
                            ),
                            severity=IssueSeverity.ERROR,
                            code="MISSING_DEPENDENCY",
                        )
                    )

        status = self._get_value(
            task,
            "status",
            None,
        )

        if status is not None:

            status_value = getattr(
                status,
                "value",
                str(status),
            )

            if status_value == "failed":

                result.issues.append(
                    VerificationIssue(
                        message="Task is already marked as failed.",
                        severity=IssueSeverity.WARNING,
                        code="TASK_ALREADY_FAILED",
                    )
                )

        if level == VerificationLevel.STRICT:

            description = self._get_value(
                task,
                "description",
                "",
            )

            if not str(description).strip():

                result.issues.append(
                    VerificationIssue(
                        message=(
                            "Strict verification requires "
                            "a task description."
                        ),
                        severity=IssueSeverity.ERROR,
                        code="TASK_DESCRIPTION_MISSING",
                    )
                )

        return self._finalize(
            result
        )

    # ========================================================================
    # PLAN VERIFICATION
    # ========================================================================

    def verify_plan(
        self,
        plan: Any,
        *,
        goal: Optional[str] = None,
        level: Optional[
            VerificationLevel
        ] = None,
    ) -> VerificationResult:
        """
        Verify an execution plan or task graph.
        """

        result = self._base_result()

        tasks = self._get_value(
            plan,
            "tasks",
            [],
        )

        result.checks_performed.append(
            "plan_has_tasks"
        )

        if not tasks:

            result.issues.append(
                VerificationIssue(
                    message="Plan contains no tasks.",
                    severity=IssueSeverity.ERROR,
                    code="EMPTY_PLAN",
                )
            )

            return self._finalize(
                result
            )

        task_ids = set()

        for task in tasks:

            task_id = self._get_value(
                task,
                "id",
                None,
            )

            if not task_id:
                continue

            if task_id in task_ids:

                result.issues.append(
                    VerificationIssue(
                        message=(
                            f"Duplicate task ID: "
                            f"{task_id}"
                        ),
                        severity=IssueSeverity.ERROR,
                        code="DUPLICATE_TASK_ID",
                    )
                )

            task_ids.add(task_id)

        result.checks_performed.append(
            "unique_task_ids"
        )

        # Dependency validation.
        for task in tasks:

            task_id = self._get_value(
                task,
                "id",
                "",
            )

            dependencies = self._get_value(
                task,
                "dependencies",
                [],
            )

            for dependency in dependencies:

                if dependency not in task_ids:

                    result.issues.append(
                        VerificationIssue(
                            message=(
                                f"Task '{task_id}' "
                                f"depends on missing "
                                f"task '{dependency}'."
                            ),
                            severity=IssueSeverity.ERROR,
                            code="PLAN_MISSING_DEPENDENCY",
                        )
                    )

                if dependency == task_id:

                    result.issues.append(
                        VerificationIssue(
                            message=(
                                f"Task '{task_id}' "
                                "depends on itself."
                            ),
                            severity=IssueSeverity.ERROR,
                            code="SELF_DEPENDENCY",
                        )
                    )

        result.checks_performed.append(
            "dependency_integrity"
        )

        # Cycle detection.
        if self._contains_cycle(
            tasks
        ):

            result.issues.append(
                VerificationIssue(
                    message=(
                        "Plan contains a dependency cycle."
                    ),
                    severity=IssueSeverity.CRITICAL,
                    code="DEPENDENCY_CYCLE",
                )
            )

        result.checks_performed.append(
            "cycle_detection"
        )

        # Goal presence.
        if goal:

            result.checks_performed.append(
                "goal_presence"
            )

            if not self._plan_mentions_goal(
                plan,
                goal,
            ):
                result.issues.append(
                    VerificationIssue(
                        message=(
                            "The plan does not clearly "
                            "reference the requested goal."
                        ),
                        severity=IssueSeverity.WARNING,
                        code="GOAL_NOT_EVIDENT",
                    )
                )

        if level == VerificationLevel.STRICT:

            for task in tasks:

                description = self._get_value(
                    task,
                    "description",
                    "",
                )

                if not str(
                    description
                ).strip():

                    result.issues.append(
                        VerificationIssue(
                            message=(
                                "Strict plan verification "
                                "requires every task to "
                                "have a description."
                            ),
                            severity=IssueSeverity.ERROR,
                            code="TASK_DESCRIPTION_MISSING",
                            field=self._get_value(
                                task,
                                "id",
                                None,
                            ),
                        )
                    )

        return self._finalize(
            result
        )

    # ========================================================================
    # OUTPUT VERIFICATION
    # ========================================================================

    def verify_output(
        self,
        output: Any,
        *,
        expected_type: Optional[
            type | tuple[type, ...]
        ] = None,
        required_fields: Optional[
            Iterable[str]
        ] = None,
        forbidden_values: Optional[
            Iterable[Any]
        ] = None,
        level: Optional[
            VerificationLevel
        ] = None,
    ) -> VerificationResult:
        """
        Verify generated output.
        """

        result = self._base_result()

        result.checks_performed.append(
            "output_not_empty"
        )

        if self._is_empty(output):

            result.issues.append(
                VerificationIssue(
                    message="Output is empty.",
                    severity=IssueSeverity.ERROR,
                    code="EMPTY_OUTPUT",
                )
            )

        if expected_type is not None:

            result.checks_performed.append(
                "output_type"
            )

            if not isinstance(
                output,
                expected_type,
            ):
                result.issues.append(
                    VerificationIssue(
                        message=(
                            "Output has an unexpected type."
                        ),
                        severity=IssueSeverity.ERROR,
                        code="OUTPUT_TYPE_MISMATCH",
                        evidence={
                            "actual_type": (
                                type(output).__name__
                            ),
                            "expected_type": (
                                str(expected_type)
                            ),
                        },
                    )
                )

        if required_fields:

            result.checks_performed.append(
                "required_fields"
            )

            for field_name in required_fields:

                if not self._has_field(
                    output,
                    field_name,
                ):
                    result.issues.append(
                        VerificationIssue(
                            message=(
                                f"Required output field "
                                f"is missing: {field_name}"
                            ),
                            severity=IssueSeverity.ERROR,
                            code="MISSING_OUTPUT_FIELD",
                            field=field_name,
                        )
                    )

        if forbidden_values:

            result.checks_performed.append(
                "forbidden_values"
            )

            for value in forbidden_values:

                if self._contains_value(
                    output,
                    value,
                ):
                    result.issues.append(
                        VerificationIssue(
                            message=(
                                "Output contains a "
                                f"forbidden value: {value}"
                            ),
                            severity=IssueSeverity.ERROR,
                            code="FORBIDDEN_OUTPUT_VALUE",
                            evidence=value,
                        )
                    )

        if (
            level == VerificationLevel.STRICT
            or (
                level is None
                and self.default_level
                == VerificationLevel.STRICT
            )
        ):
            self._run_strict_checks(
                output,
                result,
            )

        return self._finalize(
            result
        )

    # ========================================================================
    # CLAIM VERIFICATION
    # ========================================================================

    def verify_claim(
        self,
        claim: str,
        *,
        evidence: Optional[
            Sequence[Any]
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> VerificationResult:
        """
        Verify a textual claim against supplied evidence.

        This does not perform external fact-checking by itself.
        If an AI callback is configured, the callback can perform
        advanced reasoning or external-source validation.
        """

        result = self._base_result()

        result.checks_performed.append(
            "claim_presence"
        )

        if not claim or not claim.strip():

            result.issues.append(
                VerificationIssue(
                    message="Claim is empty.",
                    severity=IssueSeverity.ERROR,
                    code="EMPTY_CLAIM",
                )
            )

            return self._finalize(
                result
            )

        result.checks_performed.append(
            "claim_structure"
        )

        if len(
            claim.strip()
        ) < 3:

            result.issues.append(
                VerificationIssue(
                    message=(
                        "Claim is too short to "
                        "verify meaningfully."
                    ),
                    severity=IssueSeverity.WARNING,
                    code="CLAIM_TOO_SHORT",
                )
            )

        supplied_evidence = list(
            evidence or []
        )

        result.evidence.extend(
            supplied_evidence
        )

        if not supplied_evidence:

            result.issues.append(
                VerificationIssue(
                    message=(
                        "No evidence was supplied "
                        "for the claim."
                    ),
                    severity=IssueSeverity.WARNING,
                    code="NO_EVIDENCE",
                    suggestion=(
                        "Provide reliable evidence or "
                        "connect an external verification "
                        "provider."
                    ),
                )
            )

        if self.ai_callback:

            ai_result = self._run_ai_verification(
                VerificationRequest(
                    subject=claim,
                    expected=None,
                    criteria=[
                        "claim is supported by evidence"
                    ],
                    context={
                        "evidence": supplied_evidence,
                        **(
                            context
                            or {}
                        ),
                    },
                    level=self.default_level,
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
    # CONSISTENCY
    # ========================================================================

    def check_consistency(
        self,
        values: Sequence[Any],
    ) -> VerificationResult:
        """
        Check a collection for obvious contradictory values.
        """

        result = self._base_result()

        result.checks_performed.append(
            "value_consistency"
        )

        normalized = [
            self._normalize_for_comparison(
                value
            )
            for value in values
        ]

        contradictory_pairs = []

        for index, left in enumerate(
            normalized
        ):
            for right in normalized[
                index + 1 :
            ]:

                if self._contradictory(
                    left,
                    right,
                ):
                    contradictory_pairs.append(
                        (
                            values[index],
                            values[
                                index + 1
                            ],
                        )
                    )

        if contradictory_pairs:

            result.issues.append(
                VerificationIssue(
                    message=(
                        "Potential contradictory "
                        "values were detected."
                    ),
                    severity=IssueSeverity.WARNING,
                    code="CONTRADICTION_DETECTED",
                    evidence=contradictory_pairs,
                )
            )

        return self._finalize(
            result
        )

    # ========================================================================
    # AI VERIFICATION
    # ========================================================================

    def _run_ai_verification(
        self,
        request: VerificationRequest,
    ) -> Any:
        """Run optional advanced AI verification."""

        try:

            return self.ai_callback(
                subject=request.subject,
                expected=request.expected,
                criteria=request.criteria,
                context=request.context,
                level=request.level.value,
            )

        except Exception as exc:

            return {
                "status": "inconclusive",
                "confidence": 0.0,
                "summary": (
                    "Advanced AI verification "
                    f"could not be completed: {exc}"
                ),
                "issues": [
                    {
                        "message": str(exc),
                        "severity": "warning",
                        "code": "AI_VERIFIER_ERROR",
                    }
                ],
            }

    def _merge_ai_result(
        self,
        result: VerificationResult,
        ai_result: Any,
    ) -> None:
        """Merge AI verification output into a result."""

        if isinstance(
            ai_result,
            VerificationResult,
        ):

            result.issues.extend(
                ai_result.issues
            )

            result.checks_performed.extend(
                ai_result.checks_performed
            )

            result.evidence.extend(
                ai_result.evidence
            )

            if ai_result.summary:
                result.metadata[
                    "ai_summary"
                ] = ai_result.summary

            result.metadata[
                "ai_status"
            ] = ai_result.status.value

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

        if ai_result.get(
            "summary"
        ):
            result.metadata[
                "ai_summary"
            ] = ai_result[
                "summary"
            ]

        raw_issues = ai_result.get(
            "issues",
            [],
        )

        for raw_issue in raw_issues:

            if isinstance(
                raw_issue,
                VerificationIssue,
            ):
                result.issues.append(
                    raw_issue
                )
                continue

            if isinstance(
                raw_issue,
                dict,
            ):

                severity = self._parse_severity(
                    raw_issue.get(
                        "severity",
                        "warning",
                    )
                )

                result.issues.append(
                    VerificationIssue(
                        message=str(
                            raw_issue.get(
                                "message",
                                "AI verification issue.",
                            )
                        ),
                        severity=severity,
                        code=raw_issue.get(
                            "code"
                        ),
                        field=raw_issue.get(
                            "field"
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

        status = str(
            ai_result.get(
                "status",
                "",
            )
        ).lower()

        if status == "failed":

            result.issues.append(
                VerificationIssue(
                    message=(
                        "Advanced AI verification "
                        "marked the result as failed."
                    ),
                    severity=IssueSeverity.ERROR,
                    code="AI_VERIFICATION_FAILED",
                )
            )

        elif status == "warning":

            result.issues.append(
                VerificationIssue(
                    message=(
                        "Advanced AI verification "
                        "returned warnings."
                    ),
                    severity=IssueSeverity.WARNING,
                    code="AI_VERIFICATION_WARNING",
                )
            )

    # ========================================================================
    # BASIC CHECKS
    # ========================================================================

    def _verify_structure(
        self,
        subject: Any,
        result: VerificationResult,
    ) -> None:
        """Run basic structural checks."""

        if isinstance(
            subject,
            str,
        ):

            result.checks_performed.append(
                "text_structure"
            )

            if not subject.strip():

                result.issues.append(
                    VerificationIssue(
                        message="Text contains no meaningful content.",
                        severity=IssueSeverity.ERROR,
                        code="BLANK_TEXT",
                    )
                )

            if self._looks_like_error_message(
                subject
            ):

                result.issues.append(
                    VerificationIssue(
                        message=(
                            "Text appears to contain "
                            "an error or failure message."
                        ),
                        severity=IssueSeverity.WARNING,
                        code="ERROR_LIKE_OUTPUT",
                    )
                )

        elif isinstance(
            subject,
            dict,
        ):

            result.checks_performed.append(
                "dictionary_structure"
            )

            if not subject:
                result.issues.append(
                    VerificationIssue(
                        message="Dictionary is empty.",
                        severity=IssueSeverity.WARNING,
                        code="EMPTY_DICTIONARY",
                    )
                )

        elif isinstance(
            subject,
            (list, tuple, set),
        ):

            result.checks_performed.append(
                "collection_structure"
            )

            if not subject:
                result.issues.append(
                    VerificationIssue(
                        message="Collection is empty.",
                        severity=IssueSeverity.WARNING,
                        code="EMPTY_COLLECTION",
                    )
                )

    def _run_strict_checks(
        self,
        subject: Any,
        result: VerificationResult,
    ) -> None:
        """Additional conservative checks."""

        result.checks_performed.append(
            "strict_content_check"
        )

        if isinstance(
            subject,
            str,
        ):

            suspicious_patterns = [
                r"\bTODO\b",
                r"\bFIXME\b",
                r"\bUNKNOWN\b",
                r"\bNOT IMPLEMENTED\b",
                r"\bFAILED\b",
            ]

            for pattern in suspicious_patterns:

                if re.search(
                    pattern,
                    subject,
                    re.IGNORECASE,
                ):

                    result.issues.append(
                        VerificationIssue(
                            message=(
                                "Strict verification found "
                                f"suspicious marker: {pattern}"
                            ),
                            severity=IssueSeverity.WARNING,
                            code="SUSPICIOUS_MARKER",
                        )
                    )

    # ========================================================================
    # FINALIZATION
    # ========================================================================

    def _base_result(
        self,
    ) -> VerificationResult:
        """Create a fresh verification result."""

        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            confidence=0.0,
        )

    def _finalize(
        self,
        result: VerificationResult,
    ) -> VerificationResult:
        """Determine final status and confidence."""

        critical = any(
            issue.severity
            == IssueSeverity.CRITICAL
            for issue in result.issues
        )

        errors = any(
            issue.severity
            == IssueSeverity.ERROR
            for issue in result.issues
        )

        warnings = any(
            issue.severity
            == IssueSeverity.WARNING
            for issue in result.issues
        )

        if critical or errors:

            result.status = (
                VerificationStatus.FAILED
            )

        elif warnings:

            result.status = (
                VerificationStatus.WARNING
            )

        else:

            result.status = (
                VerificationStatus.PASSED
            )

        issue_penalty = (
            len(result.errors) * 0.25
            + len(result.warnings) * 0.05
        )

        result.confidence = max(
            0.0,
            min(
                1.0,
                1.0 - issue_penalty,
            ),
        )

        if not result.summary:

            if result.status == VerificationStatus.PASSED:

                result.summary = (
                    "Verification passed successfully."
                )

            elif result.status == VerificationStatus.WARNING:

                result.summary = (
                    "Verification completed with warnings."
                )

            else:

                result.summary = (
                    "Verification failed."
                )

        return result

    # ========================================================================
    # MATCHING HELPERS
    # ========================================================================

    @staticmethod
    def _matches_expected(
        actual: Any,
        expected: Any,
    ) -> bool:
        """Compare actual and expected values."""

        if actual == expected:
            return True

        if (
            isinstance(actual, str)
            and isinstance(expected, str)
        ):

            return (
                actual.strip().lower()
                == expected.strip().lower()
            )

        return False

    @staticmethod
    def _criterion_satisfied(
        subject: Any,
        criterion: str,
    ) -> bool:
        """
        Perform a conservative criterion check.

        This deliberately avoids claiming semantic verification
        when no AI verifier is connected.
        """

        if not criterion:
            return True

        text = Verifier._to_searchable_text(
            subject
        )

        criterion_text = criterion.strip().lower()

        if not criterion_text:
            return True

        # Exact phrase.
        if criterion_text in text.lower():
            return True

        # Basic keyword check.
        words = [
            word
            for word in re.findall(
                r"[A-Za-z0-9_]+",
                criterion_text,
            )
            if len(word) > 2
        ]

        if not words:
            return True

        matches = sum(
            word in text.lower()
            for word in words
        )

        return (
            matches / len(words)
            >= 0.6
        )

    @staticmethod
    def _constraint_violated(
        solution: Any,
        constraint: str,
    ) -> bool:
        """
        Detect obvious textual constraint violations.

        Complex semantic constraints should be checked by the
        connected AI verifier.
        """

        if not constraint:
            return False

        text = Verifier._to_searchable_text(
            solution
        ).lower()

        normalized = (
            constraint
            .strip()
            .lower()
        )

        negative_patterns = [
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
            for marker in negative_patterns
        ):
            return False

        # Extract obvious restricted words after negative markers.
        parts = re.split(
            r"\b(?:must not|do not|don't|cannot|can't|without|avoid|never)\b",
            normalized,
            maxsplit=1,
        )

        if len(parts) != 2:
            return False

        restricted = parts[1].strip()

        if not restricted:
            return False

        return restricted in text

    # ========================================================================
    # GRAPH HELPERS
    # ========================================================================

    @staticmethod
    def _contains_cycle(
        tasks: Sequence[Any],
    ) -> bool:
        """Detect dependency cycles."""

        task_map = {
            Verifier._get_value(
                task,
                "id",
                "",
            ): task
            for task in tasks
        }

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(
            task_id: str,
        ) -> bool:

            if task_id in visiting:
                return True

            if task_id in visited:
                return False

            visiting.add(task_id)

            task = task_map.get(
                task_id
            )

            if task:

                dependencies = Verifier._get_value(
                    task,
                    "dependencies",
                    [],
                )

                for dependency in dependencies:

                    if dependency in task_map:

                        if visit(
                            dependency
                        ):
                            return True

            visiting.remove(task_id)

            visited.add(task_id)

            return False

        for task_id in task_map:

            if visit(task_id):
                return True

        return False

    @staticmethod
    def _get_graph_task(
        graph: Any,
        task_id: str,
    ) -> Any:
        """Retrieve a task from a graph-like object."""

        getter = getattr(
            graph,
            "get_task",
            None,
        )

        if callable(getter):

            try:
                return getter(task_id)
            except Exception:
                return None

        tasks = Verifier._get_value(
            graph,
            "tasks",
            [],
        )

        for task in tasks:

            if (
                Verifier._get_value(
                    task,
                    "id",
                    None,
                )
                == task_id
            ):
                return task

        return None

    # ========================================================================
    # GENERAL HELPERS
    # ========================================================================

    @staticmethod
    def _get_value(
        obj: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        """Read from dictionaries or objects."""

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
    def _has_field(
        obj: Any,
        field_name: str,
    ) -> bool:
        """Check whether an object contains a field."""

        if isinstance(
            obj,
            dict,
        ):
            return field_name in obj

        return hasattr(
            obj,
            field_name,
        )

    @staticmethod
    def _is_empty(
        value: Any,
    ) -> bool:
        """Determine whether a value is meaningfully empty."""

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
    def _to_searchable_text(
        value: Any,
    ) -> str:
        """Convert a value into searchable text."""

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
            return " ".join(
                Verifier._to_searchable_text(
                    key
                )
                + " "
                + Verifier._to_searchable_text(
                    item
                )
                for key, item in value.items()
            )

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return " ".join(
                Verifier._to_searchable_text(
                    item
                )
                for item in value
            )

        return str(value)

    @staticmethod
    def _contains_value(
        container: Any,
        value: Any,
    ) -> bool:
        """Recursively search for a value."""

        if container == value:
            return True

        if isinstance(
            container,
            dict,
        ):
            return any(
                Verifier._contains_value(
                    key,
                    value,
                )
                or Verifier._contains_value(
                    item,
                    value,
                )
                for key, item in container.items()
            )

        if isinstance(
            container,
            (list, tuple, set),
        ):
            return any(
                Verifier._contains_value(
                    item,
                    value,
                )
                for item in container
            )

        return False

    @staticmethod
    def _looks_like_error_message(
        text: str,
    ) -> bool:
        """Detect obvious error output."""

        patterns = [
            r"\btraceback\b",
            r"\bexception\b",
            r"\berror:\b",
            r"\bpermission denied\b",
            r"\bcommand failed\b",
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    @staticmethod
    def _normalize_for_comparison(
        value: Any,
    ) -> str:
        """Normalize a value for consistency checks."""

        return re.sub(
            r"\s+",
            " ",
            str(value).strip().lower(),
        )

    @staticmethod
    def _contradictory(
        left: str,
        right: str,
    ) -> bool:
        """
        Detect a small set of obvious textual contradictions.
        """

        negative_pairs = [
            ("true", "false"),
            ("yes", "no"),
            ("enabled", "disabled"),
            ("on", "off"),
            ("open", "closed"),
            ("active", "inactive"),
            ("available", "unavailable"),
            ("success", "failure"),
            ("successful", "failed"),
        ]

        for first, second in negative_pairs:

            if (
                first in left
                and second in right
            ) or (
                second in left
                and first in right
            ):
                return True

        return False

    @staticmethod
    def _parse_severity(
        value: Any,
    ) -> IssueSeverity:
        """Parse issue severity."""

        if isinstance(
            value,
            IssueSeverity,
        ):
            return value

        try:
            return IssueSeverity(
                str(value).lower()
            )
        except ValueError:
            return IssueSeverity.WARNING

    @staticmethod
    def _plan_mentions_goal(
        plan: Any,
        goal: str,
    ) -> bool:
        """Check whether plan content references the goal."""

        plan_text = Verifier._to_searchable_text(
            plan
        ).lower()

        goal_words = [
            word
            for word in re.findall(
                r"[A-Za-z0-9_]+",
                goal.lower(),
            )
            if len(word) > 3
        ]

        if not goal_words:
            return True

        matches = sum(
            word in plan_text
            for word in goal_words
        )

        return (
            matches / len(goal_words)
            >= 0.3
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def verify(
    subject: Any,
    *,
    expected: Any = None,
    criteria: Optional[
        Sequence[str]
    ] = None,
    context: Optional[
        Dict[str, Any]
    ] = None,
) -> VerificationResult:
    """
    Convenience verification function.
    """

    verifier = Verifier()

    return verifier.verify(
        subject,
        expected=expected,
        criteria=criteria,
        context=context,
    )


def verify_solution(
    problem: Any,
    solution: Any,
    *,
    context: Optional[
        Dict[str, Any]
    ] = None,
) -> VerificationResult:
    """
    Convenience solution verification function.
    """

    verifier = Verifier()

    return verifier.verify_solution(
        problem,
        solution,
        context=context,
    )


def verify_plan(
    plan: Any,
    *,
    goal: Optional[str] = None,
) -> VerificationResult:
    """
    Convenience plan verification function.
    """

    verifier = Verifier()

    return verifier.verify_plan(
        plan,
        goal=goal,
    )


def verify_output(
    output: Any,
    *,
    expected_type: Optional[
        type | tuple[type, ...]
    ] = None,
    required_fields: Optional[
        Iterable[str]
    ] = None,
) -> VerificationResult:
    """
    Convenience output verification function.
    """

    verifier = Verifier()

    return verifier.verify_output(
        output,
        expected_type=expected_type,
        required_fields=required_fields,
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================


__all__ = [
    "VerificationStatus",
    "VerificationLevel",
    "IssueSeverity",
    "VerificationIssue",
    "VerificationResult",
    "VerificationRequest",
    "Verifier",
    "ReasoningVerifier",
    "verify",
    "verify_solution",
    "verify_plan",
    "verify_output",
]


ReasoningVerifier = Verifier
