"""
RENIX AI
Problem Solver

Provides structured problem-solving capabilities for RENIX.

Responsibilities:
- Understand a problem
- Identify objectives and constraints
- Generate possible solutions
- Compare solutions
- Select a solution
- Execute a solver callback when available
- Verify solution completeness
- Return structured results
"""

from __future__ import annotations

import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence


# ============================================================================
# ENUMS
# ============================================================================


class ProblemStatus(str, Enum):
    """Current state of a problem-solving operation."""

    CREATED = "created"
    ANALYZING = "analyzing"
    SOLVING = "solving"
    VERIFYING = "verifying"
    SOLVED = "solved"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SolutionStatus(str, Enum):
    """Current state of a candidate solution."""

    PROPOSED = "proposed"
    EVALUATING = "evaluating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


# ============================================================================
# PROBLEM
# ============================================================================


@dataclass
class Problem:
    """
    Structured representation of a problem.
    """

    id: str
    statement: str

    objective: str = ""

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    constraints: List[str] = field(
        default_factory=list
    )

    requirements: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    known_facts: List[str] = field(
        default_factory=list
    )

    unknowns: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Convert the problem to a dictionary."""

        return {
            "id": self.id,
            "statement": self.statement,
            "objective": self.objective,
            "context": dict(self.context),
            "constraints": list(self.constraints),
            "requirements": list(self.requirements),
            "assumptions": list(self.assumptions),
            "known_facts": list(self.known_facts),
            "unknowns": list(self.unknowns),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


# ============================================================================
# SOLUTION
# ============================================================================


@dataclass
class Solution:
    """
    Candidate solution for a problem.
    """

    id: str

    title: str

    description: str

    status: SolutionStatus = (
        SolutionStatus.PROPOSED
    )

    score: float = 0.0

    confidence: float = 0.0

    advantages: List[str] = field(
        default_factory=list
    )

    disadvantages: List[str] = field(
        default_factory=list
    )

    risks: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    steps: List[str] = field(
        default_factory=list
    )

    result: Any = None

    verification: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================================
    # STATE
    # ========================================================================

    def evaluate(
        self,
        score: float,
        confidence: Optional[float] = None,
    ) -> None:
        """Set evaluation values."""

        self.status = (
            SolutionStatus.EVALUATING
        )

        self.score = max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

        if confidence is not None:
            self.confidence = max(
                0.0,
                min(
                    1.0,
                    float(confidence),
                ),
            )

    def accept(
        self,
    ) -> None:
        """Accept the solution."""

        self.status = (
            SolutionStatus.ACCEPTED
        )

    def reject(
        self,
        reason: Optional[str] = None,
    ) -> None:
        """Reject the solution."""

        self.status = (
            SolutionStatus.REJECTED
        )

        if reason:
            self.metadata[
                "rejection_reason"
            ] = reason

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Convert solution to a dictionary."""

        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
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
            "assumptions": list(
                self.assumptions
            ),
            "steps": list(
                self.steps
            ),
            "result": self.result,
            "verification": dict(
                self.verification
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# SOLVING RESULT
# ============================================================================


@dataclass
class ProblemSolvingResult:
    """
    Final result returned by the problem solver.
    """

    id: str

    problem_id: str

    status: ProblemStatus

    solution: Optional[Solution] = None

    candidates: List[Solution] = field(
        default_factory=list
    )

    reasoning_summary: str = ""

    verification_summary: str = ""

    confidence: float = 0.0

    elapsed_seconds: float = 0.0

    error: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================================
    # STATE
    # ========================================================================

    @property
    def solved(self) -> bool:
        """Return whether the problem was solved."""

        return (
            self.status
            == ProblemStatus.SOLVED
            and self.solution is not None
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Convert result to a dictionary."""

        return {
            "id": self.id,
            "problem_id": self.problem_id,
            "status": self.status.value,
            "solution": (
                self.solution.to_dict()
                if self.solution
                else None
            ),
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "reasoning_summary": (
                self.reasoning_summary
            ),
            "verification_summary": (
                self.verification_summary
            ),
            "confidence": self.confidence,
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "error": self.error,
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# PROBLEM SOLVER
# ============================================================================


class ProblemSolver:
    """
    Main RENIX problem-solving engine.

    The solver can operate with callback-based AI reasoning while
    retaining a deterministic fallback for basic problems.
    """

    def __init__(
        self,
        *,
        analysis_callback: Optional[
            Callable[..., Any]
        ] = None,
        solution_callback: Optional[
            Callable[..., Any]
        ] = None,
        verification_callback: Optional[
            Callable[..., Any]
        ] = None,
    ) -> None:

        self.analysis_callback = (
            analysis_callback
        )

        self.solution_callback = (
            solution_callback
        )

        self.verification_callback = (
            verification_callback
        )

        self._problems: Dict[
            str,
            Problem,
        ] = {}

        self._results: Dict[
            str,
            ProblemSolvingResult,
        ] = {}

        self._cancelled: set[str] = set()

    # ========================================================================
    # PROBLEM CREATION
    # ========================================================================

    def create_problem(
        self,
        statement: str,
        *,
        objective: str = "",
        context: Optional[
            Dict[str, Any]
        ] = None,
        constraints: Optional[
            Sequence[str]
        ] = None,
        requirements: Optional[
            Sequence[str]
        ] = None,
        assumptions: Optional[
            Sequence[str]
        ] = None,
        known_facts: Optional[
            Sequence[str]
        ] = None,
        unknowns: Optional[
            Sequence[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Problem:
        """Create and register a structured problem."""

        if not statement or not statement.strip():
            raise ValueError(
                "Problem statement cannot be empty."
            )

        problem = Problem(
            id=self._generate_id("problem"),
            statement=statement.strip(),
            objective=objective.strip(),
            context=dict(
                context or {}
            ),
            constraints=list(
                constraints or []
            ),
            requirements=list(
                requirements or []
            ),
            assumptions=list(
                assumptions or []
            ),
            known_facts=list(
                known_facts or []
            ),
            unknowns=list(
                unknowns or []
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        self._problems[
            problem.id
        ] = problem

        return problem

    # ========================================================================
    # MAIN SOLVE METHOD
    # ========================================================================

    def solve(
        self,
        problem: Problem | str,
        *,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ProblemSolvingResult:
        """
        Solve a problem.

        ``problem`` may be a Problem instance or a plain string.
        """

        started = time.time()

        if isinstance(
            problem,
            str,
        ):
            problem = self.create_problem(
                problem,
                context=context,
            )

        elif not isinstance(
            problem,
            Problem,
        ):
            raise TypeError(
                "problem must be a Problem or string."
            )

        self._problems[
            problem.id
        ] = problem

        result = ProblemSolvingResult(
            id=self._generate_id("solution"),
            problem_id=problem.id,
            status=ProblemStatus.CREATED,
        )

        self._results[
            result.id
        ] = result

        try:

            if self.is_cancelled(
                result.id
            ):
                result.status = (
                    ProblemStatus.CANCELLED
                )
                return self._finish(
                    result,
                    started,
                )

            # --------------------------------------------------------------
            # ANALYSIS
            # --------------------------------------------------------------

            result.status = (
                ProblemStatus.ANALYZING
            )

            analysis = self.analyze(
                problem,
                context=context,
            )

            result.metadata[
                "analysis"
            ] = analysis

            if self.is_cancelled(
                result.id
            ):
                result.status = (
                    ProblemStatus.CANCELLED
                )
                return self._finish(
                    result,
                    started,
                )

            # --------------------------------------------------------------
            # SOLUTION GENERATION
            # --------------------------------------------------------------

            result.status = (
                ProblemStatus.SOLVING
            )

            candidates = self.generate_solutions(
                problem,
                analysis=analysis,
                context=context,
            )

            result.candidates = candidates

            if not candidates:
                raise RuntimeError(
                    "No candidate solutions were generated."
                )

            # --------------------------------------------------------------
            # EVALUATION
            # --------------------------------------------------------------

            self.evaluate_solutions(
                problem,
                candidates,
                analysis=analysis,
            )

            selected = self.select_solution(
                candidates
            )

            if selected is None:
                raise RuntimeError(
                    "Unable to select a valid solution."
                )

            # --------------------------------------------------------------
            # VERIFICATION
            # --------------------------------------------------------------

            result.status = (
                ProblemStatus.VERIFYING
            )

            verified = self.verify_solution(
                problem,
                selected,
                context=context,
            )

            if not verified:
                selected.reject(
                    "Verification failed."
                )

                alternatives = [
                    candidate
                    for candidate in candidates
                    if candidate.id
                    != selected.id
                    and candidate.status
                    != SolutionStatus.REJECTED
                ]

                for alternative in alternatives:
                    if self.verify_solution(
                        problem,
                        alternative,
                        context=context,
                    ):
                        selected = alternative
                        verified = True
                        break

            if not verified:
                raise RuntimeError(
                    "No candidate solution passed verification."
                )

            selected.accept()

            result.solution = selected

            result.confidence = selected.confidence

            result.verification_summary = (
                selected.verification.get(
                    "summary",
                    "Solution verified.",
                )
            )

            result.status = (
                ProblemStatus.SOLVED
            )

            result.reasoning_summary = (
                self._create_reasoning_summary(
                    problem,
                    analysis,
                    selected,
                )
            )

        except Exception as exc:

            result.status = (
                ProblemStatus.FAILED
            )

            result.error = str(
                exc
            )

        return self._finish(
            result,
            started,
        )

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    def analyze(
        self,
        problem: Problem,
        *,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the problem before generating solutions.
        """

        if self.analysis_callback:

            analysis = (
                self.analysis_callback(
                    problem=problem,
                    context=context or {},
                )
            )

            if isinstance(
                analysis,
                dict,
            ):
                return analysis

            return {
                "summary": str(
                    analysis
                )
            }

        return self._default_analysis(
            problem
        )

    def _default_analysis(
        self,
        problem: Problem,
    ) -> Dict[str, Any]:
        """Generate deterministic basic analysis."""

        objective = (
            problem.objective
            or problem.statement
        )

        return {
            "summary": (
                "Problem analyzed using "
                "RENIX structured reasoning."
            ),
            "problem_statement": (
                problem.statement
            ),
            "objective": objective,
            "constraints": list(
                problem.constraints
            ),
            "requirements": list(
                problem.requirements
            ),
            "known_facts": list(
                problem.known_facts
            ),
            "unknowns": list(
                problem.unknowns
            ),
            "assumptions": list(
                problem.assumptions
            ),
            "context": dict(
                problem.context
            ),
        }

    # ========================================================================
    # SOLUTION GENERATION
    # ========================================================================

    def generate_solutions(
        self,
        problem: Problem,
        *,
        analysis: Optional[
            Dict[str, Any]
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> List[Solution]:
        """
        Generate candidate solutions.
        """

        if self.solution_callback:

            generated = (
                self.solution_callback(
                    problem=problem,
                    analysis=analysis or {},
                    context=context or {},
                )
            )

            return self._normalize_solutions(
                generated
            )

        return self._default_solutions(
            problem
        )

    def _normalize_solutions(
        self,
        generated: Any,
    ) -> List[Solution]:
        """Normalize callback output into Solution objects."""

        if isinstance(
            generated,
            Solution,
        ):
            return [
                generated
            ]

        if not isinstance(
            generated,
            (list, tuple),
        ):
            generated = [
                generated
            ]

        solutions: List[
            Solution
        ] = []

        for index, item in enumerate(
            generated,
            start=1,
        ):

            if isinstance(
                item,
                Solution,
            ):
                solutions.append(
                    item
                )
                continue

            if isinstance(
                item,
                str,
            ):
                solutions.append(
                    Solution(
                        id=self._generate_id(
                            "candidate"
                        ),
                        title=(
                            f"Candidate {index}"
                        ),
                        description=item,
                    )
                )
                continue

            if isinstance(
                item,
                dict,
            ):
                solutions.append(
                    Solution(
                        id=item.get(
                            "id",
                            self._generate_id(
                                "candidate"
                            ),
                        ),
                        title=item.get(
                            "title",
                            f"Candidate {index}",
                        ),
                        description=item.get(
                            "description",
                            "",
                        ),
                        score=float(
                            item.get(
                                "score",
                                0.0,
                            )
                        ),
                        confidence=float(
                            item.get(
                                "confidence",
                                0.0,
                            )
                        ),
                        advantages=list(
                            item.get(
                                "advantages",
                                [],
                            )
                        ),
                        disadvantages=list(
                            item.get(
                                "disadvantages",
                                [],
                            )
                        ),
                        risks=list(
                            item.get(
                                "risks",
                                [],
                            )
                        ),
                        assumptions=list(
                            item.get(
                                "assumptions",
                                [],
                            )
                        ),
                        steps=list(
                            item.get(
                                "steps",
                                [],
                            )
                        ),
                        result=item.get(
                            "result"
                        ),
                        metadata=dict(
                            item.get(
                                "metadata",
                                {},
                            )
                        ),
                    )
                )
                continue

            raise ValueError(
                (
                    "Unsupported solution "
                    f"type at index {index}."
                )
            )

        return solutions

    def _default_solutions(
        self,
        problem: Problem,
    ) -> List[Solution]:
        """
        Basic deterministic fallback solution.
        """

        solution = Solution(
            id=self._generate_id(
                "candidate"
            ),
            title="Direct solution",
            description=(
                "Execute the requested objective "
                "directly while respecting the "
                "identified constraints."
            ),
            advantages=[
                "Simple",
                "Direct",
                "Low planning overhead",
            ],
            disadvantages=[
                "Requires additional reasoning "
                "for complex tasks.",
            ],
            risks=[
                "Complex or ambiguous requirements "
                "may require clarification.",
            ],
            steps=[
                (
                    "Understand the requested objective."
                ),
                (
                    "Apply the required actions."
                ),
                (
                    "Check the result against "
                    "the requirements."
                ),
            ],
        )

        return [
            solution
        ]

    # ========================================================================
    # EVALUATION
    # ========================================================================

    def evaluate_solutions(
        self,
        problem: Problem,
        solutions: List[Solution],
        *,
        analysis: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """
        Evaluate candidate solutions.

        The default evaluator uses completeness, risks,
        advantages and disadvantages as basic signals.
        """

        for solution in solutions:

            score = 0.5

            if solution.steps:
                score += 0.15

            if solution.advantages:
                score += 0.10

            if solution.disadvantages:
                score -= 0.05

            if solution.risks:
                score -= min(
                    0.15,
                    len(solution.risks)
                    * 0.03,
                )

            if solution.assumptions:
                score -= min(
                    0.10,
                    len(solution.assumptions)
                    * 0.02,
                )

            score = max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

            confidence = score

            solution.evaluate(
                score,
                confidence,
            )

    # ========================================================================
    # SELECTION
    # ========================================================================

    def select_solution(
        self,
        solutions: Sequence[
            Solution
        ],
    ) -> Optional[Solution]:
        """
        Select the highest-scoring candidate.
        """

        valid = [
            solution
            for solution in solutions
            if solution.status
            != SolutionStatus.REJECTED
        ]

        if not valid:
            return None

        return max(
            valid,
            key=lambda solution: (
                solution.score,
                solution.confidence,
            ),
        )

    # ========================================================================
    # VERIFICATION
    # ========================================================================

    def verify_solution(
        self,
        problem: Problem,
        solution: Solution,
        *,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> bool:
        """
        Verify a candidate solution.
        """

        if self.verification_callback:

            verification = (
                self.verification_callback(
                    problem=problem,
                    solution=solution,
                    context=context or {},
                )
            )

            if isinstance(
                verification,
                dict,
            ):
                valid = bool(
                    verification.get(
                        "valid",
                        False,
                    )
                )

                solution.verification.update(
                    verification
                )

                return valid

            valid = bool(
                verification
            )

            solution.verification[
                "valid"
            ] = valid

            return valid

        # --------------------------------------------------------------
        # Basic deterministic verification
        # --------------------------------------------------------------

        errors: List[str] = []

        if not solution.description.strip():
            errors.append(
                "Solution description is empty."
            )

        if not solution.steps:
            errors.append(
                "Solution has no execution steps."
            )

        if (
            solution.score < 0.0
            or solution.score > 1.0
        ):
            errors.append(
                "Solution score is invalid."
            )

        valid = not errors

        solution.verification = {
            "valid": valid,
            "errors": errors,
            "summary": (
                "Solution passed basic verification."
                if valid
                else "Solution failed verification."
            ),
        }

        return valid

    # ========================================================================
    # COMPARISON
    # ========================================================================

    def compare_solutions(
        self,
        solutions: Sequence[
            Solution
        ],
    ) -> List[Dict[str, Any]]:
        """
        Compare solutions and return them ordered by score.
        """

        ordered = sorted(
            solutions,
            key=lambda solution: (
                solution.score,
                solution.confidence,
            ),
            reverse=True,
        )

        comparison = []

        for rank, solution in enumerate(
            ordered,
            start=1,
        ):
            comparison.append(
                {
                    "rank": rank,
                    "id": solution.id,
                    "title": solution.title,
                    "score": solution.score,
                    "confidence": (
                        solution.confidence
                    ),
                    "advantages": list(
                        solution.advantages
                    ),
                    "disadvantages": list(
                        solution.disadvantages
                    ),
                    "risks": list(
                        solution.risks
                    ),
                }
            )

        return comparison

    # ========================================================================
    # RESULT MANAGEMENT
    # ========================================================================

    def get_problem(
        self,
        problem_id: str,
    ) -> Optional[Problem]:
        """Get a registered problem."""

        return self._problems.get(
            problem_id
        )

    def get_result(
        self,
        result_id: str,
    ) -> Optional[
        ProblemSolvingResult
    ]:
        """Get a solving result."""

        return self._results.get(
            result_id
        )

    def list_problems(
        self,
    ) -> List[Problem]:
        """Return all registered problems."""

        return list(
            self._problems.values()
        )

    def list_results(
        self,
    ) -> List[
        ProblemSolvingResult
    ]:
        """Return all solving results."""

        return list(
            self._results.values()
        )

    # ========================================================================
    # CANCELLATION
    # ========================================================================

    def cancel(
        self,
        result_id: str,
    ) -> bool:
        """
        Cancel a solving operation.
        """

        if result_id not in self._results:
            return False

        self._cancelled.add(
            result_id
        )

        result = self._results[
            result_id
        ]

        if not result.status in {
            ProblemStatus.SOLVED,
            ProblemStatus.FAILED,
            ProblemStatus.CANCELLED,
        }:
            result.status = (
                ProblemStatus.CANCELLED
            )

        return True

    def is_cancelled(
        self,
        result_id: str,
    ) -> bool:
        """Check whether a result has been cancelled."""

        return result_id in (
            self._cancelled
        )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _finish(
        self,
        result: ProblemSolvingResult,
        started: float,
    ) -> ProblemSolvingResult:
        """Finalize elapsed-time information."""

        result.elapsed_seconds = (
            time.time()
            - started
        )

        return result

    @staticmethod
    def _generate_id(
        prefix: str,
    ) -> str:
        """Generate a unique identifier."""

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    @staticmethod
    def _create_reasoning_summary(
        problem: Problem,
        analysis: Dict[str, Any],
        solution: Solution,
    ) -> str:
        """Create a concise reasoning summary."""

        objective = (
            problem.objective
            or problem.statement
        )

        return (
            f"RENIX analyzed the objective "
            f"'{objective}' and selected "
            f"'{solution.title}' as the "
            f"highest-confidence verified solution."
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def solve_problem(
    statement: str,
    *,
    objective: str = "",
    context: Optional[
        Dict[str, Any]
    ] = None,
) -> ProblemSolvingResult:
    """
    Convenience function for solving a simple problem.
    """

    solver = ProblemSolver()

    problem = solver.create_problem(
        statement,
        objective=objective,
        context=context,
    )

    return solver.solve(
        problem
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================


__all__ = [
    "ProblemStatus",
    "SolutionStatus",
    "Problem",
    "Solution",
    "ProblemSolvingResult",
    "ProblemSolver",
    "solve_problem",
]


