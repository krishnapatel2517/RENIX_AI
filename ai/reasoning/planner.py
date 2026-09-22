"""
RENIX AI
Reasoning Planner

Creates structured execution plans for RENIX tasks.

Responsibilities:
- Convert goals into executable plans
- Break complex requests into steps
- Define dependencies
- Estimate priorities
- Track plan state
- Validate plans
- Support replanning
"""

from __future__ import annotations

import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ============================================================================
# ENUMS
# ============================================================================


class PlanStatus(str, Enum):
    """
    Current state of a plan.
    """

    CREATED = "created"

    READY = "ready"

    RUNNING = "running"

    PAUSED = "paused"

    COMPLETED = "completed"

    FAILED = "failed"

    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """
    Current state of an individual plan step.
    """

    PENDING = "pending"

    READY = "ready"

    RUNNING = "running"

    COMPLETED = "completed"

    FAILED = "failed"

    SKIPPED = "skipped"

    CANCELLED = "cancelled"


class StepPriority(str, Enum):
    """
    Priority of a plan step.
    """

    LOW = "low"

    NORMAL = "normal"

    HIGH = "high"

    CRITICAL = "critical"


# ============================================================================
# PLAN STEP
# ============================================================================


@dataclass
class PlanStep:
    """
    One executable step inside a RENIX plan.
    """

    id: str

    title: str

    description: str = ""

    action: Optional[str] = None

    agent: Optional[str] = None

    tool: Optional[str] = None

    dependencies: List[str] = field(
        default_factory=list
    )

    priority: StepPriority = (
        StepPriority.NORMAL
    )

    status: StepStatus = (
        StepStatus.PENDING
    )

    optional: bool = False

    estimated_seconds: float = 0.0

    input_data: Dict[str, Any] = field(
        default_factory=dict
    )

    output_data: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    # ========================================================================
    # STATE
    # ========================================================================

    def mark_ready(self) -> None:
        """
        Mark the step as ready.
        """

        if self.status in {
            StepStatus.PENDING,
            StepStatus.READY,
        }:
            self.status = (
                StepStatus.READY
            )

    def start(self) -> None:
        """
        Start the step.
        """

        self.status = StepStatus.RUNNING

        self.error = None

    def complete(
        self,
        output: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """
        Mark the step as completed.
        """

        self.status = (
            StepStatus.COMPLETED
        )

        if output:
            self.output_data.update(
                output
            )

        self.error = None

    def fail(
        self,
        error: Exception | str,
    ) -> None:
        """
        Mark the step as failed.
        """

        self.status = StepStatus.FAILED

        self.error = str(error)

    def skip(
        self,
        reason: Optional[str] = None,
    ) -> None:
        """
        Skip the step.
        """

        self.status = StepStatus.SKIPPED

        if reason:
            self.metadata[
                "skip_reason"
            ] = reason

    def cancel(self) -> None:
        """
        Cancel the step.
        """

        self.status = (
            StepStatus.CANCELLED
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert the step into a dictionary.
        """

        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "agent": self.agent,
            "tool": self.tool,
            "dependencies": list(
                self.dependencies
            ),
            "priority": self.priority.value,
            "status": self.status.value,
            "optional": self.optional,
            "estimated_seconds": (
                self.estimated_seconds
            ),
            "input_data": dict(
                self.input_data
            ),
            "output_data": dict(
                self.output_data
            ),
            "metadata": dict(
                self.metadata
            ),
            "error": self.error,
        }


# ============================================================================
# PLAN
# ============================================================================


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a RENIX task.
    """

    id: str

    goal: str

    description: str = ""

    steps: List[PlanStep] = field(
        default_factory=list
    )

    status: PlanStatus = (
        PlanStatus.CREATED
    )

    created_at: float = 0.0

    started_at: Optional[float] = None

    completed_at: Optional[float] = None

    current_step_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    result: Dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    # ========================================================================
    # STEP MANAGEMENT
    # ========================================================================

    def add_step(
        self,
        step: PlanStep,
    ) -> None:
        """
        Add a step to the plan.
        """

        if any(
            existing.id == step.id
            for existing in self.steps
        ):
            raise ValueError(
                f"Duplicate plan step ID: {step.id}"
            )

        self.steps.append(
            step
        )

    def get_step(
        self,
        step_id: str,
    ) -> Optional[PlanStep]:
        """
        Get a step by ID.
        """

        for step in self.steps:
            if step.id == step_id:
                return step

        return None

    def require_step(
        self,
        step_id: str,
    ) -> PlanStep:
        """
        Get a step or raise an error.
        """

        step = self.get_step(
            step_id
        )

        if step is None:
            raise KeyError(
                f"Unknown plan step: {step_id}"
            )

        return step

    def remove_step(
        self,
        step_id: str,
    ) -> bool:
        """
        Remove a step.
        """

        step = self.get_step(
            step_id
        )

        if step is None:
            return False

        self.steps.remove(
            step
        )

        for other in self.steps:
            if step_id in other.dependencies:
                other.dependencies.remove(
                    step_id
                )

        return True

    # ========================================================================
    # DEPENDENCIES
    # ========================================================================

    def dependencies_completed(
        self,
        step: PlanStep,
    ) -> bool:
        """
        Check whether all dependencies are complete.
        """

        for dependency_id in (
            step.dependencies
        ):
            dependency = self.get_step(
                dependency_id
            )

            if dependency is None:
                return False

            if dependency.status != (
                StepStatus.COMPLETED
            ):
                return False

        return True

    def get_ready_steps(
        self,
    ) -> List[PlanStep]:
        """
        Return steps that can currently execute.
        """

        ready: List[
            PlanStep
        ] = []

        for step in self.steps:

            if step.status not in {
                StepStatus.PENDING,
                StepStatus.READY,
            }:
                continue

            if self.dependencies_completed(
                step
            ):
                step.mark_ready()

                ready.append(
                    step
                )

        return sorted(
            ready,
            key=lambda item: self._priority_value(
                item.priority
            ),
            reverse=True,
        )

    @staticmethod
    def _priority_value(
        priority: StepPriority,
    ) -> int:
        """
        Convert priority into a sortable value.
        """

        values = {
            StepPriority.LOW: 1,
            StepPriority.NORMAL: 2,
            StepPriority.HIGH: 3,
            StepPriority.CRITICAL: 4,
        }

        return values.get(
            priority,
            2,
        )

    # ========================================================================
    # PLAN STATE
    # ========================================================================

    def start(
        self,
    ) -> None:
        """
        Start the execution plan.
        """

        self.status = (
            PlanStatus.RUNNING
        )

        self.current_step_id = None

    def pause(
        self,
    ) -> None:
        """
        Pause the plan.
        """

        self.status = (
            PlanStatus.PAUSED
        )

    def cancel(
        self,
    ) -> None:
        """
        Cancel the plan.
        """

        self.status = (
            PlanStatus.CANCELLED
        )

        for step in self.steps:
            if step.status in {
                StepStatus.PENDING,
                StepStatus.READY,
                StepStatus.RUNNING,
            }:
                step.cancel()

    def complete(
        self,
        result: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """
        Complete the plan.
        """

        self.status = (
            PlanStatus.COMPLETED
        )

        self.current_step_id = None

        if result:
            self.result.update(
                result
            )

    def fail(
        self,
        error: Exception | str,
    ) -> None:
        """
        Mark the plan as failed.
        """

        self.status = (
            PlanStatus.FAILED
        )

        self.error = str(error)

    # ========================================================================
    # PROGRESS
    # ========================================================================

    @property
    def completed_steps(self) -> int:
        """
        Number of completed steps.
        """

        return sum(
            step.status
            == StepStatus.COMPLETED
            for step in self.steps
        )

    @property
    def failed_steps(self) -> int:
        """
        Number of failed steps.
        """

        return sum(
            step.status
            == StepStatus.FAILED
            for step in self.steps
        )

    @property
    def progress(self) -> float:
        """
        Return plan progress from 0 to 1.
        """

        if not self.steps:
            return 1.0

        return (
            self.completed_steps
            / len(self.steps)
        )

    @property
    def is_finished(self) -> bool:
        """
        Check whether the plan has reached a terminal state.
        """

        return self.status in {
            PlanStatus.COMPLETED,
            PlanStatus.FAILED,
            PlanStatus.CANCELLED,
        }

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate(
        self,
    ) -> List[str]:
        """
        Validate the plan structure.

        Returns a list of validation errors.
        """

        errors: List[str] = []

        step_ids = {
            step.id
            for step in self.steps
        }

        if not self.goal.strip():
            errors.append(
                "Plan goal cannot be empty."
            )

        for step in self.steps:

            for dependency in (
                step.dependencies
            ):
                if dependency not in step_ids:
                    errors.append(
                        (
                            f"Step '{step.id}' "
                            f"depends on missing "
                            f"step '{dependency}'."
                        )
                    )

                if dependency == step.id:
                    errors.append(
                        (
                            f"Step '{step.id}' "
                            "cannot depend on itself."
                        )
                    )

        if self._has_cycle():
            errors.append(
                "Plan contains a dependency cycle."
            )

        return errors

    def _has_cycle(
        self,
    ) -> bool:
        """
        Detect dependency cycles.
        """

        visiting: set[str] = set()

        visited: set[str] = set()

        def visit(
            step_id: str,
        ) -> bool:

            if step_id in visiting:
                return True

            if step_id in visited:
                return False

            visiting.add(
                step_id
            )

            step = self.get_step(
                step_id
            )

            if step:
                for dependency in (
                    step.dependencies
                ):
                    if visit(
                        dependency
                    ):
                        return True

            visiting.remove(
                step_id
            )

            visited.add(
                step_id
            )

            return False

        for step in self.steps:
            if visit(step.id):
                return True

        return False

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert the complete plan into a dictionary.
        """

        return {
            "id": self.id,
            "goal": self.goal,
            "description": self.description,
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_step_id": (
                self.current_step_id
            ),
            "metadata": dict(
                self.metadata
            ),
            "result": dict(
                self.result
            ),
            "error": self.error,
            "progress": self.progress,
        }


# ============================================================================
# PLANNER
# ============================================================================


class ReasoningPlanner:
    """
    Main RENIX reasoning planner.

    The planner converts natural-language goals into structured
    execution plans.

    Actual LLM reasoning can be supplied through ``planning_callback``.
    """

    def __init__(
        self,
        *,
        planning_callback: Optional[
            Any
        ] = None,
    ) -> None:

        self.planning_callback = (
            planning_callback
        )

        self._plans: Dict[
            str,
            ExecutionPlan,
        ] = {}

    # ========================================================================
    # PLAN CREATION
    # ========================================================================

    def create_plan(
        self,
        goal: str,
        *,
        description: str = "",
        steps: Optional[
            Sequence[PlanStep]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ExecutionPlan:
        """
        Create a new execution plan.
        """

        import time

        plan = ExecutionPlan(
            id=self._generate_plan_id(),
            goal=goal.strip(),
            description=description,
            created_at=time.time(),
            metadata=dict(
                metadata or {}
            ),
        )

        if steps:
            for step in steps:
                plan.add_step(
                    step
                )

        validation_errors = (
            plan.validate()
        )

        if validation_errors:
            raise ValueError(
                "; ".join(
                    validation_errors
                )
            )

        plan.status = (
            PlanStatus.READY
        )

        self._plans[
            plan.id
        ] = plan

        return plan

    # ========================================================================
    # AI PLANNING
    # ========================================================================

    def plan(
        self,
        goal: str,
        *,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ExecutionPlan:
        """
        Generate a plan for a natural-language goal.

        If an external planning callback is configured, RENIX
        delegates plan generation to it.

        Otherwise a basic fallback plan is generated.
        """

        if self.planning_callback:

            generated = (
                self.planning_callback(
                    goal,
                    context or {},
                )
            )

            return self._normalize_generated_plan(
                goal,
                generated,
            )

        return self._create_basic_plan(
            goal
        )

    def _normalize_generated_plan(
        self,
        goal: str,
        generated: Any,
    ) -> ExecutionPlan:
        """
        Normalize plans returned by an AI planner.
        """

        if isinstance(
            generated,
            ExecutionPlan,
        ):
            self._plans[
                generated.id
            ] = generated

            return generated

        if not isinstance(
            generated,
            (list, tuple),
        ):
            raise ValueError(
                "Planning callback must return "
                "an ExecutionPlan or a sequence of steps."
            )

        steps: List[
            PlanStep
        ] = []

        for index, item in enumerate(
            generated,
            start=1,
        ):

            if isinstance(
                item,
                PlanStep,
            ):
                steps.append(
                    item
                )
                continue

            if isinstance(
                item,
                str,
            ):
                steps.append(
                    PlanStep(
                        id=f"step_{index}",
                        title=item,
                        description=item,
                    )
                )
                continue

            if isinstance(
                item,
                dict,
            ):
                steps.append(
                    PlanStep(
                        id=item.get(
                            "id",
                            f"step_{index}",
                        ),
                        title=item.get(
                            "title",
                            f"Step {index}",
                        ),
                        description=item.get(
                            "description",
                            "",
                        ),
                        action=item.get(
                            "action"
                        ),
                        agent=item.get(
                            "agent"
                        ),
                        tool=item.get(
                            "tool"
                        ),
                        dependencies=list(
                            item.get(
                                "dependencies",
                                [],
                            )
                        ),
                        priority=self._parse_priority(
                            item.get(
                                "priority",
                                "normal",
                            )
                        ),
                        optional=bool(
                            item.get(
                                "optional",
                                False,
                            )
                        ),
                        estimated_seconds=float(
                            item.get(
                                "estimated_seconds",
                                0,
                            )
                        ),
                        input_data=dict(
                            item.get(
                                "input_data",
                                {},
                            )
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
                    "Unsupported generated "
                    f"plan step at index {index}."
                )
            )

        return self.create_plan(
            goal,
            steps=steps,
        )

    @staticmethod
    def _parse_priority(
        value: Any,
    ) -> StepPriority:
        """
        Convert arbitrary priority values to StepPriority.
        """

        if isinstance(
            value,
            StepPriority,
        ):
            return value

        normalized = str(
            value
        ).strip().lower()

        try:
            return StepPriority(
                normalized
            )
        except ValueError:
            return StepPriority.NORMAL

    # ========================================================================
    # BASIC PLANNER
    # ========================================================================

    def _create_basic_plan(
        self,
        goal: str,
    ) -> ExecutionPlan:
        """
        Create a minimal plan when no external reasoning
        callback is available.
        """

        step = PlanStep(
            id="step_1",
            title="Execute requested task",
            description=goal.strip(),
            action="execute_task",
            priority=StepPriority.NORMAL,
        )

        return self.create_plan(
            goal,
            steps=[
                step
            ],
        )

    # ========================================================================
    # PLAN MANAGEMENT
    # ========================================================================

    def get_plan(
        self,
        plan_id: str,
    ) -> Optional[
        ExecutionPlan
    ]:
        """
        Get a plan by ID.
        """

        return self._plans.get(
            plan_id
        )

    def require_plan(
        self,
        plan_id: str,
    ) -> ExecutionPlan:
        """
        Get a plan or raise an error.
        """

        plan = self.get_plan(
            plan_id
        )

        if plan is None:
            raise KeyError(
                f"Unknown plan: {plan_id}"
            )

        return plan

    def delete_plan(
        self,
        plan_id: str,
    ) -> bool:
        """
        Delete a plan.
        """

        return (
            self._plans.pop(
                plan_id,
                None,
            )
            is not None
        )

    def list_plans(
        self,
    ) -> List[ExecutionPlan]:
        """
        Return all stored plans.
        """

        return list(
            self._plans.values()
        )

    # ========================================================================
    # REPLANNING
    # ========================================================================

    def replan(
        self,
        plan_id: str,
        *,
        reason: str = "",
        additional_context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ExecutionPlan:
        """
        Generate a replacement plan based on the previous plan.
        """

        old_plan = self.require_plan(
            plan_id
        )

        context = {
            "previous_plan": (
                old_plan.to_dict()
            ),
            "replan_reason": reason,
            "additional_context": (
                additional_context or {}
            ),
        }

        new_plan = self.plan(
            old_plan.goal,
            context=context,
        )

        new_plan.metadata[
            "replanned_from"
        ] = old_plan.id

        new_plan.metadata[
            "replan_reason"
        ] = reason

        return new_plan

    # ========================================================================
    # READY STEP CALCULATION
    # ========================================================================

    def get_next_steps(
        self,
        plan_id: str,
    ) -> List[PlanStep]:
        """
        Return currently executable steps.
        """

        plan = self.require_plan(
            plan_id
        )

        return plan.get_ready_steps()

    def get_next_step(
        self,
        plan_id: str,
    ) -> Optional[PlanStep]:
        """
        Return the highest-priority ready step.
        """

        steps = self.get_next_steps(
            plan_id
        )

        if not steps:
            return None

        return steps[0]

    # ========================================================================
    # PLAN UPDATES
    # ========================================================================

    def complete_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        output: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ExecutionPlan:
        """
        Complete one plan step.
        """

        plan = self.require_plan(
            plan_id
        )

        step = plan.require_step(
            step_id
        )

        step.complete(
            output
        )

        plan.current_step_id = None

        if all(
            item.status
            in {
                StepStatus.COMPLETED,
                StepStatus.SKIPPED,
            }
            for item in plan.steps
        ):
            plan.complete()

        return plan

    def fail_step(
        self,
        plan_id: str,
        step_id: str,
        error: Exception | str,
    ) -> ExecutionPlan:
        """
        Mark one plan step as failed.
        """

        plan = self.require_plan(
            plan_id
        )

        step = plan.require_step(
            step_id
        )

        step.fail(
            error
        )

        if not step.optional:
            plan.fail(
                error
            )

        return plan

    # ========================================================================
    # IDENTIFIERS
    # ========================================================================

    @staticmethod
    def _generate_plan_id() -> str:
        """
        Generate a unique plan ID.
        """

        return (
            "plan_"
            + uuid.uuid4().hex
        )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================


__all__ = [
    "PlanStatus",
    "StepStatus",
    "StepPriority",
    "PlanStep",
    "ExecutionPlan",
    "ReasoningPlanner",
]


