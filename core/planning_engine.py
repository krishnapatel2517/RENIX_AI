"""
RENIX AI
Core Planning Engine

Responsible for:
- Turning reasoning results into executable plans
- Breaking complex objectives into tasks
- Ordering tasks by dependency
- Assigning capabilities to tasks
- Tracking plan status
- Validating plans before execution
- Producing structured execution plans

Execution itself is handled by execution_engine.py.
"""

from __future__ import annotations

import logging
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(
    "RENIX.PlanningEngine"
)


# ============================================================================
# PLAN TASK
# ============================================================================

@dataclass
class PlanTask:
    """
    Represents one task inside a RENIX execution plan.
    """

    task_id: str

    name: str

    description: str

    action: Optional[str] = None

    capability: Optional[str] = None

    dependencies: list[str] = field(
        default_factory=list
    )

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    status: str = "pending"

    priority: int = 5

    required: bool = True

    reversible: bool = True

    requires_confirmation: bool = False

    result: Any = None

    error: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    started_at: Optional[float] = None

    completed_at: Optional[float] = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the task into a dictionary.
        """

        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "capability": self.capability,
            "dependencies": list(
                self.dependencies
            ),
            "parameters": dict(
                self.parameters
            ),
            "status": self.status,
            "priority": self.priority,
            "required": self.required,
            "reversible": self.reversible,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "result": self.result,
            "error": self.error,
            "metadata": dict(
                self.metadata
            ),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# ============================================================================
# PLAN
# ============================================================================

@dataclass
class Plan:
    """
    Represents a complete RENIX execution plan.
    """

    plan_id: str

    objective: str

    tasks: list[PlanTask] = field(
        default_factory=list
    )

    status: str = "draft"

    confidence: float = 0.0

    requires_confirmation: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    started_at: Optional[float] = None

    completed_at: Optional[float] = None

    error: Optional[str] = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the plan into a dictionary.
        """

        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "tasks": [
                task.to_dict()
                for task in self.tasks
            ],
            "status": self.status,
            "confidence": self.confidence,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "metadata": dict(
                self.metadata
            ),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ============================================================================
# PLANNING ENGINE
# ============================================================================

class PlanningEngine:
    """
    Main planning engine for RENIX.
    """

    def __init__(
        self,
        reasoning_engine: Any = None,
        capability_manager: Any = None,
        context_engine: Any = None,
    ) -> None:

        self.logger = logger

        self.reasoning_engine = (
            reasoning_engine
        )

        self.capability_manager = (
            capability_manager
        )

        self.context_engine = (
            context_engine
        )

        self.initialized = False

        self.created_at = time.time()

        self.plans: dict[
            str,
            Plan
        ] = {}

        self.active_plan: Optional[
            Plan
        ] = None

        self.plan_count = 0

        self.completed_plan_count = 0

        self.failed_plan_count = 0

        self.max_history = 500

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the planning engine.
        """

        self.initialized = True

        self.logger.info(
            "RENIX Planning Engine initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the planning engine.
        """

        self.initialized = False

        self.logger.info(
            "RENIX Planning Engine shutdown."
        )

    # ========================================================================
    # ID GENERATION
    # ========================================================================

    @staticmethod
    def _generate_id(
        prefix: str = "plan",
    ) -> str:
        """
        Generate a unique identifier.
        """

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    # ========================================================================
    # NORMALIZE OBJECTIVE
    # ========================================================================

    @staticmethod
    def _normalize_objective(
        objective: Any,
    ) -> str:
        """
        Normalize a planning objective.
        """

        if objective is None:

            return ""

        return str(
            objective
        ).strip()

    # ========================================================================
    # CREATE TASK
    # ========================================================================

    def create_task(
        self,
        name: str,
        description: str,
        *,
        action: Optional[str] = None,
        capability: Optional[str] = None,
        dependencies: Optional[
            list[str]
        ] = None,
        parameters: Optional[
            dict[str, Any]
        ] = None,
        priority: int = 5,
        required: bool = True,
        reversible: bool = True,
        requires_confirmation: bool = False,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> PlanTask:
        """
        Create a plan task.
        """

        return PlanTask(
            task_id=self._generate_id(
                "task"
            ),
            name=name,
            description=description,
            action=action,
            capability=capability,
            dependencies=(
                list(dependencies)
                if dependencies
                else []
            ),
            parameters=(
                dict(parameters)
                if parameters
                else {}
            ),
            priority=priority,
            required=required,
            reversible=reversible,
            requires_confirmation=(
                requires_confirmation
            ),
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

    # ========================================================================
    # CREATE PLAN
    # ========================================================================

    def create_plan(
        self,
        objective: str,
        tasks: Optional[
            list[PlanTask]
        ] = None,
        *,
        confidence: float = 0.0,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Plan:
        """
        Create a new plan.
        """

        normalized = (
            self._normalize_objective(
                objective
            )
        )

        plan = Plan(
            plan_id=self._generate_id(
                "plan"
            ),
            objective=normalized,
            tasks=(
                list(tasks)
                if tasks
                else []
            ),
            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        plan.requires_confirmation = (
            any(
                task.requires_confirmation
                for task in plan.tasks
            )
        )

        self.plans[
            plan.plan_id
        ] = plan

        self.plan_count += 1

        return plan

    # ========================================================================
    # ADD TASK
    # ========================================================================

    def add_task(
        self,
        plan: Plan,
        task: PlanTask,
    ) -> Plan:
        """
        Add a task to an existing plan.
        """

        if any(
            existing.task_id
            == task.task_id
            for existing
            in plan.tasks
        ):

            raise ValueError(
                f"Task {task.task_id} "
                "already exists in plan."
            )

        plan.tasks.append(
            task
        )

        if task.requires_confirmation:

            plan.requires_confirmation = True

        return plan

    # ========================================================================
    # REMOVE TASK
    # ========================================================================

    def remove_task(
        self,
        plan: Plan,
        task_id: str,
    ) -> bool:
        """
        Remove a task from a plan.

        A task cannot be removed if another task
        depends on it.
        """

        dependent_tasks = [
            task
            for task in plan.tasks
            if task_id
            in task.dependencies
        ]

        if dependent_tasks:

            return False

        original_count = len(
            plan.tasks
        )

        plan.tasks = [
            task
            for task in plan.tasks
            if task.task_id
            != task_id
        ]

        removed = (
            len(plan.tasks)
            != original_count
        )

        plan.requires_confirmation = (
            any(
                task.requires_confirmation
                for task in plan.tasks
            )
        )

        return removed

    # ========================================================================
    # GET PLAN
    # ========================================================================

    def get_plan(
        self,
        plan_id: str,
    ) -> Optional[Plan]:
        """
        Get a plan by ID.
        """

        return self.plans.get(
            plan_id
        )

    # ========================================================================
    # SET ACTIVE PLAN
    # ========================================================================

    def set_active_plan(
        self,
        plan: Optional[Plan],
    ) -> None:
        """
        Set the current active plan.
        """

        self.active_plan = plan

    # ========================================================================
    # GET ACTIVE PLAN
    # ========================================================================

    def get_active_plan(
        self,
    ) -> Optional[Plan]:
        """
        Return the current active plan.
        """

        return self.active_plan

    # ========================================================================
    # BUILD CONTEXT
    # ========================================================================

    def _build_context(
        self,
        objective: str,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> dict[str, Any]:
        """
        Build planning context.
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
                    "Unable to load planning context: %s",
                    exc,
                )

        if context:

            result.update(
                context
            )

        result[
            "objective"
        ] = objective

        return result

    # ========================================================================
    # PLAN FROM REASONING
    # ========================================================================

    async def plan_from_reasoning(
        self,
        objective: str,
        *,
        reasoning_result: Any = None,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> Plan:
        """
        Convert a reasoning result into a structured plan.
        """

        normalized = (
            self._normalize_objective(
                objective
            )
        )

        if not normalized:

            raise ValueError(
                "Planning objective cannot be empty."
            )

        plan = self.create_plan(
            normalized
        )

        reasoning_data = None

        if reasoning_result is not None:

            if hasattr(
                reasoning_result,
                "to_dict",
            ):

                reasoning_data = (
                    reasoning_result
                    .to_dict()
                )

            elif isinstance(
                reasoning_result,
                dict,
            ):

                reasoning_data = (
                    reasoning_result
                )

        if reasoning_data:

            plan.metadata[
                "reasoning"
            ] = reasoning_data

        built_context = (
            self._build_context(
                normalized,
                context,
            )
        )

        plan.metadata[
            "context"
        ] = built_context

        # ---------------------------------------------------------------
        # Use reasoning steps if available
        # ---------------------------------------------------------------

        if reasoning_data:

            reasoning_steps = (
                reasoning_data.get(
                    "steps",
                    [],
                )
            )

            previous_task_id = None

            for index, step in enumerate(
                reasoning_steps
            ):

                if isinstance(
                    step,
                    dict,
                ):

                    description = (
                        step.get(
                            "description"
                        )
                        or f"Planning step {index + 1}"
                    )

                    purpose = (
                        step.get(
                            "purpose",
                            "",
                        )
                    )

                else:

                    description = (
                        str(step)
                    )

                    purpose = ""

                task = self.create_task(
                    name=(
                        f"Step {index + 1}"
                    ),
                    description=(
                        description
                    ),
                    action="reasoning_step",
                    capability="reasoning",
                    dependencies=(
                        [previous_task_id]
                        if previous_task_id
                        else []
                    ),
                    metadata={
                        "purpose": purpose,
                    },
                )

                self.add_task(
                    plan,
                    task,
                )

                previous_task_id = (
                    task.task_id
                )

        # ---------------------------------------------------------------
        # If no reasoning steps exist,
        # create a general objective task.
        # ---------------------------------------------------------------

        if not plan.tasks:

            task = self.create_task(
                name="Execute objective",
                description=normalized,
                action="execute_objective",
                capability="general",
                priority=1,
            )

            self.add_task(
                plan,
                task,
            )

        plan.confidence = self._calculate_plan_confidence(
            plan
        )

        plan.status = "ready"

        self.set_active_plan(
            plan
        )

        return plan

    # ========================================================================
    # BUILD PLAN FROM OBJECTIVE
    # ========================================================================

    async def build_plan(
        self,
        objective: str,
        *,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> Plan:
        """
        Build a plan directly from an objective.

        If a reasoning engine is connected, it is used first.
        """

        normalized = (
            self._normalize_objective(
                objective
            )
        )

        if not normalized:

            raise ValueError(
                "Planning objective cannot be empty."
            )

        reasoning_result = None

        if self.reasoning_engine is not None:

            try:

                if hasattr(
                    self.reasoning_engine,
                    "reason",
                ):

                    reasoning_result = (
                        await self.reasoning_engine.reason(
                            normalized,
                            context=context,
                        )
                    )

            except Exception as exc:

                self.logger.warning(
                    "Reasoning engine unavailable during planning: %s",
                    exc,
                )

        return await self.plan_from_reasoning(
            normalized,
            reasoning_result=reasoning_result,
            context=context,
        )

    # ========================================================================
    # DEPENDENCY VALIDATION
    # ========================================================================

    def validate_dependencies(
        self,
        plan: Plan,
    ) -> tuple[
        bool,
        list[str],
    ]:
        """
        Validate task dependencies.
        """

        errors: list[str] = []

        task_ids = {
            task.task_id
            for task in plan.tasks
        }

        for task in plan.tasks:

            for dependency in (
                task.dependencies
            ):

                if dependency not in task_ids:

                    errors.append(
                        (
                            f"Task {task.task_id} "
                            f"depends on missing task "
                            f"{dependency}."
                        )
                    )

        # --------------------------------------------------------------------
        # Cycle detection
        # --------------------------------------------------------------------

        graph = {
            task.task_id: list(
                task.dependencies
            )
            for task in plan.tasks
        }

        visiting: set[str] = set()

        visited: set[str] = set()

        def visit(
            node: str,
        ) -> bool:

            if node in visiting:

                return True

            if node in visited:

                return False

            visiting.add(
                node
            )

            for dependency in graph.get(
                node,
                [],
            ):

                if dependency in graph:

                    if visit(
                        dependency
                    ):

                        return True

            visiting.remove(
                node
            )

            visited.add(
                node
            )

            return False

        for task_id in graph:

            if visit(
                task_id
            ):

                errors.append(
                    "Circular dependency detected."
                )

                break

        return (
            not errors,
            errors,
        )

    # ========================================================================
    # TOPOLOGICAL SORT
    # ========================================================================

    def order_tasks(
        self,
        plan: Plan,
    ) -> list[PlanTask]:
        """
        Return tasks in dependency-safe execution order.
        """

        valid, errors = (
            self.validate_dependencies(
                plan
            )
        )

        if not valid:

            raise ValueError(
                "; ".join(errors)
            )

        task_map = {
            task.task_id: task
            for task in plan.tasks
        }

        remaining = set(
            task_map.keys()
        )

        ordered: list[
            PlanTask
        ] = []

        while remaining:

            ready = []

            for task_id in remaining:

                task = task_map[
                    task_id
                ]

                if all(
                    dependency
                    not in remaining
                    for dependency
                    in task.dependencies
                ):

                    ready.append(
                        task
                    )

            if not ready:

                raise ValueError(
                    "Unable to resolve "
                    "task dependencies."
                )

            ready.sort(
                key=lambda task: (
                    task.priority,
                    task.created_at,
                )
            )

            for task in ready:

                ordered.append(
                    task
                )

                remaining.remove(
                    task.task_id
                )

        return ordered

    # ========================================================================
    # CALCULATE CONFIDENCE
    # ========================================================================

    def _calculate_plan_confidence(
        self,
        plan: Plan,
    ) -> float:
        """
        Calculate overall plan confidence.
        """

        if not plan.tasks:

            return 0.0

        confidence = 0.90

        for task in plan.tasks:

            if not task.action:

                confidence -= 0.02

            if (
                task.capability
                is None
            ):

                confidence -= 0.02

            if task.error:

                confidence -= 0.20

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # ========================================================================
    # VALIDATE PLAN
    # ========================================================================

    def validate_plan(
        self,
        plan: Plan,
    ) -> tuple[
        bool,
        list[str],
    ]:
        """
        Validate a complete plan before execution.
        """

        errors: list[str] = []

        if not plan.objective:

            errors.append(
                "Plan objective is empty."
            )

        if not plan.tasks:

            errors.append(
                "Plan contains no tasks."
            )

        dependency_valid, dependency_errors = (
            self.validate_dependencies(
                plan
            )
        )

        if not dependency_valid:

            errors.extend(
                dependency_errors
            )

        task_names: set[str] = set()

        for task in plan.tasks:

            if not task.name:

                errors.append(
                    (
                        f"Task {task.task_id} "
                        "has no name."
                    )
                )

            if task.name in task_names:

                self.logger.warning(
                    "Duplicate task name: %s",
                    task.name,
                )

            task_names.add(
                task.name
            )

            if (
                task.priority
                < 0
            ):

                errors.append(
                    (
                        f"Task {task.task_id} "
                        "has an invalid priority."
                    )
                )

            if (
                not task.action
                and not task.description
            ):

                errors.append(
                    (
                        f"Task {task.task_id} "
                        "has no action or description."
                    )
                )

        return (
            not errors,
            errors,
        )

    # ========================================================================
    # PREPARE PLAN
    # ========================================================================

    def prepare_plan(
        self,
        plan: Plan,
    ) -> Plan:
        """
        Validate and prepare a plan for execution.
        """

        valid, errors = (
            self.validate_plan(
                plan
            )
        )

        if not valid:

            plan.status = "invalid"

            plan.error = (
                "; ".join(errors)
            )

            return plan

        ordered = (
            self.order_tasks(
                plan
            )
        )

        plan.tasks = ordered

        plan.confidence = (
            self._calculate_plan_confidence(
                plan
            )
        )

        plan.status = "ready"

        plan.error = None

        return plan

    # ========================================================================
    # NEXT TASK
    # ========================================================================

    def get_next_task(
        self,
        plan: Plan,
    ) -> Optional[
        PlanTask
    ]:
        """
        Return the next task that is ready to run.
        """

        for task in self.order_tasks(
            plan
        ):

            if task.status != "pending":

                continue

            dependencies_complete = all(
                self._task_is_complete(
                    plan,
                    dependency,
                )
                for dependency
                in task.dependencies
            )

            if dependencies_complete:

                return task

        return None

    # ========================================================================
    # TASK COMPLETE CHECK
    # ========================================================================

    @staticmethod
    def _task_is_complete(
        plan: Plan,
        task_id: str,
    ) -> bool:
        """
        Check whether a task is complete.
        """

        for task in plan.tasks:

            if task.task_id == task_id:

                return (
                    task.status
                    == "completed"
                )

        return False

    # ========================================================================
    # START PLAN
    # ========================================================================

    def start_plan(
        self,
        plan: Plan,
    ) -> Plan:
        """
        Mark a plan as running.
        """

        valid, errors = (
            self.validate_plan(
                plan
            )
        )

        if not valid:

            plan.status = "invalid"

            plan.error = (
                "; ".join(errors)
            )

            return plan

        plan.status = "running"

        plan.started_at = time.time()

        self.set_active_plan(
            plan
        )

        return plan

    # ========================================================================
    # START TASK
    # ========================================================================

    def start_task(
        self,
        plan: Plan,
        task_id: str,
    ) -> Optional[PlanTask]:
        """
        Mark a task as running.
        """

        task = self._find_task(
            plan,
            task_id,
        )

        if task is None:

            return None

        if not all(
            self._task_is_complete(
                plan,
                dependency,
            )
            for dependency
            in task.dependencies
        ):

            raise RuntimeError(
                (
                    f"Task {task_id} "
                    "has incomplete dependencies."
                )
            )

        task.status = "running"

        task.started_at = time.time()

        return task

    # ========================================================================
    # COMPLETE TASK
    # ========================================================================

    def complete_task(
        self,
        plan: Plan,
        task_id: str,
        result: Any = None,
    ) -> Optional[PlanTask]:
        """
        Mark a task as completed.
        """

        task = self._find_task(
            plan,
            task_id,
        )

        if task is None:

            return None

        task.status = "completed"

        task.result = result

        task.error = None

        task.completed_at = time.time()

        self._update_plan_status(
            plan
        )

        return task

    # ========================================================================
    # FAIL TASK
    # ========================================================================

    def fail_task(
        self,
        plan: Plan,
        task_id: str,
        error: Any,
    ) -> Optional[PlanTask]:
        """
        Mark a task as failed.
        """

        task = self._find_task(
            plan,
            task_id,
        )

        if task is None:

            return None

        task.status = "failed"

        task.error = str(
            error
        )

        task.completed_at = time.time()

        self._update_plan_status(
            plan
        )

        return task

    # ========================================================================
    # FIND TASK
    # ========================================================================

    @staticmethod
    def _find_task(
        plan: Plan,
        task_id: str,
    ) -> Optional[PlanTask]:
        """
        Find a task inside a plan.
        """

        for task in plan.tasks:

            if task.task_id == task_id:

                return task

        return None

    # ========================================================================
    # UPDATE PLAN STATUS
    # ========================================================================

    def _update_plan_status(
        self,
        plan: Plan,
    ) -> None:
        """
        Update overall plan status from task states.
        """

        if not plan.tasks:

            plan.status = "empty"

            return

        if any(
            task.status == "failed"
            and task.required
            for task in plan.tasks
        ):

            plan.status = "failed"

            plan.error = (
                "A required task failed."
            )

            self.failed_plan_count += 1

            return

        if all(
            task.status == "completed"
            for task in plan.tasks
        ):

            plan.status = "completed"

            plan.completed_at = time.time()

            self.completed_plan_count += 1

            if (
                self.active_plan
                and self.active_plan.plan_id
                == plan.plan_id
            ):

                self.active_plan = None

            return

        if any(
            task.status == "running"
            for task in plan.tasks
        ):

            plan.status = "running"

            return

        plan.status = "ready"

    # ========================================================================
    # CANCEL PLAN
    # ========================================================================

    def cancel_plan(
        self,
        plan: Plan,
        reason: str = "Cancelled by user.",
    ) -> Plan:
        """
        Cancel a plan.
        """

        if plan.status in {
            "completed",
            "failed",
            "cancelled",
        }:

            return plan

        plan.status = "cancelled"

        plan.error = reason

        for task in plan.tasks:

            if task.status in {
                "pending",
                "running",
            }:

                task.status = "cancelled"

        if (
            self.active_plan
            and self.active_plan.plan_id
            == plan.plan_id
        ):

            self.active_plan = None

        return plan

    # ========================================================================
    # REPLAN
    # ========================================================================

    async def replan(
        self,
        plan: Plan,
        *,
        reason: Optional[str] = None,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> Plan:
        """
        Create a fresh plan after a failure or changed objective.
        """

        objective = plan.objective

        if reason:

            objective = (
                f"{objective}\n"
                f"Replanning reason: {reason}"
            )

        self.cancel_plan(
            plan,
            reason=(
                "Replaced by a new plan."
            ),
        )

        return await self.build_plan(
            objective,
            context=context,
        )

    # ========================================================================
    # PLAN SUMMARY
    # ========================================================================

    def summarize_plan(
        self,
        plan: Plan,
    ) -> dict[str, Any]:
        """
        Return a compact plan summary.
        """

        total = len(
            plan.tasks
        )

        completed = sum(
            task.status == "completed"
            for task in plan.tasks
        )

        running = sum(
            task.status == "running"
            for task in plan.tasks
        )

        pending = sum(
            task.status == "pending"
            for task in plan.tasks
        )

        failed = sum(
            task.status == "failed"
            for task in plan.tasks
        )

        cancelled = sum(
            task.status == "cancelled"
            for task in plan.tasks
        )

        return {
            "plan_id": plan.plan_id,
            "objective": plan.objective,
            "status": plan.status,
            "confidence": plan.confidence,
            "total_tasks": total,
            "completed_tasks": completed,
            "running_tasks": running,
            "pending_tasks": pending,
            "failed_tasks": failed,
            "cancelled_tasks": cancelled,
            "requires_confirmation": (
                plan.requires_confirmation
            ),
        }

    # ========================================================================
    # LIST PLANS
    # ========================================================================

    def list_plans(
        self,
        status: Optional[str] = None,
    ) -> list[Plan]:
        """
        List plans, optionally filtered by status.
        """

        plans = list(
            self.plans.values()
        )

        if status is not None:

            plans = [
                plan
                for plan in plans
                if plan.status
                == status
            ]

        plans.sort(
            key=lambda plan: (
                plan.created_at
            ),
            reverse=True,
        )

        return plans[
            :self.max_history
        ]

    # ========================================================================
    # DELETE PLAN
    # ========================================================================

    def delete_plan(
        self,
        plan_id: str,
    ) -> bool:
        """
        Delete a stored plan.
        """

        if plan_id not in self.plans:

            return False

        if (
            self.active_plan
            and self.active_plan.plan_id
            == plan_id
        ):

            self.active_plan = None

        del self.plans[
            plan_id
        ]

        return True

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return planning engine statistics.
        """

        return {
            "initialized": (
                self.initialized
            ),
            "plan_count": (
                self.plan_count
            ),
            "stored_plans": len(
                self.plans
            ),
            "completed_plan_count": (
                self.completed_plan_count
            ),
            "failed_plan_count": (
                self.failed_plan_count
            ),
            "active_plan": (
                self.active_plan.plan_id
                if self.active_plan
                else None
            ),
            "created_at": (
                self.created_at
            ),
        }


# ============================================================================
# GLOBAL PLANNING ENGINE
# ============================================================================

planning_engine = PlanningEngine()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def build_plan(
    objective: str,
    *,
    context: Optional[
        dict[str, Any]
    ] = None,
) -> Plan:
    """
    Convenience wrapper around the global planning engine.
    """

    return await planning_engine.build_plan(
        objective,
        context=context,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PlanTask",
    "Plan",
    "PlanningEngine",
    "planning_engine",
    "build_plan",
]


