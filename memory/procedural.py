"""
RENIX Procedural Memory
=======================

Stores knowledge about HOW RENIX performs tasks.

Procedural memory is different from semantic memory:

    Semantic:
        "Python is installed."

    Procedural:
        "To create a Python project, create the virtual environment,
         install dependencies, create the package structure, then run tests."

This module provides:
    - Procedures
    - Steps
    - Preconditions
    - Postconditions
    - Failure handling
    - Procedure versions
    - Success statistics
    - User/project/task association
    - Procedure search
    - Procedure execution tracking
    - Import/export
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Iterable, List, Optional
import logging
import uuid


logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class ProcedureStatus(str, Enum):
    """Lifecycle state of a procedure."""

    ACTIVE = "active"
    DRAFT = "draft"
    DISABLED = "disabled"
    ARCHIVED = "archived"


class StepType(str, Enum):
    """Types of actions that can occur inside a procedure."""

    ACTION = "action"
    CONDITION = "condition"
    INPUT = "input"
    OUTPUT = "output"
    WAIT = "wait"
    LOOP = "loop"
    SUBTASK = "subtask"


class ExecutionStatus(str, Enum):
    """Result of a procedure execution."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ProceduralMemoryError(Exception):
    """Base exception for procedural-memory failures."""


class ProcedureNotFoundError(
    ProceduralMemoryError
):
    """Raised when a procedure cannot be found."""


class ProcedureValidationError(
    ProceduralMemoryError
):
    """Raised when a procedure is invalid."""


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ProcedureStep:
    """
    A single step inside a procedure.

    A step contains the instruction but does not execute it.

    Execution belongs to RENIX's execution/automation layers.
    """

    step_id: str

    order: int

    instruction: str

    step_type: StepType = StepType.ACTION

    description: Optional[str] = None

    tool: Optional[str] = None

    agent: Optional[str] = None

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    expected_result: Optional[str] = None

    timeout_seconds: Optional[float] = None

    optional: bool = False

    retry_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Procedure:
    """
    A reusable RENIX procedure.

    Example:

        name:
            create_python_project

        trigger:
            "create a python project"

        steps:
            1. Create project directory.
            2. Create virtual environment.
            3. Create source structure.
            4. Install dependencies.
            5. Run tests.
    """

    procedure_id: str

    name: str

    description: str

    steps: List[ProcedureStep] = field(
        default_factory=list
    )

    triggers: List[str] = field(
        default_factory=list
    )

    preconditions: List[str] = field(
        default_factory=list
    )

    postconditions: List[str] = field(
        default_factory=list
    )

    failure_handlers: List[str] = field(
        default_factory=list
    )

    tags: List[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    user_id: Optional[str] = None

    project_id: Optional[str] = None

    task_id: Optional[str] = None

    version: int = 1

    status: ProcedureStatus = (
        ProcedureStatus.ACTIVE
    )

    confidence: float = 1.0

    success_count: int = 0

    failure_count: int = 0

    execution_count: int = 0

    created_at: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    updated_at: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    last_executed_at: Optional[str] = None


@dataclass
class ProcedureExecution:
    """
    Historical record of one procedure execution.
    """

    execution_id: str

    procedure_id: str

    status: ExecutionStatus = (
        ExecutionStatus.RUNNING
    )

    started_at: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    finished_at: Optional[str] = None

    completed_steps: int = 0

    failed_step: Optional[int] = None

    error: Optional[str] = None

    input_data: dict[str, Any] = field(
        default_factory=dict
    )

    output_data: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# PROCEDURAL MEMORY
# ============================================================================


class ProceduralMemory:
    """
    RENIX procedural-memory manager.

    This class stores reusable task-solving procedures.

    It intentionally does NOT execute arbitrary instructions.

    Instead it provides structured procedural knowledge to:

        planning_engine
        reasoning_engine
        task_manager
        automation_engine
        execution_engine
        agents

    This separation prevents procedural memory from becoming an unsafe
    arbitrary-command execution layer.
    """

    def __init__(
        self,
    ) -> None:

        self._procedures: dict[
            str,
            Procedure,
        ] = {}

        self._executions: dict[
            str,
            ProcedureExecution,
        ] = {}

        self._lock = RLock()

    # ========================================================================
    # CREATE
    # ========================================================================

    def create(
        self,
        name: str,
        description: str,
        *,
        steps: Optional[
            Iterable[
                ProcedureStep
            ]
        ] = None,
        triggers: Optional[
            Iterable[str]
        ] = None,
        preconditions: Optional[
            Iterable[str]
        ] = None,
        postconditions: Optional[
            Iterable[str]
        ] = None,
        failure_handlers: Optional[
            Iterable[str]
        ] = None,
        tags: Optional[
            Iterable[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        confidence: float = 1.0,
        status: ProcedureStatus = (
            ProcedureStatus.ACTIVE
        ),
    ) -> Procedure:
        """
        Create a new procedure.
        """

        name = str(
            name
        ).strip()

        description = str(
            description
        ).strip()

        if not name:
            raise ProcedureValidationError(
                "Procedure name cannot be empty."
            )

        if not description:
            raise ProcedureValidationError(
                "Procedure description cannot be empty."
            )

        confidence = self._clamp_confidence(
            confidence
        )

        normalized_steps = (
            self._normalize_steps(
                steps
            )
        )

        procedure = Procedure(
            procedure_id=self._new_id(
                "proc"
            ),
            name=name,
            description=description,
            steps=normalized_steps,
            triggers=self._clean_list(
                triggers
            ),
            preconditions=self._clean_list(
                preconditions
            ),
            postconditions=self._clean_list(
                postconditions
            ),
            failure_handlers=self._clean_list(
                failure_handlers
            ),
            tags=self._clean_list(
                tags
            ),
            metadata=dict(
                metadata or {}
            ),
            user_id=user_id,
            project_id=project_id,
            task_id=task_id,
            confidence=confidence,
            status=status,
        )

        with self._lock:

            self._procedures[
                procedure.procedure_id
            ] = procedure

        return procedure

    # ========================================================================
    # READ
    # ========================================================================

    def get(
        self,
        procedure_id: str,
    ) -> Optional[Procedure]:
        """
        Get a procedure by ID.
        """

        with self._lock:

            return self._procedures.get(
                procedure_id
            )

    def require(
        self,
        procedure_id: str,
    ) -> Procedure:
        """
        Get a procedure or raise an error.
        """

        procedure = self.get(
            procedure_id
        )

        if procedure is None:
            raise ProcedureNotFoundError(
                f"Procedure not found: "
                f"{procedure_id}"
            )

        return procedure

    def exists(
        self,
        procedure_id: str,
    ) -> bool:

        with self._lock:

            return (
                procedure_id
                in self._procedures
            )

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        procedure: Procedure,
    ) -> Procedure:
        """
        Update an existing procedure.
        """

        if not isinstance(
            procedure,
            Procedure,
        ):
            raise TypeError(
                "Expected Procedure."
            )

        self._validate_procedure(
            procedure
        )

        with self._lock:

            if (
                procedure.procedure_id
                not in self._procedures
            ):
                raise ProcedureNotFoundError(
                    f"Procedure not found: "
                    f"{procedure.procedure_id}"
                )

            procedure.updated_at = (
                self._utc_now()
            )

            self._procedures[
                procedure.procedure_id
            ] = procedure

            return procedure

    def rename(
        self,
        procedure_id: str,
        name: str,
    ) -> Optional[Procedure]:
        """
        Rename a procedure.
        """

        with self._lock:

            procedure = self.get(
                procedure_id
            )

            if procedure is None:
                return None

            name = str(
                name
            ).strip()

            if not name:
                raise ProcedureValidationError(
                    "Procedure name cannot be empty."
                )

            procedure.name = name

            procedure.updated_at = (
                self._utc_now()
            )

            return procedure

    # ========================================================================
    # STEP MANAGEMENT
    # ========================================================================

    def add_step(
        self,
        procedure_id: str,
        instruction: str,
        *,
        step_type: StepType = StepType.ACTION,
        order: Optional[int] = None,
        description: Optional[str] = None,
        tool: Optional[str] = None,
        agent: Optional[str] = None,
        parameters: Optional[
            dict[str, Any]
        ] = None,
        expected_result: Optional[str] = None,
        timeout_seconds: Optional[
            float
        ] = None,
        optional: bool = False,
        retry_count: int = 0,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ProcedureStep:
        """
        Add a step to a procedure.
        """

        instruction = str(
            instruction
        ).strip()

        if not instruction:
            raise ProcedureValidationError(
                "Step instruction cannot be empty."
            )

        with self._lock:

            procedure = self.require(
                procedure_id
            )

            if order is None:
                order = (
                    len(
                        procedure.steps
                    )
                    + 1
                )

            step = ProcedureStep(
                step_id=self._new_id(
                    "step"
                ),
                order=int(order),
                instruction=instruction,
                step_type=step_type,
                description=description,
                tool=tool,
                agent=agent,
                parameters=dict(
                    parameters or {}
                ),
                expected_result=(
                    expected_result
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
                optional=bool(
                    optional
                ),
                retry_count=max(
                    0,
                    int(
                        retry_count
                    ),
                ),
                metadata=dict(
                    metadata or {}
                ),
            )

            procedure.steps.append(
                step
            )

            self._renumber_steps(
                procedure
            )

            procedure.updated_at = (
                self._utc_now()
            )

            return step

    def remove_step(
        self,
        procedure_id: str,
        step_id: str,
    ) -> bool:
        """
        Remove a procedure step.
        """

        with self._lock:

            procedure = self.get(
                procedure_id
            )

            if procedure is None:
                return False

            original = len(
                procedure.steps
            )

            procedure.steps = [
                step
                for step
                in procedure.steps
                if step.step_id
                != step_id
            ]

            if len(
                procedure.steps
            ) == original:
                return False

            self._renumber_steps(
                procedure
            )

            procedure.updated_at = (
                self._utc_now()
            )

            return True

    def reorder_steps(
        self,
        procedure_id: str,
        ordered_step_ids: List[str],
    ) -> bool:
        """
        Reorder procedure steps.

        All existing step IDs must be present exactly once.
        """

        with self._lock:

            procedure = self.get(
                procedure_id
            )

            if procedure is None:
                return False

            existing_ids = {
                step.step_id
                for step
                in procedure.steps
            }

            requested_ids = set(
                ordered_step_ids
            )

            if (
                existing_ids
                != requested_ids
            ):
                raise ProcedureValidationError(
                    "Reordering must include "
                    "every existing step exactly once."
                )

            mapping = {
                step.step_id: step
                for step
                in procedure.steps
            }

            procedure.steps = [
                mapping[
                    step_id
                ]
                for step_id
                in ordered_step_ids
            ]

            self._renumber_steps(
                procedure
            )

            procedure.updated_at = (
                self._utc_now()
            )

            return True

    # ========================================================================
    # SEARCH
    # ========================================================================

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> List[Procedure]:
        """
        Search procedures using lexical relevance.

        This is intentionally lightweight. A future retrieval layer can
        add embeddings/vector similarity without changing this API.
        """

        query = str(
            query
        ).strip()

        if not query:
            return []

        query_lower = query.casefold()

        query_tokens = self._tokenize(
            query
        )

        with self._lock:

            procedures = list(
                self._procedures.values()
            )

        scored: list[
            tuple[
                float,
                Procedure,
            ]
        ] = []

        for procedure in procedures:

            if (
                procedure.status
                in {
                    ProcedureStatus.ARCHIVED,
                    ProcedureStatus.DISABLED,
                }
                and not include_disabled
            ):
                continue

            if (
                user_id is not None
                and procedure.user_id
                != user_id
            ):
                continue

            if (
                project_id is not None
                and procedure.project_id
                != project_id
            ):
                continue

            if (
                task_id is not None
                and procedure.task_id
                != task_id
            ):
                continue

            searchable = " ".join(
                [
                    procedure.name,
                    procedure.description,
                    *procedure.triggers,
                    *procedure.tags,
                    *[
                        step.instruction
                        for step
                        in procedure.steps
                    ],
                ]
            ).casefold()

            score = 0.0

            if (
                query_lower
                in procedure.name.casefold()
            ):
                score += 10.0

            if (
                query_lower
                in procedure.description.casefold()
            ):
                score += 5.0

            for trigger in procedure.triggers:

                if (
                    query_lower
                    in trigger.casefold()
                ):
                    score += 8.0

            tokens = self._tokenize(
                searchable
            )

            overlap = (
                query_tokens
                & tokens
            )

            score += (
                len(overlap)
                * 1.5
            )

            score += (
                procedure.confidence
                * 0.5
            )

            if score > 0:

                scored.append(
                    (
                        score,
                        procedure,
                    )
                )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].success_rate,
                item[1].confidence,
                self._timestamp(
                    item[1].updated_at
                ),
            ),
            reverse=True,
        )

        return [
            procedure
            for _, procedure
            in scored[
                :max(
                    1,
                    int(limit),
                )
            ]
        ]

    # ========================================================================
    # TRIGGER MATCHING
    # ========================================================================

    def match_trigger(
        self,
        user_input: str,
        *,
        limit: int = 5,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[Procedure]:
        """
        Find procedures whose triggers best match user input.
        """

        user_input = str(
            user_input
        ).strip()

        if not user_input:
            return []

        input_tokens = self._tokenize(
            user_input
        )

        with self._lock:

            procedures = list(
                self._procedures.values()
            )

        matches: list[
            tuple[
                float,
                Procedure,
            ]
        ] = []

        for procedure in procedures:

            if (
                procedure.status
                != ProcedureStatus.ACTIVE
            ):
                continue

            if (
                user_id is not None
                and procedure.user_id
                not in {
                    None,
                    user_id,
                }
            ):
                continue

            if (
                project_id is not None
                and procedure.project_id
                not in {
                    None,
                    project_id,
                }
            ):
                continue

            best_score = 0.0

            for trigger in procedure.triggers:

                trigger_lower = (
                    trigger.casefold()
                )

                if (
                    trigger_lower
                    == user_input.casefold()
                ):
                    best_score = max(
                        best_score,
                        100.0,
                    )
                    continue

                if (
                    trigger_lower
                    in user_input.casefold()
                    or user_input.casefold()
                    in trigger_lower
                ):
                    best_score = max(
                        best_score,
                        50.0,
                    )

                trigger_tokens = (
                    self._tokenize(
                        trigger
                    )
                )

                if trigger_tokens:

                    overlap = (
                        input_tokens
                        & trigger_tokens
                    )

                    ratio = (
                        len(overlap)
                        / len(
                            trigger_tokens
                        )
                    )

                    best_score = max(
                        best_score,
                        ratio * 40.0,
                    )

            if best_score > 0:

                matches.append(
                    (
                        best_score,
                        procedure,
                    )
                )

        matches.sort(
            key=lambda item: (
                item[0],
                item[1].success_rate,
                item[1].confidence,
            ),
            reverse=True,
        )

        return [
            procedure
            for _, procedure
            in matches[
                :max(
                    1,
                    int(limit),
                )
            ]
        ]

    # ========================================================================
    # EXECUTION TRACKING
    # ========================================================================

    def start_execution(
        self,
        procedure_id: str,
        *,
        input_data: Optional[
            dict[str, Any]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ProcedureExecution:
        """
        Start tracking a procedure execution.
        """

        with self._lock:

            procedure = self.require(
                procedure_id
            )

            if (
                procedure.status
                != ProcedureStatus.ACTIVE
            ):
                raise ProceduralMemoryError(
                    "Cannot execute an inactive "
                    "procedure."
                )

            execution = ProcedureExecution(
                execution_id=self._new_id(
                    "exec"
                ),
                procedure_id=procedure_id,
                status=ExecutionStatus.RUNNING,
                input_data=dict(
                    input_data or {}
                ),
                metadata=dict(
                    metadata or {}
                ),
            )

            self._executions[
                execution.execution_id
            ] = execution

            procedure.execution_count += 1
            procedure.last_executed_at = (
                execution.started_at
            )
            procedure.updated_at = (
                self._utc_now()
            )

            return execution

    def complete_execution(
        self,
        execution_id: str,
        *,
        output_data: Optional[
            dict[str, Any]
        ] = None,
        completed_steps: Optional[int] = None,
    ) -> ProcedureExecution:
        """
        Mark execution successful.
        """

        with self._lock:

            execution = (
                self._require_execution(
                    execution_id
                )
            )

            procedure = self.require(
                execution.procedure_id
            )

            execution.status = (
                ExecutionStatus.SUCCESS
            )

            execution.finished_at = (
                self._utc_now()
            )

            execution.output_data = dict(
                output_data or {}
            )

            if completed_steps is None:
                completed_steps = len(
                    procedure.steps
                )

            execution.completed_steps = max(
                0,
                int(
                    completed_steps
                ),
            )

            procedure.success_count += 1

            procedure.updated_at = (
                self._utc_now()
            )

            self._recalculate_confidence(
                procedure
            )

            return execution

    def fail_execution(
        self,
        execution_id: str,
        error: str,
        *,
        failed_step: Optional[int] = None,
        completed_steps: int = 0,
        output_data: Optional[
            dict[str, Any]
        ] = None,
    ) -> ProcedureExecution:
        """
        Mark execution as failed.
        """

        with self._lock:

            execution = (
                self._require_execution(
                    execution_id
                )
            )

            procedure = self.require(
                execution.procedure_id
            )

            execution.status = (
                ExecutionStatus.FAILED
            )

            execution.finished_at = (
                self._utc_now()
            )

            execution.error = str(
                error
            )

            execution.failed_step = (
                failed_step
            )

            execution.completed_steps = max(
                0,
                int(
                    completed_steps
                ),
            )

            execution.output_data = dict(
                output_data or {}
            )

            procedure.failure_count += 1

            procedure.updated_at = (
                self._utc_now()
            )

            self._recalculate_confidence(
                procedure
            )

            return execution

    def cancel_execution(
        self,
        execution_id: str,
    ) -> Optional[ProcedureExecution]:
        """
        Cancel a running procedure execution.
        """

        with self._lock:

            execution = (
                self._executions.get(
                    execution_id
                )
            )

            if execution is None:
                return None

            if (
                execution.status
                != ExecutionStatus.RUNNING
            ):
                return execution

            execution.status = (
                ExecutionStatus.CANCELLED
            )

            execution.finished_at = (
                self._utc_now()
            )

            return execution

    def partial_execution(
        self,
        execution_id: str,
        *,
        completed_steps: int,
        output_data: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[ProcedureExecution]:
        """
        Mark execution as partially successful.
        """

        with self._lock:

            execution = (
                self._executions.get(
                    execution_id
                )
            )

            if execution is None:
                return None

            execution.status = (
                ExecutionStatus.PARTIAL
            )

            execution.finished_at = (
                self._utc_now()
            )

            execution.completed_steps = max(
                0,
                int(
                    completed_steps
                ),
            )

            execution.output_data = dict(
                output_data or {}
            )

            return execution

    # ========================================================================
    # EXECUTION HISTORY
    # ========================================================================

    def get_execution(
        self,
        execution_id: str,
    ) -> Optional[ProcedureExecution]:
        """
        Retrieve an execution record.
        """

        with self._lock:

            return self._executions.get(
                execution_id
            )

    def execution_history(
        self,
        procedure_id: str,
        *,
        limit: int = 50,
    ) -> List[ProcedureExecution]:
        """
        Return execution history for a procedure.
        """

        with self._lock:

            executions = [
                execution
                for execution
                in self._executions.values()
                if (
                    execution.procedure_id
                    == procedure_id
                )
            ]

        executions.sort(
            key=lambda execution: (
                self._timestamp(
                    execution.started_at
                )
            ),
            reverse=True,
        )

        return executions[
            :max(
                1,
                int(limit),
            )
        ]

    # ========================================================================
    # VERSIONING
    # ========================================================================

    def create_version(
        self,
        procedure_id: str,
        *,
        steps: Optional[
            Iterable[
                ProcedureStep
            ]
        ] = None,
        description: Optional[str] = None,
        triggers: Optional[
            Iterable[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[Procedure]:
        """
        Create a new version of an existing procedure.

        The original procedure remains untouched.
        """

        with self._lock:

            original = self.get(
                procedure_id
            )

            if original is None:
                return None

            new_steps = (
                self._clone_steps(
                    steps
                    if steps is not None
                    else original.steps
                )
            )

            version = Procedure(
                procedure_id=self._new_id(
                    "proc"
                ),
                name=original.name,
                description=(
                    description
                    if description is not None
                    else original.description
                ),
                steps=new_steps,
                triggers=(
                    self._clean_list(
                        triggers
                    )
                    if triggers is not None
                    else list(
                        original.triggers
                    )
                ),
                preconditions=list(
                    original.preconditions
                ),
                postconditions=list(
                    original.postconditions
                ),
                failure_handlers=list(
                    original.failure_handlers
                ),
                tags=list(
                    original.tags
                ),
                metadata={
                    **original.metadata,
                    **(
                        metadata or {}
                    ),
                    "parent_procedure_id": (
                        original.procedure_id
                    ),
                },
                user_id=original.user_id,
                project_id=original.project_id,
                task_id=original.task_id,
                version=(
                    original.version
                    + 1
                ),
                confidence=original.confidence,
                status=ProcedureStatus.ACTIVE,
            )

            self._procedures[
                version.procedure_id
            ] = version

            return version

    # ========================================================================
    # STATUS
    # ========================================================================

    def enable(
        self,
        procedure_id: str,
    ) -> bool:

        with self._lock:

            procedure = self.get(
                procedure_id
            )

            if procedure is None:
                return False

            procedure.status = (
                ProcedureStatus.ACTIVE
            )

            procedure.updated_at = (
                self._utc_now()
            )

            return True

    def disable(
        self,
        procedure_id: str,
    ) -> bool:

        with self._lock:

            procedure = self.get(
                procedure_id
            )

            if procedure is None:
                return False

            procedure.status = (
                ProcedureStatus.DISABLED
            )

            procedure.updated_at = (
                self._utc_now()
            )

            return True

    def archive(
        self,
        procedure_id: str,
    ) -> bool:

        with self._lock:

            procedure = self.get(
                procedure_id
            )

            if procedure is None:
                return False

            procedure.status = (
                ProcedureStatus.ARCHIVED
            )

            procedure.updated_at = (
                self._utc_now()
            )

            return True

    # ========================================================================
    # LISTING
    # ========================================================================

    def list_all(
        self,
        *,
        status: Optional[
            ProcedureStatus
        ] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Procedure]:
        """
        List procedures using optional filters.
        """

        with self._lock:

            procedures = [
                procedure
                for procedure
                in self._procedures.values()
                if (
                    status is None
                    or procedure.status
                    == status
                )
                and (
                    user_id is None
                    or procedure.user_id
                    == user_id
                )
                and (
                    project_id is None
                    or procedure.project_id
                    == project_id
                )
                and (
                    task_id is None
                    or procedure.task_id
                    == task_id
                )
            ]

        procedures.sort(
            key=lambda procedure: (
                self._timestamp(
                    procedure.updated_at
                )
            ),
            reverse=True,
        )

        return procedures[
            :max(
                1,
                int(limit),
            )
        ]

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(self) -> dict[str, Any]:
        """
        Return procedural-memory statistics.
        """

        with self._lock:

            procedures = list(
                self._procedures.values()
            )

            executions = list(
                self._executions.values()
            )

        active = sum(
            1
            for procedure
            in procedures
            if procedure.status
            == ProcedureStatus.ACTIVE
        )

        archived = sum(
            1
            for procedure
            in procedures
            if procedure.status
            == ProcedureStatus.ARCHIVED
        )

        disabled = sum(
            1
            for procedure
            in procedures
            if procedure.status
            == ProcedureStatus.DISABLED
        )

        successful_executions = sum(
            1
            for execution
            in executions
            if execution.status
            == ExecutionStatus.SUCCESS
        )

        failed_executions = sum(
            1
            for execution
            in executions
            if execution.status
            == ExecutionStatus.FAILED
        )

        return {
            "procedures": len(
                procedures
            ),
            "active": active,
            "archived": archived,
            "disabled": disabled,
            "executions": len(
                executions
            ),
            "successful_executions": (
                successful_executions
            ),
            "failed_executions": (
                failed_executions
            ),
        }

    def health(self) -> dict[str, Any]:
        """
        Return health information for RENIX diagnostics.
        """

        return {
            "status": "healthy",
            **self.statistics(),
        }

    # ========================================================================
    # EXPORT
    # ========================================================================

    def to_dict(
        self,
        procedure_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Convert a procedure to a JSON-compatible dictionary.
        """

        procedure = self.get(
            procedure_id
        )

        if procedure is None:
            return None

        return self._procedure_to_dict(
            procedure
        )

    def export_all(
        self,
    ) -> List[dict[str, Any]]:
        """
        Export all procedures.
        """

        with self._lock:

            procedures = list(
                self._procedures.values()
            )

        return [
            self._procedure_to_dict(
                procedure
            )
            for procedure
            in procedures
        ]

    def export_executions(
        self,
    ) -> List[dict[str, Any]]:
        """
        Export execution history.
        """

        with self._lock:

            executions = list(
                self._executions.values()
            )

        return [
            self._execution_to_dict(
                execution
            )
            for execution
            in executions
        ]

    # ========================================================================
    # IMPORT
    # ========================================================================

    def import_procedure(
        self,
        data: dict[str, Any],
        *,
        replace: bool = False,
    ) -> Procedure:
        """
        Import a procedure from a dictionary.

        Existing IDs are rejected unless replace=True.
        """

        if not isinstance(
            data,
            dict,
        ):
            raise ProcedureValidationError(
                "Procedure data must be a dictionary."
            )

        procedure_id = str(
            data.get(
                "procedure_id"
            )
            or self._new_id(
                "proc"
            )
        )

        raw_steps = data.get(
            "steps",
            [],
        )

        steps = []

        for raw in raw_steps:

            if isinstance(
                raw,
                ProcedureStep,
            ):
                steps.append(
                    raw
                )
                continue

            steps.append(
                ProcedureStep(
                    step_id=str(
                        raw.get(
                            "step_id"
                        )
                        or self._new_id(
                            "step"
                        )
                    ),
                    order=int(
                        raw.get(
                            "order",
                            len(
                                steps
                            ) + 1,
                        )
                    ),
                    instruction=str(
                        raw.get(
                            "instruction",
                            "",
                        )
                    ),
                    step_type=self._enum_value(
                        StepType,
                        raw.get(
                            "step_type",
                            StepType.ACTION.value,
                        ),
                        StepType.ACTION,
                    ),
                    description=raw.get(
                        "description"
                    ),
                    tool=raw.get(
                        "tool"
                    ),
                    agent=raw.get(
                        "agent"
                    ),
                    parameters=dict(
                        raw.get(
                            "parameters",
                            {},
                        )
                    ),
                    expected_result=raw.get(
                        "expected_result"
                    ),
                    timeout_seconds=raw.get(
                        "timeout_seconds"
                    ),
                    optional=bool(
                        raw.get(
                            "optional",
                            False,
                        )
                    ),
                    retry_count=max(
                        0,
                        int(
                            raw.get(
                                "retry_count",
                                0,
                            )
                        ),
                    ),
                    metadata=dict(
                        raw.get(
                            "metadata",
                            {},
                        )
                    ),
                )
            )

        procedure = Procedure(
            procedure_id=procedure_id,
            name=str(
                data.get(
                    "name",
                    "",
                )
            ),
            description=str(
                data.get(
                    "description",
                    "",
                )
            ),
            steps=steps,
            triggers=self._clean_list(
                data.get(
                    "triggers",
                    [],
                )
            ),
            preconditions=self._clean_list(
                data.get(
                    "preconditions",
                    [],
                )
            ),
            postconditions=self._clean_list(
                data.get(
                    "postconditions",
                    [],
                )
            ),
            failure_handlers=self._clean_list(
                data.get(
                    "failure_handlers",
                    [],
                )
            ),
            tags=self._clean_list(
                data.get(
                    "tags",
                    [],
                )
            ),
            metadata=dict(
                data.get(
                    "metadata",
                    {},
                )
            ),
            user_id=data.get(
                "user_id"
            ),
            project_id=data.get(
                "project_id"
            ),
            task_id=data.get(
                "task_id"
            ),
            version=int(
                data.get(
                    "version",
                    1,
                )
            ),
            status=self._enum_value(
                ProcedureStatus,
                data.get(
                    "status",
                    ProcedureStatus.ACTIVE.value,
                ),
                ProcedureStatus.ACTIVE,
            ),
            confidence=self._clamp_confidence(
                data.get(
                    "confidence",
                    1.0,
                )
            ),
            success_count=max(
                0,
                int(
                    data.get(
                        "success_count",
                        0,
                    )
                ),
            ),
            failure_count=max(
                0,
                int(
                    data.get(
                        "failure_count",
                        0,
                    )
                ),
            ),
            execution_count=max(
                0,
                int(
                    data.get(
                        "execution_count",
                        0,
                    )
                ),
            ),
            created_at=str(
                data.get(
                    "created_at",
                    self._utc_now(),
                )
            ),
            updated_at=str(
                data.get(
                    "updated_at",
                    self._utc_now(),
                )
            ),
            last_executed_at=data.get(
                "last_executed_at"
            ),
        )

        self._validate_procedure(
            procedure
        )

        with self._lock:

            if (
                procedure_id
                in self._procedures
                and not replace
            ):
                raise ProceduralMemoryError(
                    f"Procedure already exists: "
                    f"{procedure_id}"
                )

            self._procedures[
                procedure_id
            ] = procedure

        return procedure

    # ========================================================================
    # INTERNAL VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_procedure(
        procedure: Procedure,
    ) -> None:
        """
        Validate procedure structure.
        """

        if not procedure.name.strip():
            raise ProcedureValidationError(
                "Procedure name cannot be empty."
            )

        if not procedure.description.strip():
            raise ProcedureValidationError(
                "Procedure description cannot be empty."
            )

        step_ids = set()

        for step in procedure.steps:

            if not step.instruction.strip():
                raise ProcedureValidationError(
                    "Every procedure step must "
                    "have an instruction."
                )

            if step.step_id in step_ids:
                raise ProcedureValidationError(
                    f"Duplicate step ID: "
                    f"{step.step_id}"
                )

            step_ids.add(
                step.step_id
            )

            if step.retry_count < 0:
                raise ProcedureValidationError(
                    "retry_count cannot be negative."
                )

            if (
                step.timeout_seconds
                is not None
                and step.timeout_seconds
                <= 0
            ):
                raise ProcedureValidationError(
                    "timeout_seconds must be positive."
                )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    @staticmethod
    def _new_id(
        prefix: str,
    ) -> str:

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    @staticmethod
    def _utc_now() -> str:

        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    @staticmethod
    def _timestamp(
        value: Optional[str],
    ) -> float:

        if not value:
            return 0.0

        try:

            return datetime.fromisoformat(
                str(value).replace(
                    "Z",
                    "+00:00",
                )
            ).timestamp()

        except (
            ValueError,
            TypeError,
        ):
            return 0.0

    @staticmethod
    def _clamp_confidence(
        value: Any,
    ) -> float:

        try:
            value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    @staticmethod
    def _clean_list(
        values: Optional[
            Iterable[Any]
        ],
    ) -> List[str]:

        if values is None:
            return []

        result = []

        for value in values:

            value = str(
                value
            ).strip()

            if value:
                result.append(
                    value
                )

        return list(
            dict.fromkeys(
                result
            )
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:

        normalized = (
            str(text)
            .casefold()
        )

        punctuation = (
            ",",
            ".",
            "!",
            "?",
            ":",
            ";",
            "/",
            "\\",
            "-",
            "_",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "'",
            '"',
        )

        for char in punctuation:

            normalized = (
                normalized.replace(
                    char,
                    " ",
                )
            )

        return {
            token
            for token
            in normalized.split()
            if len(token) >= 2
        }

    @staticmethod
    def _normalize_steps(
        steps: Optional[
            Iterable[
                ProcedureStep
            ]
        ],
    ) -> List[ProcedureStep]:

        if steps is None:
            return []

        normalized = []

        for index, step in enumerate(
            steps,
            start=1,
        ):

            if not isinstance(
                step,
                ProcedureStep,
            ):
                raise ProcedureValidationError(
                    "All procedure steps "
                    "must be ProcedureStep objects."
                )

            step.order = index

            if not step.step_id:
                step.step_id = (
                    "step_"
                    + uuid.uuid4().hex
                )

            normalized.append(
                step
            )

        return normalized

    @staticmethod
    def _clone_steps(
        steps: Iterable[
            ProcedureStep
        ],
    ) -> List[ProcedureStep]:

        cloned = []

        for index, step in enumerate(
            steps,
            start=1,
        ):

            cloned.append(
                ProcedureStep(
                    step_id=(
                        "step_"
                        + uuid.uuid4().hex
                    ),
                    order=index,
                    instruction=step.instruction,
                    step_type=step.step_type,
                    description=step.description,
                    tool=step.tool,
                    agent=step.agent,
                    parameters=dict(
                        step.parameters
                    ),
                    expected_result=(
                        step.expected_result
                    ),
                    timeout_seconds=(
                        step.timeout_seconds
                    ),
                    optional=step.optional,
                    retry_count=step.retry_count,
                    metadata=dict(
                        step.metadata
                    ),
                )
            )

        return cloned

    @staticmethod
    def _renumber_steps(
        procedure: Procedure,
    ) -> None:

        procedure.steps.sort(
            key=lambda step: step.order
        )

        for index, step in enumerate(
            procedure.steps,
            start=1,
        ):
            step.order = index

    def _require_execution(
        self,
        execution_id: str,
    ) -> ProcedureExecution:

        execution = (
            self._executions.get(
                execution_id
            )
        )

        if execution is None:
            raise ProceduralMemoryError(
                f"Execution not found: "
                f"{execution_id}"
            )

        return execution

    @staticmethod
    def _enum_value(
        enum_class,
        value,
        default,
    ):

        if isinstance(
            value,
            enum_class,
        ):
            return value

        try:
            return enum_class(
                value
            )
        except (
            ValueError,
            TypeError,
        ):
            return default

    @staticmethod
    def _recalculate_confidence(
        procedure: Procedure,
    ) -> None:
        """
        Adjust procedural confidence from execution history.

        The confidence is intentionally conservative:
        - no executions -> unchanged
        - successful executions increase confidence
        - failures decrease confidence
        """

        total = (
            procedure.success_count
            + procedure.failure_count
        )

        if total <= 0:
            return

        success_rate = (
            procedure.success_count
            / total
        )

        procedure.confidence = max(
            0.05,
            min(
                1.0,
                (
                    0.35
                    + (
                        success_rate
                        * 0.65
                    )
                ),
            ),
        )

    @staticmethod
    def _procedure_to_dict(
        procedure: Procedure,
    ) -> dict[str, Any]:

        return {
            "procedure_id": (
                procedure.procedure_id
            ),
            "name": procedure.name,
            "description": (
                procedure.description
            ),
            "steps": [
                {
                    "step_id": step.step_id,
                    "order": step.order,
                    "instruction": (
                        step.instruction
                    ),
                    "step_type": (
                        step.step_type.value
                    ),
                    "description": (
                        step.description
                    ),
                    "tool": step.tool,
                    "agent": step.agent,
                    "parameters": dict(
                        step.parameters
                    ),
                    "expected_result": (
                        step.expected_result
                    ),
                    "timeout_seconds": (
                        step.timeout_seconds
                    ),
                    "optional": step.optional,
                    "retry_count": (
                        step.retry_count
                    ),
                    "metadata": dict(
                        step.metadata
                    ),
                }
                for step
                in procedure.steps
            ],
            "triggers": list(
                procedure.triggers
            ),
            "preconditions": list(
                procedure.preconditions
            ),
            "postconditions": list(
                procedure.postconditions
            ),
            "failure_handlers": list(
                procedure.failure_handlers
            ),
            "tags": list(
                procedure.tags
            ),
            "metadata": dict(
                procedure.metadata
            ),
            "user_id": procedure.user_id,
            "project_id": procedure.project_id,
            "task_id": procedure.task_id,
            "version": procedure.version,
            "status": (
                procedure.status.value
            ),
            "confidence": (
                procedure.confidence
            ),
            "success_count": (
                procedure.success_count
            ),
            "failure_count": (
                procedure.failure_count
            ),
            "execution_count": (
                procedure.execution_count
            ),
            "success_rate": (
                procedure.success_rate
            ),
            "created_at": (
                procedure.created_at
            ),
            "updated_at": (
                procedure.updated_at
            ),
            "last_executed_at": (
                procedure.last_executed_at
            ),
        }

    @staticmethod
    def _execution_to_dict(
        execution: ProcedureExecution,
    ) -> dict[str, Any]:

        return {
            "execution_id": (
                execution.execution_id
            ),
            "procedure_id": (
                execution.procedure_id
            ),
            "status": (
                execution.status.value
            ),
            "started_at": (
                execution.started_at
            ),
            "finished_at": (
                execution.finished_at
            ),
            "completed_steps": (
                execution.completed_steps
            ),
            "failed_step": (
                execution.failed_step
            ),
            "error": execution.error,
            "input_data": dict(
                execution.input_data
            ),
            "output_data": dict(
                execution.output_data
            ),
            "metadata": dict(
                execution.metadata
            ),
        }


# ============================================================================
# PROCEDURE COMPUTED PROPERTIES
# ============================================================================


def _procedure_success_rate(
    self: Procedure,
) -> float:
    """
    Calculate historical procedure success rate.
    """

    total = (
        self.success_count
        + self.failure_count
    )

    if total <= 0:
        return 0.0

    return (
        self.success_count
        / total
    )


# Attach as a property so the dataclass remains simple and serializable.
Procedure.success_rate = property(
    _procedure_success_rate
)


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "ProcedureStatus",
    "StepType",
    "ExecutionStatus",
    "ProceduralMemoryError",
    "ProcedureNotFoundError",
    "ProcedureValidationError",
    "ProcedureStep",
    "Procedure",
    "ProcedureExecution",
    "ProceduralMemory",
]


