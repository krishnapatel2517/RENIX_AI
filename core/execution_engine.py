"""
RENIX AI
Core Execution Engine

Responsible for:
- Executing approved decisions
- Executing actions through registered capabilities/services
- Handling synchronous and asynchronous executors
- Tracking execution state
- Handling timeouts
- Handling failures
- Supporting cancellation
- Returning structured execution results

The Execution Engine performs WHAT the Decision Engine decided.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.ExecutionEngine"
)


# ============================================================================
# EXECUTION RESULT
# ============================================================================

@dataclass
class ExecutionResult:
    """
    Result returned after an action is executed.
    """

    execution_id: str

    decision_id: Optional[str]

    action: Optional[str]

    status: str = "pending"

    success: bool = False

    result: Any = None

    error: Optional[str] = None

    started_at: Optional[float] = None

    completed_at: Optional[float] = None

    duration: float = 0.0

    cancelled: bool = False

    timed_out: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert execution result into a dictionary.
        """

        return {
            "execution_id": self.execution_id,
            "decision_id": self.decision_id,
            "action": self.action,
            "status": self.status,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# EXECUTION TASK
# ============================================================================

@dataclass
class ExecutionTask:
    """
    Represents an active execution.
    """

    execution_id: str

    action: str

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    decision_id: Optional[str] = None

    status: str = "pending"

    task: Optional[
        asyncio.Task
    ] = None

    created_at: float = field(
        default_factory=time.time
    )

    started_at: Optional[float] = None

    completed_at: Optional[float] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """
    Main RENIX execution engine.

    It does not decide what action should be performed.
    It executes an already-approved action.
    """

    def __init__(
        self,
        service_registry: Any = None,
        capability_manager: Any = None,
        event_bus: Any = None,
        decision_engine: Any = None,
    ) -> None:

        self.logger = logger

        self.service_registry = (
            service_registry
        )

        self.capability_manager = (
            capability_manager
        )

        self.event_bus = event_bus

        self.decision_engine = (
            decision_engine
        )

        self.initialized = False

        self.created_at = time.time()

        self.executors: dict[
            str,
            Callable[..., Any],
        ] = {}

        self.active_tasks: dict[
            str,
            ExecutionTask,
        ] = {}

        self.history: dict[
            str,
            ExecutionResult,
        ] = {}

        self.execution_count = 0

        self.success_count = 0

        self.failure_count = 0

        self.cancelled_count = 0

        self.timeout_count = 0

        self.default_timeout = 120.0

        self.max_history = 500

        self.running = False

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the execution engine.
        """

        self.initialized = True
        self.running = True

        self.logger.info(
            "RENIX Execution Engine initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the execution engine.
        """

        active_ids = list(
            self.active_tasks.keys()
        )

        for execution_id in active_ids:

            try:

                await self.cancel(
                    execution_id
                )

            except Exception as exc:

                self.logger.warning(
                    "Failed to cancel execution %s: %s",
                    execution_id,
                    exc,
                )

        self.running = False
        self.initialized = False

        self.logger.info(
            "RENIX Execution Engine shutdown."
        )

    # ========================================================================
    # ID
    # ========================================================================

    @staticmethod
    def _generate_id(
        prefix: str = "execution",
    ) -> str:
        """
        Generate a unique execution ID.
        """

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    # ========================================================================
    # REGISTER EXECUTOR
    # ========================================================================

    def register_executor(
        self,
        action: str,
        executor: Callable[..., Any],
    ) -> None:
        """
        Register a Python callable for an action.

        Example:

            engine.register_executor(
                "open_browser",
                open_browser,
            )
        """

        if not action:

            raise ValueError(
                "Action name cannot be empty."
            )

        if not callable(executor):

            raise TypeError(
                "Executor must be callable."
            )

        self.executors[
            action
        ] = executor

        self.logger.debug(
            "Registered executor: %s",
            action,
        )

    # ========================================================================
    # UNREGISTER EXECUTOR
    # ========================================================================

    def unregister_executor(
        self,
        action: str,
    ) -> bool:
        """
        Remove an executor.
        """

        if action not in self.executors:

            return False

        del self.executors[
            action
        ]

        return True

    # ========================================================================
    # GET EXECUTOR
    # ========================================================================

    def get_executor(
        self,
        action: str,
    ) -> Optional[
        Callable[..., Any]
    ]:
        """
        Retrieve a registered executor.
        """

        return self.executors.get(
            action
        )

    # ========================================================================
    # RESOLVE EXECUTOR FROM SERVICE REGISTRY
    # ========================================================================

    def _resolve_from_service_registry(
        self,
        action: str,
    ) -> Optional[
        Callable[..., Any]
    ]:
        """
        Attempt to resolve an action from the service registry.
        """

        registry = (
            self.service_registry
        )

        if registry is None:

            return None

        try:

            if hasattr(
                registry,
                "get_service",
            ):

                service = (
                    registry.get_service(
                        action
                    )
                )

                if callable(service):

                    return service

                if service is not None:

                    if hasattr(
                        service,
                        "execute",
                    ):

                        return service.execute

                    if hasattr(
                        service,
                        "run",
                    ):

                        return service.run

            if hasattr(
                registry,
                "resolve",
            ):

                service = (
                    registry.resolve(
                        action
                    )
                )

                if callable(service):

                    return service

                if service is not None:

                    if hasattr(
                        service,
                        "execute",
                    ):

                        return service.execute

                    if hasattr(
                        service,
                        "run",
                    ):

                        return service.run

        except Exception as exc:

            self.logger.warning(
                "Service resolution failed for %s: %s",
                action,
                exc,
            )

        return None

    # ========================================================================
    # RESOLVE EXECUTOR
    # ========================================================================

    def resolve_executor(
        self,
        action: str,
    ) -> Optional[
        Callable[..., Any]
    ]:
        """
        Resolve an executor.

        Resolution order:
        1. Locally registered executor
        2. Service registry
        3. Capability manager
        """

        executor = (
            self.get_executor(
                action
            )
        )

        if executor is not None:

            return executor

        executor = (
            self._resolve_from_service_registry(
                action
            )
        )

        if executor is not None:

            return executor

        manager = (
            self.capability_manager
        )

        if manager is not None:

            try:

                if hasattr(
                    manager,
                    "get_executor",
                ):

                    executor = (
                        manager.get_executor(
                            action
                        )
                    )

                    if callable(
                        executor
                    ):

                        return executor

                if hasattr(
                    manager,
                    "get_capability",
                ):

                    capability = (
                        manager.get_capability(
                            action
                        )
                    )

                    if callable(
                        capability
                    ):

                        return capability

                    if capability is not None:

                        if hasattr(
                            capability,
                            "execute",
                        ):

                            return (
                                capability.execute
                            )

                        if hasattr(
                            capability,
                            "run",
                        ):

                            return (
                                capability.run
                            )

            except Exception as exc:

                self.logger.warning(
                    "Capability resolution failed for %s: %s",
                    action,
                    exc,
                )

        return None

    # ========================================================================
    # CHECK DECISION
    # ========================================================================

    def validate_decision(
        self,
        decision: Any,
    ) -> tuple[
        bool,
        Optional[str],
    ]:
        """
        Validate whether a decision is executable.
        """

        if decision is None:

            return (
                False,
                "Decision is missing.",
            )

        status = getattr(
            decision,
            "status",
            None,
        )

        if status not in {
            "approved",
            "executing",
        }:

            return (
                False,
                (
                    "Decision is not approved. "
                    f"Current status: {status}"
                ),
            )

        option = getattr(
            decision,
            "selected_option",
            None,
        )

        if option is None:

            return (
                False,
                "Decision has no selected option.",
            )

        if getattr(
            option,
            "blocked",
            False,
        ):

            return (
                False,
                "Selected option is blocked.",
            )

        if not getattr(
            option,
            "available",
            True,
        ):

            return (
                False,
                "Selected option is unavailable.",
            )

        return True, None

    # ========================================================================
    # CALL EXECUTOR
    # ========================================================================

    async def _call_executor(
        self,
        executor: Callable[..., Any],
        parameters: dict[str, Any],
    ) -> Any:
        """
        Execute a callable regardless of whether
        it is synchronous or asynchronous.
        """

        result = executor(
            **parameters
        )

        if inspect.isawaitable(
            result
        ):

            return await result

        return result

    # ========================================================================
    # EMIT EVENT
    # ========================================================================

    async def _emit_event(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Send an execution event through the event bus
        when available.
        """

        if self.event_bus is None:

            return

        try:

            if hasattr(
                self.event_bus,
                "publish",
            ):

                result = (
                    self.event_bus.publish(
                        event_name,
                        payload,
                    )
                )

                if inspect.isawaitable(
                    result
                ):

                    await result

                return

            if hasattr(
                self.event_bus,
                "emit",
            ):

                result = (
                    self.event_bus.emit(
                        event_name,
                        payload,
                    )
                )

                if inspect.isawaitable(
                    result
                ):

                    await result

        except Exception as exc:

            self.logger.debug(
                "Unable to emit event %s: %s",
                event_name,
                exc,
            )

    # ========================================================================
    # EXECUTE
    # ========================================================================

    async def execute(
        self,
        action: str,
        parameters: Optional[
            dict[str, Any]
        ] = None,
        *,
        decision_id: Optional[str] = None,
        timeout: Optional[float] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ExecutionResult:
        """
        Execute an action.
        """

        if not action:

            raise ValueError(
                "Action cannot be empty."
            )

        parameters = (
            dict(parameters)
            if parameters
            else {}
        )

        execution_id = (
            self._generate_id()
        )

        result = ExecutionResult(
            execution_id=execution_id,
            decision_id=decision_id,
            action=action,
            status="pending",
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        executor = (
            self.resolve_executor(
                action
            )
        )

        if executor is None:

            result.status = "failed"

            result.success = False

            result.error = (
                f"No executor registered "
                f"for action '{action}'."
            )

            result.completed_at = time.time()

            self.history[
                execution_id
            ] = result

            self.failure_count += 1

            self.execution_count += 1

            await self._emit_event(
                "execution.failed",
                result.to_dict(),
            )

            return result

        execution_task = ExecutionTask(
            execution_id=execution_id,
            action=action,
            parameters=parameters,
            decision_id=decision_id,
            status="running",
            started_at=time.time(),
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        self.active_tasks[
            execution_id
        ] = execution_task

        result.status = "running"

        result.started_at = (
            execution_task.started_at
        )

        self.execution_count += 1

        await self._emit_event(
            "execution.started",
            result.to_dict(),
        )

        timeout_value = (
            timeout
            if timeout is not None
            else self.default_timeout
        )

        try:

            execution_task.task = (
                asyncio.current_task()
            )

            if (
                timeout_value is not None
                and timeout_value > 0
            ):

                output = await asyncio.wait_for(
                    self._call_executor(
                        executor,
                        parameters,
                    ),
                    timeout=timeout_value,
                )

            else:

                output = await self._call_executor(
                    executor,
                    parameters,
                )

            result.result = output

            result.status = "completed"

            result.success = True

            result.completed_at = (
                time.time()
            )

            result.duration = (
                result.completed_at
                - result.started_at
            )

            execution_task.status = (
                "completed"
            )

            execution_task.completed_at = (
                result.completed_at
            )

            self.success_count += 1

            await self._emit_event(
                "execution.completed",
                result.to_dict(),
            )

        except asyncio.TimeoutError:

            result.status = "timeout"

            result.success = False

            result.timed_out = True

            result.error = (
                f"Execution timed out "
                f"after {timeout_value} seconds."
            )

            result.completed_at = (
                time.time()
            )

            result.duration = (
                result.completed_at
                - result.started_at
            )

            execution_task.status = (
                "timeout"
            )

            execution_task.completed_at = (
                result.completed_at
            )

            self.timeout_count += 1

            self.failure_count += 1

            await self._emit_event(
                "execution.timeout",
                result.to_dict(),
            )

        except asyncio.CancelledError:

            result.status = "cancelled"

            result.success = False

            result.cancelled = True

            result.error = (
                "Execution was cancelled."
            )

            result.completed_at = (
                time.time()
            )

            if result.started_at:

                result.duration = (
                    result.completed_at
                    - result.started_at
                )

            execution_task.status = (
                "cancelled"
            )

            execution_task.completed_at = (
                result.completed_at
            )

            self.cancelled_count += 1

            await self._emit_event(
                "execution.cancelled",
                result.to_dict(),
            )

        except Exception as exc:

            result.status = "failed"

            result.success = False

            result.error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            result.completed_at = (
                time.time()
            )

            if result.started_at:

                result.duration = (
                    result.completed_at
                    - result.started_at
                )

            execution_task.status = (
                "failed"
            )

            execution_task.completed_at = (
                result.completed_at
            )

            self.failure_count += 1

            self.logger.exception(
                "Execution failed for action '%s'.",
                action,
            )

            await self._emit_event(
                "execution.failed",
                result.to_dict(),
            )

        finally:

            self.active_tasks.pop(
                execution_id,
                None,
            )

            self.history[
                execution_id
            ] = result

            self._trim_history()

        return result

    # ========================================================================
    # EXECUTE DECISION
    # ========================================================================

    async def execute_decision(
        self,
        decision: Any,
        *,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute the selected option from a Decision object.
        """

        valid, reason = (
            self.validate_decision(
                decision
            )
        )

        if not valid:

            execution_id = (
                self._generate_id()
            )

            result = ExecutionResult(
                execution_id=execution_id,
                decision_id=getattr(
                    decision,
                    "decision_id",
                    None,
                ),
                action=None,
                status="rejected",
                success=False,
                error=reason,
            )

            result.completed_at = (
                time.time()
            )

            self.history[
                execution_id
            ] = result

            self.failure_count += 1

            return result

        option = (
            decision.selected_option
        )

        action = getattr(
            option,
            "action",
            None,
        )

        if not action:

            execution_id = (
                self._generate_id()
            )

            result = ExecutionResult(
                execution_id=execution_id,
                decision_id=getattr(
                    decision,
                    "decision_id",
                    None,
                ),
                action=None,
                status="failed",
                success=False,
                error=(
                    "Selected option does not "
                    "contain an executable action."
                ),
            )

            result.completed_at = (
                time.time()
            )

            self.history[
                execution_id
            ] = result

            self.failure_count += 1

            return result

        # --------------------------------------------------------------------
        # Update decision status
        # --------------------------------------------------------------------

        if self.decision_engine is not None:

            try:

                if hasattr(
                    self.decision_engine,
                    "mark_executing",
                ):

                    self.decision_engine.mark_executing(
                        decision
                    )

            except Exception as exc:

                self.logger.debug(
                    "Unable to update decision state: %s",
                    exc,
                )

        else:

            try:

                decision.status = (
                    "executing"
                )

            except Exception:

                pass

        result = await self.execute(
            action,
            getattr(
                option,
                "parameters",
                {},
            ),
            decision_id=getattr(
                decision,
                "decision_id",
                None,
            ),
            timeout=timeout,
            metadata={
                "decision_objective": getattr(
                    decision,
                    "objective",
                    None,
                ),
                "option_name": getattr(
                    option,
                    "name",
                    None,
                ),
            },
        )

        # --------------------------------------------------------------------
        # Update decision after execution
        # --------------------------------------------------------------------

        if self.decision_engine is not None:

            try:

                if result.success:

                    if hasattr(
                        self.decision_engine,
                        "mark_completed",
                    ):

                        self.decision_engine.mark_completed(
                            decision,
                            result.result,
                        )

                else:

                    if hasattr(
                        self.decision_engine,
                        "mark_failed",
                    ):

                        self.decision_engine.mark_failed(
                            decision,
                            result.error,
                        )

            except Exception as exc:

                self.logger.debug(
                    "Unable to update final decision state: %s",
                    exc,
                )

        else:

            try:

                if result.success:

                    decision.status = (
                        "completed"
                    )

                else:

                    decision.status = (
                        "failed"
                    )

            except Exception:

                pass

        return result

    # ========================================================================
    # EXECUTE IN BACKGROUND
    # ========================================================================

    async def execute_background(
        self,
        action: str,
        parameters: Optional[
            dict[str, Any]
        ] = None,
        *,
        decision_id: Optional[str] = None,
        timeout: Optional[float] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Start an action in the background.

        Returns the execution ID.
        """

        execution_id = (
            self._generate_id()
        )

        # --------------------------------------------------------------
        # Background wrapper
        # --------------------------------------------------------------

        async def runner() -> None:

            await self._execute_with_id(
                execution_id=execution_id,
                action=action,
                parameters=parameters,
                decision_id=decision_id,
                timeout=timeout,
                metadata=metadata,
            )

        task = asyncio.create_task(
            runner()
        )

        self.active_tasks[
            execution_id
        ] = ExecutionTask(
            execution_id=execution_id,
            action=action,
            parameters=(
                dict(parameters)
                if parameters
                else {}
            ),
            decision_id=decision_id,
            status="queued",
            task=task,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        return execution_id

    # ========================================================================
    # INTERNAL EXECUTION WITH ID
    # ========================================================================

    async def _execute_with_id(
        self,
        execution_id: str,
        action: str,
        parameters: Optional[
            dict[str, Any]
        ],
        *,
        decision_id: Optional[str],
        timeout: Optional[float],
        metadata: Optional[
            dict[str, Any]
        ],
    ) -> ExecutionResult:
        """
        Internal execution method used for background jobs.
        """

        result = await self.execute(
            action,
            parameters,
            decision_id=decision_id,
            timeout=timeout,
            metadata=metadata,
        )

        # Keep the externally assigned ID available
        # through metadata when execute() creates its own ID.
        result.metadata[
            "background_execution_id"
        ] = execution_id

        return result

    # ========================================================================
    # CANCEL
    # ========================================================================

    async def cancel(
        self,
        execution_id: str,
    ) -> bool:
        """
        Cancel an active execution.
        """

        execution = self.active_tasks.get(
            execution_id
        )

        if execution is None:

            return False

        task = execution.task

        if task is None:

            execution.status = (
                "cancelled"
            )

            return True

        if task.done():

            return False

        task.cancel()

        try:

            await task

        except asyncio.CancelledError:

            pass

        except Exception:

            pass

        return True

    # ========================================================================
    # GET RESULT
    # ========================================================================

    def get_result(
        self,
        execution_id: str,
    ) -> Optional[
        ExecutionResult
    ]:
        """
        Get an execution result.
        """

        return self.history.get(
            execution_id
        )

    # ========================================================================
    # GET ACTIVE EXECUTION
    # ========================================================================

    def get_active_execution(
        self,
        execution_id: str,
    ) -> Optional[
        ExecutionTask
    ]:
        """
        Get an active execution.
        """

        return self.active_tasks.get(
            execution_id
        )

    # ========================================================================
    # LIST ACTIVE
    # ========================================================================

    def list_active(
        self,
    ) -> list[ExecutionTask]:
        """
        Return all currently active executions.
        """

        return list(
            self.active_tasks.values()
        )

    # ========================================================================
    # LIST HISTORY
    # ========================================================================

    def list_history(
        self,
        status: Optional[str] = None,
    ) -> list[ExecutionResult]:
        """
        Return execution history.
        """

        results = list(
            self.history.values()
        )

        if status is not None:

            results = [
                result
                for result in results
                if result.status
                == status
            ]

        results.sort(
            key=lambda result: (
                result.completed_at
                or result.started_at
                or 0.0
            ),
            reverse=True,
        )

        return results[
            :self.max_history
        ]

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear execution history.
        """

        self.history.clear()

    # ========================================================================
    # TRIM HISTORY
    # ========================================================================

    def _trim_history(
        self,
    ) -> None:
        """
        Keep history within the configured limit.
        """

        if len(
            self.history
        ) <= self.max_history:

            return

        ordered = sorted(
            self.history.items(),
            key=lambda item: (
                item[1].completed_at
                or item[1].started_at
                or 0.0
            ),
            reverse=True,
        )

        keep = dict(
            ordered[
                :self.max_history
            ]
        )

        self.history = keep

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return execution statistics.
        """

        total = (
            self.execution_count
        )

        success_rate = (
            self.success_count / total
            if total > 0
            else 0.0
        )

        return {
            "initialized": (
                self.initialized
            ),
            "running": (
                self.running
            ),
            "registered_executors": len(
                self.executors
            ),
            "active_executions": len(
                self.active_tasks
            ),
            "history_size": len(
                self.history
            ),
            "execution_count": (
                self.execution_count
            ),
            "success_count": (
                self.success_count
            ),
            "failure_count": (
                self.failure_count
            ),
            "cancelled_count": (
                self.cancelled_count
            ),
            "timeout_count": (
                self.timeout_count
            ),
            "success_rate": (
                success_rate
            ),
            "default_timeout": (
                self.default_timeout
            ),
            "created_at": (
                self.created_at
            ),
        }


# ============================================================================
# GLOBAL ENGINE
# ============================================================================

execution_engine = ExecutionEngine()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def execute(
    action: str,
    parameters: Optional[
        dict[str, Any]
    ] = None,
    *,
    decision_id: Optional[str] = None,
    timeout: Optional[float] = None,
) -> ExecutionResult:
    """
    Convenience wrapper around the global execution engine.
    """

    return await execution_engine.execute(
        action,
        parameters,
        decision_id=decision_id,
        timeout=timeout,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ExecutionResult",
    "ExecutionTask",
    "ExecutionEngine",
    "execution_engine",
    "execute",
]


