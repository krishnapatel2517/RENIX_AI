"""
RENIX AI
Base Agent

The foundational agent class used by every specialized RENIX agent.

All agents should inherit from BaseAgent so the rest of RENIX can interact
with them through one consistent interface.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# ENUMS
# ============================================================================


class AgentStatus(str, Enum):
    """Current lifecycle state of an agent."""

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AgentPriority(str, Enum):
    """Execution priority of an agent."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AgentResultStatus(str, Enum):
    """Status of an individual agent execution."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_CONFIRMATION = "requires_confirmation"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class AgentTask:
    """A task assigned to an agent."""

    task_id: str

    instruction: str

    parameters: Dict[str, Any] = field(
        default_factory=dict
    )

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    priority: AgentPriority = (
        AgentPriority.NORMAL
    )

    parent_task_id: Optional[str] = None

    created_at: float = field(
        default_factory=time.time
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def create(
        cls,
        instruction: str,
        *,
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
        priority: AgentPriority = AgentPriority.NORMAL,
        parent_task_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> "AgentTask":
        """Create a new agent task."""

        return cls(
            task_id=str(
                uuid.uuid4()
            ),
            instruction=instruction,
            parameters=dict(
                parameters or {}
            ),
            context=dict(
                context or {}
            ),
            priority=priority,
            parent_task_id=parent_task_id,
            metadata=dict(
                metadata or {}
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary."""

        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "parameters": dict(
                self.parameters
            ),
            "context": dict(
                self.context
            ),
            "priority": self.priority.value,
            "parent_task_id": self.parent_task_id,
            "created_at": self.created_at,
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class AgentResult:
    """Standard result returned by an agent."""

    task_id: str

    status: AgentResultStatus

    output: Any = None

    message: str = ""

    error: Optional[str] = None

    started_at: Optional[float] = None

    completed_at: Optional[float] = None

    execution_time: float = 0.0

    confidence: float = 1.0

    requires_confirmation: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def success(self) -> bool:
        """Return True when execution succeeded."""

        return self.status in {
            AgentResultStatus.SUCCESS,
            AgentResultStatus.PARTIAL,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary."""

        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class AgentCapability:
    """Describes one capability exposed by an agent."""

    name: str

    description: str

    enabled: bool = True

    requires_confirmation: bool = False

    dangerous: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary."""

        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "dangerous": self.dangerous,
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class AgentEvent:
    """Event generated during agent execution."""

    event_type: str

    agent_id: str

    timestamp: float = field(
        default_factory=time.time
    )

    task_id: Optional[str] = None

    data: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""

        return {
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "data": dict(
                self.data
            ),
        }


# ============================================================================
# BASE AGENT
# ============================================================================


class BaseAgent(ABC):
    """
    Base class for all RENIX agents.

    Specialized agents should inherit from this class and implement:

        execute()

    Optional lifecycle hooks can also be overridden.
    """

    # ------------------------------------------------------------------------
    # CLASS METADATA
    # ------------------------------------------------------------------------

    agent_name: str = "base_agent"

    agent_description: str = (
        "Base RENIX autonomous agent."
    )

    agent_version: str = "1.0.0"

    # ------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: AgentPriority = AgentPriority.NORMAL,
        logger: Optional[
            logging.Logger
        ] = None,
        event_callback: Optional[
            Callable[[AgentEvent], Any]
        ] = None,
        confirmation_callback: Optional[
            Callable[..., Any]
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:

        self.agent_id = (
            agent_id
            or str(uuid.uuid4())
        )

        self.name = (
            name
            or self.agent_name
        )

        self.description = (
            description
            or self.agent_description
        )

        self.priority = priority

        self.logger = (
            logger
            or logging.getLogger(
                f"renix.agent.{self.name}"
            )
        )

        self.event_callback = (
            event_callback
        )

        self.confirmation_callback = (
            confirmation_callback
        )

        self.context: Dict[str, Any] = dict(
            context or {}
        )

        self.status = AgentStatus.CREATED

        self.current_task: Optional[
            AgentTask
        ] = None

        self.last_result: Optional[
            AgentResult
        ] = None

        self.execution_count = 0

        self.success_count = 0

        self.failure_count = 0

        self.created_at = time.time()

        self.started_at: Optional[
            float
        ] = None

        self.completed_at: Optional[
            float
        ] = None

        self._stop_requested = False

        self._pause_requested = False

        self._capabilities: Dict[
            str,
            AgentCapability,
        ] = {}

        self._register_default_capabilities()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    async def initialize(
        self,
    ) -> bool:
        """
        Initialize the agent.

        Returns True when initialization succeeds.
        """

        try:

            self.status = (
                AgentStatus.INITIALIZING
            )

            self.emit_event(
                "agent.initializing"
            )

            result = self.on_initialize()

            if inspect.isawaitable(
                result
            ):
                result = await result

            if result is False:

                self.status = (
                    AgentStatus.FAILED
                )

                self.emit_event(
                    "agent.initialization_failed"
                )

                return False

            self.status = (
                AgentStatus.READY
            )

            self.emit_event(
                "agent.ready"
            )

            return True

        except Exception as exc:

            self.status = (
                AgentStatus.FAILED
            )

            self.logger.exception(
                "Agent initialization failed: %s",
                exc,
            )

            self.emit_event(
                "agent.initialization_failed",
                error=str(exc),
            )

            return False

    async def start(
        self,
    ) -> bool:
        """
        Start the agent.

        If the agent has not been initialized yet,
        initialization is performed automatically.
        """

        if self.status == AgentStatus.CREATED:

            initialized = await self.initialize()

            if not initialized:
                return False

        if self.status in {
            AgentStatus.FAILED,
            AgentStatus.STOPPED,
        }:

            return False

        try:

            result = self.on_start()

            if inspect.isawaitable(
                result
            ):
                result = await result

            self.emit_event(
                "agent.started"
            )

            return (
                True
                if result is not False
                else False
            )

        except Exception as exc:

            self.status = (
                AgentStatus.FAILED
            )

            self.logger.exception(
                "Agent start failed: %s",
                exc,
            )

            self.emit_event(
                "agent.start_failed",
                error=str(exc),
            )

            return False

    async def stop(
        self,
    ) -> None:
        """Stop the agent."""

        self._stop_requested = True

        try:

            result = self.on_stop()

            if inspect.isawaitable(
                result
            ):
                await result

        except Exception as exc:

            self.logger.exception(
                "Agent stop hook failed: %s",
                exc,
            )

        finally:

            self.status = (
                AgentStatus.STOPPED
            )

            self.emit_event(
                "agent.stopped"
            )

    async def pause(
        self,
    ) -> None:
        """Pause future execution where supported."""

        self._pause_requested = True

        self.status = (
            AgentStatus.PAUSED
        )

        self.emit_event(
            "agent.paused"
        )

        result = self.on_pause()

        if inspect.isawaitable(
            result
        ):
            await result

    async def resume(
        self,
    ) -> None:
        """Resume execution."""

        self._pause_requested = False

        if self.status == AgentStatus.PAUSED:

            self.status = (
                AgentStatus.READY
            )

        self.emit_event(
            "agent.resumed"
        )

        result = self.on_resume()

        if inspect.isawaitable(
            result
        ):
            await result

    # ========================================================================
    # EXECUTION
    # ========================================================================

    async def run(
        self,
        task: AgentTask | str,
        *,
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> AgentResult:
        """
        Execute one task through the complete agent lifecycle.

        This method handles:
        - task normalization
        - state transitions
        - validation
        - confirmation
        - execution
        - errors
        - metrics
        - events
        """

        normalized_task = (
            self._normalize_task(
                task,
                parameters=parameters,
                context=context,
            )
        )

        self.current_task = (
            normalized_task
        )

        self.execution_count += 1

        started = time.time()

        self.started_at = started

        self._stop_requested = False

        # --------------------------------------------------------------------
        # STATE
        # --------------------------------------------------------------------

        if self.status in {
            AgentStatus.CREATED,
            AgentStatus.FAILED,
        }:

            initialized = await self.initialize()

            if not initialized:

                return self._failed_result(
                    normalized_task,
                    "Agent initialization failed.",
                    started,
                )

        if self.status == AgentStatus.STOPPED:

            return self._failed_result(
                normalized_task,
                "Agent is stopped.",
                started,
            )

        # --------------------------------------------------------------------
        # PAUSE
        # --------------------------------------------------------------------

        while self._pause_requested:

            self.status = (
                AgentStatus.PAUSED
            )

            await asyncio.sleep(
                0.05
            )

            if self._stop_requested:

                return self._cancelled_result(
                    normalized_task,
                    started,
                )

        # --------------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------------

        try:

            validation = self.validate_task(
                normalized_task
            )

            if inspect.isawaitable(
                validation
            ):

                validation = await validation

            if validation is False:

                self.failure_count += 1

                return self._failed_result(
                    normalized_task,
                    "Task validation failed.",
                    started,
                )

        except Exception as exc:

            self.failure_count += 1

            return self._failed_result(
                normalized_task,
                str(exc),
                started,
            )

        # --------------------------------------------------------------------
        # CONFIRMATION
        # --------------------------------------------------------------------

        if self.requires_confirmation(
            normalized_task
        ):

            confirmed = await self.request_confirmation(
                normalized_task
            )

            if not confirmed:

                result = AgentResult(
                    task_id=normalized_task.task_id,
                    status=(
                        AgentResultStatus.REQUIRES_CONFIRMATION
                    ),
                    message=(
                        "Execution requires confirmation."
                    ),
                    started_at=started,
                    completed_at=time.time(),
                    execution_time=(
                        time.time()
                        - started
                    ),
                    requires_confirmation=True,
                )

                self.last_result = result

                self.status = (
                    AgentStatus.WAITING
                )

                self.emit_event(
                    "agent.confirmation_required",
                    task_id=(
                        normalized_task.task_id
                    ),
                )

                return result

        # --------------------------------------------------------------------
        # EXECUTION
        # --------------------------------------------------------------------

        self.status = (
            AgentStatus.RUNNING
        )

        self.emit_event(
            "agent.task_started",
            task_id=(
                normalized_task.task_id
            ),
            instruction=(
                normalized_task.instruction
            ),
        )

        try:

            output = self.execute(
                normalized_task
            )

            if inspect.isawaitable(
                output
            ):

                output = await output

            if self._stop_requested:

                return self._cancelled_result(
                    normalized_task,
                    started,
                )

            result = self._normalize_result(
                normalized_task,
                output,
                started,
            )

            self.last_result = result

            if result.success:

                self.success_count += 1

                self.status = (
                    AgentStatus.COMPLETED
                )

            else:

                self.failure_count += 1

                self.status = (
                    AgentStatus.FAILED
                )

            self.completed_at = (
                result.completed_at
            )

            self.emit_event(
                "agent.task_completed",
                task_id=(
                    normalized_task.task_id
                ),
                status=result.status.value,
            )

            return result

        except asyncio.CancelledError:

            self.failure_count += 1

            result = self._cancelled_result(
                normalized_task,
                started,
            )

            self.last_result = result

            raise

        except Exception as exc:

            self.failure_count += 1

            self.logger.exception(
                "Agent execution failed: %s",
                exc,
            )

            result = self._failed_result(
                normalized_task,
                str(exc),
                started,
            )

            self.last_result = result

            self.status = (
                AgentStatus.FAILED
            )

            self.emit_event(
                "agent.task_failed",
                task_id=(
                    normalized_task.task_id
                ),
                error=str(exc),
            )

            return result

        finally:

            self.current_task = None

    # ========================================================================
    # ABSTRACT EXECUTION
    # ========================================================================

    @abstractmethod
    async def execute(
        self,
        task: AgentTask,
    ) -> Any:
        """
        Execute an agent task.

        Every specialized RENIX agent must implement this method.

        The method may return:
        - raw output
        - AgentResult
        - dictionary
        - None
        """

        raise NotImplementedError

    # ========================================================================
    # TASK VALIDATION
    # ========================================================================

    def validate_task(
        self,
        task: AgentTask,
    ) -> bool:
        """
        Validate an incoming task.

        Subclasses may override this method.
        """

        if not task.instruction.strip():

            return False

        return True

    # ========================================================================
    # CONFIRMATION
    # ========================================================================

    def requires_confirmation(
        self,
        task: AgentTask,
    ) -> bool:
        """
        Determine whether a task requires confirmation.

        A task requires confirmation when its requested capability is
        marked as dangerous or confirmation-required.
        """

        capability_name = self._get_requested_capability(
            task
        )

        if not capability_name:
            return False

        capability = self._capabilities.get(
            capability_name
        )

        if capability is None:
            return False

        return (
            capability.requires_confirmation
            or capability.dangerous
        )

    async def request_confirmation(
        self,
        task: AgentTask,
    ) -> bool:
        """
        Request user confirmation.

        If no confirmation callback exists, confirmation is denied.
        """

        self.emit_event(
            "agent.confirmation_requested",
            task_id=task.task_id,
        )

        if self.confirmation_callback is None:

            return False

        try:

            result = self.confirmation_callback(
                task=task,
                agent=self,
            )

            if inspect.isawaitable(
                result
            ):
                result = await result

            return bool(result)

        except Exception as exc:

            self.logger.exception(
                "Confirmation callback failed: %s",
                exc,
            )

            return False

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def register_capability(
        self,
        capability: AgentCapability,
    ) -> None:
        """Register an agent capability."""

        self._capabilities[
            capability.name
        ] = capability

        self.emit_event(
            "agent.capability_registered",
            capability=capability.to_dict(),
        )

    def unregister_capability(
        self,
        name: str,
    ) -> bool:
        """Remove an agent capability."""

        if name not in self._capabilities:
            return False

        del self._capabilities[
            name
        ]

        self.emit_event(
            "agent.capability_unregistered",
            capability=name,
        )

        return True

    def has_capability(
        self,
        name: str,
    ) -> bool:
        """Check whether a capability exists and is enabled."""

        capability = self._capabilities.get(
            name
        )

        return bool(
            capability
            and capability.enabled
        )

    def get_capability(
        self,
        name: str,
    ) -> Optional[AgentCapability]:
        """Return a capability."""

        return self._capabilities.get(
            name
        )

    def get_capabilities(
        self,
    ) -> List[AgentCapability]:
        """Return all capabilities."""

        return list(
            self._capabilities.values()
        )

    def capability_names(
        self,
    ) -> List[str]:
        """Return capability names."""

        return list(
            self._capabilities.keys()
        )

    def _register_default_capabilities(
        self,
    ) -> None:
        """
        Register generic capabilities.

        Specialized agents should override this method and call
        super() when appropriate.
        """

        self.register_capability(
            AgentCapability(
                name="execute",
                description=(
                    "Execute an assigned task."
                ),
            )
        )

        self.register_capability(
            AgentCapability(
                name="status",
                description=(
                    "Report agent status."
                ),
            )
        )

    def _get_requested_capability(
        self,
        task: AgentTask,
    ) -> Optional[str]:
        """Extract the requested capability from a task."""

        capability = task.parameters.get(
            "capability"
        )

        if capability:
            return str(
                capability
            )

        return None

    # ========================================================================
    # CONTEXT
    # ========================================================================

    def set_context(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set shared agent context."""

        self.context[key] = value

        self.emit_event(
            "agent.context_updated",
            key=key,
        )

    def get_context(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Read shared agent context."""

        return self.context.get(
            key,
            default,
        )

    def update_context(
        self,
        values: Dict[str, Any],
    ) -> None:
        """Update multiple context values."""

        self.context.update(
            values
        )

        self.emit_event(
            "agent.context_updated",
            keys=list(
                values.keys()
            ),
        )

    def clear_context(
        self,
    ) -> None:
        """Clear shared context."""

        self.context.clear()

        self.emit_event(
            "agent.context_cleared"
        )

    # ========================================================================
    # EVENTS
    # ========================================================================

    def emit_event(
        self,
        event_type: str,
        **data: Any,
    ) -> AgentEvent:
        """Emit an agent event."""

        event = AgentEvent(
            event_type=event_type,
            agent_id=self.agent_id,
            task_id=(
                self.current_task.task_id
                if self.current_task
                else None
            ),
            data=data,
        )

        if self.event_callback:

            try:

                callback_result = (
                    self.event_callback(
                        event
                    )
                )

                # The callback is intentionally not awaited here because
                # emit_event is synchronous. Async callers can provide a
                # synchronous bridge or override this behavior.

                if inspect.isawaitable(
                    callback_result
                ):

                    # Avoid silently leaking a coroutine.
                    try:
                        callback_result.close()
                    except Exception:
                        pass

            except Exception as exc:

                self.logger.exception(
                    "Agent event callback failed: %s",
                    exc,
                )

        return event

    # ========================================================================
    # LIFECYCLE HOOKS
    # ========================================================================

    def on_initialize(
        self,
    ) -> Any:
        """Optional initialization hook."""

        return True

    def on_start(
        self,
    ) -> Any:
        """Optional start hook."""

        return True

    def on_stop(
        self,
    ) -> Any:
        """Optional stop hook."""

        return True

    def on_pause(
        self,
    ) -> Any:
        """Optional pause hook."""

        return True

    def on_resume(
        self,
    ) -> Any:
        """Optional resume hook."""

        return True

    # ========================================================================
    # RESULT HELPERS
    # ========================================================================

    def success_result(
        self,
        task: AgentTask,
        output: Any = None,
        *,
        message: str = "",
        confidence: float = 1.0,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        started_at: Optional[
            float
        ] = None,
    ) -> AgentResult:
        """Create a successful result."""

        completed = time.time()

        started = (
            started_at
            if started_at is not None
            else completed
        )

        return AgentResult(
            task_id=task.task_id,
            status=AgentResultStatus.SUCCESS,
            output=output,
            message=message,
            started_at=started,
            completed_at=completed,
            execution_time=(
                completed - started
            ),
            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            metadata=dict(
                metadata or {}
            ),
        )

    def partial_result(
        self,
        task: AgentTask,
        output: Any = None,
        *,
        message: str = "",
        confidence: float = 0.5,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        started_at: Optional[
            float
        ] = None,
    ) -> AgentResult:
        """Create a partial-success result."""

        completed = time.time()

        started = (
            started_at
            if started_at is not None
            else completed
        )

        return AgentResult(
            task_id=task.task_id,
            status=AgentResultStatus.PARTIAL,
            output=output,
            message=message,
            started_at=started,
            completed_at=completed,
            execution_time=(
                completed - started
            ),
            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            metadata=dict(
                metadata or {}
            ),
        )

    def failure_result(
        self,
        task: AgentTask,
        error: str,
        *,
        output: Any = None,
        message: str = "",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        started_at: Optional[
            float
        ] = None,
    ) -> AgentResult:
        """Create a failed result."""

        completed = time.time()

        started = (
            started_at
            if started_at is not None
            else completed
        )

        return AgentResult(
            task_id=task.task_id,
            status=AgentResultStatus.FAILED,
            output=output,
            message=message,
            error=error,
            started_at=started,
            completed_at=completed,
            execution_time=(
                completed - started
            ),
            confidence=0.0,
            metadata=dict(
                metadata or {}
            ),
        )

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
    ) -> AgentStatus:
        """Return current status."""

        return self.status

    def is_ready(
        self,
    ) -> bool:
        """Return whether the agent is ready."""

        return self.status == AgentStatus.READY

    def is_running(
        self,
    ) -> bool:
        """Return whether the agent is running."""

        return self.status == AgentStatus.RUNNING

    def is_busy(
        self,
    ) -> bool:
        """Return whether the agent is currently occupied."""

        return self.status in {
            AgentStatus.INITIALIZING,
            AgentStatus.RUNNING,
            AgentStatus.WAITING,
            AgentStatus.PAUSED,
        }

    # ========================================================================
    # METRICS
    # ========================================================================

    def get_metrics(
        self,
    ) -> Dict[str, Any]:
        """Return execution metrics."""

        success_rate = 0.0

        if self.execution_count:

            success_rate = (
                self.success_count
                / self.execution_count
            )

        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.agent_version,
            "status": self.status.value,
            "execution_count": (
                self.execution_count
            ),
            "success_count": (
                self.success_count
            ),
            "failure_count": (
                self.failure_count
            ),
            "success_rate": success_rate,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Serialize agent information."""

        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.agent_version,
            "priority": self.priority.value,
            "status": self.status.value,
            "capabilities": [
                capability.to_dict()
                for capability in self.get_capabilities()
            ],
            "context": dict(
                self.context
            ),
            "metrics": self.get_metrics(),
            "last_result": (
                self.last_result.to_dict()
                if self.last_result
                else None
            ),
        }

    def __repr__(
        self,
    ) -> str:
        """Developer-friendly representation."""

        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id!r}, "
            f"name={self.name!r}, "
            f"status={self.status.value!r}"
            f")"
        )

    # ========================================================================
    # INTERNAL TASK NORMALIZATION
    # ========================================================================

    def _normalize_task(
        self,
        task: AgentTask | str,
        *,
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        context: Optional[
            Dict[str, Any]
        ] = None,
    ) -> AgentTask:
        """Normalize a string or AgentTask into AgentTask."""

        if isinstance(
            task,
            AgentTask,
        ):

            if parameters:
                task.parameters.update(
                    parameters
                )

            if context:
                task.context.update(
                    context
                )

            return task

        return AgentTask.create(
            instruction=str(
                task
            ),
            parameters=parameters,
            context=context,
            priority=self.priority,
        )

    def _normalize_result(
        self,
        task: AgentTask,
        output: Any,
        started: float,
    ) -> AgentResult:
        """Normalize arbitrary execute() output."""

        if isinstance(
            output,
            AgentResult,
        ):

            if output.started_at is None:
                output.started_at = started

            if output.completed_at is None:
                output.completed_at = time.time()

            output.execution_time = (
                output.completed_at
                - output.started_at
            )

            return output

        if isinstance(
            output,
            dict,
        ):

            status = output.get(
                "status",
                AgentResultStatus.SUCCESS.value,
            )

            try:

                result_status = (
                    AgentResultStatus(
                        str(status).lower()
                    )
                )

            except ValueError:

                result_status = (
                    AgentResultStatus.SUCCESS
                )

            completed = time.time()

            return AgentResult(
                task_id=task.task_id,
                status=result_status,
                output=output.get(
                    "output",
                    output,
                ),
                message=str(
                    output.get(
                        "message",
                        "",
                    )
                ),
                error=output.get(
                    "error"
                ),
                started_at=started,
                completed_at=completed,
                execution_time=(
                    completed - started
                ),
                confidence=float(
                    output.get(
                        "confidence",
                        1.0,
                    )
                ),
                requires_confirmation=bool(
                    output.get(
                        "requires_confirmation",
                        False,
                    )
                ),
                metadata=dict(
                    output.get(
                        "metadata",
                        {},
                    )
                ),
            )

        return self.success_result(
            task,
            output,
            started_at=started,
        )

    def _failed_result(
        self,
        task: AgentTask,
        error: str,
        started: float,
    ) -> AgentResult:
        """Create a failure result."""

        result = self.failure_result(
            task,
            error,
            started_at=started,
        )

        self.last_result = result

        self.status = (
            AgentStatus.FAILED
        )

        self.completed_at = (
            result.completed_at
        )

        self.emit_event(
            "agent.task_failed",
            task_id=task.task_id,
            error=error,
        )

        return result

    def _cancelled_result(
        self,
        task: AgentTask,
        started: float,
    ) -> AgentResult:
        """Create a cancellation result."""

        completed = time.time()

        result = AgentResult(
            task_id=task.task_id,
            status=(
                AgentResultStatus.CANCELLED
            ),
            message="Task cancelled.",
            started_at=started,
            completed_at=completed,
            execution_time=(
                completed - started
            ),
            confidence=0.0,
        )

        self.last_result = result

        self.status = (
            AgentStatus.STOPPED
        )

        self.emit_event(
            "agent.task_cancelled",
            task_id=task.task_id,
        )

        return result


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "AgentStatus",
    "AgentPriority",
    "AgentResultStatus",
    "AgentTask",
    "AgentResult",
    "AgentCapability",
    "AgentEvent",
    "BaseAgent",
]


