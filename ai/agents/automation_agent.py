"""
RENIX - Automation Agent

Handles automation-related requests for RENIX.

Responsibilities:
- Execute automation workflows
- Run predefined routines
- Manage automation triggers
- Execute scheduled automation
- Run individual automation actions
- Validate automation requests
- Provide automation status

The actual workflow/routine implementations belong to the
automation package. This agent acts as the AI-facing interface
for those capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import threading
import time
import uuid


class AutomationStatus(str, Enum):
    """Status of an automation execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AutomationResult:
    """Standard result returned by the automation agent."""

    success: bool
    message: str
    status: AutomationStatus
    automation_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "status": self.status.value,
            "automation_id": self.automation_id,
            "data": self.data,
            "error": self.error,
        }


@dataclass
class AutomationTask:
    """Represents a registered automation task."""

    automation_id: str
    name: str
    description: str
    action: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: AutomationStatus = AutomationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


class AutomationAgent:
    """
    AI-facing automation controller.

    The agent is intentionally designed so that the real automation
    engine can be connected later without changing the public API.
    """

    name = "automation_agent"
    description = "Controls RENIX automation workflows, routines and triggers."

    def __init__(self) -> None:
        self.running = True

        self._tasks: Dict[str, AutomationTask] = {}
        self._routines: Dict[str, Callable[..., Any]] = {}
        self._workflows: Dict[str, Callable[..., Any]] = {}
        self._triggers: Dict[str, Dict[str, Any]] = {}

        self._lock = threading.RLock()

    # ================================================================
    # MAIN ENTRY POINT
    # ================================================================

    def execute(
        self,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AutomationResult:
        """
        Execute an automation action.

        Supported actions:
        - run
        - run_routine
        - run_workflow
        - trigger
        - register_routine
        - register_workflow
        - cancel
        - status
        - list
        """

        parameters = parameters or {}

        if not action:
            return self._failure(
                "No automation action was provided.",
                "missing_action",
            )

        action = action.strip().lower()

        handlers = {
            "run": self._execute_registered_task,
            "execute": self._execute_registered_task,
            "run_routine": self._run_routine,
            "routine": self._run_routine,
            "run_workflow": self._run_workflow,
            "workflow": self._run_workflow,
            "trigger": self._trigger,
            "register_routine": self._register_routine_from_parameters,
            "register_workflow": self._register_workflow_from_parameters,
            "cancel": self._cancel_task,
            "status": self._get_status,
            "list": self._list_automations,
        }

        handler = handlers.get(action)

        if handler is None:
            return self._failure(
                f"Unsupported automation action: {action}",
                "unsupported_action",
            )

        try:
            result = handler(parameters)

            if isinstance(result, AutomationResult):
                return result

            return self._success(
                "Automation action completed.",
                data={"result": result},
            )

        except Exception as exc:
            return self._failure(
                "Automation action failed.",
                str(exc),
            )

    # ================================================================
    # TASK REGISTRATION
    # ================================================================

    def register_task(
        self,
        name: str,
        description: str,
        action: Callable[..., Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register an executable automation task.
        """

        if not name:
            raise ValueError("Task name cannot be empty.")

        if not callable(action):
            raise TypeError("action must be callable.")

        automation_id = self._generate_id()

        task = AutomationTask(
            automation_id=automation_id,
            name=name,
            description=description,
            action=action,
            parameters=parameters or {},
        )

        with self._lock:
            self._tasks[automation_id] = task

        return automation_id

    def register_routine(
        self,
        name: str,
        handler: Callable[..., Any],
    ) -> bool:
        """Register a reusable automation routine."""

        if not name:
            return False

        if not callable(handler):
            return False

        with self._lock:
            self._routines[name.lower()] = handler

        return True

    def register_workflow(
        self,
        name: str,
        handler: Callable[..., Any],
    ) -> bool:
        """Register a reusable automation workflow."""

        if not name:
            return False

        if not callable(handler):
            return False

        with self._lock:
            self._workflows[name.lower()] = handler

        return True

    # ================================================================
    # TASK EXECUTION
    # ================================================================

    def run_task(
        self,
        automation_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AutomationResult:
        """Execute a registered automation task."""

        with self._lock:
            task = self._tasks.get(automation_id)

        if task is None:
            return self._failure(
                f"Automation '{automation_id}' was not found.",
                "automation_not_found",
                automation_id=automation_id,
            )

        if not self.running:
            return self._failure(
                "Automation agent is stopped.",
                "agent_stopped",
                automation_id=automation_id,
            )

        task_parameters = dict(task.parameters)

        if parameters:
            task_parameters.update(parameters)

        task.status = AutomationStatus.RUNNING
        task.started_at = time.time()
        task.error = None

        try:
            result = task.action(**task_parameters)

            task.status = AutomationStatus.COMPLETED
            task.completed_at = time.time()

            return AutomationResult(
                success=True,
                message=f"Automation '{task.name}' completed.",
                status=AutomationStatus.COMPLETED,
                automation_id=automation_id,
                data={
                    "result": result,
                    "name": task.name,
                },
            )

        except Exception as exc:
            task.status = AutomationStatus.FAILED
            task.completed_at = time.time()
            task.error = str(exc)

            return self._failure(
                f"Automation '{task.name}' failed.",
                str(exc),
                automation_id=automation_id,
            )

    # ================================================================
    # ROUTINES
    # ================================================================

    def run_routine(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AutomationResult:
        """Execute a registered routine."""

        if not name:
            return self._failure(
                "No routine name was provided.",
                "missing_routine",
            )

        routine_name = name.strip().lower()

        with self._lock:
            routine = self._routines.get(routine_name)

        if routine is None:
            return self._failure(
                f"Routine '{name}' was not found.",
                "routine_not_found",
            )

        automation_id = self._generate_id()

        try:
            result = routine(**(parameters or {}))

            return AutomationResult(
                success=True,
                message=f"Routine '{name}' completed.",
                status=AutomationStatus.COMPLETED,
                automation_id=automation_id,
                data={
                    "routine": name,
                    "result": result,
                },
            )

        except Exception as exc:
            return AutomationResult(
                success=False,
                message=f"Routine '{name}' failed.",
                status=AutomationStatus.FAILED,
                automation_id=automation_id,
                error=str(exc),
            )

    # ================================================================
    # WORKFLOWS
    # ================================================================

    def run_workflow(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AutomationResult:
        """Execute a registered workflow."""

        if not name:
            return self._failure(
                "No workflow name was provided.",
                "missing_workflow",
            )

        workflow_name = name.strip().lower()

        with self._lock:
            workflow = self._workflows.get(workflow_name)

        if workflow is None:
            return self._failure(
                f"Workflow '{name}' was not found.",
                "workflow_not_found",
            )

        automation_id = self._generate_id()

        try:
            result = workflow(**(parameters or {}))

            return AutomationResult(
                success=True,
                message=f"Workflow '{name}' completed.",
                status=AutomationStatus.COMPLETED,
                automation_id=automation_id,
                data={
                    "workflow": name,
                    "result": result,
                },
            )

        except Exception as exc:
            return AutomationResult(
                success=False,
                message=f"Workflow '{name}' failed.",
                status=AutomationStatus.FAILED,
                automation_id=automation_id,
                error=str(exc),
            )

    # ================================================================
    # TRIGGERS
    # ================================================================

    def register_trigger(
        self,
        name: str,
        event: str,
        automation: str,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register an event-based automation trigger."""

        if not name or not event or not automation:
            return False

        trigger = {
            "name": name,
            "event": event,
            "automation": automation,
            "conditions": conditions or {},
            "enabled": True,
            "created_at": time.time(),
        }

        with self._lock:
            self._triggers[name.lower()] = trigger

        return True

    def trigger(
        self,
        event: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[AutomationResult]:
        """
        Fire all matching triggers for an event.
        """

        if not event:
            return []

        event = event.strip().lower()

        with self._lock:
            matching = [
                trigger
                for trigger in self._triggers.values()
                if trigger["enabled"]
                and trigger["event"].lower() == event
            ]

        results: List[AutomationResult] = []

        for trigger in matching:
            automation_name = trigger["automation"]

            if automation_name.lower() in self._routines:
                result = self.run_routine(
                    automation_name,
                    parameters,
                )
                results.append(result)

            elif automation_name.lower() in self._workflows:
                result = self.run_workflow(
                    automation_name,
                    parameters,
                )
                results.append(result)

        return results

    # ================================================================
    # CANCELLATION
    # ================================================================

    def cancel_task(
        self,
        automation_id: str,
    ) -> AutomationResult:
        """
        Cancel a pending/running task.

        Python cannot safely terminate arbitrary running functions,
        so running tasks are marked cancelled and cooperative
        cancellation can be implemented by the underlying handler.
        """

        with self._lock:
            task = self._tasks.get(automation_id)

        if task is None:
            return self._failure(
                "Automation was not found.",
                "automation_not_found",
                automation_id=automation_id,
            )

        if task.status == AutomationStatus.COMPLETED:
            return self._failure(
                "Automation has already completed.",
                "already_completed",
                automation_id=automation_id,
            )

        if task.status == AutomationStatus.FAILED:
            return self._failure(
                "Automation has already failed.",
                "already_failed",
                automation_id=automation_id,
            )

        task.status = AutomationStatus.CANCELLED
        task.completed_at = time.time()

        return AutomationResult(
            success=True,
            message=f"Automation '{task.name}' cancelled.",
            status=AutomationStatus.CANCELLED,
            automation_id=automation_id,
        )

    # ================================================================
    # STATUS
    # ================================================================

    def get_status(
        self,
        automation_id: str,
    ) -> AutomationResult:
        """Return the status of an automation."""

        with self._lock:
            task = self._tasks.get(automation_id)

        if task is None:
            return self._failure(
                "Automation was not found.",
                "automation_not_found",
                automation_id=automation_id,
            )

        return AutomationResult(
            success=True,
            message="Automation status retrieved.",
            status=task.status,
            automation_id=automation_id,
            data={
                "name": task.name,
                "description": task.description,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error": task.error,
            },
        )

    # ================================================================
    # LISTING
    # ================================================================

    def list_automations(self) -> AutomationResult:
        """Return all registered automations."""

        with self._lock:
            tasks = [
                {
                    "automation_id": task.automation_id,
                    "name": task.name,
                    "description": task.description,
                    "status": task.status.value,
                }
                for task in self._tasks.values()
            ]

            routines = list(self._routines.keys())
            workflows = list(self._workflows.keys())
            triggers = list(self._triggers.keys())

        return AutomationResult(
            success=True,
            message="Automation registry retrieved.",
            status=AutomationStatus.COMPLETED,
            automation_id=self._generate_id(),
            data={
                "tasks": tasks,
                "routines": routines,
                "workflows": workflows,
                "triggers": triggers,
            },
        )

    # ================================================================
    # INTERNAL EXECUTION HANDLERS
    # ================================================================

    def _execute_registered_task(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        automation_id = (
            parameters.get("automation_id")
            or parameters.get("id")
        )

        if not automation_id:
            return self._failure(
                "No automation ID was provided.",
                "missing_automation_id",
            )

        execution_parameters = parameters.get("parameters", {})

        return self.run_task(
            str(automation_id),
            execution_parameters,
        )

    def _run_routine(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        name = (
            parameters.get("name")
            or parameters.get("routine")
        )

        routine_parameters = parameters.get("parameters", {})

        return self.run_routine(
            str(name) if name else "",
            routine_parameters,
        )

    def _run_workflow(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        name = (
            parameters.get("name")
            or parameters.get("workflow")
        )

        workflow_parameters = parameters.get("parameters", {})

        return self.run_workflow(
            str(name) if name else "",
            workflow_parameters,
        )

    def _trigger(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        event = parameters.get("event")

        if not event:
            return self._failure(
                "No trigger event was provided.",
                "missing_event",
            )

        results = self.trigger(
            str(event),
            parameters.get("parameters", {}),
        )

        successful = all(result.success for result in results)

        return AutomationResult(
            success=successful,
            message=f"Trigger '{event}' processed.",
            status=(
                AutomationStatus.COMPLETED
                if successful
                else AutomationStatus.FAILED
            ),
            automation_id=self._generate_id(),
            data={
                "event": event,
                "results": [
                    result.to_dict()
                    for result in results
                ],
            },
        )

    def _register_routine_from_parameters(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        """
        Registering executable callables through the generic execute()
        API is intentionally unsupported.

        Call register_routine() directly from trusted application code.
        """

        return self._failure(
            "Routines must be registered through register_routine().",
            "direct_callable_registration_required",
        )

    def _register_workflow_from_parameters(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        """
        Registering executable callables through the generic execute()
        API is intentionally unsupported.
        """

        return self._failure(
            "Workflows must be registered through register_workflow().",
            "direct_callable_registration_required",
        )

    def _cancel_task(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        automation_id = (
            parameters.get("automation_id")
            or parameters.get("id")
        )

        if not automation_id:
            return self._failure(
                "No automation ID was provided.",
                "missing_automation_id",
            )

        return self.cancel_task(str(automation_id))

    def _get_status(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        automation_id = (
            parameters.get("automation_id")
            or parameters.get("id")
        )

        if not automation_id:
            return self._failure(
                "No automation ID was provided.",
                "missing_automation_id",
            )

        return self.get_status(str(automation_id))

    def _list_automations(
        self,
        parameters: Dict[str, Any],
    ) -> AutomationResult:
        return self.list_automations()

    # ================================================================
    # CAPABILITIES / HEALTH
    # ================================================================

    def get_capabilities(self) -> List[str]:
        """Return capabilities supported by this agent."""

        return [
            "run",
            "execute",
            "run_routine",
            "routine",
            "run_workflow",
            "workflow",
            "trigger",
            "register_routine",
            "register_workflow",
            "cancel",
            "status",
            "list",
        ]

    def health_check(self) -> Dict[str, Any]:
        """Return the current health of the agent."""

        with self._lock:
            return {
                "agent": self.name,
                "status": "healthy" if self.running else "stopped",
                "registered_tasks": len(self._tasks),
                "registered_routines": len(self._routines),
                "registered_workflows": len(self._workflows),
                "registered_triggers": len(self._triggers),
            }

    def start(self) -> None:
        """Start the automation agent."""

        self.running = True

    def stop(self) -> None:
        """Stop the automation agent."""

        self.running = False

    # ================================================================
    # RESULT HELPERS
    # ================================================================

    def _success(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AutomationResult:
        return AutomationResult(
            success=True,
            message=message,
            status=AutomationStatus.COMPLETED,
            automation_id=self._generate_id(),
            data=data or {},
        )

    def _failure(
        self,
        message: str,
        error: Optional[str] = None,
        automation_id: Optional[str] = None,
    ) -> AutomationResult:
        return AutomationResult(
            success=False,
            message=message,
            status=AutomationStatus.FAILED,
            automation_id=automation_id or self._generate_id(),
            error=error,
        )

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique automation ID."""

        return f"automation_{uuid.uuid4().hex}"


# ====================================================================
# DEFAULT GLOBAL INSTANCE
# ====================================================================

automation_agent = AutomationAgent()


__all__ = [
    "AutomationStatus",
    "AutomationResult",
    "AutomationTask",
    "AutomationAgent",
    "automation_agent",
]


