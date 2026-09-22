"""
RENIX Automation Engine
=======================

Central coordinator for RENIX automation.

Responsibilities:
    - Register and manage workflows
    - Execute workflows
    - Execute individual actions
    - Manage triggers
    - Manage routines
    - Manage scheduled tasks
    - Manage macros
    - Track running automation jobs
    - Provide automation status
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid

from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AutomationEngine:
    """
    Central automation coordinator.

    The engine is intentionally dependency-light. Other RENIX
    components can be injected into it rather than hard-coded.
    """

    def __init__(
        self,
        *,
        workflow_engine: Any = None,
        action_executor: Any = None,
        trigger_manager: Any = None,
        routine_manager: Any = None,
        scheduled_task_manager: Any = None,
        macro_manager: Any = None,
        event_bus: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:

        self.workflow_engine = workflow_engine
        self.action_executor = action_executor
        self.trigger_manager = trigger_manager
        self.routine_manager = routine_manager
        self.scheduled_task_manager = scheduled_task_manager
        self.macro_manager = macro_manager
        self.event_bus = event_bus

        self.context: dict[str, Any] = dict(
            context or {}
        )

        self._running = False

        self._jobs: dict[
            str,
            dict[str, Any],
        ] = {}

        self._workflows: dict[
            str,
            dict[str, Any],
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        self._lock = threading.RLock()

        self._loop: asyncio.AbstractEventLoop | None = None

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def start(self) -> None:
        """Start the automation engine."""

        with self._lock:

            if self._running:
                return

            self._running = True

            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        self._start_component(
            self.trigger_manager
        )

        self._start_component(
            self.routine_manager
        )

        self._start_component(
            self.scheduled_task_manager
        )

        logger.info(
            "RENIX Automation Engine started."
        )

        self._emit(
            "automation.engine.started",
            {
                "timestamp": self._timestamp(),
            },
        )

    def stop(self) -> None:
        """Stop the automation engine."""

        with self._lock:

            if not self._running:
                return

            self._running = False

        self._stop_component(
            self.scheduled_task_manager
        )

        self._stop_component(
            self.routine_manager
        )

        self._stop_component(
            self.trigger_manager
        )

        logger.info(
            "RENIX Automation Engine stopped."
        )

        self._emit(
            "automation.engine.stopped",
            {
                "timestamp": self._timestamp(),
            },
        )

    def is_running(self) -> bool:
        """Return whether the engine is running."""

        return self._running

    # ========================================================
    # WORKFLOWS
    # ========================================================

    def register_workflow(
        self,
        workflow: dict[str, Any],
    ) -> str:
        """
        Register a workflow.

        Expected structure:

        {
            "name": "morning_routine",
            "description": "...",
            "actions": [...]
        }
        """

        if not isinstance(
            workflow,
            dict,
        ):
            raise TypeError(
                "workflow must be a dictionary."
            )

        name = str(
            workflow.get(
                "name",
                ""
            )
        ).strip()

        if not name:
            raise ValueError(
                "Workflow name is required."
            )

        workflow_id = str(
            workflow.get(
                "id"
            )
            or uuid.uuid4()
        )

        stored = dict(
            workflow
        )

        stored["id"] = workflow_id
        stored["name"] = name
        stored.setdefault(
            "enabled",
            True,
        )

        with self._lock:
            self._workflows[
                workflow_id
            ] = stored

        if self.workflow_engine is not None:
            register = getattr(
                self.workflow_engine,
                "register",
                None,
            )

            if callable(register):
                register(
                    stored
                )

        self._emit(
            "automation.workflow.registered",
            stored,
        )

        return workflow_id

    def unregister_workflow(
        self,
        workflow_id: str,
    ) -> bool:
        """Remove a registered workflow."""

        with self._lock:

            if workflow_id not in self._workflows:
                return False

            workflow = self._workflows.pop(
                workflow_id
            )

        if self.workflow_engine is not None:
            unregister = getattr(
                self.workflow_engine,
                "unregister",
                None,
            )

            if callable(unregister):
                unregister(
                    workflow_id
                )

        self._emit(
            "automation.workflow.unregistered",
            workflow,
        )

        return True

    def get_workflow(
        self,
        workflow_id: str,
    ) -> dict[str, Any] | None:
        """Return a workflow by ID."""

        with self._lock:
            workflow = self._workflows.get(
                workflow_id
            )

            if workflow is None:
                return None

            return dict(
                workflow
            )

    def list_workflows(
        self,
        *,
        enabled_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return registered workflows."""

        with self._lock:

            workflows = list(
                self._workflows.values()
            )

        if enabled_only:
            workflows = [
                workflow
                for workflow in workflows
                if workflow.get(
                    "enabled",
                    True,
                )
            ]

        return [
            dict(workflow)
            for workflow in workflows
        ]

    def enable_workflow(
        self,
        workflow_id: str,
    ) -> bool:
        return self._set_workflow_enabled(
            workflow_id,
            True,
        )

    def disable_workflow(
        self,
        workflow_id: str,
    ) -> bool:
        return self._set_workflow_enabled(
            workflow_id,
            False,
        )

    def _set_workflow_enabled(
        self,
        workflow_id: str,
        enabled: bool,
    ) -> bool:

        with self._lock:

            workflow = self._workflows.get(
                workflow_id
            )

            if workflow is None:
                return False

            workflow["enabled"] = enabled

        self._emit(
            "automation.workflow.updated",
            {
                "id": workflow_id,
                "enabled": enabled,
            },
        )

        return True

    # ========================================================
    # EXECUTION
    # ========================================================

    def execute_workflow(
        self,
        workflow_id: str,
        *,
        variables: dict[str, Any] | None = None,
        background: bool = False,
    ) -> Any:
        """
        Execute a registered workflow.

        If background=True, returns a job ID.
        """

        workflow = self.get_workflow(
            workflow_id
        )

        if workflow is None:
            raise KeyError(
                f"Workflow not found: {workflow_id}"
            )

        if not workflow.get(
            "enabled",
            True,
        ):
            raise RuntimeError(
                f"Workflow is disabled: {workflow_id}"
            )

        job_id = self._create_job(
            workflow_id=workflow_id,
            variables=variables,
        )

        if background:

            thread = threading.Thread(
                target=self._execute_job_sync,
                args=(
                    job_id,
                    workflow,
                    variables or {},
                ),
                daemon=True,
            )

            thread.start()

            return job_id

        return self._execute_job_sync(
            job_id,
            workflow,
            variables or {},
        )

    def execute_actions(
        self,
        actions: list[dict[str, Any]],
        *,
        variables: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Execute a list of automation actions."""

        if not isinstance(
            actions,
            list,
        ):
            raise TypeError(
                "actions must be a list."
            )

        results: list[Any] = []

        for action in actions:

            result = self.execute_action(
                action,
                variables=variables,
            )

            results.append(
                result
            )

        return results

    def execute_action(
        self,
        action: dict[str, Any],
        *,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a single automation action."""

        if not isinstance(
            action,
            dict,
        ):
            raise TypeError(
                "action must be a dictionary."
            )

        if self.action_executor is None:
            raise RuntimeError(
                "Action executor is not configured."
            )

        execute = getattr(
            self.action_executor,
            "execute",
            None,
        )

        if not callable(execute):
            raise RuntimeError(
                "Action executor does not provide execute()."
            )

        merged_variables = dict(
            self.context
        )

        merged_variables.update(
            variables or {}
        )

        return execute(
            action,
            context=merged_variables,
        )

    # ========================================================
    # JOBS
    # ========================================================

    def _create_job(
        self,
        *,
        workflow_id: str,
        variables: dict[str, Any] | None,
    ) -> str:

        job_id = str(
            uuid.uuid4()
        )

        job = {
            "id": job_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "created_at": self._timestamp(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "variables": dict(
                variables or {}
            ),
        }

        with self._lock:
            self._jobs[
                job_id
            ] = job

        return job_id

    def _execute_job_sync(
        self,
        job_id: str,
        workflow: dict[str, Any],
        variables: dict[str, Any],
    ) -> Any:

        self._update_job(
            job_id,
            status="running",
            started_at=self._timestamp(),
        )

        self._emit(
            "automation.job.started",
            {
                "job_id": job_id,
                "workflow_id": workflow["id"],
            },
        )

        try:

            result = self._execute_workflow_internal(
                workflow,
                variables,
            )

            self._update_job(
                job_id,
                status="completed",
                finished_at=self._timestamp(),
                result=result,
            )

            self._emit(
                "automation.job.completed",
                {
                    "job_id": job_id,
                    "workflow_id": workflow["id"],
                    "result": result,
                },
            )

            return result

        except Exception as exc:

            logger.exception(
                "Automation job failed: %s",
                job_id,
            )

            self._update_job(
                job_id,
                status="failed",
                finished_at=self._timestamp(),
                error=str(exc),
            )

            self._emit(
                "automation.job.failed",
                {
                    "job_id": job_id,
                    "workflow_id": workflow["id"],
                    "error": str(exc),
                },
            )

            raise

    def _execute_workflow_internal(
        self,
        workflow: dict[str, Any],
        variables: dict[str, Any],
    ) -> Any:

        if self.workflow_engine is not None:

            execute = getattr(
                self.workflow_engine,
                "execute",
                None,
            )

            if callable(execute):
                return execute(
                    workflow,
                    context={
                        **self.context,
                        **variables,
                    },
                )

        actions = workflow.get(
            "actions",
            [],
        )

        return self.execute_actions(
            actions,
            variables=variables,
        )

    def _update_job(
        self,
        job_id: str,
        **updates: Any,
    ) -> None:

        with self._lock:

            job = self._jobs.get(
                job_id
            )

            if job is None:
                return

            job.update(
                updates
            )

    def get_job(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        """Return job information."""

        with self._lock:

            job = self._jobs.get(
                job_id
            )

            if job is None:
                return None

            return dict(
                job
            )

    def list_jobs(
        self,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List automation jobs."""

        with self._lock:

            jobs = list(
                self._jobs.values()
            )

        if status is not None:
            jobs = [
                job
                for job in jobs
                if job.get(
                    "status"
                ) == status
            ]

        return [
            dict(job)
            for job in jobs
        ]

    # ========================================================
    # CANCELLATION
    # ========================================================

    def cancel_job(
        self,
        job_id: str,
    ) -> bool:
        """
        Mark a queued/running job as cancelled.

        Actual action interruption is delegated to the workflow
        or action executor when supported.
        """

        with self._lock:

            job = self._jobs.get(
                job_id
            )

            if job is None:
                return False

            if job["status"] not in {
                "queued",
                "running",
            }:
                return False

            job["status"] = "cancelled"
            job["finished_at"] = self._timestamp()

        self._emit(
            "automation.job.cancelled",
            {
                "job_id": job_id,
            },
        )

        return True

    # ========================================================
    # CONTEXT
    # ========================================================

    def set_context(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set a global automation context value."""

        if not key.strip():
            raise ValueError(
                "context key cannot be empty."
            )

        self.context[key] = value

    def update_context(
        self,
        values: dict[str, Any],
    ) -> None:
        """Update multiple context values."""

        if not isinstance(
            values,
            dict,
        ):
            raise TypeError(
                "values must be a dictionary."
            )

        self.context.update(
            values
        )

    def get_context(
        self,
    ) -> dict[str, Any]:
        """Return a copy of automation context."""

        return dict(
            self.context
        )

    # ========================================================
    # EVENTS
    # ========================================================

    def subscribe(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> None:
        """Subscribe to an engine event."""

        if not event.strip():
            raise ValueError(
                "event cannot be empty."
            )

        if not callable(callback):
            raise TypeError(
                "callback must be callable."
            )

        self._callbacks.setdefault(
            event,
            [],
        ).append(
            callback
        )

    def unsubscribe(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> bool:
        """Remove an event callback."""

        callbacks = self._callbacks.get(
            event,
            [],
        )

        if callback not in callbacks:
            return False

        callbacks.remove(
            callback
        )

        return True

    def _emit(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> None:

        callbacks = list(
            self._callbacks.get(
                event,
                [],
            )
        )

        for callback in callbacks:

            try:
                callback(
                    payload
                )
            except Exception:
                logger.exception(
                    "Automation event callback failed."
                )

        if self.event_bus is not None:

            publish = getattr(
                self.event_bus,
                "publish",
                None,
            )

            if callable(publish):

                try:
                    publish(
                        event,
                        payload,
                    )
                except Exception:
                    logger.exception(
                        "Event bus publish failed."
                    )

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict[str, Any]:
        """Return complete automation engine status."""

        with self._lock:

            jobs = list(
                self._jobs.values()
            )

            workflows = list(
                self._workflows.values()
            )

        return {
            "running": self._running,
            "workflow_count": len(
                workflows
            ),
            "job_count": len(
                jobs
            ),
            "running_jobs": sum(
                1
                for job in jobs
                if job["status"] == "running"
            ),
            "completed_jobs": sum(
                1
                for job in jobs
                if job["status"] == "completed"
            ),
            "failed_jobs": sum(
                1
                for job in jobs
                if job["status"] == "failed"
            ),
        }

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(
            timespec="seconds"
        )

    @staticmethod
    def _start_component(
        component: Any,
    ) -> None:

        if component is None:
            return

        start = getattr(
            component,
            "start",
            None,
        )

        if callable(start):
            try:
                start()
            except Exception:
                logger.exception(
                    "Failed to start automation component."
                )

    @staticmethod
    def _stop_component(
        component: Any,
    ) -> None:

        if component is None:
            return

        stop = getattr(
            component,
            "stop",
            None,
        )

        if callable(stop):
            try:
                stop()
            except Exception:
                logger.exception(
                    "Failed to stop automation component."
                )


__all__ = [
    "AutomationEngine",
]


