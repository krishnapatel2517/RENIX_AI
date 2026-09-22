"""
RENIX Workflow Engine
=====================

Executes RENIX automation workflows.

A workflow is a dictionary such as:

{
    "id": "morning",
    "name": "Morning Routine",
    "enabled": True,
    "actions": [
        {
            "type": "notification",
            "message": "Good morning!"
        },
        {
            "type": "delay",
            "seconds": 2
        }
    ]
}

The actual execution of individual actions is delegated to
ActionExecutor.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Workflow registration, validation and execution engine."""

    def __init__(
        self,
        *,
        action_executor: Any = None,
        event_bus: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:

        self.action_executor = action_executor
        self.event_bus = event_bus

        self.context: dict[str, Any] = dict(
            context or {}
        )

        self._workflows: dict[
            str,
            dict[str, Any],
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        self._lock = threading.RLock()

    # ========================================================
    # WORKFLOW REGISTRATION
    # ========================================================

    def register(
        self,
        workflow: dict[str, Any],
    ) -> str:
        """Register a workflow."""

        normalized = self.validate_workflow(
            workflow
        )

        workflow_id = normalized["id"]

        with self._lock:
            self._workflows[
                workflow_id
            ] = normalized

        self._emit(
            "workflow.registered",
            normalized,
        )

        return workflow_id

    def unregister(
        self,
        workflow_id: str,
    ) -> bool:
        """Unregister a workflow."""

        with self._lock:

            if workflow_id not in self._workflows:
                return False

            workflow = self._workflows.pop(
                workflow_id
            )

        self._emit(
            "workflow.unregistered",
            workflow,
        )

        return True

    def get(
        self,
        workflow_id: str,
    ) -> dict[str, Any] | None:
        """Return a workflow."""

        with self._lock:

            workflow = self._workflows.get(
                workflow_id
            )

            if workflow is None:
                return None

            return dict(
                workflow
            )

    def list(
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

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_workflow(
        self,
        workflow: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate and normalize a workflow.

        Returns a safe copy.
        """

        if not isinstance(
            workflow,
            dict,
        ):
            raise TypeError(
                "workflow must be a dictionary."
            )

        normalized = dict(
            workflow
        )

        workflow_id = str(
            normalized.get(
                "id"
            ) or uuid.uuid4()
        ).strip()

        name = str(
            normalized.get(
                "name",
                "",
            )
        ).strip()

        if not name:
            raise ValueError(
                "Workflow name is required."
            )

        actions = normalized.get(
            "actions",
            [],
        )

        if actions is None:
            actions = []

        if not isinstance(
            actions,
            list,
        ):
            raise TypeError(
                "workflow.actions must be a list."
            )

        normalized["id"] = workflow_id
        normalized["name"] = name
        normalized["actions"] = [
            self.validate_action(
                action
            )
            for action in actions
        ]

        normalized.setdefault(
            "enabled",
            True,
        )

        normalized.setdefault(
            "description",
            "",
        )

        normalized.setdefault(
            "variables",
            {},
        )

        if not isinstance(
            normalized["variables"],
            dict,
        ):
            raise TypeError(
                "workflow.variables must be a dictionary."
            )

        return normalized

    def validate_action(
        self,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and normalize an action."""

        if not isinstance(
            action,
            dict,
        ):
            raise TypeError(
                "Each workflow action must be a dictionary."
            )

        normalized = dict(
            action
        )

        action_type = str(
            normalized.get(
                "type",
                "",
            )
        ).strip().lower()

        if not action_type:
            raise ValueError(
                "Workflow action requires a type."
            )

        normalized["type"] = action_type

        normalized.setdefault(
            "enabled",
            True,
        )

        return normalized

    # ========================================================
    # EXECUTION
    # ========================================================

    def execute(
        self,
        workflow: dict[str, Any] | str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a workflow synchronously.

        `workflow` may be either a workflow dictionary or
        a registered workflow ID.
        """

        if isinstance(
            workflow,
            str,
        ):

            stored = self.get(
                workflow
            )

            if stored is None:
                raise KeyError(
                    f"Workflow not found: {workflow}"
                )

            workflow = stored

        workflow = self.validate_workflow(
            workflow
        )

        if not workflow.get(
            "enabled",
            True,
        ):
            raise RuntimeError(
                f"Workflow is disabled: {workflow['id']}"
            )

        variables = dict(
            self.context
        )

        variables.update(
            workflow.get(
                "variables",
                {},
            )
        )

        variables.update(
            context or {}
        )

        execution_id = str(
            uuid.uuid4()
        )

        started_at = self._timestamp()

        self._emit(
            "workflow.started",
            {
                "execution_id": execution_id,
                "workflow_id": workflow["id"],
                "workflow_name": workflow["name"],
            },
        )

        results: list[Any] = []

        try:

            for index, action in enumerate(
                workflow["actions"]
            ):

                if not action.get(
                    "enabled",
                    True,
                ):
                    results.append(
                        {
                            "skipped": True,
                            "reason": "disabled",
                        }
                    )

                    continue

                self._emit(
                    "workflow.action.started",
                    {
                        "execution_id": execution_id,
                        "workflow_id": workflow["id"],
                        "action_index": index,
                        "action": action,
                    },
                )

                result = self.execute_action(
                    action,
                    context=variables,
                )

                results.append(
                    result
                )

                self._emit(
                    "workflow.action.completed",
                    {
                        "execution_id": execution_id,
                        "workflow_id": workflow["id"],
                        "action_index": index,
                        "result": result,
                    },
                )

            finished_at = self._timestamp()

            result = {
                "execution_id": execution_id,
                "workflow_id": workflow["id"],
                "workflow_name": workflow["name"],
                "status": "completed",
                "started_at": started_at,
                "finished_at": finished_at,
                "results": results,
            }

            self._emit(
                "workflow.completed",
                result,
            )

            return result

        except Exception as exc:

            logger.exception(
                "Workflow execution failed: %s",
                workflow["id"],
            )

            result = {
                "execution_id": execution_id,
                "workflow_id": workflow["id"],
                "workflow_name": workflow["name"],
                "status": "failed",
                "started_at": started_at,
                "finished_at": self._timestamp(),
                "results": results,
                "error": str(exc),
            }

            self._emit(
                "workflow.failed",
                result,
            )

            raise

    def execute_action(
        self,
        action: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute one workflow action."""

        action = self.validate_action(
            action
        )

        action_type = action["type"]

        # ----------------------------------------------------
        # Built-in delay
        # ----------------------------------------------------

        if action_type in {
            "delay",
            "wait",
            "sleep",
        }:

            seconds = float(
                action.get(
                    "seconds",
                    action.get(
                        "duration",
                        0,
                    ),
                )
            )

            if seconds < 0:
                raise ValueError(
                    "Delay cannot be negative."
                )

            time.sleep(
                seconds
            )

            return {
                "type": action_type,
                "seconds": seconds,
                "status": "completed",
            }

        # ----------------------------------------------------
        # Conditional action
        # ----------------------------------------------------

        if action_type in {
            "condition",
            "if",
        }:

            condition = action.get(
                "condition"
            )

            if not callable(
                condition
            ):
                raise ValueError(
                    "Condition action requires a callable "
                    "'condition'."
                )

            passed = bool(
                condition(
                    context or {}
                )
            )

            if passed:

                nested = action.get(
                    "then",
                    [],
                )

                return self.execute_actions(
                    nested,
                    context=context,
                )

            nested = action.get(
                "else",
                [],
            )

            return self.execute_actions(
                nested,
                context=context,
            )

        # ----------------------------------------------------
        # Set variable
        # ----------------------------------------------------

        if action_type in {
            "set_variable",
            "set",
        }:

            key = str(
                action.get(
                    "key",
                    "",
                )
            ).strip()

            if not key:
                raise ValueError(
                    "set_variable requires a key."
                )

            value = action.get(
                "value"
            )

            if context is not None:
                context[key] = value

            return {
                "key": key,
                "value": value,
            }

        # ----------------------------------------------------
        # Delegate all other actions
        # ----------------------------------------------------

        if self.action_executor is None:
            raise RuntimeError(
                "Action executor is not configured."
            )

        execute = getattr(
            self.action_executor,
            "execute",
            None,
        )

        if not callable(
            execute
        ):
            raise RuntimeError(
                "Action executor does not provide execute()."
            )

        return execute(
            action,
            context=context or {},
        )

    def execute_actions(
        self,
        actions: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Execute multiple actions sequentially."""

        if not isinstance(
            actions,
            list,
        ):
            raise TypeError(
                "actions must be a list."
            )

        results: list[Any] = []

        for action in actions:

            results.append(
                self.execute_action(
                    action,
                    context=context,
                )
            )

        return results

    # ========================================================
    # CONTEXT
    # ========================================================

    def set_context(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set a workflow context value."""

        if not key.strip():
            raise ValueError(
                "Context key cannot be empty."
            )

        self.context[key] = value

    def update_context(
        self,
        values: dict[str, Any],
    ) -> None:
        """Update workflow context."""

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
        """Return workflow context."""

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
        """Subscribe to workflow events."""

        if not event.strip():
            raise ValueError(
                "event cannot be empty."
            )

        if not callable(
            callback
        ):
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
        """Unsubscribe from a workflow event."""

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

        for callback in list(
            self._callbacks.get(
                event,
                [],
            )
        ):

            try:
                callback(
                    payload
                )
            except Exception:
                logger.exception(
                    "Workflow callback failed."
                )

        if self.event_bus is not None:

            publish = getattr(
                self.event_bus,
                "publish",
                None,
            )

            if callable(
                publish
            ):

                try:
                    publish(
                        event,
                        payload,
                    )
                except Exception:
                    logger.exception(
                        "Workflow event publish failed."
                    )

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def start(self) -> None:
        """Start workflow engine."""

        logger.info(
            "RENIX Workflow Engine started."
        )

    def stop(self) -> None:
        """Stop workflow engine."""

        logger.info(
            "RENIX Workflow Engine stopped."
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(
            timespec="seconds"
        )


__all__ = [
    "WorkflowEngine",
]


