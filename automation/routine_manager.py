"""
RENIX Routine Manager
=====================

Manages reusable RENIX routines.

A routine is a named collection of automation actions that can
be executed manually, from a workflow, or through a trigger.

Example routine:

{
    "name": "study_mode",
    "actions": [
        {
            "type": "smart_home",
            "method": "set_light",
            "data": {
                "brightness": 40
            }
        },
        {
            "type": "media",
            "method": "play",
            "data": {
                "playlist": "study"
            }
        }
    ]
}
"""

from __future__ import annotations

import logging
import threading
import uuid

from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RoutineManager:
    """Manages reusable RENIX automation routines."""

    def __init__(
        self,
        *,
        workflow_engine: Any = None,
        action_executor: Any = None,
        event_bus: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:

        self.workflow_engine = workflow_engine
        self.action_executor = action_executor
        self.event_bus = event_bus

        self.context: dict[str, Any] = dict(
            context or {}
        )

        self._routines: dict[
            str,
            dict[str, Any],
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def start(self) -> None:
        """Start the routine manager."""

        with self._lock:

            if self._running:
                return

            self._running = True

        self._emit(
            "routine_manager.started",
            {},
        )

        logger.info(
            "RENIX Routine Manager started."
        )

    def stop(self) -> None:
        """Stop the routine manager."""

        with self._lock:

            if not self._running:
                return

            self._running = False

        self._emit(
            "routine_manager.stopped",
            {},
        )

        logger.info(
            "RENIX Routine Manager stopped."
        )

    def is_running(self) -> bool:
        return self._running

    # ========================================================
    # REGISTRATION
    # ========================================================

    def register(
        self,
        routine: dict[str, Any],
    ) -> str:
        """Register a routine."""

        normalized = self.validate(
            routine
        )

        routine_id = normalized["id"]

        with self._lock:
            self._routines[
                routine_id
            ] = normalized

        self._emit(
            "routine.registered",
            normalized,
        )

        return routine_id

    def unregister(
        self,
        routine_id: str,
    ) -> bool:
        """Remove a routine."""

        with self._lock:

            routine = self._routines.pop(
                routine_id,
                None,
            )

        if routine is None:
            return False

        self._emit(
            "routine.unregistered",
            routine,
        )

        return True

    def get(
        self,
        routine_id: str,
    ) -> dict[str, Any] | None:
        """Return a routine by ID."""

        with self._lock:

            routine = self._routines.get(
                routine_id
            )

            if routine is None:
                return None

            return dict(
                routine
            )

    def get_by_name(
        self,
        name: str,
    ) -> dict[str, Any] | None:
        """Find a routine by name."""

        name = name.strip().lower()

        with self._lock:

            for routine in self._routines.values():

                if routine.get(
                    "name",
                    "",
                ).lower() == name:
                    return dict(
                        routine
                    )

        return None

    def list(
        self,
        *,
        enabled_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List routines."""

        with self._lock:
            routines = list(
                self._routines.values()
            )

        if enabled_only:
            routines = [
                routine
                for routine in routines
                if routine.get(
                    "enabled",
                    True,
                )
            ]

        return [
            dict(routine)
            for routine in routines
        ]

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        routine: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and normalize a routine."""

        if not isinstance(
            routine,
            dict,
        ):
            raise TypeError(
                "routine must be a dictionary."
            )

        normalized = dict(
            routine
        )

        routine_id = str(
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
                "Routine name is required."
            )

        actions = normalized.get(
            "actions",
            [],
        )

        if not isinstance(
            actions,
            list,
        ):
            raise TypeError(
                "routine.actions must be a list."
            )

        normalized["id"] = routine_id
        normalized["name"] = name
        normalized["actions"] = [
            dict(action)
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
                "routine.variables must be a dictionary."
            )

        return normalized

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(
        self,
        routine_id: str,
    ) -> bool:
        return self._set_enabled(
            routine_id,
            True,
        )

    def disable(
        self,
        routine_id: str,
    ) -> bool:
        return self._set_enabled(
            routine_id,
            False,
        )

    def _set_enabled(
        self,
        routine_id: str,
        enabled: bool,
    ) -> bool:

        with self._lock:

            routine = self._routines.get(
                routine_id
            )

            if routine is None:
                return False

            routine["enabled"] = enabled

        self._emit(
            "routine.updated",
            {
                "id": routine_id,
                "enabled": enabled,
            },
        )

        return True

    # ========================================================
    # EXECUTION
    # ========================================================

    def execute(
        self,
        routine: str | dict[str, Any],
        *,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a routine.

        `routine` can be:
            - routine ID
            - routine name
            - routine dictionary
        """

        if isinstance(
            routine,
            str,
        ):

            stored = self.get(
                routine
            )

            if stored is None:
                stored = self.get_by_name(
                    routine
                )

            if stored is None:
                raise KeyError(
                    f"Routine not found: {routine}"
                )

            routine = stored

        routine = self.validate(
            routine
        )

        if not routine.get(
            "enabled",
            True,
        ):
            raise RuntimeError(
                f"Routine is disabled: {routine['name']}"
            )

        execution_id = str(
            uuid.uuid4()
        )

        started_at = self._timestamp()

        context = dict(
            self.context
        )

        context.update(
            routine.get(
                "variables",
                {},
            )
        )

        context.update(
            variables or {}
        )

        results: list[Any] = []

        self._emit(
            "routine.started",
            {
                "execution_id": execution_id,
                "routine_id": routine["id"],
                "routine_name": routine["name"],
            },
        )

        try:

            if self.workflow_engine is not None:

                execute_actions = getattr(
                    self.workflow_engine,
                    "execute_actions",
                    None,
                )

                if callable(
                    execute_actions
                ):

                    results = execute_actions(
                        routine["actions"],
                        context=context,
                    )

                else:
                    results = self._execute_actions(
                        routine["actions"],
                        context,
                    )

            else:

                results = self._execute_actions(
                    routine["actions"],
                    context,
                )

            result = {
                "execution_id": execution_id,
                "routine_id": routine["id"],
                "routine_name": routine["name"],
                "status": "completed",
                "started_at": started_at,
                "finished_at": self._timestamp(),
                "results": results,
            }

            self._emit(
                "routine.completed",
                result,
            )

            return result

        except Exception as exc:

            logger.exception(
                "Routine execution failed: %s",
                routine["name"],
            )

            result = {
                "execution_id": execution_id,
                "routine_id": routine["id"],
                "routine_name": routine["name"],
                "status": "failed",
                "started_at": started_at,
                "finished_at": self._timestamp(),
                "results": results,
                "error": str(exc),
            }

            self._emit(
                "routine.failed",
                result,
            )

            raise

    def _execute_actions(
        self,
        actions: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[Any]:
        """Execute actions directly."""

        if self.action_executor is None:
            raise RuntimeError(
                "Action executor is not configured."
            )

        results: list[Any] = []

        for action in actions:

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

            results.append(
                execute(
                    action,
                    context=context,
                )
            )

        return results

    # ========================================================
    # ROUTINE COMPOSITION
    # ========================================================

    def add_action(
        self,
        routine_id: str,
        action: dict[str, Any],
    ) -> bool:
        """Add an action to a routine."""

        if not isinstance(
            action,
            dict,
        ):
            raise TypeError(
                "action must be a dictionary."
            )

        with self._lock:

            routine = self._routines.get(
                routine_id
            )

            if routine is None:
                return False

            routine.setdefault(
                "actions",
                [],
            ).append(
                dict(action)
            )

        self._emit(
            "routine.action_added",
            {
                "routine_id": routine_id,
                "action": action,
            },
        )

        return True

    def remove_action(
        self,
        routine_id: str,
        index: int,
    ) -> bool:
        """Remove an action by index."""

        with self._lock:

            routine = self._routines.get(
                routine_id
            )

            if routine is None:
                return False

            actions = routine.get(
                "actions",
                [],
            )

            if not (
                0 <= index < len(actions)
            ):
                return False

            removed = actions.pop(
                index
            )

        self._emit(
            "routine.action_removed",
            {
                "routine_id": routine_id,
                "index": index,
                "action": removed,
            },
        )

        return True

    def clear_actions(
        self,
        routine_id: str,
    ) -> bool:
        """Remove every action from a routine."""

        with self._lock:

            routine = self._routines.get(
                routine_id
            )

            if routine is None:
                return False

            routine["actions"] = []

        self._emit(
            "routine.actions_cleared",
            {
                "routine_id": routine_id,
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

        if not key.strip():
            raise ValueError(
                "Context key cannot be empty."
            )

        self.context[key] = value

    def update_context(
        self,
        values: dict[str, Any],
    ) -> None:

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
        return dict(
            self.context
        )

    # ========================================================
    # CALLBACKS
    # ========================================================

    def subscribe(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> None:
        """Subscribe to routine events."""

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

    # ========================================================
    # EVENTS
    # ========================================================

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
                    "Routine callback failed."
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
                        "Failed to publish routine event."
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
    "RoutineManager",
]


