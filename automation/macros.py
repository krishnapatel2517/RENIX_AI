"""
RENIX Macro Engine
==================

Provides reusable macro recording and playback for RENIX.

A macro is a named sequence of automation actions.

Features:
    - Create macros
    - Record actions
    - Add/remove actions
    - Play macros
    - Pause/resume playback
    - Stop playback
    - Enable/disable macros
    - Import/export-friendly dictionaries
    - Variable/context support
    - Event notifications
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid

from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MacroManager:
    """Manage RENIX automation macros."""

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

        self._macros: dict[
            str,
            dict[str, Any],
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        self._lock = threading.RLock()

        self._playback_thread: threading.Thread | None = None

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        self._playing_macro_id: str | None = None

    # ========================================================
    # MACRO CREATION
    # ========================================================

    def create(
        self,
        name: str,
        *,
        description: str = "",
        actions: list[dict[str, Any]] | None = None,
        variables: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> str:
        """Create a new macro."""

        name = name.strip()

        if not name:
            raise ValueError(
                "Macro name cannot be empty."
            )

        macro_id = str(
            uuid.uuid4()
        )

        macro = {
            "id": macro_id,
            "name": name,
            "description": description,
            "actions": copy.deepcopy(
                actions or []
            ),
            "variables": dict(
                variables or {}
            ),
            "enabled": bool(enabled),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "run_count": 0,
            "last_run": None,
        }

        self._validate_macro(
            macro
        )

        with self._lock:
            self._macros[
                macro_id
            ] = macro

        self._emit(
            "macro.created",
            self._serialize(
                macro
            ),
        )

        return macro_id

    # ========================================================
    # ACCESS
    # ========================================================

    def get(
        self,
        macro_id: str,
    ) -> dict[str, Any] | None:

        with self._lock:

            macro = self._macros.get(
                macro_id
            )

            if macro is None:
                return None

            return self._serialize(
                macro
            )

    def get_by_name(
        self,
        name: str,
    ) -> dict[str, Any] | None:

        name = name.strip().lower()

        with self._lock:

            for macro in self._macros.values():

                if macro["name"].lower() == name:

                    return self._serialize(
                        macro
                    )

        return None

    def list(
        self,
        *,
        enabled_only: bool = False,
    ) -> list[dict[str, Any]]:

        with self._lock:
            macros = list(
                self._macros.values()
            )

        if enabled_only:

            macros = [
                macro
                for macro in macros
                if macro["enabled"]
            ]

        return [
            self._serialize(
                macro
            )
            for macro in macros
        ]

    def delete(
        self,
        macro_id: str,
    ) -> bool:

        with self._lock:

            if macro_id not in self._macros:
                return False

            del self._macros[
                macro_id
            ]

        self._emit(
            "macro.deleted",
            {
                "id": macro_id,
            },
        )

        return True

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(
        self,
        macro_id: str,
    ) -> bool:

        return self._set_enabled(
            macro_id,
            True,
        )

    def disable(
        self,
        macro_id: str,
    ) -> bool:

        return self._set_enabled(
            macro_id,
            False,
        )

    def _set_enabled(
        self,
        macro_id: str,
        enabled: bool,
    ) -> bool:

        with self._lock:

            macro = self._macros.get(
                macro_id
            )

            if macro is None:
                return False

            macro["enabled"] = enabled

            macro["updated_at"] = datetime.now()

        self._emit(
            "macro.updated",
            self._serialize(
                macro
            ),
        )

        return True

    # ========================================================
    # ACTION MANAGEMENT
    # ========================================================

    def add_action(
        self,
        macro_id: str,
        action: dict[str, Any],
    ) -> bool:

        if not isinstance(
            action,
            dict,
        ):
            raise TypeError(
                "action must be a dictionary."
            )

        with self._lock:

            macro = self._macros.get(
                macro_id
            )

            if macro is None:
                return False

            macro["actions"].append(
                copy.deepcopy(
                    action
                )
            )

            macro["updated_at"] = datetime.now()

        self._emit(
            "macro.action_added",
            {
                "macro_id": macro_id,
                "action": action,
            },
        )

        return True

    def insert_action(
        self,
        macro_id: str,
        index: int,
        action: dict[str, Any],
    ) -> bool:

        if not isinstance(
            action,
            dict,
        ):
            raise TypeError(
                "action must be a dictionary."
            )

        with self._lock:

            macro = self._macros.get(
                macro_id
            )

            if macro is None:
                return False

            if not (
                0 <= index <= len(
                    macro["actions"]
                )
            ):
                raise IndexError(
                    "Action index out of range."
                )

            macro["actions"].insert(
                index,
                copy.deepcopy(
                    action
                ),
            )

            macro["updated_at"] = datetime.now()

        return True

    def remove_action(
        self,
        macro_id: str,
        index: int,
    ) -> bool:

        with self._lock:

            macro = self._macros.get(
                macro_id
            )

            if macro is None:
                return False

            actions = macro["actions"]

            if not (
                0 <= index < len(actions)
            ):
                return False

            actions.pop(
                index
            )

            macro["updated_at"] = datetime.now()

        self._emit(
            "macro.action_removed",
            {
                "macro_id": macro_id,
                "index": index,
            },
        )

        return True

    def clear_actions(
        self,
        macro_id: str,
    ) -> bool:

        with self._lock:

            macro = self._macros.get(
                macro_id
            )

            if macro is None:
                return False

            macro["actions"] = []

            macro["updated_at"] = datetime.now()

        self._emit(
            "macro.actions_cleared",
            {
                "macro_id": macro_id,
            },
        )

        return True

    # ========================================================
    # RECORDING
    # ========================================================

    def record_action(
        self,
        macro_id: str,
        action: dict[str, Any],
    ) -> bool:
        """
        Add an action while recording.

        This method can be called by computer/voice/gesture
        modules whenever an action should become part of a macro.
        """

        return self.add_action(
            macro_id,
            action,
        )

    # ========================================================
    # PLAYBACK
    # ========================================================

    def play(
        self,
        macro: str | dict[str, Any],
        *,
        variables: dict[str, Any] | None = None,
        blocking: bool = True,
    ) -> dict[str, Any]:
        """
        Play a macro.

        `macro` may be:
            - macro ID
            - macro name
            - macro dictionary
        """

        macro_data = self._resolve_macro(
            macro
        )

        if not macro_data["enabled"]:
            raise RuntimeError(
                f"Macro is disabled: {macro_data['name']}"
            )

        if self.is_playing():

            raise RuntimeError(
                "Another macro is already playing."
            )

        self._stop_event.clear()
        self._pause_event.clear()

        execution_id = str(
            uuid.uuid4()
        )

        self._playing_macro_id = macro_data[
            "id"
        ]

        context = dict(
            self.context
        )

        context.update(
            macro_data.get(
                "variables",
                {},
            )
        )

        context.update(
            variables or {}
        )

        if blocking:

            return self._play(
                macro_data,
                execution_id,
                context,
            )

        self._playback_thread = threading.Thread(
            target=self._play_thread,
            args=(
                macro_data,
                execution_id,
                context,
            ),
            name="RENIX-Macro-Playback",
            daemon=True,
        )

        self._playback_thread.start()

        return {
            "status": "started",
            "execution_id": execution_id,
            "macro_id": macro_data["id"],
            "macro_name": macro_data["name"],
        }

    def _play_thread(
        self,
        macro: dict[str, Any],
        execution_id: str,
        context: dict[str, Any],
    ) -> None:

        try:

            self._play(
                macro,
                execution_id,
                context,
            )

        except Exception:
            logger.exception(
                "Macro playback failed."
            )

    def _play(
        self,
        macro: dict[str, Any],
        execution_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:

        started_at = datetime.now()

        results: list[Any] = []

        self._emit(
            "macro.started",
            {
                "execution_id": execution_id,
                "macro_id": macro["id"],
                "macro_name": macro["name"],
            },
        )

        try:

            for index, action in enumerate(
                macro["actions"]
            ):

                if self._stop_event.is_set():

                    return self._finish_playback(
                        macro,
                        execution_id,
                        started_at,
                        results,
                        "stopped",
                    )

                self._wait_if_paused()

                self._emit(
                    "macro.action.started",
                    {
                        "execution_id": execution_id,
                        "macro_id": macro["id"],
                        "index": index,
                        "action": action,
                    },
                )

                result = self._execute_action(
                    action,
                    context,
                )

                results.append(
                    result
                )

                self._emit(
                    "macro.action.completed",
                    {
                        "execution_id": execution_id,
                        "macro_id": macro["id"],
                        "index": index,
                        "result": result,
                    },
                )

            return self._finish_playback(
                macro,
                execution_id,
                started_at,
                results,
                "completed",
            )

        except Exception as exc:

            self._emit(
                "macro.failed",
                {
                    "execution_id": execution_id,
                    "macro_id": macro["id"],
                    "error": str(exc),
                },
            )

            raise

        finally:

            self._playing_macro_id = None

    # ========================================================
    # ACTION EXECUTION
    # ========================================================

    def _execute_action(
        self,
        action: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:

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
            context=context,
        )

    # ========================================================
    # PLAYBACK CONTROL
    # ========================================================

    def pause(self) -> bool:
        """Pause the current macro."""

        if not self.is_playing():
            return False

        self._pause_event.set()

        self._emit(
            "macro.paused",
            {
                "macro_id": self._playing_macro_id,
            },
        )

        return True

    def resume(self) -> bool:
        """Resume the current macro."""

        if not self.is_playing():
            return False

        self._pause_event.clear()

        self._emit(
            "macro.resumed",
            {
                "macro_id": self._playing_macro_id,
            },
        )

        return True

    def stop(self) -> bool:
        """Stop the current macro."""

        if not self.is_playing():
            return False

        self._stop_event.set()
        self._pause_event.clear()

        self._emit(
            "macro.stop_requested",
            {
                "macro_id": self._playing_macro_id,
            },
        )

        return True

    def is_playing(self) -> bool:
        thread = self._playback_thread

        return (
            self._playing_macro_id is not None
            and (
                thread is None
                or thread.is_alive()
            )
        )

    def _wait_if_paused(self) -> None:

        while self._pause_event.is_set():

            if self._stop_event.is_set():
                return

            time.sleep(
                0.05
            )

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

    # ========================================================
    # IMPORT / EXPORT
    # ========================================================

    def export_macro(
        self,
        macro_id: str,
    ) -> dict[str, Any]:
        """Return a JSON-friendly macro definition."""

        macro = self._macros.get(
            macro_id
        )

        if macro is None:
            raise KeyError(
                f"Macro not found: {macro_id}"
            )

        return self._serialize(
            macro
        )

    def import_macro(
        self,
        data: dict[str, Any],
        *,
        replace_id: bool = False,
    ) -> str:
        """Import a macro definition."""

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "Macro data must be a dictionary."
            )

        imported = copy.deepcopy(
            data
        )

        if (
            replace_id
            or not imported.get("id")
        ):
            imported["id"] = str(
                uuid.uuid4()
            )

        imported.setdefault(
            "created_at",
            datetime.now(),
        )

        imported.setdefault(
            "updated_at",
            datetime.now(),
        )

        imported.setdefault(
            "run_count",
            0,
        )

        imported.setdefault(
            "last_run",
            None,
        )

        self._validate_macro(
            imported
        )

        with self._lock:

            self._macros[
                imported["id"]
            ] = imported

        return imported["id"]

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_macro(
        macro: dict[str, Any],
    ) -> None:

        if not macro.get("id"):
            raise ValueError(
                "Macro ID is required."
            )

        if not macro.get("name"):
            raise ValueError(
                "Macro name is required."
            )

        actions = macro.get(
            "actions",
            [],
        )

        if not isinstance(
            actions,
            list,
        ):
            raise TypeError(
                "Macro actions must be a list."
            )

        for action in actions:

            if not isinstance(
                action,
                dict,
            ):
                raise TypeError(
                    "Every macro action must be a dictionary."
                )

    # ========================================================
    # FINISH
    # ========================================================

    def _finish_playback(
        self,
        macro: dict[str, Any],
        execution_id: str,
        started_at: datetime,
        results: list[Any],
        status: str,
    ) -> dict[str, Any]:

        finished_at = datetime.now()

        with self._lock:

            stored = self._macros.get(
                macro["id"]
            )

            if stored is not None:

                stored["run_count"] += 1
                stored["last_run"] = finished_at
                stored["updated_at"] = finished_at

        result = {
            "status": status,
            "execution_id": execution_id,
            "macro_id": macro["id"],
            "macro_name": macro["name"],
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "results": results,
        }

        self._emit(
            f"macro.{status}",
            result,
        )

        return result

    # ========================================================
    # EVENTS
    # ========================================================

    def subscribe(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> None:

        if not event.strip():
            raise ValueError(
                "Event cannot be empty."
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
                    "Macro callback failed."
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
                        "Failed to publish macro event."
                    )

    # ========================================================
    # HELPERS
    # ========================================================

    def _resolve_macro(
        self,
        macro: str | dict[str, Any],
    ) -> dict[str, Any]:

        if isinstance(
            macro,
            dict,
        ):
            data = copy.deepcopy(
                macro
            )

        elif isinstance(
            macro,
            str,
        ):

            data = self.get(
                macro
            )

            if data is None:
                data = self.get_by_name(
                    macro
                )

            if data is None:
                raise KeyError(
                    f"Macro not found: {macro}"
                )

        else:
            raise TypeError(
                "macro must be a string or dictionary."
            )

        return data

    @staticmethod
    def _serialize(
        macro: dict[str, Any],
    ) -> dict[str, Any]:

        data = copy.deepcopy(
            macro
        )

        for key in (
            "created_at",
            "updated_at",
            "last_run",
        ):

            value = data.get(
                key
            )

            if isinstance(
                value,
                datetime,
            ):
                data[key] = value.isoformat()

        return data


__all__ = [
    "MacroManager",
]


