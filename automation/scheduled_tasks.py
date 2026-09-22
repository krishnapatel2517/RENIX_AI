"""
RENIX Scheduled Tasks
=====================

Handles scheduled automation tasks.

Supports:

    - One-time execution
    - Daily schedules
    - Weekly schedules
    - Interval schedules
    - Specific datetime execution
    - Enable / disable
    - Repeating tasks
    - Callback notifications

This module is intentionally independent from the actual
workflow/action implementation. It delegates execution to
the configured callback or workflow engine.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from datetime import datetime, timedelta
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ScheduledTaskManager:
    """Manager for RENIX scheduled automation tasks."""

    def __init__(
        self,
        *,
        executor: Callable[..., Any] | None = None,
        event_bus: Any = None,
        check_interval: float = 1.0,
    ) -> None:

        self.executor = executor
        self.event_bus = event_bus

        self.check_interval = max(
            0.1,
            float(check_interval),
        )

        self._tasks: dict[
            str,
            dict[str, Any],
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False
        self._thread: threading.Thread | None = None

        self._stop_event = threading.Event()

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def start(self) -> None:
        """Start the scheduler loop."""

        with self._lock:

            if self._running:
                return

            self._running = True

            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._scheduler_loop,
                name="RENIX-Scheduler",
                daemon=True,
            )

            self._thread.start()

        self._emit(
            "scheduled_tasks.started",
            {},
        )

        logger.info(
            "RENIX Scheduled Task Manager started."
        )

    def stop(
        self,
        timeout: float = 5.0,
    ) -> None:
        """Stop the scheduler."""

        with self._lock:

            if not self._running:
                return

            self._running = False

            self._stop_event.set()

            thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(
                timeout=max(
                    0.0,
                    timeout,
                )
            )

        self._emit(
            "scheduled_tasks.stopped",
            {},
        )

    def is_running(self) -> bool:
        return self._running

    # ========================================================
    # TASK REGISTRATION
    # ========================================================

    def schedule(
        self,
        *,
        name: str,
        run_at: datetime | str | None = None,
        action: Any = None,
        callback: Callable[..., Any] | None = None,
        repeat: str | None = None,
        interval_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> str:
        """
        Schedule a task.

        Examples:

            schedule(
                name="Morning routine",
                run_at="07:00",
                action="morning_routine",
                repeat="daily",
            )

        Or:

            schedule(
                name="Backup",
                run_at=datetime.now() + timedelta(hours=1),
                action="backup",
            )
        """

        if not name.strip():
            raise ValueError(
                "Task name cannot be empty."
            )

        task_id = str(
            uuid.uuid4()
        )

        next_run = self._parse_run_at(
            run_at
        )

        if interval_seconds is not None:

            interval_seconds = float(
                interval_seconds
            )

            if interval_seconds <= 0:
                raise ValueError(
                    "interval_seconds must be greater than zero."
                )

        normalized_repeat = (
            repeat.strip().lower()
            if isinstance(
                repeat,
                str,
            )
            else None
        )

        allowed_repeat = {
            None,
            "once",
            "daily",
            "hourly",
            "weekly",
            "interval",
        }

        if normalized_repeat not in allowed_repeat:
            raise ValueError(
                f"Unsupported repeat mode: {repeat}"
            )

        if (
            normalized_repeat == "interval"
            and interval_seconds is None
        ):
            raise ValueError(
                "Interval schedules require interval_seconds."
            )

        task = {
            "id": task_id,
            "name": name.strip(),
            "run_at": next_run,
            "action": action,
            "callback": callback,
            "repeat": normalized_repeat or "once",
            "interval_seconds": interval_seconds,
            "metadata": dict(
                metadata or {}
            ),
            "enabled": bool(enabled),
            "created_at": datetime.now(),
            "last_run": None,
            "run_count": 0,
            "last_error": None,
        }

        with self._lock:
            self._tasks[
                task_id
            ] = task

        self._emit(
            "scheduled_task.created",
            self._serialize_task(task),
        )

        return task_id

    def unschedule(
        self,
        task_id: str,
    ) -> bool:
        """Remove a scheduled task."""

        with self._lock:

            task = self._tasks.pop(
                task_id,
                None,
            )

        if task is None:
            return False

        self._emit(
            "scheduled_task.removed",
            self._serialize_task(task),
        )

        return True

    # ========================================================
    # TASK ACCESS
    # ========================================================

    def get(
        self,
        task_id: str,
    ) -> dict[str, Any] | None:

        with self._lock:

            task = self._tasks.get(
                task_id
            )

            if task is None:
                return None

            return self._serialize_task(
                task
            )

    def list(
        self,
        *,
        enabled_only: bool = False,
    ) -> list[dict[str, Any]]:

        with self._lock:

            tasks = list(
                self._tasks.values()
            )

        if enabled_only:
            tasks = [
                task
                for task in tasks
                if task["enabled"]
            ]

        tasks.sort(
            key=lambda task: (
                task["run_at"]
                or datetime.max
            )
        )

        return [
            self._serialize_task(task)
            for task in tasks
        ]

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(
        self,
        task_id: str,
    ) -> bool:

        return self._set_enabled(
            task_id,
            True,
        )

    def disable(
        self,
        task_id: str,
    ) -> bool:

        return self._set_enabled(
            task_id,
            False,
        )

    def _set_enabled(
        self,
        task_id: str,
        enabled: bool,
    ) -> bool:

        with self._lock:

            task = self._tasks.get(
                task_id
            )

            if task is None:
                return False

            task["enabled"] = enabled

        self._emit(
            "scheduled_task.updated",
            {
                "id": task_id,
                "enabled": enabled,
            },
        )

        return True

    # ========================================================
    # MANUAL EXECUTION
    # ========================================================

    def run_now(
        self,
        task_id: str,
    ) -> Any:
        """Execute a scheduled task immediately."""

        with self._lock:

            task = self._tasks.get(
                task_id
            )

        if task is None:
            raise KeyError(
                f"Scheduled task not found: {task_id}"
            )

        return self._execute_task(
            task
        )

    # ========================================================
    # SCHEDULER LOOP
    # ========================================================

    def _scheduler_loop(self) -> None:

        logger.info(
            "RENIX scheduler loop started."
        )

        while not self._stop_event.is_set():

            try:
                self._check_tasks()

            except Exception:
                logger.exception(
                    "Error in RENIX scheduler loop."
                )

            self._stop_event.wait(
                self.check_interval
            )

        logger.info(
            "RENIX scheduler loop stopped."
        )

    def _check_tasks(self) -> None:

        now = datetime.now()

        with self._lock:

            due_tasks = [
                task
                for task in self._tasks.values()
                if (
                    task["enabled"]
                    and task["run_at"] is not None
                    and task["run_at"] <= now
                )
            ]

        for task in due_tasks:

            try:
                self._execute_task(
                    task
                )

            except Exception:
                logger.exception(
                    "Scheduled task failed: %s",
                    task["name"],
                )

    # ========================================================
    # EXECUTION
    # ========================================================

    def _execute_task(
        self,
        task: dict[str, Any],
    ) -> Any:

        task_id = task["id"]

        self._emit(
            "scheduled_task.started",
            self._serialize_task(task),
        )

        result = None
        error = None

        try:

            callback = task.get(
                "callback"
            )

            if callback is not None:

                if not callable(callback):
                    raise TypeError(
                        "Scheduled task callback must be callable."
                    )

                result = callback(
                    task
                )

            elif self.executor is not None:

                result = self.executor(
                    task.get("action"),
                    task,
                )

            else:

                raise RuntimeError(
                    "No executor or callback configured."
                )

            with self._lock:

                task["last_run"] = datetime.now()

                task["run_count"] += 1

                task["last_error"] = None

                self._schedule_next_run(
                    task
                )

            self._emit(
                "scheduled_task.completed",
                {
                    "task": self._serialize_task(
                        task
                    ),
                    "result": result,
                },
            )

            return result

        except Exception as exc:

            error = str(exc)

            with self._lock:

                task["last_run"] = datetime.now()

                task["run_count"] += 1

                task["last_error"] = error

                self._schedule_next_run(
                    task
                )

            self._emit(
                "scheduled_task.failed",
                {
                    "task": self._serialize_task(
                        task
                    ),
                    "error": error,
                },
            )

            raise

    # ========================================================
    # NEXT RUN CALCULATION
    # ========================================================

    def _schedule_next_run(
        self,
        task: dict[str, Any],
    ) -> None:

        repeat = task.get(
            "repeat",
            "once",
        )

        current = task.get(
            "run_at"
        )

        if repeat == "once":

            task["enabled"] = False
            task["run_at"] = None

            return

        if current is None:
            current = datetime.now()

        if repeat == "hourly":

            task["run_at"] = (
                current
                + timedelta(
                    hours=1
                )
            )

        elif repeat == "daily":

            task["run_at"] = (
                current
                + timedelta(
                    days=1
                )
            )

        elif repeat == "weekly":

            task["run_at"] = (
                current
                + timedelta(
                    weeks=1
                )
            )

        elif repeat == "interval":

            seconds = task.get(
                "interval_seconds"
            )

            if not seconds:
                task["enabled"] = False
                task["run_at"] = None
                return

            task["run_at"] = (
                datetime.now()
                + timedelta(
                    seconds=seconds
                )
            )

        else:

            task["enabled"] = False
            task["run_at"] = None

    # ========================================================
    # TIME PARSING
    # ========================================================

    @staticmethod
    def _parse_run_at(
        value: datetime | str | None,
    ) -> datetime | None:

        if value is None:
            return None

        if isinstance(
            value,
            datetime,
        ):
            return value

        value = value.strip()

        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%H:%M:%S",
            "%H:%M",
        )

        for fmt in formats:

            try:

                parsed = datetime.strptime(
                    value,
                    fmt,
                )

                # Time-only schedules are interpreted
                # as today's time, unless already passed.
                if fmt in {
                    "%H:%M:%S",
                    "%H:%M",
                }:

                    now = datetime.now()

                    parsed = parsed.replace(
                        year=now.year,
                        month=now.month,
                        day=now.day,
                    )

                    if parsed <= now:
                        parsed += timedelta(
                            days=1
                        )

                return parsed

            except ValueError:
                continue

        try:
            return datetime.fromisoformat(
                value
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid run_at value: {value}"
            ) from exc

    # ========================================================
    # CALLBACKS
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
                    "Scheduled task callback failed."
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
                        "Failed to publish scheduler event."
                    )

    # ========================================================
    # SERIALIZATION
    # ========================================================

    @staticmethod
    def _serialize_task(
        task: dict[str, Any],
    ) -> dict[str, Any]:

        serialized = dict(
            task
        )

        callback = serialized.get(
            "callback"
        )

        if callback is not None:
            serialized["callback"] = getattr(
                callback,
                "__name__",
                str(callback),
            )

        for key in (
            "run_at",
            "created_at",
            "last_run",
        ):

            value = serialized.get(
                key
            )

            if isinstance(
                value,
                datetime,
            ):
                serialized[key] = value.isoformat()

        return serialized


__all__ = [
    "ScheduledTaskManager",
]


