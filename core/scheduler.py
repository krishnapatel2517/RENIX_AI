"""
RENIX AI
Core Scheduler

Central scheduling system for RENIX.

Responsibilities:
- Schedule one-time tasks
- Schedule recurring tasks
- Delayed execution
- Periodic execution
- Task cancellation
- Task pausing/resuming
- Event-based scheduling
- Async and sync callbacks
- Persistent scheduler state hooks
- Execution history
- Retry handling
- Scheduler statistics
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Optional


logger = logging.getLogger(
    "RENIX.Scheduler"
)


# ============================================================================
# TYPES
# ============================================================================

TaskCallback = Callable[..., Any]


# ============================================================================
# SCHEDULE TYPES
# ============================================================================

class ScheduleType(str, Enum):
    """
    Types of schedules supported by RENIX.
    """

    ONCE = "once"

    INTERVAL = "interval"

    DAILY = "daily"

    HOURLY = "hourly"

    MINUTELY = "minutely"

    EVENT = "event"


# ============================================================================
# TASK STATUS
# ============================================================================

class TaskStatus(str, Enum):
    """
    Scheduler task states.
    """

    PENDING = "pending"

    RUNNING = "running"

    PAUSED = "paused"

    COMPLETED = "completed"

    FAILED = "failed"

    CANCELLED = "cancelled"

    WAITING = "waiting"


# ============================================================================
# SCHEDULED TASK
# ============================================================================

@dataclass
class ScheduledTask:
    """
    Represents one scheduled RENIX task.
    """

    name: str

    callback: TaskCallback

    schedule_type: ScheduleType = (
        ScheduleType.ONCE
    )

    task_id: str = field(
        default_factory=lambda:
        f"scheduled_{uuid.uuid4().hex}"
    )

    run_at: Optional[float] = None

    interval: Optional[float] = None

    args: tuple[Any, ...] = ()

    kwargs: dict[str, Any] = field(
        default_factory=dict
    )

    status: TaskStatus = (
        TaskStatus.PENDING
    )

    enabled: bool = True

    repeat: bool = False

    max_runs: Optional[int] = None

    run_count: int = 0

    failure_count: int = 0

    max_retries: int = 0

    retry_delay: float = 1.0

    created_at: float = field(
        default_factory=time.time
    )

    last_run_at: Optional[float] = None

    next_run_at: Optional[float] = None

    completed_at: Optional[float] = None

    last_error: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    asyncio_task: Optional[
        asyncio.Task
    ] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def is_finished(self) -> bool:
        """
        Check whether the task has reached a final state.
        """

        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELLED,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the scheduled task to a dictionary.
        """

        return {
            "task_id": self.task_id,
            "name": self.name,
            "schedule_type": (
                self.schedule_type.value
            ),
            "status": (
                self.status.value
            ),
            "enabled": self.enabled,
            "repeat": self.repeat,
            "run_at": self.run_at,
            "interval": self.interval,
            "run_count": self.run_count,
            "failure_count": self.failure_count,
            "max_runs": self.max_runs,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "completed_at": self.completed_at,
            "last_error": self.last_error,
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# SCHEDULER
# ============================================================================

class Scheduler:
    """
    RENIX central scheduler.

    Supports:

    - delayed tasks
    - recurring tasks
    - daily tasks
    - hourly tasks
    - minute tasks
    - retries
    - cancellation
    - pause/resume
    - event-driven scheduling
    """

    def __init__(
        self,
        *,
        max_history: int = 1000,
    ) -> None:

        self.logger = logger

        self.max_history = max(
            1,
            int(max_history),
        )

        self.tasks: dict[
            str,
            ScheduledTask,
        ] = {}

        self.history: list[
            dict[str, Any]
        ] = []

        self.running = False

        self.initialized = False

        self.started_at = time.time()

        self.total_executions = 0

        self.total_failures = 0

        self.total_cancellations = 0

        self.total_completed = 0

        self._lock = asyncio.Lock()

        self._runner_task: Optional[
            asyncio.Task
        ] = None

        self._shutdown_event = (
            asyncio.Event()
        )

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the scheduler.
        """

        if self.initialized:

            return

        self.initialized = True

        self.logger.info(
            "RENIX Scheduler initialized."
        )

    # ========================================================================
    # START
    # ========================================================================

    async def start(
        self,
    ) -> None:
        """
        Start the scheduler loop.
        """

        if not self.initialized:

            await self.initialize()

        if self.running:

            return

        self.running = True

        self._shutdown_event.clear()

        self._runner_task = asyncio.create_task(
            self._scheduler_loop()
        )

        self.logger.info(
            "RENIX Scheduler started."
        )

    # ========================================================================
    # STOP
    # ========================================================================

    async def stop(
        self,
    ) -> None:
        """
        Stop scheduler execution.
        """

        self.running = False

        self._shutdown_event.set()

        if (
            self._runner_task is not None
            and not self._runner_task.done()
        ):

            self._runner_task.cancel()

            try:

                await self._runner_task

            except asyncio.CancelledError:

                pass

        self._runner_task = None

        self.logger.info(
            "RENIX Scheduler stopped."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Completely shut down scheduler.
        """

        await self.stop()

        async with self._lock:

            for task in self.tasks.values():

                if (
                    task.asyncio_task is not None
                    and not task.asyncio_task.done()
                ):

                    task.asyncio_task.cancel()

                if not task.is_finished():

                    task.status = (
                        TaskStatus.CANCELLED
                    )

            self.tasks.clear()

        self.initialized = False

        self.logger.info(
            "RENIX Scheduler shutdown."
        )

    # ========================================================================
    # SCHEDULER LOOP
    # ========================================================================

    async def _scheduler_loop(
        self,
    ) -> None:
        """
        Main scheduler loop.
        """

        while self.running:

            try:

                await self._process_due_tasks()

                try:

                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=0.25,
                    )

                except asyncio.TimeoutError:

                    pass

            except asyncio.CancelledError:

                break

            except Exception:

                self.logger.exception(
                    "Scheduler loop error."
                )

                await asyncio.sleep(
                    1
                )

    # ========================================================================
    # PROCESS DUE TASKS
    # ========================================================================

    async def _process_due_tasks(
        self,
    ) -> None:
        """
        Find and execute due tasks.
        """

        now = time.time()

        due_tasks: list[
            ScheduledTask
        ] = []

        async with self._lock:

            for task in self.tasks.values():

                if not task.enabled:

                    continue

                if task.status in {
                    TaskStatus.CANCELLED,
                    TaskStatus.COMPLETED,
                    TaskStatus.RUNNING,
                    TaskStatus.PAUSED,
                }:

                    continue

                if task.next_run_at is None:

                    continue

                if task.next_run_at <= now:

                    due_tasks.append(
                        task
                    )

        for task in due_tasks:

            task.asyncio_task = (
                asyncio.create_task(
                    self._execute_task(
                        task
                    )
                )
            )

    # ========================================================================
    # ADD TASK
    # ========================================================================

    async def add_task(
        self,
        name: str,
        callback: TaskCallback,
        *,
        delay: Optional[float] = None,
        run_at: Optional[
            datetime | float
        ] = None,
        interval: Optional[float] = None,
        schedule_type: ScheduleType = (
            ScheduleType.ONCE
        ),
        args: tuple[Any, ...] = (),
        kwargs: Optional[
            dict[str, Any]
        ] = None,
        repeat: bool = False,
        max_runs: Optional[int] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Add a scheduled task.

        Parameters:

        delay:
            Seconds from now before execution.

        run_at:
            Unix timestamp or datetime.

        interval:
            Seconds between recurring executions.
        """

        if not callable(callback):

            raise TypeError(
                "callback must be callable."
            )

        if not name.strip():

            raise ValueError(
                "Task name cannot be empty."
            )

        now = time.time()

        calculated_run_at: Optional[
            float
        ] = None

        if delay is not None:

            if delay < 0:

                raise ValueError(
                    "delay cannot be negative."
                )

            calculated_run_at = (
                now + float(delay)
            )

        elif run_at is not None:

            if isinstance(
                run_at,
                datetime,
            ):

                calculated_run_at = (
                    run_at.timestamp()
                )

            else:

                calculated_run_at = float(
                    run_at
                )

        else:

            calculated_run_at = now

        if interval is not None:

            if interval <= 0:

                raise ValueError(
                    "interval must be greater than zero."
                )

        task = ScheduledTask(
            name=name.strip(),
            callback=callback,
            schedule_type=schedule_type,
            run_at=calculated_run_at,
            interval=interval,
            args=tuple(args),
            kwargs=dict(
                kwargs or {}
            ),
            repeat=repeat,
            max_runs=max_runs,
            max_retries=max(
                0,
                int(max_retries),
            ),
            retry_delay=max(
                0.0,
                float(retry_delay),
            ),
            next_run_at=(
                calculated_run_at
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        if schedule_type == (
            ScheduleType.EVENT
        ):

            task.status = (
                TaskStatus.WAITING
            )

        async with self._lock:

            self.tasks[
                task.task_id
            ] = task

        self.logger.info(
            "Scheduled task %s (%s)",
            task.name,
            task.task_id,
        )

        return task.task_id

    # ========================================================================
    # SCHEDULE
    # ========================================================================

    async def schedule(
        self,
        name: str,
        callback: TaskCallback,
        delay: float = 0,
        **kwargs: Any,
    ) -> str:
        """
        Simple scheduling helper.
        """

        return await self.add_task(
            name,
            callback,
            delay=delay,
            **kwargs,
        )

    # ========================================================================
    # SCHEDULE ONCE
    # ========================================================================

    async def schedule_once(
        self,
        name: str,
        callback: TaskCallback,
        *,
        delay: float = 0,
        args: tuple[Any, ...] = (),
        kwargs: Optional[
            dict[str, Any]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Schedule a task for one execution.
        """

        return await self.add_task(
            name,
            callback,
            delay=delay,
            schedule_type=(
                ScheduleType.ONCE
            ),
            args=args,
            kwargs=kwargs,
            repeat=False,
            metadata=metadata,
        )

    # ========================================================================
    # SCHEDULE REPEATING
    # ========================================================================

    async def schedule_interval(
        self,
        name: str,
        callback: TaskCallback,
        interval: float,
        *,
        immediate: bool = False,
        args: tuple[Any, ...] = (),
        kwargs: Optional[
            dict[str, Any]
        ] = None,
        max_runs: Optional[int] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Schedule a repeating task.
        """

        delay = (
            0
            if immediate
            else interval
        )

        return await self.add_task(
            name,
            callback,
            delay=delay,
            interval=interval,
            schedule_type=(
                ScheduleType.INTERVAL
            ),
            args=args,
            kwargs=kwargs,
            repeat=True,
            max_runs=max_runs,
            metadata=metadata,
        )

    # ========================================================================
    # DAILY
    # ========================================================================

    async def schedule_daily(
        self,
        name: str,
        callback: TaskCallback,
        hour: int,
        minute: int = 0,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Optional[
            dict[str, Any]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Schedule a task daily at a specified local time.
        """

        if not 0 <= hour <= 23:

            raise ValueError(
                "hour must be between 0 and 23."
            )

        if not 0 <= minute <= 59:

            raise ValueError(
                "minute must be between 0 and 59."
            )

        now = datetime.now()

        target = now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )

        if target <= now:

            target += timedelta(
                days=1
            )

        return await self.add_task(
            name,
            callback,
            run_at=target,
            interval=86400,
            schedule_type=(
                ScheduleType.DAILY
            ),
            args=args,
            kwargs=kwargs,
            repeat=True,
            metadata=metadata,
        )

    # ========================================================================
    # HOURLY
    # ========================================================================

    async def schedule_hourly(
        self,
        name: str,
        callback: TaskCallback,
        *,
        minute: int = 0,
        args: tuple[Any, ...] = (),
        kwargs: Optional[
            dict[str, Any]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Schedule a task every hour at a specific minute.
        """

        if not 0 <= minute <= 59:

            raise ValueError(
                "minute must be between 0 and 59."
            )

        now = datetime.now()

        target = now.replace(
            minute=minute,
            second=0,
            microsecond=0,
        )

        if target <= now:

            target += timedelta(
                hours=1
            )

        return await self.add_task(
            name,
            callback,
            run_at=target,
            interval=3600,
            schedule_type=(
                ScheduleType.HOURLY
            ),
            args=args,
            kwargs=kwargs,
            repeat=True,
            metadata=metadata,
        )

    # ========================================================================
    # MINUTELY
    # ========================================================================

    async def schedule_minutely(
        self,
        name: str,
        callback: TaskCallback,
        interval_minutes: int = 1,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Optional[
            dict[str, Any]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> str:
        """
        Schedule a task every N minutes.
        """

        if interval_minutes <= 0:

            raise ValueError(
                "interval_minutes must be greater than zero."
            )

        interval = (
            interval_minutes * 60
        )

        return await self.add_task(
            name,
            callback,
            delay=interval,
            interval=interval,
            schedule_type=(
                ScheduleType.MINUTELY
            ),
            args=args,
            kwargs=kwargs,
            repeat=True,
            metadata=metadata,
        )

    # ========================================================================
    # EXECUTE TASK
    # ========================================================================

    async def _execute_task(
        self,
        task: ScheduledTask,
    ) -> None:
        """
        Execute a scheduled task.
        """

        if not task.enabled:

            return

        if task.status == (
            TaskStatus.PAUSED
        ):

            return

        task.status = TaskStatus.RUNNING

        task.last_run_at = time.time()

        task.run_count += 1

        self.total_executions += 1

        execution_start = time.time()

        try:

            result = task.callback(
                *task.args,
                **task.kwargs,
            )

            if inspect.isawaitable(
                result
            ):

                result = await result

            execution_time = (
                time.time()
                - execution_start
            )

            task.last_error = None

            self._record_execution(
                task,
                success=True,
                result=result,
                execution_time=(
                    execution_time
                ),
            )

            if self._should_repeat(
                task
            ):

                self._schedule_next_run(
                    task
                )

                task.status = (
                    TaskStatus.PENDING
                )

            else:

                task.status = (
                    TaskStatus.COMPLETED
                )

                task.completed_at = (
                    time.time()
                )

                self.total_completed += 1

        except asyncio.CancelledError:

            task.status = (
                TaskStatus.CANCELLED
            )

            self.total_cancellations += 1

            raise

        except Exception as exc:

            task.failure_count += 1

            self.total_failures += 1

            task.last_error = str(
                exc
            )

            self._record_execution(
                task,
                success=False,
                error=str(exc),
                execution_time=(
                    time.time()
                    - execution_start
                ),
            )

            self.logger.exception(
                "Scheduled task failed: %s",
                task.name,
            )

            if (
                task.failure_count
                <= task.max_retries
            ):

                task.status = (
                    TaskStatus.PENDING
                )

                task.next_run_at = (
                    time.time()
                    + task.retry_delay
                )

            elif self._should_repeat(
                task
            ):

                self._schedule_next_run(
                    task
                )

                task.status = (
                    TaskStatus.PENDING
                )

            else:

                task.status = (
                    TaskStatus.FAILED
                )

    # ========================================================================
    # SHOULD REPEAT
    # ========================================================================

    def _should_repeat(
        self,
        task: ScheduledTask,
    ) -> bool:
        """
        Determine whether a task should run again.
        """

        if not task.repeat:

            return False

        if task.max_runs is not None:

            if task.run_count >= (
                task.max_runs
            ):

                return False

        if task.interval is None:

            return False

        return True

    # ========================================================================
    # SCHEDULE NEXT RUN
    # ========================================================================

    def _schedule_next_run(
        self,
        task: ScheduledTask,
    ) -> None:
        """
        Calculate next execution time.
        """

        if task.interval is None:

            task.next_run_at = None

            return

        task.next_run_at = (
            time.time()
            + task.interval
        )

    # ========================================================================
    # RECORD EXECUTION
    # ========================================================================

    def _record_execution(
        self,
        task: ScheduledTask,
        *,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        execution_time: float = 0,
    ) -> None:
        """
        Store execution history.
        """

        record = {
            "task_id": task.task_id,
            "task_name": task.name,
            "timestamp": time.time(),
            "success": success,
            "result": result,
            "error": error,
            "execution_time": execution_time,
            "run_count": task.run_count,
        }

        self.history.append(
            record
        )

        if len(
            self.history
        ) > self.max_history:

            self.history = self.history[
                -self.max_history:
            ]

    # ========================================================================
    # CANCEL
    # ========================================================================

    async def cancel(
        self,
        task_id: str,
    ) -> bool:
        """
        Cancel a scheduled task.
        """

        task = self.tasks.get(
            task_id
        )

        if task is None:

            return False

        task.enabled = False

        task.status = (
            TaskStatus.CANCELLED
        )

        task.next_run_at = None

        if (
            task.asyncio_task is not None
            and not task.asyncio_task.done()
        ):

            task.asyncio_task.cancel()

        self.total_cancellations += 1

        return True

    # ========================================================================
    # PAUSE
    # ========================================================================

    async def pause(
        self,
        task_id: str,
    ) -> bool:
        """
        Pause a task.
        """

        task = self.tasks.get(
            task_id
        )

        if task is None:

            return False

        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELLED,
            TaskStatus.FAILED,
        }:

            return False

        task.status = (
            TaskStatus.PAUSED
        )

        task.enabled = False

        return True

    # ========================================================================
    # RESUME
    # ========================================================================

    async def resume(
        self,
        task_id: str,
    ) -> bool:
        """
        Resume a paused task.
        """

        task = self.tasks.get(
            task_id
        )

        if task is None:

            return False

        if task.status != (
            TaskStatus.PAUSED
        ):

            return False

        task.enabled = True

        task.status = (
            TaskStatus.PENDING
        )

        if task.next_run_at is None:

            task.next_run_at = (
                time.time()
            )

        return True

    # ========================================================================
    # REMOVE TASK
    # ========================================================================

    async def remove(
        self,
        task_id: str,
    ) -> bool:
        """
        Cancel and remove a task.
        """

        task = self.tasks.get(
            task_id
        )

        if task is None:

            return False

        await self.cancel(
            task_id
        )

        async with self._lock:

            self.tasks.pop(
                task_id,
                None,
            )

        return True

    # ========================================================================
    # GET TASK
    # ========================================================================

    def get_task(
        self,
        task_id: str,
    ) -> Optional[ScheduledTask]:
        """
        Get a scheduled task.
        """

        return self.tasks.get(
            task_id
        )

    # ========================================================================
    # GET ALL TASKS
    # ========================================================================

    def get_tasks(
        self,
        *,
        status: Optional[
            TaskStatus
        ] = None,
    ) -> list[ScheduledTask]:
        """
        Get all scheduled tasks.
        """

        tasks = list(
            self.tasks.values()
        )

        if status is not None:

            tasks = [
                task
                for task in tasks
                if task.status == status
            ]

        return tasks

    # ========================================================================
    # FIND TASK
    # ========================================================================

    def find_task(
        self,
        name: str,
    ) -> Optional[ScheduledTask]:
        """
        Find the first task matching a name.
        """

        for task in self.tasks.values():

            if task.name == name:

                return task

        return None

    # ========================================================================
    # CLEAR COMPLETED
    # ========================================================================

    async def clear_completed(
        self,
    ) -> int:
        """
        Remove completed tasks.
        """

        completed = [
            task_id
            for task_id, task
            in self.tasks.items()
            if task.status
            == TaskStatus.COMPLETED
        ]

        for task_id in completed:

            self.tasks.pop(
                task_id,
                None,
            )

        return len(
            completed
        )

    # ========================================================================
    # CLEAR ALL
    # ========================================================================

    async def clear_all(
        self,
    ) -> None:
        """
        Cancel and remove all tasks.
        """

        task_ids = list(
            self.tasks.keys()
        )

        for task_id in task_ids:

            await self.cancel(
                task_id
            )

        self.tasks.clear()

    # ========================================================================
    # HISTORY
    # ========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Return execution history.
        """

        if limit is None:

            return list(
                self.history
            )

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:

            return []

        return self.history[
            -limit:
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
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return scheduler statistics.
        """

        status_counts: dict[
            str,
            int,
        ] = {}

        for task in self.tasks.values():

            status = task.status.value

            status_counts[
                status
            ] = (
                status_counts.get(
                    status,
                    0,
                )
                + 1
            )

        return {
            "initialized": (
                self.initialized
            ),
            "running": self.running,
            "task_count": len(
                self.tasks
            ),
            "history_count": len(
                self.history
            ),
            "total_executions": (
                self.total_executions
            ),
            "total_failures": (
                self.total_failures
            ),
            "total_cancellations": (
                self.total_cancellations
            ),
            "total_completed": (
                self.total_completed
            ),
            "status_counts": (
                status_counts
            ),
            "started_at": (
                self.started_at
            ),
        }

    # ========================================================================
    # EXPORT STATE
    # ========================================================================

    def export_state(
        self,
    ) -> dict[str, Any]:
        """
        Export scheduler state.
        """

        return {
            "initialized": (
                self.initialized
            ),
            "running": self.running,
            "tasks": [
                task.to_dict()
                for task in self.tasks.values()
            ],
            "statistics": self.statistics(),
        }


# ============================================================================
# GLOBAL SCHEDULER
# ============================================================================

scheduler = Scheduler()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def schedule(
    name: str,
    callback: TaskCallback,
    delay: float = 0,
    **kwargs: Any,
) -> str:
    """
    Schedule a task using the global scheduler.
    """

    return await scheduler.schedule(
        name,
        callback,
        delay,
        **kwargs,
    )


async def schedule_once(
    name: str,
    callback: TaskCallback,
    *,
    delay: float = 0,
    args: tuple[Any, ...] = (),
    kwargs: Optional[
        dict[str, Any]
    ] = None,
) -> str:
    """
    Schedule a one-time task globally.
    """

    return await scheduler.schedule_once(
        name,
        callback,
        delay=delay,
        args=args,
        kwargs=kwargs,
    )


async def schedule_interval(
    name: str,
    callback: TaskCallback,
    interval: float,
    *,
    immediate: bool = False,
) -> str:
    """
    Schedule a recurring task globally.
    """

    return await scheduler.schedule_interval(
        name,
        callback,
        interval,
        immediate=immediate,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ScheduleType",
    "TaskStatus",
    "ScheduledTask",
    "Scheduler",
    "scheduler",
    "schedule",
    "schedule_once",
    "schedule_interval",
]


