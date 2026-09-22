"""
RENIX AI - Task Manager
=======================

Central task-management system for RENIX.

Responsibilities:
    - Create tasks
    - Queue tasks
    - Prioritize tasks
    - Run tasks asynchronously
    - Track task status
    - Pause tasks
    - Resume tasks
    - Cancel tasks
    - Retry failed tasks
    - Track progress
    - Store task results
    - Handle dependencies
    - Handle task timeouts
    - Maintain task history
    - Provide task statistics

This module is designed to work as the central task layer
between RENIX's reasoning/planning systems and execution systems.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import traceback

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Union,
)


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("RENIX.TaskManager")


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

TaskFunction = Callable[..., Any]

AsyncTaskFunction = Callable[
    ...,
    Awaitable[Any],
]

TaskCallable = Union[
    TaskFunction,
    AsyncTaskFunction,
]


# ============================================================================
# ENUMS
# ============================================================================

class TaskStatus(str, Enum):
    """
    Current state of a task.
    """

    CREATED = "created"

    QUEUED = "queued"

    RUNNING = "running"

    PAUSED = "paused"

    COMPLETED = "completed"

    FAILED = "failed"

    CANCELLED = "cancelled"

    TIMEOUT = "timeout"

    RETRYING = "retrying"


class TaskPriority(int, Enum):
    """
    Task priority.

    Higher number = higher priority.
    """

    LOWEST = 0

    LOW = 1

    NORMAL = 2

    HIGH = 3

    CRITICAL = 4


# ============================================================================
# TASK
# ============================================================================

@dataclass
class RENIXTask:
    """
    Represents one RENIX task.
    """

    task_id: str

    name: str

    function: Optional[TaskCallable] = None

    args: tuple[Any, ...] = field(
        default_factory=tuple
    )

    kwargs: dict[str, Any] = field(
        default_factory=dict
    )

    priority: TaskPriority = (
        TaskPriority.NORMAL
    )

    status: TaskStatus = (
        TaskStatus.CREATED
    )

    description: str = ""

    dependencies: list[str] = field(
        default_factory=list
    )

    max_retries: int = 0

    retry_count: int = 0

    timeout: Optional[float] = None

    progress: float = 0.0

    result: Any = None

    error: Optional[str] = None

    traceback_text: Optional[str] = None

    created_at: float = field(
        default_factory=time.time
    )

    queued_at: Optional[float] = None

    started_at: Optional[float] = None

    completed_at: Optional[float] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    cancellation_requested: bool = False

    pause_requested: bool = False

    _asyncio_task: Optional[
        asyncio.Task
    ] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert task to a serializable dictionary.
        """

        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority.name,
            "priority_value": int(
                self.priority.value
            ),
            "status": self.status.value,
            "description": self.description,
            "dependencies": list(
                self.dependencies
            ),
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback_text,
            "created_at": self.created_at,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": dict(
                self.metadata
            ),
            "cancellation_requested": (
                self.cancellation_requested
            ),
            "pause_requested": (
                self.pause_requested
            ),
        }


# ============================================================================
# TASK RESULT
# ============================================================================

@dataclass
class TaskResult:
    """
    Result returned after task execution.
    """

    task_id: str

    status: TaskStatus

    result: Any = None

    error: Optional[str] = None

    duration: float = 0.0

    retry_count: int = 0

    completed_at: float = field(
        default_factory=time.time
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration": self.duration,
            "retry_count": self.retry_count,
            "completed_at": self.completed_at,
        }


# ============================================================================
# TASK MANAGER
# ============================================================================

class TaskManager:
    """
    Central asynchronous task manager for RENIX.
    """

    def __init__(
        self,
        *,
        max_concurrent_tasks: int = 4,
        max_history: int = 500,
    ) -> None:

        self.logger = logger

        self.max_concurrent_tasks = max(
            1,
            int(
                max_concurrent_tasks
            ),
        )

        self.max_history = max(
            1,
            int(
                max_history
            ),
        )

        # ------------------------------------------------------------
        # Task storage
        # ------------------------------------------------------------

        self.tasks: dict[
            str,
            RENIXTask,
        ] = {}

        self.history: list[
            RENIXTask
        ] = []

        # ------------------------------------------------------------
        # Runtime
        # ------------------------------------------------------------

        self.running = False

        self._worker_task: Optional[
            asyncio.Task
        ] = None

        self._queue_event = (
            asyncio.Event()
        )

        self._semaphore = (
            asyncio.Semaphore(
                self.max_concurrent_tasks
            )
        )

        self._lock = (
            asyncio.Lock()
        )

        # ------------------------------------------------------------
        # Counter
        # ------------------------------------------------------------

        self._task_counter = 0

        # ------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------

        self.total_created = 0

        self.total_completed = 0

        self.total_failed = 0

        self.total_cancelled = 0

        self.total_timeout = 0

        self.total_retried = 0

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Start the task manager worker.
        """

        if self.running:
            return

        self.running = True

        self._worker_task = (
            asyncio.create_task(
                self._worker_loop()
            )
        )

        self.logger.info(
            "RENIX Task Manager initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
        *,
        cancel_running: bool = False,
    ) -> None:
        """
        Shut down the task manager.
        """

        self.running = False

        self._queue_event.set()

        if cancel_running:

            for task in list(
                self.tasks.values()
            ):

                if task.status in {
                    TaskStatus.RUNNING,
                    TaskStatus.QUEUED,
                    TaskStatus.RETRYING,
                }:

                    await self.cancel_task(
                        task.task_id
                    )

        if self._worker_task:

            try:

                await self._worker_task

            except asyncio.CancelledError:

                pass

            finally:

                self._worker_task = None

        self.logger.info(
            "RENIX Task Manager shutdown."
        )

    # ========================================================================
    # GENERATE TASK ID
    # ========================================================================

    def _generate_task_id(
        self,
    ) -> str:
        """
        Generate a unique task ID.
        """

        self._task_counter += 1

        timestamp = int(
            time.time() * 1000
        )

        return (
            f"renix_task_"
            f"{timestamp}_"
            f"{self._task_counter}"
        )

    # ========================================================================
    # CREATE TASK
    # ========================================================================

    async def create_task(
        self,
        name: str,
        function: Optional[
            TaskCallable
        ] = None,
        *,
        args: Optional[
            tuple[Any, ...]
        ] = None,
        kwargs: Optional[
            dict[str, Any]
        ] = None,
        priority: TaskPriority = (
            TaskPriority.NORMAL
        ),
        description: str = "",
        dependencies: Optional[
            list[str]
        ] = None,
        max_retries: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        auto_queue: bool = True,
    ) -> RENIXTask:
        """
        Create a new task.
        """

        if not name:

            raise ValueError(
                "Task name cannot be empty."
            )

        if not isinstance(
            priority,
            TaskPriority,
        ):

            priority = TaskPriority(
                int(priority)
            )

        task = RENIXTask(
            task_id=(
                self._generate_task_id()
            ),
            name=name,
            function=function,
            args=args or (),
            kwargs=kwargs or {},
            priority=priority,
            description=description,
            dependencies=(
                dependencies or []
            ),
            max_retries=max(
                0,
                int(max_retries),
            ),
            timeout=timeout,
            metadata=(
                metadata or {}
            ),
        )

        async with self._lock:

            self.tasks[
                task.task_id
            ] = task

            self.total_created += 1

        if auto_queue:

            await self.queue_task(
                task.task_id
            )

        return task

    # ========================================================================
    # QUEUE TASK
    # ========================================================================

    async def queue_task(
        self,
        task_id: str,
    ) -> bool:
        """
        Put a task into the execution queue.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELLED,
        }:

            return False

        task.status = (
            TaskStatus.QUEUED
        )

        task.queued_at = time.time()

        task.cancellation_requested = False

        self._queue_event.set()

        return True

    # ========================================================================
    # WORKER LOOP
    # ========================================================================

    async def _worker_loop(
        self,
    ) -> None:
        """
        Continuously searches for queued tasks.
        """

        while self.running:

            try:

                task = (
                    self._get_next_queued_task()
                )

                if task is None:

                    self._queue_event.clear()

                    try:

                        await asyncio.wait_for(
                            self._queue_event.wait(),
                            timeout=0.5,
                        )

                    except asyncio.TimeoutError:

                        pass

                    continue

                asyncio.create_task(
                    self._execute_with_semaphore(
                        task
                    )
                )

                await asyncio.sleep(
                    0
                )

            except asyncio.CancelledError:

                break

            except Exception:

                self.logger.exception(
                    "Task worker loop error."
                )

                await asyncio.sleep(
                    0.5
                )

    # ========================================================================
    # GET NEXT TASK
    # ========================================================================

    def _get_next_queued_task(
        self,
    ) -> Optional[RENIXTask]:
        """
        Return the highest-priority runnable task.
        """

        candidates: list[
            RENIXTask
        ] = []

        for task in (
            self.tasks.values()
        ):

            if task.status != (
                TaskStatus.QUEUED
            ):

                continue

            if not self._dependencies_complete(
                task
            ):

                continue

            candidates.append(
                task
            )

        if not candidates:

            return None

        candidates.sort(
            key=lambda task: (
                -int(
                    task.priority.value
                ),
                task.created_at,
            )
        )

        return candidates[0]

    # ========================================================================
    # DEPENDENCY CHECK
    # ========================================================================

    def _dependencies_complete(
        self,
        task: RENIXTask,
    ) -> bool:
        """
        Check whether all task dependencies are completed.
        """

        if not task.dependencies:

            return True

        for dependency_id in (
            task.dependencies
        ):

            dependency = self.get_task(
                dependency_id
            )

            if dependency is None:

                return False

            if dependency.status != (
                TaskStatus.COMPLETED
            ):

                return False

        return True

    # ========================================================================
    # EXECUTE WITH SEMAPHORE
    # ========================================================================

    async def _execute_with_semaphore(
        self,
        task: RENIXTask,
    ) -> None:
        """
        Execute a task while respecting concurrency limits.
        """

        async with self._semaphore:

            await self._execute_task(
                task
            )

    # ========================================================================
    # EXECUTE TASK
    # ========================================================================

    async def _execute_task(
        self,
        task: RENIXTask,
    ) -> None:
        """
        Execute an individual task.
        """

        if task.status != (
            TaskStatus.QUEUED
        ):

            return

        if task.cancellation_requested:

            await self.cancel_task(
                task.task_id
            )

            return

        task.status = (
            TaskStatus.RUNNING
        )

        task.started_at = time.time()

        task.progress = 0.0

        try:

            if task.function is None:

                raise RuntimeError(
                    "Task has no executable function."
                )

            task._asyncio_task = (
                asyncio.current_task()
            )

            if task.timeout is not None:

                result = await asyncio.wait_for(
                    self._call_function(
                        task
                    ),
                    timeout=float(
                        task.timeout
                    ),
                )

            else:

                result = await self._call_function(
                    task
                )

            if task.cancellation_requested:

                task.status = (
                    TaskStatus.CANCELLED
                )

                self.total_cancelled += 1

            else:

                task.result = result

                task.progress = 100.0

                task.status = (
                    TaskStatus.COMPLETED
                )

                self.total_completed += 1

        except asyncio.TimeoutError:

            task.status = (
                TaskStatus.TIMEOUT
            )

            task.error = (
                "Task execution timed out."
            )

            self.total_timeout += 1

            if await self._retry_if_possible(
                task
            ):

                return

        except asyncio.CancelledError:

            task.status = (
                TaskStatus.CANCELLED
            )

            task.error = (
                "Task was cancelled."
            )

            self.total_cancelled += 1

        except Exception as exc:

            task.error = str(
                exc
            )

            task.traceback_text = (
                traceback.format_exc()
            )

            if await self._retry_if_possible(
                task
            ):

                return

            task.status = (
                TaskStatus.FAILED
            )

            self.total_failed += 1

            self.logger.exception(
                "Task failed: %s",
                task.name,
            )

        finally:

            task.completed_at = time.time()

            task._asyncio_task = None

            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            }:

                self._move_to_history(
                    task
                )

    # ========================================================================
    # CALL FUNCTION
    # ========================================================================

    async def _call_function(
        self,
        task: RENIXTask,
    ) -> Any:
        """
        Execute synchronous or asynchronous task functions.
        """

        function = task.function

        if function is None:

            raise RuntimeError(
                "Task function is missing."
            )

        if inspect.iscoroutinefunction(
            function
        ):

            return await function(
                *task.args,
                **task.kwargs,
            )

        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            None,
            lambda: function(
                *task.args,
                **task.kwargs,
            ),
        )

    # ========================================================================
    # RETRY
    # ========================================================================

    async def _retry_if_possible(
        self,
        task: RENIXTask,
    ) -> bool:
        """
        Retry a failed task if retries remain.
        """

        if (
            task.retry_count
            >= task.max_retries
        ):

            return False

        if task.cancellation_requested:

            return False

        task.retry_count += 1

        self.total_retried += 1

        task.status = (
            TaskStatus.RETRYING
        )

        task.progress = 0.0

        await asyncio.sleep(
            min(
                2.0 * task.retry_count,
                10.0,
            )
        )

        if task.cancellation_requested:

            task.status = (
                TaskStatus.CANCELLED
            )

            return False

        task.status = (
            TaskStatus.QUEUED
        )

        task.queued_at = time.time()

        self._queue_event.set()

        return True

    # ========================================================================
    # GET TASK
    # ========================================================================

    def get_task(
        self,
        task_id: str,
    ) -> Optional[RENIXTask]:
        """
        Retrieve a task by ID.
        """

        return self.tasks.get(
            task_id
        )

    # ========================================================================
    # GET TASKS
    # ========================================================================

    def get_tasks(
        self,
        *,
        status: Optional[
            TaskStatus
        ] = None,
    ) -> list[RENIXTask]:
        """
        Return tasks, optionally filtered by status.
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
    # PAUSE TASK
    # ========================================================================

    async def pause_task(
        self,
        task_id: str,
    ) -> bool:
        """
        Request a task pause.

        A currently running arbitrary Python function cannot always
        be forcefully paused safely, so the task receives a pause
        request. Queued tasks can be paused immediately.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        if task.status == (
            TaskStatus.QUEUED
        ):

            task.status = (
                TaskStatus.PAUSED
            )

            task.pause_requested = True

            return True

        if task.status == (
            TaskStatus.RUNNING
        ):

            task.pause_requested = True

            return True

        return False

    # ========================================================================
    # RESUME TASK
    # ========================================================================

    async def resume_task(
        self,
        task_id: str,
    ) -> bool:
        """
        Resume a paused task.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        if task.status != (
            TaskStatus.PAUSED
        ):

            return False

        task.pause_requested = False

        task.status = (
            TaskStatus.QUEUED
        )

        task.queued_at = time.time()

        self._queue_event.set()

        return True

    # ========================================================================
    # CANCEL TASK
    # ========================================================================

    async def cancel_task(
        self,
        task_id: str,
    ) -> bool:
        """
        Cancel a task.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        if task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELLED,
        }:

            return False

        task.cancellation_requested = True

        if task._asyncio_task is not None:

            current = (
                asyncio.current_task()
            )

            if (
                task._asyncio_task
                is not current
            ):

                task._asyncio_task.cancel()

        else:

            task.status = (
                TaskStatus.CANCELLED
            )

            task.completed_at = time.time()

            self.total_cancelled += 1

            self._move_to_history(
                task
            )

        return True

    # ========================================================================
    # CANCEL ALL
    # ========================================================================

    async def cancel_all(
        self,
    ) -> int:
        """
        Cancel all active tasks.
        """

        count = 0

        for task in list(
            self.tasks.values()
        ):

            if task.status in {
                TaskStatus.CREATED,
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
                TaskStatus.PAUSED,
                TaskStatus.RETRYING,
            }:

                if await self.cancel_task(
                    task.task_id
                ):

                    count += 1

        return count

    # ========================================================================
    # RETRY TASK
    # ========================================================================

    async def retry_task(
        self,
        task_id: str,
    ) -> bool:
        """
        Manually retry a failed task.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        if task.status not in {
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
        }:

            return False

        task.error = None

        task.traceback_text = None

        task.retry_count += 1

        task.progress = 0.0

        task.status = (
            TaskStatus.QUEUED
        )

        task.queued_at = time.time()

        self.total_retried += 1

        self._queue_event.set()

        return True

    # ========================================================================
    # UPDATE PROGRESS
    # ========================================================================

    def update_progress(
        self,
        task_id: str,
        progress: float,
    ) -> bool:
        """
        Update task progress from 0 to 100.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        task.progress = max(
            0.0,
            min(
                100.0,
                float(progress),
            ),
        )

        return True

    # ========================================================================
    # COMPLETE TASK MANUALLY
    # ========================================================================

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
    ) -> bool:
        """
        Manually mark a task as completed.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        if task.status == (
            TaskStatus.COMPLETED
        ):

            return False

        task.result = result

        task.progress = 100.0

        task.status = (
            TaskStatus.COMPLETED
        )

        task.completed_at = time.time()

        self.total_completed += 1

        self._move_to_history(
            task
        )

        return True

    # ========================================================================
    # FAIL TASK MANUALLY
    # ========================================================================

    async def fail_task(
        self,
        task_id: str,
        error: str,
    ) -> bool:
        """
        Manually mark a task as failed.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        task.error = str(
            error
        )

        task.status = (
            TaskStatus.FAILED
        )

        task.completed_at = time.time()

        self.total_failed += 1

        self._move_to_history(
            task
        )

        return True

    # ========================================================================
    # WAIT FOR TASK
    # ========================================================================

    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[TaskResult]:
        """
        Wait until a task reaches a final state.
        """

        start = time.time()

        while True:

            task = self.get_task(
                task_id
            )

            if task is None:

                return None

            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            }:

                duration = 0.0

                if task.started_at:

                    end = (
                        task.completed_at
                        or time.time()
                    )

                    duration = (
                        end
                        - task.started_at
                    )

                return TaskResult(
                    task_id=task.task_id,
                    status=task.status,
                    result=task.result,
                    error=task.error,
                    duration=duration,
                    retry_count=task.retry_count,
                    completed_at=(
                        task.completed_at
                        or time.time()
                    ),
                )

            if (
                timeout is not None
                and (
                    time.time()
                    - start
                    >= timeout
                )
            ):

                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.TIMEOUT,
                    error=(
                        "Waiting for task timed out."
                    ),
                    retry_count=task.retry_count,
                )

            await asyncio.sleep(
                0.05
            )

    # ========================================================================
    # WAIT FOR ALL
    # ========================================================================

    async def wait_for_all(
        self,
        task_ids: list[str],
        timeout: Optional[float] = None,
    ) -> list[TaskResult]:
        """
        Wait for multiple tasks.
        """

        async def wait_one(
            task_id: str,
        ) -> Optional[TaskResult]:

            return await self.wait_for_task(
                task_id,
                timeout=timeout,
            )

        results = await asyncio.gather(
            *[
                wait_one(
                    task_id
                )
                for task_id in task_ids
            ]
        )

        return [
            result
            for result in results
            if result is not None
        ]

    # ========================================================================
    # MOVE TO HISTORY
    # ========================================================================

    def _move_to_history(
        self,
        task: RENIXTask,
    ) -> None:
        """
        Add completed task to history.
        """

        if task not in self.history:

            self.history.append(
                task
            )

        if len(
            self.history
        ) > self.max_history:

            self.history = self.history[
                -self.max_history:
            ]

    # ========================================================================
    # REMOVE TASK
    # ========================================================================

    async def remove_task(
        self,
        task_id: str,
    ) -> bool:
        """
        Remove a task from active storage.
        """

        task = self.get_task(
            task_id
        )

        if task is None:

            return False

        if task.status in {
            TaskStatus.RUNNING,
            TaskStatus.QUEUED,
            TaskStatus.RETRYING,
        }:

            return False

        del self.tasks[
            task_id
        ]

        return True

    # ========================================================================
    # CLEAR COMPLETED
    # ========================================================================

    async def clear_completed(
        self,
    ) -> int:
        """
        Remove completed/cancelled/failed tasks
        from active task storage.
        """

        removable = [
            task_id
            for task_id, task
            in self.tasks.items()
            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            }
        ]

        for task_id in removable:

            self.tasks.pop(
                task_id,
                None,
            )

        return len(
            removable
        )

    # ========================================================================
    # GET HISTORY
    # ========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[RENIXTask]:
        """
        Return task history.
        """

        if limit is None:

            return list(
                self.history
            )

        limit = max(
            0,
            int(limit),
        )

        return list(
            self.history[
                -limit:
            ]
        )

    # ========================================================================
    # FIND BY NAME
    # ========================================================================

    def find_by_name(
        self,
        name: str,
    ) -> list[RENIXTask]:
        """
        Find tasks by name.
        """

        name_lower = (
            name.lower()
        )

        return [
            task
            for task in self.tasks.values()
            if task.name.lower()
            == name_lower
        ]

    # ========================================================================
    # ACTIVE TASKS
    # ========================================================================

    def active_tasks(
        self,
    ) -> list[RENIXTask]:
        """
        Return all currently active tasks.
        """

        active_statuses = {
            TaskStatus.CREATED,
            TaskStatus.QUEUED,
            TaskStatus.RUNNING,
            TaskStatus.PAUSED,
            TaskStatus.RETRYING,
        }

        return [
            task
            for task in self.tasks.values()
            if task.status
            in active_statuses
        ]

    # ========================================================================
    # RUNNING TASKS
    # ========================================================================

    def running_tasks(
        self,
    ) -> list[RENIXTask]:
        """
        Return currently running tasks.
        """

        return [
            task
            for task in self.tasks.values()
            if task.status
            == TaskStatus.RUNNING
        ]

    # ========================================================================
    # QUEUED TASKS
    # ========================================================================

    def queued_tasks(
        self,
    ) -> list[RENIXTask]:
        """
        Return queued tasks.
        """

        return [
            task
            for task in self.tasks.values()
            if task.status
            == TaskStatus.QUEUED
        ]

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return task-manager statistics.
        """

        active = self.active_tasks()

        running = self.running_tasks()

        queued = self.queued_tasks()

        return {
            "running": self.running,
            "max_concurrent_tasks": (
                self.max_concurrent_tasks
            ),
            "active_tasks": len(
                active
            ),
            "running_tasks": len(
                running
            ),
            "queued_tasks": len(
                queued
            ),
            "history_tasks": len(
                self.history
            ),
            "total_created": (
                self.total_created
            ),
            "total_completed": (
                self.total_completed
            ),
            "total_failed": (
                self.total_failed
            ),
            "total_cancelled": (
                self.total_cancelled
            ),
            "total_timeout": (
                self.total_timeout
            ),
            "total_retried": (
                self.total_retried
            ),
        }

    # ========================================================================
    # EXPORT TASKS
    # ========================================================================

    def export_tasks(
        self,
    ) -> list[dict[str, Any]]:
        """
        Export all active tasks.
        """

        return [
            task.to_dict()
            for task in self.tasks.values()
        ]

    # ========================================================================
    # EXPORT HISTORY
    # ========================================================================

    def export_history(
        self,
    ) -> list[dict[str, Any]]:
        """
        Export task history.
        """

        return [
            task.to_dict()
            for task in self.history
        ]


# ============================================================================
# GLOBAL TASK MANAGER
# ============================================================================

task_manager = TaskManager()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_task(
    name: str,
    function: Optional[
        TaskCallable
    ] = None,
    *,
    args: Optional[
        tuple[Any, ...]
    ] = None,
    kwargs: Optional[
        dict[str, Any]
    ] = None,
    priority: TaskPriority = (
        TaskPriority.NORMAL
    ),
    description: str = "",
    dependencies: Optional[
        list[str]
    ] = None,
    max_retries: int = 0,
    timeout: Optional[float] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
    auto_queue: bool = True,
) -> RENIXTask:
    """
    Convenience wrapper for creating a task.
    """

    if not task_manager.running:

        await task_manager.initialize()

    return await task_manager.create_task(
        name=name,
        function=function,
        args=args,
        kwargs=kwargs,
        priority=priority,
        description=description,
        dependencies=dependencies,
        max_retries=max_retries,
        timeout=timeout,
        metadata=metadata,
        auto_queue=auto_queue,
    )


async def run_task(
    name: str,
    function: TaskCallable,
    *,
    args: Optional[
        tuple[Any, ...]
    ] = None,
    kwargs: Optional[
        dict[str, Any]
    ] = None,
    priority: TaskPriority = (
        TaskPriority.NORMAL
    ),
    timeout: Optional[float] = None,
    max_retries: int = 0,
) -> TaskResult:
    """
    Create a task and wait for its completion.
    """

    task = await create_task(
        name=name,
        function=function,
        args=args,
        kwargs=kwargs,
        priority=priority,
        timeout=timeout,
        max_retries=max_retries,
    )

    result = await task_manager.wait_for_task(
        task.task_id
    )

    if result is None:

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error="Task disappeared.",
        )

    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "TaskStatus",
    "TaskPriority",
    "RENIXTask",
    "TaskResult",
    "TaskManager",
    "task_manager",
    "create_task",
    "run_task",
]


