"""
RENIX Personal Task Manager

Handles personal tasks, priorities, deadlines, subtasks,
status tracking, tags, searching, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable
import uuid


@dataclass
class Task:
    """Represents a personal RENIX task."""

    title: str
    description: str = ""
    priority: str = "normal"
    due_date: datetime | None = None
    task_id: str | None = None
    status: str = "pending"
    category: str = ""
    tags: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = f"TASK-{uuid.uuid4().hex[:12].upper()}"

        self.priority = (
            self.priority or "normal"
        ).lower()

        self.status = (
            self.status or "pending"
        ).lower()

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "due_date": (
                self.due_date.isoformat()
                if self.due_date
                else None
            ),
            "status": self.status,
            "category": self.category,
            "tags": list(self.tags),
            "subtasks": list(self.subtasks),
            "metadata": dict(self.metadata),
        }


class TaskManager:
    """Manages RENIX personal tasks."""

    VALID_PRIORITIES = {
        "low",
        "normal",
        "high",
        "urgent",
    }

    VALID_STATUSES = {
        "pending",
        "in_progress",
        "completed",
        "cancelled",
        "paused",
    }

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    # ============================================================
    # CREATE
    # ============================================================

    def create_task(
        self,
        title: str,
        *,
        description: str = "",
        priority: str = "normal",
        due_date: datetime | None = None,
        category: str = "",
        tags: Iterable[str] | None = None,
        subtasks: Iterable[str] | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:

        title = (title or "").strip()

        if not title:
            raise ValueError(
                "Task title cannot be empty."
            )

        priority = (
            priority or "normal"
        ).lower()

        if priority not in self.VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority: {priority}"
            )

        task = Task(
            title=title,
            description=description,
            priority=priority,
            due_date=due_date,
            category=category,
            tags=list(tags or []),
            subtasks=list(subtasks or []),
            task_id=task_id,
            metadata=dict(metadata or {}),
        )

        if task.task_id in self.tasks:
            raise ValueError(
                f"Task '{task.task_id}' already exists."
            )

        self.tasks[task.task_id] = task

        return task

    def add_task(self, task: Task) -> str:

        if not isinstance(task, Task):
            raise TypeError(
                "task must be a Task."
            )

        if task.task_id in self.tasks:
            raise ValueError(
                f"Task '{task.task_id}' already exists."
            )

        self.tasks[task.task_id] = task

        return task.task_id

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def get_task(
        self,
        task_id: str,
    ) -> Task | None:

        return self.tasks.get(task_id)

    def get_all_tasks(
        self,
        *,
        include_completed: bool = True,
        include_cancelled: bool = False,
    ) -> list[Task]:

        tasks = list(self.tasks.values())

        if not include_completed:
            tasks = [
                task
                for task in tasks
                if task.status != "completed"
            ]

        if not include_cancelled:
            tasks = [
                task
                for task in tasks
                if task.status != "cancelled"
            ]

        return sorted(
            tasks,
            key=self._sort_key,
        )

    def get_pending_tasks(self) -> list[Task]:

        return self.get_all_tasks(
            include_completed=False,
            include_cancelled=False,
        )

    def get_overdue_tasks(
        self,
        now: datetime | None = None,
    ) -> list[Task]:

        now = now or datetime.now()

        return sorted(
            [
                task
                for task in self.tasks.values()
                if task.due_date
                and task.due_date < now
                and task.status
                not in {
                    "completed",
                    "cancelled",
                }
            ],
            key=lambda task: task.due_date,
        )

    def get_due_today(
        self,
        now: datetime | None = None,
    ) -> list[Task]:

        now = now or datetime.now()

        return [
            task
            for task in self.tasks.values()
            if task.due_date
            and task.due_date.date() == now.date()
            and task.status
            not in {
                "completed",
                "cancelled",
            }
        ]

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
    ) -> list[Task]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        results = []

        for task in self.tasks.values():

            searchable = " ".join(
                [
                    task.title,
                    task.description,
                    task.priority,
                    task.category,
                    task.status,
                    " ".join(task.tags),
                    " ".join(task.subtasks),
                ]
            ).casefold()

            if query in searchable:
                results.append(task)

        return sorted(
            results,
            key=self._sort_key,
        )

    def by_category(
        self,
        category: str,
    ) -> list[Task]:

        category = (
            category or ""
        ).strip().casefold()

        return sorted(
            [
                task
                for task in self.tasks.values()
                if task.category.casefold()
                == category
            ],
            key=self._sort_key,
        )

    def by_priority(
        self,
        priority: str,
    ) -> list[Task]:

        priority = (
            priority or ""
        ).strip().casefold()

        return sorted(
            [
                task
                for task in self.tasks.values()
                if task.priority == priority
            ],
            key=self._sort_key,
        )

    # ============================================================
    # UPDATE
    # ============================================================

    def update_task(
        self,
        task_id: str,
        **changes: Any,
    ) -> bool:

        task = self.get_task(task_id)

        if task is None:
            return False

        allowed_fields = {
            "title",
            "description",
            "priority",
            "due_date",
            "status",
            "category",
            "tags",
            "subtasks",
            "metadata",
        }

        for key, value in changes.items():

            if key not in allowed_fields:
                continue

            if key == "title":
                value = str(value).strip()

                if not value:
                    raise ValueError(
                        "Task title cannot be empty."
                    )

            if key == "priority":

                value = str(value).lower()

                if value not in self.VALID_PRIORITIES:
                    raise ValueError(
                        f"Invalid priority: {value}"
                    )

            if key == "status":

                value = str(value).lower()

                if value not in self.VALID_STATUSES:
                    raise ValueError(
                        f"Invalid status: {value}"
                    )

            if key == "due_date":

                if (
                    value is not None
                    and not isinstance(value, datetime)
                ):
                    raise TypeError(
                        "due_date must be a datetime or None."
                    )

            setattr(
                task,
                key,
                value,
            )

        return True

    # ============================================================
    # STATUS
    # ============================================================

    def start_task(
        self,
        task_id: str,
    ) -> bool:

        return self.update_task(
            task_id,
            status="in_progress",
        )

    def pause_task(
        self,
        task_id: str,
    ) -> bool:

        return self.update_task(
            task_id,
            status="paused",
        )

    def complete_task(
        self,
        task_id: str,
    ) -> bool:

        return self.update_task(
            task_id,
            status="completed",
        )

    def cancel_task(
        self,
        task_id: str,
    ) -> bool:

        return self.update_task(
            task_id,
            status="cancelled",
        )

    def reopen_task(
        self,
        task_id: str,
    ) -> bool:

        return self.update_task(
            task_id,
            status="pending",
        )

    # ============================================================
    # SUBTASKS
    # ============================================================

    def add_subtask(
        self,
        task_id: str,
        subtask: str,
    ) -> bool:

        task = self.get_task(task_id)

        if task is None:
            return False

        subtask = (
            subtask or ""
        ).strip()

        if not subtask:
            return False

        task.subtasks.append(subtask)

        return True

    def remove_subtask(
        self,
        task_id: str,
        subtask: str,
    ) -> bool:

        task = self.get_task(task_id)

        if task is None:
            return False

        if subtask not in task.subtasks:
            return False

        task.subtasks.remove(subtask)

        return True

    # ============================================================
    # DELETE
    # ============================================================

    def delete_task(
        self,
        task_id: str,
    ) -> bool:

        if task_id not in self.tasks:
            return False

        del self.tasks[task_id]

        return True

    def clear_completed(self) -> int:

        completed_ids = [
            task_id
            for task_id, task in self.tasks.items()
            if task.status == "completed"
        ]

        for task_id in completed_ids:
            del self.tasks[task_id]

        return len(completed_ids)

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(self) -> dict[str, Any]:

        tasks = list(self.tasks.values())

        total = len(tasks)

        completed = sum(
            task.status == "completed"
            for task in tasks
        )

        pending = sum(
            task.status == "pending"
            for task in tasks
        )

        in_progress = sum(
            task.status == "in_progress"
            for task in tasks
        )

        cancelled = sum(
            task.status == "cancelled"
            for task in tasks
        )

        overdue = len(
            self.get_overdue_tasks()
        )

        completion_rate = (
            (completed / total) * 100
            if total
            else 0.0
        )

        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "in_progress": in_progress,
            "cancelled": cancelled,
            "overdue": overdue,
            "completion_rate": round(
                completion_rate,
                2,
            ),
        }

    # ============================================================
    # SIMPLE COMMAND HANDLER
    # ============================================================

    def handle_command(
        self,
        command: str,
    ) -> dict[str, Any]:

        command = (
            command or ""
        ).strip()

        if not command:
            return {
                "success": False,
                "message": "Empty task command.",
            }

        lowered = command.casefold()

        if lowered in {
            "tasks",
            "show tasks",
            "show my tasks",
        }:

            return {
                "success": True,
                "tasks": [
                    task.to_dict()
                    for task in self.get_pending_tasks()
                ],
            }

        if (
            "overdue"
            in lowered
        ):

            return {
                "success": True,
                "tasks": [
                    task.to_dict()
                    for task in self.get_overdue_tasks()
                ],
            }

        if lowered == "completed tasks":

            return {
                "success": True,
                "tasks": [
                    task.to_dict()
                    for task in self.get_all_tasks()
                    if task.status == "completed"
                ],
            }

        if lowered.startswith("find task "):

            query = command[
                len("find task "):
            ].strip()

            return {
                "success": True,
                "tasks": [
                    task.to_dict()
                    for task in self.search(query)
                ],
            }

        if lowered.startswith("complete "):

            task_id = command[
                len("complete "):
            ].strip()

            return {
                "success": self.complete_task(
                    task_id
                ),
                "task_id": task_id,
            }

        if lowered.startswith("start "):

            task_id = command[
                len("start "):
            ].strip()

            return {
                "success": self.start_task(
                    task_id
                ),
                "task_id": task_id,
            }

        if lowered.startswith("delete "):

            task_id = command[
                len("delete "):
            ].strip()

            return {
                "success": self.delete_task(
                    task_id
                ),
                "task_id": task_id,
            }

        return {
            "success": False,
            "message": "Unknown task command.",
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(self) -> dict[str, Any]:

        return {
            "tasks": [
                task.to_dict()
                for task in self.tasks.values()
            ]
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Task state must be a dictionary."
            )

        self.tasks.clear()

        raw_tasks = data.get(
            "tasks",
            [],
        )

        if not isinstance(raw_tasks, list):
            return

        for raw in raw_tasks:

            if not isinstance(raw, dict):
                continue

            try:

                due_raw = raw.get(
                    "due_date"
                )

                due_date = (
                    datetime.fromisoformat(
                        due_raw
                    )
                    if due_raw
                    else None
                )

                task = Task(
                    title=str(
                        raw.get(
                            "title",
                            "Untitled Task",
                        )
                    ),
                    description=str(
                        raw.get(
                            "description",
                            "",
                        )
                    ),
                    priority=str(
                        raw.get(
                            "priority",
                            "normal",
                        )
                    ),
                    due_date=due_date,
                    task_id=raw.get(
                        "task_id"
                    ),
                    status=str(
                        raw.get(
                            "status",
                            "pending",
                        )
                    ),
                    category=str(
                        raw.get(
                            "category",
                            "",
                        )
                    ),
                    tags=list(
                        raw.get(
                            "tags",
                            [],
                        )
                    ),
                    subtasks=list(
                        raw.get(
                            "subtasks",
                            [],
                        )
                    ),
                    metadata=dict(
                        raw.get(
                            "metadata",
                            {},
                        )
                    ),
                )

                self.tasks[
                    task.task_id
                ] = task

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    @staticmethod
    def _priority_value(
        priority: str,
    ) -> int:

        return {
            "low": 1,
            "normal": 2,
            "high": 3,
            "urgent": 4,
        }.get(
            priority,
            2,
        )

    @classmethod
    def _sort_key(
        cls,
        task: Task,
    ) -> tuple:

        due_timestamp = (
            task.due_date.timestamp()
            if task.due_date
            else float("inf")
        )

        return (
            -cls._priority_value(
                task.priority
            ),
            due_timestamp,
            task.title.casefold(),
        )


__all__ = [
    "Task",
    "TaskManager",
]


