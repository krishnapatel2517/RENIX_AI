"""
RENIX Tasks Widget
"""

from __future__ import annotations

import uuid
from typing import Any


class TasksWidget:

    def __init__(self) -> None:

        self.visible = True

        self.tasks: list[dict[str, Any]] = []

    def add_task(
        self,
        title: str,
        priority: str = "normal",
    ) -> dict[str, Any]:

        task = {
            "id": str(uuid.uuid4()),
            "title": title,
            "priority": priority,
            "completed": False,
        }

        self.tasks.append(task)

        return task

    def complete_task(
        self,
        task_id: str,
    ) -> bool:

        for task in self.tasks:

            if task["id"] == task_id:

                task["completed"] = True

                return True

        return False

    def delete_task(
        self,
        task_id: str,
    ) -> bool:

        for task in self.tasks:

            if task["id"] == task_id:

                self.tasks.remove(task)

                return True

        return False

    def get_pending(
        self,
    ) -> list[dict[str, Any]]:

        return [
            task
            for task in self.tasks
            if not task["completed"]
        ]

    def get_completed(
        self,
    ) -> list[dict[str, Any]]:

        return [
            task
            for task in self.tasks
            if task["completed"]
        ]

    def get_data(self) -> dict[str, Any]:

        return {
            "total": len(self.tasks),
            "pending": len(
                self.get_pending()
            ),
            "completed": len(
                self.get_completed()
            ),
            "tasks": list(self.tasks),
        }


