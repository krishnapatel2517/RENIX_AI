"""
RENIX Personal Daily Planner

Creates and manages daily plans, time blocks, priorities,
tasks, routines, progress, and daily summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable
import uuid


@dataclass
class PlanItem:
    """Represents one item in a daily plan."""

    title: str
    start_time: time | None = None
    end_time: time | None = None
    priority: str = "medium"
    category: str = ""
    completed: bool = False
    notes: str = ""
    item_id: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        if not self.item_id:
            self.item_id = (
                f"PLANITEM-{uuid.uuid4().hex[:12].upper()}"
            )

        self.title = (
            self.title or ""
        ).strip()

        if not self.title:
            raise ValueError(
                "Plan item title cannot be empty."
            )

        self.priority = (
            self.priority or "medium"
        ).lower()

        if self.priority not in {
            "low",
            "medium",
            "high",
            "urgent",
        }:
            self.priority = "medium"

        if (
            self.start_time
            and self.end_time
            and self.end_time < self.start_time
        ):
            raise ValueError(
                "end_time cannot be before start_time."
            )

    @property
    def duration_minutes(self) -> int:

        if not self.start_time or not self.end_time:
            return 0

        start = datetime.combine(
            date.today(),
            self.start_time,
        )

        end = datetime.combine(
            date.today(),
            self.end_time,
        )

        return max(
            0,
            int(
                (
                    end - start
                ).total_seconds()
                // 60
            ),
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "item_id": self.item_id,
            "title": self.title,
            "start_time": (
                self.start_time.isoformat()
                if self.start_time
                else None
            ),
            "end_time": (
                self.end_time.isoformat()
                if self.end_time
                else None
            ),
            "priority": self.priority,
            "category": self.category,
            "completed": self.completed,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


@dataclass
class DailyPlan:
    """Represents a complete plan for one day."""

    plan_date: date
    items: list[PlanItem] = field(
        default_factory=list
    )
    goals: list[str] = field(
        default_factory=list
    )
    notes: str = ""
    created_at: datetime = field(
        default_factory=datetime.now
    )
    updated_at: datetime = field(
        default_factory=datetime.now
    )

    def add_item(
        self,
        item: PlanItem,
    ) -> None:

        self.items.append(item)
        self.updated_at = datetime.now()

    def remove_item(
        self,
        item_id: str,
    ) -> bool:

        for item in self.items:
            if item.item_id == item_id:
                self.items.remove(item)
                self.updated_at = datetime.now()
                return True

        return False

    def get_item(
        self,
        item_id: str,
    ) -> PlanItem | None:

        for item in self.items:
            if item.item_id == item_id:
                return item

        return None

    def to_dict(self) -> dict[str, Any]:

        return {
            "plan_date": self.plan_date.isoformat(),
            "items": [
                item.to_dict()
                for item in self.items
            ],
            "goals": list(self.goals),
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class DailyPlanner:
    """Central daily-planning system for RENIX."""

    def __init__(self) -> None:

        self.plans: dict[
            str,
            DailyPlan,
        ] = {}

    # ============================================================
    # PLAN MANAGEMENT
    # ============================================================

    def create_plan(
        self,
        plan_date: date | None = None,
        *,
        goals: Iterable[str] | None = None,
        notes: str = "",
    ) -> DailyPlan:

        plan_date = plan_date or date.today()

        if not isinstance(
            plan_date,
            date,
        ):
            raise TypeError(
                "plan_date must be a date."
            )

        key = plan_date.isoformat()

        if key in self.plans:
            return self.plans[key]

        plan = DailyPlan(
            plan_date=plan_date,
            goals=list(goals or []),
            notes=notes,
        )

        self.plans[key] = plan

        return plan

    def get_plan(
        self,
        plan_date: date | None = None,
    ) -> DailyPlan | None:

        plan_date = plan_date or date.today()

        return self.plans.get(
            plan_date.isoformat()
        )

    def get_or_create_plan(
        self,
        plan_date: date | None = None,
    ) -> DailyPlan:

        plan = self.get_plan(
            plan_date
        )

        if plan is not None:
            return plan

        return self.create_plan(
            plan_date
        )

    def delete_plan(
        self,
        plan_date: date,
    ) -> bool:

        key = plan_date.isoformat()

        if key not in self.plans:
            return False

        del self.plans[key]

        return True

    # ============================================================
    # ITEMS
    # ============================================================

    def add_item(
        self,
        title: str,
        *,
        plan_date: date | None = None,
        start_time: time | None = None,
        end_time: time | None = None,
        priority: str = "medium",
        category: str = "",
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PlanItem:

        plan = self.get_or_create_plan(
            plan_date
        )

        item = PlanItem(
            title=title,
            start_time=start_time,
            end_time=end_time,
            priority=priority,
            category=category,
            notes=notes,
            metadata=dict(
                metadata or {}
            ),
        )

        plan.add_item(item)

        return item

    def add_task(
        self,
        title: str,
        *,
        plan_date: date | None = None,
        priority: str = "medium",
        category: str = "",
    ) -> PlanItem:

        return self.add_item(
            title,
            plan_date=plan_date,
            priority=priority,
            category=category,
        )

    def get_item(
        self,
        item_id: str,
    ) -> PlanItem | None:

        for plan in self.plans.values():

            item = plan.get_item(
                item_id
            )

            if item is not None:
                return item

        return None

    def remove_item(
        self,
        item_id: str,
    ) -> bool:

        for plan in self.plans.values():

            if plan.remove_item(
                item_id
            ):
                return True

        return False

    # ============================================================
    # COMPLETION
    # ============================================================

    def complete_item(
        self,
        item_id: str,
    ) -> bool:

        item = self.get_item(
            item_id
        )

        if item is None:
            return False

        item.completed = True

        for plan in self.plans.values():
            if plan.get_item(item_id):
                plan.updated_at = datetime.now()
                break

        return True

    def uncomplete_item(
        self,
        item_id: str,
    ) -> bool:

        item = self.get_item(
            item_id
        )

        if item is None:
            return False

        item.completed = False

        return True

    # ============================================================
    # GOALS
    # ============================================================

    def add_goal(
        self,
        goal: str,
        plan_date: date | None = None,
    ) -> bool:

        goal = (
            goal or ""
        ).strip()

        if not goal:
            return False

        plan = self.get_or_create_plan(
            plan_date
        )

        if goal not in plan.goals:
            plan.goals.append(goal)
            plan.updated_at = datetime.now()

        return True

    def remove_goal(
        self,
        goal: str,
        plan_date: date | None = None,
    ) -> bool:

        plan = self.get_plan(
            plan_date
        )

        if plan is None:
            return False

        if goal not in plan.goals:
            return False

        plan.goals.remove(goal)
        plan.updated_at = datetime.now()

        return True

    # ============================================================
    # PRIORITIES
    # ============================================================

    def get_by_priority(
        self,
        priority: str,
        plan_date: date | None = None,
    ) -> list[PlanItem]:

        plan = self.get_plan(
            plan_date
        )

        if plan is None:
            return []

        priority = (
            priority or ""
        ).lower()

        return [
            item
            for item in plan.items
            if item.priority == priority
        ]

    def get_urgent_items(
        self,
        plan_date: date | None = None,
    ) -> list[PlanItem]:

        plan = self.get_plan(
            plan_date
        )

        if plan is None:
            return []

        return [
            item
            for item in plan.items
            if item.priority == "urgent"
            and not item.completed
        ]

    # ============================================================
    # SCHEDULE
    # ============================================================

    def get_scheduled_items(
        self,
        plan_date: date | None = None,
    ) -> list[PlanItem]:

        plan = self.get_plan(
            plan_date
        )

        if plan is None:
            return []

        return sorted(
            [
                item
                for item in plan.items
                if item.start_time is not None
            ],
            key=lambda item: item.start_time,
        )

    def get_current_item(
        self,
        now: datetime | None = None,
    ) -> PlanItem | None:

        now = now or datetime.now()

        plan = self.get_plan(
            now.date()
        )

        if plan is None:
            return None

        current_time = now.time()

        for item in plan.items:

            if item.completed:
                continue

            if (
                item.start_time
                and item.end_time
                and item.start_time
                <= current_time
                <= item.end_time
            ):
                return item

        return None

    def get_next_item(
        self,
        now: datetime | None = None,
    ) -> PlanItem | None:

        now = now or datetime.now()

        plan = self.get_plan(
            now.date()
        )

        if plan is None:
            return None

        upcoming = [
            item
            for item in plan.items
            if (
                not item.completed
                and item.start_time
                and item.start_time
                > now.time()
            )
        ]

        if not upcoming:
            return None

        return min(
            upcoming,
            key=lambda item: item.start_time,
        )

    # ============================================================
    # CONFLICT DETECTION
    # ============================================================

    def find_conflicts(
        self,
        plan_date: date | None = None,
    ) -> list[tuple[PlanItem, PlanItem]]:

        scheduled = self.get_scheduled_items(
            plan_date
        )

        conflicts = []

        for index, first in enumerate(
            scheduled
        ):

            if not first.end_time:
                continue

            for second in scheduled[
                index + 1:
            ]:

                if not second.start_time:
                    continue

                if (
                    second.start_time
                    < first.end_time
                ):
                    conflicts.append(
                        (
                            first,
                            second,
                        )
                    )

        return conflicts

    # ============================================================
    # PROGRESS
    # ============================================================

    def completion_rate(
        self,
        plan_date: date | None = None,
    ) -> float:

        plan = self.get_plan(
            plan_date
        )

        if plan is None or not plan.items:
            return 0.0

        completed = sum(
            item.completed
            for item in plan.items
        )

        return round(
            completed
            / len(plan.items)
            * 100,
            2,
        )

    def summary(
        self,
        plan_date: date | None = None,
    ) -> dict[str, Any]:

        plan = self.get_plan(
            plan_date
        )

        if plan is None:
            return {
                "date": (
                    plan_date
                    or date.today()
                ).isoformat(),
                "exists": False,
                "items": 0,
                "completed": 0,
                "pending": 0,
                "completion_rate": 0.0,
                "goals": [],
            }

        completed = sum(
            item.completed
            for item in plan.items
        )

        return {
            "date": plan.plan_date.isoformat(),
            "exists": True,
            "items": len(plan.items),
            "completed": completed,
            "pending": (
                len(plan.items)
                - completed
            ),
            "completion_rate": self.completion_rate(
                plan.plan_date
            ),
            "goals": list(plan.goals),
            "urgent": len(
                self.get_urgent_items(
                    plan.plan_date
                )
            ),
            "conflicts": len(
                self.find_conflicts(
                    plan.plan_date
                )
            ),
        }

    # ============================================================
    # DAILY COPY / TEMPLATES
    # ============================================================

    def copy_plan(
        self,
        source_date: date,
        target_date: date,
    ) -> DailyPlan | None:

        source = self.get_plan(
            source_date
        )

        if source is None:
            return None

        target = self.create_plan(
            target_date,
            goals=source.goals,
            notes=source.notes,
        )

        target.items.clear()

        for source_item in source.items:

            copied = PlanItem(
                title=source_item.title,
                start_time=source_item.start_time,
                end_time=source_item.end_time,
                priority=source_item.priority,
                category=source_item.category,
                completed=False,
                notes=source_item.notes,
                metadata=dict(
                    source_item.metadata
                ),
            )

            target.items.append(
                copied
            )

        target.updated_at = datetime.now()

        return target

    def clear_completed(
        self,
        plan_date: date | None = None,
    ) -> int:

        plan = self.get_plan(
            plan_date
        )

        if plan is None:
            return 0

        original = len(
            plan.items
        )

        plan.items = [
            item
            for item in plan.items
            if not item.completed
        ]

        removed = (
            original
            - len(plan.items)
        )

        if removed:
            plan.updated_at = datetime.now()

        return removed

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        plan_date: date | None = None,
    ) -> list[PlanItem]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        plan = self.get_plan(
            plan_date
        )

        if plan is None:
            return []

        return [
            item
            for item in plan.items
            if query
            in " ".join(
                [
                    item.title,
                    item.category,
                    item.notes,
                    item.priority,
                ]
            ).casefold()
        ]

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "plans": [
                plan.to_dict()
                for plan in self.plans.values()
            ]
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "Planner state must be a dictionary."
            )

        self.plans.clear()

        raw_plans = data.get(
            "plans",
            [],
        )

        if not isinstance(
            raw_plans,
            list,
        ):
            return

        for raw_plan in raw_plans:

            if not isinstance(
                raw_plan,
                dict,
            ):
                continue

            try:

                plan_date = date.fromisoformat(
                    raw_plan["plan_date"]
                )

                created_raw = raw_plan.get(
                    "created_at"
                )

                updated_raw = raw_plan.get(
                    "updated_at"
                )

                created_at = (
                    datetime.fromisoformat(
                        created_raw
                    )
                    if created_raw
                    else datetime.now()
                )

                updated_at = (
                    datetime.fromisoformat(
                        updated_raw
                    )
                    if updated_raw
                    else created_at
                )

                plan = DailyPlan(
                    plan_date=plan_date,
                    goals=list(
                        raw_plan.get(
                            "goals",
                            [],
                        )
                    ),
                    notes=str(
                        raw_plan.get(
                            "notes",
                            "",
                        )
                    ),
                    created_at=created_at,
                    updated_at=updated_at,
                )

                for raw_item in raw_plan.get(
                    "items",
                    [],
                ):

                    if not isinstance(
                        raw_item,
                        dict,
                    ):
                        continue

                    start_raw = raw_item.get(
                        "start_time"
                    )

                    end_raw = raw_item.get(
                        "end_time"
                    )

                    start_time = (
                        time.fromisoformat(
                            start_raw
                        )
                        if start_raw
                        else None
                    )

                    end_time = (
                        time.fromisoformat(
                            end_raw
                        )
                        if end_raw
                        else None
                    )

                    item = PlanItem(
                        title=str(
                            raw_item.get(
                                "title",
                                "Untitled",
                            )
                        ),
                        start_time=start_time,
                        end_time=end_time,
                        priority=str(
                            raw_item.get(
                                "priority",
                                "medium",
                            )
                        ),
                        category=str(
                            raw_item.get(
                                "category",
                                "",
                            )
                        ),
                        completed=bool(
                            raw_item.get(
                                "completed",
                                False,
                            )
                        ),
                        notes=str(
                            raw_item.get(
                                "notes",
                                "",
                            )
                        ),
                        item_id=raw_item.get(
                            "item_id"
                        ),
                        metadata=dict(
                            raw_item.get(
                                "metadata",
                                {},
                            )
                        ),
                    )

                    plan.items.append(
                        item
                    )

                self.plans[
                    plan_date.isoformat()
                ] = plan

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue


__all__ = [
    "PlanItem",
    "DailyPlan",
    "DailyPlanner",
]


