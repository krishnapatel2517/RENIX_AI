"""
RENIX Personal Habit Manager

Tracks habits, daily completions, streaks, goals, reminders,
progress, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable
import uuid


@dataclass
class Habit:
    """Represents a recurring personal habit."""

    name: str
    description: str = ""
    frequency: str = "daily"
    target_count: int = 1
    habit_id: str | None = None
    category: str = ""
    tags: list[str] = field(default_factory=list)
    reminder_time: str | None = None
    active: bool = True
    start_date: date = field(
        default_factory=date.today
    )
    completion_dates: list[str] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        if not self.habit_id:
            self.habit_id = (
                f"HABIT-{uuid.uuid4().hex[:12].upper()}"
            )

        self.name = (
            self.name or ""
        ).strip()

        if not self.name:
            raise ValueError(
                "Habit name cannot be empty."
            )

        self.frequency = (
            self.frequency or "daily"
        ).lower()

        self.target_count = max(
            1,
            int(self.target_count),
        )

        if not isinstance(
            self.start_date,
            date,
        ):
            self.start_date = date.today()

    def is_completed_on(
        self,
        day: date,
    ) -> bool:

        return day.isoformat() in (
            self.completion_dates
        )

    def mark_complete(
        self,
        day: date | None = None,
    ) -> bool:

        day = day or date.today()
        value = day.isoformat()

        if value in self.completion_dates:
            return False

        self.completion_dates.append(value)
        self.completion_dates.sort()

        return True

    def mark_incomplete(
        self,
        day: date | None = None,
    ) -> bool:

        day = day or date.today()
        value = day.isoformat()

        if value not in self.completion_dates:
            return False

        self.completion_dates.remove(value)

        return True

    def to_dict(self) -> dict[str, Any]:

        return {
            "habit_id": self.habit_id,
            "name": self.name,
            "description": self.description,
            "frequency": self.frequency,
            "target_count": self.target_count,
            "category": self.category,
            "tags": list(self.tags),
            "reminder_time": self.reminder_time,
            "active": self.active,
            "start_date": self.start_date.isoformat(),
            "completion_dates": list(
                self.completion_dates
            ),
            "metadata": dict(self.metadata),
        }


class HabitManager:
    """Central habit-tracking manager for RENIX."""

    VALID_FREQUENCIES = {
        "daily",
        "weekly",
        "monthly",
    }

    def __init__(self) -> None:

        self.habits: dict[
            str,
            Habit,
        ] = {}

    # ============================================================
    # CREATE
    # ============================================================

    def create_habit(
        self,
        name: str,
        *,
        description: str = "",
        frequency: str = "daily",
        target_count: int = 1,
        category: str = "",
        tags: Iterable[str] | None = None,
        reminder_time: str | None = None,
        start_date: date | None = None,
        habit_id: str | None = None,
        active: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Habit:

        frequency = (
            frequency or "daily"
        ).lower()

        if frequency not in self.VALID_FREQUENCIES:
            raise ValueError(
                f"Invalid habit frequency: {frequency}"
            )

        habit = Habit(
            name=name,
            description=description,
            frequency=frequency,
            target_count=target_count,
            habit_id=habit_id,
            category=category,
            tags=list(tags or []),
            reminder_time=reminder_time,
            active=active,
            start_date=start_date or date.today(),
            metadata=dict(
                metadata or {}
            ),
        )

        if habit.habit_id in self.habits:
            raise ValueError(
                f"Habit '{habit.habit_id}' already exists."
            )

        self.habits[
            habit.habit_id
        ] = habit

        return habit

    def add_habit(
        self,
        habit: Habit,
    ) -> str:

        if not isinstance(
            habit,
            Habit,
        ):
            raise TypeError(
                "habit must be a Habit."
            )

        if habit.habit_id in self.habits:
            raise ValueError(
                f"Habit '{habit.habit_id}' already exists."
            )

        self.habits[
            habit.habit_id
        ] = habit

        return habit.habit_id

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def get_habit(
        self,
        habit_id: str,
    ) -> Habit | None:

        return self.habits.get(
            habit_id
        )

    def get_all_habits(
        self,
        *,
        include_inactive: bool = False,
    ) -> list[Habit]:

        habits = list(
            self.habits.values()
        )

        if not include_inactive:
            habits = [
                habit
                for habit in habits
                if habit.active
            ]

        return sorted(
            habits,
            key=lambda habit: habit.name.casefold(),
        )

    def get_today_habits(
        self,
        day: date | None = None,
    ) -> list[Habit]:

        day = day or date.today()

        return [
            habit
            for habit in self.get_all_habits()
            if self.is_due_on(
                habit,
                day,
            )
        ]

    # ============================================================
    # COMPLETION
    # ============================================================

    def complete_habit(
        self,
        habit_id: str,
        day: date | None = None,
    ) -> bool:

        habit = self.get_habit(
            habit_id
        )

        if habit is None or not habit.active:
            return False

        return habit.mark_complete(day)

    def uncomplete_habit(
        self,
        habit_id: str,
        day: date | None = None,
    ) -> bool:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return False

        return habit.mark_incomplete(day)

    def is_completed(
        self,
        habit_id: str,
        day: date | None = None,
    ) -> bool:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return False

        return habit.is_completed_on(
            day or date.today()
        )

    # ============================================================
    # FREQUENCY
    # ============================================================

    def is_due_on(
        self,
        habit: Habit,
        day: date,
    ) -> bool:

        if not habit.active:
            return False

        if day < habit.start_date:
            return False

        if habit.frequency == "daily":
            return True

        if habit.frequency == "weekly":
            days_since_start = (
                day - habit.start_date
            ).days

            return (
                days_since_start % 7 == 0
            )

        if habit.frequency == "monthly":
            return (
                day.day
                == habit.start_date.day
            )

        return False

    # ============================================================
    # STREAKS
    # ============================================================

    def current_streak(
        self,
        habit_id: str,
        day: date | None = None,
    ) -> int:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return 0

        day = day or date.today()

        if not habit.is_completed_on(day):

            # A streak can still be active if
            # today's occurrence has not happened yet.
            previous_day = day - timedelta(days=1)

            if not habit.is_completed_on(
                previous_day
            ):
                return 0

            day = previous_day

        streak = 0

        while habit.is_completed_on(day):

            streak += 1
            day -= timedelta(days=1)

        return streak

    def longest_streak(
        self,
        habit_id: str,
    ) -> int:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return 0

        if not habit.completion_dates:
            return 0

        dates = sorted(
            {
                date.fromisoformat(value)
                for value in habit.completion_dates
            }
        )

        longest = 1
        current = 1

        for previous, current_day in zip(
            dates,
            dates[1:],
        ):

            if (
                current_day
                == previous + timedelta(days=1)
            ):
                current += 1
                longest = max(
                    longest,
                    current,
                )
            else:
                current = 1

        return longest

    # ============================================================
    # PROGRESS
    # ============================================================

    def completion_count(
        self,
        habit_id: str,
    ) -> int:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return 0

        return len(
            habit.completion_dates
        )

    def completion_rate(
        self,
        habit_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> float:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return 0.0

        start = (
            start
            or habit.start_date
        )

        end = (
            end
            or date.today()
        )

        if end < start:
            return 0.0

        total_days = (
            end - start
        ).days + 1

        completed = sum(
            start
            <= date.fromisoformat(value)
            <= end
            for value in habit.completion_dates
        )

        return round(
            (
                completed
                / total_days
            )
            * 100,
            2,
        )

    def weekly_progress(
        self,
        habit_id: str,
        day: date | None = None,
    ) -> dict[str, Any]:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return {}

        day = day or date.today()

        start = (
            day
            - timedelta(
                days=day.weekday()
            )
        )

        result = {}

        for offset in range(7):

            current_day = (
                start
                + timedelta(
                    days=offset
                )
            )

            result[
                current_day.isoformat()
            ] = habit.is_completed_on(
                current_day
            )

        completed = sum(
            result.values()
        )

        return {
            "habit_id": habit_id,
            "week_start": start.isoformat(),
            "completed_days": completed,
            "total_days": 7,
            "completion_rate": round(
                completed / 7 * 100,
                2,
            ),
            "days": result,
        }

    # ============================================================
    # UPDATE
    # ============================================================

    def update_habit(
        self,
        habit_id: str,
        **changes: Any,
    ) -> bool:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return False

        allowed_fields = {
            "name",
            "description",
            "frequency",
            "target_count",
            "category",
            "tags",
            "reminder_time",
            "active",
            "start_date",
            "metadata",
        }

        for key, value in changes.items():

            if key not in allowed_fields:
                continue

            if key == "name":

                value = str(
                    value
                ).strip()

                if not value:
                    raise ValueError(
                        "Habit name cannot be empty."
                    )

            if key == "frequency":

                value = str(
                    value
                ).lower()

                if value not in self.VALID_FREQUENCIES:
                    raise ValueError(
                        f"Invalid habit frequency: {value}"
                    )

            if key == "target_count":

                value = max(
                    1,
                    int(value),
                )

            if key == "start_date":

                if not isinstance(
                    value,
                    date,
                ):
                    raise TypeError(
                        "start_date must be a date."
                    )

            setattr(
                habit,
                key,
                value,
            )

        return True

    # ============================================================
    # ACTIVATE / DEACTIVATE
    # ============================================================

    def activate(
        self,
        habit_id: str,
    ) -> bool:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return False

        habit.active = True

        return True

    def deactivate(
        self,
        habit_id: str,
    ) -> bool:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return False

        habit.active = False

        return True

    # ============================================================
    # TAGS
    # ============================================================

    def add_tag(
        self,
        habit_id: str,
        tag: str,
    ) -> bool:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return False

        tag = (
            tag or ""
        ).strip()

        if not tag:
            return False

        existing = {
            item.casefold()
            for item in habit.tags
        }

        if tag.casefold() not in existing:
            habit.tags.append(tag)

        return True

    def remove_tag(
        self,
        habit_id: str,
        tag: str,
    ) -> bool:

        habit = self.get_habit(
            habit_id
        )

        if habit is None:
            return False

        target = (
            tag or ""
        ).strip().casefold()

        for existing in list(
            habit.tags
        ):
            if existing.casefold() == target:
                habit.tags.remove(existing)
                return True

        return False

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
    ) -> list[Habit]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        return sorted(
            [
                habit
                for habit in self.habits.values()
                if query
                in " ".join(
                    [
                        habit.name,
                        habit.description,
                        habit.category,
                        habit.frequency,
                        " ".join(habit.tags),
                    ]
                ).casefold()
            ],
            key=lambda habit: habit.name.casefold(),
        )

    def by_category(
        self,
        category: str,
    ) -> list[Habit]:

        category = (
            category or ""
        ).strip().casefold()

        return [
            habit
            for habit in self.get_all_habits()
            if habit.category.casefold()
            == category
        ]

    # ============================================================
    # DELETE
    # ============================================================

    def delete_habit(
        self,
        habit_id: str,
    ) -> bool:

        if habit_id not in self.habits:
            return False

        del self.habits[
            habit_id
        ]

        return True

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(self) -> dict[str, Any]:

        habits = list(
            self.habits.values()
        )

        if not habits:
            return {
                "total": 0,
                "active": 0,
                "inactive": 0,
                "completed_today": 0,
                "average_completion_rate": 0.0,
                "longest_streak": 0,
            }

        active = [
            habit
            for habit in habits
            if habit.active
        ]

        rates = [
            self.completion_rate(
                habit.habit_id
            )
            for habit in active
        ]

        return {
            "total": len(habits),
            "active": len(active),
            "inactive": (
                len(habits) - len(active)
            ),
            "completed_today": sum(
                habit.is_completed_on(
                    date.today()
                )
                for habit in active
            ),
            "average_completion_rate": round(
                sum(rates) / len(rates),
                2,
            )
            if rates
            else 0.0,
            "longest_streak": max(
                (
                    self.longest_streak(
                        habit.habit_id
                    )
                    for habit in habits
                ),
                default=0,
            ),
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "habits": [
                habit.to_dict()
                for habit in self.habits.values()
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
                "Habit state must be a dictionary."
            )

        self.habits.clear()

        raw_habits = data.get(
            "habits",
            [],
        )

        if not isinstance(
            raw_habits,
            list,
        ):
            return

        for raw in raw_habits:

            if not isinstance(
                raw,
                dict,
            ):
                continue

            try:

                start_raw = raw.get(
                    "start_date"
                )

                start_date = (
                    date.fromisoformat(
                        start_raw
                    )
                    if start_raw
                    else date.today()
                )

                habit = Habit(
                    name=str(
                        raw.get(
                            "name",
                            "Unnamed Habit",
                        )
                    ),
                    description=str(
                        raw.get(
                            "description",
                            "",
                        )
                    ),
                    frequency=str(
                        raw.get(
                            "frequency",
                            "daily",
                        )
                    ),
                    target_count=int(
                        raw.get(
                            "target_count",
                            1,
                        )
                    ),
                    habit_id=raw.get(
                        "habit_id"
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
                    reminder_time=raw.get(
                        "reminder_time"
                    ),
                    active=bool(
                        raw.get(
                            "active",
                            True,
                        )
                    ),
                    start_date=start_date,
                    completion_dates=list(
                        raw.get(
                            "completion_dates",
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

                self.habits[
                    habit.habit_id
                ] = habit

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue


__all__ = [
    "Habit",
    "HabitManager",
]


