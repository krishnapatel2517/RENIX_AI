"""
RENIX Personal Timer Manager

Handles countdown timers, pause/resume, reset, cancellation,
completion, repeating timers, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import uuid


@dataclass
class Timer:
    """Represents a RENIX countdown timer."""

    title: str
    duration_seconds: int
    timer_id: str | None = None
    started_at: datetime | None = None
    ends_at: datetime | None = None
    remaining_seconds: int | None = None
    status: str = "ready"
    repeating: bool = False
    sound: str = "default"
    volume: int = 100
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        if not self.timer_id:
            self.timer_id = (
                f"TIMER-{uuid.uuid4().hex[:12].upper()}"
            )

        self.title = (
            self.title or ""
        ).strip()

        if not self.title:
            self.title = "RENIX Timer"

        self.duration_seconds = max(
            1,
            int(self.duration_seconds),
        )

        self.volume = max(
            0,
            min(
                100,
                int(self.volume),
            ),
        )

        if self.remaining_seconds is None:
            self.remaining_seconds = (
                self.duration_seconds
            )

    def to_dict(self) -> dict[str, Any]:

        return {
            "timer_id": self.timer_id,
            "title": self.title,
            "duration_seconds": self.duration_seconds,
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "ends_at": (
                self.ends_at.isoformat()
                if self.ends_at
                else None
            ),
            "remaining_seconds": (
                self.remaining_seconds
            ),
            "status": self.status,
            "repeating": self.repeating,
            "sound": self.sound,
            "volume": self.volume,
            "metadata": dict(self.metadata),
        }


class TimerManager:
    """Central countdown timer manager for RENIX."""

    VALID_STATUSES = {
        "ready",
        "running",
        "paused",
        "completed",
        "cancelled",
    }

    def __init__(self) -> None:

        self.timers: dict[
            str,
            Timer,
        ] = {}

    # ============================================================
    # CREATE
    # ============================================================

    def create_timer(
        self,
        title: str,
        duration_seconds: int,
        *,
        repeating: bool = False,
        sound: str = "default",
        volume: int = 100,
        timer_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Timer:

        duration_seconds = max(
            1,
            int(duration_seconds),
        )

        timer = Timer(
            title=title,
            duration_seconds=duration_seconds,
            timer_id=timer_id,
            repeating=repeating,
            sound=sound,
            volume=volume,
            metadata=dict(
                metadata or {}
            ),
        )

        if timer.timer_id in self.timers:
            raise ValueError(
                f"Timer '{timer.timer_id}' already exists."
            )

        self.timers[
            timer.timer_id
        ] = timer

        return timer

    def add_timer(
        self,
        timer: Timer,
    ) -> str:

        if not isinstance(
            timer,
            Timer,
        ):
            raise TypeError(
                "timer must be a Timer."
            )

        if timer.timer_id in self.timers:
            raise ValueError(
                f"Timer '{timer.timer_id}' already exists."
            )

        self.timers[
            timer.timer_id
        ] = timer

        return timer.timer_id

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def get_timer(
        self,
        timer_id: str,
    ) -> Timer | None:

        return self.timers.get(
            timer_id
        )

    def get_all_timers(
        self,
        *,
        include_completed: bool = True,
        include_cancelled: bool = False,
    ) -> list[Timer]:

        timers = list(
            self.timers.values()
        )

        if not include_completed:
            timers = [
                timer
                for timer in timers
                if timer.status != "completed"
            ]

        if not include_cancelled:
            timers = [
                timer
                for timer in timers
                if timer.status != "cancelled"
            ]

        return sorted(
            timers,
            key=lambda timer: (
                timer.status,
                timer.title.casefold(),
            ),
        )

    def get_running_timers(self) -> list[Timer]:

        return [
            timer
            for timer in self.timers.values()
            if timer.status == "running"
        ]

    # ============================================================
    # START
    # ============================================================

    def start(
        self,
        timer_id: str,
        now: datetime | None = None,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        if timer.status == "running":
            return True

        if timer.status in {
            "completed",
            "cancelled",
        }:
            return False

        now = now or datetime.now()

        remaining = max(
            1,
            int(
                timer.remaining_seconds
                or timer.duration_seconds
            ),
        )

        timer.started_at = now
        timer.ends_at = (
            now
            + timedelta(
                seconds=remaining
            )
        )
        timer.remaining_seconds = remaining
        timer.status = "running"

        return True

    # ============================================================
    # PAUSE
    # ============================================================

    def pause(
        self,
        timer_id: str,
        now: datetime | None = None,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        if timer.status != "running":
            return False

        now = now or datetime.now()

        timer.remaining_seconds = (
            self._calculate_remaining(
                timer,
                now,
            )
        )

        timer.ends_at = None
        timer.status = "paused"

        return True

    # ============================================================
    # RESUME
    # ============================================================

    def resume(
        self,
        timer_id: str,
        now: datetime | None = None,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        if timer.status != "paused":
            return False

        now = now or datetime.now()

        remaining = max(
            1,
            int(
                timer.remaining_seconds
                or timer.duration_seconds
            ),
        )

        timer.started_at = now
        timer.ends_at = (
            now
            + timedelta(
                seconds=remaining
            )
        )
        timer.status = "running"

        return True

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        timer_id: str,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        timer.started_at = None
        timer.ends_at = None
        timer.remaining_seconds = (
            timer.duration_seconds
        )
        timer.status = "ready"

        return True

    # ============================================================
    # CANCEL
    # ============================================================

    def cancel(
        self,
        timer_id: str,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        timer.ends_at = None
        timer.status = "cancelled"

        return True

    def restore(
        self,
        timer_id: str,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        timer.started_at = None
        timer.ends_at = None
        timer.remaining_seconds = (
            timer.duration_seconds
        )
        timer.status = "ready"

        return True

    # ============================================================
    # UPDATE
    # ============================================================

    def update_timer(
        self,
        timer_id: str,
        **changes: Any,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        allowed_fields = {
            "title",
            "duration_seconds",
            "repeating",
            "sound",
            "volume",
            "metadata",
        }

        for key, value in changes.items():

            if key not in allowed_fields:
                continue

            if key == "title":

                value = str(
                    value
                ).strip()

                if not value:
                    raise ValueError(
                        "Timer title cannot be empty."
                    )

            if key == "duration_seconds":

                value = max(
                    1,
                    int(value),
                )

                if timer.status in {
                    "ready",
                    "completed",
                    "cancelled",
                }:
                    timer.remaining_seconds = value

            if key == "volume":

                value = max(
                    0,
                    min(
                        100,
                        int(value),
                    ),
                )

            setattr(
                timer,
                key,
                value,
            )

        return True

    # ============================================================
    # UPDATE / PROCESS
    # ============================================================

    def update(
        self,
        now: datetime | None = None,
    ) -> list[Timer]:

        """
        Updates all running timers.

        Returns timers that completed during this update.
        """

        now = now or datetime.now()

        completed: list[Timer] = []

        for timer in list(
            self.timers.values()
        ):

            if timer.status != "running":
                continue

            remaining = self._calculate_remaining(
                timer,
                now,
            )

            timer.remaining_seconds = remaining

            if remaining > 0:
                continue

            if timer.repeating:

                timer.remaining_seconds = (
                    timer.duration_seconds
                )
                timer.started_at = now
                timer.ends_at = (
                    now
                    + timedelta(
                        seconds=timer.duration_seconds
                    )
                )
                timer.status = "running"

            else:

                timer.remaining_seconds = 0
                timer.ends_at = None
                timer.status = "completed"

            completed.append(timer)

        return completed

    def _calculate_remaining(
        self,
        timer: Timer,
        now: datetime,
    ) -> int:

        if timer.ends_at is None:
            return max(
                0,
                int(
                    timer.remaining_seconds
                    or 0
                ),
            )

        remaining = (
            timer.ends_at - now
        ).total_seconds()

        return max(
            0,
            int(remaining + 0.999),
        )

    # ============================================================
    # TIME REMAINING
    # ============================================================

    def remaining_seconds(
        self,
        timer_id: str,
        now: datetime | None = None,
    ) -> int | None:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return None

        now = now or datetime.now()

        if timer.status == "running":
            return self._calculate_remaining(
                timer,
                now,
            )

        return max(
            0,
            int(
                timer.remaining_seconds
                or 0
            ),
        )

    def remaining(
        self,
        timer_id: str,
        now: datetime | None = None,
    ) -> timedelta | None:

        seconds = self.remaining_seconds(
            timer_id,
            now,
        )

        if seconds is None:
            return None

        return timedelta(
            seconds=seconds
        )

    # ============================================================
    # DELETE
    # ============================================================

    def delete_timer(
        self,
        timer_id: str,
    ) -> bool:

        if timer_id not in self.timers:
            return False

        del self.timers[
            timer_id
        ]

        return True

    def clear_completed(self) -> int:

        ids = [
            timer_id
            for timer_id, timer
            in self.timers.items()
            if timer.status == "completed"
        ]

        for timer_id in ids:
            del self.timers[
                timer_id
            ]

        return len(ids)

    # ============================================================
    # SOUND
    # ============================================================

    def set_sound(
        self,
        timer_id: str,
        sound: str,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        timer.sound = (
            sound or "default"
        )

        return True

    def set_volume(
        self,
        timer_id: str,
        volume: int,
    ) -> bool:

        timer = self.get_timer(
            timer_id
        )

        if timer is None:
            return False

        timer.volume = max(
            0,
            min(
                100,
                int(volume),
            ),
        )

        return True

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
    ) -> list[Timer]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        return [
            timer
            for timer in self.timers.values()
            if query in " ".join(
                [
                    timer.title,
                    timer.sound,
                    timer.status,
                ]
            ).casefold()
        ]

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(self) -> dict[str, Any]:

        timers = list(
            self.timers.values()
        )

        return {
            "total": len(timers),
            "ready": sum(
                timer.status == "ready"
                for timer in timers
            ),
            "running": sum(
                timer.status == "running"
                for timer in timers
            ),
            "paused": sum(
                timer.status == "paused"
                for timer in timers
            ),
            "completed": sum(
                timer.status == "completed"
                for timer in timers
            ),
            "cancelled": sum(
                timer.status == "cancelled"
                for timer in timers
            ),
            "repeating": sum(
                timer.repeating
                for timer in timers
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
                "message": "Empty timer command.",
            }

        lowered = command.casefold()

        if lowered in {
            "timers",
            "show timers",
            "show my timers",
        }:

            return {
                "success": True,
                "timers": [
                    timer.to_dict()
                    for timer in self.get_all_timers(
                        include_completed=False,
                        include_cancelled=False,
                    )
                ],
            }

        if lowered in {
            "running timers",
            "active timers",
        }:

            return {
                "success": True,
                "timers": [
                    timer.to_dict()
                    for timer in self.get_running_timers()
                ],
            }

        if lowered.startswith(
            "start timer "
        ):

            timer_id = command[
                len("start timer "):
            ].strip()

            return {
                "success": self.start(
                    timer_id
                ),
                "timer_id": timer_id,
            }

        if lowered.startswith(
            "pause timer "
        ):

            timer_id = command[
                len("pause timer "):
            ].strip()

            return {
                "success": self.pause(
                    timer_id
                ),
                "timer_id": timer_id,
            }

        if lowered.startswith(
            "resume timer "
        ):

            timer_id = command[
                len("resume timer "):
            ].strip()

            return {
                "success": self.resume(
                    timer_id
                ),
                "timer_id": timer_id,
            }

        if lowered.startswith(
            "reset timer "
        ):

            timer_id = command[
                len("reset timer "):
            ].strip()

            return {
                "success": self.reset(
                    timer_id
                ),
                "timer_id": timer_id,
            }

        if lowered.startswith(
            "cancel timer "
        ):

            timer_id = command[
                len("cancel timer "):
            ].strip()

            return {
                "success": self.cancel(
                    timer_id
                ),
                "timer_id": timer_id,
            }

        return {
            "success": False,
            "message": "Unknown timer command.",
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "timers": [
                timer.to_dict()
                for timer in self.timers.values()
            ]
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Timer state must be a dictionary."
            )

        self.timers.clear()

        raw_timers = data.get(
            "timers",
            [],
        )

        if not isinstance(
            raw_timers,
            list,
        ):
            return

        for raw in raw_timers:

            if not isinstance(
                raw,
                dict,
            ):
                continue

            try:

                started_raw = raw.get(
                    "started_at"
                )

                ends_raw = raw.get(
                    "ends_at"
                )

                started_at = (
                    datetime.fromisoformat(
                        started_raw
                    )
                    if started_raw
                    else None
                )

                ends_at = (
                    datetime.fromisoformat(
                        ends_raw
                    )
                    if ends_raw
                    else None
                )

                timer = Timer(
                    title=str(
                        raw.get(
                            "title",
                            "RENIX Timer",
                        )
                    ),
                    duration_seconds=int(
                        raw.get(
                            "duration_seconds",
                            1,
                        )
                    ),
                    timer_id=raw.get(
                        "timer_id"
                    ),
                    started_at=started_at,
                    ends_at=ends_at,
                    remaining_seconds=raw.get(
                        "remaining_seconds"
                    ),
                    status=str(
                        raw.get(
                            "status",
                            "ready",
                        )
                    ),
                    repeating=bool(
                        raw.get(
                            "repeating",
                            False,
                        )
                    ),
                    sound=str(
                        raw.get(
                            "sound",
                            "default",
                        )
                    ),
                    volume=int(
                        raw.get(
                            "volume",
                            100,
                        )
                    ),
                    metadata=dict(
                        raw.get(
                            "metadata",
                            {},
                        )
                    ),
                )

                self.timers[
                    timer.timer_id
                ] = timer

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue


__all__ = [
    "Timer",
    "TimerManager",
]


