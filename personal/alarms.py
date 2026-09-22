"""
RENIX Personal Alarm Manager

Handles alarms, recurring alarms, enabling/disabling,
snoozing, dismissal, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import uuid


@dataclass
class Alarm:
    """Represents a RENIX alarm."""

    title: str
    alarm_time: datetime
    alarm_id: str | None = None
    enabled: bool = True
    recurring: bool = False
    recurrence_minutes: int | None = None
    sound: str = "default"
    volume: int = 100
    label: str = ""
    snoozed_until: datetime | None = None
    dismissed: bool = False
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        if not self.alarm_id:
            self.alarm_id = (
                f"ALARM-{uuid.uuid4().hex[:12].upper()}"
            )

        self.title = (
            self.title or ""
        ).strip()

        if not self.title:
            self.title = "RENIX Alarm"

        self.volume = max(
            0,
            min(
                100,
                int(self.volume),
            ),
        )

        if self.recurrence_minutes is not None:
            self.recurrence_minutes = max(
                1,
                int(self.recurrence_minutes),
            )

    def is_due(
        self,
        now: datetime | None = None,
    ) -> bool:

        now = now or datetime.now()

        if not self.enabled or self.dismissed:
            return False

        target = (
            self.snoozed_until
            or self.alarm_time
        )

        return target <= now

    def to_dict(self) -> dict[str, Any]:

        return {
            "alarm_id": self.alarm_id,
            "title": self.title,
            "alarm_time": self.alarm_time.isoformat(),
            "enabled": self.enabled,
            "recurring": self.recurring,
            "recurrence_minutes": (
                self.recurrence_minutes
            ),
            "sound": self.sound,
            "volume": self.volume,
            "label": self.label,
            "snoozed_until": (
                self.snoozed_until.isoformat()
                if self.snoozed_until
                else None
            ),
            "dismissed": self.dismissed,
            "metadata": dict(self.metadata),
        }


class AlarmManager:
    """Central alarm manager for RENIX."""

    def __init__(self) -> None:

        self.alarms: dict[
            str,
            Alarm,
        ] = {}

    # ============================================================
    # CREATE
    # ============================================================

    def create_alarm(
        self,
        title: str,
        alarm_time: datetime,
        *,
        recurring: bool = False,
        recurrence_minutes: int | None = None,
        sound: str = "default",
        volume: int = 100,
        label: str = "",
        alarm_id: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Alarm:

        if not isinstance(
            alarm_time,
            datetime,
        ):
            raise TypeError(
                "alarm_time must be a datetime."
            )

        alarm = Alarm(
            title=title,
            alarm_time=alarm_time,
            alarm_id=alarm_id,
            enabled=enabled,
            recurring=recurring,
            recurrence_minutes=recurrence_minutes,
            sound=sound,
            volume=volume,
            label=label,
            metadata=dict(
                metadata or {}
            ),
        )

        if alarm.alarm_id in self.alarms:
            raise ValueError(
                f"Alarm '{alarm.alarm_id}' already exists."
            )

        self.alarms[
            alarm.alarm_id
        ] = alarm

        return alarm

    def add_alarm(
        self,
        alarm: Alarm,
    ) -> str:

        if not isinstance(
            alarm,
            Alarm,
        ):
            raise TypeError(
                "alarm must be an Alarm."
            )

        if alarm.alarm_id in self.alarms:
            raise ValueError(
                f"Alarm '{alarm.alarm_id}' already exists."
            )

        self.alarms[
            alarm.alarm_id
        ] = alarm

        return alarm.alarm_id

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def get_alarm(
        self,
        alarm_id: str,
    ) -> Alarm | None:

        return self.alarms.get(
            alarm_id
        )

    def get_all_alarms(
        self,
        *,
        include_disabled: bool = True,
    ) -> list[Alarm]:

        alarms = list(
            self.alarms.values()
        )

        if not include_disabled:
            alarms = [
                alarm
                for alarm in alarms
                if alarm.enabled
            ]

        return sorted(
            alarms,
            key=lambda alarm: alarm.alarm_time,
        )

    def get_active_alarms(self) -> list[Alarm]:

        return self.get_all_alarms(
            include_disabled=False
        )

    def get_due_alarms(
        self,
        now: datetime | None = None,
    ) -> list[Alarm]:

        now = now or datetime.now()

        return sorted(
            [
                alarm
                for alarm in self.alarms.values()
                if alarm.is_due(now)
            ],
            key=lambda alarm: alarm.alarm_time,
        )

    def get_next_alarm(
        self,
        now: datetime | None = None,
    ) -> Alarm | None:

        now = now or datetime.now()

        upcoming = [
            alarm
            for alarm in self.alarms.values()
            if alarm.enabled
            and not alarm.dismissed
            and (
                alarm.snoozed_until
                or alarm.alarm_time
            ) >= now
        ]

        if not upcoming:
            return None

        return min(
            upcoming,
            key=lambda alarm: (
                alarm.snoozed_until
                or alarm.alarm_time
            ),
        )

    # ============================================================
    # ENABLE / DISABLE
    # ============================================================

    def enable(
        self,
        alarm_id: str,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        alarm.enabled = True
        alarm.dismissed = False

        return True

    def disable(
        self,
        alarm_id: str,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        alarm.enabled = False

        return True

    def toggle(
        self,
        alarm_id: str,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        alarm.enabled = not alarm.enabled

        return alarm.enabled

    # ============================================================
    # DISMISS
    # ============================================================

    def dismiss(
        self,
        alarm_id: str,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        if alarm.recurring:
            return self._advance_recurring(
                alarm
            )

        alarm.dismissed = True
        alarm.snoozed_until = None

        return True

    def reset(
        self,
        alarm_id: str,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        alarm.dismissed = False
        alarm.snoozed_until = None
        alarm.enabled = True

        return True

    # ============================================================
    # SNOOZE
    # ============================================================

    def snooze(
        self,
        alarm_id: str,
        minutes: int = 10,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        if not alarm.enabled:
            return False

        minutes = max(
            1,
            int(minutes),
        )

        alarm.snoozed_until = (
            datetime.now()
            + timedelta(
                minutes=minutes
            )
        )

        return True

    def snooze_until(
        self,
        alarm_id: str,
        target: datetime,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        if not isinstance(
            target,
            datetime,
        ):
            raise TypeError(
                "target must be a datetime."
            )

        alarm.snoozed_until = target

        return True

    # ============================================================
    # UPDATE
    # ============================================================

    def update_alarm(
        self,
        alarm_id: str,
        **changes: Any,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        allowed_fields = {
            "title",
            "alarm_time",
            "enabled",
            "recurring",
            "recurrence_minutes",
            "sound",
            "volume",
            "label",
            "metadata",
        }

        for key, value in changes.items():

            if key not in allowed_fields:
                continue

            if key == "alarm_time":
                if not isinstance(
                    value,
                    datetime,
                ):
                    raise TypeError(
                        "alarm_time must be a datetime."
                    )

            if key == "volume":
                value = max(
                    0,
                    min(
                        100,
                        int(value),
                    ),
                )

            if key == "recurrence_minutes":
                if value is not None:
                    value = max(
                        1,
                        int(value),
                    )

            if key == "title":
                value = str(value).strip()

                if not value:
                    raise ValueError(
                        "Alarm title cannot be empty."
                    )

            setattr(
                alarm,
                key,
                value,
            )

        return True

    # ============================================================
    # DELETE
    # ============================================================

    def delete_alarm(
        self,
        alarm_id: str,
    ) -> bool:

        if alarm_id not in self.alarms:
            return False

        del self.alarms[
            alarm_id
        ]

        return True

    def clear_disabled(self) -> int:

        ids = [
            alarm_id
            for alarm_id, alarm
            in self.alarms.items()
            if not alarm.enabled
        ]

        for alarm_id in ids:
            del self.alarms[
                alarm_id
            ]

        return len(ids)

    # ============================================================
    # RECURRING ALARMS
    # ============================================================

    def process_due(
        self,
        now: datetime | None = None,
    ) -> list[Alarm]:

        now = now or datetime.now()

        due = self.get_due_alarms(
            now
        )

        processed = []

        for alarm in due:

            if alarm.recurring:
                self._advance_recurring(
                    alarm
                )
            else:
                alarm.dismissed = True

            processed.append(alarm)

        return processed

    def _advance_recurring(
        self,
        alarm: Alarm,
    ) -> bool:

        if (
            not alarm.recurring
            or not alarm.recurrence_minutes
        ):
            alarm.dismissed = True
            return True

        alarm.alarm_time = (
            alarm.alarm_time
            + timedelta(
                minutes=alarm.recurrence_minutes
            )
        )

        alarm.snoozed_until = None
        alarm.dismissed = False
        alarm.enabled = True

        return True

    # ============================================================
    # SOUND / VOLUME
    # ============================================================

    def set_sound(
        self,
        alarm_id: str,
        sound: str,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        alarm.sound = (
            sound or "default"
        )

        return True

    def set_volume(
        self,
        alarm_id: str,
        volume: int,
    ) -> bool:

        alarm = self.get_alarm(
            alarm_id
        )

        if alarm is None:
            return False

        alarm.volume = max(
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
    ) -> list[Alarm]:

        query = (
            query or ""
        ).strip().casefold()

        if not query:
            return []

        return sorted(
            [
                alarm
                for alarm in self.alarms.values()
                if query
                in " ".join(
                    [
                        alarm.title,
                        alarm.label,
                        alarm.sound,
                    ]
                ).casefold()
            ],
            key=lambda alarm: alarm.alarm_time,
        )

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(self) -> dict[str, Any]:

        alarms = list(
            self.alarms.values()
        )

        return {
            "total": len(alarms),
            "enabled": sum(
                alarm.enabled
                for alarm in alarms
            ),
            "disabled": sum(
                not alarm.enabled
                for alarm in alarms
            ),
            "recurring": sum(
                alarm.recurring
                for alarm in alarms
            ),
            "due": len(
                self.get_due_alarms()
            ),
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "alarms": [
                alarm.to_dict()
                for alarm in self.alarms.values()
            ]
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(data, dict):
            raise TypeError(
                "Alarm state must be a dictionary."
            )

        self.alarms.clear()

        raw_alarms = data.get(
            "alarms",
            [],
        )

        if not isinstance(
            raw_alarms,
            list,
        ):
            return

        for raw in raw_alarms:

            if not isinstance(
                raw,
                dict,
            ):
                continue

            try:

                alarm_time = datetime.fromisoformat(
                    raw["alarm_time"]
                )

                snoozed_raw = raw.get(
                    "snoozed_until"
                )

                snoozed_until = (
                    datetime.fromisoformat(
                        snoozed_raw
                    )
                    if snoozed_raw
                    else None
                )

                alarm = Alarm(
                    title=str(
                        raw.get(
                            "title",
                            "RENIX Alarm",
                        )
                    ),
                    alarm_time=alarm_time,
                    alarm_id=raw.get(
                        "alarm_id"
                    ),
                    enabled=bool(
                        raw.get(
                            "enabled",
                            True,
                        )
                    ),
                    recurring=bool(
                        raw.get(
                            "recurring",
                            False,
                        )
                    ),
                    recurrence_minutes=raw.get(
                        "recurrence_minutes"
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
                    label=str(
                        raw.get(
                            "label",
                            "",
                        )
                    ),
                    snoozed_until=snoozed_until,
                    dismissed=bool(
                        raw.get(
                            "dismissed",
                            False,
                        )
                    ),
                    metadata=dict(
                        raw.get(
                            "metadata",
                            {},
                        )
                    ),
                )

                self.alarms[
                    alarm.alarm_id
                ] = alarm

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue


__all__ = [
    "Alarm",
    "AlarmManager",
]


