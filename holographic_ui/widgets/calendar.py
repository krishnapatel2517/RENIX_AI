"""
RENIX Calendar Widget
"""

from __future__ import annotations

import calendar as calendar_module
from datetime import datetime
from typing import Any


class CalendarWidget:

    def __init__(self) -> None:

        self.visible = True

        self.selected_date = datetime.now()

        self.events: list[dict[str, Any]] = []

    def set_date(
        self,
        date: datetime,
    ) -> None:

        self.selected_date = date

    def add_event(
        self,
        title: str,
        date: str,
        time_text: str = "",
    ) -> dict[str, Any]:

        event = {
            "title": title,
            "date": date,
            "time": time_text,
        }

        self.events.append(event)

        return event

    def remove_event(
        self,
        title: str,
    ) -> bool:

        for event in self.events:

            if event["title"] == title:

                self.events.remove(event)

                return True

        return False

    def get_month(
        self,
        year: int | None = None,
        month: int | None = None,
    ) -> list[list[int]]:

        now = self.selected_date

        year = year or now.year
        month = month or now.month

        return calendar_module.monthcalendar(
            year,
            month,
        )

    def get_events(self) -> list[dict[str, Any]]:

        return list(self.events)

    def get_data(self) -> dict[str, Any]:

        date = self.selected_date

        return {
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "month_name": date.strftime(
                "%B"
            ),
            "events": self.get_events(),
        }


