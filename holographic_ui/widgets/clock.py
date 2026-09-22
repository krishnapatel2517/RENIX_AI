"""
RENIX Clock Widget
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional


class ClockWidget:

    def __init__(
        self,
        timezone: Optional[str] = None,
        use_24_hour: bool = False,
    ) -> None:

        self.timezone = timezone
        self.use_24_hour = use_24_hour

        self.visible = True

        self.last_update = 0.0

        self.data: dict[str, Any] = {}

        self.update()

    def update(self) -> dict[str, Any]:

        now = datetime.now()

        if self.use_24_hour:
            time_text = now.strftime("%H:%M:%S")
        else:
            time_text = now.strftime("%I:%M:%S %p")

        self.data = {
            "time": time_text,
            "date": now.strftime("%d %B %Y"),
            "day": now.strftime("%A"),
            "timestamp": time.time(),
        }

        self.last_update = time.time()

        return self.data

    def get_data(self) -> dict[str, Any]:

        return dict(self.data)

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def toggle(self) -> bool:

        self.visible = not self.visible

        return self.visible

    def render_data(self) -> dict[str, Any]:

        self.update()

        return {
            "visible": self.visible,
            **self.data,
        }


