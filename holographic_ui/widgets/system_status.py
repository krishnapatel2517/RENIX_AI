"""
RENIX System Status Widget
"""

from __future__ import annotations

import platform
import time
from typing import Any


class SystemStatusWidget:

    def __init__(self) -> None:

        self.visible = True

        self.status = "ONLINE"

        self.data: dict[str, Any] = {}

        self.update()

    def update(
        self,
        **overrides: Any,
    ) -> dict[str, Any]:

        self.data = {
            "status": overrides.get(
                "status",
                self.status,
            ),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "timestamp": time.time(),
        }

        return self.data

    def set_status(
        self,
        status: str,
    ) -> None:

        self.status = str(status).upper()

        self.update()

    def online(self) -> None:
        self.set_status("ONLINE")

    def offline(self) -> None:
        self.set_status("OFFLINE")

    def warning(self) -> None:
        self.set_status("WARNING")

    def error(self) -> None:
        self.set_status("ERROR")

    def get_data(self) -> dict[str, Any]:
        return dict(self.data)


