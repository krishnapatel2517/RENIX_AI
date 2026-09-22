"""
RENIX RAM Widget
"""

from __future__ import annotations

import time
from typing import Any


class RAMWidget:

    def __init__(self) -> None:

        self.visible = True

        self.data: dict[str, Any] = {}

        self.update()

    def update(self) -> dict[str, Any]:

        total = 0.0
        used = 0.0
        available = 0.0
        percent = 0.0

        try:

            import psutil

            memory = psutil.virtual_memory()

            total = memory.total / (
                1024 ** 3
            )

            used = memory.used / (
                1024 ** 3
            )

            available = memory.available / (
                1024 ** 3
            )

            percent = float(
                memory.percent
            )

        except Exception:
            pass

        self.data = {
            "total_gb": round(
                total,
                2,
            ),
            "used_gb": round(
                used,
                2,
            ),
            "available_gb": round(
                available,
                2,
            ),
            "usage": round(
                percent,
                1,
            ),
            "timestamp": time.time(),
        }

        return self.data

    def get_data(self) -> dict[str, Any]:
        return dict(self.data)


