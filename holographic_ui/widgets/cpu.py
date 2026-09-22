"""
RENIX CPU Widget
"""

from __future__ import annotations

import os
import time
from typing import Any


class CPUWidget:

    def __init__(self) -> None:

        self.visible = True

        self.usage = 0.0
        self.cores = os.cpu_count() or 1

        self.data: dict[str, Any] = {}

        self.update()

    def _get_usage(self) -> float:

        try:

            import psutil

            return float(
                psutil.cpu_percent(
                    interval=None
                )
            )

        except Exception:

            return self.usage

    def update(self) -> dict[str, Any]:

        self.usage = self._get_usage()

        self.data = {
            "usage": round(
                self.usage,
                1,
            ),
            "cores": self.cores,
            "load": (
                "HIGH"
                if self.usage >= 80
                else "MEDIUM"
                if self.usage >= 50
                else "LOW"
            ),
            "timestamp": time.time(),
        }

        return self.data

    def get_data(self) -> dict[str, Any]:
        return dict(self.data)


