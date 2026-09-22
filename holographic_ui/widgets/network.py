"""
RENIX Network Widget
"""

from __future__ import annotations

import time
from typing import Any


class NetworkWidget:

    def __init__(self) -> None:

        self.visible = True

        self.data: dict[str, Any] = {}

        self.update()

    def update(self) -> dict[str, Any]:

        sent = 0
        received = 0

        connected = False

        try:

            import psutil

            stats = psutil.net_io_counters()

            sent = stats.bytes_sent
            received = stats.bytes_recv

            connected = True

        except Exception:
            pass

        self.data = {
            "connected": connected,
            "bytes_sent": sent,
            "bytes_received": received,
            "status": (
                "CONNECTED"
                if connected
                else "OFFLINE"
            ),
            "timestamp": time.time(),
        }

        return self.data

    def get_data(self) -> dict[str, Any]:
        return dict(self.data)


