"""
RENIX GPU Widget
"""

from __future__ import annotations

import time
from typing import Any


class GPUWidget:

    def __init__(self) -> None:

        self.visible = True

        self.data: dict[str, Any] = {}

        self.update()

    def update(self) -> dict[str, Any]:

        gpu_name = "Unknown GPU"
        gpu_usage = 0.0
        gpu_memory = 0.0

        try:

            import GPUtil

            gpus = GPUtil.getGPUs()

            if gpus:

                gpu = gpus[0]

                gpu_name = gpu.name

                gpu_usage = float(
                    gpu.load * 100
                )

                gpu_memory = float(
                    gpu.memoryUtil * 100
                )

        except Exception:
            pass

        self.data = {
            "name": gpu_name,
            "usage": round(
                gpu_usage,
                1,
            ),
            "memory_usage": round(
                gpu_memory,
                1,
            ),
            "timestamp": time.time(),
        }

        return self.data

    def get_data(self) -> dict[str, Any]:
        return dict(self.data)


