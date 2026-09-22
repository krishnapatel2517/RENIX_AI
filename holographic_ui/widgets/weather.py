"""
RENIX Weather Widget

The widget intentionally does not make network requests itself.
An integration/provider can push weather data into it.
"""

from __future__ import annotations

import time
from typing import Any, Optional


class WeatherWidget:

    def __init__(
        self,
        location: str = "Mumbai",
    ) -> None:

        self.location = location

        self.visible = True

        self.data: dict[str, Any] = {
            "location": location,
            "temperature": None,
            "condition": "Unknown",
            "humidity": None,
            "wind_speed": None,
            "timestamp": time.time(),
        }

    def update(
        self,
        temperature: Optional[float] = None,
        condition: str = "Unknown",
        humidity: Optional[float] = None,
        wind_speed: Optional[float] = None,
    ) -> dict[str, Any]:

        self.data = {
            "location": self.location,
            "temperature": temperature,
            "condition": condition,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "timestamp": time.time(),
        }

        return self.data

    def set_location(
        self,
        location: str,
    ) -> None:

        self.location = location

        self.data["location"] = location

    def get_data(self) -> dict[str, Any]:
        return dict(self.data)


