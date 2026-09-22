"""
RENIX Volume Manager

Provides a unified interface for controlling media volume,
mute state, volume limits, and volume stepping.

The actual system-level audio backend can be connected later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VolumeState:
    """Current RENIX volume state."""

    volume: float = 50.0
    muted: bool = False
    previous_volume: float = 50.0
    minimum: float = 0.0
    maximum: float = 100.0
    step: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume": self.volume,
            "muted": self.muted,
            "previous_volume": self.previous_volume,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
        }


class VolumeManager:
    """
    RENIX media volume controller.

    This class maintains volume state independently from the
    operating-system audio backend.

    A platform-specific implementation can later synchronize
    this state with Windows, Linux, macOS, Bluetooth devices,
    speakers, or other audio hardware.
    """

    def __init__(
        self,
        initial_volume: float = 50.0,
        minimum: float = 0.0,
        maximum: float = 100.0,
        step: float = 5.0,
    ) -> None:

        if maximum <= minimum:
            raise ValueError(
                "maximum must be greater than minimum."
            )

        if step <= 0:
            raise ValueError(
                "step must be greater than zero."
            )

        self.state = VolumeState(
            volume=initial_volume,
            muted=False,
            previous_volume=initial_volume,
            minimum=minimum,
            maximum=maximum,
            step=step,
        )

        self.state.volume = self._clamp(
            initial_volume
        )

        self.state.previous_volume = (
            self.state.volume
        )

    # ============================================================
    # BASIC VOLUME
    # ============================================================

    def get_volume(self) -> float:
        """Return current volume percentage."""

        return self.state.volume

    def set_volume(
        self,
        volume: float,
    ) -> dict[str, Any]:

        volume = self._clamp(
            float(volume)
        )

        if volume > 0:
            self.state.previous_volume = volume

        self.state.volume = volume

        if volume > 0:
            self.state.muted = False

        return self.status()

    def increase(
        self,
        amount: float | None = None,
    ) -> dict[str, Any]:

        amount = (
            self.state.step
            if amount is None
            else float(amount)
        )

        return self.set_volume(
            self.state.volume + amount
        )

    def decrease(
        self,
        amount: float | None = None,
    ) -> dict[str, Any]:

        amount = (
            self.state.step
            if amount is None
            else float(amount)
        )

        return self.set_volume(
            self.state.volume - amount
        )

    def max_volume(self) -> dict[str, Any]:

        return self.set_volume(
            self.state.maximum
        )

    def min_volume(self) -> dict[str, Any]:

        return self.set_volume(
            self.state.minimum
        )

    # ============================================================
    # MUTE
    # ============================================================

    def mute(self) -> dict[str, Any]:

        if not self.state.muted:

            if self.state.volume > 0:
                self.state.previous_volume = (
                    self.state.volume
                )

            self.state.muted = True

        return self.status()

    def unmute(self) -> dict[str, Any]:

        if self.state.muted:

            self.state.muted = False

            if self.state.volume <= 0:

                self.state.volume = (
                    self.state.previous_volume
                    if self.state.previous_volume > 0
                    else self.state.step
                )

                self.state.volume = self._clamp(
                    self.state.volume
                )

        return self.status()

    def toggle_mute(self) -> dict[str, Any]:

        if self.state.muted:
            return self.unmute()

        return self.mute()

    def is_muted(self) -> bool:

        return self.state.muted

    # ============================================================
    # PERCENTAGE / NORMALIZATION
    # ============================================================

    def get_normalized_volume(self) -> float:
        """
        Return volume between 0.0 and 1.0.
        """

        span = (
            self.state.maximum
            - self.state.minimum
        )

        if span <= 0:
            return 0.0

        return (
            self.state.volume
            - self.state.minimum
        ) / span

    def set_normalized_volume(
        self,
        value: float,
    ) -> dict[str, Any]:

        value = max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

        volume = (
            self.state.minimum
            + (
                self.state.maximum
                - self.state.minimum
            )
            * value
        )

        return self.set_volume(
            volume
        )

    # ============================================================
    # STEPS
    # ============================================================

    def set_step(
        self,
        step: float,
    ) -> float:

        step = float(step)

        if step <= 0:
            raise ValueError(
                "Volume step must be greater than zero."
            )

        self.state.step = step

        return self.state.step

    def get_step(self) -> float:

        return self.state.step

    # ============================================================
    # LIMITS
    # ============================================================

    def set_limits(
        self,
        minimum: float,
        maximum: float,
    ) -> dict[str, Any]:

        minimum = float(minimum)
        maximum = float(maximum)

        if maximum <= minimum:
            raise ValueError(
                "maximum must be greater than minimum."
            )

        self.state.minimum = minimum
        self.state.maximum = maximum

        self.state.volume = self._clamp(
            self.state.volume
        )

        self.state.previous_volume = self._clamp(
            self.state.previous_volume
        )

        return self.status()

    def get_limits(self) -> tuple[float, float]:

        return (
            self.state.minimum,
            self.state.maximum,
        )

    # ============================================================
    # PREVIOUS VOLUME
    # ============================================================

    def restore_previous_volume(
        self,
    ) -> dict[str, Any]:

        return self.set_volume(
            self.state.previous_volume
        )

    def remember_current_volume(self) -> float:

        if self.state.volume > 0:

            self.state.previous_volume = (
                self.state.volume
            )

        return self.state.previous_volume

    # ============================================================
    # STATUS
    # ============================================================

    def status(self) -> dict[str, Any]:

        data = self.state.to_dict()

        data["normalized"] = (
            self.get_normalized_volume()
        )

        data["effective_volume"] = (
            0.0
            if self.state.muted
            else self.state.volume
        )

        return data

    # ============================================================
    # COMMAND INTERPRETATION
    # ============================================================

    def handle_command(
        self,
        command: str,
    ) -> dict[str, Any]:

        """
        Convert simple natural-language volume commands
        into volume operations.

        Examples:

            "volume up"
            "increase volume"
            "volume down"
            "mute"
            "unmute"
            "set volume 70"
        """

        command = (
            command or ""
        ).strip().lower()

        if not command:

            return {
                "success": False,
                "message": "Empty volume command.",
            }

        if (
            command in {
                "mute",
                "mute volume",
                "silence",
            }
        ):

            return self.mute()

        if (
            command in {
                "unmute",
                "unmute volume",
            }
        ):

            return self.unmute()

        if (
            command in {
                "toggle mute",
                "toggle mute volume",
            }
        ):

            return self.toggle_mute()

        if any(
            phrase in command
            for phrase in (
                "volume up",
                "increase volume",
                "turn volume up",
                "louder",
            )
        ):

            return self.increase()

        if any(
            phrase in command
            for phrase in (
                "volume down",
                "decrease volume",
                "turn volume down",
                "quieter",
                "lower volume",
            )
        ):

            return self.decrease()

        if (
            "maximum volume"
            in command
            or command == "max volume"
        ):

            return self.max_volume()

        if (
            "minimum volume"
            in command
            or command == "min volume"
        ):

            return self.min_volume()

        # --------------------------------------------------------
        # Numeric volume commands
        # --------------------------------------------------------

        import re

        match = re.search(
            r"(?:volume|set volume)\s*(?:to)?\s*(\d+(?:\.\d+)?)",
            command,
        )

        if match:

            value = float(
                match.group(1)
            )

            return self.set_volume(
                value
            )

        return {
            "success": False,
            "message": (
                f"Unknown volume command: {command}"
            ),
        }

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return self.status()

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "Volume state must be a dictionary."
            )

        minimum = float(
            data.get(
                "minimum",
                self.state.minimum,
            )
        )

        maximum = float(
            data.get(
                "maximum",
                self.state.maximum,
            )
        )

        if maximum <= minimum:
            raise ValueError(
                "Invalid volume limits."
            )

        self.state.minimum = minimum
        self.state.maximum = maximum

        self.state.volume = self._clamp(
            float(
                data.get(
                    "volume",
                    self.state.volume,
                )
            )
        )

        self.state.previous_volume = (
            self._clamp(
                float(
                    data.get(
                        "previous_volume",
                        self.state.previous_volume,
                    )
                )
            )
        )

        self.state.step = max(
            0.01,
            float(
                data.get(
                    "step",
                    self.state.step,
                )
            ),
        )

        self.state.muted = bool(
            data.get(
                "muted",
                False,
            )
        )

    # ============================================================
    # INTERNAL
    # ============================================================

    def _clamp(
        self,
        value: float,
    ) -> float:

        return max(
            self.state.minimum,
            min(
                self.state.maximum,
                value,
            ),
        )


class CastingManager:
    """
    Media casting facade with graceful fallback.

    A real backend can be injected when Chromecast, DLNA, AirPlay,
    or another provider is configured. Without a backend, casting
    capabilities are reported as unavailable.
    """

    def __init__(self, backend: Any | None = None) -> None:
        self.backend = backend

    def is_available(self) -> bool:
        return self.backend is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "available": self.is_available(),
            "backend": (
                type(self.backend).__name__
                if self.backend is not None
                else None
            ),
        }

    def list_devices(self) -> list[Any]:
        if self.backend is None:
            return []

        method = getattr(self.backend, "list_devices", None)
        if not callable(method):
            return []

        return list(method())

    def cast(self, media: Any, device: str | None = None) -> dict[str, Any]:
        if self.backend is None:
            return {
                "success": False,
                "error": "Casting backend is not configured.",
            }

        method = getattr(self.backend, "cast", None)
        if not callable(method):
            return {
                "success": False,
                "error": "Casting backend does not implement cast().",
            }

        result = method(media, device=device)
        if isinstance(result, dict):
            return result

        return {
            "success": bool(result),
            "result": result,
        }

    def stop(self, device: str | None = None) -> dict[str, Any]:
        if self.backend is None:
            return {
                "success": False,
                "error": "Casting backend is not configured.",
            }

        method = getattr(self.backend, "stop", None)
        if not callable(method):
            return {
                "success": False,
                "error": "Casting backend does not implement stop().",
            }

        result = method(device=device)
        if isinstance(result, dict):
            return result

        return {
            "success": bool(result),
            "result": result,
        }


__all__ = [
    "VolumeState",
    "VolumeManager",
    "CastingManager",
]


