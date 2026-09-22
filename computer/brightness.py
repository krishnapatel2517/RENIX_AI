"""
RENIX Brightness Controller
===========================

Windows brightness controller for RENIX.

Features:
- Get brightness
- Set brightness
- Increase/decrease brightness
- Set minimum/maximum brightness
- Open Windows display settings
- Get available monitors
- Per-monitor brightness where supported

Primary backend:
    WMI / WMI-compatible Windows monitor interface

Install:
    pip install wmi pywin32

Note:
    Some external monitors do not expose brightness
    through Windows WMI. In that case RENIX reports
    the monitor as unsupported instead of failing silently.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BrightnessController:
    """
    High-level Windows brightness controller for RENIX.

    Example:

        brightness = BrightnessController()

        brightness.set_brightness(70)
        brightness.increase_brightness(10)
        brightness.decrease_brightness(10)
    """

    def __init__(
        self,
        enabled: bool = True,
    ) -> None:

        self.enabled = enabled

        self._wmi = None
        self._initialized = False

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:

        if os.name != "nt":

            logger.warning(
                "BrightnessController currently "
                "supports Windows only."
            )

            return

        try:

            import wmi

            self._wmi = wmi.WMI(
                namespace="root\\WMI"
            )

            self._initialized = True

            logger.info(
                "RENIX brightness backend initialized."
            )

        except Exception as exc:

            logger.warning(
                "Unable to initialize brightness backend: %s",
                exc,
            )

            self._wmi = None
            self._initialized = False

    # ==========================================================
    # STATE
    # ==========================================================

    def enable(self) -> None:

        self.enabled = True

    def disable(self) -> None:

        self.enabled = False

    def is_enabled(self) -> bool:

        return self.enabled

    def _check(self) -> None:

        if not self.enabled:

            raise RuntimeError(
                "RENIX brightness control is disabled."
            )

        if os.name != "nt":

            raise RuntimeError(
                "BrightnessController currently "
                "supports Windows only."
            )

        if self._wmi is None:

            raise RuntimeError(
                "Brightness backend unavailable. "
                "Install wmi and pywin32."
            )

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> int:

        return int(
            max(
                0,
                min(
                    100,
                    round(float(value)),
                ),
            )
        )

    # ==========================================================
    # MONITORS
    # ==========================================================

    def get_monitors(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return monitors exposing Windows
        brightness controls.
        """

        self._check()

        monitors: list[
            dict[str, Any]
        ] = []

        try:

            brightness_methods = (
                self._wmi.WmiMonitorBrightness()
            )

        except Exception as exc:

            logger.warning(
                "Unable to query monitor brightness: %s",
                exc,
            )

            return monitors

        for index, monitor in enumerate(
            brightness_methods
        ):

            try:

                current = int(
                    monitor.CurrentBrightness
                )

            except Exception:

                current = None

            try:

                instance_name = str(
                    monitor.InstanceName
                )

            except Exception:

                instance_name = (
                    f"Monitor {index + 1}"
                )

            monitors.append(
                {
                    "index": index,
                    "name": instance_name,
                    "brightness": current,
                    "supported": True,
                }
            )

        return monitors

    def get_monitor_count(self) -> int:

        return len(
            self.get_monitors()
        )

    # ==========================================================
    # GET BRIGHTNESS
    # ==========================================================

    def get_brightness(
        self,
        monitor_index: int = 0,
    ) -> int:
        """
        Return brightness from 0 to 100.
        """

        self._check()

        monitor_index = int(
            monitor_index
        )

        monitors = (
            self._wmi.WmiMonitorBrightness()
        )

        if not monitors:

            raise RuntimeError(
                "No brightness-capable monitors found."
            )

        if not (
            0
            <= monitor_index
            < len(monitors)
        ):

            raise IndexError(
                f"Monitor index {monitor_index} "
                f"is out of range."
            )

        return int(
            monitors[
                monitor_index
            ].CurrentBrightness
        )

    # ==========================================================
    # SET BRIGHTNESS
    # ==========================================================

    def set_brightness(
        self,
        brightness: float,
        monitor_index: int = 0,
    ) -> int:
        """
        Set monitor brightness from 0 to 100.
        """

        self._check()

        brightness = self._clamp(
            brightness
        )

        monitor_index = int(
            monitor_index
        )

        monitors = (
            self._wmi.WmiMonitorBrightnessMethods()
        )

        if not monitors:

            raise RuntimeError(
                "No brightness-capable monitors found."
            )

        if not (
            0
            <= monitor_index
            < len(monitors)
        ):

            raise IndexError(
                f"Monitor index {monitor_index} "
                f"is out of range."
            )

        method = monitors[
            monitor_index
        ]

        try:

            method.WmiSetBrightness(
                Timeout=1,
                Brightness=brightness,
            )

        except TypeError:

            # Some WMI implementations expose
            # the method with positional arguments.
            method.WmiSetBrightness(
                1,
                brightness,
            )

        return self.get_brightness(
            monitor_index
        )

    # ==========================================================
    # INCREASE
    # ==========================================================

    def increase_brightness(
        self,
        amount: float = 10,
        monitor_index: int = 0,
    ) -> int:

        current = self.get_brightness(
            monitor_index
        )

        return self.set_brightness(
            current + amount,
            monitor_index,
        )

    # ==========================================================
    # DECREASE
    # ==========================================================

    def decrease_brightness(
        self,
        amount: float = 10,
        monitor_index: int = 0,
    ) -> int:

        current = self.get_brightness(
            monitor_index
        )

        return self.set_brightness(
            current - amount,
            monitor_index,
        )

    # ==========================================================
    # PRESETS
    # ==========================================================

    def set_minimum(
        self,
        monitor_index: int = 0,
    ) -> int:

        return self.set_brightness(
            0,
            monitor_index,
        )

    def set_low(
        self,
        monitor_index: int = 0,
    ) -> int:

        return self.set_brightness(
            25,
            monitor_index,
        )

    def set_medium(
        self,
        monitor_index: int = 0,
    ) -> int:

        return self.set_brightness(
            50,
            monitor_index,
        )

    def set_high(
        self,
        monitor_index: int = 0,
    ) -> int:

        return self.set_brightness(
            75,
            monitor_index,
        )

    def set_maximum(
        self,
        monitor_index: int = 0,
    ) -> int:

        return self.set_brightness(
            100,
            monitor_index,
        )

    # ==========================================================
    # ALL MONITORS
    # ==========================================================

    def set_all_brightness(
        self,
        brightness: float,
    ) -> list[int]:

        self._check()

        brightness = self._clamp(
            brightness
        )

        monitors = (
            self.get_monitors()
        )

        results: list[int] = []

        for monitor in monitors:

            try:

                result = self.set_brightness(
                    brightness,
                    monitor["index"],
                )

                results.append(result)

            except Exception as exc:

                logger.warning(
                    "Unable to set monitor %s brightness: %s",
                    monitor["index"],
                    exc,
                )

        return results

    # ==========================================================
    # WINDOWS DISPLAY SETTINGS
    # ==========================================================

    def open_display_settings(self) -> bool:

        if os.name != "nt":
            return False

        try:

            subprocess.Popen(
                [
                    "explorer.exe",
                    "ms-settings:display",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception as exc:

            logger.warning(
                "Unable to open display settings: %s",
                exc,
            )

            return False

    # ==========================================================
    # NIGHT LIGHT
    # ==========================================================

    def open_night_light_settings(self) -> bool:

        if os.name != "nt":
            return False

        try:

            subprocess.Popen(
                [
                    "explorer.exe",
                    "ms-settings:nightlight",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception as exc:

            logger.warning(
                "Unable to open Night Light settings: %s",
                exc,
            )

            return False

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        status: dict[str, Any] = {
            "enabled": self.enabled,
            "platform": os.name,
            "backend_available": (
                self._wmi is not None
            ),
            "initialized": self._initialized,
        }

        if (
            self.enabled
            and self._wmi is not None
        ):

            try:

                monitors = (
                    self.get_monitors()
                )

                status[
                    "monitor_count"
                ] = len(monitors)

                status[
                    "monitors"
                ] = monitors

            except Exception as exc:

                status[
                    "error"
                ] = str(exc)

        return status

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        self._wmi = None
        self._initialized = False

        logger.info(
            "RENIX brightness controller shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_brightness: Optional[
    BrightnessController
] = None


def get_brightness_controller() -> BrightnessController:
    """
    Return the shared RENIX brightness controller.
    """

    global _default_brightness

    if _default_brightness is None:

        _default_brightness = (
            BrightnessController()
        )

    return _default_brightness


