"""
RENIX Volume Controller
=======================

Windows audio-volume controller for RENIX.

Features:
- Get master volume
- Set master volume
- Increase/decrease volume
- Mute/unmute
- Toggle mute
- Open Windows sound settings
- Read mute state
- Safe clamping between 0 and 100

Backend:
    pycaw + comtypes

Install:
    pip install pycaw comtypes
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VolumeController:
    """
    High-level Windows volume controller for RENIX.

    Example:

        volume = VolumeController()

        volume.set_volume(50)
        volume.increase_volume(10)
        volume.mute()
    """

    def __init__(
        self,
        enabled: bool = True,
    ) -> None:

        self.enabled = enabled

        self._audio_device = None
        self._volume_interface = None

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:

        if os.name != "nt":
            logger.warning(
                "RENIX VolumeController is "
                "currently designed for Windows."
            )
            return

        try:

            from pycaw.pycaw import (
                AudioUtilities,
                IAudioEndpointVolume,
            )

            from comtypes import CLSCTX_ALL

            devices = (
                AudioUtilities
                .GetSpeakers()
            )

            interface = devices.Activate(
                IAudioEndpointVolume._iid_,
                CLSCTX_ALL,
                None,
            )

            self._volume_interface = (
                interface.QueryInterface(
                    IAudioEndpointVolume
                )
            )

            self._audio_device = devices

            logger.info(
                "RENIX audio volume backend initialized."
            )

        except Exception as exc:

            logger.warning(
                "Unable to initialize audio backend: %s",
                exc,
            )

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
                "RENIX volume control is disabled."
            )

        if os.name != "nt":
            raise RuntimeError(
                "VolumeController currently "
                "supports Windows only."
            )

        if self._volume_interface is None:
            raise RuntimeError(
                "Audio backend unavailable. "
                "Install pycaw and comtypes."
            )

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 100.0,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    # ==========================================================
    # GET VOLUME
    # ==========================================================

    def get_volume(self) -> float:
        """
        Return master volume as a percentage.
        """

        self._check()

        scalar = (
            self._volume_interface
            .GetMasterVolumeLevelScalar()
        )

        return round(
            float(scalar) * 100.0,
            2,
        )

    # ==========================================================
    # SET VOLUME
    # ==========================================================

    def set_volume(
        self,
        volume: float,
    ) -> float:
        """
        Set master volume.

        Args:
            volume:
                Percentage from 0 to 100.

        Returns:
            Actual volume percentage.
        """

        self._check()

        volume = self._clamp(
            volume
        )

        scalar = volume / 100.0

        self._volume_interface.SetMasterVolumeLevelScalar(
            scalar,
            None,
        )

        return self.get_volume()

    # ==========================================================
    # INCREASE
    # ==========================================================

    def increase_volume(
        self,
        amount: float = 10.0,
    ) -> float:

        current = self.get_volume()

        return self.set_volume(
            current + float(amount)
        )

    # ==========================================================
    # DECREASE
    # ==========================================================

    def decrease_volume(
        self,
        amount: float = 10.0,
    ) -> float:

        current = self.get_volume()

        return self.set_volume(
            current - float(amount)
        )

    # ==========================================================
    # MUTE
    # ==========================================================

    def mute(self) -> bool:

        self._check()

        self._volume_interface.SetMute(
            1,
            None,
        )

        return self.is_muted()

    # ==========================================================
    # UNMUTE
    # ==========================================================

    def unmute(self) -> bool:

        self._check()

        self._volume_interface.SetMute(
            0,
            None,
        )

        return self.is_muted()

    # ==========================================================
    # TOGGLE MUTE
    # ==========================================================

    def toggle_mute(self) -> bool:

        self._check()

        current = self.is_muted()

        self._volume_interface.SetMute(
            0 if current else 1,
            None,
        )

        return self.is_muted()

    # ==========================================================
    # MUTE STATE
    # ==========================================================

    def is_muted(self) -> bool:

        self._check()

        return bool(
            self._volume_interface
            .GetMute()
        )

    # ==========================================================
    # VOLUME SHORTCUTS
    # ==========================================================

    def volume_up(
        self,
        amount: float = 5.0,
    ) -> float:

        return self.increase_volume(
            amount
        )

    def volume_down(
        self,
        amount: float = 5.0,
    ) -> float:

        return self.decrease_volume(
            amount
        )

    def set_zero(self) -> float:

        return self.set_volume(0)

    def set_half(self) -> float:

        return self.set_volume(50)

    def set_max(self) -> float:

        return self.set_volume(100)

    # ==========================================================
    # WINDOWS SOUND SETTINGS
    # ==========================================================

    def open_sound_settings(self) -> bool:

        if os.name != "nt":
            return False

        try:

            subprocess.Popen(
                [
                    "explorer.exe",
                    "ms-settings:sound",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception as exc:

            logger.warning(
                "Unable to open sound settings: %s",
                exc,
            )

            return False

    # ==========================================================
    # DEFAULT DEVICE
    # ==========================================================

    def get_device_name(self) -> Optional[str]:

        self._check()

        try:

            if self._audio_device is None:
                return None

            device = (
                self._audio_device
                .GetSpeakers()
            )

            if hasattr(
                device,
                "FriendlyName",
            ):
                return str(
                    device.FriendlyName
                )

        except Exception as exc:

            logger.debug(
                "Unable to read audio device name: %s",
                exc,
            )

        return None

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        status: dict[str, Any] = {
            "enabled": self.enabled,
            "platform": os.name,
            "backend_available": (
                self._volume_interface
                is not None
            ),
        }

        if (
            self.enabled
            and self._volume_interface
            is not None
        ):

            try:

                status[
                    "volume"
                ] = self.get_volume()

                status[
                    "muted"
                ] = self.is_muted()

                status[
                    "device"
                ] = self.get_device_name()

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

        self._volume_interface = None
        self._audio_device = None

        logger.info(
            "RENIX volume controller shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_volume: Optional[
    VolumeController
] = None


def get_volume_controller() -> VolumeController:
    """
    Return the shared RENIX volume controller.
    """

    global _default_volume

    if _default_volume is None:

        _default_volume = (
            VolumeController()
        )

    return _default_volume


