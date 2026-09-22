"""
RENIX Display Controller
========================

Windows display-management controller for RENIX.

Features:
- Get display information
- Get screen resolution
- Get monitor count
- Get primary monitor
- Set display resolution
- Change orientation where supported
- Open Windows display settings
- Open advanced display settings
- Enumerate connected monitors
- Get virtual desktop dimensions

Dependencies:
    pip install screeninfo pywin32

Windows-focused implementation.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# DATA MODEL
# ==============================================================


@dataclass
class DisplayInfo:
    """
    Information about one physical display.
    """

    index: int
    name: str
    x: int
    y: int
    width: int
    height: int
    is_primary: bool
    refresh_rate: Optional[int] = None

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================
# DISPLAY CONTROLLER
# ==============================================================


class DisplayController:
    """
    High-level display controller for RENIX.

    Example:

        display = DisplayController()

        print(display.get_resolution())
        print(display.get_monitors())

        display.open_display_settings()
    """

    def __init__(
        self,
        enabled: bool = True,
    ) -> None:

        self.enabled = enabled

        self._screeninfo_available = False

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:

        try:

            import screeninfo  # noqa: F401

            self._screeninfo_available = True

            logger.info(
                "RENIX display backend initialized."
            )

        except Exception as exc:

            logger.warning(
                "screeninfo is unavailable: %s",
                exc,
            )

            self._screeninfo_available = False

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
                "RENIX display control is disabled."
            )

        if os.name != "nt":

            raise RuntimeError(
                "DisplayController currently "
                "supports Windows only."
            )

    # ==========================================================
    # MONITOR ENUMERATION
    # ==========================================================

    def get_monitors(
        self,
    ) -> list[DisplayInfo]:
        """
        Enumerate connected monitors.
        """

        self._check()

        if not self._screeninfo_available:

            raise RuntimeError(
                "screeninfo is not installed. "
                "Install it with: pip install screeninfo"
            )

        try:

            from screeninfo import get_monitors

            monitors = get_monitors()

        except Exception as exc:

            raise RuntimeError(
                f"Unable to enumerate displays: {exc}"
            ) from exc

        results: list[
            DisplayInfo
        ] = []

        for index, monitor in enumerate(
            monitors
        ):

            width = int(
                getattr(
                    monitor,
                    "width",
                    0,
                )
            )

            height = int(
                getattr(
                    monitor,
                    "height",
                    0,
                )
            )

            x = int(
                getattr(
                    monitor,
                    "x",
                    0,
                )
            )

            y = int(
                getattr(
                    monitor,
                    "y",
                    0,
                )
            )

            name = str(
                getattr(
                    monitor,
                    "name",
                    f"Display {index + 1}",
                )
            )

            is_primary = (
                x == 0
                and y == 0
            )

            refresh_rate = getattr(
                monitor,
                "refresh_rate",
                None,
            )

            if refresh_rate is not None:

                try:
                    refresh_rate = int(
                        refresh_rate
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    refresh_rate = None

            results.append(
                DisplayInfo(
                    index=index,
                    name=name,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    is_primary=is_primary,
                    refresh_rate=refresh_rate,
                )
            )

        return results

    # ==========================================================
    # MONITOR COUNT
    # ==========================================================

    def get_monitor_count(self) -> int:

        return len(
            self.get_monitors()
        )

    # ==========================================================
    # PRIMARY DISPLAY
    # ==========================================================

    def get_primary_display(
        self,
    ) -> Optional[DisplayInfo]:

        monitors = self.get_monitors()

        if not monitors:
            return None

        for monitor in monitors:

            if monitor.is_primary:

                return monitor

        return monitors[0]

    # ==========================================================
    # RESOLUTION
    # ==========================================================

    def get_resolution(
        self,
        monitor_index: int = 0,
    ) -> tuple[int, int]:

        monitors = self.get_monitors()

        if not monitors:

            raise RuntimeError(
                "No displays detected."
            )

        if not (
            0
            <= monitor_index
            < len(monitors)
        ):

            raise IndexError(
                f"Display index {monitor_index} "
                f"is out of range."
            )

        monitor = monitors[
            monitor_index
        ]

        return (
            monitor.width,
            monitor.height,
        )

    def get_primary_resolution(
        self,
    ) -> tuple[int, int]:

        primary = (
            self.get_primary_display()
        )

        if primary is None:

            raise RuntimeError(
                "No primary display detected."
            )

        return (
            primary.width,
            primary.height,
        )

    # ==========================================================
    # VIRTUAL DESKTOP
    # ==========================================================

    def get_virtual_desktop_bounds(
        self,
    ) -> dict[str, int]:

        monitors = self.get_monitors()

        if not monitors:

            return {
                "left": 0,
                "top": 0,
                "right": 0,
                "bottom": 0,
                "width": 0,
                "height": 0,
            }

        left = min(
            monitor.x
            for monitor in monitors
        )

        top = min(
            monitor.y
            for monitor in monitors
        )

        right = max(
            monitor.x + monitor.width
            for monitor in monitors
        )

        bottom = max(
            monitor.y + monitor.height
            for monitor in monitors
        )

        return {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": right - left,
            "height": bottom - top,
        }

    # ==========================================================
    # WINDOWS API
    # ==========================================================

    def _get_windows_screen_size(
        self,
    ) -> tuple[int, int]:

        self._check()

        user32 = ctypes.windll.user32

        return (
            int(
                user32.GetSystemMetrics(0)
            ),
            int(
                user32.GetSystemMetrics(1)
            ),
        )

    # ==========================================================
    # SET RESOLUTION
    # ==========================================================

    def set_resolution(
        self,
        width: int,
        height: int,
        monitor_index: int = 0,
    ) -> bool:
        """
        Change display resolution.

        This implementation uses the Windows display
        configuration API for the selected display.

        Returns True if Windows accepts the new mode.
        """

        self._check()

        width = int(width)
        height = int(height)

        if width <= 0 or height <= 0:

            raise ValueError(
                "Width and height must be positive."
            )

        monitors = self.get_monitors()

        if not (
            0
            <= monitor_index
            < len(monitors)
        ):

            raise IndexError(
                f"Display index {monitor_index} "
                f"is out of range."
            )

        if os.name != "nt":
            return False

        try:

            return self._set_windows_resolution(
                width,
                height,
                monitor_index,
            )

        except Exception as exc:

            logger.warning(
                "Unable to change resolution: %s",
                exc,
            )

            return False

    def _set_windows_resolution(
        self,
        width: int,
        height: int,
        monitor_index: int,
    ) -> bool:
        """
        Low-level Windows DEVMODE resolution change.
        """

        user32 = ctypes.windll.user32

        ENUM_CURRENT_SETTINGS = -1

        CDS_UPDATEREGISTRY = 0x00000001
        DISP_CHANGE_SUCCESSFUL = 0

        CCHDEVICENAME = 32
        CCHFORMNAME = 32

        class POINTL(ctypes.Structure):

            _fields_ = [
                ("x", ctypes.c_long),
                ("y", ctypes.c_long),
            ]

        class DEVMODEW(ctypes.Structure):

            _fields_ = [
                (
                    "dmDeviceName",
                    ctypes.c_wchar * CCHDEVICENAME,
                ),
                (
                    "dmSpecVersion",
                    ctypes.c_ushort,
                ),
                (
                    "dmDriverVersion",
                    ctypes.c_ushort,
                ),
                (
                    "dmSize",
                    ctypes.c_ushort,
                ),
                (
                    "dmDriverExtra",
                    ctypes.c_ushort,
                ),
                (
                    "dmFields",
                    ctypes.c_ulong,
                ),
                (
                    "dmPosition",
                    POINTL,
                ),
                (
                    "dmDisplayOrientation",
                    ctypes.c_ulong,
                ),
                (
                    "dmDisplayFixedOutput",
                    ctypes.c_ulong,
                ),
                (
                    "dmColor",
                    ctypes.c_short,
                ),
                (
                    "dmDuplex",
                    ctypes.c_short,
                ),
                (
                    "dmYResolution",
                    ctypes.c_short,
                ),
                (
                    "dmTTOption",
                    ctypes.c_short,
                ),
                (
                    "dmCollate",
                    ctypes.c_short,
                ),
                (
                    "dmFormName",
                    ctypes.c_wchar * CCHFORMNAME,
                ),
                (
                    "dmLogPixels",
                    ctypes.c_ushort,
                ),
                (
                    "dmBitsPerPel",
                    ctypes.c_ulong,
                ),
                (
                    "dmPelsWidth",
                    ctypes.c_ulong,
                ),
                (
                    "dmPelsHeight",
                    ctypes.c_ulong,
                ),
                (
                    "dmDisplayFlags",
                    ctypes.c_ulong,
                ),
                (
                    "dmDisplayFrequency",
                    ctypes.c_ulong,
                ),
                (
                    "dmICMMethod",
                    ctypes.c_ulong,
                ),
                (
                    "dmICMIntent",
                    ctypes.c_ulong,
                ),
                (
                    "dmMediaType",
                    ctypes.c_ulong,
                ),
                (
                    "dmDitherType",
                    ctypes.c_ulong,
                ),
                (
                    "dmReserved1",
                    ctypes.c_ulong,
                ),
                (
                    "dmReserved2",
                    ctypes.c_ulong,
                ),
                (
                    "dmPanningWidth",
                    ctypes.c_ulong,
                ),
                (
                    "dmPanningHeight",
                    ctypes.c_ulong,
                ),
            ]

        DM_PELSWIDTH = 0x80000
        DM_PELSHEIGHT = 0x100000

        monitor_name = (
            monitors_name := self._get_device_name(
                monitor_index
            )
        )

        if not monitor_name:

            return False

        devmode = DEVMODEW()

        devmode.dmSize = ctypes.sizeof(
            DEVMODEW
        )

        result = user32.EnumDisplaySettingsW(
            monitor_name,
            ENUM_CURRENT_SETTINGS,
            ctypes.byref(devmode),
        )

        if not result:

            return False

        devmode.dmPelsWidth = width
        devmode.dmPelsHeight = height

        devmode.dmFields = (
            DM_PELSWIDTH
            | DM_PELSHEIGHT
        )

        result = user32.ChangeDisplaySettingsExW(
            monitor_name,
            ctypes.byref(devmode),
            None,
            CDS_UPDATEREGISTRY,
            None,
        )

        return (
            result
            == DISP_CHANGE_SUCCESSFUL
        )

    def _get_device_name(
        self,
        monitor_index: int,
    ) -> Optional[str]:

        if os.name != "nt":
            return None

        try:

            user32 = ctypes.windll.user32

            class MONITORINFOEXW(
                ctypes.Structure
            ):

                _fields_ = [
                    (
                        "cbSize",
                        ctypes.c_ulong,
                    ),
                    (
                        "rcMonitor",
                        ctypes.c_long * 4,
                    ),
                    (
                        "rcWork",
                        ctypes.c_long * 4,
                    ),
                    (
                        "dwFlags",
                        ctypes.c_ulong,
                    ),
                    (
                        "szDevice",
                        ctypes.c_wchar * 32,
                    ),
                ]

            monitors_found: list[
                str
            ] = []

            MONITORINFOF_PRIMARY = 1

            def callback(
                hmonitor,
                hdc,
                rect,
                data,
            ):

                info = (
                    MONITORINFOEXW()
                )

                info.cbSize = ctypes.sizeof(
                    MONITORINFOEXW
                )

                if user32.GetMonitorInfoW(
                    hmonitor,
                    ctypes.byref(info),
                ):

                    monitors_found.append(
                        info.szDevice
                    )

                return 1

            MonitorEnumProc = (
                ctypes.WINFUNCTYPE(
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(
                        ctypes.c_long
                    ),
                    ctypes.c_double,
                )
            )

            callback_proc = MonitorEnumProc(
                callback
            )

            user32.EnumDisplayMonitors(
                None,
                None,
                callback_proc,
                0,
            )

            if (
                0
                <= monitor_index
                < len(monitors_found)
            ):

                return monitors_found[
                    monitor_index
                ]

        except Exception as exc:

            logger.debug(
                "Unable to obtain display device name: %s",
                exc,
            )

        return None

    # ==========================================================
    # ORIENTATION
    # ==========================================================

    def get_orientation(
        self,
        monitor_index: int = 0,
    ) -> Optional[str]:
        """
        Return approximate orientation from
        current display dimensions.
        """

        width, height = self.get_resolution(
            monitor_index
        )

        if width > height:
            return "landscape"

        if height > width:
            return "portrait"

        return "square"

    # ==========================================================
    # SETTINGS
    # ==========================================================

    def open_display_settings(self) -> bool:

        self._check()

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

    def open_advanced_display_settings(
        self,
    ) -> bool:

        self._check()

        try:

            subprocess.Popen(
                [
                    "explorer.exe",
                    "ms-settings:display-advanced",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return True

        except Exception as exc:

            logger.warning(
                "Unable to open advanced display settings: %s",
                exc,
            )

            return False

    # ==========================================================
    # DISPLAY IDENTIFICATION
    # ==========================================================

    def get_display_summary(
        self,
    ) -> list[dict[str, Any]]:

        monitors = self.get_monitors()

        return [
            monitor.to_dict()
            for monitor in monitors
        ]

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        status: dict[str, Any] = {
            "enabled": self.enabled,
            "platform": os.name,
            "screeninfo_available": (
                self._screeninfo_available
            ),
        }

        if self.enabled:

            try:

                monitors = (
                    self.get_monitors()
                )

                status[
                    "monitor_count"
                ] = len(monitors)

                status[
                    "monitors"
                ] = [
                    monitor.to_dict()
                    for monitor in monitors
                ]

                status[
                    "virtual_desktop"
                ] = (
                    self.get_virtual_desktop_bounds()
                )

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

        logger.info(
            "RENIX display controller shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_display: Optional[
    DisplayController
] = None


def get_display_controller() -> DisplayController:
    """
    Return the shared RENIX display controller.
    """

    global _default_display

    if _default_display is None:

        _default_display = (
            DisplayController()
        )

    return _default_display


