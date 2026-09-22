"""
RENIX Screen Controller
=======================

Provides screen-level operations for RENIX.

Responsibilities:
- Detect screen resolution
- Read cursor position
- Capture screen pixels
- Locate colors
- Check screen bounds
- Support multi-monitor information where available
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ScreenController:
    """High-level screen controller for RENIX."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

        self._pyautogui: Optional[Any] = None
        self._initialized = False

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:
        try:
            import pyautogui

            self._pyautogui = pyautogui
            self._initialized = True

            logger.info(
                "RENIX screen controller initialized."
            )

        except Exception as exc:
            self._initialized = False

            logger.warning(
                "Screen backend unavailable: %s",
                exc,
            )

    def _check(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "RENIX screen control is disabled."
            )

        if not self._initialized:
            self._initialize()

        if self._pyautogui is None:
            raise RuntimeError(
                "Screen backend is unavailable. "
                "Install PyAutoGUI."
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

    def is_available(self) -> bool:
        return self._initialized

    # ==========================================================
    # SCREEN SIZE
    # ==========================================================

    def get_size(self) -> tuple[int, int]:
        """
        Return primary screen size as:

            (width, height)
        """

        self._check()

        size = self._pyautogui.size()

        return (
            int(size.width),
            int(size.height),
        )

    def width(self) -> int:
        return self.get_size()[0]

    def height(self) -> int:
        return self.get_size()[1]

    # ==========================================================
    # SCREEN CENTER
    # ==========================================================

    def get_center(self) -> tuple[int, int]:
        width, height = self.get_size()

        return (
            width // 2,
            height // 2,
        )

    # ==========================================================
    # BOUNDS
    # ==========================================================

    def is_inside(
        self,
        x: int,
        y: int,
    ) -> bool:
        width, height = self.get_size()

        return (
            0 <= int(x) < width
            and
            0 <= int(y) < height
        )

    def clamp(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int]:
        """
        Clamp coordinates to the primary screen.
        """

        width, height = self.get_size()

        x = max(
            0,
            min(
                int(x),
                width - 1,
            ),
        )

        y = max(
            0,
            min(
                int(y),
                height - 1,
            ),
        )

        return x, y

    # ==========================================================
    # SCREENSHOT
    # ==========================================================

    def screenshot(
        self,
        region: Optional[
            tuple[int, int, int, int]
        ] = None,
    ) -> Any:
        """
        Capture the screen.

        region:
            (left, top, width, height)

        Returns a PIL Image object.
        """

        self._check()

        return self._pyautogui.screenshot(
            region=region
        )

    def capture(
        self,
        region: Optional[
            tuple[int, int, int, int]
        ] = None,
    ) -> Any:
        return self.screenshot(region)

    def save_screenshot(
        self,
        path: str,
        region: Optional[
            tuple[int, int, int, int]
        ] = None,
    ) -> str:

        image = self.screenshot(region)

        image.save(path)

        return path

    # ==========================================================
    # PIXEL ACCESS
    # ==========================================================

    def get_pixel(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int, int]:

        self._check()

        x, y = self.clamp(x, y)

        pixel = self._pyautogui.pixel(
            x,
            y,
        )

        return (
            int(pixel[0]),
            int(pixel[1]),
            int(pixel[2]),
        )

    def pixel_matches(
        self,
        x: int,
        y: int,
        rgb: tuple[int, int, int],
        tolerance: int = 0,
    ) -> bool:

        current = self.get_pixel(
            x,
            y,
        )

        tolerance = max(
            0,
            int(tolerance),
        )

        return all(
            abs(
                current[index]
                - int(rgb[index])
            ) <= tolerance
            for index in range(3)
        )

    # ==========================================================
    # COLOR SEARCH
    # ==========================================================

    def locate_color(
        self,
        rgb: tuple[int, int, int],
        tolerance: int = 10,
        region: Optional[
            tuple[int, int, int, int]
        ] = None,
    ) -> Optional[tuple[int, int]]:

        self._check()

        image = self.screenshot(region)

        target = tuple(
            int(value)
            for value in rgb
        )

        tolerance = max(
            0,
            int(tolerance),
        )

        width, height = image.size

        for y in range(height):
            for x in range(width):

                pixel = image.getpixel(
                    (x, y)
                )

                if all(
                    abs(
                        pixel[index]
                        - target[index]
                    ) <= tolerance
                    for index in range(3)
                ):

                    if region is not None:
                        return (
                            x + region[0],
                            y + region[1],
                        )

                    return (
                        x,
                        y,
                    )

        return None

    # ==========================================================
    # DISPLAY INFORMATION
    # ==========================================================

    def get_primary_display(self) -> dict[str, Any]:
        width, height = self.get_size()

        return {
            "index": 0,
            "primary": True,
            "width": width,
            "height": height,
            "center": (
                width // 2,
                height // 2,
            ),
        }

    def get_display_count(self) -> int:
        """
        Return monitor count.

        Uses screeninfo when available.
        Falls back to one monitor.
        """

        try:
            from screeninfo import (
                get_monitors,
            )

            monitors = get_monitors()

            return len(monitors)

        except Exception:
            return 1

    def get_displays(self) -> list[dict[str, Any]]:
        """
        Return information about available displays.

        Uses screeninfo when installed.
        """

        try:
            from screeninfo import (
                get_monitors,
            )

            monitors = get_monitors()

            displays = []

            for index, monitor in enumerate(
                monitors
            ):
                displays.append(
                    {
                        "index": index,
                        "name": getattr(
                            monitor,
                            "name",
                            None,
                        ),
                        "x": int(
                            getattr(
                                monitor,
                                "x",
                                0,
                            )
                        ),
                        "y": int(
                            getattr(
                                monitor,
                                "y",
                                0,
                            )
                        ),
                        "width": int(
                            monitor.width
                        ),
                        "height": int(
                            monitor.height
                        ),
                        "primary": index == 0,
                    }
                )

            return displays

        except Exception:

            primary = (
                self.get_primary_display()
            )

            return [primary]

    # ==========================================================
    # SCREEN INFORMATION
    # ==========================================================

    def get_info(self) -> dict[str, Any]:

        width, height = self.get_size()

        return {
            "enabled": self.enabled,
            "available": self._initialized,
            "width": width,
            "height": height,
            "center": (
                width // 2,
                height // 2,
            ),
            "display_count": (
                self.get_display_count()
            ),
            "displays": self.get_displays(),
        }

    # ==========================================================
    # UTILITY
    # ==========================================================

    def normalize_coordinates(
        self,
        x: int,
        y: int,
    ) -> tuple[float, float]:
        """
        Convert pixel coordinates into normalized
        0.0 - 1.0 coordinates.

        Useful for the RENIX gesture engine.
        """

        width, height = self.get_size()

        return (
            float(x) / max(1, width - 1),
            float(y) / max(1, height - 1),
        )

    def denormalize_coordinates(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int]:
        """
        Convert normalized coordinates back to pixels.
        """

        width, height = self.get_size()

        px = int(
            max(0.0, min(1.0, x))
            * (width - 1)
        )

        py = int(
            max(0.0, min(1.0, y))
            * (height - 1)
        )

        return px, py

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def shutdown(self) -> None:
        self.enabled = False

        logger.info(
            "RENIX screen controller shut down."
        )


