"""
RENIX Mouse Controller
======================

Low-level mouse control for RENIX.

Supports:
- Move cursor
- Click
- Double click
- Mouse down/up
- Drag
- Scroll
- Position detection
- Screen-bound clamping
- Safe enable/disable
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MouseController:
    """
    Cross-platform mouse controller.

    RENIX can use this module directly or control it through
    computer_controller.py and the gesture system.
    """

    VALID_BUTTONS = {
        "left",
        "right",
        "middle",
    }

    def __init__(
        self,
        enabled: bool = True,
        fail_safe: bool = True,
    ) -> None:

        self.enabled = enabled
        self.fail_safe = fail_safe

        self._pyautogui: Optional[Any] = None
        self._initialized = False

        self.last_position = (0, 0)

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:
        """
        Initialize PyAutoGUI lazily.

        This prevents RENIX from crashing during import if the
        dependency is not installed yet.
        """

        try:

            import pyautogui

            self._pyautogui = pyautogui

            if self.fail_safe:
                pyautogui.FAILSAFE = True

            self._initialized = True

            self.last_position = tuple(
                pyautogui.position()
            )

            logger.info(
                "RENIX mouse controller initialized"
            )

        except Exception as exc:

            self._initialized = False

            logger.warning(
                "Mouse backend unavailable: %s",
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

    def is_available(self) -> bool:
        return self._initialized

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX mouse control is disabled."
            )

        if not self._initialized:
            self._initialize()

        if self._pyautogui is None:
            raise RuntimeError(
                "Mouse backend is unavailable. "
                "Install PyAutoGUI."
            )

    # ==========================================================
    # POSITION
    # ==========================================================

    def get_position(self) -> tuple[int, int]:

        self._check()

        position = self._pyautogui.position()

        self.last_position = (
            int(position.x),
            int(position.y),
        )

        return self.last_position

    def position(self) -> tuple[int, int]:
        return self.get_position()

    def get_screen_size(self) -> tuple[int, int]:

        self._check()

        size = self._pyautogui.size()

        return (
            int(size.width),
            int(size.height),
        )

    # ==========================================================
    # MOVEMENT
    # ==========================================================

    def move(
        self,
        x: int,
        y: int,
        duration: float = 0.0,
    ) -> tuple[int, int]:

        self._check()

        x, y = self._clamp_coordinates(
            x,
            y,
        )

        self._pyautogui.moveTo(
            x,
            y,
            duration=max(
                0.0,
                duration,
            ),
        )

        self.last_position = (
            x,
            y,
        )

        return self.last_position

    def move_relative(
        self,
        dx: int,
        dy: int,
        duration: float = 0.0,
    ) -> tuple[int, int]:

        self._check()

        self._pyautogui.moveRel(
            dx,
            dy,
            duration=max(
                0.0,
                duration,
            ),
        )

        return self.get_position()

    def _clamp_coordinates(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int]:

        width, height = self.get_screen_size()

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
    # CLICK
    # ==========================================================

    def click(
        self,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.0,
    ) -> None:

        self._check()

        button = self._validate_button(
            button
        )

        clicks = max(
            1,
            int(clicks),
        )

        self._pyautogui.click(
            button=button,
            clicks=clicks,
            interval=max(
                0.0,
                interval,
            ),
        )

    def double_click(
        self,
        button: str = "left",
        interval: float = 0.1,
    ) -> None:

        self.click(
            button=button,
            clicks=2,
            interval=interval,
        )

    def triple_click(
        self,
        button: str = "left",
        interval: float = 0.1,
    ) -> None:

        self.click(
            button=button,
            clicks=3,
            interval=interval,
        )

    # ==========================================================
    # BUTTON CONTROL
    # ==========================================================

    def button_down(
        self,
        button: str = "left",
    ) -> None:

        self._check()

        button = self._validate_button(
            button
        )

        self._pyautogui.mouseDown(
            button=button
        )

    def button_up(
        self,
        button: str = "left",
    ) -> None:

        self._check()

        button = self._validate_button(
            button
        )

        self._pyautogui.mouseUp(
            button=button
        )

    def mouse_down(
        self,
        button: str = "left",
    ) -> None:

        self.button_down(button)

    def mouse_up(
        self,
        button: str = "left",
    ) -> None:

        self.button_up(button)

    def _validate_button(
        self,
        button: str,
    ) -> str:

        button = str(
            button
        ).lower().strip()

        if button not in self.VALID_BUTTONS:
            raise ValueError(
                f"Invalid mouse button: {button}. "
                f"Expected one of "
                f"{sorted(self.VALID_BUTTONS)}."
            )

        return button

    # ==========================================================
    # DRAGGING
    # ==========================================================

    def drag_to(
        self,
        x: int,
        y: int,
        duration: float = 0.5,
        button: str = "left",
    ) -> tuple[int, int]:

        self._check()

        button = self._validate_button(
            button
        )

        x, y = self._clamp_coordinates(
            x,
            y,
        )

        self._pyautogui.dragTo(
            x,
            y,
            duration=max(
                0.0,
                duration,
            ),
            button=button,
        )

        self.last_position = (
            x,
            y,
        )

        return self.last_position

    def drag_relative(
        self,
        dx: int,
        dy: int,
        duration: float = 0.5,
        button: str = "left",
    ) -> tuple[int, int]:

        self._check()

        button = self._validate_button(
            button
        )

        self._pyautogui.dragRel(
            dx,
            dy,
            duration=max(
                0.0,
                duration,
            ),
            button=button,
        )

        return self.get_position()

    def drag_from_to(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
        button: str = "left",
    ) -> tuple[int, int]:

        self._check()

        self.move(
            start_x,
            start_y,
        )

        time.sleep(0.05)

        self.button_down(
            button
        )

        try:

            self.move(
                end_x,
                end_y,
                duration=duration,
            )

        finally:

            self.button_up(
                button
            )

        return self.get_position()

    # ==========================================================
    # SCROLL
    # ==========================================================

    def scroll(
        self,
        amount: int,
    ) -> None:

        self._check()

        self._pyautogui.scroll(
            int(amount)
        )

    def scroll_up(
        self,
        amount: int = 3,
    ) -> None:

        self.scroll(
            abs(int(amount))
        )

    def scroll_down(
        self,
        amount: int = 3,
    ) -> None:

        self.scroll(
            -abs(int(amount))
        )

    def horizontal_scroll(
        self,
        amount: int,
    ) -> None:

        self._check()

        method = getattr(
            self._pyautogui,
            "hscroll",
            None,
        )

        if method is None:
            raise RuntimeError(
                "Horizontal scrolling is not "
                "supported by the current backend."
            )

        method(
            int(amount)
        )

    # ==========================================================
    # POSITIONING HELPERS
    # ==========================================================

    def move_to_center(
        self,
        duration: float = 0.3,
    ) -> tuple[int, int]:

        width, height = (
            self.get_screen_size()
        )

        return self.move(
            width // 2,
            height // 2,
            duration=duration,
        )

    def move_to_top_left(
        self,
        margin: int = 0,
        duration: float = 0.2,
    ) -> tuple[int, int]:

        return self.move(
            margin,
            margin,
            duration=duration,
        )

    def move_to_top_right(
        self,
        margin: int = 0,
        duration: float = 0.2,
    ) -> tuple[int, int]:

        width, _ = self.get_screen_size()

        return self.move(
            width - 1 - margin,
            margin,
            duration=duration,
        )

    def move_to_bottom_left(
        self,
        margin: int = 0,
        duration: float = 0.2,
    ) -> tuple[int, int]:

        _, height = self.get_screen_size()

        return self.move(
            margin,
            height - 1 - margin,
            duration=duration,
        )

    def move_to_bottom_right(
        self,
        margin: int = 0,
        duration: float = 0.2,
    ) -> tuple[int, int]:

        width, height = (
            self.get_screen_size()
        )

        return self.move(
            width - 1 - margin,
            height - 1 - margin,
            duration=duration,
        )

    # ==========================================================
    # PAUSE / FAILSAFE
    # ==========================================================

    def set_failsafe(
        self,
        enabled: bool,
    ) -> None:

        self.fail_safe = bool(enabled)

        if self._pyautogui is not None:
            self._pyautogui.FAILSAFE = (
                self.fail_safe
            )

    def set_pause(
        self,
        seconds: float,
    ) -> None:

        self._check()

        self._pyautogui.PAUSE = max(
            0.0,
            float(seconds),
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        position = None

        if self._initialized:

            try:
                position = self.get_position()
            except Exception:
                position = None

        return {
            "enabled": self.enabled,
            "available": self._initialized,
            "failsafe": self.fail_safe,
            "position": position,
        }

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def release_all_buttons(self) -> None:

        if not self._initialized:
            return

        try:

            for button in self.VALID_BUTTONS:

                self._pyautogui.mouseUp(
                    button=button
                )

        except Exception:

            logger.exception(
                "Failed to release mouse buttons."
            )

    def shutdown(self) -> None:

        self.release_all_buttons()

        self.enabled = False

        logger.info(
            "RENIX mouse controller shut down."
        )


