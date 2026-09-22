"""
RENIX Keyboard Controller
=========================

Keyboard input controller for RENIX.

Supports:
- Single key presses
- Key release
- Text typing
- Hotkeys
- Key combinations
- Repeated keys
- Clipboard-friendly typing
- Safe enable/disable state
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class KeyboardController:
    """
    High-level keyboard controller used by RENIX.

    The implementation uses PyAutoGUI as the default backend
    and keeps the backend lazily initialized so importing RENIX
    does not immediately fail if the dependency is unavailable.
    """

    # Common aliases accepted by RENIX.
    KEY_ALIASES = {
        "return": "enter",
        "esc": "escape",
        "del": "delete",
        "bksp": "backspace",
        "spacebar": "space",
        "ctrl": "ctrl",
        "control": "ctrl",
        "cmd": "command",
        "win": "win",
        "windows": "win",
        "altgr": "altright",
        "shiftleft": "shiftleft",
        "shiftright": "shiftright",
    }

    SPECIAL_KEYS = {
        "backspace",
        "delete",
        "down",
        "end",
        "enter",
        "escape",
        "home",
        "insert",
        "left",
        "pagedown",
        "pageup",
        "pause",
        "printscreen",
        "right",
        "scrolllock",
        "space",
        "tab",
        "up",
        "capslock",
        "numlock",
        "shift",
        "ctrl",
        "alt",
        "command",
        "win",
        "apps",
        "menu",
    }

    MODIFIER_KEYS = {
        "shift",
        "ctrl",
        "alt",
        "command",
        "win",
    }

    def __init__(
        self,
        enabled: bool = True,
        typing_interval: float = 0.0,
        fail_safe: bool = True,
    ) -> None:

        self.enabled = enabled
        self.typing_interval = max(
            0.0,
            float(typing_interval),
        )
        self.fail_safe = fail_safe

        self._pyautogui: Optional[Any] = None
        self._initialized = False

        self.pressed_keys: set[str] = set()

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:
        """
        Lazily initialize the keyboard backend.
        """

        try:
            import pyautogui

            self._pyautogui = pyautogui

            if self.fail_safe:
                pyautogui.FAILSAFE = True

            self._initialized = True

            logger.info(
                "RENIX keyboard controller initialized."
            )

        except Exception as exc:

            self._initialized = False

            logger.warning(
                "Keyboard backend unavailable: %s",
                exc,
            )

    # ==========================================================
    # STATE
    # ==========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.release_all()

        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def is_available(self) -> bool:
        return self._initialized

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX keyboard control is disabled."
            )

        if not self._initialized:
            self._initialize()

        if self._pyautogui is None:
            raise RuntimeError(
                "Keyboard backend is unavailable. "
                "Install PyAutoGUI."
            )

    # ==========================================================
    # KEY NORMALIZATION
    # ==========================================================

    def normalize_key(
        self,
        key: str,
    ) -> str:

        if not isinstance(key, str):
            raise TypeError(
                "Keyboard key must be a string."
            )

        normalized = (
            key.strip()
            .lower()
        )

        normalized = self.KEY_ALIASES.get(
            normalized,
            normalized,
        )

        return normalized

    def _validate_key(
        self,
        key: str,
    ) -> str:

        key = self.normalize_key(key)

        if not key:
            raise ValueError(
                "Keyboard key cannot be empty."
            )

        if self._pyautogui is not None:

            valid_keys = getattr(
                self._pyautogui,
                "KEYBOARD_KEYS",
                [],
            )

            if (
                key not in valid_keys
                and len(key) != 1
            ):
                raise ValueError(
                    f"Unsupported keyboard key: {key}"
                )

        return key

    # ==========================================================
    # PRESS
    # ==========================================================

    def press(
        self,
        key: str,
        presses: int = 1,
        interval: float = 0.0,
    ) -> None:

        self._check()

        key = self._validate_key(key)

        presses = max(
            1,
            int(presses),
        )

        interval = max(
            0.0,
            float(interval),
        )

        self._pyautogui.press(
            key,
            presses=presses,
            interval=interval,
        )

    def press_key(
        self,
        key: str,
    ) -> None:

        self.press(key)

    def repeat(
        self,
        key: str,
        count: int,
        interval: float = 0.05,
    ) -> None:

        self.press(
            key,
            presses=max(1, int(count)),
            interval=max(0.0, interval),
        )

    # ==========================================================
    # KEY DOWN / UP
    # ==========================================================

    def key_down(
        self,
        key: str,
    ) -> None:

        self._check()

        key = self._validate_key(key)

        self._pyautogui.keyDown(key)

        self.pressed_keys.add(key)

    def key_up(
        self,
        key: str,
    ) -> None:

        self._check()

        key = self._validate_key(key)

        self._pyautogui.keyUp(key)

        self.pressed_keys.discard(key)

    def hold(
        self,
        key: str,
        duration: float = 0.1,
    ) -> None:

        self.key_down(key)

        try:
            time.sleep(
                max(0.0, float(duration))
            )
        finally:
            self.key_up(key)

    # ==========================================================
    # TEXT TYPING
    # ==========================================================

    def type_text(
        self,
        text: str,
        interval: Optional[float] = None,
    ) -> None:

        self._check()

        if not isinstance(text, str):
            text = str(text)

        if interval is None:
            interval = self.typing_interval

        interval = max(
            0.0,
            float(interval),
        )

        self._pyautogui.write(
            text,
            interval=interval,
        )

    def write(
        self,
        text: str,
        interval: Optional[float] = None,
    ) -> None:

        self.type_text(
            text,
            interval,
        )

    def type_slowly(
        self,
        text: str,
        characters_per_second: float = 30.0,
    ) -> None:

        self._check()

        cps = max(
            1.0,
            float(characters_per_second),
        )

        interval = 1.0 / cps

        self.type_text(
            text,
            interval=interval,
        )

    # ==========================================================
    # HOTKEYS
    # ==========================================================

    def hotkey(
        self,
        *keys: str,
        interval: float = 0.0,
    ) -> None:

        self._check()

        if not keys:
            raise ValueError(
                "At least one key is required."
            )

        normalized_keys = [
            self._validate_key(key)
            for key in keys
        ]

        interval = max(
            0.0,
            float(interval),
        )

        self._pyautogui.hotkey(
            *normalized_keys,
            interval=interval,
        )

    def shortcut(
        self,
        *keys: str,
    ) -> None:

        self.hotkey(*keys)

    # ==========================================================
    # COMMON SHORTCUTS
    # ==========================================================

    def copy(self) -> None:
        self.hotkey("ctrl", "c")

    def paste(self) -> None:
        self.hotkey("ctrl", "v")

    def cut(self) -> None:
        self.hotkey("ctrl", "x")

    def undo(self) -> None:
        self.hotkey("ctrl", "z")

    def redo(self) -> None:
        self.hotkey("ctrl", "y")

    def select_all(self) -> None:
        self.hotkey("ctrl", "a")

    def save(self) -> None:
        self.hotkey("ctrl", "s")

    def save_as(self) -> None:
        self.hotkey(
            "ctrl",
            "shift",
            "s",
        )

    def find(self) -> None:
        self.hotkey("ctrl", "f")

    def refresh(self) -> None:
        self.press("f5")

    def close_window(self) -> None:
        self.hotkey(
            "alt",
            "f4",
        )

    def switch_window(self) -> None:
        self.hotkey(
            "alt",
            "tab",
        )

    def open_task_manager(self) -> None:
        self.hotkey(
            "ctrl",
            "shift",
            "esc",
        )

    def lock_computer(self) -> None:
        self.hotkey(
            "win",
            "l",
        )

    # ==========================================================
    # FUNCTION KEYS
    # ==========================================================

    def function_key(
        self,
        number: int,
    ) -> None:

        number = int(number)

        if not 1 <= number <= 24:
            raise ValueError(
                "Function key must be between F1 and F24."
            )

        self.press(
            f"f{number}"
        )

    # ==========================================================
    # NAVIGATION
    # ==========================================================

    def enter(self) -> None:
        self.press("enter")

    def escape(self) -> None:
        self.press("escape")

    def tab(self) -> None:
        self.press("tab")

    def backspace(self) -> None:
        self.press("backspace")

    def delete(self) -> None:
        self.press("delete")

    def space(self) -> None:
        self.press("space")

    def arrow_up(self) -> None:
        self.press("up")

    def arrow_down(self) -> None:
        self.press("down")

    def arrow_left(self) -> None:
        self.press("left")

    def arrow_right(self) -> None:
        self.press("right")

    # ==========================================================
    # MODIFIER OPERATIONS
    # ==========================================================

    def hold_modifier(
        self,
        modifier: str,
    ) -> None:

        modifier = self.normalize_key(
            modifier
        )

        if modifier not in self.MODIFIER_KEYS:
            raise ValueError(
                f"{modifier} is not a modifier key."
            )

        self.key_down(modifier)

    def release_modifier(
        self,
        modifier: str,
    ) -> None:

        modifier = self.normalize_key(
            modifier
        )

        if modifier not in self.MODIFIER_KEYS:
            raise ValueError(
                f"{modifier} is not a modifier key."
            )

        self.key_up(modifier)

    # ==========================================================
    # TYPING HELPERS
    # ==========================================================

    def type_line(
        self,
        text: str,
        press_enter: bool = True,
    ) -> None:

        self.type_text(text)

        if press_enter:
            self.press("enter")

    def type_lines(
        self,
        lines: list[str],
        interval: Optional[float] = None,
    ) -> None:

        for index, line in enumerate(lines):

            self.type_text(
                line,
                interval=interval,
            )

            if index < len(lines) - 1:
                self.press("enter")

    def clear_line(self) -> None:

        self.hotkey(
            "home"
        )

        self.hotkey(
            "shift",
            "end",
        )

        self.press(
            "backspace"
        )

    # ==========================================================
    # FAILSAFE / BACKEND
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

        return {
            "enabled": self.enabled,
            "available": self._initialized,
            "failsafe": self.fail_safe,
            "typing_interval": self.typing_interval,
            "pressed_keys": sorted(
                self.pressed_keys
            ),
        }

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def release_all(self) -> None:
        """
        Release every key RENIX believes it is holding.
        """

        if not self._initialized:
            self.pressed_keys.clear()
            return

        for key in list(
            self.pressed_keys
        ):

            try:
                self._pyautogui.keyUp(key)
            except Exception:
                logger.exception(
                    "Failed to release key: %s",
                    key,
                )

        self.pressed_keys.clear()

    def shutdown(self) -> None:

        self.release_all()

        self.enabled = False

        logger.info(
            "RENIX keyboard controller shut down."
        )


