"""
RENIX Clipboard Controller
==========================

Clipboard management for RENIX.

Supports:
- Read text from clipboard
- Write text to clipboard
- Clear clipboard
- Copy using keyboard shortcuts
- Paste using keyboard shortcuts
- Clipboard history maintained by RENIX
- Safe clipboard state inspection

Primary backend:
    tkinter

Keyboard integration:
    RENIX KeyboardController
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ClipboardController:
    """
    High-level clipboard controller for RENIX.

    Example:

        clipboard = ClipboardController()

        clipboard.set_text("Hello RENIX")
        print(clipboard.get_text())

        clipboard.copy()
        clipboard.paste()
    """

    def __init__(
        self,
        enabled: bool = True,
        history_enabled: bool = True,
        max_history: int = 50,
    ) -> None:

        self.enabled = enabled

        self.history_enabled = history_enabled
        self.max_history = max(
            1,
            int(max_history),
        )

        self._tk_root: Optional[Any] = None
        self._initialized = False

        self._lock = threading.RLock()

        self.history: deque[str] = deque(
            maxlen=self.max_history
        )

        self._keyboard: Optional[Any] = None

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:
        """
        Initialize tkinter clipboard access.

        Tk is created lazily and hidden from the user.
        """

        try:

            import tkinter as tk

            root = tk.Tk()

            root.withdraw()

            try:
                root.attributes(
                    "-topmost",
                    False,
                )
            except Exception:
                pass

            self._tk_root = root
            self._initialized = True

            logger.info(
                "RENIX clipboard controller initialized."
            )

        except Exception as exc:

            self._initialized = False
            self._tk_root = None

            logger.warning(
                "Clipboard backend unavailable: %s",
                exc,
            )

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX clipboard control is disabled."
            )

        if not self._initialized:
            self._initialize()

        if self._tk_root is None:
            raise RuntimeError(
                "Clipboard backend is unavailable."
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
    # KEYBOARD BACKEND
    # ==========================================================

    def _get_keyboard(self) -> Any:

        if self._keyboard is None:

            try:

                from .keyboard import (
                    KeyboardController
                )

                self._keyboard = (
                    KeyboardController()
                )

            except Exception as exc:

                logger.warning(
                    "Keyboard controller unavailable: %s",
                    exc,
                )

                self._keyboard = False

        if self._keyboard is False:
            return None

        return self._keyboard

    # ==========================================================
    # GET CLIPBOARD
    # ==========================================================

    def get_text(
        self,
        default: str = "",
    ) -> str:
        """
        Read text from the system clipboard.
        """

        self._check()

        with self._lock:

            try:

                self._tk_root.update()

                value = self._tk_root.clipboard_get()

                if value is None:
                    return default

                value = str(value)

                self._record_history(
                    value
                )

                return value

            except Exception as exc:

                logger.debug(
                    "Unable to read clipboard: %s",
                    exc,
                )

                return default

    def read(self) -> str:
        return self.get_text()

    # ==========================================================
    # SET CLIPBOARD
    # ==========================================================

    def set_text(
        self,
        text: str,
    ) -> bool:
        """
        Put text into the system clipboard.
        """

        self._check()

        if text is None:
            text = ""

        text = str(text)

        with self._lock:

            try:

                self._tk_root.clipboard_clear()

                self._tk_root.clipboard_append(
                    text
                )

                self._tk_root.update()

                self._record_history(
                    text
                )

                return True

            except Exception as exc:

                logger.exception(
                    "Unable to write clipboard."
                )

                raise RuntimeError(
                    f"Unable to write clipboard: "
                    f"{exc}"
                ) from exc

    def write(
        self,
        text: str,
    ) -> bool:

        return self.set_text(text)

    # ==========================================================
    # CLEAR
    # ==========================================================

    def clear(self) -> bool:

        self._check()

        with self._lock:

            try:

                self._tk_root.clipboard_clear()

                self._tk_root.update()

                return True

            except Exception as exc:

                logger.warning(
                    "Unable to clear clipboard: %s",
                    exc,
                )

                return False

    # ==========================================================
    # COPY
    # ==========================================================

    def copy(
        self,
        delay: float = 0.05,
    ) -> str:
        """
        Copy currently selected content using Ctrl+C.

        Returns the resulting clipboard text.
        """

        self._check()

        keyboard = self._get_keyboard()

        if keyboard is None:
            raise RuntimeError(
                "Keyboard controller unavailable."
            )

        keyboard.hotkey(
            "ctrl",
            "c",
        )

        time.sleep(
            max(
                0.0,
                float(delay),
            )
        )

        return self.get_text()

    # ==========================================================
    # PASTE
    # ==========================================================

    def paste(
        self,
        text: Optional[str] = None,
        delay: float = 0.05,
    ) -> None:
        """
        Paste clipboard content.

        If text is provided, RENIX first places that text
        into the clipboard.
        """

        self._check()

        keyboard = self._get_keyboard()

        if keyboard is None:
            raise RuntimeError(
                "Keyboard controller unavailable."
            )

        if text is not None:
            self.set_text(text)

        time.sleep(
            max(
                0.0,
                float(delay),
            )
        )

        keyboard.hotkey(
            "ctrl",
            "v",
        )

    # ==========================================================
    # COPY + RETURN
    # ==========================================================

    def copy_selected_text(
        self,
        delay: float = 0.1,
    ) -> str:

        return self.copy(
            delay=delay
        )

    # ==========================================================
    # HISTORY
    # ==========================================================

    def _record_history(
        self,
        text: str,
    ) -> None:

        if not self.history_enabled:
            return

        if not text:
            return

        text = str(text)

        if (
            self.history
            and self.history[-1] == text
        ):
            return

        self.history.append(text)

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[str]:

        if limit is None:
            limit = len(self.history)

        limit = max(
            0,
            int(limit),
        )

        if limit == 0:
            return []

        return list(
            self.history
        )[-limit:][::-1]

    def history_count(self) -> int:
        return len(self.history)

    def clear_history(self) -> None:
        self.history.clear()

    def remove_history_item(
        self,
        index: int,
    ) -> bool:

        items = list(self.history)

        if not items:
            return False

        index = int(index)

        if index < 0:
            index += len(items)

        if not (
            0 <= index < len(items)
        ):
            return False

        items.pop(index)

        self.history = deque(
            items,
            maxlen=self.max_history,
        )

        return True

    # ==========================================================
    # HISTORY PASTE
    # ==========================================================

    def paste_history_item(
        self,
        index: int,
        delay: float = 0.05,
    ) -> str:

        items = self.get_history()

        if not (
            0 <= int(index) < len(items)
        ):
            raise IndexError(
                "Clipboard history index out of range."
            )

        text = items[
            int(index)
        ]

        self.paste(
            text=text,
            delay=delay,
        )

        return text

    # ==========================================================
    # CLIPBOARD INFORMATION
    # ==========================================================

    def has_text(self) -> bool:

        text = self.get_text(
            default=""
        )

        return bool(
            text.strip()
        )

    def get_length(self) -> int:

        return len(
            self.get_text()
        )

    def get_preview(
        self,
        max_length: int = 100,
    ) -> str:

        text = self.get_text()

        max_length = max(
            1,
            int(max_length),
        )

        if len(text) <= max_length:
            return text

        return (
            text[: max_length - 3]
            + "..."
        )

    # ==========================================================
    # SAFE CLIPBOARD OPERATIONS
    # ==========================================================

    def replace_text(
        self,
        old: str,
        new: str,
        count: int = -1,
    ) -> str:
        """
        Replace text currently in the clipboard.

        Does not modify the active application.
        """

        current = self.get_text()

        result = current.replace(
            str(old),
            str(new),
            int(count),
        )

        self.set_text(result)

        return result

    def append_text(
        self,
        text: str,
        separator: str = "",
    ) -> str:

        current = self.get_text()

        if current:
            result = (
                current
                + separator
                + str(text)
            )
        else:
            result = str(text)

        self.set_text(result)

        return result

    # ==========================================================
    # EXPORT / IMPORT
    # ==========================================================

    def export_history(self) -> list[dict[str, Any]]:

        return [
            {
                "index": index,
                "text": value,
            }
            for index, value in enumerate(
                self.get_history()
            )
        ]

    def import_history(
        self,
        values: list[str],
        replace: bool = False,
    ) -> None:

        if replace:
            self.clear_history()

        for value in values:

            if value is None:
                continue

            self._record_history(
                str(value)
            )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        current = ""

        if self._initialized:

            try:
                current = self.get_text()
            except Exception:
                current = ""

        return {
            "enabled": self.enabled,
            "available": self._initialized,
            "history_enabled": (
                self.history_enabled
            ),
            "history_count": len(
                self.history
            ),
            "clipboard_length": len(
                current
            ),
        }

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        self.history.clear()

        if self._tk_root is not None:

            try:
                self._tk_root.destroy()
            except Exception:
                pass

        self._tk_root = None
        self._initialized = False

        logger.info(
            "RENIX clipboard controller shut down."
        )


# ==============================================================
# OPTIONAL SINGLETON
# ==============================================================

_default_clipboard: Optional[
    ClipboardController
] = None


def get_clipboard() -> ClipboardController:
    """
    Return the shared RENIX clipboard controller.
    """

    global _default_clipboard

    if _default_clipboard is None:
        _default_clipboard = (
            ClipboardController()
        )

    return _default_clipboard


ClipboardManager = ClipboardController


