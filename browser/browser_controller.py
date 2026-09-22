"""
RENIX Browser Controller
========================

Low-level browser interaction layer.

Handles:
    - Clicking
    - Typing
    - Selecting elements
    - Waiting
    - Scrolling
    - Keyboard actions
    - Element lookup
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class BrowserController:
    """Control interactions with a browser page."""

    def __init__(
        self,
        browser_manager: Any = None,
    ) -> None:
        self.browser_manager = browser_manager

    # ========================================================
    # PAGE
    # ========================================================

    def get_page(
        self,
        page_id: str | None = None,
    ) -> Any:
        """Get a browser page from BrowserManager."""

        if self.browser_manager is None:
            raise RuntimeError(
                "BrowserManager is not configured."
            )

        return self.browser_manager.get_page(
            page_id
        )

    # ========================================================
    # CLICK
    # ========================================================

    def click(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:
        """Click an element using a CSS selector."""

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        method = getattr(
            element,
            "click",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Element does not support click()."
            )

        return method()

    # ========================================================
    # DOUBLE CLICK
    # ========================================================

    def double_click(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        method = getattr(
            element,
            "dblclick",
            None,
        )

        if not callable(method):
            method = getattr(
                element,
                "double_click",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "Element does not support double-click."
            )

        return method()

    # ========================================================
    # TYPE
    # ========================================================

    def type_text(
        self,
        selector: str,
        text: str,
        *,
        page_id: str | None = None,
        clear: bool = True,
        timeout: float = 10.0,
    ) -> Any:
        """Type text into an input element."""

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        if clear:
            self.clear(
                selector,
                page_id=page_id,
                timeout=timeout,
            )

        method = getattr(
            element,
            "fill",
            None,
        )

        if callable(method):
            return method(
                text
            )

        method = getattr(
            element,
            "type",
            None,
        )

        if callable(method):
            return method(
                text
            )

        raise RuntimeError(
            "Element does not support text input."
        )

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        method = getattr(
            element,
            "fill",
            None,
        )

        if callable(method):
            return method("")

        method = getattr(
            element,
            "clear",
            None,
        )

        if callable(method):
            return method()

        raise RuntimeError(
            "Element does not support clearing."
        )

    # ========================================================
    # PRESS KEY
    # ========================================================

    def press(
        self,
        key: str,
        *,
        selector: str | None = None,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:
        """Press a keyboard key."""

        if selector:

            element = self.find(
                selector,
                page_id=page_id,
                timeout=timeout,
            )

            method = getattr(
                element,
                "press",
                None,
            )

            if callable(method):
                return method(
                    key
                )

        page = self.get_page(
            page_id
        )

        keyboard = getattr(
            page,
            "keyboard",
            None,
        )

        if keyboard is None:
            raise RuntimeError(
                "Browser page has no keyboard interface."
            )

        method = getattr(
            keyboard,
            "press",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Keyboard does not support press()."
            )

        return method(
            key
        )

    # ========================================================
    # SCROLL
    # ========================================================

    def scroll(
        self,
        amount: int = 600,
        *,
        page_id: str | None = None,
    ) -> Any:
        """Scroll vertically."""

        page = self.get_page(
            page_id
        )

        script = (
            "window.scrollBy(0, arguments[0]);"
        )

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if callable(evaluator):
            return evaluator(
                script,
                amount,
            )

        raise RuntimeError(
            "Browser page does not support JavaScript evaluation."
        )

    def scroll_to_top(
        self,
        *,
        page_id: str | None = None,
    ) -> Any:

        page = self.get_page(
            page_id
        )

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(evaluator):
            raise RuntimeError(
                "Browser page does not support JavaScript evaluation."
            )

        return evaluator(
            "window.scrollTo(0, 0);"
        )

    def scroll_to_bottom(
        self,
        *,
        page_id: str | None = None,
    ) -> Any:

        page = self.get_page(
            page_id
        )

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(evaluator):
            raise RuntimeError(
                "Browser page does not support JavaScript evaluation."
            )

        return evaluator(
            "window.scrollTo(0, document.body.scrollHeight);"
        )

    # ========================================================
    # FIND
    # ========================================================

    def find(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> Any:
        """Find an element using a CSS selector."""

        if not selector.strip():
            raise ValueError(
                "selector cannot be empty."
            )

        page = self.get_page(
            page_id
        )

        deadline = (
            time.monotonic()
            + timeout
        )

        while time.monotonic() < deadline:

            element = self._query(
                page,
                selector,
            )

            if element is not None:
                return element

            time.sleep(
                poll_interval
            )

        raise TimeoutError(
            f"Element not found: {selector}"
        )

    def _query(
        self,
        page: Any,
        selector: str,
    ) -> Any:

        method = getattr(
            page,
            "query_selector",
            None,
        )

        if callable(method):
            return method(
                selector
            )

        method = getattr(
            page,
            "locator",
            None,
        )

        if callable(method):
            locator = method(
                selector
            )

            count = getattr(
                locator,
                "count",
                None,
            )

            if callable(count):

                try:
                    if count() > 0:
                        return locator
                except Exception:
                    pass

            return locator

        raise RuntimeError(
            "Browser backend does not support element lookup."
        )

    # ========================================================
    # WAIT
    # ========================================================

    def wait(
        self,
        seconds: float,
    ) -> None:
        """Wait for a specified number of seconds."""

        if seconds < 0:
            raise ValueError(
                "seconds cannot be negative."
            )

        time.sleep(
            seconds
        )

    def wait_for_selector(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:
        """Wait until an element exists."""

        return self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

    # ========================================================
    # ELEMENT TEXT
    # ========================================================

    def get_text(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> str:

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        method = getattr(
            element,
            "inner_text",
            None,
        )

        if callable(method):
            return str(
                method()
            )

        method = getattr(
            element,
            "text_content",
            None,
        )

        if callable(method):
            return str(
                method()
            )

        value = getattr(
            element,
            "text",
            None,
        )

        if value is not None:
            return str(
                value
            )

        raise RuntimeError(
            "Element does not expose text."
        )

    # ========================================================
    # ATTRIBUTE
    # ========================================================

    def get_attribute(
        self,
        selector: str,
        attribute: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> str | None:

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        method = getattr(
            element,
            "get_attribute",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Element does not support get_attribute()."
            )

        return method(
            attribute
        )

    # ========================================================
    # SELECT
    # ========================================================

    def select(
        self,
        selector: str,
        value: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        method = getattr(
            element,
            "select_option",
            None,
        )

        if callable(method):
            return method(
                value
            )

        method = getattr(
            element,
            "select",
            None,
        )

        if callable(method):
            return method(
                value
            )

        raise RuntimeError(
            "Element does not support selection."
        )

    # ========================================================
    # CHECKBOX
    # ========================================================

    def check(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        method = getattr(
            element,
            "check",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Element does not support check()."
            )

        return method()

    def uncheck(
        self,
        selector: str,
        *,
        page_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:

        element = self.find(
            selector,
            page_id=page_id,
            timeout=timeout,
        )

        method = getattr(
            element,
            "uncheck",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Element does not support uncheck()."
            )

        return method()


__all__ = [
    "BrowserController",
]


