"""
RENIX Browser Manager
=====================

High-level browser lifecycle manager.

Responsibilities:
    - Start and stop browser sessions
    - Manage browser pages/tabs
    - Navigate to URLs
    - Create and close tabs
    - Track the active page
    - Provide a common interface for browser automation

The actual browser implementation can be injected through
a backend/driver object, keeping RENIX independent of a
specific browser automation library.
"""

from __future__ import annotations

import logging
import threading
import uuid

from typing import Any, Callable

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manage RENIX browser sessions and pages."""

    def __init__(
        self,
        *,
        backend: Any = None,
        event_bus: Any = None,
        headless: bool = False,
        browser_name: str = "default",
    ) -> None:

        self.backend = backend
        self.event_bus = event_bus

        self.headless = bool(headless)
        self.browser_name = browser_name

        self._browser: Any = None

        self._pages: dict[
            str,
            Any,
        ] = {}

        self._page_metadata: dict[
            str,
            dict[str, Any],
        ] = {}

        self._active_page_id: str | None = None

        self._running = False

        self._lock = threading.RLock()

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {}

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def start(self) -> Any:
        """Start the browser."""

        with self._lock:

            if self._running:
                return self._browser

            if self.backend is None:
                raise RuntimeError(
                    "Browser backend is not configured."
                )

            launcher = getattr(
                self.backend,
                "start",
                None,
            )

            if not callable(launcher):
                raise RuntimeError(
                    "Browser backend must provide start()."
                )

            self._browser = launcher(
                headless=self.headless,
                browser_name=self.browser_name,
            )

            self._running = True

        self._emit(
            "browser.started",
            {
                "browser_name": self.browser_name,
                "headless": self.headless,
            },
        )

        logger.info(
            "RENIX browser started."
        )

        return self._browser

    def stop(self) -> None:
        """Stop the browser and close all pages."""

        with self._lock:

            if not self._running:
                return

            browser = self._browser

            self._pages.clear()
            self._page_metadata.clear()
            self._active_page_id = None

            self._browser = None
            self._running = False

        if browser is not None:

            closer = getattr(
                browser,
                "close",
                None,
            )

            if callable(closer):

                try:
                    closer()
                except Exception:
                    logger.exception(
                        "Failed to close browser."
                    )

        self._emit(
            "browser.stopped",
            {},
        )

    def restart(self) -> Any:
        """Restart the browser."""

        self.stop()
        return self.start()

    def is_running(self) -> bool:
        return self._running

    # ========================================================
    # PAGE MANAGEMENT
    # ========================================================

    def new_page(
        self,
        *,
        url: str | None = None,
        activate: bool = True,
    ) -> str:
        """Create a new browser page/tab."""

        self._ensure_running()

        creator = getattr(
            self._browser,
            "new_page",
            None,
        )

        if not callable(creator):
            raise RuntimeError(
                "Browser backend does not provide new_page()."
            )

        page = creator()

        page_id = str(
            uuid.uuid4()
        )

        with self._lock:

            self._pages[
                page_id
            ] = page

            self._page_metadata[
                page_id
            ] = {
                "id": page_id,
                "url": None,
                "title": None,
            }

            if activate:
                self._active_page_id = page_id

        if url:
            self.navigate(
                url,
                page_id=page_id,
            )

        self._emit(
            "browser.page_created",
            self._page_metadata[
                page_id
            ],
        )

        return page_id

    def close_page(
        self,
        page_id: str,
    ) -> bool:
        """Close a page/tab."""

        with self._lock:

            page = self._pages.pop(
                page_id,
                None,
            )

            self._page_metadata.pop(
                page_id,
                None,
            )

            was_active = (
                self._active_page_id
                == page_id
            )

            if was_active:
                self._active_page_id = (
                    next(
                        iter(
                            self._pages
                        ),
                        None,
                    )
                )

        if page is None:
            return False

        closer = getattr(
            page,
            "close",
            None,
        )

        if callable(closer):

            try:
                closer()
            except Exception:
                logger.exception(
                    "Failed to close browser page."
                )

        self._emit(
            "browser.page_closed",
            {
                "page_id": page_id,
            },
        )

        return True

    def get_page(
        self,
        page_id: str | None = None,
    ) -> Any:
        """Return a page object."""

        page_id = (
            page_id
            or self._active_page_id
        )

        if page_id is None:
            raise RuntimeError(
                "No active browser page."
            )

        with self._lock:

            page = self._pages.get(
                page_id
            )

        if page is None:
            raise KeyError(
                f"Browser page not found: {page_id}"
            )

        return page

    def list_pages(
        self,
    ) -> list[dict[str, Any]]:
        """Return metadata for all open pages."""

        with self._lock:

            return [
                dict(metadata)
                for metadata in
                self._page_metadata.values()
            ]

    def activate_page(
        self,
        page_id: str,
    ) -> bool:
        """Set a page as the active page."""

        with self._lock:

            if page_id not in self._pages:
                return False

            self._active_page_id = page_id

        self._emit(
            "browser.page_activated",
            {
                "page_id": page_id,
            },
        )

        return True

    @property
    def active_page_id(
        self,
    ) -> str | None:
        return self._active_page_id

    # ========================================================
    # NAVIGATION
    # ========================================================

    def navigate(
        self,
        url: str,
        *,
        page_id: str | None = None,
        wait_until: str | None = None,
    ) -> Any:
        """Navigate a page to a URL."""

        self._ensure_running()

        url = url.strip()

        if not url:
            raise ValueError(
                "URL cannot be empty."
            )

        page = self.get_page(
            page_id
        )

        navigator = getattr(
            page,
            "goto",
            None,
        )

        if not callable(navigator):
            navigator = getattr(
                page,
                "navigate",
                None,
            )

        if not callable(navigator):
            raise RuntimeError(
                "Browser page does not provide navigation."
            )

        if wait_until is None:
            result = navigator(
                url
            )
        else:
            result = navigator(
                url,
                wait_until=wait_until,
            )

        actual_page_id = (
            page_id
            or self._active_page_id
        )

        if actual_page_id is not None:

            with self._lock:

                metadata = self._page_metadata.get(
                    actual_page_id
                )

                if metadata is not None:
                    metadata["url"] = url

        self._emit(
            "browser.navigation",
            {
                "page_id": actual_page_id,
                "url": url,
            },
        )

        return result

    def back(
        self,
        *,
        page_id: str | None = None,
    ) -> Any:
        """Navigate backward."""

        page = self.get_page(
            page_id
        )

        method = getattr(
            page,
            "go_back",
            None,
        )

        if not callable(method):
            method = getattr(
                page,
                "back",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "Browser page does not support back navigation."
            )

        return method()

    def forward(
        self,
        *,
        page_id: str | None = None,
    ) -> Any:
        """Navigate forward."""

        page = self.get_page(
            page_id
        )

        method = getattr(
            page,
            "go_forward",
            None,
        )

        if not callable(method):
            method = getattr(
                page,
                "forward",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "Browser page does not support forward navigation."
            )

        return method()

    def reload(
        self,
        *,
        page_id: str | None = None,
    ) -> Any:
        """Reload the current page."""

        page = self.get_page(
            page_id
        )

        method = getattr(
            page,
            "reload",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Browser page does not support reload()."
            )

        return method()

    # ========================================================
    # PAGE INFORMATION
    # ========================================================

    def get_url(
        self,
        *,
        page_id: str | None = None,
    ) -> str | None:

        page = self.get_page(
            page_id
        )

        value = getattr(
            page,
            "url",
            None,
        )

        if callable(value):
            value = value()

        return value

    def get_title(
        self,
        *,
        page_id: str | None = None,
    ) -> str | None:

        page = self.get_page(
            page_id
        )

        value = getattr(
            page,
            "title",
            None,
        )

        if callable(value):
            value = value()

        return value

    def update_page_metadata(
        self,
        page_id: str,
    ) -> dict[str, Any]:
        """Refresh stored page metadata."""

        page = self.get_page(
            page_id
        )

        url = getattr(
            page,
            "url",
            None,
        )

        title = getattr(
            page,
            "title",
            None,
        )

        if callable(url):
            url = url()

        if callable(title):
            title = title()

        with self._lock:

            metadata = self._page_metadata.get(
                page_id
            )

            if metadata is None:
                raise KeyError(
                    f"Page metadata not found: {page_id}"
                )

            metadata["url"] = url
            metadata["title"] = title

            return dict(
                metadata
            )

    # ========================================================
    # ACTIVE PAGE HELPERS
    # ========================================================

    def ensure_page(self) -> str:
        """Return the active page or create one."""

        if self._active_page_id is not None:

            if self._active_page_id in self._pages:
                return self._active_page_id

        return self.new_page()

    def open(
        self,
        url: str,
    ) -> str:
        """Convenience method to open a URL."""

        page_id = self.ensure_page()

        self.navigate(
            url,
            page_id=page_id,
        )

        return page_id

    # ========================================================
    # SCREENSHOT
    # ========================================================

    def screenshot(
        self,
        path: str | None = None,
        *,
        page_id: str | None = None,
    ) -> Any:
        """Capture the current page."""

        page = self.get_page(
            page_id
        )

        method = getattr(
            page,
            "screenshot",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Browser page does not support screenshots."
            )

        if path is None:
            return method()

        return method(
            path=path
        )

    # ========================================================
    # JAVASCRIPT
    # ========================================================

    def execute_script(
        self,
        script: str,
        *args: Any,
        page_id: str | None = None,
    ) -> Any:
        """Execute JavaScript in the active page."""

        if not script.strip():
            raise ValueError(
                "script cannot be empty."
            )

        page = self.get_page(
            page_id
        )

        method = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(method):
            method = getattr(
                page,
                "execute_script",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "Browser page does not support script execution."
            )

        return method(
            script,
            *args,
        )

    # ========================================================
    # BACKEND
    # ========================================================

    def set_backend(
        self,
        backend: Any,
    ) -> None:
        """Set the browser backend."""

        if self._running:
            raise RuntimeError(
                "Cannot replace backend while browser is running."
            )

        self.backend = backend

    def set_headless(
        self,
        headless: bool,
    ) -> None:
        """Configure headless mode."""

        if self._running:
            raise RuntimeError(
                "Cannot change headless mode while browser is running."
            )

        self.headless = bool(
            headless
        )

    # ========================================================
    # EVENTS
    # ========================================================

    def subscribe(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> None:

        event = event.strip()

        if not event:
            raise ValueError(
                "Event cannot be empty."
            )

        if not callable(callback):
            raise TypeError(
                "callback must be callable."
            )

        self._callbacks.setdefault(
            event,
            [],
        ).append(
            callback
        )

    def unsubscribe(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> bool:

        callbacks = self._callbacks.get(
            event,
            [],
        )

        if callback not in callbacks:
            return False

        callbacks.remove(
            callback
        )

        return True

    def _emit(
        self,
        event: str,
        payload: dict[str, Any],
    ) -> None:

        for callback in list(
            self._callbacks.get(
                event,
                [],
            )
        ):

            try:
                callback(
                    payload
                )

            except Exception:
                logger.exception(
                    "Browser event callback failed."
                )

        if self.event_bus is not None:

            publish = getattr(
                self.event_bus,
                "publish",
                None,
            )

            if callable(publish):

                try:
                    publish(
                        event,
                        payload,
                    )

                except Exception:
                    logger.exception(
                        "Failed to publish browser event."
                    )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _ensure_running(self) -> None:

        if not self._running:
            self.start()


__all__ = [
    "BrowserManager",
]


