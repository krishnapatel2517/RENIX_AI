"""
RENIX Holographic UI
Browser UI
=================

UI/state layer for RENIX browser control.

Responsibilities:
- Browser tabs
- Current URL and page state
- Navigation controls
- Search/address bar
- Loading state
- Page title
- History
- Bookmarks
- Downloads
- Find-in-page state
- Browser UI events
- Gesture/voice friendly command interface

Actual browser automation should be handled by:
    RENIX/browser/browser_manager.py
    RENIX/browser/browser_controller.py
    RENIX/browser/page_reader.py
    RENIX/browser/web_search.py
    RENIX/browser/form_controller.py
"""

from __future__ import annotations

import threading
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# ENUMS
# ============================================================================


class BrowserTabState(str, Enum):
    ACTIVE = "active"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    CLOSED = "closed"


class BrowserAction(str, Enum):
    BACK = "back"
    FORWARD = "forward"
    RELOAD = "reload"
    STOP = "stop"
    HOME = "home"
    SEARCH = "search"
    OPEN_URL = "open_url"
    NEW_TAB = "new_tab"
    CLOSE_TAB = "close_tab"
    DUPLICATE_TAB = "duplicate_tab"
    NEXT_TAB = "next_tab"
    PREVIOUS_TAB = "previous_tab"
    BOOKMARK = "bookmark"
    DOWNLOAD = "download"
    FIND = "find"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    RESET_ZOOM = "reset_zoom"
    OPEN_DEVTOOLS = "open_devtools"


class BrowserViewMode(str, Enum):
    NORMAL = "normal"
    IMMERSIVE = "immersive"
    SPLIT = "split"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class BrowserTab:

    id: str

    url: str = "about:blank"

    title: str = "New Tab"

    favicon: Optional[str] = None

    state: BrowserTabState = (
        BrowserTabState.ACTIVE
    )

    can_go_back: bool = False

    can_go_forward: bool = False

    loading_progress: float = 0.0

    zoom: float = 100.0

    error: Optional[str] = None

    created_at: float = field(
        default_factory=time.time
    )

    last_updated: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class BrowserHistoryEntry:

    url: str

    title: str = ""

    timestamp: float = field(
        default_factory=time.time
    )


@dataclass
class Bookmark:

    id: str

    url: str

    title: str

    folder: str = "Bookmarks"

    created_at: float = field(
        default_factory=time.time
    )

    favicon: Optional[str] = None


@dataclass
class DownloadItem:

    id: str

    url: str

    filename: str

    destination: str = ""

    progress: float = 0.0

    completed: bool = False

    failed: bool = False

    error: Optional[str] = None

    created_at: float = field(
        default_factory=time.time
    )


@dataclass
class BrowserActionResult:

    action: BrowserAction

    success: bool

    message: str = ""

    data: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# BROWSER UI
# ============================================================================


class BrowserUI:

    def __init__(
        self,
        *,
        home_url: str = "https://www.google.com",
        max_history: int = 200,
        max_tabs: int = 50,
    ) -> None:

        self.home_url = home_url

        self.max_history = max(
            1,
            int(max_history),
        )

        self.max_tabs = max(
            1,
            int(max_tabs),
        )

        self._tabs: dict[
            str,
            BrowserTab,
        ] = {}

        self._tab_order: list[str] = []

        self._active_tab_id: Optional[
            str
        ] = None

        self._history: list[
            BrowserHistoryEntry
        ] = []

        self._history_index = -1

        self._bookmarks: dict[
            str,
            Bookmark,
        ] = {}

        self._downloads: dict[
            str,
            DownloadItem,
        ] = {}

        self._address_text = ""

        self._search_text = ""

        self._find_text = ""

        self._find_visible = False

        self._find_match_count = 0

        self._find_current_match = 0

        self._view_mode = (
            BrowserViewMode.NORMAL
        )

        self._sidebar_visible = False

        self._running = False

        self._lock = threading.RLock()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._last_update = time.monotonic()

        self._create_initial_tab()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:
                return

            self._running = True

        self._emit(
            "started",
            self,
        )

    def stop(self) -> None:

        with self._lock:

            if not self._running:
                return

            self._running = False

        self._emit(
            "stopped",
            self,
        )

    @property
    def running(self) -> bool:

        with self._lock:
            return self._running

    def update(
        self,
        delta_time: Optional[float] = None,
    ) -> None:

        now = time.monotonic()

        if delta_time is None:

            delta_time = (
                now - self._last_update
            )

        self._last_update = now

        self._emit(
            "updated",
            delta_time,
        )

    # ========================================================================
    # INITIAL TAB
    # ========================================================================

    def _create_initial_tab(
        self,
    ) -> None:

        tab = BrowserTab(
            id=uuid.uuid4().hex,
            url=self.home_url,
            title="New Tab",
            state=BrowserTabState.ACTIVE,
        )

        self._tabs[tab.id] = tab

        self._tab_order.append(
            tab.id
        )

        self._active_tab_id = tab.id

        self._address_text = self.home_url

    # ========================================================================
    # TABS
    # ========================================================================

    @property
    def active_tab_id(
        self,
    ) -> Optional[str]:

        with self._lock:
            return self._active_tab_id

    def get_active_tab(
        self,
    ) -> Optional[BrowserTab]:

        with self._lock:

            if (
                self._active_tab_id
                is None
            ):
                return None

            return self._tabs.get(
                self._active_tab_id
            )

    def get_tab(
        self,
        tab_id: str,
    ) -> Optional[BrowserTab]:

        with self._lock:
            return self._tabs.get(
                tab_id
            )

    def get_tabs(
        self,
    ) -> list[BrowserTab]:

        with self._lock:

            return [
                self._tabs[tab_id]
                for tab_id
                in self._tab_order
                if tab_id in self._tabs
            ]

    def create_tab(
        self,
        url: Optional[str] = None,
        *,
        activate: bool = True,
    ) -> BrowserTab:

        with self._lock:

            if (
                len(self._tab_order)
                >= self.max_tabs
            ):

                raise RuntimeError(
                    "Maximum browser tabs reached."
                )

            target_url = (
                url or self.home_url
            )

            tab = BrowserTab(
                id=uuid.uuid4().hex,
                url=target_url,
                title="New Tab",
                state=(
                    BrowserTabState.LOADING
                    if target_url
                    != "about:blank"
                    else BrowserTabState.ACTIVE
                ),
            )

            self._tabs[tab.id] = tab

            self._tab_order.append(
                tab.id
            )

            if activate:

                self._active_tab_id = (
                    tab.id
                )

                self._address_text = (
                    target_url
                )

        self._emit(
            "tab_created",
            tab,
        )

        if activate:

            self._emit(
                "active_tab_changed",
                tab,
            )

        return tab

    def activate_tab(
        self,
        tab_id: str,
    ) -> bool:

        tab = self.get_tab(
            tab_id
        )

        if tab is None:
            return False

        with self._lock:

            self._active_tab_id = tab_id

            self._address_text = (
                tab.url
            )

        self._emit(
            "active_tab_changed",
            tab,
        )

        return True

    def close_tab(
        self,
        tab_id: Optional[str] = None,
    ) -> bool:

        with self._lock:

            target_id = (
                tab_id
                or self._active_tab_id
            )

            if target_id is None:
                return False

            if target_id not in self._tabs:
                return False

            was_active = (
                target_id
                == self._active_tab_id
            )

            self._tabs.pop(
                target_id
            )

            if target_id in self._tab_order:

                index = self._tab_order.index(
                    target_id
                )

                self._tab_order.remove(
                    target_id
                )

            else:

                index = 0

            if not self._tab_order:

                new_tab = self.create_tab(
                    self.home_url,
                    activate=True,
                )

                self._emit(
                    "tab_closed",
                    target_id,
                    new_tab,
                )

                return True

            if was_active:

                new_index = min(
                    index,
                    len(
                        self._tab_order
                    ) - 1,
                )

                new_id = self._tab_order[
                    new_index
                ]

                self._active_tab_id = (
                    new_id
                )

                self._address_text = (
                    self._tabs[
                        new_id
                    ].url
                )

        self._emit(
            "tab_closed",
            target_id,
        )

        self._emit(
            "active_tab_changed",
            self.get_active_tab(),
        )

        return True

    def duplicate_tab(
        self,
        tab_id: Optional[str] = None,
    ) -> Optional[BrowserTab]:

        original = (
            self.get_tab(
                tab_id
                or self._active_tab_id
            )
        )

        if original is None:
            return None

        duplicate = self.create_tab(
            original.url,
            activate=True,
        )

        duplicate.title = original.title

        duplicate.favicon = original.favicon

        duplicate.zoom = original.zoom

        duplicate.metadata = dict(
            original.metadata
        )

        self._emit(
            "tab_duplicated",
            original,
            duplicate,
        )

        return duplicate

    def next_tab(self) -> Optional[BrowserTab]:

        with self._lock:

            if not self._tab_order:
                return None

            if (
                self._active_tab_id
                not in self._tab_order
            ):

                target_id = (
                    self._tab_order[0]
                )

            else:

                index = (
                    self._tab_order.index(
                        self._active_tab_id
                    )
                )

                target_id = (
                    self._tab_order[
                        (index + 1)
                        % len(
                            self._tab_order
                        )
                    ]
                )

        self.activate_tab(
            target_id
        )

        return self.get_tab(
            target_id
        )

    def previous_tab(
        self,
    ) -> Optional[BrowserTab]:

        with self._lock:

            if not self._tab_order:
                return None

            if (
                self._active_tab_id
                not in self._tab_order
            ):

                target_id = (
                    self._tab_order[0]
                )

            else:

                index = (
                    self._tab_order.index(
                        self._active_tab_id
                    )
                )

                target_id = (
                    self._tab_order[
                        (index - 1)
                        % len(
                            self._tab_order
                        )
                    ]
                )

        self.activate_tab(
            target_id
        )

        return self.get_tab(
            target_id
        )

    # ========================================================================
    # ADDRESS BAR / URL
    # ========================================================================

    @property
    def address_text(self) -> str:

        with self._lock:
            return self._address_text

    def set_address_text(
        self,
        text: str,
    ) -> None:

        with self._lock:

            self._address_text = (
                text or ""
            )

        self._emit(
            "address_changed",
            self._address_text,
        )

    def open_url(
        self,
        url: str,
    ) -> BrowserActionResult:

        if not url or not url.strip():

            return BrowserActionResult(
                action=BrowserAction.OPEN_URL,
                success=False,
                message="URL cannot be empty.",
            )

        normalized = self._normalize_url(
            url.strip()
        )

        tab = self.get_active_tab()

        if tab is None:

            tab = self.create_tab(
                normalized
            )

        with self._lock:

            tab.url = normalized

            tab.title = normalized

            tab.state = (
                BrowserTabState.LOADING
            )

            tab.loading_progress = 0.0

            tab.error = None

            tab.last_updated = (
                time.time()
            )

            self._address_text = (
                normalized
            )

            self._push_history_locked(
                normalized,
                tab.title,
            )

        self._emit(
            "navigation_requested",
            normalized,
            tab,
        )

        self._emit(
            "tab_updated",
            tab,
        )

        return BrowserActionResult(
            action=BrowserAction.OPEN_URL,
            success=True,
            message="Navigation requested.",
            data={
                "url": normalized,
                "tab_id": tab.id,
            },
        )

    def search(
        self,
        query: str,
    ) -> BrowserActionResult:

        query = query.strip()

        if not query:

            return BrowserActionResult(
                action=BrowserAction.SEARCH,
                success=False,
                message="Search query is empty.",
            )

        self._search_text = query

        url = (
            "https://www.google.com/search?q="
            + self._url_encode(query)
        )

        result = self.open_url(
            url
        )

        self._emit(
            "search_requested",
            query,
        )

        return result

    @staticmethod
    def _normalize_url(
        value: str,
    ) -> str:

        lowered = value.lower()

        if (
            lowered.startswith(
                (
                    "http://",
                    "https://",
                    "file://",
                    "about:",
                )
            )
        ):

            return value

        if " " in value:

            return (
                "https://www.google.com/search?q="
                + BrowserUI._url_encode(
                    value
                )
            )

        return "https://" + value

    @staticmethod
    def _url_encode(
        value: str,
    ) -> str:

        from urllib.parse import quote_plus

        return quote_plus(
            value
        )

    # ========================================================================
    # NAVIGATION STATE
    # ========================================================================

    def set_loading(
        self,
        progress: float = 0.0,
    ) -> None:

        tab = self.get_active_tab()

        if tab is None:
            return

        with self._lock:

            tab.state = (
                BrowserTabState.LOADING
            )

            tab.loading_progress = max(
                0.0,
                min(
                    100.0,
                    float(progress),
                ),
            )

            tab.last_updated = (
                time.time()
            )

        self._emit(
            "loading_changed",
            tab,
        )

    def set_loaded(
        self,
        *,
        title: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:

        tab = self.get_active_tab()

        if tab is None:
            return

        with self._lock:

            tab.state = (
                BrowserTabState.LOADED
            )

            tab.loading_progress = 100.0

            if title is not None:

                tab.title = title

            if url is not None:

                tab.url = url

                self._address_text = (
                    url
                )

            tab.error = None

            tab.last_updated = (
                time.time()
            )

        self._emit(
            "page_loaded",
            tab,
        )

        self._emit(
            "tab_updated",
            tab,
        )

    def set_error(
        self,
        error: str,
    ) -> None:

        tab = self.get_active_tab()

        if tab is None:
            return

        with self._lock:

            tab.state = (
                BrowserTabState.ERROR
            )

            tab.error = error

            tab.last_updated = (
                time.time()
            )

        self._emit(
            "page_error",
            tab,
            error,
        )

    def go_back(self) -> BrowserActionResult:

        tab = self.get_active_tab()

        if tab is None:

            return BrowserActionResult(
                action=BrowserAction.BACK,
                success=False,
                message="No active tab.",
            )

        self._emit(
            "back_requested",
            tab,
        )

        return BrowserActionResult(
            action=BrowserAction.BACK,
            success=tab.can_go_back,
            message=(
                "Back navigation requested."
                if tab.can_go_back
                else "No previous page."
            ),
        )

    def go_forward(
        self,
    ) -> BrowserActionResult:

        tab = self.get_active_tab()

        if tab is None:

            return BrowserActionResult(
                action=BrowserAction.FORWARD,
                success=False,
                message="No active tab.",
            )

        self._emit(
            "forward_requested",
            tab,
        )

        return BrowserActionResult(
            action=BrowserAction.FORWARD,
            success=tab.can_go_forward,
            message=(
                "Forward navigation requested."
                if tab.can_go_forward
                else "No next page."
            ),
        )

    def reload(self) -> BrowserActionResult:

        tab = self.get_active_tab()

        if tab is None:

            return BrowserActionResult(
                action=BrowserAction.RELOAD,
                success=False,
                message="No active tab.",
            )

        with self._lock:

            tab.state = (
                BrowserTabState.LOADING
            )

            tab.loading_progress = 0.0

        self._emit(
            "reload_requested",
            tab,
        )

        return BrowserActionResult(
            action=BrowserAction.RELOAD,
            success=True,
            message="Reload requested.",
        )

    def stop_loading(
        self,
    ) -> BrowserActionResult:

        tab = self.get_active_tab()

        if tab is None:

            return BrowserActionResult(
                action=BrowserAction.STOP,
                success=False,
                message="No active tab.",
            )

        with self._lock:

            tab.state = (
                BrowserTabState.LOADED
            )

        self._emit(
            "stop_requested",
            tab,
        )

        return BrowserActionResult(
            action=BrowserAction.STOP,
            success=True,
            message="Loading stopped.",
        )

    def home(self) -> BrowserActionResult:

        return self.open_url(
            self.home_url
        )

    # ========================================================================
    # HISTORY
    # ========================================================================

    def _push_history_locked(
        self,
        url: str,
        title: str = "",
    ) -> None:

        if (
            self._history_index
            >= 0
            and self._history[
                self._history_index
            ].url
            == url
        ):

            return

        if (
            self._history_index
            < len(self._history) - 1
        ):

            self._history = self._history[
                : self._history_index + 1
            ]

        self._history.append(
            BrowserHistoryEntry(
                url=url,
                title=title,
            )
        )

        self._history_index = (
            len(self._history) - 1
        )

        if len(self._history) > self.max_history:

            overflow = (
                len(self._history)
                - self.max_history
            )

            self._history = (
                self._history[
                    overflow:
                ]
            )

            self._history_index = (
                len(self._history) - 1
            )

    def get_history(
        self,
    ) -> list[BrowserHistoryEntry]:

        with self._lock:

            return list(
                self._history
            )

    def clear_history(self) -> None:

        with self._lock:

            self._history.clear()

            self._history_index = -1

        self._emit(
            "history_cleared"
        )

    # ========================================================================
    # BOOKMARKS
    # ========================================================================

    def add_bookmark(
        self,
        url: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Bookmark:

        tab = self.get_active_tab()

        target_url = (
            url
            or (
                tab.url
                if tab
                else self.home_url
            )
        )

        target_title = (
            title
            or (
                tab.title
                if tab
                else target_url
            )
        )

        bookmark = Bookmark(
            id=uuid.uuid4().hex,
            url=target_url,
            title=target_title,
            favicon=(
                tab.favicon
                if tab
                else None
            ),
        )

        with self._lock:

            self._bookmarks[
                bookmark.id
            ] = bookmark

        self._emit(
            "bookmark_added",
            bookmark,
        )

        return bookmark

    def remove_bookmark(
        self,
        bookmark_id: str,
    ) -> bool:

        with self._lock:

            if (
                bookmark_id
                not in self._bookmarks
            ):
                return False

            bookmark = self._bookmarks.pop(
                bookmark_id
            )

        self._emit(
            "bookmark_removed",
            bookmark,
        )

        return True

    def get_bookmarks(
        self,
    ) -> list[Bookmark]:

        with self._lock:

            return list(
                self._bookmarks.values()
            )

    def is_bookmarked(
        self,
        url: Optional[str] = None,
    ) -> bool:

        tab = self.get_active_tab()

        target = (
            url
            or (
                tab.url
                if tab
                else ""
            )
        )

        with self._lock:

            return any(
                bookmark.url == target
                for bookmark
                in self._bookmarks.values()
            )

    # ========================================================================
    # DOWNLOADS
    # ========================================================================

    def create_download(
        self,
        url: str,
        filename: str,
        destination: str = "",
    ) -> DownloadItem:

        item = DownloadItem(
            id=uuid.uuid4().hex,
            url=url,
            filename=filename,
            destination=destination,
        )

        with self._lock:

            self._downloads[
                item.id
            ] = item

        self._emit(
            "download_created",
            item,
        )

        return item

    def update_download(
        self,
        download_id: str,
        *,
        progress: Optional[float] = None,
        completed: Optional[bool] = None,
        failed: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> bool:

        with self._lock:

            item = self._downloads.get(
                download_id
            )

            if item is None:
                return False

            if progress is not None:

                item.progress = max(
                    0.0,
                    min(
                        100.0,
                        float(progress),
                    ),
                )

            if completed is not None:

                item.completed = bool(
                    completed
                )

            if failed is not None:

                item.failed = bool(
                    failed
                )

            if error is not None:

                item.error = error

        self._emit(
            "download_updated",
            item,
        )

        return True

    def get_downloads(
        self,
    ) -> list[DownloadItem]:

        with self._lock:

            return list(
                self._downloads.values()
            )

    # ========================================================================
    # FIND IN PAGE
    # ========================================================================

    @property
    def find_visible(self) -> bool:

        with self._lock:
            return self._find_visible

    @property
    def find_text(self) -> str:

        with self._lock:
            return self._find_text

    def open_find(
        self,
        text: str = "",
    ) -> None:

        with self._lock:

            self._find_visible = True

            self._find_text = text

            self._find_current_match = 0

        self._emit(
            "find_opened",
            text,
        )

    def set_find_text(
        self,
        text: str,
        match_count: int = 0,
    ) -> None:

        with self._lock:

            self._find_text = text

            self._find_match_count = max(
                0,
                int(match_count),
            )

            self._find_current_match = 0

        self._emit(
            "find_changed",
            text,
            match_count,
        )

    def next_find_match(self) -> int:

        with self._lock:

            if self._find_match_count <= 0:

                return 0

            self._find_current_match = (
                (
                    self._find_current_match
                    + 1
                )
                % self._find_match_count
            )

            result = (
                self._find_current_match
            )

        self._emit(
            "find_match_changed",
            result,
        )

        return result

    def previous_find_match(
        self,
    ) -> int:

        with self._lock:

            if self._find_match_count <= 0:

                return 0

            self._find_current_match = (
                (
                    self._find_current_match
                    - 1
                )
                % self._find_match_count
            )

            result = (
                self._find_current_match
            )

        self._emit(
            "find_match_changed",
            result,
        )

        return result

    def close_find(self) -> None:

        with self._lock:

            self._find_visible = False

        self._emit(
            "find_closed"
        )

    # ========================================================================
    # ZOOM
    # ========================================================================

    def set_zoom(
        self,
        value: float,
    ) -> float:

        tab = self.get_active_tab()

        if tab is None:
            return 100.0

        with self._lock:

            tab.zoom = max(
                25.0,
                min(
                    500.0,
                    float(value),
                ),
            )

            value = tab.zoom

        self._emit(
            "zoom_changed",
            value,
            tab,
        )

        return value

    def zoom_in(
        self,
        amount: float = 10.0,
    ) -> float:

        tab = self.get_active_tab()

        if tab is None:
            return 100.0

        return self.set_zoom(
            tab.zoom + amount
        )

    def zoom_out(
        self,
        amount: float = 10.0,
    ) -> float:

        tab = self.get_active_tab()

        if tab is None:
            return 100.0

        return self.set_zoom(
            tab.zoom - amount
        )

    def reset_zoom(self) -> float:

        return self.set_zoom(
            100.0
        )

    # ========================================================================
    # VIEW
    # ========================================================================

    @property
    def view_mode(
        self,
    ) -> BrowserViewMode:

        with self._lock:
            return self._view_mode

    def set_view_mode(
        self,
        mode: BrowserViewMode,
    ) -> None:

        with self._lock:

            self._view_mode = mode

        self._emit(
            "view_mode_changed",
            mode,
        )

    @property
    def sidebar_visible(
        self,
    ) -> bool:

        with self._lock:
            return self._sidebar_visible

    def toggle_sidebar(self) -> bool:

        with self._lock:

            self._sidebar_visible = (
                not self._sidebar_visible
            )

            visible = (
                self._sidebar_visible
            )

        self._emit(
            "sidebar_changed",
            visible,
        )

        return visible

    # ========================================================================
    # ACTION DISPATCH
    # ========================================================================

    def execute_action(
        self,
        action: BrowserAction,
        **kwargs: Any,
    ) -> BrowserActionResult:

        if action == BrowserAction.BACK:
            return self.go_back()

        if action == BrowserAction.FORWARD:
            return self.go_forward()

        if action == BrowserAction.RELOAD:
            return self.reload()

        if action == BrowserAction.STOP:
            return self.stop_loading()

        if action == BrowserAction.HOME:
            return self.home()

        if action == BrowserAction.SEARCH:

            return self.search(
                kwargs.get(
                    "query",
                    self._search_text,
                )
            )

        if action == BrowserAction.OPEN_URL:

            return self.open_url(
                kwargs.get(
                    "url",
                    self._address_text,
                )
            )

        if action == BrowserAction.NEW_TAB:

            tab = self.create_tab(
                kwargs.get(
                    "url"
                ),
                activate=True,
            )

            return BrowserActionResult(
                action=action,
                success=True,
                message="New tab created.",
                data={
                    "tab_id": tab.id
                },
            )

        if action == BrowserAction.CLOSE_TAB:

            success = self.close_tab(
                kwargs.get(
                    "tab_id"
                )
            )

            return BrowserActionResult(
                action=action,
                success=success,
                message=(
                    "Tab closed."
                    if success
                    else "Unable to close tab."
                ),
            )

        if action == BrowserAction.DUPLICATE_TAB:

            tab = self.duplicate_tab(
                kwargs.get(
                    "tab_id"
                )
            )

            return BrowserActionResult(
                action=action,
                success=tab is not None,
                message=(
                    "Tab duplicated."
                    if tab
                    else "Unable to duplicate tab."
                ),
                data={
                    "tab_id": (
                        tab.id
                        if tab
                        else None
                    )
                },
            )

        if action == BrowserAction.NEXT_TAB:

            tab = self.next_tab()

            return BrowserActionResult(
                action=action,
                success=tab is not None,
                message="Next tab activated.",
            )

        if action == BrowserAction.PREVIOUS_TAB:

            tab = self.previous_tab()

            return BrowserActionResult(
                action=action,
                success=tab is not None,
                message="Previous tab activated.",
            )

        if action == BrowserAction.BOOKMARK:

            bookmark = self.add_bookmark(
                kwargs.get("url"),
                kwargs.get("title"),
            )

            return BrowserActionResult(
                action=action,
                success=True,
                message="Bookmark added.",
                data={
                    "bookmark_id": bookmark.id
                },
            )

        if action == BrowserAction.DOWNLOAD:

            item = self.create_download(
                kwargs.get(
                    "url",
                    self.address_text,
                ),
                kwargs.get(
                    "filename",
                    "download",
                ),
                kwargs.get(
                    "destination",
                    "",
                ),
            )

            self._emit(
                "download_requested",
                item,
            )

            return BrowserActionResult(
                action=action,
                success=True,
                message="Download requested.",
                data={
                    "download_id": item.id
                },
            )

        if action == BrowserAction.FIND:

            self.open_find(
                kwargs.get(
                    "text",
                    "",
                )
            )

            return BrowserActionResult(
                action=action,
                success=True,
                message="Find opened.",
            )

        if action == BrowserAction.ZOOM_IN:

            value = self.zoom_in(
                kwargs.get(
                    "amount",
                    10.0,
                )
            )

            return BrowserActionResult(
                action=action,
                success=True,
                message=(
                    f"Zoom set to "
                    f"{value:.0f}%."
                ),
                data={
                    "zoom": value
                },
            )

        if action == BrowserAction.ZOOM_OUT:

            value = self.zoom_out(
                kwargs.get(
                    "amount",
                    10.0,
                )
            )

            return BrowserActionResult(
                action=action,
                success=True,
                message=(
                    f"Zoom set to "
                    f"{value:.0f}%."
                ),
                data={
                    "zoom": value
                },
            )

        if action == BrowserAction.RESET_ZOOM:

            value = self.reset_zoom()

            return BrowserActionResult(
                action=action,
                success=True,
                message="Zoom reset.",
                data={
                    "zoom": value
                },
            )

        if action == BrowserAction.OPEN_DEVTOOLS:

            self._emit(
                "devtools_requested"
            )

            return BrowserActionResult(
                action=action,
                success=True,
                message="Developer tools requested.",
            )

        return BrowserActionResult(
            action=action,
            success=False,
            message="Unsupported browser action.",
        )

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def get_snapshot(
        self,
    ) -> dict[str, Any]:

        active_tab = (
            self.get_active_tab()
        )

        with self._lock:

            tabs = list(
                self.get_tabs()
            )

            bookmarks = list(
                self._bookmarks.values()
            )

            downloads = list(
                self._downloads.values()
            )

            history = list(
                self._history
            )

            return {
                "running": self._running,
                "active_tab_id": (
                    self._active_tab_id
                ),
                "address": self._address_text,
                "search": self._search_text,
                "view_mode": (
                    self._view_mode.value
                ),
                "sidebar_visible": (
                    self._sidebar_visible
                ),
                "tabs": [
                    self._serialize_tab(
                        tab
                    )
                    for tab in tabs
                ],
                "active_tab": (
                    self._serialize_tab(
                        active_tab
                    )
                    if active_tab
                    else None
                ),
                "history": [
                    {
                        "url": entry.url,
                        "title": entry.title,
                        "timestamp": entry.timestamp,
                    }
                    for entry in history
                ],
                "bookmarks": [
                    {
                        "id": bookmark.id,
                        "url": bookmark.url,
                        "title": bookmark.title,
                        "folder": bookmark.folder,
                        "created_at": (
                            bookmark.created_at
                        ),
                        "favicon": bookmark.favicon,
                    }
                    for bookmark in bookmarks
                ],
                "downloads": [
                    {
                        "id": item.id,
                        "url": item.url,
                        "filename": item.filename,
                        "destination": (
                            item.destination
                        ),
                        "progress": item.progress,
                        "completed": (
                            item.completed
                        ),
                        "failed": item.failed,
                        "error": item.error,
                        "created_at": (
                            item.created_at
                        ),
                    }
                    for item in downloads
                ],
                "find": {
                    "visible": (
                        self._find_visible
                    ),
                    "text": self._find_text,
                    "match_count": (
                        self._find_match_count
                    ),
                    "current_match": (
                        self._find_current_match
                    ),
                },
            }

    @staticmethod
    def _serialize_tab(
        tab: Optional[BrowserTab],
    ) -> Optional[dict[str, Any]]:

        if tab is None:
            return None

        return {
            "id": tab.id,
            "url": tab.url,
            "title": tab.title,
            "favicon": tab.favicon,
            "state": tab.state.value,
            "can_go_back": (
                tab.can_go_back
            ),
            "can_go_forward": (
                tab.can_go_forward
            ),
            "loading_progress": (
                tab.loading_progress
            ),
            "zoom": tab.zoom,
            "error": tab.error,
            "created_at": tab.created_at,
            "last_updated": (
                tab.last_updated
            ),
            "metadata": dict(
                tab.metadata
            ),
        }

    # ========================================================================
    # EVENTS
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not event:
            raise ValueError(
                "Event name cannot be empty."
            )

        if not callable(callback):
            raise TypeError(
                "Callback must be callable."
            )

        with self._lock:

            self._callbacks.setdefault(
                event,
                [],
            ).append(callback)

    def off(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        with self._lock:

            callbacks = self._callbacks.get(
                event,
                [],
            )

            if callback in callbacks:

                callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks.get(
                    event,
                    [],
                )
            )

        for callback in callbacks:

            try:

                callback(
                    *args,
                    **kwargs,
                )

            except Exception:

                # UI callbacks must never
                # crash RENIX.
                pass

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(self) -> dict[str, Any]:

        active = self.get_active_tab()

        return {
            "running": self.running,
            "tabs": len(
                self._tab_order
            ),
            "active_tab_id": (
                self._active_tab_id
            ),
            "active_url": (
                active.url
                if active
                else None
            ),
            "active_title": (
                active.title
                if active
                else None
            ),
            "active_state": (
                active.state.value
                if active
                else None
            ),
            "zoom": (
                active.zoom
                if active
                else 100.0
            ),
            "bookmarks": len(
                self._bookmarks
            ),
            "downloads": len(
                self._downloads
            ),
            "history": len(
                self._history
            ),
            "find_visible": (
                self._find_visible
            ),
            "view_mode": (
                self._view_mode.value
            ),
        }


# ============================================================================
# FACTORY
# ============================================================================


def create_browser_ui() -> BrowserUI:

    ui = BrowserUI(
        home_url="https://www.google.com",
        max_history=200,
        max_tabs=50,
    )

    ui.start()

    return ui


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "BrowserTabState",
    "BrowserAction",
    "BrowserViewMode",
    "BrowserTab",
    "BrowserHistoryEntry",
    "Bookmark",
    "DownloadItem",
    "BrowserActionResult",
    "BrowserUI",
    "create_browser_ui",
]


