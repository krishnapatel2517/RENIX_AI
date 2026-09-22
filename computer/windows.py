"""
RENIX Window Manager
====================

Windows desktop window management for RENIX.

Supports:
- Listing visible windows
- Finding windows by title
- Focusing windows
- Minimizing / maximizing / restoring
- Moving and resizing windows
- Closing windows
- Getting the active window
- Basic window state inspection

Primary backend:
    pygetwindow

Designed for Windows 11 while keeping the module graceful when
the optional backend is unavailable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Normalized window information."""

    title: str
    left: int
    top: int
    width: int
    height: int
    visible: bool = True
    minimized: bool = False
    maximized: bool = False

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.left + self.width // 2,
            self.top + self.height // 2,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["right"] = self.right
        data["bottom"] = self.bottom
        data["center"] = self.center

        return data


class WindowManager:
    """
    High-level Windows window manager.

    Example:

        manager = WindowManager()

        manager.focus("Chrome")
        manager.maximize("Chrome")
        manager.move("Chrome", 100, 100)
    """

    def __init__(
        self,
        enabled: bool = True,
        case_sensitive: bool = False,
    ) -> None:

        self.enabled = enabled
        self.case_sensitive = case_sensitive

        self._backend: Optional[Any] = None
        self._initialized = False

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:

        try:

            import pygetwindow as gw

            self._backend = gw
            self._initialized = True

            logger.info(
                "RENIX window manager initialized."
            )

        except Exception as exc:

            self._initialized = False

            logger.warning(
                "Window backend unavailable: %s",
                exc,
            )

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX window management is disabled."
            )

        if not self._initialized:
            self._initialize()

        if self._backend is None:
            raise RuntimeError(
                "Window backend unavailable. "
                "Install PyGetWindow."
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
    # INTERNAL HELPERS
    # ==========================================================

    def _window_title(
        self,
        window: Any,
    ) -> str:

        return str(
            getattr(
                window,
                "title",
                "",
            )
            or ""
        )

    def _matches(
        self,
        window: Any,
        query: str,
        exact: bool = False,
    ) -> bool:

        title = self._window_title(
            window
        )

        if not self.case_sensitive:
            title = title.lower()
            query = query.lower()

        if exact:
            return title == query

        return query in title

    def _find_window(
        self,
        query: str,
        exact: bool = False,
    ) -> Any:

        windows = self._backend.getAllWindows()

        matches = [
            window
            for window in windows
            if self._matches(
                window,
                query,
                exact=exact,
            )
        ]

        if not matches:
            raise LookupError(
                f"No window found matching: {query}"
            )

        return matches[0]

    def _to_info(
        self,
        window: Any,
    ) -> WindowInfo:

        left = int(
            getattr(
                window,
                "left",
                0,
            )
        )

        top = int(
            getattr(
                window,
                "top",
                0,
            )
        )

        width = int(
            getattr(
                window,
                "width",
                0,
            )
        )

        height = int(
            getattr(
                window,
                "height",
                0,
            )
        )

        minimized = bool(
            getattr(
                window,
                "isMinimized",
                False,
            )
        )

        maximized = bool(
            getattr(
                window,
                "isMaximized",
                False,
            )
        )

        visible = bool(
            getattr(
                window,
                "visible",
                True,
            )
        )

        return WindowInfo(
            title=self._window_title(window),
            left=left,
            top=top,
            width=width,
            height=height,
            visible=visible,
            minimized=minimized,
            maximized=maximized,
        )

    # ==========================================================
    # LIST WINDOWS
    # ==========================================================

    def get_all_windows(
        self,
        include_empty: bool = False,
    ) -> list[Any]:

        self._check()

        windows = self._backend.getAllWindows()

        if include_empty:
            return list(windows)

        return [
            window
            for window in windows
            if self._window_title(window).strip()
        ]

    def list_windows(
        self,
        include_empty: bool = False,
    ) -> list[dict[str, Any]]:

        windows = self.get_all_windows(
            include_empty=include_empty
        )

        return [
            self._to_info(
                window
            ).to_dict()
            for window in windows
        ]

    # ==========================================================
    # FINDING WINDOWS
    # ==========================================================

    def find(
        self,
        query: str,
        exact: bool = False,
    ) -> list[Any]:

        self._check()

        if not query:
            return []

        return [
            window
            for window in self.get_all_windows()
            if self._matches(
                window,
                query,
                exact=exact,
            )
        ]

    def find_one(
        self,
        query: str,
        exact: bool = False,
    ) -> Any:

        return self._find_window(
            query,
            exact=exact,
        )

    def exists(
        self,
        query: str,
        exact: bool = False,
    ) -> bool:

        try:

            return bool(
                self.find(
                    query,
                    exact=exact,
                )
            )

        except Exception:

            return False

    # ==========================================================
    # ACTIVE WINDOW
    # ==========================================================

    def get_active_window(self) -> Optional[Any]:

        self._check()

        try:

            return self._backend.getActiveWindow()

        except Exception as exc:

            logger.warning(
                "Unable to determine active window: %s",
                exc,
            )

            return None

    def get_active_window_info(
        self,
    ) -> Optional[dict[str, Any]]:

        window = self.get_active_window()

        if window is None:
            return None

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # FOCUS
    # ==========================================================

    def focus(
        self,
        query: str,
        exact: bool = False,
        restore_if_minimized: bool = True,
    ) -> dict[str, Any]:

        self._check()

        window = self._find_window(
            query,
            exact=exact,
        )

        if (
            restore_if_minimized
            and bool(
                getattr(
                    window,
                    "isMinimized",
                    False,
                )
            )
        ):

            try:
                window.restore()
            except Exception:
                logger.exception(
                    "Failed to restore window."
                )

        window.activate()

        time.sleep(0.05)

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # MINIMIZE
    # ==========================================================

    def minimize(
        self,
        query: str,
        exact: bool = False,
    ) -> dict[str, Any]:

        self._check()

        window = self._find_window(
            query,
            exact=exact,
        )

        window.minimize()

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # MAXIMIZE
    # ==========================================================

    def maximize(
        self,
        query: str,
        exact: bool = False,
    ) -> dict[str, Any]:

        self._check()

        window = self._find_window(
            query,
            exact=exact,
        )

        window.maximize()

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # RESTORE
    # ==========================================================

    def restore(
        self,
        query: str,
        exact: bool = False,
    ) -> dict[str, Any]:

        self._check()

        window = self._find_window(
            query,
            exact=exact,
        )

        window.restore()

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(
        self,
        query: str,
        exact: bool = False,
    ) -> bool:

        self._check()

        window = self._find_window(
            query,
            exact=exact,
        )

        window.close()

        return True

    # ==========================================================
    # MOVE
    # ==========================================================

    def move(
        self,
        query: str,
        x: int,
        y: int,
        exact: bool = False,
    ) -> dict[str, Any]:

        self._check()

        window = self._find_window(
            query,
            exact=exact,
        )

        window.moveTo(
            int(x),
            int(y),
        )

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # RESIZE
    # ==========================================================

    def resize(
        self,
        query: str,
        width: int,
        height: int,
        exact: bool = False,
    ) -> dict[str, Any]:

        self._check()

        width = max(
            1,
            int(width),
        )

        height = max(
            1,
            int(height),
        )

        window = self._find_window(
            query,
            exact=exact,
        )

        window.resizeTo(
            width,
            height,
        )

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # MOVE + RESIZE
    # ==========================================================

    def set_geometry(
        self,
        query: str,
        x: int,
        y: int,
        width: int,
        height: int,
        exact: bool = False,
    ) -> dict[str, Any]:

        self._check()

        window = self._find_window(
            query,
            exact=exact,
        )

        window.moveTo(
            int(x),
            int(y),
        )

        window.resizeTo(
            max(1, int(width)),
            max(1, int(height)),
        )

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # POSITION / SIZE
    # ==========================================================

    def get_window_info(
        self,
        query: str,
        exact: bool = False,
    ) -> dict[str, Any]:

        window = self._find_window(
            query,
            exact=exact,
        )

        return self._to_info(
            window
        ).to_dict()

    def get_position(
        self,
        query: str,
        exact: bool = False,
    ) -> tuple[int, int]:

        info = self.get_window_info(
            query,
            exact=exact,
        )

        return (
            info["left"],
            info["top"],
        )

    def get_size(
        self,
        query: str,
        exact: bool = False,
    ) -> tuple[int, int]:

        info = self.get_window_info(
            query,
            exact=exact,
        )

        return (
            info["width"],
            info["height"],
        )

    # ==========================================================
    # WINDOW STATE
    # ==========================================================

    def is_minimized(
        self,
        query: str,
        exact: bool = False,
    ) -> bool:

        window = self._find_window(
            query,
            exact=exact,
        )

        return bool(
            getattr(
                window,
                "isMinimized",
                False,
            )
        )

    def is_maximized(
        self,
        query: str,
        exact: bool = False,
    ) -> bool:

        window = self._find_window(
            query,
            exact=exact,
        )

        return bool(
            getattr(
                window,
                "isMaximized",
                False,
            )
        )

    # ==========================================================
    # CENTERING
    # ==========================================================

    def center(
        self,
        query: str,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        exact: bool = False,
    ) -> dict[str, Any]:

        self._check()

        window = self._find_window(
            query,
            exact=exact,
        )

        if (
            screen_width is None
            or screen_height is None
        ):

            try:

                import pyautogui

                size = pyautogui.size()

                screen_width = int(
                    size.width
                )

                screen_height = int(
                    size.height
                )

            except Exception:

                raise RuntimeError(
                    "Screen dimensions unavailable."
                )

        width = int(
            getattr(
                window,
                "width",
                0,
            )
        )

        height = int(
            getattr(
                window,
                "height",
                0,
            )
        )

        x = (
            int(screen_width)
            - width
        ) // 2

        y = (
            int(screen_height)
            - height
        ) // 2

        window.moveTo(
            x,
            y,
        )

        return self._to_info(
            window
        ).to_dict()

    # ==========================================================
    # TILE HELPERS
    # ==========================================================

    def tile_left(
        self,
        query: str,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        exact: bool = False,
    ) -> dict[str, Any]:

        if screen_width is None or screen_height is None:
            try:
                import pyautogui

                size = pyautogui.size()

                screen_width = int(size.width)
                screen_height = int(size.height)

            except Exception as exc:
                raise RuntimeError(
                    "Screen dimensions unavailable."
                ) from exc

        return self.set_geometry(
            query,
            0,
            0,
            int(screen_width) // 2,
            int(screen_height),
            exact=exact,
        )

    def tile_right(
        self,
        query: str,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        exact: bool = False,
    ) -> dict[str, Any]:

        if screen_width is None or screen_height is None:
            try:
                import pyautogui

                size = pyautogui.size()

                screen_width = int(size.width)
                screen_height = int(size.height)

            except Exception as exc:
                raise RuntimeError(
                    "Screen dimensions unavailable."
                ) from exc

        half_width = (
            int(screen_width) // 2
        )

        return self.set_geometry(
            query,
            half_width,
            0,
            int(screen_width) - half_width,
            int(screen_height),
            exact=exact,
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        active = None

        if self._initialized:

            try:

                active_window = (
                    self.get_active_window()
                )

                if active_window is not None:
                    active = (
                        self._window_title(
                            active_window
                        )
                    )

            except Exception:
                active = None

        count = 0

        if self._initialized:

            try:
                count = len(
                    self.get_all_windows()
                )
            except Exception:
                count = 0

        return {
            "enabled": self.enabled,
            "available": self._initialized,
            "window_count": count,
            "active_window": active,
        }

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX window manager shut down."
        )


