"""
RENIX Screenshot Manager
========================

Screenshot capture and management for RENIX.

Features:
- Full-screen screenshots
- Region screenshots
- Monitor-specific screenshots
- Active-window screenshots
- Save screenshots to disk
- Timestamped filenames
- Screenshot metadata
- Clipboard copy support
- Capture history
- Image format handling
- Safe cleanup

Primary backend:
    PIL / Pillow

Optional:
    pyautogui
    mss
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotInfo:
    """Metadata describing a captured screenshot."""

    path: Optional[str]
    width: int
    height: int
    timestamp: float
    source: str
    format: str

    @property
    def datetime(self) -> str:
        return datetime.fromtimestamp(
            self.timestamp
        ).isoformat()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["datetime"] = self.datetime
        return data


class ScreenshotManager:
    """
    High-level screenshot manager for RENIX.

    Examples:

        manager = ScreenshotManager()

        manager.capture()

        manager.capture(
            save_path="data/screenshots/test.png"
        )

        manager.capture_region(
            100,
            100,
            800,
            600,
        )
    """

    def __init__(
        self,
        enabled: bool = True,
        screenshot_directory: Optional[str] = None,
        history_limit: int = 50,
    ) -> None:

        self.enabled = enabled

        self.history_limit = max(
            1,
            int(history_limit),
        )

        if screenshot_directory is None:

            screenshot_directory = os.path.join(
                "data",
                "screenshots",
            )

        self.screenshot_directory = (
            Path(screenshot_directory)
            .expanduser()
            .resolve()
        )

        self.screenshot_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history: list[
            ScreenshotInfo
        ] = []

        self._lock = threading.RLock()

        self._pyautogui = None
        self._mss = None
        self._pil_image = None

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:

        try:

            from PIL import Image

            self._pil_image = Image

        except Exception as exc:

            logger.warning(
                "Pillow unavailable: %s",
                exc,
            )

        try:

            import pyautogui

            self._pyautogui = pyautogui

        except Exception as exc:

            logger.debug(
                "PyAutoGUI unavailable: %s",
                exc,
            )

        try:

            import mss

            self._mss = mss

        except Exception as exc:

            logger.debug(
                "MSS unavailable: %s",
                exc,
            )

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX screenshot manager is disabled."
            )

        if (
            self._pyautogui is None
            and self._mss is None
        ):
            raise RuntimeError(
                "No screenshot backend available. "
                "Install Pillow + PyAutoGUI or MSS."
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

    # ==========================================================
    # DIRECTORY
    # ==========================================================

    def set_directory(
        self,
        directory: str,
    ) -> Path:

        path = (
            Path(directory)
            .expanduser()
            .resolve()
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.screenshot_directory = path

        return path

    def get_directory(self) -> str:

        return str(
            self.screenshot_directory
        )

    # ==========================================================
    # FILENAME
    # ==========================================================

    def generate_filename(
        self,
        prefix: str = "renix",
        extension: str = "png",
    ) -> str:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        extension = extension.lstrip(
            "."
        )

        safe_prefix = "".join(
            character
            if (
                character.isalnum()
                or character
                in ("-", "_")
            )
            else "_"
            for character in str(prefix)
        )

        return (
            f"{safe_prefix}_"
            f"{timestamp}."
            f"{extension}"
        )

    def generate_path(
        self,
        prefix: str = "renix",
        extension: str = "png",
    ) -> Path:

        return (
            self.screenshot_directory
            / self.generate_filename(
                prefix=prefix,
                extension=extension,
            )
        )

    # ==========================================================
    # RECORD HISTORY
    # ==========================================================

    def _record(
        self,
        info: ScreenshotInfo,
    ) -> None:

        with self._lock:

            self.history.append(info)

            if (
                len(self.history)
                > self.history_limit
            ):

                self.history = (
                    self.history[
                        -self.history_limit:
                    ]
                )

    # ==========================================================
    # SAVE IMAGE
    # ==========================================================

    def _save_image(
        self,
        image: Any,
        path: Path,
        image_format: Optional[str] = None,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if image_format is None:

            image_format = (
                path.suffix
                .lstrip(".")
                .upper()
            )

        if not image_format:
            image_format = "PNG"

        if image_format == "JPG":
            image_format = "JPEG"

        image.save(
            str(path),
            format=image_format,
        )

    # ==========================================================
    # FULL SCREEN
    # ==========================================================

    def capture(
        self,
        save_path: Optional[str] = None,
        prefix: str = "renix",
        image_format: str = "PNG",
    ) -> Any:
        """
        Capture the entire primary screen.
        """

        self._check()

        with self._lock:

            image = None

            # Prefer MSS when available.
            if self._mss is not None:

                try:

                    with self._mss.mss() as sct:

                        monitor = sct.monitors[1]

                        screenshot = (
                            sct.grab(monitor)
                        )

                        from PIL import Image

                        image = Image.frombytes(
                            "RGB",
                            screenshot.size,
                            screenshot.rgb,
                        )

                except Exception as exc:

                    logger.warning(
                        "MSS capture failed: %s",
                        exc,
                    )

            # Fallback to PyAutoGUI.
            if image is None:

                if self._pyautogui is None:
                    raise RuntimeError(
                        "No screenshot backend available."
                    )

                image = (
                    self._pyautogui.screenshot()
                )

            path = None

            if save_path is not None:

                path = (
                    Path(save_path)
                    .expanduser()
                    .resolve()
                )

            else:

                path = self.generate_path(
                    prefix=prefix,
                    extension=image_format.lower(),
                )

            self._save_image(
                image,
                path,
                image_format=image_format,
            )

            info = ScreenshotInfo(
                path=str(path),
                width=int(image.width),
                height=int(image.height),
                timestamp=time.time(),
                source="full_screen",
                format=image_format.upper(),
            )

            self._record(info)

            return image

    # ==========================================================
    # REGION
    # ==========================================================

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        save_path: Optional[str] = None,
        prefix: str = "renix_region",
        image_format: str = "PNG",
    ) -> Any:
        """
        Capture a rectangular screen region.
        """

        self._check()

        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        if width <= 0 or height <= 0:
            raise ValueError(
                "Width and height must be positive."
            )

        with self._lock:

            image = None

            if self._pyautogui is not None:

                try:

                    image = (
                        self._pyautogui.screenshot(
                            region=(
                                x,
                                y,
                                width,
                                height,
                            )
                        )
                    )

                except Exception as exc:

                    logger.warning(
                        "PyAutoGUI region capture failed: %s",
                        exc,
                    )

            if image is None:

                if self._mss is None:
                    raise RuntimeError(
                        "No region screenshot backend available."
                    )

                with self._mss.mss() as sct:

                    monitor = {
                        "left": x,
                        "top": y,
                        "width": width,
                        "height": height,
                    }

                    screenshot = sct.grab(
                        monitor
                    )

                    from PIL import Image

                    image = Image.frombytes(
                        "RGB",
                        screenshot.size,
                        screenshot.rgb,
                    )

            if save_path is not None:

                path = (
                    Path(save_path)
                    .expanduser()
                    .resolve()
                )

            else:

                path = self.generate_path(
                    prefix=prefix,
                    extension=image_format.lower(),
                )

            self._save_image(
                image,
                path,
                image_format=image_format,
            )

            info = ScreenshotInfo(
                path=str(path),
                width=int(image.width),
                height=int(image.height),
                timestamp=time.time(),
                source="region",
                format=image_format.upper(),
            )

            self._record(info)

            return image

    # ==========================================================
    # MONITOR CAPTURE
    # ==========================================================

    def get_monitors(self) -> list[dict[str, int]]:

        self._check()

        if self._mss is None:

            if self._pyautogui is None:
                return []

            size = self._pyautogui.size()

            return [
                {
                    "index": 1,
                    "left": 0,
                    "top": 0,
                    "width": int(size.width),
                    "height": int(size.height),
                }
            ]

        with self._mss.mss() as sct:

            monitors = []

            for index, monitor in enumerate(
                sct.monitors[1:],
                start=1,
            ):

                monitors.append(
                    {
                        "index": index,
                        "left": int(
                            monitor["left"]
                        ),
                        "top": int(
                            monitor["top"]
                        ),
                        "width": int(
                            monitor["width"]
                        ),
                        "height": int(
                            monitor["height"]
                        ),
                    }
                )

            return monitors

    def capture_monitor(
        self,
        monitor_index: int = 1,
        save_path: Optional[str] = None,
        prefix: str = "renix_monitor",
        image_format: str = "PNG",
    ) -> Any:

        self._check()

        monitor_index = int(
            monitor_index
        )

        if self._mss is None:

            if monitor_index != 1:
                raise RuntimeError(
                    "Multi-monitor capture requires MSS."
                )

            return self.capture(
                save_path=save_path,
                prefix=prefix,
                image_format=image_format,
            )

        with self._mss.mss() as sct:

            monitors = sct.monitors

            if not (
                1 <= monitor_index
                < len(monitors)
            ):
                raise IndexError(
                    "Invalid monitor index."
                )

            monitor = monitors[
                monitor_index
            ]

            screenshot = sct.grab(
                monitor
            )

            from PIL import Image

            image = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.rgb,
            )

        if save_path is not None:

            path = (
                Path(save_path)
                .expanduser()
                .resolve()
            )

        else:

            path = self.generate_path(
                prefix=prefix,
                extension=image_format.lower(),
            )

        self._save_image(
            image,
            path,
            image_format=image_format,
        )

        info = ScreenshotInfo(
            path=str(path),
            width=int(image.width),
            height=int(image.height),
            timestamp=time.time(),
            source=f"monitor_{monitor_index}",
            format=image_format.upper(),
        )

        self._record(info)

        return image

    # ==========================================================
    # ACTIVE WINDOW
    # ==========================================================

    def capture_active_window(
        self,
        save_path: Optional[str] = None,
        prefix: str = "renix_window",
        image_format: str = "PNG",
    ) -> Any:
        """
        Capture the currently active window.

        Uses the RENIX WindowManager to obtain geometry.
        """

        try:

            from .windows import (
                WindowManager
            )

            manager = WindowManager()

            info = manager.get_active_window_info()

        except Exception as exc:

            raise RuntimeError(
                "Unable to determine active window."
            ) from exc

        if not info:
            raise RuntimeError(
                "No active window found."
            )

        return self.capture_region(
            x=info["left"],
            y=info["top"],
            width=info["width"],
            height=info["height"],
            save_path=save_path,
            prefix=prefix,
            image_format=image_format,
        )

    # ==========================================================
    # CLIPBOARD
    # ==========================================================

    def copy_to_clipboard(
        self,
        image: Any,
    ) -> bool:
        """
        Copy a screenshot image to the Windows clipboard.

        Uses Pillow + Windows clipboard APIs.
        """

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        try:

            import io
            import win32clipboard
            from PIL import Image

            if image.mode != "RGB":
                image = image.convert("RGB")

            output = io.BytesIO()

            image.save(
                output,
                "BMP",
            )

            data = output.getvalue()[
                14:
            ]

            win32clipboard.OpenClipboard()

            try:

                win32clipboard.EmptyClipboard()

                win32clipboard.SetClipboardData(
                    win32clipboard.CF_DIB,
                    data,
                )

            finally:

                win32clipboard.CloseClipboard()

            return True

        except ImportError as exc:

            raise RuntimeError(
                "pywin32 is required for "
                "image clipboard support."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unable to copy screenshot to clipboard."
            )

            return False

    # ==========================================================
    # CAPTURE + CLIPBOARD
    # ==========================================================

    def capture_to_clipboard(self) -> Any:

        image = self.capture()

        self.copy_to_clipboard(
            image
        )

        return image

    # ==========================================================
    # HISTORY
    # ==========================================================

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:

        items = self.history

        if limit is not None:

            limit = max(
                0,
                int(limit),
            )

            items = items[-limit:]

        return [
            item.to_dict()
            for item in reversed(
                items
            )
        ]

    def clear_history(
        self,
        delete_files: bool = False,
    ) -> int:

        removed = 0

        if delete_files:

            for item in self.history:

                if not item.path:
                    continue

                try:

                    path = Path(
                        item.path
                    )

                    if path.exists():
                        path.unlink()
                        removed += 1

                except Exception as exc:

                    logger.warning(
                        "Unable to delete screenshot %s: %s",
                        item.path,
                        exc,
                    )

        self.history.clear()

        return removed

    # ==========================================================
    # FILE OPERATIONS
    # ==========================================================

    def delete_screenshot(
        self,
        path: str,
    ) -> bool:

        file_path = (
            Path(path)
            .expanduser()
            .resolve()
        )

        if not file_path.exists():
            return False

        try:

            file_path.unlink()

            return True

        except Exception as exc:

            logger.warning(
                "Unable to delete screenshot: %s",
                exc,
            )

            return False

    def list_screenshots(self) -> list[str]:

        if not self.screenshot_directory.exists():
            return []

        supported = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp",
        }

        files = [
            path
            for path in
            self.screenshot_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in supported
            )
        ]

        files.sort(
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

        return [
            str(path)
            for path in files
        ]

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "pyautogui_available": (
                self._pyautogui is not None
            ),
            "mss_available": (
                self._mss is not None
            ),
            "pillow_available": (
                self._pil_image is not None
            ),
            "directory": str(
                self.screenshot_directory
            ),
            "history_count": len(
                self.history
            ),
        }

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        self.history.clear()

        self._pyautogui = None
        self._mss = None
        self._pil_image = None

        logger.info(
            "RENIX screenshot manager shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_manager: Optional[
    ScreenshotManager
] = None


def get_screenshot_manager() -> ScreenshotManager:
    """Return the shared RENIX screenshot manager."""

    global _default_manager

    if _default_manager is None:

        _default_manager = (
            ScreenshotManager()
        )

    return _default_manager


