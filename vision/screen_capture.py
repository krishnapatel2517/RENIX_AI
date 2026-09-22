"""
RENIX Vision — Screen Capture Engine

Captures the Windows desktop/screen for RENIX vision processing.

Features:
- Full-screen capture
- Specific monitor capture
- Region capture
- Window-friendly frame capture
- Multi-monitor support
- Screenshot saving
- OpenCV-compatible BGR frames
- RGB frame support
- Continuous capture
- FPS tracking
- Thread-safe access
- Graceful dependency handling

This module ONLY captures screen pixels.
It does not interpret, click, type, or execute anything.
Screen understanding belongs to screen_understanding.py.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mss
except ImportError:
    mss = None


logger = logging.getLogger(
    "RENIX.vision.screen_capture"
)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class CaptureRegion:
    """
    Rectangle used for screen capture.
    """

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    def is_valid(self) -> bool:
        return (
            self.width > 0
            and self.height > 0
        )

    def as_mss_monitor(
        self,
    ) -> dict[str, int]:
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class MonitorInfo:
    """
    Information about one physical/display monitor.
    """

    index: int

    name: str

    x: int

    y: int

    width: int

    height: int

    is_primary: bool = False

    scale: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def region(self) -> CaptureRegion:
        return CaptureRegion(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "is_primary": self.is_primary,
            "scale": self.scale,
            "metadata": dict(self.metadata),
        }


@dataclass
class ScreenFrame:
    """
    One captured screen frame.
    """

    image: Any

    timestamp: float

    frame_number: int

    width: int

    height: int

    monitor_index: Optional[int] = None

    region: Optional[CaptureRegion] = None

    fps: float = 0.0

    color_format: str = "BGR"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def shape(self) -> tuple[int, ...]:
        try:
            return tuple(
                self.image.shape
            )
        except Exception:
            return ()

    def to_dict(
        self,
        include_image: bool = False,
    ) -> dict[str, Any]:

        result = {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "width": self.width,
            "height": self.height,
            "monitor_index": self.monitor_index,
            "region": (
                self.region.to_dict()
                if self.region
                else None
            ),
            "fps": self.fps,
            "color_format": self.color_format,
            "metadata": dict(
                self.metadata
            ),
        }

        if include_image:
            result["image"] = self.image

        return result


@dataclass
class ScreenCaptureConfig:
    """
    Screen capture configuration.
    """

    monitor_index: int = 1

    target_fps: float = 30.0

    color_format: str = "BGR"

    include_cursor: bool = False

    max_width: Optional[int] = None

    max_height: Optional[int] = None

    region: Optional[CaptureRegion] = None

    continuous: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# SCREEN CAPTURE ENGINE
# ============================================================================


class ScreenCapture:
    """
    Main RENIX screen-capture engine.

    Example:

        capture = ScreenCapture()

        frame = capture.capture()

        if frame:
            image = frame.image
    """

    def __init__(
        self,
        config: Optional[
            ScreenCaptureConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or ScreenCaptureConfig()
        )

        self._lock = threading.RLock()

        self._sct = None

        self._running = False

        self._frame_number = 0

        self._last_frame: Optional[
            ScreenFrame
        ] = None

        self._last_capture_time = 0.0

        self._fps = 0.0

        self._capture_thread: Optional[
            threading.Thread
        ] = None

        self._stop_event = (
            threading.Event()
        )

        self._callbacks: list[
            Callable[
                [ScreenFrame],
                None,
            ]
        ] = []

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:

        if mss is None:

            logger.warning(
                "mss is not installed. "
                "Screen capture is unavailable."
            )

            return

        try:

            self._sct = mss.mss()

        except Exception as exc:

            logger.exception(
                "Failed to initialize screen capture: %s",
                exc,
            )

            self._sct = None

    # ========================================================================
    # AVAILABILITY
    # ========================================================================

    def is_available(self) -> bool:

        return self._sct is not None

    # ========================================================================
    # MONITORS
    # ========================================================================

    def get_monitors(
        self,
    ) -> list[MonitorInfo]:

        if self._sct is None:
            return []

        monitors: list[
            MonitorInfo
        ] = []

        try:

            raw_monitors = (
                self._sct.monitors
            )

            # mss index 0 = virtual combined
            # monitor, so expose physical
            # monitors starting at 1.

            for index, monitor in enumerate(
                raw_monitors[1:],
                start=1,
            ):

                monitors.append(
                    MonitorInfo(
                        index=index,
                        name=(
                            f"Monitor {index}"
                        ),
                        x=int(
                            monitor.get(
                                "left",
                                0,
                            )
                        ),
                        y=int(
                            monitor.get(
                                "top",
                                0,
                            )
                        ),
                        width=int(
                            monitor.get(
                                "width",
                                0,
                            )
                        ),
                        height=int(
                            monitor.get(
                                "height",
                                0,
                            )
                        ),
                        is_primary=(
                            index == 1
                        ),
                        scale=1.0,
                    )
                )

        except Exception as exc:

            logger.exception(
                "Failed to enumerate monitors: %s",
                exc,
            )

        return monitors

    def get_monitor(
        self,
        index: Optional[int] = None,
    ) -> Optional[MonitorInfo]:

        if index is None:
            index = self.config.monitor_index

        monitors = self.get_monitors()

        for monitor in monitors:

            if monitor.index == index:
                return monitor

        return None

    # ========================================================================
    # CAPTURE
    # ========================================================================

    def capture(
        self,
        monitor_index: Optional[int] = None,
        region: Optional[
            CaptureRegion
        ] = None,
    ) -> Optional[ScreenFrame]:

        if self._sct is None:

            logger.error(
                "Screen capture backend "
                "is unavailable."
            )

            return None

        if monitor_index is None:
            monitor_index = (
                self.config.monitor_index
            )

        started = time.perf_counter()

        try:

            monitor = self._resolve_capture_region(
                monitor_index=monitor_index,
                region=region,
            )

            if monitor is None:
                return None

            raw = self._sct.grab(
                monitor
            )

            image = self._convert_frame(
                raw
            )

            if image is None:
                return None

            image = self._resize_if_needed(
                image
            )

            timestamp = time.time()

            with self._lock:

                self._frame_number += 1

                current_number = (
                    self._frame_number
                )

                self._update_fps(
                    timestamp
                )

            height, width = (
                image.shape[:2]
            )

            capture_region = (
                CaptureRegion(
                    x=int(
                        monitor["left"]
                    ),
                    y=int(
                        monitor["top"]
                    ),
                    width=int(
                        monitor["width"]
                    ),
                    height=int(
                        monitor["height"]
                    ),
                )
            )

            frame = ScreenFrame(
                image=image,
                timestamp=timestamp,
                frame_number=current_number,
                width=int(width),
                height=int(height),
                monitor_index=monitor_index,
                region=capture_region,
                fps=self._fps,
                color_format=(
                    self.config.color_format
                ),
                metadata={
                    "capture_time": (
                        time.perf_counter()
                        - started
                    ),
                },
            )

            with self._lock:

                self._last_frame = frame

            return frame

        except Exception as exc:

            logger.exception(
                "Screen capture failed: %s",
                exc
            )

            return None

    # ========================================================================
    # REGION RESOLUTION
    # ========================================================================

    def _resolve_capture_region(
        self,
        monitor_index: int,
        region: Optional[
            CaptureRegion
        ],
    ) -> Optional[
        dict[str, int]
    ]:

        if region is not None:

            if not region.is_valid():

                logger.error(
                    "Invalid capture region."
                )

                return None

            return (
                region.as_mss_monitor()
            )

        if (
            self.config.region is not None
        ):

            configured = (
                self.config.region
            )

            if not configured.is_valid():

                logger.error(
                    "Configured capture "
                    "region is invalid."
                )

                return None

            return (
                configured.as_mss_monitor()
            )

        if monitor_index == 0:

            try:

                return dict(
                    self._sct.monitors[0]
                )

            except Exception:
                return None

        try:

            monitors = (
                self._sct.monitors
            )

            if (
                monitor_index < 0
                or monitor_index
                >= len(monitors)
            ):

                logger.error(
                    "Monitor index %s "
                    "does not exist.",
                    monitor_index,
                )

                return None

            return dict(
                monitors[
                    monitor_index
                ]
            )

        except Exception as exc:

            logger.exception(
                "Failed to resolve monitor: %s",
                exc,
            )

            return None

    # ========================================================================
    # FRAME CONVERSION
    # ========================================================================

    def _convert_frame(
        self,
        raw: Any,
    ) -> Any:

        if np is None:

            raise RuntimeError(
                "NumPy is required "
                "for screen capture."
            )

        image = np.asarray(
            raw,
            dtype=np.uint8,
        )

        # mss returns BGRA.
        if image.ndim == 3:

            channels = image.shape[2]

            if channels == 4:

                if (
                    self.config.color_format
                    .upper()
                    == "RGB"
                ):

                    image = image[
                        :, :, :3
                    ][:, :, ::-1]

                elif (
                    self.config.color_format
                    .upper()
                    == "BGR"
                ):

                    image = image[
                        :, :, :3
                    ]

                elif (
                    self.config.color_format
                    .upper()
                    == "RGBA"
                ):

                    image = (
                        image[:, :, ::-1]
                    )

                elif (
                    self.config.color_format
                    .upper()
                    == "BGRA"
                ):

                    pass

                else:

                    image = image[
                        :, :, :3
                    ]

            elif channels == 3:

                if (
                    self.config.color_format
                    .upper()
                    == "RGB"
                ):

                    image = image[
                        :, :, ::-1
                    ]

        return image

    # ========================================================================
    # RESIZING
    # ========================================================================

    def _resize_if_needed(
        self,
        image: Any,
    ) -> Any:

        if cv2 is None:
            return image

        try:

            height, width = (
                image.shape[:2]
            )

            max_width = (
                self.config.max_width
            )

            max_height = (
                self.config.max_height
            )

            if (
                max_width is None
                and max_height is None
            ):
                return image

            scale = 1.0

            if (
                max_width is not None
                and width > max_width
            ):

                scale = min(
                    scale,
                    max_width / width,
                )

            if (
                max_height is not None
                and height > max_height
            ):

                scale = min(
                    scale,
                    max_height / height,
                )

            if scale >= 1.0:
                return image

            new_width = max(
                1,
                int(width * scale),
            )

            new_height = max(
                1,
                int(height * scale),
            )

            return cv2.resize(
                image,
                (
                    new_width,
                    new_height,
                ),
                interpolation=(
                    cv2.INTER_AREA
                ),
            )

        except Exception as exc:

            logger.debug(
                "Frame resizing failed: %s",
                exc,
            )

            return image

    # ========================================================================
    # FPS
    # ========================================================================

    def _update_fps(
        self,
        timestamp: float,
    ) -> None:

        if (
            self._last_capture_time
            <= 0
        ):

            self._last_capture_time = (
                timestamp
            )

            self._fps = 0.0

            return

        delta = (
            timestamp
            - self._last_capture_time
        )

        self._last_capture_time = (
            timestamp
        )

        if delta > 0:

            instant_fps = (
                1.0 / delta
            )

            if self._fps <= 0:

                self._fps = instant_fps

            else:

                # Exponential smoothing.
                self._fps = (
                    self._fps * 0.90
                    + instant_fps * 0.10
                )

    def get_fps(self) -> float:

        with self._lock:
            return float(
                self._fps
            )

    # ========================================================================
    # LAST FRAME
    # ========================================================================

    def get_last_frame(
        self,
    ) -> Optional[ScreenFrame]:

        with self._lock:
            return self._last_frame

    # ========================================================================
    # SCREENSHOT
    # ========================================================================

    def save_screenshot(
        self,
        path: str | Path,
        monitor_index: Optional[
            int
        ] = None,
        region: Optional[
            CaptureRegion
        ] = None,
    ) -> bool:

        if cv2 is None:

            logger.error(
                "OpenCV is required "
                "to save screenshots."
            )

            return False

        frame = self.capture(
            monitor_index=monitor_index,
            region=region,
        )

        if frame is None:
            return False

        destination = Path(
            path
        ).expanduser()

        try:

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            image = frame.image

            if (
                frame.color_format.upper()
                == "RGB"
            ):

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR,
                )

            success = cv2.imwrite(
                str(destination),
                image,
            )

            if not success:

                logger.error(
                    "Failed to write screenshot: %s",
                    destination,
                )

            return bool(success)

        except Exception as exc:

            logger.exception(
                "Screenshot save failed: %s",
                exc,
            )

            return False

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def add_callback(
        self,
        callback: Callable[
            [ScreenFrame],
            None,
        ],
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "callback must be callable."
            )

        with self._lock:

            if callback not in self._callbacks:

                self._callbacks.append(
                    callback
                )

    def remove_callback(
        self,
        callback: Callable[
            [ScreenFrame],
            None,
        ],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def clear_callbacks(
        self,
    ) -> None:

        with self._lock:
            self._callbacks.clear()

    def _notify_callbacks(
        self,
        frame: ScreenFrame,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks
            )

        for callback in callbacks:

            try:

                callback(frame)

            except Exception:

                logger.exception(
                    "Screen capture callback failed."
                )

    # ========================================================================
    # CONTINUOUS CAPTURE
    # ========================================================================

    def start(
        self,
        monitor_index: Optional[
            int
        ] = None,
        region: Optional[
            CaptureRegion
        ] = None,
        target_fps: Optional[
            float
        ] = None,
    ) -> bool:

        if not self.is_available():

            logger.error(
                "Cannot start screen capture: "
                "backend unavailable."
            )

            return False

        with self._lock:

            if self._running:
                return True

            self._running = True

            self._stop_event.clear()

        fps = (
            target_fps
            if target_fps is not None
            else self.config.target_fps
        )

        if fps <= 0:
            fps = 30.0

        self._capture_thread = (
            threading.Thread(
                target=self._capture_loop,
                args=(
                    monitor_index,
                    region,
                    fps,
                ),
                name="RENIX-ScreenCapture",
                daemon=True,
            )
        )

        self._capture_thread.start()

        return True

    def _capture_loop(
        self,
        monitor_index: Optional[
            int
        ],
        region: Optional[
            CaptureRegion
        ],
        target_fps: float,
    ) -> None:

        interval = (
            1.0 / target_fps
        )

        next_capture = (
            time.perf_counter()
        )

        while not self._stop_event.is_set():

            loop_started = (
                time.perf_counter()
            )

            frame = self.capture(
                monitor_index=monitor_index,
                region=region,
            )

            if frame is not None:

                self._notify_callbacks(
                    frame
                )

            next_capture += interval

            sleep_time = (
                next_capture
                - time.perf_counter()
            )

            if sleep_time > 0:

                self._stop_event.wait(
                    sleep_time
                )

            else:

                # We fell behind. Reset the
                # schedule instead of building
                # an increasingly large delay.
                next_capture = (
                    time.perf_counter()
                )

            # Prevent accidental busy loops.
            if (
                time.perf_counter()
                - loop_started
                > interval * 4
            ):

                next_capture = (
                    time.perf_counter()
                    + interval
                )

        with self._lock:
            self._running = False

    def stop(
        self,
        timeout: float = 2.0,
    ) -> None:

        with self._lock:

            if not self._running:

                return

            self._stop_event.set()

            thread = (
                self._capture_thread
            )

        if (
            thread is not None
            and thread.is_alive()
        ):

            thread.join(
                timeout=max(
                    0.0,
                    timeout,
                )
            )

        with self._lock:

            self._running = False

            self._capture_thread = None

    def is_running(self) -> bool:

        with self._lock:
            return bool(
                self._running
            )

    # ========================================================================
    # FRAME STREAM
    # ========================================================================

    def frames(
        self,
        monitor_index: Optional[
            int
        ] = None,
        region: Optional[
            CaptureRegion
        ] = None,
        target_fps: Optional[
            float
        ] = None,
    ) -> Iterator[
        ScreenFrame
    ]:

        fps = (
            target_fps
            if target_fps is not None
            else self.config.target_fps
        )

        if fps <= 0:
            fps = 30.0

        interval = (
            1.0 / fps
        )

        while True:

            started = (
                time.perf_counter()
            )

            frame = self.capture(
                monitor_index=monitor_index,
                region=region,
            )

            if frame is not None:
                yield frame

            elapsed = (
                time.perf_counter()
                - started
            )

            remaining = (
                interval - elapsed
            )

            if remaining > 0:
                time.sleep(
                    remaining
                )

    # ========================================================================
    # REGION HELPERS
    # ========================================================================

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Optional[ScreenFrame]:

        region = CaptureRegion(
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
        )

        return self.capture(
            region=region
        )

    def capture_monitor(
        self,
        monitor_index: int,
    ) -> Optional[ScreenFrame]:

        return self.capture(
            monitor_index=monitor_index
        )

    def capture_all_monitors(
        self,
    ) -> Optional[ScreenFrame]:

        return self.capture(
            monitor_index=0
        )

    # ========================================================================
    # CENTER / HALF SCREEN HELPERS
    # ========================================================================

    def capture_center(
        self,
        width: int,
        height: int,
        monitor_index: Optional[
            int
        ] = None,
    ) -> Optional[ScreenFrame]:

        monitor = self.get_monitor(
            monitor_index
        )

        if monitor is None:
            return None

        width = min(
            width,
            monitor.width,
        )

        height = min(
            height,
            monitor.height,
        )

        x = (
            monitor.x
            + (
                monitor.width
                - width
            )
            // 2
        )

        y = (
            monitor.y
            + (
                monitor.height
                - height
            )
            // 2
        )

        return self.capture_region(
            x=x,
            y=y,
            width=width,
            height=height,
        )

    # ========================================================================
    # RESOURCE CLEANUP
    # ========================================================================

    def close(self) -> None:

        self.stop()

        with self._lock:

            self._callbacks.clear()

            self._last_frame = None

            sct = self._sct

            self._sct = None

        if sct is not None:

            try:

                sct.close()

            except Exception:

                logger.debug(
                    "Screen capture backend "
                    "close failed.",
                    exc_info=True,
                )

    def __enter__(
        self,
    ) -> "ScreenCapture":

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.close()


# ============================================================================
# DEFAULT INSTANCE
# ============================================================================


_default_capture: Optional[
    ScreenCapture
] = None

_default_lock = (
    threading.RLock()
)


def get_screen_capture(
) -> ScreenCapture:

    global _default_capture

    with _default_lock:

        if _default_capture is None:

            _default_capture = (
                ScreenCapture()
            )

        return _default_capture


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def capture_screen(
    monitor_index: int = 1,
) -> Optional[ScreenFrame]:

    return get_screen_capture().capture(
        monitor_index=monitor_index
    )


def capture_region(
    x: int,
    y: int,
    width: int,
    height: int,
) -> Optional[ScreenFrame]:

    return get_screen_capture().capture_region(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def capture_all_monitors(
) -> Optional[ScreenFrame]:

    return (
        get_screen_capture()
        .capture_all_monitors()
    )


def save_screenshot(
    path: str | Path,
    monitor_index: int = 1,
) -> bool:

    return (
        get_screen_capture()
        .save_screenshot(
            path=path,
            monitor_index=monitor_index,
        )
    )


def get_monitors() -> list[MonitorInfo]:

    return (
        get_screen_capture()
        .get_monitors()
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "CaptureRegion",
    "MonitorInfo",
    "ScreenFrame",
    "ScreenCaptureConfig",
    "ScreenCapture",
    "get_screen_capture",
    "capture_screen",
    "capture_region",
    "capture_all_monitors",
    "save_screenshot",
    "get_monitors",
]


