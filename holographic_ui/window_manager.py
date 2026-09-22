"""
RENIX Holographic UI - Window Manager
=====================================

Manages holographic windows inside the RENIX interface.

Responsibilities:
    - Create and remove UI windows
    - Move and resize windows
    - Focus / minimize / maximize / restore
    - Z-order management
    - Window visibility
    - Spatial positioning
    - Window state
    - Hit testing
    - Gesture-friendly manipulation
    - Window events

This module does NOT render windows.
The renderer/engine is responsible for drawing them.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.holographic_ui.window_manager"
)


# ============================================================================
# ENUMS
# ============================================================================


class WindowState(str, Enum):
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    FULLSCREEN = "fullscreen"
    HIDDEN = "hidden"


class WindowLayer(str, Enum):
    BACKGROUND = "background"
    NORMAL = "normal"
    FLOATING = "floating"
    POPUP = "popup"
    OVERLAY = "overlay"
    SYSTEM = "system"


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class WindowPosition:
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0

    def copy(self) -> "WindowPosition":
        return WindowPosition(
            self.x,
            self.y,
            self.z,
        )


@dataclass
class WindowSize:
    width: float = 0.30
    height: float = 0.25

    min_width: float = 0.05
    min_height: float = 0.05

    max_width: float = 1.0
    max_height: float = 1.0

    def copy(self) -> "WindowSize":
        return WindowSize(
            width=self.width,
            height=self.height,
            min_width=self.min_width,
            min_height=self.min_height,
            max_width=self.max_width,
            max_height=self.max_height,
        )


@dataclass
class HolographicWindow:
    """
    Represents one holographic UI window.
    """

    id: str

    title: str

    app_id: str = ""

    position: WindowPosition = field(
        default_factory=WindowPosition
    )

    size: WindowSize = field(
        default_factory=WindowSize
    )

    state: WindowState = WindowState.NORMAL

    layer: WindowLayer = WindowLayer.NORMAL

    visible: bool = True

    focused: bool = False

    draggable: bool = True

    resizable: bool = True

    closable: bool = True

    minimizable: bool = True

    maximizable: bool = True

    always_on_top: bool = False

    opacity: float = 1.0

    rotation: float = 0.0

    scale: float = 1.0

    z_index: int = 0

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def touch(self) -> None:
        self.updated_at = time.time()

    def contains_point(
        self,
        x: float,
        y: float,
    ) -> bool:

        half_width = (
            self.size.width * self.scale
        ) / 2.0

        half_height = (
            self.size.height * self.scale
        ) / 2.0

        return (
            self.position.x - half_width
            <= x
            <= self.position.x + half_width
            and
            self.position.y - half_height
            <= y
            <= self.position.y + half_height
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "id": self.id,
            "title": self.title,
            "app_id": self.app_id,
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z,
            },
            "size": {
                "width": self.size.width,
                "height": self.size.height,
            },
            "state": self.state.value,
            "layer": self.layer.value,
            "visible": self.visible,
            "focused": self.focused,
            "draggable": self.draggable,
            "resizable": self.resizable,
            "closable": self.closable,
            "minimizable": self.minimizable,
            "maximizable": self.maximizable,
            "always_on_top": self.always_on_top,
            "opacity": self.opacity,
            "rotation": self.rotation,
            "scale": self.scale,
            "z_index": self.z_index,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# WINDOW MANAGER
# ============================================================================


class WindowManager:
    """
    Central manager for RENIX holographic windows.
    """

    def __init__(
        self,
        *,
        screen_width: float = 1.0,
        screen_height: float = 1.0,
    ) -> None:

        self.screen_width = max(
            0.01,
            float(screen_width),
        )

        self.screen_height = max(
            0.01,
            float(screen_height),
        )

        self._windows: dict[
            str,
            HolographicWindow,
        ] = {}

        self._z_order: list[str] = []

        self._focused_window: Optional[str] = None

        self._lock = threading.RLock()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._running = False

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            self._running = True

    def stop(self) -> None:

        with self._lock:

            self._running = False

    def update(
        self,
        delta_time: float,
    ) -> None:

        # Window manager is primarily event-driven.
        # This method exists so RENIXApp can treat all
        # UI components uniformly.
        return None

    # ========================================================================
    # WINDOW CREATION
    # ========================================================================

    def create_window(
        self,
        title: str,
        *,
        app_id: str = "",
        x: float = 0.5,
        y: float = 0.5,
        z: float = 0.0,
        width: float = 0.30,
        height: float = 0.25,
        state: WindowState = WindowState.NORMAL,
        layer: WindowLayer = WindowLayer.NORMAL,
        visible: bool = True,
        draggable: bool = True,
        resizable: bool = True,
        closable: bool = True,
        minimizable: bool = True,
        maximizable: bool = True,
        always_on_top: bool = False,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> HolographicWindow:

        if not title:

            raise ValueError(
                "Window title cannot be empty"
            )

        window_id = uuid.uuid4().hex

        window = HolographicWindow(
            id=window_id,
            title=title,
            app_id=app_id,
            position=WindowPosition(
                self._clamp_x(x),
                self._clamp_y(y),
                float(z),
            ),
            size=WindowSize(
                width=self._clamp_width(
                    width
                ),
                height=self._clamp_height(
                    height
                ),
            ),
            state=state,
            layer=layer,
            visible=visible,
            draggable=draggable,
            resizable=resizable,
            closable=closable,
            minimizable=minimizable,
            maximizable=maximizable,
            always_on_top=always_on_top,
            metadata=dict(metadata or {}),
        )

        with self._lock:

            self._windows[window_id] = window

            self._z_order.append(
                window_id
            )

            self._recalculate_z_order()

            if visible:

                self.focus_window(
                    window_id
                )

        self._emit(
            "window_created",
            window,
        )

        return window

    # ========================================================================
    # WINDOW REMOVAL
    # ========================================================================

    def close_window(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            if not window.closable:

                return False

            self._windows.pop(
                window_id,
                None,
            )

            if window_id in self._z_order:

                self._z_order.remove(
                    window_id
                )

            was_focused = (
                self._focused_window
                == window_id
            )

            if was_focused:

                self._focused_window = None

                self._focus_topmost_visible()

            self._recalculate_z_order()

        self._emit(
            "window_closed",
            window,
        )

        return True

    # ========================================================================
    # WINDOW LOOKUP
    # ========================================================================

    def get_window(
        self,
        window_id: str,
    ) -> Optional[HolographicWindow]:

        with self._lock:

            return self._windows.get(
                window_id
            )

    def get_windows(
        self,
        *,
        visible_only: bool = False,
    ) -> list[HolographicWindow]:

        with self._lock:

            windows = list(
                self._windows.values()
            )

            if visible_only:

                windows = [
                    window
                    for window in windows
                    if window.visible
                    and window.state
                    != WindowState.HIDDEN
                    and window.state
                    != WindowState.MINIMIZED
                ]

            return sorted(
                windows,
                key=lambda item: item.z_index,
            )

    def get_focused_window(
        self,
    ) -> Optional[HolographicWindow]:

        with self._lock:

            if self._focused_window is None:

                return None

            return self._windows.get(
                self._focused_window
            )

    # ========================================================================
    # FOCUS
    # ========================================================================

    def focus_window(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            if not window.visible:

                return False

            if window.state in (
                WindowState.HIDDEN,
                WindowState.MINIMIZED,
            ):

                return False

            previous = self._focused_window

            for item in self._windows.values():

                item.focused = False

            window.focused = True

            self._focused_window = window_id

            if window_id in self._z_order:

                self._z_order.remove(
                    window_id
                )

            self._z_order.append(
                window_id
            )

            self._recalculate_z_order()

            window.touch()

        if previous != window_id:

            self._emit(
                "window_focused",
                window,
            )

        return True

    # ========================================================================
    # POSITION
    # ========================================================================

    def move_window(
        self,
        window_id: str,
        x: float,
        y: float,
        *,
        z: Optional[float] = None,
        clamp: bool = True,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            if not window.draggable:

                return False

            old_position = (
                window.position.copy()
            )

            if clamp:

                x = self._clamp_x(x)

                y = self._clamp_y(y)

            window.position.x = float(x)

            window.position.y = float(y)

            if z is not None:

                window.position.z = float(z)

            window.touch()

        self._emit(
            "window_moved",
            window,
            old_position,
        )

        return True

    def move_window_by(
        self,
        window_id: str,
        dx: float,
        dy: float,
        dz: float = 0.0,
    ) -> bool:

        window = self.get_window(
            window_id
        )

        if window is None:

            return False

        return self.move_window(
            window_id,
            window.position.x + dx,
            window.position.y + dy,
            z=window.position.z + dz,
        )

    # ========================================================================
    # RESIZE
    # ========================================================================

    def resize_window(
        self,
        window_id: str,
        width: float,
        height: float,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            if not window.resizable:

                return False

            old_size = (
                window.size.copy()
            )

            window.size.width = (
                self._clamp_width_for_window(
                    window,
                    width,
                )
            )

            window.size.height = (
                self._clamp_height_for_window(
                    window,
                    height,
                )
            )

            window.touch()

        self._emit(
            "window_resized",
            window,
            old_size,
        )

        return True

    # ========================================================================
    # SCALE
    # ========================================================================

    def scale_window(
        self,
        window_id: str,
        scale: float,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            old_scale = window.scale

            window.scale = max(
                0.10,
                min(
                    5.0,
                    float(scale),
                ),
            )

            window.touch()

        self._emit(
            "window_scaled",
            window,
            old_scale,
        )

        return True

    def scale_window_by(
        self,
        window_id: str,
        factor: float,
    ) -> bool:

        window = self.get_window(
            window_id
        )

        if window is None:

            return False

        return self.scale_window(
            window_id,
            window.scale * factor,
        )

    # ========================================================================
    # ROTATION
    # ========================================================================

    def rotate_window(
        self,
        window_id: str,
        rotation: float,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            old_rotation = (
                window.rotation
            )

            window.rotation = float(
                rotation
            )

            window.touch()

        self._emit(
            "window_rotated",
            window,
            old_rotation,
        )

        return True

    # ========================================================================
    # STATE
    # ========================================================================

    def minimize_window(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if (
                window is None
                or not window.minimizable
            ):

                return False

            window.state = (
                WindowState.MINIMIZED
            )

            window.visible = False
            window.focused = False

            if self._focused_window == window_id:

                self._focused_window = None

                self._focus_topmost_visible()

            window.touch()

        self._emit(
            "window_minimized",
            window,
        )

        return True

    def maximize_window(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if (
                window is None
                or not window.maximizable
            ):

                return False

            window.state = (
                WindowState.MAXIMIZED
            )

            window.visible = True

            window.position.x = 0.5
            window.position.y = 0.5

            window.size.width = 1.0
            window.size.height = 1.0

            window.touch()

            self._focused_window = window_id

        self.focus_window(
            window_id
        )

        self._emit(
            "window_maximized",
            window,
        )

        return True

    def restore_window(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            window.state = (
                WindowState.NORMAL
            )

            window.visible = True

            window.touch()

        self.focus_window(
            window_id
        )

        self._emit(
            "window_restored",
            window,
        )

        return True

    def hide_window(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            window.visible = False

            window.state = (
                WindowState.HIDDEN
            )

            window.focused = False

            if self._focused_window == window_id:

                self._focused_window = None

                self._focus_topmost_visible()

            window.touch()

        self._emit(
            "window_hidden",
            window,
        )

        return True

    def show_window(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            window = self._windows.get(
                window_id
            )

            if window is None:

                return False

            window.visible = True

            window.state = (
                WindowState.NORMAL
            )

            window.touch()

        self.focus_window(
            window_id
        )

        self._emit(
            "window_shown",
            window,
        )

        return True

    # ========================================================================
    # HIT TESTING
    # ========================================================================

    def hit_test(
        self,
        x: float,
        y: float,
    ) -> Optional[HolographicWindow]:

        with self._lock:

            candidates = self.get_windows(
                visible_only=True
            )

            candidates.reverse()

            for window in candidates:

                if window.contains_point(
                    x,
                    y,
                ):

                    return window

            return None

    def focus_at(
        self,
        x: float,
        y: float,
    ) -> Optional[HolographicWindow]:

        window = self.hit_test(
            x,
            y,
        )

        if window is not None:

            self.focus_window(
                window.id
            )

        return window

    # ========================================================================
    # Z ORDER
    # ========================================================================

    def bring_to_front(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            if window_id not in self._windows:

                return False

            if window_id in self._z_order:

                self._z_order.remove(
                    window_id
                )

            self._z_order.append(
                window_id
            )

            self._recalculate_z_order()

            return True

    def send_to_back(
        self,
        window_id: str,
    ) -> bool:

        with self._lock:

            if window_id not in self._windows:

                return False

            if window_id in self._z_order:

                self._z_order.remove(
                    window_id
                )

            self._z_order.insert(
                0,
                window_id,
            )

            self._recalculate_z_order()

            return True

    def _recalculate_z_order(
        self,
    ) -> None:

        for index, window_id in enumerate(
            self._z_order
        ):

            window = self._windows.get(
                window_id
            )

            if window is not None:

                window.z_index = index

    # ========================================================================
    # INTERNAL FOCUS
    # ========================================================================

    def _focus_topmost_visible(
        self,
    ) -> None:

        for window_id in reversed(
            self._z_order
        ):

            window = self._windows.get(
                window_id
            )

            if (
                window is not None
                and window.visible
                and window.state
                not in (
                    WindowState.HIDDEN,
                    WindowState.MINIMIZED,
                )
            ):

                window.focused = True

                self._focused_window = (
                    window_id
                )

                self._recalculate_z_order()

                return

    # ========================================================================
    # EVENT SYSTEM
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "callback must be callable"
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

                logger.exception(
                    "Window event callback failed: %s",
                    event,
                )

    # ========================================================================
    # SCREEN
    # ========================================================================

    def set_screen_size(
        self,
        width: float,
        height: float,
    ) -> None:

        with self._lock:

            self.screen_width = max(
                0.01,
                float(width),
            )

            self.screen_height = max(
                0.01,
                float(height),
            )

    # ========================================================================
    # CLAMPING
    # ========================================================================

    def _clamp_x(
        self,
        x: float,
    ) -> float:

        return max(
            0.0,
            min(
                self.screen_width,
                float(x),
            ),
        )

    def _clamp_y(
        self,
        y: float,
    ) -> float:

        return max(
            0.0,
            min(
                self.screen_height,
                float(y),
            ),
        )

    @staticmethod
    def _clamp_width(
        width: float,
    ) -> float:

        return max(
            0.05,
            min(
                1.0,
                float(width),
            ),
        )

    @staticmethod
    def _clamp_height(
        height: float,
    ) -> float:

        return max(
            0.05,
            min(
                1.0,
                float(height),
            ),
        )

    @staticmethod
    def _clamp_width_for_window(
        window: HolographicWindow,
        width: float,
    ) -> float:

        return max(
            window.size.min_width,
            min(
                window.size.max_width,
                float(width),
            ),
        )

    @staticmethod
    def _clamp_height_for_window(
        window: HolographicWindow,
        height: float,
    ) -> float:

        return max(
            window.size.min_height,
            min(
                window.size.max_height,
                float(height),
            ),
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "focused_window": (
                    self._focused_window
                ),
                "z_order": list(
                    self._z_order
                ),
                "windows": [
                    window.to_dict()
                    for window in self.get_windows()
                ],
            }


__all__ = [
    "WindowState",
    "WindowLayer",
    "WindowPosition",
    "WindowSize",
    "HolographicWindow",
    "WindowManager",
]


