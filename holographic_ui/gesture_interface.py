"""
RENIX Holographic UI - Gesture Interface
========================================

Connects the gesture system with the holographic UI.

Responsibilities:
    - Receive gesture events
    - Map gestures to UI actions
    - Move the holographic cursor
    - Handle pinch selection
    - Handle grab/drag
    - Handle swipe
    - Handle zoom
    - Handle rotation
    - Handle two-hand interactions
    - Provide callbacks for UI components

This module does NOT perform hand detection.
The gestures/ package is responsible for detecting and classifying gestures.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.holographic_ui.gesture_interface"
)


# ============================================================================
# ENUMS
# ============================================================================


class UIInteraction(str, Enum):
    NONE = "none"
    MOVE_CURSOR = "move_cursor"
    HOVER = "hover"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    DRAG_START = "drag_start"
    DRAG = "drag"
    DRAG_END = "drag_end"
    SWIPE = "swipe"
    ZOOM = "zoom"
    ROTATE = "rotate"
    GRAB = "grab"
    RELEASE = "release"
    TWO_HAND_ZOOM = "two_hand_zoom"
    TWO_HAND_ROTATE = "two_hand_rotate"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class GesturePoint:
    """Normalized gesture position."""

    x: float = 0.5
    y: float = 0.5
    z: float = 0.0

    confidence: float = 1.0

    timestamp: float = field(
        default_factory=time.time
    )


@dataclass
class GestureEvent:
    """Normalized event received from the gesture engine."""

    gesture: str

    point: Optional[GesturePoint] = None

    second_point: Optional[GesturePoint] = None

    confidence: float = 1.0

    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0

    scale: float = 1.0

    rotation: float = 0.0

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    hand: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    timestamp: float = field(
        default_factory=time.time
    )


@dataclass
class CursorState:
    """Current holographic cursor state."""

    x: float = 0.5
    y: float = 0.5
    z: float = 0.0

    visible: bool = True

    hovering: bool = False

    pressed: bool = False

    dragging: bool = False

    grabbed: bool = False

    confidence: float = 1.0

    target_id: Optional[str] = None

    timestamp: float = field(
        default_factory=time.time
    )


@dataclass
class InteractionState:
    """Current gesture/UI interaction state."""

    active: bool = False

    interaction: UIInteraction = (
        UIInteraction.NONE
    )

    target_id: Optional[str] = None

    started_at: Optional[float] = None

    last_update: Optional[float] = None

    initial_x: float = 0.0
    initial_y: float = 0.0

    last_x: float = 0.0
    last_y: float = 0.0

    scale: float = 1.0

    rotation: float = 0.0


# ============================================================================
# GESTURE INTERFACE
# ============================================================================


class GestureInterface:
    """
    Converts gesture-engine events into holographic UI interactions.
    """

    def __init__(
        self,
        *,
        gesture_engine: Any = None,
        spatial_manager: Any = None,
        scene_manager: Any = None,
        min_confidence: float = 0.55,
        cursor_smoothing: float = 0.25,
    ) -> None:

        self.gesture_engine = gesture_engine
        self.spatial_manager = spatial_manager
        self.scene_manager = scene_manager

        self.min_confidence = max(
            0.0,
            min(
                1.0,
                float(min_confidence),
            ),
        )

        self.cursor_smoothing = max(
            0.0,
            min(
                1.0,
                float(cursor_smoothing),
            ),
        )

        self.cursor = CursorState()

        self.interaction = InteractionState()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False

        self._last_event_time = 0.0

        self._drag_threshold = 0.015

        self._click_timeout = 0.5

        self._last_click_time = 0.0

        self._last_click_target: Optional[
            str
        ] = None

        logger.debug(
            "Gesture interface created"
        )

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:

                return

            self._running = True

        self._connect_gesture_engine()

        self._emit(
            "started",
            self,
        )

        logger.info(
            "Gesture interface started"
        )

    def stop(self) -> None:

        with self._lock:

            if not self._running:

                return

            self._running = False

            self._reset_interaction()

        self._disconnect_gesture_engine()

        self._emit(
            "stopped",
            self,
        )

        logger.info(
            "Gesture interface stopped"
        )

    # ========================================================================
    # GESTURE ENGINE CONNECTION
    # ========================================================================

    def _connect_gesture_engine(self) -> None:

        engine = self.gesture_engine

        if engine is None:

            return

        try:

            if hasattr(engine, "on"):

                engine.on(
                    "gesture",
                    self.handle_gesture,
                )

            elif hasattr(
                engine,
                "register_callback",
            ):

                engine.register_callback(
                    self.handle_gesture
                )

        except Exception:

            logger.exception(
                "Failed to connect gesture engine"
            )

    def _disconnect_gesture_engine(self) -> None:

        engine = self.gesture_engine

        if engine is None:

            return

        try:

            if hasattr(engine, "off"):

                engine.off(
                    "gesture",
                    self.handle_gesture,
                )

            elif hasattr(
                engine,
                "remove_callback",
            ):

                engine.remove_callback(
                    self.handle_gesture
                )

        except Exception:

            logger.exception(
                "Failed to disconnect gesture engine"
            )

    # ========================================================================
    # EVENT INPUT
    # ========================================================================

    def handle_gesture(
        self,
        event: GestureEvent | dict[str, Any] | Any,
    ) -> None:
        """
        Main entry point for gesture events.
        """

        if not self._running:

            return

        event = self._normalize_event(
            event
        )

        if event is None:

            return

        if (
            event.confidence
            < self.min_confidence
        ):

            return

        self._last_event_time = (
            event.timestamp
        )

        gesture = (
            event.gesture
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        handlers = {
            "move": self._handle_move,
            "point": self._handle_point,
            "hover": self._handle_hover,
            "pinch": self._handle_pinch,
            "pinch_start": self._handle_pinch_start,
            "pinch_end": self._handle_pinch_end,
            "click": self._handle_click,
            "grab": self._handle_grab,
            "grab_start": self._handle_grab_start,
            "grab_end": self._handle_grab_end,
            "release": self._handle_release,
            "drag": self._handle_drag,
            "drag_start": self._handle_drag_start,
            "drag_end": self._handle_drag_end,
            "swipe": self._handle_swipe,
            "zoom": self._handle_zoom,
            "rotate": self._handle_rotate,
            "two_hand_zoom": self._handle_two_hand_zoom,
            "two_hand_rotate": self._handle_two_hand_rotate,
            "two_hand": self._handle_two_hand,
        }

        handler = handlers.get(
            gesture
        )

        if handler is not None:

            try:

                handler(event)

            except Exception:

                logger.exception(
                    "Gesture handler failed: %s",
                    gesture,
                )

        else:

            self._emit(
                "unknown_gesture",
                event,
            )

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    def _normalize_event(
        self,
        event: GestureEvent | dict[str, Any] | Any,
    ) -> Optional[GestureEvent]:

        if isinstance(
            event,
            GestureEvent,
        ):

            return event

        if isinstance(
            event,
            dict,
        ):

            point = self._normalize_point(
                event.get("point")
            )

            second_point = (
                self._normalize_point(
                    event.get("second_point")
                )
                if event.get("second_point")
                is not None
                else None
            )

            return GestureEvent(
                gesture=str(
                    event.get(
                        "gesture",
                        "",
                    )
                ),
                point=point,
                second_point=second_point,
                confidence=float(
                    event.get(
                        "confidence",
                        1.0,
                    )
                ),
                delta_x=float(
                    event.get(
                        "delta_x",
                        0.0,
                    )
                ),
                delta_y=float(
                    event.get(
                        "delta_y",
                        0.0,
                    )
                ),
                delta_z=float(
                    event.get(
                        "delta_z",
                        0.0,
                    )
                ),
                scale=float(
                    event.get(
                        "scale",
                        1.0,
                    )
                ),
                rotation=float(
                    event.get(
                        "rotation",
                        0.0,
                    )
                ),
                velocity_x=float(
                    event.get(
                        "velocity_x",
                        0.0,
                    )
                ),
                velocity_y=float(
                    event.get(
                        "velocity_y",
                        0.0,
                    )
                ),
                hand=event.get(
                    "hand"
                ),
                metadata=dict(
                    event.get(
                        "metadata",
                        {},
                    )
                ),
                timestamp=float(
                    event.get(
                        "timestamp",
                        time.time(),
                    )
                ),
            )

        gesture = getattr(
            event,
            "gesture",
            None,
        )

        if gesture is None:

            return None

        return GestureEvent(
            gesture=str(
                gesture
            ),
            point=self._normalize_point(
                getattr(
                    event,
                    "point",
                    None,
                )
            ),
            confidence=float(
                getattr(
                    event,
                    "confidence",
                    1.0,
                )
            ),
            delta_x=float(
                getattr(
                    event,
                    "delta_x",
                    0.0,
                )
            ),
            delta_y=float(
                getattr(
                    event,
                    "delta_y",
                    0.0,
                )
            ),
            delta_z=float(
                getattr(
                    event,
                    "delta_z",
                    0.0,
                )
            ),
            scale=float(
                getattr(
                    event,
                    "scale",
                    1.0,
                )
            ),
            rotation=float(
                getattr(
                    event,
                    "rotation",
                    0.0,
                )
            ),
            velocity_x=float(
                getattr(
                    event,
                    "velocity_x",
                    0.0,
                )
            ),
            velocity_y=float(
                getattr(
                    event,
                    "velocity_y",
                    0.0,
                )
            ),
            hand=getattr(
                event,
                "hand",
                None,
            ),
            timestamp=float(
                getattr(
                    event,
                    "timestamp",
                    time.time(),
                )
            ),
        )

    @staticmethod
    def _normalize_point(
        point: Any,
    ) -> Optional[GesturePoint]:

        if point is None:

            return None

        if isinstance(
            point,
            GesturePoint,
        ):

            return point

        if isinstance(
            point,
            dict,
        ):

            return GesturePoint(
                x=float(
                    point.get(
                        "x",
                        0.5,
                    )
                ),
                y=float(
                    point.get(
                        "y",
                        0.5,
                    )
                ),
                z=float(
                    point.get(
                        "z",
                        0.0,
                    )
                ),
                confidence=float(
                    point.get(
                        "confidence",
                        1.0,
                    )
                ),
                timestamp=float(
                    point.get(
                        "timestamp",
                        time.time(),
                    )
                ),
            )

        return GesturePoint(
            x=float(
                getattr(
                    point,
                    "x",
                    0.5,
                )
            ),
            y=float(
                getattr(
                    point,
                    "y",
                    0.5,
                )
            ),
            z=float(
                getattr(
                    point,
                    "z",
                    0.0,
                )
            ),
            confidence=float(
                getattr(
                    point,
                    "confidence",
                    1.0,
                )
            ),
        )

    # ========================================================================
    # CURSOR
    # ========================================================================

    def _update_cursor(
        self,
        point: Optional[GesturePoint],
    ) -> None:

        if point is None:

            return

        x = max(
            0.0,
            min(
                1.0,
                point.x,
            ),
        )

        y = max(
            0.0,
            min(
                1.0,
                point.y,
            ),
        )

        z = point.z

        smoothing = self.cursor_smoothing

        with self._lock:

            if smoothing > 0.0:

                factor = 1.0 - smoothing

                x = (
                    self.cursor.x * smoothing
                    + x * factor
                )

                y = (
                    self.cursor.y * smoothing
                    + y * factor
                )

                z = (
                    self.cursor.z * smoothing
                    + z * factor
                )

            self.cursor.x = x
            self.cursor.y = y
            self.cursor.z = z

            self.cursor.visible = True

            self.cursor.confidence = (
                point.confidence
            )

            self.cursor.timestamp = (
                time.time()
            )

        self._emit(
            "cursor_moved",
            self.cursor,
        )

    def get_cursor(
        self,
    ) -> CursorState:

        with self._lock:

            return CursorState(
                **vars(self.cursor)
            )

    def set_cursor_target(
        self,
        target_id: Optional[str],
    ) -> None:

        with self._lock:

            self.cursor.target_id = target_id

            self.cursor.hovering = (
                target_id is not None
            )

        self._emit(
            "cursor_target_changed",
            target_id,
        )

    # ========================================================================
    # BASIC MOVEMENT
    # ========================================================================

    def _handle_move(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        self._emit(
            "move_cursor",
            event,
            self.cursor,
        )

    def _handle_point(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            self.cursor.hovering = True

        self._emit(
            "point",
            event,
            self.cursor,
        )

    def _handle_hover(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            self.cursor.hovering = True

        self._emit(
            "hover",
            event,
            self.cursor,
        )

    # ========================================================================
    # PINCH / CLICK
    # ========================================================================

    def _handle_pinch(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        if not self.interaction.active:

            self._handle_pinch_start(
                event
            )

        else:

            self._emit(
                "pinch",
                event,
                self.cursor,
            )

    def _handle_pinch_start(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            self.cursor.pressed = True

            self.interaction.active = True

            self.interaction.interaction = (
                UIInteraction.CLICK
            )

            self.interaction.target_id = (
                self.cursor.target_id
            )

            self.interaction.started_at = (
                event.timestamp
            )

            self.interaction.initial_x = (
                self.cursor.x
            )

            self.interaction.initial_y = (
                self.cursor.y
            )

            self.interaction.last_x = (
                self.cursor.x
            )

            self.interaction.last_y = (
                self.cursor.y
            )

        self._emit(
            "pinch_start",
            event,
            self.cursor,
        )

    def _handle_pinch_end(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        target = (
            self.interaction.target_id
        )

        with self._lock:

            self.cursor.pressed = False

            self.interaction.active = False

        self._emit(
            "pinch_end",
            event,
            self.cursor,
        )

        if target is not None:

            self._emit(
                "click",
                event,
                target,
            )

    def _handle_click(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        target = self.cursor.target_id

        now = time.time()

        if (
            target is not None
            and target == self._last_click_target
            and now - self._last_click_time
            <= self._click_timeout
        ):

            self._emit(
                "double_click",
                event,
                target,
            )

            self._last_click_time = 0.0
            self._last_click_target = None

        else:

            self._emit(
                "click",
                event,
                target,
            )

            self._last_click_time = now
            self._last_click_target = target

    # ========================================================================
    # GRAB
    # ========================================================================

    def _handle_grab(
        self,
        event: GestureEvent,
    ) -> None:

        if not self.interaction.active:

            self._handle_grab_start(
                event
            )

        else:

            self._handle_drag(
                event
            )

    def _handle_grab_start(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            self.cursor.grabbed = True

            self.interaction.active = True

            self.interaction.interaction = (
                UIInteraction.GRAB
            )

            self.interaction.target_id = (
                self.cursor.target_id
            )

            self.interaction.started_at = (
                event.timestamp
            )

        self._emit(
            "grab_start",
            event,
            self.cursor,
        )

    def _handle_release(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            self.cursor.grabbed = False

            self.interaction.active = False

            self.interaction.interaction = (
                UIInteraction.RELEASE
            )

        self._emit(
            "release",
            event,
            self.cursor,
        )

        self._reset_interaction()

    def _handle_grab_end(
        self,
        event: GestureEvent,
    ) -> None:

        self._handle_release(
            event
        )

    # ========================================================================
    # DRAG
    # ========================================================================

    def _handle_drag_start(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            self.cursor.dragging = True

            self.interaction.active = True

            self.interaction.interaction = (
                UIInteraction.DRAG_START
            )

            self.interaction.target_id = (
                self.cursor.target_id
            )

            self.interaction.started_at = (
                event.timestamp
            )

            self.interaction.initial_x = (
                self.cursor.x
            )

            self.interaction.initial_y = (
                self.cursor.y
            )

        self._emit(
            "drag_start",
            event,
            self.cursor,
        )

    def _handle_drag(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            if not self.interaction.active:

                self._handle_drag_start(
                    event
                )
                return

            self.cursor.dragging = True

            self.interaction.interaction = (
                UIInteraction.DRAG
            )

            self.interaction.last_x = (
                self.cursor.x
            )

            self.interaction.last_y = (
                self.cursor.y
            )

        self._emit(
            "drag",
            event,
            self.cursor,
        )

    def _handle_drag_end(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            self.cursor.dragging = False

            self.interaction.interaction = (
                UIInteraction.DRAG_END
            )

        self._emit(
            "drag_end",
            event,
            self.cursor,
        )

        self._reset_interaction()

    # ========================================================================
    # SWIPE
    # ========================================================================

    def _handle_swipe(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        self._emit(
            "swipe",
            event,
            self.cursor,
        )

        direction = self._swipe_direction(
            event
        )

        if direction:

            self._emit(
                f"swipe_{direction}",
                event,
                self.cursor,
            )

    @staticmethod
    def _swipe_direction(
        event: GestureEvent,
    ) -> Optional[str]:

        dx = event.delta_x

        dy = event.delta_y

        if (
            abs(dx)
            < 0.01
            and abs(dy)
            < 0.01
        ):

            dx = event.velocity_x
            dy = event.velocity_y

        if (
            abs(dx)
            < 0.01
            and abs(dy)
            < 0.01
        ):

            return None

        if abs(dx) >= abs(dy):

            return (
                "right"
                if dx > 0
                else "left"
            )

        return (
            "down"
            if dy > 0
            else "up"
        )

    # ========================================================================
    # ZOOM
    # ========================================================================

    def _handle_zoom(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        scale = event.scale

        if scale <= 0:

            scale = 1.0

        with self._lock:

            self.interaction.interaction = (
                UIInteraction.ZOOM
            )

            self.interaction.scale = scale

        self._emit(
            "zoom",
            event,
            scale,
        )

        if scale > 1.0:

            self._emit(
                "zoom_in",
                event,
                scale,
            )

        elif scale < 1.0:

            self._emit(
                "zoom_out",
                event,
                scale,
            )

    # ========================================================================
    # ROTATION
    # ========================================================================

    def _handle_rotate(
        self,
        event: GestureEvent,
    ) -> None:

        self._update_cursor(
            event.point
        )

        with self._lock:

            self.interaction.interaction = (
                UIInteraction.ROTATE
            )

            self.interaction.rotation = (
                event.rotation
            )

        self._emit(
            "rotate",
            event,
            event.rotation,
        )

    # ========================================================================
    # TWO-HAND
    # ========================================================================

    def _handle_two_hand(
        self,
        event: GestureEvent,
    ) -> None:

        if (
            event.second_point is None
            or event.point is None
        ):

            return

        self._emit(
            "two_hand",
            event,
        )

    def _handle_two_hand_zoom(
        self,
        event: GestureEvent,
    ) -> None:

        self._handle_two_hand(
            event
        )

        scale = event.scale

        if scale <= 0:

            scale = 1.0

        self._emit(
            "two_hand_zoom",
            event,
            scale,
        )

        if scale > 1.0:

            self._emit(
                "zoom_in",
                event,
                scale,
            )

        elif scale < 1.0:

            self._emit(
                "zoom_out",
                event,
                scale,
            )

    def _handle_two_hand_rotate(
        self,
        event: GestureEvent,
    ) -> None:

        self._handle_two_hand(
            event
        )

        self._emit(
            "two_hand_rotate",
            event,
            event.rotation,
        )

    # ========================================================================
    # TARGETING
    # ========================================================================

    def target_object(
        self,
        object_id: Optional[str],
    ) -> None:

        self.set_cursor_target(
            object_id
        )

    def clear_target(self) -> None:

        self.set_cursor_target(
            None
        )

    # ========================================================================
    # RESET
    # ========================================================================

    def _reset_interaction(self) -> None:

        with self._lock:

            self.interaction = (
                InteractionState()
            )

            self.cursor.pressed = False

            self.cursor.dragging = False

            self.cursor.grabbed = False

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not event:

            raise ValueError(
                "event cannot be empty"
            )

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
                    "Gesture UI callback failed: %s",
                    event,
                )

    # ========================================================================
    # STATUS
    # ========================================================================

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    def get_interaction_state(
        self,
    ) -> InteractionState:

        with self._lock:

            return InteractionState(
                **vars(
                    self.interaction
                )
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "min_confidence": (
                    self.min_confidence
                ),
                "cursor": vars(
                    self.cursor
                ).copy(),
                "interaction": {
                    key: (
                        value.value
                        if isinstance(
                            value,
                            Enum,
                        )
                        else value
                    )
                    for key, value
                    in vars(
                        self.interaction
                    ).items()
                },
            }


__all__ = [
    "UIInteraction",
    "GesturePoint",
    "GestureEvent",
    "CursorState",
    "InteractionState",
    "GestureInterface",
]


