"""
RENIX - Holographic UI Gesture Manager
======================================

Central gesture orchestration layer.

Responsibilities:
- Connect RENIX gesture engine to holographic UI
- Register/unregister gesture handlers
- Convert gesture events into UI actions
- Maintain gesture state
- Manage gesture cursor
- Handle one-hand and two-hand interactions
- Support pinch, swipe, grab, rotate, zoom, drag, point and palm
- Provide callbacks for the rest of the UI
- Prevent duplicate gesture firing
- Apply gesture sensitivity/smoothing
- Support enable/disable and interaction modes

IMPORTANT:
This file does NOT implement hand tracking itself.
The existing gestures/ package remains responsible for detection/classification.

Flow:

Camera / Vision
      ↓
gestures.gesture_engine
      ↓
GestureManager
      ↓
Holographic UI
      ↓
Scene / HUD / Windows / Widgets
"""

from __future__ import annotations

import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# ENUMS
# ============================================================================


class GestureType(str, Enum):
    NONE = "none"

    PINCH = "pinch"
    SWIPE = "swipe"
    GRAB = "grab"
    ROTATE = "rotate"
    ZOOM = "zoom"
    DRAG = "drag"
    POINT = "point"
    PALM = "palm"

    TWO_HAND = "two_hand"

    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"

    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"

    OPEN_MENU = "open_menu"
    CLOSE_MENU = "close_menu"


class GestureState(str, Enum):
    IDLE = "idle"
    STARTED = "started"
    ACTIVE = "active"
    UPDATED = "updated"
    ENDED = "ended"
    CANCELLED = "cancelled"


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class InteractionMode(str, Enum):
    GESTURE = "gesture"
    VOICE = "voice"
    HYBRID = "hybrid"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class GesturePoint:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class GestureEvent:
    """
    Normalized gesture event consumed by the holographic UI.
    """

    gesture: GestureType = GestureType.NONE

    state: GestureState = GestureState.ACTIVE

    hand: HandSide = HandSide.UNKNOWN

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_z: float = 0.0

    scale: float = 1.0

    rotation: float = 0.0

    confidence: float = 1.0

    timestamp: float = field(
        default_factory=time.time
    )

    data: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class GestureCursor:
    """
    Virtual cursor controlled by hand movement.
    """

    x: float = 0.5
    y: float = 0.5

    target_x: float = 0.5
    target_y: float = 0.5

    visible: bool = True

    active: bool = False

    smoothing: float = 0.70

    hand: HandSide = HandSide.RIGHT

    confidence: float = 0.0

    last_update: float = field(
        default_factory=time.time
    )

    def update(
        self,
        x: float,
        y: float,
        confidence: float = 1.0,
    ) -> None:

        self.target_x = max(
            0.0,
            min(1.0, x),
        )

        self.target_y = max(
            0.0,
            min(1.0, y),
        )

        alpha = 1.0 - self.smoothing

        self.x += (
            self.target_x - self.x
        ) * alpha

        self.y += (
            self.target_y - self.y
        ) * alpha

        self.confidence = confidence

        self.last_update = time.time()


@dataclass
class GestureManagerConfig:
    enabled: bool = True

    sensitivity: float = 1.0

    smoothing: float = 0.70

    minimum_confidence: float = 0.55

    cursor_enabled: bool = True

    cursor_hand: HandSide = HandSide.RIGHT

    interaction_mode: InteractionMode = (
        InteractionMode.HYBRID
    )

    duplicate_threshold: float = 0.12

    pinch_click_enabled: bool = True

    palm_menu_enabled: bool = True

    swipe_navigation_enabled: bool = True

    two_hand_enabled: bool = True


# ============================================================================
# GESTURE MANAGER
# ============================================================================


class GestureManager:
    """
    Main bridge between RENIX gesture detection and holographic UI.
    """

    def __init__(
        self,
        gesture_engine: Any = None,
        config: Optional[
            GestureManagerConfig
        ] = None,
    ) -> None:

        self._lock = threading.RLock()

        self.engine = gesture_engine

        self.config = (
            config
            if config is not None
            else GestureManagerConfig()
        )

        self.cursor = GestureCursor(
            smoothing=self.config.smoothing,
            hand=self.config.cursor_hand,
        )

        self._running = False

        self._enabled = (
            self.config.enabled
        )

        self._last_gesture = (
            GestureType.NONE
        )

        self._last_gesture_time = 0.0

        self._active_gestures: set[
            GestureType
        ] = set()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._handlers: dict[
            GestureType,
            Callable[
                [GestureEvent],
                Any,
            ],
        ] = {}

        self._event_history: list[
            GestureEvent
        ] = []

        self._max_history = 100

        self._stats = {
            "events": 0,
            "accepted": 0,
            "rejected": 0,
            "gestures": {},
        }

        self._register_default_handlers()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:
                return

            self._running = True

        self._connect_engine()

        self._emit(
            "started",
            self,
        )

    def stop(self) -> None:

        with self._lock:

            if not self._running:
                return

            self._running = False

        self._disconnect_engine()

        self._active_gestures.clear()

        self.cursor.active = False

        self._emit(
            "stopped",
            self,
        )

    def enable(self) -> None:

        with self._lock:
            self._enabled = True

        self._emit(
            "enabled",
        )

    def disable(self) -> None:

        with self._lock:
            self._enabled = False

            self._active_gestures.clear()

            self.cursor.active = False

        self._emit(
            "disabled",
        )

    def toggle(self) -> bool:

        if self._enabled:
            self.disable()
        else:
            self.enable()

        return self._enabled

    @property
    def running(self) -> bool:
        return self._running

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ========================================================================
    # ENGINE CONNECTION
    # ========================================================================

    def _connect_engine(self) -> None:

        if self.engine is None:
            return

        # Support multiple possible event APIs.
        try:

            if hasattr(
                self.engine,
                "on",
            ):

                self.engine.on(
                    "gesture",
                    self.process,
                )

                self.engine.on(
                    "gesture_detected",
                    self.process,
                )

                self.engine.on(
                    "hand_update",
                    self.process_hand,
                )

        except Exception:

            pass

    def _disconnect_engine(self) -> None:

        if self.engine is None:
            return

        try:

            if hasattr(
                self.engine,
                "off",
            ):

                self.engine.off(
                    "gesture",
                    self.process,
                )

                self.engine.off(
                    "gesture_detected",
                    self.process,
                )

                self.engine.off(
                    "hand_update",
                    self.process_hand,
                )

        except Exception:

            pass

    # ========================================================================
    # EVENT PROCESSING
    # ========================================================================

    def process(
        self,
        event: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[GestureEvent]:

        if not self._enabled:
            return None

        normalized = (
            self._normalize_event(
                event,
                *args,
                **kwargs,
            )
        )

        if normalized is None:
            return None

        self._stats["events"] += 1

        if not self._validate_event(
            normalized
        ):

            self._stats[
                "rejected"
            ] += 1

            return None

        if self._is_duplicate(
            normalized
        ):

            self._stats[
                "rejected"
            ] += 1

            return None

        self._stats[
            "accepted"
        ] += 1

        self._record_event(
            normalized
        )

        self._update_state(
            normalized
        )

        self._update_cursor(
            normalized
        )

        self._execute_handler(
            normalized
        )

        self._emit(
            "gesture",
            normalized,
        )

        self._emit(
            normalized.gesture.value,
            normalized,
        )

        return normalized

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    def _normalize_event(
        self,
        event: Any,
        *args: Any,
        **kwargs: Any,
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

            raw_gesture = (
                event.get(
                    "gesture",
                    event.get(
                        "type",
                        event.get(
                            "name",
                            "none",
                        ),
                    ),
                )
            )

            gesture = (
                self._normalize_gesture(
                    raw_gesture
                )
            )

            state = self._normalize_state(
                event.get(
                    "state",
                    "active",
                )
            )

            hand = self._normalize_hand(
                event.get(
                    "hand",
                    event.get(
                        "side",
                        "unknown",
                    ),
                )
            )

            return GestureEvent(
                gesture=gesture,
                state=state,
                hand=hand,
                x=float(
                    event.get("x", 0.0)
                ),
                y=float(
                    event.get("y", 0.0)
                ),
                z=float(
                    event.get("z", 0.0)
                ),
                delta_x=float(
                    event.get(
                        "delta_x",
                        event.get(
                            "dx",
                            0.0,
                        ),
                    )
                ),
                delta_y=float(
                    event.get(
                        "delta_y",
                        event.get(
                            "dy",
                            0.0,
                        ),
                    )
                ),
                delta_z=float(
                    event.get(
                        "delta_z",
                        event.get(
                            "dz",
                            0.0,
                        ),
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
                confidence=float(
                    event.get(
                        "confidence",
                        1.0,
                    )
                ),
                timestamp=float(
                    event.get(
                        "timestamp",
                        time.time(),
                    )
                ),
                data=dict(
                    event.get(
                        "data",
                        {},
                    )
                ),
            )

        if isinstance(
            event,
            str,
        ):

            return GestureEvent(
                gesture=self._normalize_gesture(
                    event
                )
            )

        return None

    @staticmethod
    def _normalize_gesture(
        gesture: Any,
    ) -> GestureType:

        if isinstance(
            gesture,
            GestureType,
        ):

            return gesture

        value = str(
            gesture
        ).lower().strip()

        aliases = {

            "pinch_start": GestureType.PINCH,
            "pinch_hold": GestureType.PINCH,
            "pinch_end": GestureType.PINCH,

            "swipe_left": GestureType.SWIPE,
            "swipe_right": GestureType.SWIPE,
            "swipe_up": GestureType.SWIPE,
            "swipe_down": GestureType.SWIPE,

            "grab_start": GestureType.GRAB,
            "grab_hold": GestureType.GRAB,
            "grab_end": GestureType.GRAB,

            "rotate_left": GestureType.ROTATE,
            "rotate_right": GestureType.ROTATE,

            "zoom_in": GestureType.ZOOM,
            "zoom_out": GestureType.ZOOM,

            "drag_start": GestureType.DRAG,
            "drag_move": GestureType.DRAG,
            "drag_end": GestureType.DRAG,

            "index_point": GestureType.POINT,

            "open_palm": GestureType.PALM,

            "two_hands": GestureType.TWO_HAND,

            "click": GestureType.LEFT_CLICK,
            "left_click": GestureType.LEFT_CLICK,

            "rightclick": GestureType.RIGHT_CLICK,

            "doubleclick": GestureType.DOUBLE_CLICK,

        }

        if value in aliases:
            return aliases[value]

        try:

            return GestureType(value)

        except ValueError:

            return GestureType.NONE

    @staticmethod
    def _normalize_state(
        state: Any,
    ) -> GestureState:

        if isinstance(
            state,
            GestureState,
        ):
            return state

        try:

            return GestureState(
                str(
                    state
                ).lower()
            )

        except ValueError:

            return GestureState.ACTIVE

    @staticmethod
    def _normalize_hand(
        hand: Any,
    ) -> HandSide:

        if isinstance(
            hand,
            HandSide,
        ):
            return hand

        value = str(
            hand
        ).lower()

        if "left" in value:
            return HandSide.LEFT

        if "right" in value:
            return HandSide.RIGHT

        return HandSide.UNKNOWN

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_event(
        self,
        event: GestureEvent,
    ) -> bool:

        if (
            event.gesture
            == GestureType.NONE
        ):
            return False

        if (
            event.confidence
            < self.config.minimum_confidence
        ):
            return False

        if (
            self.config.interaction_mode
            == InteractionMode.VOICE
        ):
            return False

        return True

    def _is_duplicate(
        self,
        event: GestureEvent,
    ) -> bool:

        now = time.time()

        if (
            event.gesture
            != self._last_gesture
        ):

            self._last_gesture = (
                event.gesture
            )

            self._last_gesture_time = now

            return False

        if (
            now
            - self._last_gesture_time
            < self.config.duplicate_threshold
        ):

            return (
                event.state
                == GestureState.STARTED
            )

        self._last_gesture_time = now

        return False

    # ========================================================================
    # STATE
    # ========================================================================

    def _update_state(
        self,
        event: GestureEvent,
    ) -> None:

        gesture = event.gesture

        if event.state in (
            GestureState.STARTED,
            GestureState.ACTIVE,
            GestureState.UPDATED,
        ):

            self._active_gestures.add(
                gesture
            )

        elif event.state in (
            GestureState.ENDED,
            GestureState.CANCELLED,
        ):

            self._active_gestures.discard(
                gesture
            )

    def is_gesture_active(
        self,
        gesture: GestureType | str,
    ) -> bool:

        if isinstance(
            gesture,
            str,
        ):

            gesture = (
                self._normalize_gesture(
                    gesture
                )
            )

        return gesture in (
            self._active_gestures
        )

    # ========================================================================
    # CURSOR
    # ========================================================================

    def process_hand(
        self,
        data: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        if not self._enabled:
            return

        if not self.config.cursor_enabled:
            return

        if isinstance(
            data,
            dict,
        ):

            hand = self._normalize_hand(
                data.get(
                    "hand",
                    data.get(
                        "side",
                        "unknown",
                    ),
                )
            )

            if (
                self.config.cursor_hand
                != HandSide.UNKNOWN
                and hand
                != self.config.cursor_hand
            ):
                return

            x = float(
                data.get(
                    "x",
                    0.5,
                )
            )

            y = float(
                data.get(
                    "y",
                    0.5,
                )
            )

            confidence = float(
                data.get(
                    "confidence",
                    1.0,
                )
            )

            self.update_cursor(
                x,
                y,
                confidence,
                hand,
            )

    def _update_cursor(
        self,
        event: GestureEvent,
    ) -> None:

        if not self.config.cursor_enabled:
            return

        if event.gesture not in (
            GestureType.POINT,
            GestureType.PINCH,
            GestureType.DRAG,
            GestureType.GRAB,
        ):
            return

        if (
            self.config.cursor_hand
            != HandSide.UNKNOWN
            and event.hand
            != self.config.cursor_hand
        ):
            return

        self.update_cursor(
            event.x,
            event.y,
            event.confidence,
            event.hand,
        )

    def update_cursor(
        self,
        x: float,
        y: float,
        confidence: float = 1.0,
        hand: HandSide = HandSide.UNKNOWN,
    ) -> None:

        sensitivity = (
            self.config.sensitivity
        )

        x = 0.5 + (
            x - 0.5
        ) * sensitivity

        y = 0.5 + (
            y - 0.5
        ) * sensitivity

        self.cursor.hand = hand

        self.cursor.active = True

        self.cursor.update(
            x,
            y,
            confidence,
        )

        self._emit(
            "cursor",
            self.cursor,
        )

    def hide_cursor(self) -> None:

        self.cursor.visible = False

        self._emit(
            "cursor_visibility",
            False,
        )

    def show_cursor(self) -> None:

        self.cursor.visible = True

        self._emit(
            "cursor_visibility",
            True,
        )

    def get_cursor_position(
        self,
    ) -> tuple[float, float]:

        return (
            self.cursor.x,
            self.cursor.y,
        )

    # ========================================================================
    # DEFAULT HANDLERS
    # ========================================================================

    def _register_default_handlers(
        self,
    ) -> None:

        self._handlers = {

            GestureType.PINCH:
                self._handle_pinch,

            GestureType.SWIPE:
                self._handle_swipe,

            GestureType.GRAB:
                self._handle_grab,

            GestureType.ROTATE:
                self._handle_rotate,

            GestureType.ZOOM:
                self._handle_zoom,

            GestureType.DRAG:
                self._handle_drag,

            GestureType.POINT:
                self._handle_point,

            GestureType.PALM:
                self._handle_palm,

            GestureType.TWO_HAND:
                self._handle_two_hand,

            GestureType.LEFT_CLICK:
                self._handle_left_click,

            GestureType.RIGHT_CLICK:
                self._handle_right_click,

            GestureType.DOUBLE_CLICK:
                self._handle_double_click,

        }

    def register_handler(
        self,
        gesture: GestureType | str,
        handler: Callable[
            [GestureEvent],
            Any,
        ],
    ) -> None:

        if isinstance(
            gesture,
            str,
        ):

            gesture = (
                self._normalize_gesture(
                    gesture
                )
            )

        if not callable(handler):

            raise TypeError(
                "handler must be callable"
            )

        self._handlers[
            gesture
        ] = handler

    def unregister_handler(
        self,
        gesture: GestureType | str,
    ) -> None:

        if isinstance(
            gesture,
            str,
        ):

            gesture = (
                self._normalize_gesture(
                    gesture
                )
            )

        self._handlers.pop(
            gesture,
            None,
        )

    def _execute_handler(
        self,
        event: GestureEvent,
    ) -> None:

        handler = self._handlers.get(
            event.gesture
        )

        if handler is None:
            return

        try:

            handler(event)

        except Exception as exc:

            self._emit(
                "handler_error",
                event,
                exc,
            )

    # ========================================================================
    # GESTURE HANDLERS
    # ========================================================================

    def _handle_pinch(
        self,
        event: GestureEvent,
    ) -> None:

        if not self.config.pinch_click_enabled:
            return

        self._emit(
            "ui_pinch",
            event,
        )

    def _handle_swipe(
        self,
        event: GestureEvent,
    ) -> None:

        if not self.config.swipe_navigation_enabled:
            return

        direction = self._get_swipe_direction(
            event
        )

        self._emit(
            "ui_swipe",
            event,
            direction,
        )

    def _handle_grab(
        self,
        event: GestureEvent,
    ) -> None:

        self._emit(
            "ui_grab",
            event,
        )

    def _handle_rotate(
        self,
        event: GestureEvent,
    ) -> None:

        self._emit(
            "ui_rotate",
            event,
            event.rotation,
        )

    def _handle_zoom(
        self,
        event: GestureEvent,
    ) -> None:

        self._emit(
            "ui_zoom",
            event,
            event.scale,
        )

    def _handle_drag(
        self,
        event: GestureEvent,
    ) -> None:

        self._emit(
            "ui_drag",
            event,
        )

    def _handle_point(
        self,
        event: GestureEvent,
    ) -> None:

        self._emit(
            "ui_point",
            event,
        )

    def _handle_palm(
        self,
        event: GestureEvent,
    ) -> None:

        if not self.config.palm_menu_enabled:
            return

        self._emit(
            "ui_palm",
            event,
        )

    def _handle_two_hand(
        self,
        event: GestureEvent,
    ) -> None:

        if not self.config.two_hand_enabled:
            return

        self._emit(
            "ui_two_hand",
            event,
        )

    def _handle_left_click(
        self,
        event: GestureEvent,
    ) -> None:

        self._emit(
            "ui_left_click",
            event,
        )

    def _handle_right_click(
        self,
        event: GestureEvent,
    ) -> None:

        self._emit(
            "ui_right_click",
            event,
        )

    def _handle_double_click(
        self,
        event: GestureEvent,
    ) -> None:

        self._emit(
            "ui_double_click",
            event,
        )

    # ========================================================================
    # SWIPE
    # ========================================================================

    @staticmethod
    def _get_swipe_direction(
        event: GestureEvent,
    ) -> str:

        dx = event.delta_x
        dy = event.delta_y

        if abs(dx) >= abs(dy):

            if dx > 0:
                return "right"

            return "left"

        if dy > 0:
            return "down"

        return "up"

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_sensitivity(
        self,
        value: float,
    ) -> None:

        self.config.sensitivity = max(
            0.1,
            min(3.0, float(value)),
        )

        self._emit(
            "configuration_changed",
            "sensitivity",
            self.config.sensitivity,
        )

    def set_smoothing(
        self,
        value: float,
    ) -> None:

        self.config.smoothing = max(
            0.0,
            min(1.0, float(value)),
        )

        self.cursor.smoothing = (
            self.config.smoothing
        )

        self._emit(
            "configuration_changed",
            "smoothing",
            self.config.smoothing,
        )

    def set_minimum_confidence(
        self,
        value: float,
    ) -> None:

        self.config.minimum_confidence = max(
            0.0,
            min(1.0, float(value)),
        )

    def set_interaction_mode(
        self,
        mode: InteractionMode | str,
    ) -> None:

        if isinstance(
            mode,
            str,
        ):

            mode = InteractionMode(
                mode.lower()
            )

        self.config.interaction_mode = mode

        self._emit(
            "configuration_changed",
            "interaction_mode",
            mode,
        )

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

                # UI callback errors must never
                # crash the gesture manager.
                pass

    # ========================================================================
    # HISTORY / STATISTICS
    # ========================================================================

    def _record_event(
        self,
        event: GestureEvent,
    ) -> None:

        with self._lock:

            self._event_history.append(
                event
            )

            if len(
                self._event_history
            ) > self._max_history:

                self._event_history.pop(
                    0
                )

            name = event.gesture.value

            self._stats[
                "gestures"
            ][name] = (
                self._stats[
                    "gestures"
                ].get(
                    name,
                    0,
                )
                + 1
            )

    def get_history(
        self,
        limit: int = 20,
    ) -> list[GestureEvent]:

        with self._lock:

            return list(
                self._event_history[
                    -max(1, limit):
                ]
            )

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                **self._stats,
                "active_gestures": [
                    gesture.value
                    for gesture
                    in self._active_gestures
                ],
            }

    def clear_history(self) -> None:

        with self._lock:

            self._event_history.clear()

    # ========================================================================
    # COMMAND HELPERS
    # ========================================================================

    def trigger(
        self,
        gesture: GestureType | str,
        *,
        state: GestureState = (
            GestureState.ACTIVE
        ),
        hand: HandSide = (
            HandSide.UNKNOWN
        ),
        x: float = 0.5,
        y: float = 0.5,
        confidence: float = 1.0,
        **data: Any,
    ) -> Optional[GestureEvent]:

        event = GestureEvent(
            gesture=(
                self._normalize_gesture(
                    gesture
                )
            ),
            state=state,
            hand=hand,
            x=x,
            y=y,
            confidence=confidence,
            data=data,
        )

        return self.process(
            event
        )

    def cancel_all(self) -> None:

        active = list(
            self._active_gestures
        )

        for gesture in active:

            self.process(
                GestureEvent(
                    gesture=gesture,
                    state=(
                        GestureState.CANCELLED
                    ),
                )
            )

        self._active_gestures.clear()

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
    ) -> dict[str, Any]:

        return {
            "running": self.running,
            "enabled": self.enabled,
            "cursor": {
                "visible": (
                    self.cursor.visible
                ),
                "active": (
                    self.cursor.active
                ),
                "x": self.cursor.x,
                "y": self.cursor.y,
                "confidence": (
                    self.cursor.confidence
                ),
            },
            "active_gestures": [
                gesture.value
                for gesture
                in self._active_gestures
            ],
            "interaction_mode": (
                self.config
                .interaction_mode
                .value
            ),
            "sensitivity": (
                self.config.sensitivity
            ),
            "smoothing": (
                self.config.smoothing
            ),
            "minimum_confidence": (
                self.config
                .minimum_confidence
            ),
            "statistics": (
                self.get_statistics()
            ),
        }


# ============================================================================
# FACTORY
# ============================================================================


def create_gesture_manager(
    gesture_engine: Any = None,
    config: Optional[
        GestureManagerConfig
    ] = None,
) -> GestureManager:

    manager = GestureManager(
        gesture_engine=gesture_engine,
        config=config,
    )

    manager.start()

    return manager


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "GestureType",
    "GestureState",
    "HandSide",
    "InteractionMode",
    "GesturePoint",
    "GestureEvent",
    "GestureCursor",
    "GestureManagerConfig",
    "GestureManager",
    "create_gesture_manager",
]


