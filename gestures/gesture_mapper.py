"""
RENIX Gesture Mapper
====================

Maps classified gestures to semantic RENIX actions.

Important:
    gesture_classifier.py -> WHAT gesture happened
    gesture_mapper.py     -> WHAT that gesture means
    computer / holographic_ui -> HOW the action is executed

This module intentionally does not directly move the mouse, press keys,
or manipulate the UI. It produces normalized GestureAction objects that
can be consumed by gesture_engine.py, computer_controller.py,
holographic_ui, or other RENIX components.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .gesture_classifier import Gesture, GestureType


logger = logging.getLogger("RENIX.gestures.gesture_mapper")


# ============================================================================
# ENUMS
# ============================================================================


class GestureActionType(str, Enum):
    """Semantic actions understood by RENIX."""

    NONE = "none"

    SELECT = "select"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"

    GRAB = "grab"
    RELEASE = "release"
    DRAG = "drag"

    MOVE_CURSOR = "move_cursor"

    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"

    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"

    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"

    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"

    OPEN_MENU = "open_menu"
    CLOSE_MENU = "close_menu"

    BACK = "back"
    FORWARD = "forward"

    HOME = "home"

    PLAY_PAUSE = "play_pause"
    NEXT_MEDIA = "next_media"
    PREVIOUS_MEDIA = "previous_media"

    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MUTE = "mute"

    CONFIRM = "confirm"
    CANCEL = "cancel"

    WAKE = "wake"
    SLEEP = "sleep"

    EMERGENCY_STOP = "emergency_stop"


class MappingMode(str, Enum):
    """How aggressively gestures are interpreted."""

    NORMAL = "normal"
    PRESENTATION = "presentation"
    MEDIA = "media"
    DESKTOP = "desktop"
    HOLOGRAM = "hologram"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class GestureAction:
    """Normalized action generated from a gesture."""

    action_type: GestureActionType

    confidence: float = 1.0

    source_gesture: Optional[GestureType] = None

    hand_id: Optional[str] = None

    side: str = "unknown"

    timestamp: float = field(
        default_factory=time.time
    )

    value: float = 0.0

    x: float = 0.0
    y: float = 0.0

    dx: float = 0.0
    dy: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    consumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action_type.value,
            "confidence": self.confidence,
            "source_gesture": (
                self.source_gesture.value
                if self.source_gesture
                else None
            ),
            "hand_id": self.hand_id,
            "side": self.side,
            "timestamp": self.timestamp,
            "value": self.value,
            "position": {
                "x": self.x,
                "y": self.y,
            },
            "delta": {
                "x": self.dx,
                "y": self.dy,
            },
            "metadata": dict(self.metadata),
            "consumed": self.consumed,
        }


@dataclass
class GestureMapping:
    """Configuration for one gesture-to-action mapping."""

    gesture: GestureType

    action: GestureActionType

    enabled: bool = True

    minimum_confidence: float = 0.50

    cooldown: float = 0.0

    modes: set[MappingMode] = field(
        default_factory=lambda: {
            MappingMode.NORMAL,
            MappingMode.DESKTOP,
            MappingMode.HOLOGRAM,
        }
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class GestureMapperConfig:
    """Global mapper configuration."""

    enabled: bool = True

    mode: MappingMode = MappingMode.NORMAL

    minimum_confidence: float = 0.50

    default_cooldown: float = 0.10

    pinch_click_enabled: bool = True

    point_move_enabled: bool = True

    fist_grab_enabled: bool = True

    palm_menu_enabled: bool = True

    swipe_enabled: bool = True

    two_hand_zoom_enabled: bool = True

    two_hand_rotate_enabled: bool = True

    emit_unknown: bool = False


# ============================================================================
# GESTURE MAPPER
# ============================================================================


class GestureMapper:
    """
    Converts classified gestures into RENIX semantic actions.
    """

    def __init__(
        self,
        config: Optional[
            GestureMapperConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or GestureMapperConfig()
        )

        self._lock = threading.RLock()

        self._mappings: dict[
            GestureType,
            GestureMapping,
        ] = {}

        self._last_action_time: dict[
            tuple[GestureType, GestureActionType],
            float,
        ] = {}

        self._callbacks: list[
            Callable[[GestureAction], None]
        ] = []

        self._active_grabs: set[str] = set()

        self._register_default_mappings()

    # ========================================================================
    # DEFAULT MAPPINGS
    # ========================================================================

    def _register_default_mappings(self) -> None:

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.PINCH,
                action=GestureActionType.CLICK,
                minimum_confidence=0.55,
                cooldown=0.18,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.POINT,
                action=GestureActionType.MOVE_CURSOR,
                minimum_confidence=0.60,
                cooldown=0.0,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.GRAB,
                action=GestureActionType.GRAB,
                minimum_confidence=0.55,
                cooldown=0.05,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.FIST,
                action=GestureActionType.GRAB,
                minimum_confidence=0.65,
                cooldown=0.10,
                modes={
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.OPEN_PALM,
                action=GestureActionType.RELEASE,
                minimum_confidence=0.65,
                cooldown=0.10,
                modes={
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.SWIPE_LEFT,
                action=GestureActionType.SWIPE_LEFT,
                minimum_confidence=0.55,
                cooldown=0.25,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.MEDIA,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.SWIPE_RIGHT,
                action=GestureActionType.SWIPE_RIGHT,
                minimum_confidence=0.55,
                cooldown=0.25,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.MEDIA,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.SWIPE_UP,
                action=GestureActionType.SWIPE_UP,
                minimum_confidence=0.55,
                cooldown=0.25,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.MEDIA,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.SWIPE_DOWN,
                action=GestureActionType.SWIPE_DOWN,
                minimum_confidence=0.55,
                cooldown=0.25,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.MEDIA,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.TWO_HAND_ZOOM,
                action=GestureActionType.ZOOM_IN,
                minimum_confidence=0.55,
                cooldown=0.04,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.TWO_HAND_ROTATE,
                action=GestureActionType.ROTATE_RIGHT,
                minimum_confidence=0.55,
                cooldown=0.04,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.PEACE,
                action=GestureActionType.SELECT,
                minimum_confidence=0.65,
                cooldown=0.25,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.THUMBS_UP,
                action=GestureActionType.CONFIRM,
                minimum_confidence=0.70,
                cooldown=0.30,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.THUMBS_DOWN,
                action=GestureActionType.CANCEL,
                minimum_confidence=0.70,
                cooldown=0.30,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.PRESENTATION,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

        self.register_mapping(
            GestureMapping(
                gesture=GestureType.PALM_FORWARD,
                action=GestureActionType.OPEN_MENU,
                minimum_confidence=0.65,
                cooldown=0.40,
                modes={
                    MappingMode.NORMAL,
                    MappingMode.DESKTOP,
                    MappingMode.HOLOGRAM,
                },
            )
        )

    # ========================================================================
    # REGISTER / REMOVE
    # ========================================================================

    def register_mapping(
        self,
        mapping: GestureMapping,
    ) -> None:

        with self._lock:

            self._mappings[
                mapping.gesture
            ] = mapping

    def remove_mapping(
        self,
        gesture: GestureType,
    ) -> None:

        with self._lock:

            self._mappings.pop(
                gesture,
                None,
            )

    def get_mapping(
        self,
        gesture: GestureType,
    ) -> Optional[GestureMapping]:

        with self._lock:

            mapping = self._mappings.get(
                gesture
            )

            if mapping is None:
                return None

            return GestureMapping(
                gesture=mapping.gesture,
                action=mapping.action,
                enabled=mapping.enabled,
                minimum_confidence=mapping.minimum_confidence,
                cooldown=mapping.cooldown,
                modes=set(mapping.modes),
                metadata=dict(mapping.metadata),
            )

    # ========================================================================
    # MAP
    # ========================================================================

    def map_gesture(
        self,
        gesture: Gesture,
        *,
        timestamp: Optional[float] = None,
    ) -> Optional[GestureAction]:

        if not self.config.enabled:
            return None

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        if (
            gesture.confidence
            < self.config.minimum_confidence
        ):
            return None

        mapping = self.get_mapping(
            gesture.gesture_type
        )

        if mapping is None:
            return self._special_mapping(
                gesture,
                now,
            )

        if not mapping.enabled:
            return None

        if (
            self.config.mode
            not in mapping.modes
        ):
            return None

        if (
            gesture.confidence
            < mapping.minimum_confidence
        ):
            return None

        if self._is_on_cooldown(
            gesture,
            mapping,
            now,
        ):
            return None

        action = self._create_action(
            gesture,
            mapping,
            now,
        )

        if action is None:
            return None

        self._record_action(
            gesture,
            mapping,
            now,
        )

        self._update_state(
            action
        )

        self._emit(
            action
        )

        return action

    def map(
        self,
        gesture: Gesture,
        **kwargs: Any,
    ) -> Optional[GestureAction]:

        return self.map_gesture(
            gesture,
            **kwargs,
        )

    def map_many(
        self,
        gestures: list[Gesture],
        *,
        timestamp: Optional[float] = None,
    ) -> list[GestureAction]:

        actions: list[
            GestureAction
        ] = []

        for gesture in gestures:

            action = self.map_gesture(
                gesture,
                timestamp=timestamp,
            )

            if action is not None:

                actions.append(action)

        return actions

    # ========================================================================
    # SPECIAL MAPPINGS
    # ========================================================================

    def _special_mapping(
        self,
        gesture: Gesture,
        timestamp: float,
    ) -> Optional[GestureAction]:

        if (
            gesture.gesture_type
            == GestureType.TWO_HAND_ZOOM
        ):

            if not self.config.two_hand_zoom_enabled:
                return None

            if gesture.direction_x > 0:

                action_type = (
                    GestureActionType.ZOOM_IN
                )

            else:

                action_type = (
                    GestureActionType.ZOOM_OUT
                )

            return self._emit_special(
                gesture,
                action_type,
                timestamp,
            )

        if (
            gesture.gesture_type
            == GestureType.TWO_HAND_ROTATE
        ):

            if not self.config.two_hand_rotate_enabled:
                return None

            if gesture.direction_x > 0:

                action_type = (
                    GestureActionType.ROTATE_RIGHT
                )

            else:

                action_type = (
                    GestureActionType.ROTATE_LEFT
                )

            return self._emit_special(
                gesture,
                action_type,
                timestamp,
            )

        if (
            gesture.gesture_type
            == GestureType.PINCH
            and self.config.pinch_click_enabled
        ):

            return self._emit_special(
                gesture,
                GestureActionType.CLICK,
                timestamp,
            )

        if (
            gesture.gesture_type
            == GestureType.POINT
            and self.config.point_move_enabled
        ):

            return self._emit_special(
                gesture,
                GestureActionType.MOVE_CURSOR,
                timestamp,
            )

        if (
            gesture.gesture_type
            == GestureType.GRAB
            and self.config.fist_grab_enabled
        ):

            return self._emit_special(
                gesture,
                GestureActionType.GRAB,
                timestamp,
            )

        if (
            gesture.gesture_type
            == GestureType.OPEN_PALM
        ):

            return self._emit_special(
                gesture,
                GestureActionType.RELEASE,
                timestamp,
            )

        if self.config.emit_unknown:

            return self._emit_special(
                gesture,
                GestureActionType.NONE,
                timestamp,
            )

        return None

    def _emit_special(
        self,
        gesture: Gesture,
        action_type: GestureActionType,
        timestamp: float,
    ) -> Optional[GestureAction]:

        key = (
            gesture.gesture_type,
            action_type,
        )

        last = self._last_action_time.get(
            key,
            0.0,
        )

        if (
            timestamp - last
            < self.config.default_cooldown
        ):
            return None

        self._last_action_time[
            key
        ] = timestamp

        action = self._build_action(
            gesture,
            action_type,
            timestamp,
        )

        self._update_state(
            action
        )

        self._emit(
            action
        )

        return action

    # ========================================================================
    # ACTION CREATION
    # ========================================================================

    def _create_action(
        self,
        gesture: Gesture,
        mapping: GestureMapping,
        timestamp: float,
    ) -> Optional[GestureAction]:

        action_type = mapping.action

        # Dynamic two-hand zoom direction.
        if (
            gesture.gesture_type
            == GestureType.TWO_HAND_ZOOM
        ):

            if gesture.direction_x > 0:

                action_type = (
                    GestureActionType.ZOOM_IN
                )

            else:

                action_type = (
                    GestureActionType.ZOOM_OUT
                )

        # Dynamic two-hand rotation direction.
        if (
            gesture.gesture_type
            == GestureType.TWO_HAND_ROTATE
        ):

            if gesture.direction_x > 0:

                action_type = (
                    GestureActionType.ROTATE_RIGHT
                )

            else:

                action_type = (
                    GestureActionType.ROTATE_LEFT
                )

        return self._build_action(
            gesture,
            action_type,
            timestamp,
            mapping.metadata,
        )

    def _build_action(
        self,
        gesture: Gesture,
        action_type: GestureActionType,
        timestamp: float,
        extra_metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> GestureAction:

        metadata = dict(
            gesture.metadata
        )

        if extra_metadata:

            metadata.update(
                extra_metadata
            )

        metadata[
            "gesture_name"
        ] = gesture.gesture_type.value

        return GestureAction(
            action_type=action_type,
            confidence=gesture.confidence,
            source_gesture=gesture.gesture_type,
            hand_id=gesture.hand_id,
            side=gesture.side,
            timestamp=timestamp,
            value=gesture.magnitude,
            x=gesture.center_x,
            y=gesture.center_y,
            dx=gesture.direction_x,
            dy=gesture.direction_y,
            metadata=metadata,
        )

    # ========================================================================
    # COOLDOWN
    # ========================================================================

    def _is_on_cooldown(
        self,
        gesture: Gesture,
        mapping: GestureMapping,
        timestamp: float,
    ) -> bool:

        cooldown = max(
            mapping.cooldown,
            self.config.default_cooldown
            if mapping.cooldown == 0.0
            else 0.0,
        )

        key = (
            gesture.gesture_type,
            mapping.action,
        )

        last = self._last_action_time.get(
            key,
            0.0,
        )

        return (
            timestamp - last
            < cooldown
        )

    def _record_action(
        self,
        gesture: Gesture,
        mapping: GestureMapping,
        timestamp: float,
    ) -> None:

        self._last_action_time[
            (
                gesture.gesture_type,
                mapping.action,
            )
        ] = timestamp

    # ========================================================================
    # STATE
    # ========================================================================

    def _update_state(
        self,
        action: GestureAction,
    ) -> None:

        hand_id = action.hand_id

        if (
            action.action_type
            == GestureActionType.GRAB
        ):

            if hand_id:

                self._active_grabs.add(
                    hand_id
                )

        elif (
            action.action_type
            == GestureActionType.RELEASE
        ):

            if hand_id:

                self._active_grabs.discard(
                    hand_id
                )

    def is_grabbing(
        self,
        hand_id: str,
    ) -> bool:

        with self._lock:

            return (
                hand_id
                in self._active_grabs
            )

    def get_active_grabs(
        self,
    ) -> set[str]:

        with self._lock:

            return set(
                self._active_grabs
            )

    # ========================================================================
    # MODE
    # ========================================================================

    def set_mode(
        self,
        mode: MappingMode,
    ) -> None:

        with self._lock:

            self.config.mode = mode

            logger.info(
                "Gesture mapping mode changed to %s",
                mode.value,
            )

    def get_mode(
        self,
    ) -> MappingMode:

        with self._lock:

            return self.config.mode

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def subscribe(
        self,
        callback: Callable[
            [GestureAction],
            None,
        ],
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "callback must be callable"
            )

        with self._lock:

            if callback not in self._callbacks:

                self._callbacks.append(
                    callback
                )

    def unsubscribe(
        self,
        callback: Callable[
            [GestureAction],
            None,
        ],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _emit(
        self,
        action: GestureAction,
    ) -> None:

        callbacks = list(
            self._callbacks
        )

        for callback in callbacks:

            try:

                callback(action)

            except Exception:

                logger.exception(
                    "Gesture action callback failed"
                )

    # ========================================================================
    # UTILITY
    # ========================================================================

    def clear_state(self) -> None:

        with self._lock:

            self._last_action_time.clear()
            self._active_grabs.clear()

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "enabled": self.config.enabled,
                "mode": self.config.mode.value,
                "mappings": len(
                    self._mappings
                ),
                "callbacks": len(
                    self._callbacks
                ),
                "active_grabs": len(
                    self._active_grabs
                ),
            }


# ============================================================================
# DEFAULT SINGLETON
# ============================================================================


_default_mapper: Optional[
    GestureMapper
] = None

_default_mapper_lock = threading.RLock()


def get_gesture_mapper() -> GestureMapper:

    global _default_mapper

    with _default_mapper_lock:

        if _default_mapper is None:

            _default_mapper = (
                GestureMapper()
            )

        return _default_mapper


def map_gesture(
    gesture: Gesture,
    **kwargs: Any,
) -> Optional[GestureAction]:

    return (
        get_gesture_mapper()
        .map_gesture(
            gesture,
            **kwargs,
        )
    )


def map_gestures(
    gestures: list[Gesture],
    **kwargs: Any,
) -> list[GestureAction]:

    return (
        get_gesture_mapper()
        .map_many(
            gestures,
            **kwargs,
        )
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "GestureActionType",
    "MappingMode",
    "GestureAction",
    "GestureMapping",
    "GestureMapperConfig",
    "GestureMapper",
    "get_gesture_mapper",
    "map_gesture",
    "map_gestures",
]


