"""
RENIX Holographic UI - HUD Engine
=================================

Heads-Up Display engine for RENIX.

Responsibilities:
    - Manage HUD elements
    - Create/update/remove HUD components
    - Status indicators
    - Voice state visualization
    - Gesture state visualization
    - System information
    - Notifications
    - Progress indicators
    - Labels and values
    - Visibility and opacity
    - Layer ordering
    - Animation metadata

This module is renderer-independent.
The actual drawing/rendering is handled by the UI/rendering layer.
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
    "RENIX.holographic_ui.hud_engine"
)


# ============================================================================
# ENUMS
# ============================================================================


class HUDElementType(str, Enum):
    TEXT = "text"
    LABEL = "label"
    VALUE = "value"
    STATUS = "status"
    PROGRESS = "progress"
    ICON = "icon"
    BUTTON = "button"
    PANEL = "panel"
    INDICATOR = "indicator"
    WAVEFORM = "waveform"
    NOTIFICATION = "notification"
    CLOCK = "clock"
    CUSTOM = "custom"


class HUDState(str, Enum):
    HIDDEN = "hidden"
    VISIBLE = "visible"
    ACTIVE = "active"
    DISABLED = "disabled"
    WARNING = "warning"
    ERROR = "error"


class HUDAnchor(str, Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"

    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"

    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class HUDPosition:
    """
    HUD position.

    Coordinates are normalized by default:
        x = 0.0 -> left
        x = 1.0 -> right
        y = 0.0 -> top
        y = 1.0 -> bottom
    """

    x: float = 0.5
    y: float = 0.5

    z: float = 0.0

    anchor: HUDAnchor = HUDAnchor.CENTER

    offset_x: float = 0.0
    offset_y: float = 0.0


@dataclass
class HUDStyle:
    """Visual style of a HUD element."""

    opacity: float = 1.0

    scale: float = 1.0

    glow: float = 1.0

    font_size: float = 20.0

    line_width: float = 1.0

    color: tuple[float, float, float] = (
        0.0,
        1.0,
        0.8,
    )

    background_color: tuple[
        float,
        float,
        float,
    ] = (
        0.0,
        0.05,
        0.05,
    )

    background_opacity: float = 0.25

    border_width: float = 0.0

    corner_radius: float = 0.0


@dataclass
class HUDAnimation:
    """Animation configuration."""

    enabled: bool = False

    pulse: bool = False

    pulse_speed: float = 2.0

    pulse_amount: float = 0.15

    fade: bool = False

    fade_speed: float = 2.0

    slide: bool = False

    slide_speed: float = 4.0


@dataclass
class HUDElement:
    """Complete HUD element."""

    id: str

    element_type: HUDElementType = (
        HUDElementType.CUSTOM
    )

    name: str = ""

    text: str = ""

    value: Any = None

    position: HUDPosition = field(
        default_factory=HUDPosition
    )

    style: HUDStyle = field(
        default_factory=HUDStyle
    )

    animation: HUDAnimation = field(
        default_factory=HUDAnimation
    )

    state: HUDState = HUDState.VISIBLE

    visible: bool = True

    interactive: bool = False

    layer: int = 0

    parent_id: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )


# ============================================================================
# HUD ENGINE
# ============================================================================


class HUDEngine:
    """
    Central HUD state manager for RENIX.
    """

    def __init__(
        self,
        *,
        hologram_engine: Any = None,
        theme: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:

        self.hologram_engine = (
            hologram_engine
        )

        self.theme = (
            dict(theme)
            if theme
            else {}
        )

        self._elements: dict[
            str,
            HUDElement,
        ] = {}

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._lock = threading.RLock()

        self._running = False

        self._time = 0.0

        self._last_update = time.monotonic()

        logger.debug(
            "HUD engine created"
        )

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:
                return

            self._running = True

            self._last_update = (
                time.monotonic()
            )

        self._emit(
            "started",
            self,
        )

        logger.info(
            "HUD engine started"
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

        logger.info(
            "HUD engine stopped"
        )

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        delta_time: Optional[float] = None,
    ) -> None:

        if not self._running:
            return

        now = time.monotonic()

        if delta_time is None:

            delta_time = (
                now - self._last_update
            )

        delta_time = max(
            0.0,
            min(
                0.25,
                float(delta_time),
            ),
        )

        self._last_update = now

        with self._lock:

            self._time += delta_time

            elements = list(
                self._elements.values()
            )

        for element in elements:

            if (
                element.visible
                and element.animation.enabled
            ):

                self._update_animation(
                    element,
                    delta_time,
                )

        self._emit(
            "updated",
            delta_time,
        )

    # ========================================================================
    # CREATE
    # ========================================================================

    def create(
        self,
        element_type: HUDElementType
        | str = HUDElementType.CUSTOM,
        *,
        name: str = "",
        element_id: Optional[str] = None,
        text: str = "",
        value: Any = None,
        position: Optional[
            HUDPosition
        ] = None,
        style: Optional[
            HUDStyle
        ] = None,
        animation: Optional[
            HUDAnimation
        ] = None,
        interactive: bool = False,
        layer: int = 0,
        parent_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> HUDElement:

        if isinstance(
            element_type,
            str,
        ):

            try:

                element_type = (
                    HUDElementType(
                        element_type.lower()
                    )
                )

            except ValueError:

                element_type = (
                    HUDElementType.CUSTOM
                )

        element = HUDElement(
            id=(
                element_id
                or uuid.uuid4().hex
            ),
            element_type=element_type,
            name=name,
            text=text,
            value=value,
            position=(
                position
                or HUDPosition()
            ),
            style=(
                style
                or HUDStyle()
            ),
            animation=(
                animation
                or HUDAnimation()
            ),
            interactive=interactive,
            layer=layer,
            parent_id=parent_id,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

        self.add(element)

        return element

    def add(
        self,
        element: HUDElement,
    ) -> HUDElement:

        if not element.id:

            raise ValueError(
                "HUD element ID cannot be empty"
            )

        with self._lock:

            element.updated_at = time.time()

            self._elements[
                element.id
            ] = element

        self._emit(
            "created",
            element,
        )

        return element

    # ========================================================================
    # REMOVE
    # ========================================================================

    def remove(
        self,
        element_id: str,
    ) -> bool:

        with self._lock:

            element = self._elements.pop(
                element_id,
                None,
            )

        if element is None:
            return False

        with self._lock:

            for child in self._elements.values():

                if (
                    child.parent_id
                    == element_id
                ):

                    child.parent_id = None

        self._emit(
            "removed",
            element,
        )

        return True

    def clear(self) -> None:

        with self._lock:

            elements = list(
                self._elements.values()
            )

            self._elements.clear()

        for element in elements:

            self._emit(
                "removed",
                element,
            )

    # ========================================================================
    # GETTERS
    # ========================================================================

    def get(
        self,
        element_id: str,
    ) -> Optional[HUDElement]:

        with self._lock:

            return self._elements.get(
                element_id
            )

    def get_all(self) -> list[HUDElement]:

        with self._lock:

            return list(
                self._elements.values()
            )

    def get_visible(self) -> list[HUDElement]:

        with self._lock:

            return [
                element
                for element
                in self._elements.values()
                if element.visible
                and element.state
                != HUDState.HIDDEN
            ]

    def get_by_type(
        self,
        element_type: HUDElementType
        | str,
    ) -> list[HUDElement]:

        if isinstance(
            element_type,
            str,
        ):

            try:

                element_type = (
                    HUDElementType(
                        element_type.lower()
                    )
                )

            except ValueError:

                return []

        with self._lock:

            return [
                element
                for element
                in self._elements.values()
                if element.element_type
                == element_type
            ]

    # ========================================================================
    # TEXT / VALUE
    # ========================================================================

    def set_text(
        self,
        element_id: str,
        text: str,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.text = str(text)

        self._touch(element)

        self._emit(
            "text_changed",
            element,
        )

        return True

    def append_text(
        self,
        element_id: str,
        text: str,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.text += str(text)

        self._touch(element)

        self._emit(
            "text_changed",
            element,
        )

        return True

    def set_value(
        self,
        element_id: str,
        value: Any,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.value = value

        self._touch(element)

        self._emit(
            "value_changed",
            element,
        )

        return True

    # ========================================================================
    # POSITION
    # ========================================================================

    def set_position(
        self,
        element_id: str,
        x: float,
        y: float,
        *,
        z: Optional[float] = None,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.position.x = float(x)
        element.position.y = float(y)

        if z is not None:
            element.position.z = float(z)

        self._touch(element)

        self._emit(
            "position_changed",
            element,
        )

        return True

    def move(
        self,
        element_id: str,
        dx: float,
        dy: float,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.position.x += float(dx)
        element.position.y += float(dy)

        self._touch(element)

        self._emit(
            "position_changed",
            element,
        )

        return True

    def set_anchor(
        self,
        element_id: str,
        anchor: HUDAnchor | str,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        if isinstance(
            anchor,
            str,
        ):

            try:

                anchor = HUDAnchor(
                    anchor.lower()
                )

            except ValueError:

                return False

        element.position.anchor = anchor

        self._touch(element)

        self._emit(
            "position_changed",
            element,
        )

        return True

    # ========================================================================
    # STYLE
    # ========================================================================

    def set_opacity(
        self,
        element_id: str,
        opacity: float,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.style.opacity = max(
            0.0,
            min(
                1.0,
                float(opacity),
            ),
        )

        self._touch(element)

        self._emit(
            "style_changed",
            element,
        )

        return True

    def set_glow(
        self,
        element_id: str,
        glow: float,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.style.glow = max(
            0.0,
            float(glow),
        )

        self._touch(element)

        self._emit(
            "style_changed",
            element,
        )

        return True

    def set_scale(
        self,
        element_id: str,
        scale: float,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.style.scale = max(
            0.01,
            float(scale),
        )

        self._touch(element)

        self._emit(
            "style_changed",
            element,
        )

        return True

    def set_color(
        self,
        element_id: str,
        r: float,
        g: float,
        b: float,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.style.color = (
            max(0.0, min(1.0, float(r))),
            max(0.0, min(1.0, float(g))),
            max(0.0, min(1.0, float(b))),
        )

        self._touch(element)

        self._emit(
            "style_changed",
            element,
        )

        return True

    # ========================================================================
    # VISIBILITY
    # ========================================================================

    def show(
        self,
        element_id: str,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.visible = True

        if (
            element.state
            == HUDState.HIDDEN
        ):

            element.state = (
                HUDState.VISIBLE
            )

        self._touch(element)

        self._emit(
            "visibility_changed",
            element,
        )

        return True

    def hide(
        self,
        element_id: str,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.visible = False

        element.state = HUDState.HIDDEN

        self._touch(element)

        self._emit(
            "visibility_changed",
            element,
        )

        return True

    # ========================================================================
    # STATES
    # ========================================================================

    def set_state(
        self,
        element_id: str,
        state: HUDState | str,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        if isinstance(
            state,
            str,
        ):

            try:

                state = HUDState(
                    state.lower()
                )

            except ValueError:

                return False

        element.state = state

        if state == HUDState.HIDDEN:

            element.visible = False

        elif state != HUDState.HIDDEN:

            element.visible = True

        self._touch(element)

        self._emit(
            "state_changed",
            element,
        )

        return True

    # ========================================================================
    # PROGRESS
    # ========================================================================

    def set_progress(
        self,
        element_id: str,
        value: float,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        if maximum <= minimum:

            raise ValueError(
                "maximum must be greater than minimum"
            )

        value = max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

        element.value = value

        normalized = (
            value - minimum
        ) / (
            maximum - minimum
        )

        element.metadata[
            "progress"
        ] = normalized

        element.metadata[
            "minimum"
        ] = minimum

        element.metadata[
            "maximum"
        ] = maximum

        self._touch(element)

        self._emit(
            "progress_changed",
            element,
            normalized,
        )

        return True

    # ========================================================================
    # STATUS HELPERS
    # ========================================================================

    def set_status(
        self,
        element_id: str,
        status: str,
        *,
        state: Optional[
            HUDState | str
        ] = None,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.value = status

        element.metadata[
            "status"
        ] = status

        if state is not None:

            self.set_state(
                element_id,
                state,
            )

        self._touch(element)

        self._emit(
            "status_changed",
            element,
        )

        return True

    # ========================================================================
    # PARENTING
    # ========================================================================

    def set_parent(
        self,
        element_id: str,
        parent_id: Optional[str],
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        if parent_id == element_id:
            return False

        if parent_id is not None:

            parent = self.get(
                parent_id
            )

            if parent is None:
                return False

            visited: set[str] = set()

            current = parent

            while current is not None:

                if current.id in visited:
                    return False

                visited.add(
                    current.id
                )

                if (
                    current.parent_id
                    == element_id
                ):

                    return False

                if current.parent_id is None:
                    break

                current = self.get(
                    current.parent_id
                )

        element.parent_id = parent_id

        self._touch(element)

        self._emit(
            "parent_changed",
            element,
        )

        return True

    # ========================================================================
    # LAYER
    # ========================================================================

    def set_layer(
        self,
        element_id: str,
        layer: int,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        element.layer = int(layer)

        self._touch(element)

        self._emit(
            "layer_changed",
            element,
        )

        return True

    def sorted_visible(
        self,
    ) -> list[HUDElement]:

        return sorted(
            self.get_visible(),
            key=lambda element: element.layer,
        )

    # ========================================================================
    # ANIMATION
    # ========================================================================

    def set_animation(
        self,
        element_id: str,
        *,
        enabled: Optional[bool] = None,
        pulse: Optional[bool] = None,
        pulse_speed: Optional[float] = None,
        pulse_amount: Optional[float] = None,
        fade: Optional[bool] = None,
        fade_speed: Optional[float] = None,
        slide: Optional[bool] = None,
        slide_speed: Optional[float] = None,
    ) -> bool:

        element = self.get(
            element_id
        )

        if element is None:
            return False

        animation = element.animation

        if enabled is not None:
            animation.enabled = bool(
                enabled
            )

        if pulse is not None:
            animation.pulse = bool(
                pulse
            )

        if pulse_speed is not None:
            animation.pulse_speed = float(
                pulse_speed
            )

        if pulse_amount is not None:
            animation.pulse_amount = max(
                0.0,
                float(pulse_amount),
            )

        if fade is not None:
            animation.fade = bool(
                fade
            )

        if fade_speed is not None:
            animation.fade_speed = max(
                0.01,
                float(fade_speed),
            )

        if slide is not None:
            animation.slide = bool(
                slide
            )

        if slide_speed is not None:
            animation.slide_speed = max(
                0.01,
                float(slide_speed),
            )

        self._touch(element)

        self._emit(
            "animation_changed",
            element,
        )

        return True

    def _update_animation(
        self,
        element: HUDElement,
        delta_time: float,
    ) -> None:

        animation = element.animation

        # Pulse glow/opacity.
        if animation.pulse:

            import math

            pulse = (
                math.sin(
                    self._time
                    * animation.pulse_speed
                )
                * animation.pulse_amount
            )

            element.style.glow = max(
                0.0,
                1.0 + pulse,
            )

        element.updated_at = time.time()

        self._emit(
            "animated",
            element,
            delta_time,
        )

    # ========================================================================
    # COMMON RENIX HUD COMPONENTS
    # ========================================================================

    def create_voice_indicator(
        self,
        *,
        element_id: str = "voice_indicator",
    ) -> HUDElement:

        return self.create(
            HUDElementType.INDICATOR,
            name="Voice Indicator",
            element_id=element_id,
            text="READY",
            position=HUDPosition(
                x=0.5,
                y=0.88,
                anchor=HUDAnchor.BOTTOM_CENTER,
            ),
            style=HUDStyle(
                font_size=18,
                glow=1.5,
            ),
            layer=100,
            metadata={
                "component": "voice",
                "state": "idle",
            },
        )

    def create_gesture_indicator(
        self,
        *,
        element_id: str = "gesture_indicator",
    ) -> HUDElement:

        return self.create(
            HUDElementType.INDICATOR,
            name="Gesture Indicator",
            element_id=element_id,
            text="GESTURE READY",
            position=HUDPosition(
                x=0.5,
                y=0.93,
                anchor=HUDAnchor.BOTTOM_CENTER,
            ),
            style=HUDStyle(
                font_size=14,
                glow=1.2,
            ),
            layer=101,
            metadata={
                "component": "gesture",
                "gesture": None,
            },
        )

    def create_system_status(
        self,
        *,
        element_id: str = "system_status",
    ) -> HUDElement:

        return self.create(
            HUDElementType.STATUS,
            name="System Status",
            element_id=element_id,
            text="RENIX ONLINE",
            value="online",
            position=HUDPosition(
                x=0.03,
                y=0.04,
                anchor=HUDAnchor.TOP_LEFT,
            ),
            style=HUDStyle(
                font_size=16,
                glow=1.4,
            ),
            layer=110,
            metadata={
                "component": "system_status",
            },
        )

    def create_assistant_text(
        self,
        *,
        element_id: str = "assistant_response",
    ) -> HUDElement:

        return self.create(
            HUDElementType.TEXT,
            name="Assistant Response",
            element_id=element_id,
            text="",
            position=HUDPosition(
                x=0.5,
                y=0.76,
                anchor=HUDAnchor.BOTTOM_CENTER,
            ),
            style=HUDStyle(
                font_size=22,
                glow=1.3,
            ),
            layer=120,
            metadata={
                "component": "assistant_response",
            },
        )

    def create_command_text(
        self,
        *,
        element_id: str = "command_text",
    ) -> HUDElement:

        return self.create(
            HUDElementType.TEXT,
            name="Command Text",
            element_id=element_id,
            text="",
            position=HUDPosition(
                x=0.5,
                y=0.70,
                anchor=HUDAnchor.BOTTOM_CENTER,
            ),
            style=HUDStyle(
                font_size=18,
                glow=1.1,
            ),
            layer=121,
            metadata={
                "component": "command",
            },
        )

    # ========================================================================
    # CALLBACK SYSTEM
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
                    "HUD callback failed: %s",
                    event,
                )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    @property
    def running(self) -> bool:

        with self._lock:

            return self._running

    def to_dict(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "count": len(
                    self._elements
                ),
                "visible": len(
                    [
                        element
                        for element
                        in self._elements.values()
                        if element.visible
                    ]
                ),
                "elements": [
                    self._element_to_dict(
                        element
                    )
                    for element
                    in self._elements.values()
                ],
            }

    @staticmethod
    def _element_to_dict(
        element: HUDElement,
    ) -> dict[str, Any]:

        return {
            "id": element.id,
            "name": element.name,
            "type": (
                element.element_type.value
            ),
            "text": element.text,
            "value": element.value,
            "state": element.state.value,
            "visible": element.visible,
            "interactive": element.interactive,
            "layer": element.layer,
            "parent_id": element.parent_id,
            "position": {
                "x": element.position.x,
                "y": element.position.y,
                "z": element.position.z,
                "anchor": (
                    element.position.anchor.value
                ),
                "offset_x": (
                    element.position.offset_x
                ),
                "offset_y": (
                    element.position.offset_y
                ),
            },
            "style": {
                "opacity": element.style.opacity,
                "scale": element.style.scale,
                "glow": element.style.glow,
                "font_size": (
                    element.style.font_size
                ),
                "line_width": (
                    element.style.line_width
                ),
                "color": element.style.color,
                "background_color": (
                    element.style.background_color
                ),
                "background_opacity": (
                    element.style.background_opacity
                ),
            },
            "animation": {
                "enabled": (
                    element.animation.enabled
                ),
                "pulse": (
                    element.animation.pulse
                ),
                "pulse_speed": (
                    element.animation.pulse_speed
                ),
                "pulse_amount": (
                    element.animation.pulse_amount
                ),
                "fade": (
                    element.animation.fade
                ),
                "slide": (
                    element.animation.slide
                ),
            },
            "metadata": dict(
                element.metadata
            ),
        }

    # ========================================================================
    # INTERNAL
    # ========================================================================

    @staticmethod
    def _touch(
        element: HUDElement,
    ) -> None:

        element.updated_at = time.time()


__all__ = [
    "HUDElementType",
    "HUDState",
    "HUDAnchor",
    "HUDPosition",
    "HUDStyle",
    "HUDAnimation",
    "HUDElement",
    "HUDEngine",
]


