"""
RENIX Holographic UI - Hologram Engine
======================================

Core visual-state engine for the RENIX holographic interface.

Responsibilities:
    - Manage holographic objects
    - Create/update/remove holograms
    - Manage transforms
    - Visibility and opacity
    - Glow/intensity state
    - Selection and hover state
    - Pulse effects
    - Basic animation state
    - Layer management
    - Render-state generation

Rendering backends can consume the state generated here.
This module intentionally does not depend on a specific graphics engine.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger(
    "RENIX.holographic_ui.hologram_engine"
)


# ============================================================================
# ENUMS
# ============================================================================


class HologramType(str, Enum):
    PANEL = "panel"
    TEXT = "text"
    ICON = "icon"
    BUTTON = "button"
    WINDOW = "window"
    CURSOR = "cursor"
    ORB = "orb"
    RING = "ring"
    PARTICLE = "particle"
    MODEL = "model"
    CUSTOM = "custom"


class HologramState(str, Enum):
    HIDDEN = "hidden"
    VISIBLE = "visible"
    HOVERED = "hovered"
    SELECTED = "selected"
    ACTIVE = "active"
    DISABLED = "disabled"


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class HologramTransform:
    """Position, rotation and scale."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0

    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0


@dataclass
class HologramMaterial:
    """Visual material properties."""

    opacity: float = 1.0

    glow: float = 1.0

    emission: float = 1.0

    transparency: float = 0.0

    metallic: float = 0.0

    roughness: float = 0.5

    # RGB/RGBA-independent generic values.
    color: tuple[float, float, float] = (
        0.0,
        1.0,
        0.8,
    )


@dataclass
class HologramAnimation:
    """Animation state."""

    enabled: bool = False

    rotation_speed: float = 0.0

    pulse_speed: float = 0.0

    pulse_amount: float = 0.0

    float_speed: float = 0.0

    float_amount: float = 0.0

    phase: float = 0.0


@dataclass
class Hologram:
    """Complete hologram object."""

    id: str

    hologram_type: HologramType = (
        HologramType.CUSTOM
    )

    name: str = ""

    transform: HologramTransform = field(
        default_factory=HologramTransform
    )

    material: HologramMaterial = field(
        default_factory=HologramMaterial
    )

    animation: HologramAnimation = field(
        default_factory=HologramAnimation
    )

    state: HologramState = (
        HologramState.VISIBLE
    )

    visible: bool = True

    interactive: bool = True

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
# HOLOGRAM ENGINE
# ============================================================================


class HologramEngine:
    """
    Backend-independent holographic object engine.
    """

    def __init__(
        self,
        *,
        spatial_manager: Any = None,
        animation_engine: Any = None,
    ) -> None:

        self.spatial_manager = spatial_manager

        self.animation_engine = (
            animation_engine
        )

        self._holograms: dict[
            str,
            Hologram,
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
            "Hologram engine created"
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
            "Hologram engine started"
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
            "Hologram engine stopped"
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

            holograms = list(
                self._holograms.values()
            )

        for hologram in holograms:

            if (
                hologram.animation.enabled
                and hologram.visible
            ):

                self._update_animation(
                    hologram,
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
        hologram_type: HologramType
        | str = HologramType.CUSTOM,
        *,
        name: str = "",
        hologram_id: Optional[str] = None,
        transform: Optional[
            HologramTransform
        ] = None,
        material: Optional[
            HologramMaterial
        ] = None,
        animation: Optional[
            HologramAnimation
        ] = None,
        interactive: bool = True,
        layer: int = 0,
        parent_id: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Hologram:

        if isinstance(
            hologram_type,
            str,
        ):

            try:

                hologram_type = (
                    HologramType(
                        hologram_type.lower()
                    )
                )

            except ValueError:

                hologram_type = (
                    HologramType.CUSTOM
                )

        hologram_id = (
            hologram_id
            or uuid.uuid4().hex
        )

        hologram = Hologram(
            id=hologram_id,
            hologram_type=hologram_type,
            name=name,
            transform=(
                transform
                or HologramTransform()
            ),
            material=(
                material
                or HologramMaterial()
            ),
            animation=(
                animation
                or HologramAnimation()
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

        self.add(hologram)

        return hologram

    def add(
        self,
        hologram: Hologram,
    ) -> Hologram:

        if not hologram.id:
            raise ValueError(
                "Hologram ID cannot be empty"
            )

        with self._lock:

            hologram.updated_at = time.time()

            self._holograms[
                hologram.id
            ] = hologram

        self._emit(
            "created",
            hologram,
        )

        return hologram

    # ========================================================================
    # REMOVE
    # ========================================================================

    def remove(
        self,
        hologram_id: str,
    ) -> bool:

        with self._lock:

            hologram = self._holograms.pop(
                hologram_id,
                None,
            )

        if hologram is None:
            return False

        # Remove children from their parent
        # relationship without recursively deleting
        # them.
        with self._lock:

            for child in self._holograms.values():

                if (
                    child.parent_id
                    == hologram_id
                ):

                    child.parent_id = None

        self._emit(
            "removed",
            hologram,
        )

        return True

    def clear(self) -> None:

        with self._lock:

            holograms = list(
                self._holograms.values()
            )

            self._holograms.clear()

        for hologram in holograms:

            self._emit(
                "removed",
                hologram,
            )

    # ========================================================================
    # GETTERS
    # ========================================================================

    def get(
        self,
        hologram_id: str,
    ) -> Optional[Hologram]:

        with self._lock:

            return self._holograms.get(
                hologram_id
            )

    def get_all(self) -> list[Hologram]:

        with self._lock:

            return list(
                self._holograms.values()
            )

    def get_visible(self) -> list[Hologram]:

        with self._lock:

            return [
                h
                for h in self._holograms.values()
                if h.visible
                and h.state
                != HologramState.HIDDEN
            ]

    def get_by_layer(
        self,
        layer: int,
    ) -> list[Hologram]:

        with self._lock:

            return [
                h
                for h in self._holograms.values()
                if h.layer == layer
            ]

    # ========================================================================
    # VISIBILITY
    # ========================================================================

    def show(
        self,
        hologram_id: str,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.visible = True

        if (
            hologram.state
            == HologramState.HIDDEN
        ):

            hologram.state = (
                HologramState.VISIBLE
            )

        self._touch(
            hologram
        )

        self._emit(
            "visibility_changed",
            hologram,
        )

        return True

    def hide(
        self,
        hologram_id: str,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.visible = False

        hologram.state = (
            HologramState.HIDDEN
        )

        self._touch(
            hologram
        )

        self._emit(
            "visibility_changed",
            hologram,
        )

        return True

    def set_visibility(
        self,
        hologram_id: str,
        visible: bool,
    ) -> bool:

        if visible:
            return self.show(
                hologram_id
            )

        return self.hide(
            hologram_id
        )

    # ========================================================================
    # TRANSFORM
    # ========================================================================

    def set_position(
        self,
        hologram_id: str,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.transform.x = float(x)
        hologram.transform.y = float(y)
        hologram.transform.z = float(z)

        self._touch(
            hologram
        )

        self._emit(
            "transform_changed",
            hologram,
        )

        return True

    def move(
        self,
        hologram_id: str,
        dx: float,
        dy: float,
        dz: float = 0.0,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.transform.x += float(dx)
        hologram.transform.y += float(dy)
        hologram.transform.z += float(dz)

        self._touch(
            hologram
        )

        self._emit(
            "transform_changed",
            hologram,
        )

        return True

    def set_rotation(
        self,
        hologram_id: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.transform.rotation_x = (
            float(x)
        )

        hologram.transform.rotation_y = (
            float(y)
        )

        hologram.transform.rotation_z = (
            float(z)
        )

        self._touch(
            hologram
        )

        self._emit(
            "transform_changed",
            hologram,
        )

        return True

    def rotate(
        self,
        hologram_id: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.transform.rotation_x += (
            float(x)
        )

        hologram.transform.rotation_y += (
            float(y)
        )

        hologram.transform.rotation_z += (
            float(z)
        )

        self._touch(
            hologram
        )

        self._emit(
            "transform_changed",
            hologram,
        )

        return True

    def set_scale(
        self,
        hologram_id: str,
        x: float,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        if y is None:
            y = x

        if z is None:
            z = x

        hologram.transform.scale_x = max(
            0.001,
            float(x),
        )

        hologram.transform.scale_y = max(
            0.001,
            float(y),
        )

        hologram.transform.scale_z = max(
            0.001,
            float(z),
        )

        self._touch(
            hologram
        )

        self._emit(
            "transform_changed",
            hologram,
        )

        return True

    def scale(
        self,
        hologram_id: str,
        factor: float,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        factor = max(
            0.001,
            float(factor),
        )

        hologram.transform.scale_x *= factor
        hologram.transform.scale_y *= factor
        hologram.transform.scale_z *= factor

        self._touch(
            hologram
        )

        self._emit(
            "transform_changed",
            hologram,
        )

        return True

    # ========================================================================
    # MATERIAL
    # ========================================================================

    def set_opacity(
        self,
        hologram_id: str,
        opacity: float,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.material.opacity = max(
            0.0,
            min(
                1.0,
                float(opacity),
            ),
        )

        self._touch(
            hologram
        )

        self._emit(
            "material_changed",
            hologram,
        )

        return True

    def set_glow(
        self,
        hologram_id: str,
        glow: float,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.material.glow = max(
            0.0,
            float(glow),
        )

        self._touch(
            hologram
        )

        self._emit(
            "material_changed",
            hologram,
        )

        return True

    def set_color(
        self,
        hologram_id: str,
        r: float,
        g: float,
        b: float,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.material.color = (
            max(0.0, min(1.0, float(r))),
            max(0.0, min(1.0, float(g))),
            max(0.0, min(1.0, float(b))),
        )

        self._touch(
            hologram
        )

        self._emit(
            "material_changed",
            hologram,
        )

        return True

    # ========================================================================
    # INTERACTION STATES
    # ========================================================================

    def hover(
        self,
        hologram_id: str,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        if not hologram.interactive:
            return False

        hologram.state = (
            HologramState.HOVERED
        )

        hologram.material.glow = max(
            hologram.material.glow,
            1.4,
        )

        self._touch(
            hologram
        )

        self._emit(
            "hovered",
            hologram,
        )

        return True

    def select(
        self,
        hologram_id: str,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        if not hologram.interactive:
            return False

        hologram.state = (
            HologramState.SELECTED
        )

        self._touch(
            hologram
        )

        self._emit(
            "selected",
            hologram,
        )

        return True

    def activate(
        self,
        hologram_id: str,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        if not hologram.interactive:
            return False

        hologram.state = (
            HologramState.ACTIVE
        )

        self._touch(
            hologram
        )

        self._emit(
            "activated",
            hologram,
        )

        return True

    def disable(
        self,
        hologram_id: str,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.interactive = False

        hologram.state = (
            HologramState.DISABLED
        )

        self._touch(
            hologram
        )

        self._emit(
            "disabled",
            hologram,
        )

        return True

    def enable(
        self,
        hologram_id: str,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.interactive = True

        hologram.state = (
            HologramState.VISIBLE
            if hologram.visible
            else HologramState.HIDDEN
        )

        self._touch(
            hologram
        )

        self._emit(
            "enabled",
            hologram,
        )

        return True

    def reset_state(
        self,
        hologram_id: str,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        if not hologram.visible:

            hologram.state = (
                HologramState.HIDDEN
            )

        elif not hologram.interactive:

            hologram.state = (
                HologramState.DISABLED
            )

        else:

            hologram.state = (
                HologramState.VISIBLE
            )

        self._touch(
            hologram
        )

        return True

    # ========================================================================
    # ANIMATION
    # ========================================================================

    def set_animation(
        self,
        hologram_id: str,
        *,
        enabled: Optional[bool] = None,
        rotation_speed: Optional[
            float
        ] = None,
        pulse_speed: Optional[
            float
        ] = None,
        pulse_amount: Optional[
            float
        ] = None,
        float_speed: Optional[
            float
        ] = None,
        float_amount: Optional[
            float
        ] = None,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        animation = hologram.animation

        if enabled is not None:
            animation.enabled = bool(
                enabled
            )

        if rotation_speed is not None:
            animation.rotation_speed = float(
                rotation_speed
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

        if float_speed is not None:
            animation.float_speed = float(
                float_speed
            )

        if float_amount is not None:
            animation.float_amount = max(
                0.0,
                float(float_amount),
            )

        self._touch(
            hologram
        )

        self._emit(
            "animation_changed",
            hologram,
        )

        return True

    def _update_animation(
        self,
        hologram: Hologram,
        delta_time: float,
    ) -> None:

        animation = hologram.animation

        animation.phase += delta_time

        # Rotation
        if animation.rotation_speed != 0:

            hologram.transform.rotation_z += (
                animation.rotation_speed
                * delta_time
            )

        # Floating movement
        if (
            animation.float_speed != 0
            and animation.float_amount != 0
        ):

            offset = (
                math.sin(
                    animation.phase
                    * animation.float_speed
                )
                * animation.float_amount
            )

            hologram.transform.y += (
                offset * delta_time
            )

        # Pulsing glow
        if (
            animation.pulse_speed != 0
            and animation.pulse_amount != 0
        ):

            pulse = (
                math.sin(
                    animation.phase
                    * animation.pulse_speed
                )
                * animation.pulse_amount
            )

            hologram.material.glow = max(
                0.0,
                1.0 + pulse,
            )

        hologram.updated_at = time.time()

        self._emit(
            "animated",
            hologram,
            delta_time,
        )

    # ========================================================================
    # PARENTING
    # ========================================================================

    def set_parent(
        self,
        hologram_id: str,
        parent_id: Optional[str],
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        if parent_id == hologram_id:
            return False

        if parent_id is not None:

            parent = self.get(
                parent_id
            )

            if parent is None:
                return False

            # Prevent simple parent cycles.
            current = parent

            visited: set[str] = set()

            while current is not None:

                if current.id in visited:
                    return False

                visited.add(
                    current.id
                )

                if (
                    current.parent_id
                    == hologram_id
                ):
                    return False

                if current.parent_id is None:
                    break

                current = self.get(
                    current.parent_id
                )

        hologram.parent_id = parent_id

        self._touch(
            hologram
        )

        self._emit(
            "parent_changed",
            hologram,
        )

        return True

    # ========================================================================
    # LAYER
    # ========================================================================

    def set_layer(
        self,
        hologram_id: str,
        layer: int,
    ) -> bool:

        hologram = self.get(
            hologram_id
        )

        if hologram is None:
            return False

        hologram.layer = int(
            layer
        )

        self._touch(
            hologram
        )

        self._emit(
            "layer_changed",
            hologram,
        )

        return True

    def sorted_visible(
        self,
    ) -> list[Hologram]:

        return sorted(
            self.get_visible(),
            key=lambda h: h.layer,
        )

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _touch(
        hologram: Hologram,
    ) -> None:

        hologram.updated_at = time.time()

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
                    "Hologram callback failed: %s",
                    event,
                )

    # ========================================================================
    # STATUS / SERIALIZATION
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
                    self._holograms
                ),
                "visible": len(
                    [
                        h
                        for h
                        in self._holograms.values()
                        if h.visible
                    ]
                ),
                "holograms": [
                    self._hologram_to_dict(
                        h
                    )
                    for h in self._holograms.values()
                ],
            }

    @staticmethod
    def _hologram_to_dict(
        hologram: Hologram,
    ) -> dict[str, Any]:

        return {
            "id": hologram.id,
            "name": hologram.name,
            "type": hologram.hologram_type.value,
            "state": hologram.state.value,
            "visible": hologram.visible,
            "interactive": hologram.interactive,
            "layer": hologram.layer,
            "parent_id": hologram.parent_id,
            "transform": {
                "x": hologram.transform.x,
                "y": hologram.transform.y,
                "z": hologram.transform.z,
                "rotation_x": (
                    hologram.transform.rotation_x
                ),
                "rotation_y": (
                    hologram.transform.rotation_y
                ),
                "rotation_z": (
                    hologram.transform.rotation_z
                ),
                "scale_x": (
                    hologram.transform.scale_x
                ),
                "scale_y": (
                    hologram.transform.scale_y
                ),
                "scale_z": (
                    hologram.transform.scale_z
                ),
            },
            "material": {
                "opacity": (
                    hologram.material.opacity
                ),
                "glow": (
                    hologram.material.glow
                ),
                "emission": (
                    hologram.material.emission
                ),
                "transparency": (
                    hologram.material.transparency
                ),
                "color": (
                    hologram.material.color
                ),
            },
            "animation": {
                "enabled": (
                    hologram.animation.enabled
                ),
                "rotation_speed": (
                    hologram.animation.rotation_speed
                ),
                "pulse_speed": (
                    hologram.animation.pulse_speed
                ),
                "pulse_amount": (
                    hologram.animation.pulse_amount
                ),
                "float_speed": (
                    hologram.animation.float_speed
                ),
                "float_amount": (
                    hologram.animation.float_amount
                ),
            },
            "metadata": dict(
                hologram.metadata
            ),
        }


__all__ = [
    "HologramType",
    "HologramState",
    "HologramTransform",
    "HologramMaterial",
    "HologramAnimation",
    "Hologram",
    "HologramEngine",
]


