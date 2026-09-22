"""
RENIX Holographic UI - Spatial Manager
======================================

Manages the spatial coordinate system used by the RENIX holographic UI.

Responsibilities:
    - 2D/3D coordinate conversion
    - Screen <-> normalized coordinates
    - Camera/view configuration
    - Depth management
    - Spatial transforms
    - Distance calculations
    - Point/vector utilities
    - Gesture-friendly coordinate mapping
    - World-space and screen-space positioning

This module does not detect hands or gestures.
Those systems belong to vision/ and gestures/.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(
    "RENIX.holographic_ui.spatial_manager"
)


# ============================================================================
# VECTOR TYPES
# ============================================================================


@dataclass
class Vector2:
    """Two-dimensional vector."""

    x: float = 0.0
    y: float = 0.0

    def copy(self) -> "Vector2":
        return Vector2(self.x, self.y)

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(
            self.x - other.x,
            self.y - other.y,
        )

    def __mul__(self, value: float) -> "Vector2":
        return Vector2(
            self.x * value,
            self.y * value,
        )

    def length(self) -> float:
        return math.sqrt(
            self.x * self.x
            + self.y * self.y
        )

    def normalized(self) -> "Vector2":

        length = self.length()

        if length <= 1e-9:

            return Vector2()

        return Vector2(
            self.x / length,
            self.y / length,
        )

    def distance_to(
        self,
        other: "Vector2",
    ) -> float:

        return (
            self - other
        ).length()

    def to_tuple(self) -> tuple[float, float]:

        return (
            self.x,
            self.y,
        )


@dataclass
class Vector3:
    """Three-dimensional vector."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def copy(self) -> "Vector3":
        return Vector3(
            self.x,
            self.y,
            self.z,
        )

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, value: float) -> "Vector3":
        return Vector3(
            self.x * value,
            self.y * value,
            self.z * value,
        )

    def length(self) -> float:
        return math.sqrt(
            self.x * self.x
            + self.y * self.y
            + self.z * self.z
        )

    def normalized(self) -> "Vector3":

        length = self.length()

        if length <= 1e-9:

            return Vector3()

        return Vector3(
            self.x / length,
            self.y / length,
            self.z / length,
        )

    def distance_to(
        self,
        other: "Vector3",
    ) -> float:

        return (
            self - other
        ).length()

    def dot(
        self,
        other: "Vector3",
    ) -> float:

        return (
            self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )

    def cross(
        self,
        other: "Vector3",
    ) -> "Vector3":

        return Vector3(
            self.y * other.z
            - self.z * other.y,

            self.z * other.x
            - self.x * other.z,

            self.x * other.y
            - self.y * other.x,
        )

    def to_tuple(
        self,
    ) -> tuple[float, float, float]:

        return (
            self.x,
            self.y,
            self.z,
        )


# ============================================================================
# SPATIAL TRANSFORM
# ============================================================================


@dataclass
class SpatialTransform:
    """Position, rotation and scale in 3D space."""

    position: Vector3 = field(
        default_factory=Vector3
    )

    rotation: Vector3 = field(
        default_factory=Vector3
    )

    scale: Vector3 = field(
        default_factory=lambda: Vector3(
            1.0,
            1.0,
            1.0,
        )
    )

    def copy(self) -> "SpatialTransform":

        return SpatialTransform(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            scale=self.scale.copy(),
        )


# ============================================================================
# CAMERA
# ============================================================================


@dataclass
class SpatialCamera:
    """Camera/view configuration for the holographic scene."""

    position: Vector3 = field(
        default_factory=Vector3
    )

    rotation: Vector3 = field(
        default_factory=Vector3
    )

    fov: float = 60.0

    near_clip: float = 0.01

    far_clip: float = 1000.0

    orthographic: bool = False

    orthographic_scale: float = 1.0

    def copy(self) -> "SpatialCamera":

        return SpatialCamera(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            fov=self.fov,
            near_clip=self.near_clip,
            far_clip=self.far_clip,
            orthographic=self.orthographic,
            orthographic_scale=self.orthographic_scale,
        )


# ============================================================================
# SPATIAL MANAGER
# ============================================================================


class SpatialManager:
    """
    Central spatial-coordinate manager for RENIX holographic UI.
    """

    def __init__(
        self,
        *,
        screen_width: int = 1920,
        screen_height: int = 1080,
        world_width: float = 2.0,
        world_height: float = 2.0,
    ) -> None:

        self.screen_width = max(
            1,
            int(screen_width),
        )

        self.screen_height = max(
            1,
            int(screen_height),
        )

        self.world_width = max(
            0.01,
            float(world_width),
        )

        self.world_height = max(
            0.01,
            float(world_height),
        )

        self.camera = SpatialCamera()

        self._objects: dict[
            str,
            SpatialTransform,
        ] = {}

        self._lock = threading.RLock()

        self._running = False

        logger.debug(
            "Spatial manager created"
        )

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
        """
        Update spatial state.

        The spatial manager currently uses an on-demand model,
        so no continuous calculation is required here.
        """

        return None

    # ========================================================================
    # SCREEN CONFIGURATION
    # ========================================================================

    def set_screen_size(
        self,
        width: int,
        height: int,
    ) -> None:

        with self._lock:

            self.screen_width = max(
                1,
                int(width),
            )

            self.screen_height = max(
                1,
                int(height),
            )

    def get_screen_size(
        self,
    ) -> tuple[int, int]:

        with self._lock:

            return (
                self.screen_width,
                self.screen_height,
            )

    # ========================================================================
    # NORMALIZED COORDINATES
    # ========================================================================

    def screen_to_normalized(
        self,
        x: float,
        y: float,
    ) -> Vector2:

        with self._lock:

            nx = (
                float(x)
                / self.screen_width
            )

            ny = (
                float(y)
                / self.screen_height
            )

        return Vector2(
            max(0.0, min(1.0, nx)),
            max(0.0, min(1.0, ny)),
        )

    def normalized_to_screen(
        self,
        x: float,
        y: float,
    ) -> Vector2:

        with self._lock:

            return Vector2(
                float(x)
                * self.screen_width,

                float(y)
                * self.screen_height,
            )

    # ========================================================================
    # CENTERED NORMALIZED COORDINATES
    # ========================================================================

    def screen_to_centered(
        self,
        x: float,
        y: float,
    ) -> Vector2:

        normalized = (
            self.screen_to_normalized(
                x,
                y,
            )
        )

        return Vector2(
            normalized.x * 2.0 - 1.0,
            normalized.y * 2.0 - 1.0,
        )

    def centered_to_screen(
        self,
        x: float,
        y: float,
    ) -> Vector2:

        return self.normalized_to_screen(
            (float(x) + 1.0) / 2.0,
            (float(y) + 1.0) / 2.0,
        )

    # ========================================================================
    # WORLD COORDINATES
    # ========================================================================

    def normalized_to_world(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> Vector3:

        centered = Vector2(
            float(x) * 2.0 - 1.0,
            float(y) * 2.0 - 1.0,
        )

        return Vector3(
            centered.x
            * self.world_width
            / 2.0,

            centered.y
            * self.world_height
            / 2.0,

            float(z),
        )

    def world_to_normalized(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> Vector3:

        if self.world_width == 0:

            nx = 0.5

        else:

            nx = (
                float(x)
                / self.world_width
                + 0.5
            )

        if self.world_height == 0:

            ny = 0.5

        else:

            ny = (
                float(y)
                / self.world_height
                + 0.5
            )

        return Vector3(
            max(0.0, min(1.0, nx)),
            max(0.0, min(1.0, ny)),
            float(z),
        )

    # ========================================================================
    # SCREEN <-> WORLD
    # ========================================================================

    def screen_to_world(
        self,
        x: float,
        y: float,
        *,
        depth: float = 0.0,
    ) -> Vector3:

        normalized = (
            self.screen_to_normalized(
                x,
                y,
            )
        )

        return self.normalized_to_world(
            normalized.x,
            normalized.y,
            depth,
        )

    def world_to_screen(
        self,
        position: Vector3,
    ) -> Vector2:

        normalized = (
            self.world_to_normalized(
                position.x,
                position.y,
                position.z,
            )
        )

        return self.normalized_to_screen(
            normalized.x,
            normalized.y,
        )

    # ========================================================================
    # DEPTH
    # ========================================================================

    @staticmethod
    def clamp_depth(
        depth: float,
        minimum: float = -1000.0,
        maximum: float = 1000.0,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                float(depth),
            ),
        )

    def depth_from_distance(
        self,
        distance: float,
        *,
        base_depth: float = 0.0,
    ) -> float:

        return self.clamp_depth(
            base_depth + float(distance)
        )

    # ========================================================================
    # CAMERA
    # ========================================================================

    def set_camera_position(
        self,
        x: float,
        y: float,
        z: float,
    ) -> None:

        with self._lock:

            self.camera.position = Vector3(
                x,
                y,
                z,
            )

    def set_camera_rotation(
        self,
        x: float,
        y: float,
        z: float,
    ) -> None:

        with self._lock:

            self.camera.rotation = Vector3(
                x,
                y,
                z,
            )

    def set_camera_fov(
        self,
        fov: float,
    ) -> None:

        with self._lock:

            self.camera.fov = max(
                1.0,
                min(
                    179.0,
                    float(fov),
                ),
            )

    def get_camera(
        self,
    ) -> SpatialCamera:

        with self._lock:

            return self.camera.copy()

    # ========================================================================
    # OBJECT REGISTRY
    # ========================================================================

    def register_object(
        self,
        object_id: str,
        transform: Optional[
            SpatialTransform
        ] = None,
    ) -> SpatialTransform:

        if not object_id:

            raise ValueError(
                "object_id cannot be empty"
            )

        with self._lock:

            if transform is None:

                transform = SpatialTransform()

            self._objects[
                object_id
            ] = transform.copy()

            return self._objects[
                object_id
            ]

    def unregister_object(
        self,
        object_id: str,
    ) -> bool:

        with self._lock:

            return (
                self._objects.pop(
                    object_id,
                    None,
                )
                is not None
            )

    def get_object_transform(
        self,
        object_id: str,
    ) -> Optional[SpatialTransform]:

        with self._lock:

            transform = self._objects.get(
                object_id
            )

            if transform is None:

                return None

            return transform.copy()

    def set_object_transform(
        self,
        object_id: str,
        transform: SpatialTransform,
    ) -> bool:

        with self._lock:

            if object_id not in self._objects:

                return False

            self._objects[
                object_id
            ] = transform.copy()

            return True

    # ========================================================================
    # OBJECT MOVEMENT
    # ========================================================================

    def move_object(
        self,
        object_id: str,
        dx: float,
        dy: float,
        dz: float = 0.0,
    ) -> bool:

        with self._lock:

            transform = self._objects.get(
                object_id
            )

            if transform is None:

                return False

            transform.position.x += dx
            transform.position.y += dy
            transform.position.z += dz

            return True

    def rotate_object(
        self,
        object_id: str,
        dx: float,
        dy: float,
        dz: float = 0.0,
    ) -> bool:

        with self._lock:

            transform = self._objects.get(
                object_id
            )

            if transform is None:

                return False

            transform.rotation.x += dx
            transform.rotation.y += dy
            transform.rotation.z += dz

            return True

    def scale_object(
        self,
        object_id: str,
        factor: float,
    ) -> bool:

        with self._lock:

            transform = self._objects.get(
                object_id
            )

            if transform is None:

                return False

            factor = max(
                0.001,
                float(factor),
            )

            transform.scale.x *= factor
            transform.scale.y *= factor
            transform.scale.z *= factor

            return True

    # ========================================================================
    # DISTANCE / SPATIAL UTILITIES
    # ========================================================================

    @staticmethod
    def distance_2d(
        a: Vector2,
        b: Vector2,
    ) -> float:

        return a.distance_to(b)

    @staticmethod
    def distance_3d(
        a: Vector3,
        b: Vector3,
    ) -> float:

        return a.distance_to(b)

    @staticmethod
    def midpoint_2d(
        a: Vector2,
        b: Vector2,
    ) -> Vector2:

        return Vector2(
            (a.x + b.x) / 2.0,
            (a.y + b.y) / 2.0,
        )

    @staticmethod
    def midpoint_3d(
        a: Vector3,
        b: Vector3,
    ) -> Vector3:

        return Vector3(
            (a.x + b.x) / 2.0,
            (a.y + b.y) / 2.0,
            (a.z + b.z) / 2.0,
        )

    @staticmethod
    def lerp(
        a: Vector3,
        b: Vector3,
        amount: float,
    ) -> Vector3:

        t = max(
            0.0,
            min(
                1.0,
                float(amount),
            ),
        )

        return Vector3(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t,
        )

    # ========================================================================
    # GESTURE COORDINATE MAPPING
    # ========================================================================

    def map_gesture_point(
        self,
        x: float,
        y: float,
        *,
        smoothing: float = 0.0,
    ) -> Vector2:
        """
        Convert a normalized hand/gesture point into UI coordinates.

        x/y are expected to be normalized values in [0, 1].

        smoothing is reserved for higher-level tracking systems and is
        intentionally applied only as a simple center bias here.
        """

        x = max(
            0.0,
            min(
                1.0,
                float(x),
            ),
        )

        y = max(
            0.0,
            min(
                1.0,
                float(y),
            ),
        )

        smoothing = max(
            0.0,
            min(
                1.0,
                float(smoothing),
            ),
        )

        # Small smoothing factor toward the center.
        # Temporal smoothing should ultimately be handled by the gesture
        # tracker because it has access to frame history.
        if smoothing > 0.0:

            x = (
                x * (1.0 - smoothing)
                + 0.5 * smoothing
            )

            y = (
                y * (1.0 - smoothing)
                + 0.5 * smoothing
            )

        return Vector2(
            x,
            y,
        )

    # ========================================================================
    # CLAMPING
    # ========================================================================

    @staticmethod
    def clamp_vector2(
        vector: Vector2,
        minimum: Vector2,
        maximum: Vector2,
    ) -> Vector2:

        return Vector2(
            max(
                minimum.x,
                min(
                    maximum.x,
                    vector.x,
                ),
            ),
            max(
                minimum.y,
                min(
                    maximum.y,
                    vector.y,
                ),
            ),
        )

    @staticmethod
    def clamp_vector3(
        vector: Vector3,
        minimum: Vector3,
        maximum: Vector3,
    ) -> Vector3:

        return Vector3(
            max(
                minimum.x,
                min(
                    maximum.x,
                    vector.x,
                ),
            ),
            max(
                minimum.y,
                min(
                    maximum.y,
                    vector.y,
                ),
            ),
            max(
                minimum.z,
                min(
                    maximum.z,
                    vector.z,
                ),
            ),
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "screen": {
                    "width": self.screen_width,
                    "height": self.screen_height,
                },
                "world": {
                    "width": self.world_width,
                    "height": self.world_height,
                },
                "camera": {
                    "position": (
                        self.camera.position.to_tuple()
                    ),
                    "rotation": (
                        self.camera.rotation.to_tuple()
                    ),
                    "fov": self.camera.fov,
                    "near_clip": (
                        self.camera.near_clip
                    ),
                    "far_clip": (
                        self.camera.far_clip
                    ),
                    "orthographic": (
                        self.camera.orthographic
                    ),
                },
                "object_count": len(
                    self._objects
                ),
            }


__all__ = [
    "Vector2",
    "Vector3",
    "SpatialTransform",
    "SpatialCamera",
    "SpatialManager",
]


