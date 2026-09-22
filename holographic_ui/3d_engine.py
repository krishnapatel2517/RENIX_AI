"""
RENIX Holographic UI
3D Engine
==================

Renderer-independent 3D scene utilities for RENIX.

Responsibilities:
- 3D objects
- transforms
- cameras
- projection
- ray casting
- spatial coordinates
- holographic panels
- depth sorting
- object visibility
- interaction points
- scene snapshots

The actual GPU rendering backend can consume the generated
scene snapshot later.
"""

from __future__ import annotations

import math
import threading
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ============================================================================
# ENUMS
# ============================================================================


class ObjectType(str, Enum):
    MESH = "mesh"
    PANEL = "panel"
    HOLOGRAM = "hologram"
    POINT = "point"
    LINE = "line"
    TEXT = "text"
    ICON = "icon"
    PARTICLE = "particle"


class ProjectionType(str, Enum):
    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"


class VisibilityState(str, Enum):
    VISIBLE = "visible"
    HIDDEN = "hidden"


# ============================================================================
# VECTOR 3
# ============================================================================


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def copy(self) -> "Vector3":
        return Vector3(self.x, self.y, self.z)

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

    def __truediv__(self, value: float) -> "Vector3":
        if value == 0:
            raise ZeroDivisionError("Cannot divide Vector3 by zero")

        return Vector3(
            self.x / value,
            self.y / value,
            self.z / value,
        )

    def length(self) -> float:
        return math.sqrt(
            self.x * self.x
            + self.y * self.y
            + self.z * self.z
        )

    def normalized(self) -> "Vector3":
        length = self.length()

        if length <= 1e-8:
            return Vector3()

        return self / length

    def dot(self, other: "Vector3") -> float:
        return (
            self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).length()


# ============================================================================
# ROTATION
# ============================================================================


@dataclass
class Rotation:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def copy(self) -> "Rotation":
        return Rotation(
            self.x,
            self.y,
            self.z,
        )


# ============================================================================
# SCALE
# ============================================================================


@dataclass
class Scale:
    x: float = 1.0
    y: float = 1.0
    z: float = 1.0

    def copy(self) -> "Scale":
        return Scale(
            self.x,
            self.y,
            self.z,
        )


# ============================================================================
# TRANSFORM
# ============================================================================


@dataclass
class Transform:
    position: Vector3 = field(
        default_factory=Vector3
    )

    rotation: Rotation = field(
        default_factory=Rotation
    )

    scale: Scale = field(
        default_factory=Scale
    )

    def copy(self) -> "Transform":
        return Transform(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            scale=self.scale.copy(),
        )


# ============================================================================
# MATERIAL
# ============================================================================


@dataclass
class Material:
    name: str = "holographic"

    color: tuple[float, float, float, float] = (
        0.0,
        1.0,
        0.7,
        1.0,
    )

    emission: float = 1.0

    opacity: float = 1.0

    metallic: float = 0.0

    roughness: float = 0.5

    wireframe: bool = False

    additive: bool = True

    glow: float = 1.0

    texture: Optional[str] = None


# ============================================================================
# BOUNDING BOX
# ============================================================================


@dataclass
class BoundingBox:
    minimum: Vector3 = field(
        default_factory=lambda: Vector3(
            -0.5,
            -0.5,
            -0.5,
        )
    )

    maximum: Vector3 = field(
        default_factory=lambda: Vector3(
            0.5,
            0.5,
            0.5,
        )
    )

    def contains(self, point: Vector3) -> bool:
        return (
            self.minimum.x <= point.x <= self.maximum.x
            and self.minimum.y <= point.y <= self.maximum.y
            and self.minimum.z <= point.z <= self.maximum.z
        )

    def intersects_ray(
        self,
        origin: Vector3,
        direction: Vector3,
    ) -> Optional[float]:

        direction = direction.normalized()

        t_min = -math.inf
        t_max = math.inf

        axes = (
            (
                origin.x,
                direction.x,
                self.minimum.x,
                self.maximum.x,
            ),
            (
                origin.y,
                direction.y,
                self.minimum.y,
                self.maximum.y,
            ),
            (
                origin.z,
                direction.z,
                self.minimum.z,
                self.maximum.z,
            ),
        )

        for origin_axis, direction_axis, min_axis, max_axis in axes:

            if abs(direction_axis) < 1e-8:

                if (
                    origin_axis < min_axis
                    or origin_axis > max_axis
                ):
                    return None

                continue

            inverse = 1.0 / direction_axis

            t1 = (
                min_axis - origin_axis
            ) * inverse

            t2 = (
                max_axis - origin_axis
            ) * inverse

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return None

        if t_max < 0:
            return None

        return max(0.0, t_min)


# ============================================================================
# 3D OBJECT
# ============================================================================


@dataclass
class Object3D:
    id: str

    name: str

    object_type: ObjectType = ObjectType.HOLOGRAM

    transform: Transform = field(
        default_factory=Transform
    )

    material: Material = field(
        default_factory=Material
    )

    bounds: BoundingBox = field(
        default_factory=BoundingBox
    )

    visibility: VisibilityState = (
        VisibilityState.VISIBLE
    )

    interactive: bool = True

    parent_id: Optional[str] = None

    children: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def is_visible(self) -> bool:
        return (
            self.visibility
            == VisibilityState.VISIBLE
        )


# ============================================================================
# CAMERA
# ============================================================================


@dataclass
class Camera:
    position: Vector3 = field(
        default_factory=lambda: Vector3(
            0.0,
            0.0,
            10.0,
        )
    )

    rotation: Rotation = field(
        default_factory=Rotation
    )

    projection: ProjectionType = (
        ProjectionType.PERSPECTIVE
    )

    field_of_view: float = 60.0

    near_clip: float = 0.01

    far_clip: float = 1000.0

    aspect_ratio: float = 16.0 / 9.0

    orthographic_size: float = 10.0


# ============================================================================
# RAY
# ============================================================================


@dataclass
class Ray:
    origin: Vector3

    direction: Vector3

    def normalized(self) -> "Ray":
        return Ray(
            origin=self.origin.copy(),
            direction=self.direction.normalized(),
        )

    def point_at(self, distance: float) -> Vector3:
        return (
            self.origin
            + self.direction.normalized() * distance
        )


# ============================================================================
# RAYCAST RESULT
# ============================================================================


@dataclass
class RaycastHit:
    object_id: str

    object_name: str

    distance: float

    point: Vector3

    normal: Vector3

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# 3D ENGINE
# ============================================================================


class Engine3D:

    def __init__(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
    ) -> None:

        self.width = max(
            1,
            int(width),
        )

        self.height = max(
            1,
            int(height),
        )

        self.camera = Camera(
            aspect_ratio=(
                self.width / self.height
            )
        )

        self._objects: dict[
            str,
            Object3D,
        ] = {}

        self._lock = threading.RLock()

        self._time = 0.0

        self._running = False

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def update(
        self,
        delta_time: float,
    ) -> None:

        if not self._running:
            return

        delta_time = max(
            0.0,
            min(
                0.1,
                float(delta_time),
            ),
        )

        self._time += delta_time

    # ========================================================================
    # OBJECT MANAGEMENT
    # ========================================================================

    def create_object(
        self,
        name: str,
        *,
        object_type: ObjectType = ObjectType.HOLOGRAM,
        transform: Optional[Transform] = None,
        material: Optional[Material] = None,
        bounds: Optional[BoundingBox] = None,
        interactive: bool = True,
        metadata: Optional[dict[str, Any]] = None,
        object_id: Optional[str] = None,
    ) -> Object3D:

        object_3d = Object3D(
            id=(
                object_id
                or uuid.uuid4().hex
            ),
            name=name,
            object_type=object_type,
            transform=(
                transform.copy()
                if transform
                else Transform()
            ),
            material=(
                material
                if material
                else Material()
            ),
            bounds=(
                bounds
                if bounds
                else BoundingBox()
            ),
            interactive=interactive,
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            self._objects[
                object_3d.id
            ] = object_3d

        return object_3d

    def remove_object(
        self,
        object_id: str,
    ) -> bool:

        with self._lock:

            obj = self._objects.pop(
                object_id,
                None,
            )

            if obj is None:
                return False

            if obj.parent_id:

                parent = self._objects.get(
                    obj.parent_id
                )

                if parent:

                    if object_id in parent.children:

                        parent.children.remove(
                            object_id
                        )

            for child_id in list(
                obj.children
            ):

                child = self._objects.get(
                    child_id
                )

                if child:

                    child.parent_id = None

        return True

    def get_object(
        self,
        object_id: str,
    ) -> Optional[Object3D]:

        with self._lock:

            return self._objects.get(
                object_id
            )

    def get_objects(self) -> list[Object3D]:

        with self._lock:

            return list(
                self._objects.values()
            )

    # ========================================================================
    # HIERARCHY
    # ========================================================================

    def set_parent(
        self,
        child_id: str,
        parent_id: Optional[str],
    ) -> bool:

        with self._lock:

            child = self._objects.get(
                child_id
            )

            if child is None:
                return False

            if child.parent_id:

                old_parent = self._objects.get(
                    child.parent_id
                )

                if old_parent:

                    if child_id in old_parent.children:

                        old_parent.children.remove(
                            child_id
                        )

            child.parent_id = parent_id

            if parent_id:

                parent = self._objects.get(
                    parent_id
                )

                if parent is None:

                    child.parent_id = None

                    return False

                if child_id not in parent.children:

                    parent.children.append(
                        child_id
                    )

        return True

    # ========================================================================
    # TRANSFORMS
    # ========================================================================

    def set_position(
        self,
        object_id: str,
        position: Vector3,
    ) -> bool:

        obj = self.get_object(object_id)

        if obj is None:
            return False

        obj.transform.position = (
            position.copy()
        )

        return True

    def move(
        self,
        object_id: str,
        offset: Vector3,
    ) -> bool:

        obj = self.get_object(object_id)

        if obj is None:
            return False

        obj.transform.position += offset

        return True

    def set_rotation(
        self,
        object_id: str,
        rotation: Rotation,
    ) -> bool:

        obj = self.get_object(object_id)

        if obj is None:
            return False

        obj.transform.rotation = (
            rotation.copy()
        )

        return True

    def rotate(
        self,
        object_id: str,
        rotation: Rotation,
    ) -> bool:

        obj = self.get_object(object_id)

        if obj is None:
            return False

        obj.transform.rotation.x += rotation.x
        obj.transform.rotation.y += rotation.y
        obj.transform.rotation.z += rotation.z

        return True

    def set_scale(
        self,
        object_id: str,
        scale: Scale,
    ) -> bool:

        obj = self.get_object(object_id)

        if obj is None:
            return False

        obj.transform.scale = scale.copy()

        return True

    def scale(
        self,
        object_id: str,
        factor: float,
    ) -> bool:

        obj = self.get_object(object_id)

        if obj is None:
            return False

        obj.transform.scale.x *= factor
        obj.transform.scale.y *= factor
        obj.transform.scale.z *= factor

        return True

    # ========================================================================
    # VISIBILITY
    # ========================================================================

    def set_visible(
        self,
        object_id: str,
        visible: bool,
    ) -> bool:

        obj = self.get_object(object_id)

        if obj is None:
            return False

        obj.visibility = (
            VisibilityState.VISIBLE
            if visible
            else VisibilityState.HIDDEN
        )

        return True

    # ========================================================================
    # CAMERA
    # ========================================================================

    def set_camera_position(
        self,
        position: Vector3,
    ) -> None:

        self.camera.position = (
            position.copy()
        )

    def set_camera_rotation(
        self,
        rotation: Rotation,
    ) -> None:

        self.camera.rotation = (
            rotation.copy()
        )

    def look_at(
        self,
        target: Vector3,
    ) -> None:

        direction = (
            target
            - self.camera.position
        )

        distance_xz = math.sqrt(
            direction.x * direction.x
            + direction.z * direction.z
        )

        if distance_xz > 1e-8:

            self.camera.rotation.y = math.atan2(
                direction.x,
                direction.z,
            )

        if direction.length() > 1e-8:

            self.camera.rotation.x = -math.atan2(
                direction.y,
                distance_xz,
            )

    # ========================================================================
    # WORLD / CAMERA CONVERSION
    # ========================================================================

    def world_to_camera(
        self,
        point: Vector3,
    ) -> Vector3:

        relative = (
            point
            - self.camera.position
        )

        # Inverse camera yaw.
        yaw = -self.camera.rotation.y

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        x = (
            relative.x * cos_y
            - relative.z * sin_y
        )

        z = (
            relative.x * sin_y
            + relative.z * cos_y
        )

        # Inverse camera pitch.
        pitch = -self.camera.rotation.x

        cos_x = math.cos(pitch)
        sin_x = math.sin(pitch)

        y = (
            relative.y * cos_x
            - z * sin_x
        )

        z2 = (
            relative.y * sin_x
            + z * cos_x
        )

        return Vector3(
            x,
            y,
            z2,
        )

    def world_to_screen(
        self,
        point: Vector3,
    ) -> Optional[tuple[float, float]]:

        camera_point = (
            self.world_to_camera(point)
        )

        if (
            camera_point.z
            <= self.camera.near_clip
        ):

            return None

        if (
            camera_point.z
            > self.camera.far_clip
        ):

            return None

        if (
            self.camera.projection
            == ProjectionType.ORTHOGRAPHIC
        ):

            half_height = (
                self.camera.orthographic_size
                / 2.0
            )

            half_width = (
                half_height
                * self.camera.aspect_ratio
            )

            if half_width <= 0:
                return None

            ndc_x = (
                camera_point.x
                / half_width
            )

            ndc_y = (
                camera_point.y
                / half_height
            )

        else:

            fov_radians = math.radians(
                self.camera.field_of_view
            )

            focal = 1.0 / math.tan(
                fov_radians / 2.0
            )

            ndc_x = (
                camera_point.x
                * focal
                / (
                    camera_point.z
                    * self.camera.aspect_ratio
                )
            )

            ndc_y = (
                camera_point.y
                * focal
                / camera_point.z
            )

        screen_x = (
            (ndc_x + 1.0)
            * 0.5
            * self.width
        )

        screen_y = (
            (1.0 - ndc_y)
            * 0.5
            * self.height
        )

        return (
            screen_x,
            screen_y,
        )

    def screen_to_ray(
        self,
        screen_x: float,
        screen_y: float,
    ) -> Ray:

        ndc_x = (
            2.0
            * screen_x
            / self.width
            - 1.0
        )

        ndc_y = (
            1.0
            - 2.0
            * screen_y
            / self.height
        )

        if (
            self.camera.projection
            == ProjectionType.ORTHOGRAPHIC
        ):

            half_height = (
                self.camera.orthographic_size
                / 2.0
            )

            half_width = (
                half_height
                * self.camera.aspect_ratio
            )

            origin_camera = Vector3(
                ndc_x * half_width,
                ndc_y * half_height,
                0.0,
            )

            direction_camera = Vector3(
                0.0,
                0.0,
                1.0,
            )

        else:

            fov = math.radians(
                self.camera.field_of_view
            )

            tan_half_fov = math.tan(
                fov / 2.0
            )

            direction_camera = Vector3(
                ndc_x
                * self.camera.aspect_ratio
                * tan_half_fov,

                ndc_y
                * tan_half_fov,

                1.0,
            ).normalized()

            origin_camera = Vector3()

        world_origin = (
            self.camera_to_world(
                origin_camera
            )
        )

        world_direction = (
            self.camera_direction_to_world(
                direction_camera
            )
        )

        return Ray(
            origin=world_origin,
            direction=world_direction.normalized(),
        )

    # ========================================================================
    # CAMERA ROTATION HELPERS
    # ========================================================================

    def camera_to_world(
        self,
        point: Vector3,
    ) -> Vector3:

        result = point.copy()

        # Pitch.
        pitch = self.camera.rotation.x

        cos_x = math.cos(pitch)
        sin_x = math.sin(pitch)

        y = (
            result.y * cos_x
            - result.z * sin_x
        )

        z = (
            result.y * sin_x
            + result.z * cos_x
        )

        result.y = y
        result.z = z

        # Yaw.
        yaw = self.camera.rotation.y

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        x = (
            result.x * cos_y
            + result.z * sin_y
        )

        z = (
            -result.x * sin_y
            + result.z * cos_y
        )

        result.x = x
        result.z = z

        return (
            result
            + self.camera.position
        )

    def camera_direction_to_world(
        self,
        direction: Vector3,
    ) -> Vector3:

        result = direction.copy()

        pitch = self.camera.rotation.x

        cos_x = math.cos(pitch)
        sin_x = math.sin(pitch)

        y = (
            result.y * cos_x
            - result.z * sin_x
        )

        z = (
            result.y * sin_x
            + result.z * cos_x
        )

        result.y = y
        result.z = z

        yaw = self.camera.rotation.y

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        x = (
            result.x * cos_y
            + result.z * sin_y
        )

        z = (
            -result.x * sin_y
            + result.z * cos_y
        )

        result.x = x
        result.z = z

        return result.normalized()

    # ========================================================================
    # RAYCAST
    # ========================================================================

    def raycast(
        self,
        ray: Ray,
        *,
        interactive_only: bool = False,
        max_distance: Optional[float] = None,
    ) -> Optional[RaycastHit]:

        ray = ray.normalized()

        nearest: Optional[
            RaycastHit
        ] = None

        with self._lock:

            objects = list(
                self._objects.values()
            )

        for obj in objects:

            if not obj.is_visible():
                continue

            if (
                interactive_only
                and not obj.interactive
            ):
                continue

            distance = (
                self._object_ray_distance(
                    obj,
                    ray,
                )
            )

            if distance is None:
                continue

            if (
                max_distance is not None
                and distance > max_distance
            ):
                continue

            point = ray.point_at(
                distance
            )

            normal = (
                point
                - self.world_position(obj)
            ).normalized()

            hit = RaycastHit(
                object_id=obj.id,
                object_name=obj.name,
                distance=distance,
                point=point,
                normal=normal,
                metadata=dict(
                    obj.metadata
                ),
            )

            if (
                nearest is None
                or hit.distance
                < nearest.distance
            ):

                nearest = hit

        return nearest

    def _object_ray_distance(
        self,
        obj: Object3D,
        ray: Ray,
    ) -> Optional[float]:

        center = self.world_position(
            obj
        )

        scale = obj.transform.scale

        half_x = (
            abs(
                obj.bounds.maximum.x
                - obj.bounds.minimum.x
            )
            * abs(scale.x)
            / 2.0
        )

        half_y = (
            abs(
                obj.bounds.maximum.y
                - obj.bounds.minimum.y
            )
            * abs(scale.y)
            / 2.0
        )

        half_z = (
            abs(
                obj.bounds.maximum.z
                - obj.bounds.minimum.z
            )
            * abs(scale.z)
            / 2.0
        )

        box = BoundingBox(
            minimum=Vector3(
                center.x - half_x,
                center.y - half_y,
                center.z - half_z,
            ),
            maximum=Vector3(
                center.x + half_x,
                center.y + half_y,
                center.z + half_z,
            ),
        )

        return box.intersects_ray(
            ray.origin,
            ray.direction,
        )

    # ========================================================================
    # WORLD POSITION
    # ========================================================================

    def world_position(
        self,
        obj: Object3D,
    ) -> Vector3:

        if obj.parent_id is None:

            return obj.transform.position.copy()

        parent = self.get_object(
            obj.parent_id
        )

        if parent is None:

            return obj.transform.position.copy()

        parent_position = (
            self.world_position(parent)
        )

        return (
            parent_position
            + obj.transform.position
        )

    # ========================================================================
    # DEPTH SORTING
    # ========================================================================

    def depth_sort(
        self,
    ) -> list[Object3D]:

        with self._lock:

            objects = [
                obj
                for obj
                in self._objects.values()
                if obj.is_visible()
            ]

        objects.sort(
            key=lambda obj: (
                self.world_position(obj)
                - self.camera.position
            ).length(),
            reverse=True,
        )

        return objects

    # ========================================================================
    # VISIBILITY
    # ========================================================================

    def visible_objects(
        self,
    ) -> list[Object3D]:

        with self._lock:

            return [
                obj
                for obj
                in self._objects.values()
                if obj.is_visible()
            ]

    # ========================================================================
    # HOLOGRAPHIC PANEL
    # ========================================================================

    def create_holographic_panel(
        self,
        name: str,
        *,
        position: Optional[
            Vector3
        ] = None,
        width: float = 4.0,
        height: float = 2.5,
        color: tuple[
            float,
            float,
            float,
            float
        ] = (
            0.0,
            1.0,
            0.7,
            0.85,
        ),
        glow: float = 2.0,
    ) -> Object3D:

        transform = Transform(
            position=(
                position.copy()
                if position
                else Vector3()
            ),
            rotation=Rotation(),
            scale=Scale(
                width,
                height,
                1.0,
            ),
        )

        material = Material(
            name="renix_hologram",
            color=color,
            emission=1.5,
            opacity=color[3],
            wireframe=False,
            additive=True,
            glow=glow,
        )

        bounds = BoundingBox(
            minimum=Vector3(
                -0.5,
                -0.5,
                -0.05,
            ),
            maximum=Vector3(
                0.5,
                0.5,
                0.05,
            ),
        )

        return self.create_object(
            name,
            object_type=ObjectType.PANEL,
            transform=transform,
            material=material,
            bounds=bounds,
            interactive=True,
            metadata={
                "holographic": True,
                "width": width,
                "height": height,
            },
        )

    # ========================================================================
    # INTERACTION
    # ========================================================================

    def object_at_screen_position(
        self,
        screen_x: float,
        screen_y: float,
        *,
        interactive_only: bool = True,
    ) -> Optional[RaycastHit]:

        ray = self.screen_to_ray(
            screen_x,
            screen_y,
        )

        return self.raycast(
            ray,
            interactive_only=interactive_only,
        )

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            objects = list(
                self._objects.values()
            )

        return {
            "time": self._time,

            "viewport": {
                "width": self.width,
                "height": self.height,
            },

            "camera": {
                "position": {
                    "x": self.camera.position.x,
                    "y": self.camera.position.y,
                    "z": self.camera.position.z,
                },
                "rotation": {
                    "x": self.camera.rotation.x,
                    "y": self.camera.rotation.y,
                    "z": self.camera.rotation.z,
                },
                "projection": (
                    self.camera.projection.value
                ),
                "fov": self.camera.field_of_view,
            },

            "objects": [
                {
                    "id": obj.id,
                    "name": obj.name,
                    "type": obj.object_type.value,
                    "visible": obj.is_visible(),
                    "interactive": obj.interactive,
                    "position": {
                        "x": self.world_position(obj).x,
                        "y": self.world_position(obj).y,
                        "z": self.world_position(obj).z,
                    },
                    "rotation": {
                        "x": obj.transform.rotation.x,
                        "y": obj.transform.rotation.y,
                        "z": obj.transform.rotation.z,
                    },
                    "scale": {
                        "x": obj.transform.scale.x,
                        "y": obj.transform.scale.y,
                        "z": obj.transform.scale.z,
                    },
                    "material": {
                        "color": obj.material.color,
                        "opacity": obj.material.opacity,
                        "glow": obj.material.glow,
                        "emission": obj.material.emission,
                        "wireframe": obj.material.wireframe,
                        "additive": obj.material.additive,
                    },
                    "metadata": dict(
                        obj.metadata
                    ),
                }
                for obj in objects
            ],
        }

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(self) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "objects": len(
                    self._objects
                ),
                "visible_objects": len(
                    self.visible_objects()
                ),
                "width": self.width,
                "height": self.height,
                "projection": (
                    self.camera.projection.value
                ),
            }

    # ========================================================================
    # CLEAR
    # ========================================================================

    def clear(self) -> None:

        with self._lock:

            self._objects.clear()


# ============================================================================
# FACTORY
# ============================================================================


def create_default_3d_engine(
    width: int = 1920,
    height: int = 1080,
) -> Engine3D:

    engine = Engine3D(
        width=width,
        height=height,
    )

    engine.start()

    return engine


ThreeDEngine = Engine3D


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "ObjectType",
    "ProjectionType",
    "VisibilityState",
    "Vector3",
    "Rotation",
    "Scale",
    "Transform",
    "Material",
    "BoundingBox",
    "Object3D",
    "Camera",
    "Ray",
    "RaycastHit",
    "Engine3D",
    "ThreeDEngine",
    "create_default_3d_engine",
]



