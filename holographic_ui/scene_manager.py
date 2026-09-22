"""
RENIX Holographic UI - Scene Manager
====================================

Manages holographic scenes and the objects contained inside them.

Responsibilities:
    - Create / remove scenes
    - Activate / deactivate scenes
    - Manage scene objects
    - Scene hierarchy
    - Visibility
    - Spatial transforms
    - Scene transitions
    - Object lookup
    - Layer ordering
    - Scene events

Rendering is intentionally handled by the rendering engines.
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
    "RENIX.holographic_ui.scene_manager"
)


# ============================================================================
# ENUMS
# ============================================================================


class SceneState(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    LOADING = "loading"
    UNLOADING = "unloading"


class ObjectType(str, Enum):
    GENERIC = "generic"
    PANEL = "panel"
    WIDGET = "widget"
    BUTTON = "button"
    TEXT = "text"
    IMAGE = "image"
    MODEL_3D = "model_3d"
    PARTICLE = "particle"
    CURSOR = "cursor"
    HUD = "hud"
    NOTIFICATION = "notification"


# ============================================================================
# TRANSFORM
# ============================================================================


@dataclass
class Transform:
    """3D transform used by holographic scene objects."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0

    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0

    def copy(self) -> "Transform":
        return Transform(
            x=self.x,
            y=self.y,
            z=self.z,
            rotation_x=self.rotation_x,
            rotation_y=self.rotation_y,
            rotation_z=self.rotation_z,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            scale_z=self.scale_z,
        )

    def translate(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:

        self.x += x
        self.y += y
        self.z += z

    def rotate(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:

        self.rotation_x += x
        self.rotation_y += y
        self.rotation_z += z

    def scale(
        self,
        x: float = 1.0,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> None:

        if y is None:
            y = x

        if z is None:
            z = x

        self.scale_x *= x
        self.scale_y *= y
        self.scale_z *= z

    def to_dict(self) -> dict[str, float]:

        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "rotation_x": self.rotation_x,
            "rotation_y": self.rotation_y,
            "rotation_z": self.rotation_z,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "scale_z": self.scale_z,
        }


# ============================================================================
# SCENE OBJECT
# ============================================================================


@dataclass
class SceneObject:
    """An individual object inside a holographic scene."""

    id: str

    name: str

    object_type: ObjectType = ObjectType.GENERIC

    transform: Transform = field(
        default_factory=Transform
    )

    visible: bool = True

    enabled: bool = True

    interactive: bool = True

    opacity: float = 1.0

    layer: int = 0

    parent_id: Optional[str] = None

    children: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:

        return {
            "id": self.id,
            "name": self.name,
            "object_type": self.object_type.value,
            "transform": self.transform.to_dict(),
            "visible": self.visible,
            "enabled": self.enabled,
            "interactive": self.interactive,
            "opacity": self.opacity,
            "layer": self.layer,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ============================================================================
# SCENE
# ============================================================================


@dataclass
class HolographicScene:
    """Container for all objects belonging to a scene."""

    id: str

    name: str

    state: SceneState = SceneState.INACTIVE

    visible: bool = True

    objects: dict[str, SceneObject] = field(
        default_factory=dict
    )

    root_objects: list[str] = field(
        default_factory=list
    )

    background: Optional[str] = None

    environment: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:

        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "visible": self.visible,
            "background": self.background,
            "environment": dict(self.environment),
            "metadata": dict(self.metadata),
            "objects": [
                obj.to_dict()
                for obj in self.objects.values()
            ],
            "root_objects": list(
                self.root_objects
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ============================================================================
# SCENE MANAGER
# ============================================================================


class SceneManager:
    """
    Central manager for RENIX holographic scenes.
    """

    def __init__(self) -> None:

        self._scenes: dict[
            str,
            HolographicScene,
        ] = {}

        self._active_scene_id: Optional[
            str
        ] = None

        self._lock = threading.RLock()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._running = False

        logger.debug(
            "Scene manager created"
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

        with self._lock:

            active = self.get_active_scene()

            if active is None:
                return

            if active.state != SceneState.ACTIVE:
                return

            active.touch()

    # ========================================================================
    # SCENE CREATION
    # ========================================================================

    def create_scene(
        self,
        name: str,
        *,
        background: Optional[str] = None,
        environment: Optional[
            dict[str, Any]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        activate: bool = False,
    ) -> HolographicScene:

        if not name:

            raise ValueError(
                "Scene name cannot be empty"
            )

        scene = HolographicScene(
            id=uuid.uuid4().hex,
            name=name,
            background=background,
            environment=dict(
                environment or {}
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        with self._lock:

            self._scenes[scene.id] = scene

        self._emit(
            "scene_created",
            scene,
        )

        if activate:

            self.activate_scene(
                scene.id
            )

        return scene

    # ========================================================================
    # SCENE LOOKUP
    # ========================================================================

    def get_scene(
        self,
        scene_id: str,
    ) -> Optional[HolographicScene]:

        with self._lock:

            return self._scenes.get(
                scene_id
            )

    def get_scene_by_name(
        self,
        name: str,
    ) -> Optional[HolographicScene]:

        with self._lock:

            for scene in self._scenes.values():

                if scene.name == name:

                    return scene

        return None

    def get_scenes(
        self,
    ) -> list[HolographicScene]:

        with self._lock:

            return list(
                self._scenes.values()
            )

    def get_active_scene(
        self,
    ) -> Optional[HolographicScene]:

        with self._lock:

            if self._active_scene_id is None:

                return None

            return self._scenes.get(
                self._active_scene_id
            )

    # ========================================================================
    # SCENE ACTIVATION
    # ========================================================================

    def activate_scene(
        self,
        scene_id: str,
    ) -> bool:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return False

            previous_id = (
                self._active_scene_id
            )

            if previous_id == scene_id:

                scene.state = (
                    SceneState.ACTIVE
                )

                scene.visible = True

                return True

            if previous_id is not None:

                previous = self._scenes.get(
                    previous_id
                )

                if previous is not None:

                    previous.state = (
                        SceneState.INACTIVE
                    )

            scene.state = SceneState.ACTIVE

            scene.visible = True

            self._active_scene_id = scene_id

            scene.touch()

        self._emit(
            "scene_activated",
            scene,
            previous_id,
        )

        return True

    def deactivate_scene(
        self,
        scene_id: str,
    ) -> bool:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return False

            scene.state = (
                SceneState.INACTIVE
            )

            scene.visible = False

            if self._active_scene_id == scene_id:

                self._active_scene_id = None

            scene.touch()

        self._emit(
            "scene_deactivated",
            scene,
        )

        return True

    def pause_scene(
        self,
        scene_id: str,
    ) -> bool:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return False

            if scene.state != SceneState.ACTIVE:

                return False

            scene.state = SceneState.PAUSED

            scene.touch()

        self._emit(
            "scene_paused",
            scene,
        )

        return True

    def resume_scene(
        self,
        scene_id: str,
    ) -> bool:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return False

            if scene.state != SceneState.PAUSED:

                return False

            scene.state = SceneState.ACTIVE

            scene.visible = True

            scene.touch()

        self._emit(
            "scene_resumed",
            scene,
        )

        return True

    # ========================================================================
    # SCENE REMOVAL
    # ========================================================================

    def remove_scene(
        self,
        scene_id: str,
    ) -> bool:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return False

            if self._active_scene_id == scene_id:

                self._active_scene_id = None

            self._scenes.pop(
                scene_id
            )

        self._emit(
            "scene_removed",
            scene,
        )

        return True

    # ========================================================================
    # OBJECT CREATION
    # ========================================================================

    def create_object(
        self,
        scene_id: str,
        name: str,
        *,
        object_type: ObjectType = (
            ObjectType.GENERIC
        ),
        transform: Optional[
            Transform
        ] = None,
        parent_id: Optional[str] = None,
        visible: bool = True,
        enabled: bool = True,
        interactive: bool = True,
        opacity: float = 1.0,
        layer: int = 0,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> Optional[SceneObject]:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return None

            if not name:

                raise ValueError(
                    "Object name cannot be empty"
                )

            if parent_id is not None:

                if parent_id not in scene.objects:

                    raise ValueError(
                        "Parent object does not exist"
                    )

            object_id = uuid.uuid4().hex

            obj = SceneObject(
                id=object_id,
                name=name,
                object_type=object_type,
                transform=(
                    transform.copy()
                    if transform
                    else Transform()
                ),
                parent_id=parent_id,
                visible=visible,
                enabled=enabled,
                interactive=interactive,
                opacity=max(
                    0.0,
                    min(
                        1.0,
                        float(opacity),
                    ),
                ),
                layer=int(layer),
                metadata=dict(
                    metadata or {}
                ),
            )

            scene.objects[
                object_id
            ] = obj

            if parent_id is None:

                scene.root_objects.append(
                    object_id
                )

            else:

                parent = scene.objects[
                    parent_id
                ]

                parent.children.append(
                    object_id
                )

                parent.touch()

            scene.touch()

        self._emit(
            "object_created",
            scene,
            obj,
        )

        return obj

    # ========================================================================
    # OBJECT LOOKUP
    # ========================================================================

    def get_object(
        self,
        scene_id: str,
        object_id: str,
    ) -> Optional[SceneObject]:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return None

            return scene.objects.get(
                object_id
            )

    def find_object(
        self,
        scene_id: str,
        name: str,
    ) -> Optional[SceneObject]:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return None

            for obj in scene.objects.values():

                if obj.name == name:

                    return obj

        return None

    # ========================================================================
    # OBJECT REMOVAL
    # ========================================================================

    def remove_object(
        self,
        scene_id: str,
        object_id: str,
        *,
        recursive: bool = True,
    ) -> bool:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return False

            obj = scene.objects.get(
                object_id
            )

            if obj is None:

                return False

            if recursive:

                children = list(
                    obj.children
                )

                for child_id in children:

                    self.remove_object(
                        scene_id,
                        child_id,
                        recursive=True,
                    )

            else:

                for child_id in list(
                    obj.children
                ):

                    child = scene.objects.get(
                        child_id
                    )

                    if child is not None:

                        child.parent_id = (
                            obj.parent_id
                        )

                        if obj.parent_id:

                            parent = scene.objects.get(
                                obj.parent_id
                            )

                            if (
                                parent
                                and child_id
                                not in parent.children
                            ):

                                parent.children.append(
                                    child_id
                                )

                        elif (
                            child_id
                            not in scene.root_objects
                        ):

                            scene.root_objects.append(
                                child_id
                            )

            if obj.parent_id:

                parent = scene.objects.get(
                    obj.parent_id
                )

                if parent is not None:

                    if object_id in parent.children:

                        parent.children.remove(
                            object_id
                        )

                    parent.touch()

            elif object_id in scene.root_objects:

                scene.root_objects.remove(
                    object_id
                )

            scene.objects.pop(
                object_id,
                None,
            )

            scene.touch()

        self._emit(
            "object_removed",
            scene,
            obj,
        )

        return True

    # ========================================================================
    # OBJECT TRANSFORMS
    # ========================================================================

    def set_transform(
        self,
        scene_id: str,
        object_id: str,
        transform: Transform,
    ) -> bool:

        obj = self.get_object(
            scene_id,
            object_id,
        )

        if obj is None:

            return False

        old_transform = obj.transform.copy()

        obj.transform = transform.copy()

        obj.touch()

        self._emit(
            "object_transform_changed",
            obj,
            old_transform,
        )

        return True

    def translate_object(
        self,
        scene_id: str,
        object_id: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> bool:

        obj = self.get_object(
            scene_id,
            object_id,
        )

        if obj is None:

            return False

        old_transform = obj.transform.copy()

        obj.transform.translate(
            x,
            y,
            z,
        )

        obj.touch()

        self._emit(
            "object_transform_changed",
            obj,
            old_transform,
        )

        return True

    def rotate_object(
        self,
        scene_id: str,
        object_id: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> bool:

        obj = self.get_object(
            scene_id,
            object_id,
        )

        if obj is None:

            return False

        old_transform = obj.transform.copy()

        obj.transform.rotate(
            x,
            y,
            z,
        )

        obj.touch()

        self._emit(
            "object_transform_changed",
            obj,
            old_transform,
        )

        return True

    def scale_object(
        self,
        scene_id: str,
        object_id: str,
        x: float = 1.0,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> bool:

        obj = self.get_object(
            scene_id,
            object_id,
        )

        if obj is None:

            return False

        old_transform = obj.transform.copy()

        obj.transform.scale(
            x,
            y,
            z,
        )

        obj.touch()

        self._emit(
            "object_transform_changed",
            obj,
            old_transform,
        )

        return True

    # ========================================================================
    # VISIBILITY / ENABLE
    # ========================================================================

    def set_object_visibility(
        self,
        scene_id: str,
        object_id: str,
        visible: bool,
    ) -> bool:

        obj = self.get_object(
            scene_id,
            object_id,
        )

        if obj is None:

            return False

        obj.visible = bool(visible)

        obj.touch()

        self._emit(
            "object_visibility_changed",
            obj,
        )

        return True

    def set_object_enabled(
        self,
        scene_id: str,
        object_id: str,
        enabled: bool,
    ) -> bool:

        obj = self.get_object(
            scene_id,
            object_id,
        )

        if obj is None:

            return False

        obj.enabled = bool(enabled)

        obj.touch()

        self._emit(
            "object_enabled_changed",
            obj,
        )

        return True

    # ========================================================================
    # CHILDREN
    # ========================================================================

    def get_children(
        self,
        scene_id: str,
        object_id: str,
    ) -> list[SceneObject]:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return []

            obj = scene.objects.get(
                object_id
            )

            if obj is None:

                return []

            return [
                scene.objects[child_id]
                for child_id in obj.children
                if child_id in scene.objects
            ]

    def get_root_objects(
        self,
        scene_id: str,
    ) -> list[SceneObject]:

        with self._lock:

            scene = self._scenes.get(
                scene_id
            )

            if scene is None:

                return []

            return [
                scene.objects[obj_id]
                for obj_id in scene.root_objects
                if obj_id in scene.objects
            ]

    # ========================================================================
    # SCENE TRANSITION
    # ========================================================================

    def transition_to(
        self,
        scene_id: str,
        *,
        duration: float = 0.3,
    ) -> bool:

        if self.get_scene(scene_id) is None:

            return False

        self._emit(
            "scene_transition_started",
            scene_id,
            duration,
        )

        success = self.activate_scene(
            scene_id
        )

        self._emit(
            "scene_transition_completed",
            scene_id,
            duration,
            success,
        )

        return success

    # ========================================================================
    # EVENTS
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not event:

            raise ValueError(
                "Event name cannot be empty"
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
                    "Scene event callback failed: %s",
                    event,
                )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "active_scene_id": (
                    self._active_scene_id
                ),
                "scenes": [
                    scene.to_dict()
                    for scene in self._scenes.values()
                ],
            }


__all__ = [
    "SceneState",
    "ObjectType",
    "Transform",
    "SceneObject",
    "HolographicScene",
    "SceneManager",
]


