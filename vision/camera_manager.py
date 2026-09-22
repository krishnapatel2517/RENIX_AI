"""
RENIX Vision — Camera Manager

Central manager for multiple camera devices.

Responsibilities:
- Discover available cameras
- Register/unregister cameras
- Open/close cameras
- Manage the active/default camera
- Capture frames
- Start/stop camera streams
- Switch between cameras
- Maintain camera status
- Thread-safe camera access
- Provide callbacks for camera events
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .camera import (
    Camera,
    CameraConfig,
    CameraStatus,
    Frame,
)

logger = logging.getLogger(
    "RENIX.vision.camera_manager"
)


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class CameraDevice:
    """Metadata about a registered camera."""

    camera_id: str
    device_index: int
    name: str = ""
    description: str = ""

    camera: Optional[Camera] = None

    available: bool = False
    active: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "device_index": self.device_index,
            "name": self.name,
            "description": self.description,
            "available": self.available,
            "active": self.active,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# CAMERA MANAGER
# ============================================================================


class CameraManager:
    """
    Central camera controller for RENIX.

    Typical usage:

        manager = CameraManager()

        manager.discover()

        manager.open_camera("camera_0")

        frame = manager.read()

        manager.close_all()
    """

    def __init__(
        self,
        *,
        max_devices: int = 10,
        auto_discover: bool = False,
        default_config: Optional[
            CameraConfig
        ] = None,
    ) -> None:

        self.max_devices = max(
            1,
            int(max_devices),
        )

        self.default_config = (
            default_config
            or CameraConfig()
        )

        self._lock = threading.RLock()

        self._devices: dict[
            str,
            CameraDevice,
        ] = {}

        self._active_camera_id: Optional[
            str
        ] = None

        self._callbacks: dict[
            str,
            list[Callable[..., Any]],
        ] = {
            "connected": [],
            "disconnected": [],
            "frame": [],
            "error": [],
            "switched": [],
        }

        self._initialized = False

        if auto_discover:
            self.discover()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def initialize(self) -> bool:
        """
        Initialize the camera manager.

        Discovery is performed if no devices are currently registered.
        """

        with self._lock:

            if self._initialized:
                return True

            self.discover()

            self._initialized = True

            return True

    # ========================================================================
    # DISCOVERY
    # ========================================================================

    def discover(
        self,
        *,
        max_devices: Optional[int] = None,
        register: bool = True,
    ) -> list[CameraDevice]:
        """
        Discover physically available camera devices.

        Returns a list of CameraDevice objects.
        """

        limit = (
            max_devices
            if max_devices is not None
            else self.max_devices
        )

        limit = max(
            1,
            int(limit),
        )

        devices: list[CameraDevice] = []

        available_indices = (
            Camera.available_devices(
                max_devices=limit
            )
        )

        with self._lock:

            for index in available_indices:

                camera_id = (
                    self._camera_id(index)
                )

                existing = self._devices.get(
                    camera_id
                )

                if existing is not None:

                    existing.available = True

                    devices.append(
                        existing
                    )

                    continue

                device = CameraDevice(
                    camera_id=camera_id,
                    device_index=index,
                    name=f"Camera {index}",
                    description=(
                        f"RENIX camera device "
                        f"{index}"
                    ),
                    available=True,
                )

                devices.append(device)

                if register:

                    self._devices[
                        camera_id
                    ] = device

        return devices

    @staticmethod
    def _camera_id(
        device_index: int,
    ) -> str:
        return f"camera_{device_index}"

    # ========================================================================
    # DEVICE REGISTRATION
    # ========================================================================

    def register_camera(
        self,
        device_index: int,
        *,
        camera_id: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        config: Optional[
            CameraConfig
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> CameraDevice:
        """
        Register a camera manually.
        """

        device_index = int(device_index)

        camera_id = (
            camera_id
            or self._camera_id(
                device_index
            )
        )

        with self._lock:

            if camera_id in self._devices:

                raise ValueError(
                    f"Camera '{camera_id}' "
                    "is already registered."
                )

            camera_config = (
                config
                or CameraConfig(
                    device_index=device_index
                )
            )

            camera_config.device_index = (
                device_index
            )

            camera = Camera(
                config=camera_config
            )

            device = CameraDevice(
                camera_id=camera_id,
                device_index=device_index,
                name=(
                    name
                    or f"Camera {device_index}"
                ),
                description=description,
                camera=camera,
                available=Camera.is_available(
                    device_index
                ),
                metadata=dict(
                    metadata or {}
                ),
            )

            self._devices[
                camera_id
            ] = device

            return device

    def unregister_camera(
        self,
        camera_id: str,
        *,
        close: bool = True,
    ) -> bool:
        """
        Remove a registered camera.
        """

        with self._lock:

            device = self._devices.get(
                camera_id
            )

            if device is None:
                return False

            if close and device.camera:
                device.camera.close()

            if (
                self._active_camera_id
                == camera_id
            ):

                self._active_camera_id = None

            del self._devices[
                camera_id
            ]

            return True

    # ========================================================================
    # DEVICE LOOKUP
    # ========================================================================

    def get_camera(
        self,
        camera_id: str,
    ) -> Optional[Camera]:

        with self._lock:

            device = self._devices.get(
                camera_id
            )

            if device is None:
                return None

            if device.camera is None:

                device.camera = Camera(
                    CameraConfig(
                        device_index=(
                            device.device_index
                        )
                    )
                )

            return device.camera

    def get_device(
        self,
        camera_id: str,
    ) -> Optional[CameraDevice]:

        with self._lock:
            return self._devices.get(
                camera_id
            )

    def list_devices(
        self,
    ) -> list[CameraDevice]:

        with self._lock:

            return list(
                self._devices.values()
            )

    def camera_ids(
        self,
    ) -> list[str]:

        with self._lock:

            return list(
                self._devices.keys()
            )

    # ========================================================================
    # ACTIVE CAMERA
    # ========================================================================

    def get_active_camera_id(
        self,
    ) -> Optional[str]:

        with self._lock:

            return self._active_camera_id

    def get_active_camera(
        self,
    ) -> Optional[Camera]:

        with self._lock:

            if (
                self._active_camera_id
                is None
            ):
                return None

            device = self._devices.get(
                self._active_camera_id
            )

            if device is None:
                return None

            return device.camera

    def set_active_camera(
        self,
        camera_id: str,
        *,
        open_camera: bool = True,
    ) -> bool:
        """
        Make a camera the active RENIX camera.
        """

        with self._lock:

            device = self._devices.get(
                camera_id
            )

            if device is None:

                raise ValueError(
                    f"Unknown camera: {camera_id}"
                )

            previous_id = (
                self._active_camera_id
            )

            if open_camera:

                if device.camera is None:

                    device.camera = Camera(
                        CameraConfig(
                            device_index=(
                                device.device_index
                            )
                        )
                    )

                if not device.camera.is_open():

                    if not device.camera.open():

                        self._emit(
                            "error",
                            camera_id,
                            "Unable to open "
                            "camera.",
                        )

                        return False

            for registered in (
                self._devices.values()
            ):

                registered.active = False

            device.active = True

            self._active_camera_id = (
                camera_id
            )

        if previous_id != camera_id:

            self._emit(
                "switched",
                camera_id,
                {
                    "previous": previous_id,
                    "current": camera_id,
                },
            )

        return True

    # ========================================================================
    # OPEN
    # ========================================================================

    def open_camera(
        self,
        camera_id: str,
    ) -> bool:
        """
        Open a registered camera.
        """

        with self._lock:

            device = self._devices.get(
                camera_id
            )

            if device is None:

                raise ValueError(
                    f"Unknown camera: {camera_id}"
                )

            if device.camera is None:

                device.camera = Camera(
                    CameraConfig(
                        device_index=(
                            device.device_index
                        )
                    )
                )

            success = device.camera.open()

            if success:

                device.available = True

                self._emit(
                    "connected",
                    camera_id,
                )

            else:

                self._emit(
                    "error",
                    camera_id,
                    device.camera.get_last_error(),
                )

            return success

    # ========================================================================
    # CLOSE
    # ========================================================================

    def close_camera(
        self,
        camera_id: str,
    ) -> bool:

        with self._lock:

            device = self._devices.get(
                camera_id
            )

            if device is None:
                return False

            if device.camera is None:
                return True

            device.camera.close()

            device.active = False

            if (
                self._active_camera_id
                == camera_id
            ):

                self._active_camera_id = None

            self._emit(
                "disconnected",
                camera_id,
            )

            return True

    def close_all(
        self,
    ) -> None:

        with self._lock:

            camera_ids = list(
                self._devices.keys()
            )

        for camera_id in camera_ids:

            try:
                self.close_camera(
                    camera_id
                )
            except Exception:

                logger.exception(
                    "Failed to close camera %s.",
                    camera_id,
                )

    # ========================================================================
    # READ
    # ========================================================================

    def read(
        self,
        camera_id: Optional[str] = None,
    ) -> Optional[Frame]:
        """
        Capture one frame from a camera.

        If camera_id is omitted, the active camera is used.
        """

        camera_id = (
            camera_id
            or self._active_camera_id
        )

        if camera_id is None:

            if not self._devices:

                self.discover()

            if not self._devices:

                return None

            camera_id = next(
                iter(
                    self._devices
                )
            )

            if not self.set_active_camera(
                camera_id
            ):
                return None

        camera = self.get_camera(
            camera_id
        )

        if camera is None:
            return None

        if not camera.is_open():

            if not camera.open():

                self._emit(
                    "error",
                    camera_id,
                    camera.get_last_error(),
                )

                return None

        frame = camera.read()

        if frame is None:

            self._emit(
                "error",
                camera_id,
                camera.get_last_error(),
            )

            return None

        self._emit(
            "frame",
            camera_id,
            frame,
        )

        return frame

    # ========================================================================
    # STREAMING
    # ========================================================================

    def start_stream(
        self,
        camera_id: Optional[str] = None,
        callback: Optional[
            Callable[[Frame], Any]
        ] = None,
    ) -> bool:
        """
        Start background streaming.
        """

        camera_id = (
            camera_id
            or self._active_camera_id
        )

        if camera_id is None:

            if not self._devices:
                self.discover()

            if not self._devices:
                return False

            camera_id = next(
                iter(
                    self._devices
                )
            )

            if not self.set_active_camera(
                camera_id
            ):
                return False

        camera = self.get_camera(
            camera_id
        )

        if camera is None:
            return False

        if callback is not None:

            def wrapped_callback(
                frame: Frame,
            ) -> None:

                try:
                    callback(frame)
                finally:

                    self._emit(
                        "frame",
                        camera_id,
                        frame,
                    )

            return camera.start_stream(
                wrapped_callback
            )

        return camera.start_stream(
            lambda frame: self._emit(
                "frame",
                camera_id,
                frame,
            )
        )

    def stop_stream(
        self,
        camera_id: Optional[str] = None,
    ) -> bool:

        camera_id = (
            camera_id
            or self._active_camera_id
        )

        if camera_id is None:
            return False

        camera = self.get_camera(
            camera_id
        )

        if camera is None:
            return False

        camera.stop_stream()

        return True

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
        camera_id: Optional[str] = None,
    ) -> Optional[CameraStatus]:

        camera_id = (
            camera_id
            or self._active_camera_id
        )

        if camera_id is None:
            return None

        camera = self.get_camera(
            camera_id
        )

        if camera is None:
            return None

        return camera.get_status()

    def get_all_status(
        self,
    ) -> dict[str, dict[str, Any]]:

        result: dict[
            str,
            dict[str, Any],
        ] = {}

        with self._lock:

            devices = list(
                self._devices.items()
            )

        for camera_id, device in devices:

            status = (
                device.camera.get_status()
                if device.camera
                else CameraStatus(
                    device_index=(
                        device.device_index
                    ),
                    connected=False,
                )
            )

            result[camera_id] = {
                "device": device.to_dict(),
                "status": status.to_dict(),
            }

        return result

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> None:
        """
        Register a camera manager event callback.

        Supported events:
        - connected
        - disconnected
        - frame
        - error
        - switched
        """

        if event not in self._callbacks:

            raise ValueError(
                f"Unknown camera event: {event}"
            )

        if not callable(callback):

            raise TypeError(
                "Camera event callback "
                "must be callable."
            )

        with self._lock:

            if callback not in self._callbacks[
                event
            ]:

                self._callbacks[
                    event
                ].append(callback)

    def off(
        self,
        event: str,
        callback: Callable[..., Any],
    ) -> None:

        if event not in self._callbacks:
            return

        with self._lock:

            callbacks = self._callbacks[
                event
            ]

            if callback in callbacks:

                callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: str,
        *args: Any,
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
                    *args
                )

            except Exception:

                logger.exception(
                    "Camera manager callback "
                    "failed for event '%s'.",
                    event,
                )

    # ========================================================================
    # CAMERA SWITCHING
    # ========================================================================

    def switch_camera(
        self,
        direction: int = 1,
    ) -> bool:
        """
        Switch to the next/previous registered camera.

        direction=1  -> next camera
        direction=-1 -> previous camera
        """

        with self._lock:

            ids = list(
                self._devices.keys()
            )

            if not ids:
                return False

            if self._active_camera_id not in ids:

                target = ids[
                    0
                ]

            else:

                current_index = ids.index(
                    self._active_camera_id
                )

                target_index = (
                    current_index
                    + direction
                ) % len(ids)

                target = ids[
                    target_index
                ]

        return self.set_active_camera(
            target
        )

    # ========================================================================
    # REFRESH
    # ========================================================================

    def refresh(
        self,
    ) -> list[CameraDevice]:
        """
        Re-scan camera hardware and update
        availability metadata.
        """

        discovered = self.discover(
            register=True
        )

        discovered_ids = {
            device.camera_id
            for device in discovered
        }

        with self._lock:

            for (
                camera_id,
                device,
            ) in self._devices.items():

                if (
                    camera_id
                    not in discovered_ids
                ):

                    device.available = False

        return self.list_devices()

    # ========================================================================
    # DEFAULT CAMERA
    # ========================================================================

    def ensure_default_camera(
        self,
    ) -> Optional[str]:
        """
        Ensure that RENIX has an active camera.

        Returns the active camera ID.
        """

        with self._lock:

            if self._active_camera_id:

                device = self._devices.get(
                    self._active_camera_id
                )

                if device is not None:

                    if (
                        device.camera
                        and device.camera.is_open()
                    ):
                        return (
                            self._active_camera_id
                        )

        if not self._devices:

            self.discover()

        if not self._devices:
            return None

        first_id = next(
            iter(
                self._devices
            )
        )

        if self.set_active_camera(
            first_id
        ):

            return first_id

        return None

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def shutdown(
        self,
    ) -> None:
        """Shutdown all camera resources."""

        self.close_all()

        with self._lock:

            self._active_camera_id = None

            self._initialized = False

    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================

    def __enter__(
        self,
    ) -> "CameraManager":

        self.initialize()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.shutdown()

    # ========================================================================
    # DESTRUCTOR
    # ========================================================================

    def __del__(self) -> None:

        try:
            self.shutdown()
        except Exception:
            pass


# ============================================================================
# DEFAULT GLOBAL MANAGER
# ============================================================================


_default_manager: Optional[
    CameraManager
] = None

_default_manager_lock = (
    threading.RLock()
)


def get_camera_manager() -> CameraManager:
    """
    Return the shared RENIX CameraManager.
    """

    global _default_manager

    with _default_manager_lock:

        if _default_manager is None:

            _default_manager = (
                CameraManager()
            )

        return _default_manager


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def discover_cameras(
    max_devices: int = 10,
) -> list[CameraDevice]:
    """Discover cameras through the default manager."""

    return get_camera_manager().discover(
        max_devices=max_devices
    )


def open_camera(
    camera_id: str,
) -> bool:
    """Open a camera through the default manager."""

    return get_camera_manager().open_camera(
        camera_id
    )


def close_camera(
    camera_id: str,
) -> bool:
    """Close a camera through the default manager."""

    return get_camera_manager().close_camera(
        camera_id
    )


def read_camera(
    camera_id: Optional[str] = None,
) -> Optional[Frame]:
    """Read a frame through the default manager."""

    return get_camera_manager().read(
        camera_id
    )


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "CameraDevice",
    "CameraManager",
    "get_camera_manager",
    "discover_cameras",
    "open_camera",
    "close_camera",
    "read_camera",
]


