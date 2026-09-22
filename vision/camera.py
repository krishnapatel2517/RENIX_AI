"""
RENIX Vision — Camera Interface

Provides a robust camera abstraction for RENIX.

Responsibilities:
- Open and close camera devices
- Capture frames
- Configure resolution / FPS
- Read single frames
- Continuous frame streaming
- Camera availability checks
- Device enumeration
- Thread-safe access
- Frame metadata
- Graceful handling when OpenCV is unavailable
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger("RENIX.vision.camera")

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class CameraConfig:
    """Configuration for a camera device."""

    device_index: int = 0

    width: int = 1280
    height: int = 720

    fps: int = 30

    backend: Optional[int] = None

    auto_exposure: bool = True

    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None

    buffer_size: int = 1

    flip_horizontal: bool = False
    flip_vertical: bool = False

    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0

    warmup_frames: int = 5

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Frame:
    """
    Captured camera frame.

    ``data`` contains the OpenCV BGR image when OpenCV
    is available.
    """

    data: Any

    timestamp: float

    frame_number: int

    width: int

    height: int

    fps: float

    camera_index: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def valid(self) -> bool:
        return self.data is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "camera_index": self.camera_index,
            "valid": self.valid,
            "metadata": dict(self.metadata),
        }


@dataclass
class CameraStatus:
    """Current camera state."""

    connected: bool = False
    capturing: bool = False

    device_index: int = 0

    width: int = 0
    height: int = 0

    fps: float = 0.0

    frames_captured: int = 0

    last_frame_time: Optional[float] = None

    last_error: Optional[str] = None

    started_at: Optional[float] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "capturing": self.capturing,
            "device_index": self.device_index,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames_captured": self.frames_captured,
            "last_frame_time": self.last_frame_time,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# CAMERA
# ============================================================================


class Camera:
    """
    Main RENIX camera abstraction.

    Example
    -------
    ::

        camera = Camera()

        if camera.open():
            frame = camera.read()

            if frame:
                image = frame.data

            camera.close()
    """

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        device_index: Optional[int] = None,
    ) -> None:

        self.config = config or CameraConfig()

        if device_index is not None:
            self.config.device_index = device_index

        self._capture: Any = None

        self._lock = threading.RLock()

        self._status = CameraStatus(
            device_index=self.config.device_index
        )

        self._frame_number = 0

        self._running = False

        self._stream_thread: Optional[
            threading.Thread
        ] = None

        self._stream_stop = threading.Event()

        self._latest_frame: Optional[
            Frame
        ] = None

        self._callbacks: list = []

    # ========================================================================
    # AVAILABILITY
    # ========================================================================

    @staticmethod
    def library_available() -> bool:
        """Return whether OpenCV is installed."""

        return CV2_AVAILABLE

    @classmethod
    def is_available(
        cls,
        device_index: int = 0,
    ) -> bool:
        """
        Check whether a camera can be opened.

        The camera is opened temporarily and immediately
        released.
        """

        if not CV2_AVAILABLE:
            return False

        capture = None

        try:

            capture = cv2.VideoCapture(
                device_index
            )

            return bool(
                capture.isOpened()
            )

        except Exception:

            logger.exception(
                "Camera availability check failed."
            )

            return False

        finally:

            if capture is not None:

                try:
                    capture.release()
                except Exception:
                    pass

    @classmethod
    def available_devices(
        cls,
        max_devices: int = 10,
    ) -> list[int]:
        """
        Scan for available camera device indices.
        """

        if not CV2_AVAILABLE:
            return []

        devices: list[int] = []

        for index in range(
            max(0, max_devices)
        ):

            capture = None

            try:

                capture = cv2.VideoCapture(
                    index
                )

                if capture.isOpened():
                    devices.append(index)

            except Exception:
                pass

            finally:

                if capture is not None:

                    try:
                        capture.release()
                    except Exception:
                        pass

        return devices

    # ========================================================================
    # OPEN
    # ========================================================================

    def open(
        self,
        *,
        device_index: Optional[int] = None,
    ) -> bool:
        """
        Open the camera device.
        """

        with self._lock:

            if not CV2_AVAILABLE:

                self._set_error(
                    "OpenCV is not installed."
                )

                return False

            if device_index is not None:

                self.config.device_index = (
                    device_index
                )

                self._status.device_index = (
                    device_index
                )

            if self.is_open():

                return True

            try:

                if self.config.backend is not None:

                    self._capture = (
                        cv2.VideoCapture(
                            self.config.device_index,
                            self.config.backend,
                        )
                    )

                else:

                    self._capture = (
                        cv2.VideoCapture(
                            self.config.device_index
                        )
                    )

                if not self._capture.isOpened():

                    self._set_error(
                        "Unable to open camera "
                        f"{self.config.device_index}."
                    )

                    self._release_capture()

                    return False

                self._configure_capture()

                self._frame_number = 0

                self._status.connected = True

                self._status.capturing = False

                self._status.device_index = (
                    self.config.device_index
                )

                self._status.started_at = (
                    time.time()
                )

                self._status.last_error = None

                self._warmup()

                logger.info(
                    "Camera %s opened successfully.",
                    self.config.device_index,
                )

                return True

            except Exception as exc:

                self._set_error(
                    str(exc)
                )

                self._release_capture()

                logger.exception(
                    "Failed to open camera."
                )

                return False

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def _configure_capture(self) -> None:
        """Apply camera configuration."""

        if self._capture is None:
            return

        capture = self._capture

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.config.width,
        )

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.config.height,
        )

        capture.set(
            cv2.CAP_PROP_FPS,
            self.config.fps,
        )

        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            self.config.buffer_size,
        )

        if not self.config.auto_exposure:

            capture.set(
                cv2.CAP_PROP_AUTO_EXPOSURE,
                0.25,
            )

        if self.config.brightness is not None:

            capture.set(
                cv2.CAP_PROP_BRIGHTNESS,
                self.config.brightness,
            )

        if self.config.contrast is not None:

            capture.set(
                cv2.CAP_PROP_CONTRAST,
                self.config.contrast,
            )

        if self.config.saturation is not None:

            capture.set(
                cv2.CAP_PROP_SATURATION,
                self.config.saturation,
            )

        actual_width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        actual_fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        self._status.width = (
            actual_width
        )

        self._status.height = (
            actual_height
        )

        self._status.fps = (
            actual_fps
            if actual_fps > 0
            else float(self.config.fps)
        )

    def _warmup(self) -> None:
        """Read a few frames to stabilize the camera."""

        for _ in range(
            max(
                0,
                self.config.warmup_frames,
            )
        ):

            if self._capture is None:
                break

            try:

                self._capture.read()

            except Exception:

                break

    # ========================================================================
    # STATE
    # ========================================================================

    def is_open(self) -> bool:
        """Return whether the camera is currently open."""

        with self._lock:

            return bool(
                self._capture is not None
                and self._capture.isOpened()
            )

    @property
    def connected(self) -> bool:
        return self.is_open()

    @property
    def running(self) -> bool:

        with self._lock:
            return self._running

    # ========================================================================
    # READ FRAME
    # ========================================================================

    def read(
        self,
    ) -> Optional[Frame]:
        """
        Capture one frame.

        Returns
        -------
        Frame | None
            A RENIX Frame object or None if capture failed.
        """

        with self._lock:

            if not self.is_open():

                if not self.open():
                    return None

            try:

                success, image = (
                    self._capture.read()
                )

                if not success or image is None:

                    self._set_error(
                        "Camera returned an invalid frame."
                    )

                    return None

                image = self._apply_transforms(
                    image
                )

                self._frame_number += 1

                timestamp = time.time()

                height, width = (
                    image.shape[:2]
                )

                fps = (
                    self._status.fps
                    or float(self.config.fps)
                )

                frame = Frame(
                    data=image,
                    timestamp=timestamp,
                    frame_number=self._frame_number,
                    width=int(width),
                    height=int(height),
                    fps=float(fps),
                    camera_index=(
                        self.config.device_index
                    ),
                    metadata={
                        "source": "camera",
                        "backend": (
                            self.config.backend
                        ),
                    },
                )

                self._latest_frame = frame

                self._status.connected = True

                self._status.capturing = True

                self._status.frames_captured = (
                    self._frame_number
                )

                self._status.last_frame_time = (
                    timestamp
                )

                self._status.last_error = None

                callbacks = list(
                    self._callbacks
                )

            except Exception as exc:

                self._set_error(
                    str(exc)
                )

                logger.exception(
                    "Camera frame capture failed."
                )

                return None

        self._dispatch_frame_callbacks(
            frame,
            callbacks,
        )

        return frame

    # ========================================================================
    # TRANSFORMS
    # ========================================================================

    def _apply_transforms(
        self,
        image: Any,
    ) -> Any:

        if not CV2_AVAILABLE:
            return image

        if self.config.flip_horizontal:

            image = cv2.flip(
                image,
                1,
            )

        if self.config.flip_vertical:

            image = cv2.flip(
                image,
                0,
            )

        return image

    # ========================================================================
    # LATEST FRAME
    # ========================================================================

    def get_latest_frame(
        self,
    ) -> Optional[Frame]:

        with self._lock:

            return self._latest_frame

    # ========================================================================
    # STREAMING
    # ========================================================================

    def start_stream(
        self,
        callback: Optional[
            Any
        ] = None,
        *,
        interval: Optional[float] = None,
    ) -> bool:
        """
        Start background frame capture.

        Parameters
        ----------
        callback:
            Function receiving each Frame.

        interval:
            Optional delay between captures.
        """

        with self._lock:

            if self._running:

                if callback is not None:
                    self.add_callback(callback)

                return True

            if not self.is_open():

                if not self.open():
                    return False

            if callback is not None:

                self.add_callback(callback)

            self._stream_stop.clear()

            self._running = True

            self._status.capturing = True

            self._stream_thread = (
                threading.Thread(
                    target=self._stream_loop,
                    args=(interval,),
                    name="RENIX-CameraStream",
                    daemon=True,
                )
            )

            self._stream_thread.start()

            return True

    def _stream_loop(
        self,
        interval: Optional[float],
    ) -> None:

        target_interval = interval

        if target_interval is None:

            fps = (
                self.config.fps
                if self.config.fps > 0
                else 30
            )

            target_interval = 1.0 / fps

        while not self._stream_stop.is_set():

            started = time.monotonic()

            frame = self.read()

            if frame is None:

                if self._stream_stop.wait(
                    self.config.reconnect_delay
                ):

                    break

            elapsed = (
                time.monotonic()
                - started
            )

            remaining = (
                target_interval
                - elapsed
            )

            if remaining > 0:

                if self._stream_stop.wait(
                    remaining
                ):

                    break

        with self._lock:

            self._running = False

            self._status.capturing = False

    def stop_stream(
        self,
        timeout: float = 2.0,
    ) -> None:

        self._stream_stop.set()

        thread = self._stream_thread

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):

            thread.join(
                max(
                    0.0,
                    timeout,
                )
            )

        with self._lock:

            self._running = False

            self._status.capturing = False

            self._stream_thread = None

    def stream(
        self,
        *,
        max_frames: Optional[int] = None,
    ) -> Generator[Frame, None, None]:
        """
        Synchronous frame generator.

        Example:

            for frame in camera.stream():
                process(frame.data)
        """

        if not self.is_open():

            if not self.open():
                return

        count = 0

        while True:

            frame = self.read()

            if frame is None:

                break

            yield frame

            count += 1

            if (
                max_frames is not None
                and count >= max_frames
            ):

                break

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def add_callback(
        self,
        callback: Any,
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "Camera callback must be callable."
            )

        with self._lock:

            if callback not in self._callbacks:

                self._callbacks.append(
                    callback
                )

    def remove_callback(
        self,
        callback: Any,
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _dispatch_frame_callbacks(
        self,
        frame: Frame,
        callbacks: list,
    ) -> None:

        for callback in callbacks:

            try:

                callback(frame)

            except Exception:

                logger.exception(
                    "Camera frame callback failed."
                )

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(
        self,
    ) -> CameraStatus:

        with self._lock:

            return CameraStatus(
                connected=self._status.connected,
                capturing=self._status.capturing,
                device_index=self._status.device_index,
                width=self._status.width,
                height=self._status.height,
                fps=self._status.fps,
                frames_captured=(
                    self._status.frames_captured
                ),
                last_frame_time=(
                    self._status.last_frame_time
                ),
                last_error=self._status.last_error,
                started_at=self._status.started_at,
                metadata=dict(
                    self._status.metadata
                ),
            )

    # ========================================================================
    # ERROR MANAGEMENT
    # ========================================================================

    def _set_error(
        self,
        error: str,
    ) -> None:

        with self._lock:

            self._status.last_error = (
                str(error)
            )

    def get_last_error(
        self,
    ) -> Optional[str]:

        with self._lock:

            return self._status.last_error

    # ========================================================================
    # RELEASE
    # ========================================================================

    def _release_capture(self) -> None:

        if self._capture is not None:

            try:

                self._capture.release()

            except Exception:

                logger.exception(
                    "Failed to release camera."
                )

            finally:

                self._capture = None

    def close(self) -> None:
        """
        Stop streaming and release the camera.
        """

        self.stop_stream()

        with self._lock:

            self._release_capture()

            self._status.connected = False

            self._status.capturing = False

    release = close

    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================

    def __enter__(self) -> "Camera":

        if not self.open():

            raise RuntimeError(
                "Unable to open RENIX camera."
            )

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.close()

    # ========================================================================
    # DESTRUCTOR
    # ========================================================================

    def __del__(self) -> None:

        try:
            self.close()
        except Exception:
            pass


# ============================================================================
# DEFAULT CAMERA
# ============================================================================


_default_camera: Optional[
    Camera
] = None

_default_camera_lock = threading.RLock()


def get_default_camera() -> Camera:
    """
    Return the shared RENIX default camera.
    """

    global _default_camera

    with _default_camera_lock:

        if _default_camera is None:

            _default_camera = Camera()

        return _default_camera


def open_default_camera() -> bool:
    """Open the shared default camera."""

    return get_default_camera().open()


def read_default_camera() -> Optional[Frame]:
    """Read one frame from the shared default camera."""

    return get_default_camera().read()


def close_default_camera() -> None:
    """Close the shared default camera."""

    get_default_camera().close()


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "CameraConfig",
    "Frame",
    "CameraStatus",
    "Camera",
    "get_default_camera",
    "open_default_camera",
    "read_default_camera",
    "close_default_camera",
    "CV2_AVAILABLE",
]


