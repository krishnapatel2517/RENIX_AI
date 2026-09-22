"""
RENIX - Zoom Gesture Detector
=============================

Detects two-hand pinch-distance zoom gestures.

Typical RENIX usage:
    - Pinch with both hands.
    - Move hands apart -> zoom in.
    - Move hands together -> zoom out.

This module only detects the gesture.
It does not directly control the UI, mouse, browser, camera, or desktop.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger("RENIX.gestures.zoom")


# ============================================================================
# ENUMS
# ============================================================================


class ZoomDirection(str, Enum):
    NONE = "none"
    IN = "in"
    OUT = "out"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(frozen=True)
class ZoomPoint:
    """Normalized 2D/3D point."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class ZoomConfig:
    """Configuration for two-hand zoom detection."""

    min_distance: float = 0.05

    max_distance: float = 1.50

    min_delta: float = 0.008

    activation_delta: float = 0.025

    smoothing: float = 0.35

    confidence_threshold: float = 0.45

    cooldown: float = 0.08

    max_frame_time: float = 0.30

    enabled: bool = True


@dataclass
class ZoomEvent:
    """Result produced by the zoom detector."""

    detected: bool = False

    direction: ZoomDirection = (
        ZoomDirection.NONE
    )

    distance: float = 0.0

    previous_distance: float = 0.0

    delta: float = 0.0

    scale: float = 1.0

    velocity: float = 0.0

    confidence: float = 0.0

    duration: float = 0.0

    center_x: float = 0.0
    center_y: float = 0.0

    left_x: float = 0.0
    left_y: float = 0.0

    right_x: float = 0.0
    right_y: float = 0.0

    timestamp: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "direction": self.direction.value,
            "distance": self.distance,
            "previous_distance": (
                self.previous_distance
            ),
            "delta": self.delta,
            "scale": self.scale,
            "velocity": self.velocity,
            "confidence": self.confidence,
            "duration": self.duration,
            "center": {
                "x": self.center_x,
                "y": self.center_y,
            },
            "left": {
                "x": self.left_x,
                "y": self.left_y,
            },
            "right": {
                "x": self.right_x,
                "y": self.right_y,
            },
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# ZOOM DETECTOR
# ============================================================================


class ZoomDetector:
    """
    Detects zoom from the distance between two hand control points.

    The points are typically the tips of the left and right index fingers,
    or the centers of two active pinches.

    Increasing distance:
        Zoom IN

    Decreasing distance:
        Zoom OUT
    """

    def __init__(
        self,
        config: Optional[ZoomConfig] = None,
    ) -> None:

        self.config = (
            config
            or ZoomConfig()
        )

        self._lock = threading.RLock()

        self._active = False

        self._previous_distance: Optional[
            float
        ] = None

        self._smoothed_distance: Optional[
            float
        ] = None

        self._initial_distance: Optional[
            float
        ] = None

        self._start_time: Optional[
            float
        ] = None

        self._last_time: float = 0.0

        self._last_event = ZoomEvent()

        self._last_direction = (
            ZoomDirection.NONE
        )

        self._callbacks: list[
            Callable[[ZoomEvent], None]
        ] = []

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        left: ZoomPoint | tuple[float, ...],
        right: ZoomPoint | tuple[float, ...],
        *,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> ZoomEvent:

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        left_point = self._normalize_point(
            left
        )

        right_point = self._normalize_point(
            right
        )

        confidence = self._clamp(
            confidence,
            0.0,
            1.0,
        )

        with self._lock:

            if not self.config.enabled:

                self.reset()

                return ZoomEvent(
                    timestamp=now
                )

            if (
                confidence
                < self.config.confidence_threshold
            ):

                return ZoomEvent(
                    timestamp=now,
                    confidence=confidence,
                )

            center_x = (
                left_point.x
                + right_point.x
            ) / 2.0

            center_y = (
                left_point.y
                + right_point.y
            ) / 2.0

            distance = self._distance(
                left_point,
                right_point,
            )

            if (
                distance
                < self.config.min_distance
            ):

                return ZoomEvent(
                    timestamp=now,
                    distance=distance,
                    confidence=confidence,
                    center_x=center_x,
                    center_y=center_y,
                    left_x=left_point.x,
                    left_y=left_point.y,
                    right_x=right_point.x,
                    right_y=right_point.y,
                )

            if (
                distance
                > self.config.max_distance
            ):

                distance = (
                    self.config.max_distance
                )

            # --------------------------------------------------------------
            # FIRST FRAME
            # --------------------------------------------------------------

            if (
                self._previous_distance
                is None
            ):

                self._previous_distance = (
                    distance
                )

                self._smoothed_distance = (
                    distance
                )

                self._initial_distance = (
                    distance
                )

                self._start_time = now

                self._last_time = now

                return ZoomEvent(
                    timestamp=now,
                    distance=distance,
                    previous_distance=distance,
                    confidence=confidence,
                    center_x=center_x,
                    center_y=center_y,
                    left_x=left_point.x,
                    left_y=left_point.y,
                    right_x=right_point.x,
                    right_y=right_point.y,
                )

            # --------------------------------------------------------------
            # SMOOTHING
            # --------------------------------------------------------------

            alpha = self._clamp(
                self.config.smoothing,
                0.0,
                1.0,
            )

            if (
                self._smoothed_distance
                is None
            ):

                self._smoothed_distance = (
                    distance
                )

            else:

                self._smoothed_distance = (
                    self._smoothed_distance
                    * (1.0 - alpha)
                    + distance * alpha
                )

            previous = (
                self._previous_distance
            )

            current = (
                self._smoothed_distance
            )

            delta = current - previous

            dt = max(
                now - self._last_time,
                1e-6,
            )

            # --------------------------------------------------------------
            # FRAME VALIDATION
            # --------------------------------------------------------------

            if (
                dt
                > self.config.max_frame_time
            ):

                self._previous_distance = (
                    current
                )

                self._last_time = now

                return ZoomEvent(
                    timestamp=now,
                    distance=current,
                    previous_distance=previous,
                    confidence=confidence,
                    center_x=center_x,
                    center_y=center_y,
                    left_x=left_point.x,
                    left_y=left_point.y,
                    right_x=right_point.x,
                    right_y=right_point.y,
                )

            # --------------------------------------------------------------
            # VELOCITY
            # --------------------------------------------------------------

            velocity = delta / dt

            # --------------------------------------------------------------
            # DIRECTION
            # --------------------------------------------------------------

            if (
                delta
                >= self.config.min_delta
            ):

                direction = ZoomDirection.IN

            elif (
                delta
                <= -self.config.min_delta
            ):

                direction = ZoomDirection.OUT

            else:

                direction = ZoomDirection.NONE

            # --------------------------------------------------------------
            # ACCUMULATED MOVEMENT
            # --------------------------------------------------------------

            initial_distance = (
                self._initial_distance
            )

            if initial_distance is None:

                initial_distance = current

                self._initial_distance = (
                    current
                )

            total_delta = (
                current
                - initial_distance
            )

            # --------------------------------------------------------------
            # ACTIVATION
            # --------------------------------------------------------------

            if not self._active:

                if (
                    abs(total_delta)
                    >= self.config.activation_delta
                ):

                    self._active = True

            # --------------------------------------------------------------
            # SCALE
            # --------------------------------------------------------------

            scale = (
                current
                / max(
                    initial_distance,
                    1e-6,
                )
            )

            # --------------------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------------------

            movement_confidence = (
                self._movement_confidence(
                    abs(delta)
                )
            )

            distance_confidence = (
                self._distance_confidence(
                    current
                )
            )

            direction_confidence = (
                self._direction_confidence(
                    abs(delta)
                )
            )

            final_confidence = self._clamp(
                confidence
                * movement_confidence
                * distance_confidence
                * direction_confidence,
                0.0,
                1.0,
            )

            detected = (
                self._active
                and direction
                != ZoomDirection.NONE
                and final_confidence
                >= self.config.confidence_threshold
            )

            duration = 0.0

            if self._start_time is not None:

                duration = max(
                    0.0,
                    now - self._start_time,
                )

            event = ZoomEvent(
                detected=detected,
                direction=(
                    direction
                    if detected
                    else ZoomDirection.NONE
                ),
                distance=current,
                previous_distance=previous,
                delta=delta,
                scale=scale,
                velocity=velocity,
                confidence=final_confidence,
                duration=duration,
                center_x=center_x,
                center_y=center_y,
                left_x=left_point.x,
                left_y=left_point.y,
                right_x=right_point.x,
                right_y=right_point.y,
                timestamp=now,
                metadata={
                    "input_confidence": confidence,
                    "initial_distance": (
                        initial_distance
                    ),
                    "total_delta": total_delta,
                },
            )

            self._previous_distance = current

            self._last_time = now

            if direction != ZoomDirection.NONE:

                self._last_direction = direction

            self._last_event = event

            if detected:

                self._emit(event)

            return event

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _movement_confidence(
        self,
        delta: float,
    ) -> float:

        minimum = max(
            self.config.min_delta,
            1e-6,
        )

        ratio = delta / minimum

        if ratio >= 3.0:

            return 1.0

        if ratio <= 1.0:

            return 0.55

        return self._clamp(
            0.55
            + 0.45
            * (
                (ratio - 1.0)
                / 2.0
            ),
            0.0,
            1.0,
        )

    def _distance_confidence(
        self,
        distance: float,
    ) -> float:

        minimum = max(
            self.config.min_distance,
            1e-6,
        )

        maximum = max(
            self.config.max_distance,
            minimum + 1e-6,
        )

        if (
            distance < minimum
            or distance > maximum
        ):

            return 0.0

        midpoint = (
            minimum + maximum
        ) / 2.0

        if distance >= midpoint:

            return 1.0

        return self._clamp(
            (
                distance - minimum
            )
            / max(
                midpoint - minimum,
                1e-6,
            ),
            0.50,
            1.0,
        )

    def _direction_confidence(
        self,
        delta: float,
    ) -> float:

        minimum = max(
            self.config.min_delta,
            1e-6,
        )

        ratio = delta / minimum

        if ratio >= 2.0:

            return 1.0

        return self._clamp(
            0.55
            + 0.45
            * (
                ratio - 1.0
            ),
            0.55,
            1.0,
        )

    # ========================================================================
    # STATE
    # ========================================================================

    @property
    def active(self) -> bool:

        with self._lock:

            return self._active

    def is_active(self) -> bool:

        return self.active

    def get_scale(self) -> float:

        with self._lock:

            if (
                self._initial_distance
                is None
                or self._smoothed_distance
                is None
            ):

                return 1.0

            return (
                self._smoothed_distance
                / max(
                    self._initial_distance,
                    1e-6,
                )
            )

    def get_direction(
        self,
    ) -> ZoomDirection:

        with self._lock:

            return self._last_direction

    def get_distance(self) -> float:

        with self._lock:

            return (
                self._smoothed_distance
                or 0.0
            )

    def get_velocity(self) -> float:

        with self._lock:

            return self._last_event.velocity

    def get_last_event(
        self,
    ) -> ZoomEvent:

        with self._lock:

            return self._last_event

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def subscribe(
        self,
        callback: Callable[[ZoomEvent], None],
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
        callback: Callable[[ZoomEvent], None],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: ZoomEvent,
    ) -> None:

        callbacks = list(
            self._callbacks
        )

        for callback in callbacks:

            try:

                callback(event)

            except Exception:

                logger.exception(
                    "Zoom callback failed"
                )

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_enabled(
        self,
        enabled: bool,
    ) -> None:

        with self._lock:

            self.config.enabled = bool(
                enabled
            )

            if not enabled:

                self.reset()

    def set_threshold(
        self,
        min_delta: Optional[float] = None,
        activation_delta: Optional[
            float
        ] = None,
    ) -> None:

        with self._lock:

            if min_delta is not None:

                self.config.min_delta = max(
                    0.0001,
                    float(min_delta),
                )

            if activation_delta is not None:

                self.config.activation_delta = max(
                    self.config.min_delta,
                    float(activation_delta),
                )

    def get_config(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "min_distance": (
                    self.config.min_distance
                ),
                "max_distance": (
                    self.config.max_distance
                ),
                "min_delta": (
                    self.config.min_delta
                ),
                "activation_delta": (
                    self.config.activation_delta
                ),
                "smoothing": (
                    self.config.smoothing
                ),
                "confidence_threshold": (
                    self.config.confidence_threshold
                ),
                "cooldown": (
                    self.config.cooldown
                ),
                "max_frame_time": (
                    self.config.max_frame_time
                ),
                "enabled": (
                    self.config.enabled
                ),
            }

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:

        with self._lock:

            self._active = False

            self._previous_distance = None

            self._smoothed_distance = None

            self._initial_distance = None

            self._start_time = None

            self._last_time = 0.0

            self._last_event = ZoomEvent()

            self._last_direction = (
                ZoomDirection.NONE
            )

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _distance(
        a: ZoomPoint,
        b: ZoomPoint,
    ) -> float:

        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z

        return math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )

    @staticmethod
    def _normalize_point(
        point: ZoomPoint
        | tuple[float, ...],
    ) -> ZoomPoint:

        if isinstance(
            point,
            ZoomPoint,
        ):

            return point

        if not isinstance(
            point,
            (tuple, list),
        ):

            raise TypeError(
                "Point must be ZoomPoint, tuple, or list"
            )

        if len(point) < 2:

            raise ValueError(
                "Point requires x and y"
            )

        return ZoomPoint(
            x=float(point[0]),
            y=float(point[1]),
            z=(
                float(point[2])
                if len(point) >= 3
                else 0.0
            ),
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )


# ============================================================================
# CONVENIENCE API
# ============================================================================


_default_detector: Optional[
    ZoomDetector
] = None

_default_lock = threading.RLock()


def get_zoom_detector() -> ZoomDetector:

    global _default_detector

    with _default_lock:

        if _default_detector is None:

            _default_detector = (
                ZoomDetector()
            )

        return _default_detector


def detect_zoom(
    left: ZoomPoint | tuple[float, ...],
    right: ZoomPoint | tuple[float, ...],
    *,
    confidence: float = 1.0,
    timestamp: Optional[float] = None,
) -> ZoomEvent:

    return get_zoom_detector().update(
        left,
        right,
        confidence=confidence,
        timestamp=timestamp,
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "ZoomPoint",
    "ZoomDirection",
    "ZoomConfig",
    "ZoomEvent",
    "ZoomDetector",
    "get_zoom_detector",
    "detect_zoom",
]


