"""
RENIX - Rotate Gesture Detector
===============================

Detects two-hand rotation gestures.

Typical RENIX usage:
    - Pinch with both hands
    - Move the two pinched hands around each other
    - Detect clockwise / counter-clockwise rotation
    - Provide rotation angle, delta, velocity and confidence

This module ONLY detects rotation.
It does not manipulate the UI or execute commands.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger("RENIX.gestures.rotate")


# ============================================================================
# ENUMS
# ============================================================================


class RotationDirection(str, Enum):
    NONE = "none"
    CLOCKWISE = "clockwise"
    COUNTER_CLOCKWISE = "counter_clockwise"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(frozen=True)
class RotationPoint:
    """2D/3D normalized point."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class RotateConfig:
    """Configuration for rotation detection."""

    min_radius: float = 0.025

    min_angle: float = 3.0

    activation_angle: float = 8.0

    release_angle: float = 3.0

    max_frame_delta: float = 0.30

    smoothing: float = 0.35

    confidence_threshold: float = 0.45

    cooldown: float = 0.15

    direction_deadzone: float = 0.5

    enabled: bool = True


@dataclass
class RotationEvent:
    """Result produced by the rotation detector."""

    detected: bool = False

    direction: RotationDirection = (
        RotationDirection.NONE
    )

    angle: float = 0.0

    delta_angle: float = 0.0

    absolute_angle: float = 0.0

    velocity: float = 0.0

    radius: float = 0.0

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
            "angle": self.angle,
            "delta_angle": self.delta_angle,
            "absolute_angle": self.absolute_angle,
            "velocity": self.velocity,
            "radius": self.radius,
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
# ROTATION DETECTOR
# ============================================================================


class RotateDetector:
    """
    Detects rotation from two tracked points.

    The two points can be:
        - left/right pinch positions
        - left/right hand centers
        - any two stable hand control points

    Angle is calculated from:

        atan2(right.y - left.y, right.x - left.x)

    Coordinate systems where Y increases downward are automatically handled
    so clockwise/counter-clockwise remain intuitive for screen interaction.
    """

    def __init__(
        self,
        config: Optional[RotateConfig] = None,
    ) -> None:

        self.config = (
            config
            or RotateConfig()
        )

        self._lock = threading.RLock()

        self._active = False

        self._start_angle: Optional[
            float
        ] = None

        self._previous_angle: Optional[
            float
        ] = None

        self._smoothed_angle: Optional[
            float
        ] = None

        self._total_rotation = 0.0

        self._start_time: Optional[
            float
        ] = None

        self._last_time = 0.0

        self._last_event = RotationEvent()

        self._last_direction = (
            RotationDirection.NONE
        )

        self._last_radius = 0.0

        self._last_confidence = 0.0

        self._callbacks: list[
            Callable[[RotationEvent], None]
        ] = []

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        left: RotationPoint | tuple[float, ...],
        right: RotationPoint | tuple[float, ...],
        *,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> RotationEvent:

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

                return RotationEvent(
                    timestamp=now
                )

            if (
                confidence
                < self.config.confidence_threshold
            ):

                return RotationEvent(
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

            dx = (
                right_point.x
                - left_point.x
            )

            dy = (
                right_point.y
                - left_point.y
            )

            radius = math.hypot(
                dx,
                dy
            ) / 2.0

            angle = math.degrees(
                math.atan2(
                    dy,
                    dx,
                )
            )

            if radius < self.config.min_radius:

                return RotationEvent(
                    timestamp=now,
                    confidence=confidence,
                    radius=radius,
                    center_x=center_x,
                    center_y=center_y,
                    left_x=left_point.x,
                    left_y=left_point.y,
                    right_x=right_point.x,
                    right_y=right_point.y,
                )

            if self._previous_angle is None:

                self._previous_angle = angle

                self._smoothed_angle = angle

                self._start_angle = angle

                self._start_time = now

                self._last_time = now

                self._last_radius = radius

                return RotationEvent(
                    timestamp=now,
                    confidence=confidence,
                    radius=radius,
                    center_x=center_x,
                    center_y=center_y,
                    left_x=left_point.x,
                    left_y=left_point.y,
                    right_x=right_point.x,
                    right_y=right_point.y,
                )

            # --------------------------------------------------------------
            # ANGLE DELTA
            # --------------------------------------------------------------

            raw_delta = self._angle_difference(
                angle,
                self._previous_angle,
            )

            # Ignore physically implausible frame jumps.
            if (
                abs(raw_delta)
                > self.config.max_frame_delta
                * 360.0
            ):

                self._previous_angle = angle

                return RotationEvent(
                    timestamp=now,
                    confidence=confidence,
                    radius=radius,
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

            if self._smoothed_angle is None:

                self._smoothed_angle = angle

            else:

                smoothed_delta = (
                    self._angle_difference(
                        angle,
                        self._smoothed_angle,
                    )
                )

                self._smoothed_angle = (
                    self._smoothed_angle
                    + smoothed_delta * alpha
                )

            effective_delta = (
                self._angle_difference(
                    self._smoothed_angle,
                    self._previous_angle,
                )
            )

            dt = max(
                now - self._last_time,
                1e-6,
            )

            velocity = (
                effective_delta
                / dt
            )

            self._previous_angle = (
                self._smoothed_angle
            )

            self._last_time = now

            self._last_radius = radius

            # --------------------------------------------------------------
            # ACCUMULATION
            # --------------------------------------------------------------

            self._total_rotation += (
                effective_delta
            )

            absolute_angle = abs(
                self._total_rotation
            )

            direction = (
                self._direction_from_delta(
                    effective_delta
                )
            )

            if (
                direction
                != RotationDirection.NONE
            ):

                self._last_direction = (
                    direction
                )

            # --------------------------------------------------------------
            # ACTIVATION
            # --------------------------------------------------------------

            if not self._active:

                if (
                    absolute_angle
                    >= self.config.activation_angle
                ):

                    self._active = True

                    self._start_time = (
                        self._start_time
                        if self._start_time
                        is not None
                        else now
                    )

            # --------------------------------------------------------------
            # RELEASE
            # --------------------------------------------------------------

            elif (
                absolute_angle
                >= self.config.release_angle
            ):

                self._active = True

            # --------------------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------------------

            direction_confidence = (
                self._calculate_direction_confidence(
                    effective_delta
                )
            )

            radius_confidence = (
                self._calculate_radius_confidence(
                    radius
                )
            )

            movement_confidence = (
                self._calculate_movement_confidence(
                    effective_delta,
                    dt,
                )
            )

            final_confidence = self._clamp(
                confidence
                * direction_confidence
                * radius_confidence
                * movement_confidence,
                0.0,
                1.0,
            )

            detected = (
                self._active
                and abs(effective_delta)
                >= self.config.min_angle
                and final_confidence
                >= self.config.confidence_threshold
            )

            duration = 0.0

            if self._start_time is not None:

                duration = max(
                    0.0,
                    now - self._start_time,
                )

            event = RotationEvent(
                detected=detected,
                direction=(
                    direction
                    if detected
                    else RotationDirection.NONE
                ),
                angle=effective_delta,
                delta_angle=effective_delta,
                absolute_angle=absolute_angle,
                velocity=velocity,
                radius=radius,
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
                    "raw_angle": angle,
                    "total_rotation": (
                        self._total_rotation
                    ),
                },
            )

            self._last_confidence = (
                final_confidence
            )

            self._last_event = event

            if detected:

                self._emit(event)

            return event

    # ========================================================================
    # ANGLE HELPERS
    # ========================================================================

    @staticmethod
    def _angle_difference(
        current: float,
        previous: float,
    ) -> float:

        difference = (
            current - previous
        )

        while difference > 180.0:

            difference -= 360.0

        while difference < -180.0:

            difference += 360.0

        return difference

    def _direction_from_delta(
        self,
        delta: float,
    ) -> RotationDirection:

        deadzone = max(
            0.0,
            self.config.direction_deadzone,
        )

        if delta > deadzone:

            # Screen Y normally increases downward.
            # Positive mathematical angle is interpreted as clockwise
            # for the screen-oriented gesture system.
            return RotationDirection.CLOCKWISE

        if delta < -deadzone:

            return (
                RotationDirection.COUNTER_CLOCKWISE
            )

        return RotationDirection.NONE

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _calculate_direction_confidence(
        self,
        delta: float,
    ) -> float:

        magnitude = abs(delta)

        if magnitude <= self.config.direction_deadzone:

            return 0.0

        if magnitude >= 8.0:

            return 1.0

        return self._clamp(
            magnitude / 8.0,
            0.0,
            1.0,
        )

    def _calculate_radius_confidence(
        self,
        radius: float,
    ) -> float:

        minimum = max(
            self.config.min_radius,
            1e-6,
        )

        ratio = radius / minimum

        if ratio >= 2.0:

            return 1.0

        return self._clamp(
            ratio / 2.0,
            0.0,
            1.0,
        )

    def _calculate_movement_confidence(
        self,
        delta: float,
        dt: float,
    ) -> float:

        if dt <= 0.0:

            return 0.0

        angular_speed = abs(
            delta
        ) / dt

        if angular_speed <= 1.0:

            return 0.35

        if angular_speed >= 90.0:

            return 1.0

        return self._clamp(
            angular_speed / 90.0,
            0.35,
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

    def get_total_rotation(self) -> float:

        with self._lock:

            return self._total_rotation

    def get_direction(
        self,
    ) -> RotationDirection:

        with self._lock:

            return self._last_direction

    def get_last_event(
        self,
    ) -> RotationEvent:

        with self._lock:

            return self._last_event

    def get_velocity(self) -> float:

        with self._lock:

            return self._last_event.velocity

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def subscribe(
        self,
        callback: Callable[[RotationEvent], None],
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
        callback: Callable[[RotationEvent], None],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: RotationEvent,
    ) -> None:

        callbacks = list(
            self._callbacks
        )

        for callback in callbacks:

            try:

                callback(event)

            except Exception:

                logger.exception(
                    "Rotation callback failed"
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
        min_angle: Optional[float] = None,
        activation_angle: Optional[
            float
        ] = None,
    ) -> None:

        with self._lock:

            if min_angle is not None:

                self.config.min_angle = max(
                    0.1,
                    float(min_angle),
                )

            if activation_angle is not None:

                self.config.activation_angle = max(
                    self.config.min_angle,
                    float(activation_angle),
                )

    def get_config(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "min_radius": (
                    self.config.min_radius
                ),
                "min_angle": (
                    self.config.min_angle
                ),
                "activation_angle": (
                    self.config.activation_angle
                ),
                "release_angle": (
                    self.config.release_angle
                ),
                "max_frame_delta": (
                    self.config.max_frame_delta
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
                "direction_deadzone": (
                    self.config.direction_deadzone
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

            self._start_angle = None

            self._previous_angle = None

            self._smoothed_angle = None

            self._total_rotation = 0.0

            self._start_time = None

            self._last_time = 0.0

            self._last_event = RotationEvent()

            self._last_direction = (
                RotationDirection.NONE
            )

            self._last_radius = 0.0

            self._last_confidence = 0.0

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _normalize_point(
        point: RotationPoint
        | tuple[float, ...],
    ) -> RotationPoint:

        if isinstance(
            point,
            RotationPoint,
        ):

            return point

        if not isinstance(
            point,
            (tuple, list),
        ):

            raise TypeError(
                "Point must be RotationPoint, tuple, or list"
            )

        if len(point) < 2:

            raise ValueError(
                "Point requires x and y"
            )

        return RotationPoint(
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
    RotateDetector
] = None

_default_lock = threading.RLock()


def get_rotate_detector() -> RotateDetector:

    global _default_detector

    with _default_lock:

        if _default_detector is None:

            _default_detector = (
                RotateDetector()
            )

        return _default_detector


def detect_rotation(
    left: RotationPoint | tuple[float, ...],
    right: RotationPoint | tuple[float, ...],
    *,
    confidence: float = 1.0,
    timestamp: Optional[float] = None,
) -> RotationEvent:

    return get_rotate_detector().update(
        left,
        right,
        confidence=confidence,
        timestamp=timestamp,
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "RotationPoint",
    "RotationDirection",
    "RotateConfig",
    "RotationEvent",
    "RotateDetector",
    "get_rotate_detector",
    "detect_rotation",
]


