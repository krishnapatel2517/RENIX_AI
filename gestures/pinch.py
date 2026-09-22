"""
RENIX - Pinch Gesture Detector
==============================

Detects thumb-index finger pinch gestures from hand landmark data.

Responsibilities:
    - Calculate thumb/index distance
    - Detect pinch start/end
    - Track pinch strength
    - Provide stable hysteresis to prevent flickering
    - Track pinch position
    - Support configurable thresholds
    - Produce normalized pinch state

This module does NOT:
    - Move the mouse
    - Click the desktop
    - Control the holographic UI
    - Execute commands

Those responsibilities belong to the gesture engine / mapper / UI layers.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger("RENIX.gestures.pinch")


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_PINCH_THRESHOLD = 0.055
DEFAULT_RELEASE_THRESHOLD = 0.075
DEFAULT_MAX_DISTANCE = 0.25

MIN_THRESHOLD = 0.005
MAX_THRESHOLD = 1.0


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(frozen=True)
class Point3D:
    """3D normalized landmark position."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class PinchConfig:
    """
    Configuration for pinch detection.

    threshold:
        Distance below which a pinch is considered active.

    release_threshold:
        Distance above which an active pinch is released.

    The release threshold should be greater than the pinch threshold
    to provide hysteresis and prevent rapid state switching.
    """

    threshold: float = DEFAULT_PINCH_THRESHOLD

    release_threshold: float = (
        DEFAULT_RELEASE_THRESHOLD
    )

    max_distance: float = DEFAULT_MAX_DISTANCE

    smoothing: float = 0.35

    min_confidence: float = 0.45

    debounce_time: float = 0.035

    strength_curve: float = 1.0

    enabled: bool = True


@dataclass
class PinchState:
    """Current state of one pinch detector."""

    active: bool = False

    just_started: bool = False

    just_released: bool = False

    distance: float = 0.0

    smoothed_distance: float = 0.0

    strength: float = 0.0

    confidence: float = 0.0

    x: float = 0.0

    y: float = 0.0

    z: float = 0.0

    timestamp: float = 0.0

    duration: float = 0.0

    hand_id: Optional[str] = None

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:

        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:

        return {
            "active": self.active,
            "just_started": self.just_started,
            "just_released": self.just_released,
            "distance": self.distance,
            "smoothed_distance": self.smoothed_distance,
            "strength": self.strength,
            "confidence": self.confidence,
            "position": {
                "x": self.x,
                "y": self.y,
                "z": self.z,
            },
            "timestamp": self.timestamp,
            "duration": self.duration,
            "hand_id": self.hand_id,
            "metadata": dict(
                self.metadata or {}
            ),
        }


# ============================================================================
# PINCH DETECTOR
# ============================================================================


class PinchDetector:
    """
    Detects pinch gestures using thumb-tip and index-tip landmarks.

    The detector uses hysteresis:

        distance <= threshold
            -> pinch starts

        distance >= release_threshold
            -> pinch ends

    This prevents the pinch state from rapidly toggling when the fingers
    hover close to the boundary.
    """

    def __init__(
        self,
        config: Optional[PinchConfig] = None,
        *,
        hand_id: Optional[str] = None,
    ) -> None:

        self.config = (
            config
            or PinchConfig()
        )

        self.hand_id = hand_id

        self._lock = threading.RLock()

        self._active = False

        self._start_time: Optional[
            float
        ] = None

        self._last_update = 0.0

        self._last_distance = 0.0

        self._smoothed_distance = 0.0

        self._last_strength = 0.0

        self._last_confidence = 0.0

        self._last_position = Point3D(
            0.0,
            0.0,
            0.0,
        )

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        thumb_tip: Point3D | tuple[float, ...],
        index_tip: Point3D | tuple[float, ...],
        *,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        hand_id: Optional[str] = None,
    ) -> PinchState:

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        thumb = self._normalize_point(
            thumb_tip
        )

        index = self._normalize_point(
            index_tip
        )

        confidence = self._clamp(
            confidence,
            0.0,
            1.0,
        )

        distance = self.distance(
            thumb,
            index,
        )

        with self._lock:

            previous_active = self._active

            self._last_distance = distance

            if self._smoothed_distance == 0.0:

                self._smoothed_distance = (
                    distance
                )

            else:

                alpha = self._clamp(
                    self.config.smoothing,
                    0.0,
                    1.0,
                )

                self._smoothed_distance = (
                    self._smoothed_distance
                    * (1.0 - alpha)
                    + distance * alpha
                )

            effective_distance = (
                self._smoothed_distance
            )

            effective_confidence = confidence

            if (
                not self.config.enabled
                or confidence
                < self.config.min_confidence
            ):

                self._active = False

            else:

                if not self._active:

                    if (
                        effective_distance
                        <= self.config.threshold
                    ):

                        self._active = True

                        self._start_time = now

                else:

                    if (
                        effective_distance
                        >= self.config.release_threshold
                    ):

                        self._active = False

                        self._start_time = None

            just_started = (
                not previous_active
                and self._active
            )

            just_released = (
                previous_active
                and not self._active
            )

            strength = self._calculate_strength(
                effective_distance
            )

            midpoint = self.midpoint(
                thumb,
                index,
            )

            duration = 0.0

            if (
                self._active
                and self._start_time is not None
            ):

                duration = max(
                    0.0,
                    now - self._start_time,
                )

            self._last_strength = strength

            self._last_confidence = (
                effective_confidence
            )

            self._last_position = midpoint

            self._last_update = now

            if hand_id is not None:

                self.hand_id = hand_id

            return PinchState(
                active=self._active,
                just_started=just_started,
                just_released=just_released,
                distance=distance,
                smoothed_distance=(
                    effective_distance
                ),
                strength=strength,
                confidence=(
                    effective_confidence
                ),
                x=midpoint.x,
                y=midpoint.y,
                z=midpoint.z,
                timestamp=now,
                duration=duration,
                hand_id=self.hand_id,
                metadata={
                    "threshold": (
                        self.config.threshold
                    ),
                    "release_threshold": (
                        self.config.release_threshold
                    ),
                },
            )

    # ========================================================================
    # DISTANCE
    # ========================================================================

    @staticmethod
    def distance(
        a: Point3D | tuple[float, ...],
        b: Point3D | tuple[float, ...],
    ) -> float:

        p1 = PinchDetector._normalize_point(
            a
        )

        p2 = PinchDetector._normalize_point(
            b
        )

        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z

        return math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )

    @staticmethod
    def midpoint(
        a: Point3D | tuple[float, ...],
        b: Point3D | tuple[float, ...],
    ) -> Point3D:

        p1 = PinchDetector._normalize_point(
            a
        )

        p2 = PinchDetector._normalize_point(
            b
        )

        return Point3D(
            x=(p1.x + p2.x) / 2.0,
            y=(p1.y + p2.y) / 2.0,
            z=(p1.z + p2.z) / 2.0,
        )

    # ========================================================================
    # STRENGTH
    # ========================================================================

    def _calculate_strength(
        self,
        distance: float,
    ) -> float:

        threshold = self.config.threshold

        max_distance = max(
            self.config.max_distance,
            threshold + 1e-6,
        )

        if distance <= threshold:

            strength = 1.0

        elif distance >= max_distance:

            strength = 0.0

        else:

            strength = (
                max_distance - distance
            ) / (
                max_distance - threshold
            )

        curve = max(
            0.05,
            self.config.strength_curve,
        )

        strength = strength**curve

        return self._clamp(
            strength,
            0.0,
            1.0,
        )

    # ========================================================================
    # STATE ACCESS
    # ========================================================================

    def is_active(self) -> bool:

        with self._lock:

            return self._active

    @property
    def active(self) -> bool:

        return self.is_active()

    def get_strength(self) -> float:

        with self._lock:

            return self._last_strength

    @property
    def strength(self) -> float:

        return self.get_strength()

    def get_distance(self) -> float:

        with self._lock:

            return self._last_distance

    @property
    def current_distance(self) -> float:

        return self.get_distance()

    def get_smoothed_distance(self) -> float:

        with self._lock:

            return self._smoothed_distance

    def get_position(self) -> Point3D:

        with self._lock:

            return self._last_position

    def get_confidence(self) -> float:

        with self._lock:

            return self._last_confidence

    def get_duration(
        self,
        timestamp: Optional[float] = None,
    ) -> float:

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        with self._lock:

            if (
                not self._active
                or self._start_time is None
            ):

                return 0.0

            return max(
                0.0,
                now - self._start_time,
            )

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:

        with self._lock:

            self._active = False

            self._start_time = None

            self._last_update = 0.0

            self._last_distance = 0.0

            self._smoothed_distance = 0.0

            self._last_strength = 0.0

            self._last_confidence = 0.0

            self._last_position = Point3D(
                0.0,
                0.0,
                0.0,
            )

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_threshold(
        self,
        threshold: float,
        release_threshold: Optional[
            float
        ] = None,
    ) -> None:

        threshold = self._clamp(
            threshold,
            MIN_THRESHOLD,
            MAX_THRESHOLD,
        )

        if release_threshold is None:

            release_threshold = max(
                threshold * 1.35,
                threshold + 0.005,
            )

        release_threshold = self._clamp(
            release_threshold,
            threshold,
            MAX_THRESHOLD,
        )

        with self._lock:

            self.config.threshold = (
                threshold
            )

            self.config.release_threshold = (
                release_threshold
            )

    def set_enabled(
        self,
        enabled: bool,
    ) -> None:

        with self._lock:

            self.config.enabled = bool(
                enabled
            )

            if not enabled:

                self._active = False
                self._start_time = None

    def set_smoothing(
        self,
        smoothing: float,
    ) -> None:

        with self._lock:

            self.config.smoothing = (
                self._clamp(
                    smoothing,
                    0.0,
                    1.0,
                )
            )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def get_config(self) -> dict[str, Any]:

        with self._lock:

            return {
                "threshold": (
                    self.config.threshold
                ),
                "release_threshold": (
                    self.config.release_threshold
                ),
                "max_distance": (
                    self.config.max_distance
                ),
                "smoothing": (
                    self.config.smoothing
                ),
                "min_confidence": (
                    self.config.min_confidence
                ),
                "debounce_time": (
                    self.config.debounce_time
                ),
                "strength_curve": (
                    self.config.strength_curve
                ),
                "enabled": (
                    self.config.enabled
                ),
                "hand_id": self.hand_id,
            }

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _normalize_point(
        point: Point3D | tuple[float, ...],
    ) -> Point3D:

        if isinstance(
            point,
            Point3D,
        ):

            return point

        if not isinstance(
            point,
            (tuple, list),
        ):

            raise TypeError(
                "Point must be Point3D, tuple, or list"
            )

        if len(point) < 2:

            raise ValueError(
                "Point requires at least x and y"
            )

        x = float(point[0])
        y = float(point[1])

        z = (
            float(point[2])
            if len(point) >= 3
            else 0.0
        )

        return Point3D(
            x=x,
            y=y,
            z=z,
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
# TWO-HAND / MULTI-HAND PINCH
# ============================================================================


class MultiHandPinchDetector:
    """
    Manages pinch detection for multiple hands.

    Typical use:

        detector.update(
            hand_id="left",
            thumb_tip=...,
            index_tip=...
        )

        detector.update(
            hand_id="right",
            thumb_tip=...,
            index_tip=...
        )
    """

    def __init__(
        self,
        config: Optional[PinchConfig] = None,
    ) -> None:

        self.config = (
            config
            or PinchConfig()
        )

        self._lock = threading.RLock()

        self._detectors: dict[
            str,
            PinchDetector,
        ] = {}

    def get_detector(
        self,
        hand_id: str,
    ) -> PinchDetector:

        with self._lock:

            if hand_id not in self._detectors:

                self._detectors[
                    hand_id
                ] = PinchDetector(
                    self.config,
                    hand_id=hand_id,
                )

            return self._detectors[
                hand_id
            ]

    def update(
        self,
        hand_id: str,
        thumb_tip: Point3D | tuple[float, ...],
        index_tip: Point3D | tuple[float, ...],
        *,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> PinchState:

        detector = self.get_detector(
            hand_id
        )

        return detector.update(
            thumb_tip,
            index_tip,
            confidence=confidence,
            timestamp=timestamp,
            hand_id=hand_id,
        )

    def remove(
        self,
        hand_id: str,
    ) -> None:

        with self._lock:

            detector = self._detectors.pop(
                hand_id,
                None,
            )

            if detector is not None:

                detector.reset()

    def reset(self) -> None:

        with self._lock:

            for detector in (
                self._detectors.values()
            ):

                detector.reset()

    def get_active_hands(
        self,
    ) -> list[str]:

        with self._lock:

            return [
                hand_id
                for hand_id, detector
                in self._detectors.items()
                if detector.is_active()
            ]

    def get_states(
        self,
    ) -> dict[str, dict[str, Any]]:

        with self._lock:

            result: dict[
                str,
                dict[str, Any],
            ] = {}

            for (
                hand_id,
                detector,
            ) in self._detectors.items():

                result[
                    hand_id
                ] = {
                    "active": (
                        detector.is_active()
                    ),
                    "strength": (
                        detector.get_strength()
                    ),
                    "distance": (
                        detector.get_distance()
                    ),
                    "position": (
                        detector.get_position()
                    ),
                }

            return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


_default_detector: Optional[
    PinchDetector
] = None

_default_detector_lock = threading.RLock()


def get_pinch_detector() -> PinchDetector:

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                PinchDetector()
            )

        return _default_detector


def detect_pinch(
    thumb_tip: Point3D | tuple[float, ...],
    index_tip: Point3D | tuple[float, ...],
    *,
    confidence: float = 1.0,
    timestamp: Optional[float] = None,
) -> PinchState:

    return get_pinch_detector().update(
        thumb_tip,
        index_tip,
        confidence=confidence,
        timestamp=timestamp,
    )


def is_pinch(
    thumb_tip: Point3D | tuple[float, ...],
    index_tip: Point3D | tuple[float, ...],
    *,
    threshold: float = DEFAULT_PINCH_THRESHOLD,
) -> bool:

    distance = PinchDetector.distance(
        thumb_tip,
        index_tip,
    )

    return distance <= threshold


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "Point3D",
    "PinchConfig",
    "PinchState",
    "PinchDetector",
    "MultiHandPinchDetector",
    "get_pinch_detector",
    "detect_pinch",
    "is_pinch",
]


