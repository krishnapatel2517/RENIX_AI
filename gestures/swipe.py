"""
RENIX - Swipe Gesture Detector
==============================

Detects horizontal and vertical swipe gestures from hand movement.

Responsibilities:
    - Track hand/wrist movement
    - Detect swipe direction
    - Calculate swipe distance, speed and confidence
    - Prevent repeated triggering from one movement
    - Support configurable thresholds
    - Provide stable swipe events

This module does NOT execute desktop/UI actions.
It only detects and reports swipe gestures.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


logger = logging.getLogger("RENIX.gestures.swipe")


# ============================================================================
# ENUMS
# ============================================================================


class SwipeDirection(str, Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(frozen=True)
class SwipePoint:
    """A timestamped hand position."""

    x: float
    y: float
    z: float = 0.0
    timestamp: float = 0.0


@dataclass
class SwipeConfig:
    """Configuration for swipe detection."""

    min_distance: float = 0.12

    max_duration: float = 0.75

    min_speed: float = 0.30

    max_speed: float = 8.0

    direction_ratio: float = 1.20

    confidence_threshold: float = 0.45

    cooldown: float = 0.25

    history_size: int = 20

    smoothing: float = 0.35

    enabled: bool = True

    require_return: bool = False


@dataclass
class SwipeEvent:
    """Result generated when a swipe is detected."""

    detected: bool = False

    direction: SwipeDirection = SwipeDirection.NONE

    distance: float = 0.0

    horizontal_distance: float = 0.0

    vertical_distance: float = 0.0

    speed: float = 0.0

    duration: float = 0.0

    confidence: float = 0.0

    start_x: float = 0.0
    start_y: float = 0.0

    end_x: float = 0.0
    end_y: float = 0.0

    dx: float = 0.0
    dy: float = 0.0

    hand_id: Optional[str] = None

    timestamp: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "direction": self.direction.value,
            "distance": self.distance,
            "horizontal_distance": self.horizontal_distance,
            "vertical_distance": self.vertical_distance,
            "speed": self.speed,
            "duration": self.duration,
            "confidence": self.confidence,
            "start": {
                "x": self.start_x,
                "y": self.start_y,
            },
            "end": {
                "x": self.end_x,
                "y": self.end_y,
            },
            "delta": {
                "x": self.dx,
                "y": self.dy,
            },
            "hand_id": self.hand_id,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# SWIPE DETECTOR
# ============================================================================


class SwipeDetector:
    """
    Detects swipes from a sequence of hand positions.

    A swipe is detected when:

        1. Movement exceeds min_distance.
        2. Movement happens within max_duration.
        3. Speed is above min_speed.
        4. One axis clearly dominates the other.
        5. Cooldown has elapsed since the previous swipe.

    Coordinates are expected to be normalized, normally in the range
    0.0 -> 1.0.
    """

    def __init__(
        self,
        config: Optional[SwipeConfig] = None,
        *,
        hand_id: Optional[str] = None,
    ) -> None:

        self.config = (
            config
            or SwipeConfig()
        )

        self.hand_id = hand_id

        self._lock = threading.RLock()

        self._history: deque[
            SwipePoint
        ] = deque(
            maxlen=max(
                2,
                self.config.history_size,
            )
        )

        self._last_swipe_time = 0.0

        self._last_event = SwipeEvent()

        self._callbacks: list[
            Callable[[SwipeEvent], None]
        ] = []

        self._smoothed_x: Optional[float] = None
        self._smoothed_y: Optional[float] = None

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        x: float,
        y: float,
        *,
        z: float = 0.0,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        hand_id: Optional[str] = None,
    ) -> SwipeEvent:

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        x = float(x)
        y = float(y)
        z = float(z)

        confidence = self._clamp(
            confidence,
            0.0,
            1.0,
        )

        with self._lock:

            if hand_id is not None:
                self.hand_id = hand_id

            if not self.config.enabled:

                return SwipeEvent(
                    timestamp=now,
                    hand_id=self.hand_id,
                )

            if (
                confidence
                < self.config.confidence_threshold
            ):

                return SwipeEvent(
                    timestamp=now,
                    hand_id=self.hand_id,
                    confidence=confidence,
                )

            x, y = self._smooth(
                x,
                y,
            )

            point = SwipePoint(
                x=x,
                y=y,
                z=z,
                timestamp=now,
            )

            self._history.append(point)

            event = self._detect(
                confidence=confidence,
                now=now,
            )

            if event.detected:

                self._last_swipe_time = now

                self._last_event = event

                self._emit(event)

                self._clear_after_detection()

            return event

    # ========================================================================
    # DETECTION
    # ========================================================================

    def _detect(
        self,
        *,
        confidence: float,
        now: float,
    ) -> SwipeEvent:

        if len(self._history) < 2:

            return SwipeEvent(
                timestamp=now,
                hand_id=self.hand_id,
                confidence=confidence,
            )

        if (
            now - self._last_swipe_time
            < self.config.cooldown
        ):

            return SwipeEvent(
                timestamp=now,
                hand_id=self.hand_id,
                confidence=confidence,
            )

        current = self._history[-1]

        # Search backwards for the earliest useful point
        # still inside the configured maximum duration.
        start = None

        for point in reversed(self._history):

            if (
                current.timestamp
                - point.timestamp
                <= self.config.max_duration
            ):

                start = point

            else:

                break

        if start is None:

            start = self._history[0]

        duration = (
            current.timestamp
            - start.timestamp
        )

        if duration <= 0.0:

            return SwipeEvent(
                timestamp=now,
                hand_id=self.hand_id,
                confidence=confidence,
            )

        if duration > self.config.max_duration:

            return SwipeEvent(
                timestamp=now,
                hand_id=self.hand_id,
                confidence=confidence,
            )

        dx = current.x - start.x
        dy = current.y - start.y

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        distance = math.hypot(
            dx,
            dy,
        )

        if (
            distance
            < self.config.min_distance
        ):

            return SwipeEvent(
                timestamp=now,
                hand_id=self.hand_id,
                confidence=confidence,
            )

        speed = (
            distance / duration
        )

        if speed < self.config.min_speed:

            return SwipeEvent(
                timestamp=now,
                hand_id=self.hand_id,
                confidence=confidence,
            )

        if speed > self.config.max_speed:

            speed_confidence = 0.65

        else:

            speed_confidence = 1.0

        direction = self._get_direction(
            dx,
            dy,
        )

        if (
            direction
            == SwipeDirection.NONE
        ):

            return SwipeEvent(
                timestamp=now,
                hand_id=self.hand_id,
                confidence=confidence,
            )

        distance_confidence = self._distance_confidence(
            distance
        )

        duration_confidence = self._duration_confidence(
            duration
        )

        direction_confidence = self._direction_confidence(
            abs_dx,
            abs_dy,
        )

        final_confidence = (
            confidence
            * distance_confidence
            * duration_confidence
            * direction_confidence
            * speed_confidence
        )

        final_confidence = self._clamp(
            final_confidence,
            0.0,
            1.0,
        )

        if (
            final_confidence
            < self.config.confidence_threshold
        ):

            return SwipeEvent(
                timestamp=now,
                hand_id=self.hand_id,
                confidence=final_confidence,
            )

        return SwipeEvent(
            detected=True,
            direction=direction,
            distance=distance,
            horizontal_distance=abs_dx,
            vertical_distance=abs_dy,
            speed=speed,
            duration=duration,
            confidence=final_confidence,
            start_x=start.x,
            start_y=start.y,
            end_x=current.x,
            end_y=current.y,
            dx=dx,
            dy=dy,
            hand_id=self.hand_id,
            timestamp=now,
            metadata={
                "axis_ratio": self._axis_ratio(
                    abs_dx,
                    abs_dy,
                ),
                "input_confidence": confidence,
            },
        )

    # ========================================================================
    # DIRECTION
    # ========================================================================

    def _get_direction(
        self,
        dx: float,
        dy: float,
    ) -> SwipeDirection:

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        ratio = max(
            1.0,
            self.config.direction_ratio,
        )

        if (
            abs_dx >= abs_dy * ratio
        ):

            if dx > 0:
                return SwipeDirection.RIGHT

            if dx < 0:
                return SwipeDirection.LEFT

        elif (
            abs_dy >= abs_dx * ratio
        ):

            if dy > 0:
                return SwipeDirection.DOWN

            if dy < 0:
                return SwipeDirection.UP

        return SwipeDirection.NONE

    @staticmethod
    def _axis_ratio(
        abs_dx: float,
        abs_dy: float,
    ) -> float:

        major = max(
            abs_dx,
            abs_dy,
        )

        minor = min(
            abs_dx,
            abs_dy,
        )

        if minor <= 1e-9:

            return float("inf")

        return major / minor

    def _direction_confidence(
        self,
        abs_dx: float,
        abs_dy: float,
    ) -> float:

        major = max(
            abs_dx,
            abs_dy,
        )

        minor = min(
            abs_dx,
            abs_dy,
        )

        if major <= 1e-9:

            return 0.0

        ratio = major / max(
            minor,
            1e-9,
        )

        required = max(
            1.0,
            self.config.direction_ratio,
        )

        if ratio >= required * 2.0:

            return 1.0

        if ratio <= required:

            return 0.70

        return (
            0.70
            + 0.30
            * (
                (ratio - required)
                / required
            )
        )

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _distance_confidence(
        self,
        distance: float,
    ) -> float:

        threshold = max(
            self.config.min_distance,
            1e-9,
        )

        ratio = distance / threshold

        if ratio >= 2.0:

            return 1.0

        return self._clamp(
            0.55
            + 0.45 * (ratio - 1.0),
            0.0,
            1.0,
        )

    def _duration_confidence(
        self,
        duration: float,
    ) -> float:

        if duration <= 0.0:
            return 0.0

        max_duration = max(
            self.config.max_duration,
            1e-9,
        )

        normalized = (
            duration / max_duration
        )

        # Extremely slow movements are less likely to be intentional
        # swipes, but do not punish short, fast gestures.
        if normalized <= 0.60:

            return 1.0

        return self._clamp(
            1.0
            - (
                normalized - 0.60
            )
            / 0.40
            * 0.40,
            0.60,
            1.0,
        )

    # ========================================================================
    # SMOOTHING
    # ========================================================================

    def _smooth(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:

        alpha = self._clamp(
            self.config.smoothing,
            0.0,
            1.0,
        )

        if (
            self._smoothed_x is None
            or self._smoothed_y is None
        ):

            self._smoothed_x = x
            self._smoothed_y = y

            return x, y

        self._smoothed_x = (
            self._smoothed_x
            * (1.0 - alpha)
            + x * alpha
        )

        self._smoothed_y = (
            self._smoothed_y
            * (1.0 - alpha)
            + y * alpha
        )

        return (
            self._smoothed_x,
            self._smoothed_y,
        )

    # ========================================================================
    # HISTORY
    # ========================================================================

    def _clear_after_detection(self) -> None:

        if not self._history:
            return

        last = self._history[-1]

        self._history.clear()

        self._history.append(last)

    def clear_history(self) -> None:

        with self._lock:

            self._history.clear()

            self._smoothed_x = None
            self._smoothed_y = None

    def get_history(self) -> list[SwipePoint]:

        with self._lock:

            return list(
                self._history
            )

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def subscribe(
        self,
        callback: Callable[[SwipeEvent], None],
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
        callback: Callable[[SwipeEvent], None],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: SwipeEvent,
    ) -> None:

        callbacks = list(
            self._callbacks
        )

        for callback in callbacks:

            try:

                callback(event)

            except Exception:

                logger.exception(
                    "Swipe callback failed"
                )

    # ========================================================================
    # STATE
    # ========================================================================

    def get_last_event(self) -> SwipeEvent:

        with self._lock:

            return self._last_event

    def reset(self) -> None:

        with self._lock:

            self._history.clear()

            self._last_swipe_time = 0.0

            self._last_event = SwipeEvent()

            self._smoothed_x = None
            self._smoothed_y = None

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

    # ========================================================================
    # CONFIG
    # ========================================================================

    def set_threshold(
        self,
        distance: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> None:

        with self._lock:

            if distance is not None:

                self.config.min_distance = max(
                    0.001,
                    float(distance),
                )

            if speed is not None:

                self.config.min_speed = max(
                    0.001,
                    float(speed),
                )

    def get_config(self) -> dict[str, Any]:

        with self._lock:

            return {
                "min_distance": (
                    self.config.min_distance
                ),
                "max_duration": (
                    self.config.max_duration
                ),
                "min_speed": (
                    self.config.min_speed
                ),
                "max_speed": (
                    self.config.max_speed
                ),
                "direction_ratio": (
                    self.config.direction_ratio
                ),
                "confidence_threshold": (
                    self.config.confidence_threshold
                ),
                "cooldown": (
                    self.config.cooldown
                ),
                "history_size": (
                    self.config.history_size
                ),
                "smoothing": (
                    self.config.smoothing
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
# MULTI-HAND SWIPE DETECTOR
# ============================================================================


class MultiHandSwipeDetector:
    """
    Maintains one SwipeDetector per hand.
    """

    def __init__(
        self,
        config: Optional[SwipeConfig] = None,
    ) -> None:

        self.config = (
            config
            or SwipeConfig()
        )

        self._lock = threading.RLock()

        self._detectors: dict[
            str,
            SwipeDetector,
        ] = {}

    def get_detector(
        self,
        hand_id: str,
    ) -> SwipeDetector:

        with self._lock:

            if hand_id not in self._detectors:

                self._detectors[
                    hand_id
                ] = SwipeDetector(
                    self.config,
                    hand_id=hand_id,
                )

            return self._detectors[
                hand_id
            ]

    def update(
        self,
        hand_id: str,
        x: float,
        y: float,
        *,
        z: float = 0.0,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> SwipeEvent:

        detector = self.get_detector(
            hand_id
        )

        return detector.update(
            x,
            y,
            z=z,
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

    def get_active_detectors(
        self,
    ) -> list[str]:

        with self._lock:

            return list(
                self._detectors.keys()
            )


# ============================================================================
# DEFAULT DETECTOR
# ============================================================================


_default_detector: Optional[
    SwipeDetector
] = None

_default_detector_lock = threading.RLock()


def get_swipe_detector() -> SwipeDetector:

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                SwipeDetector()
            )

        return _default_detector


def detect_swipe(
    x: float,
    y: float,
    *,
    z: float = 0.0,
    confidence: float = 1.0,
    timestamp: Optional[float] = None,
) -> SwipeEvent:

    return get_swipe_detector().update(
        x,
        y,
        z=z,
        confidence=confidence,
        timestamp=timestamp,
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "SwipeDirection",
    "SwipePoint",
    "SwipeConfig",
    "SwipeEvent",
    "SwipeDetector",
    "MultiHandSwipeDetector",
    "get_swipe_detector",
    "detect_swipe",
]


