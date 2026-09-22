"""
RENIX - Two-Hand Gesture Detection
===================================

Detects relationships and gestures involving two hands.

Supported detections:
    - two hands present
    - hands approaching
    - hands separating
    - hands moving apart
    - hands moving together
    - two-hand pinch
    - two-hand spread
    - two-hand zoom
    - two-hand rotation
    - left/right hand relationship
    - distance and distance-change tracking

This module ONLY detects two-hand interaction.
It does not directly control the desktop or UI.

Expected input:
    Two MediaPipe-style 21-landmark hand sets.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence


# ============================================================================
# ENUMS
# ============================================================================


class TwoHandGesture(str, Enum):
    NONE = "none"
    TWO_HANDS = "two_hands"
    PINCH = "two_hand_pinch"
    SPREAD = "two_hand_spread"
    MOVE_TOGETHER = "move_together"
    MOVE_APART = "move_apart"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ROTATE_CW = "rotate_clockwise"
    ROTATE_CCW = "rotate_counterclockwise"


class HandSide(str, Enum):
    UNKNOWN = "unknown"
    LEFT = "left"
    RIGHT = "right"


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass(frozen=True)
class TwoHandPoint:
    x: float
    y: float
    z: float = 0.0

    def distance_to(
        self,
        other: "TwoHandPoint",
    ) -> float:

        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z

        return math.sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        )


@dataclass
class TwoHandConfig:

    minimum_confidence: float = 0.40

    pinch_threshold: float = 0.055

    spread_threshold: float = 0.22

    zoom_threshold: float = 0.018

    rotation_threshold: float = 7.0

    smoothing: float = 0.35

    minimum_frames: int = 2

    enabled: bool = True


@dataclass
class TwoHandEvent:

    detected: bool = False

    gesture: TwoHandGesture = (
        TwoHandGesture.NONE
    )

    left_present: bool = False

    right_present: bool = False

    left_x: float = 0.0
    left_y: float = 0.0
    left_z: float = 0.0

    right_x: float = 0.0
    right_y: float = 0.0
    right_z: float = 0.0

    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0

    hand_distance: float = 0.0

    previous_distance: float = 0.0

    distance_delta: float = 0.0

    rotation: float = 0.0

    confidence: float = 0.0

    timestamp: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "detected": self.detected,
            "gesture": self.gesture.value,
            "hands": {
                "left": {
                    "present": self.left_present,
                    "x": self.left_x,
                    "y": self.left_y,
                    "z": self.left_z,
                },
                "right": {
                    "present": self.right_present,
                    "x": self.right_x,
                    "y": self.right_y,
                    "z": self.right_z,
                },
            },
            "center": {
                "x": self.center_x,
                "y": self.center_y,
                "z": self.center_z,
            },
            "hand_distance": self.hand_distance,
            "previous_distance": self.previous_distance,
            "distance_delta": self.distance_delta,
            "rotation": self.rotation,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# TWO-HAND DETECTOR
# ============================================================================


class TwoHandDetector:

    WRIST = 0

    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(
        self,
        config: Optional[TwoHandConfig] = None,
    ) -> None:

        self.config = (
            config
            or TwoHandConfig()
        )

        self._lock = threading.RLock()

        self._left_center: Optional[
            TwoHandPoint
        ] = None

        self._right_center: Optional[
            TwoHandPoint
        ] = None

        self._previous_left: Optional[
            TwoHandPoint
        ] = None

        self._previous_right: Optional[
            TwoHandPoint
        ] = None

        self._smoothed_left: Optional[
            TwoHandPoint
        ] = None

        self._smoothed_right: Optional[
            TwoHandPoint
        ] = None

        self._previous_distance = 0.0

        self._current_distance = 0.0

        self._previous_angle = 0.0

        self._current_angle = 0.0

        self._frame_count = 0

        self._last_timestamp = 0.0

        self._last_event = TwoHandEvent()

        self._callbacks: list[
            Callable[[TwoHandEvent], None]
        ] = []

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        left_landmarks: Optional[
            Sequence[
                TwoHandPoint | Sequence[float]
            ]
        ],
        right_landmarks: Optional[
            Sequence[
                TwoHandPoint | Sequence[float]
            ]
        ],
        *,
        left_confidence: float = 1.0,
        right_confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> TwoHandEvent:

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        left_confidence = self._clamp(
            left_confidence,
            0.0,
            1.0,
        )

        right_confidence = self._clamp(
            right_confidence,
            0.0,
            1.0,
        )

        with self._lock:

            if not self.config.enabled:

                self.reset()

                return TwoHandEvent(
                    timestamp=now
                )

            left = self._get_center(
                left_landmarks
            )

            right = self._get_center(
                right_landmarks
            )

            left_valid = (
                left is not None
                and left_confidence
                >= self.config.minimum_confidence
            )

            right_valid = (
                right is not None
                and right_confidence
                >= self.config.minimum_confidence
            )

            # --------------------------------------------------------------
            # NO TWO HANDS
            # --------------------------------------------------------------

            if not left_valid or not right_valid:

                self._reset_pair_state()

                return TwoHandEvent(
                    detected=False,
                    gesture=TwoHandGesture.NONE,
                    left_present=left_valid,
                    right_present=right_valid,
                    confidence=max(
                        left_confidence,
                        right_confidence,
                    ),
                    timestamp=now,
                )

            # --------------------------------------------------------------
            # SMOOTH CENTERS
            # --------------------------------------------------------------

            if self._smoothed_left is None:

                self._smoothed_left = left

            else:

                self._smoothed_left = self._lerp(
                    self._smoothed_left,
                    left,
                    self.config.smoothing,
                )

            if self._smoothed_right is None:

                self._smoothed_right = right

            else:

                self._smoothed_right = self._lerp(
                    self._smoothed_right,
                    right,
                    self.config.smoothing,
                )

            left = self._smoothed_left
            right = self._smoothed_right

            self._left_center = left
            self._right_center = right

            # --------------------------------------------------------------
            # CENTER
            # --------------------------------------------------------------

            center = TwoHandPoint(
                x=(left.x + right.x) / 2.0,
                y=(left.y + right.y) / 2.0,
                z=(left.z + right.z) / 2.0,
            )

            # --------------------------------------------------------------
            # DISTANCE
            # --------------------------------------------------------------

            distance = left.distance_to(right)

            previous_distance = (
                self._current_distance
            )

            distance_delta = (
                distance
                - previous_distance
            )

            self._previous_distance = (
                previous_distance
            )

            self._current_distance = distance

            # --------------------------------------------------------------
            # ANGLE
            # --------------------------------------------------------------

            angle = math.degrees(
                math.atan2(
                    right.y - left.y,
                    right.x - left.x,
                )
            )

            previous_angle = (
                self._current_angle
            )

            angle_delta = (
                self._angle_difference(
                    angle,
                    previous_angle,
                )
            )

            self._previous_angle = (
                previous_angle
            )

            self._current_angle = angle

            # --------------------------------------------------------------
            # GESTURE DETECTION
            # --------------------------------------------------------------

            gesture = (
                TwoHandGesture.TWO_HANDS
            )

            detected = False

            # Distance decreasing = hands moving together.
            if (
                distance_delta
                < -self.config.zoom_threshold
            ):

                gesture = (
                    TwoHandGesture.MOVE_TOGETHER
                )

                detected = True

            # Distance increasing = hands moving apart.
            elif (
                distance_delta
                > self.config.zoom_threshold
            ):

                gesture = (
                    TwoHandGesture.MOVE_APART
                )

                detected = True

            # Close hands.
            elif (
                distance
                <= self.config.pinch_threshold
            ):

                gesture = (
                    TwoHandGesture.PINCH
                )

                detected = True

            # Wide hands.
            elif (
                distance
                >= self.config.spread_threshold
            ):

                gesture = (
                    TwoHandGesture.SPREAD
                )

                detected = True

            # Rotation.
            if abs(angle_delta) >= (
                self.config.rotation_threshold
            ):

                if angle_delta > 0:

                    gesture = (
                        TwoHandGesture.ROTATE_CCW
                    )

                else:

                    gesture = (
                        TwoHandGesture.ROTATE_CW
                    )

                detected = True

            # --------------------------------------------------------------
            # ZOOM
            # --------------------------------------------------------------

            if (
                distance_delta
                > self.config.zoom_threshold
            ):

                gesture = (
                    TwoHandGesture.ZOOM_OUT
                )

                detected = True

            elif (
                distance_delta
                < -self.config.zoom_threshold
            ):

                gesture = (
                    TwoHandGesture.ZOOM_IN
                )

                detected = True

            # --------------------------------------------------------------
            # FRAME COUNT
            # --------------------------------------------------------------

            self._frame_count += 1

            if (
                self._frame_count
                < self.config.minimum_frames
            ):

                detected = False

            confidence = min(
                left_confidence,
                right_confidence,
            )

            event = TwoHandEvent(
                detected=detected,
                gesture=gesture,
                left_present=True,
                right_present=True,
                left_x=left.x,
                left_y=left.y,
                left_z=left.z,
                right_x=right.x,
                right_y=right.y,
                right_z=right.z,
                center_x=center.x,
                center_y=center.y,
                center_z=center.z,
                hand_distance=distance,
                previous_distance=previous_distance,
                distance_delta=distance_delta,
                rotation=angle_delta,
                confidence=confidence,
                timestamp=now,
                metadata={
                    "left_confidence": (
                        left_confidence
                    ),
                    "right_confidence": (
                        right_confidence
                    ),
                    "angle": angle,
                    "angle_delta": angle_delta,
                    "frame_count": (
                        self._frame_count
                    ),
                },
            )

            self._last_event = event

            self._previous_left = left
            self._previous_right = right

            self._last_timestamp = now

            if detected:

                self._emit(event)

            return event

    # ========================================================================
    # CENTER EXTRACTION
    # ========================================================================

    @staticmethod
    def _get_center(
        landmarks: Optional[
            Sequence[
                TwoHandPoint | Sequence[float]
            ]
        ],
    ) -> Optional[TwoHandPoint]:

        if landmarks is None:
            return None

        if len(landmarks) < 21:
            return None

        try:

            points = (
                TwoHandDetector._normalize(
                    landmarks
                )
            )

            if points is None:
                return None

            # Palm center based on wrist + MCP joints.
            selected = (
                points[0],
                points[5],
                points[9],
                points[13],
                points[17],
            )

            return TwoHandPoint(
                x=sum(
                    p.x
                    for p in selected
                ) / len(selected),

                y=sum(
                    p.y
                    for p in selected
                ) / len(selected),

                z=sum(
                    p.z
                    for p in selected
                ) / len(selected),
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

    # ========================================================================
    # HAND RELATIONSHIP
    # ========================================================================

    def get_hand_distance(self) -> float:

        with self._lock:

            return self._current_distance

    def get_distance_delta(self) -> float:

        with self._lock:

            return (
                self._current_distance
                - self._previous_distance
            )

    def get_rotation(self) -> float:

        with self._lock:

            return self._current_angle

    def get_last_event(self) -> TwoHandEvent:

        with self._lock:

            return self._last_event

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def subscribe(
        self,
        callback: Callable[
            [TwoHandEvent],
            None,
        ],
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
        callback: Callable[
            [TwoHandEvent],
            None,
        ],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: TwoHandEvent,
    ) -> None:

        for callback in list(
            self._callbacks
        ):

            try:

                callback(event)

            except Exception:

                # Gesture callbacks must never
                # break the tracking pipeline.
                pass

    # ========================================================================
    # RESET
    # ========================================================================

    def _reset_pair_state(self) -> None:

        self._left_center = None
        self._right_center = None

        self._previous_left = None
        self._previous_right = None

        self._smoothed_left = None
        self._smoothed_right = None

        self._previous_distance = 0.0
        self._current_distance = 0.0

        self._previous_angle = 0.0
        self._current_angle = 0.0

        self._frame_count = 0

    def reset(self) -> None:

        with self._lock:

            self._reset_pair_state()

            self._last_timestamp = 0.0

            self._last_event = (
                TwoHandEvent()
            )

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _normalize(
        landmarks: Sequence[
            TwoHandPoint | Sequence[float]
        ],
    ) -> Optional[list[TwoHandPoint]]:

        result: list[
            TwoHandPoint
        ] = []

        try:

            for landmark in landmarks:

                if isinstance(
                    landmark,
                    TwoHandPoint,
                ):

                    result.append(
                        landmark
                    )

                else:

                    if len(landmark) < 2:
                        return None

                    result.append(
                        TwoHandPoint(
                            float(
                                landmark[0]
                            ),
                            float(
                                landmark[1]
                            ),
                            (
                                float(
                                    landmark[2]
                                )
                                if len(landmark) >= 3
                                else 0.0
                            ),
                        )
                    )

        except (
            TypeError,
            ValueError,
        ):

            return None

        return result

    @staticmethod
    def _lerp(
        a: TwoHandPoint,
        b: TwoHandPoint,
        amount: float,
    ) -> TwoHandPoint:

        amount = max(
            0.0,
            min(
                1.0,
                amount,
            ),
        )

        return TwoHandPoint(
            a.x + (
                b.x - a.x
            ) * amount,

            a.y + (
                b.y - a.y
            ) * amount,

            a.z + (
                b.z - a.z
            ) * amount,
        )

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
    TwoHandDetector
] = None

_default_lock = threading.RLock()


def get_two_hand_detector() -> TwoHandDetector:

    global _default_detector

    with _default_lock:

        if _default_detector is None:

            _default_detector = (
                TwoHandDetector()
            )

        return _default_detector


def detect_two_hand(
    left_landmarks: Optional[
        Sequence[
            TwoHandPoint | Sequence[float]
        ]
    ],
    right_landmarks: Optional[
        Sequence[
            TwoHandPoint | Sequence[float]
        ]
    ],
    *,
    left_confidence: float = 1.0,
    right_confidence: float = 1.0,
    timestamp: Optional[float] = None,
) -> TwoHandEvent:

    return get_two_hand_detector().update(
        left_landmarks,
        right_landmarks,
        left_confidence=left_confidence,
        right_confidence=right_confidence,
        timestamp=timestamp,
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "TwoHandPoint",
    "TwoHandConfig",
    "TwoHandGesture",
    "HandSide",
    "TwoHandEvent",
    "TwoHandDetector",
    "get_two_hand_detector",
    "detect_two_hand",
]


