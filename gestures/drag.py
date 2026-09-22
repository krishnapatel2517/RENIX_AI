"""
RENIX - Drag Gesture Detector
=============================

Detects a natural hand drag gesture for RENIX.

Typical behavior:
    1. User pinches thumb + index finger.
    2. Pinch becomes active -> drag starts.
    3. User moves the pinched hand -> drag continues.
    4. User releases the pinch -> drag ends.

This module ONLY detects and reports drag state.
It does NOT move the mouse or manipulate the desktop directly.

Expected input:
    MediaPipe-style 21 hand landmarks.

Important landmarks:
    0  = wrist
    4  = thumb tip
    8  = index finger tip
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence


logger = logging.getLogger("RENIX.gestures.drag")


# ============================================================================
# ENUMS
# ============================================================================


class DragState(str, Enum):
    IDLE = "idle"
    POSSIBLE = "possible"
    STARTED = "started"
    DRAGGING = "dragging"
    ENDED = "ended"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(frozen=True)
class DragPoint:
    """3D normalized point."""

    x: float
    y: float
    z: float = 0.0

    def distance_to(
        self,
        other: "DragPoint",
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
class DragConfig:
    """Configuration for drag detection."""

    pinch_threshold: float = 0.055

    release_threshold: float = 0.075

    activation_frames: int = 2

    release_frames: int = 2

    minimum_confidence: float = 0.45

    movement_threshold: float = 0.004

    smoothing: float = 0.40

    max_frame_time: float = 0.30

    enabled: bool = True


@dataclass
class DragEvent:
    """Event returned by the drag detector."""

    state: DragState = DragState.IDLE

    active: bool = False

    started: bool = False

    ended: bool = False

    x: float = 0.0

    y: float = 0.0

    z: float = 0.0

    delta_x: float = 0.0

    delta_y: float = 0.0

    delta_z: float = 0.0

    distance: float = 0.0

    velocity_x: float = 0.0

    velocity_y: float = 0.0

    speed: float = 0.0

    confidence: float = 0.0

    duration: float = 0.0

    timestamp: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "active": self.active,
            "started": self.started,
            "ended": self.ended,
            "position": {
                "x": self.x,
                "y": self.y,
                "z": self.z,
            },
            "delta": {
                "x": self.delta_x,
                "y": self.delta_y,
                "z": self.delta_z,
            },
            "pinch_distance": self.distance,
            "velocity": {
                "x": self.velocity_x,
                "y": self.velocity_y,
            },
            "speed": self.speed,
            "confidence": self.confidence,
            "duration": self.duration,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# DRAG DETECTOR
# ============================================================================


class DragDetector:
    """
    Detects thumb-index pinch dragging.

    The detector is stateful.

    State flow:

        IDLE
          ↓
        POSSIBLE
          ↓
        STARTED
          ↓
        DRAGGING
          ↓
        ENDED
          ↓
        IDLE
    """

    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(
        self,
        config: Optional[DragConfig] = None,
    ) -> None:

        self.config = (
            config
            or DragConfig()
        )

        self._lock = threading.RLock()

        self._state = DragState.IDLE

        self._position: Optional[
            DragPoint
        ] = None

        self._smoothed_position: Optional[
            DragPoint
        ] = None

        self._previous_position: Optional[
            DragPoint
        ] = None

        self._start_position: Optional[
            DragPoint
        ] = None

        self._last_timestamp = 0.0

        self._start_timestamp: Optional[
            float
        ] = None

        self._last_event = DragEvent()

        self._pinch_frames = 0

        self._release_frames = 0

        self._last_confidence = 0.0

        self._callbacks: list[
            Callable[[DragEvent], None]
        ] = []

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        landmarks: Sequence[
            DragPoint | Sequence[float]
        ],
        *,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> DragEvent:

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        confidence = self._clamp(
            confidence,
            0.0,
            1.0,
        )

        with self._lock:

            if not self.config.enabled:

                self.reset()

                return DragEvent(
                    timestamp=now
                )

            if (
                confidence
                < self.config.minimum_confidence
            ):

                return self._handle_tracking_loss(
                    now,
                    confidence,
                )

            points = self._normalize_landmarks(
                landmarks
            )

            if points is None:

                return self._handle_tracking_loss(
                    now,
                    confidence,
                )

            thumb = points[
                self.THUMB_TIP
            ]

            index = points[
                self.INDEX_TIP
            ]

            pinch_distance = (
                thumb.distance_to(index)
            )

            # Pinch midpoint is used as the drag cursor.
            raw_position = DragPoint(
                x=(thumb.x + index.x) / 2.0,
                y=(thumb.y + index.y) / 2.0,
                z=(thumb.z + index.z) / 2.0,
            )

            # --------------------------------------------------------------
            # POSITION SMOOTHING
            # --------------------------------------------------------------

            if self._smoothed_position is None:

                self._smoothed_position = (
                    raw_position
                )

            else:

                self._smoothed_position = (
                    self._lerp(
                        self._smoothed_position,
                        raw_position,
                        self.config.smoothing,
                    )
                )

            current = self._smoothed_position

            previous = (
                self._previous_position
            )

            delta_x = 0.0
            delta_y = 0.0
            delta_z = 0.0

            dt = max(
                now - self._last_timestamp,
                1e-6,
            )

            if previous is not None:

                delta_x = (
                    current.x
                    - previous.x
                )

                delta_y = (
                    current.y
                    - previous.y
                )

                delta_z = (
                    current.z
                    - previous.z
                )

            velocity_x = (
                delta_x / dt
            )

            velocity_y = (
                delta_y / dt
            )

            speed = math.sqrt(
                velocity_x ** 2
                + velocity_y ** 2
            )

            # --------------------------------------------------------------
            # FRAME TIMING
            # --------------------------------------------------------------

            valid_timing = (
                self._last_timestamp == 0.0
                or dt <= self.config.max_frame_time
            )

            # --------------------------------------------------------------
            # PINCH STATE
            # --------------------------------------------------------------

            pinch_active = (
                pinch_distance
                <= self.config.pinch_threshold
            )

            release_detected = (
                pinch_distance
                >= self.config.release_threshold
            )

            started = False
            ended = False

            # --------------------------------------------------------------
            # STATE MACHINE
            # --------------------------------------------------------------

            if self._state == DragState.IDLE:

                if pinch_active:

                    self._pinch_frames += 1

                    self._release_frames = 0

                    self._state = (
                        DragState.POSSIBLE
                    )

                else:

                    self._pinch_frames = 0

            elif self._state == DragState.POSSIBLE:

                if pinch_active:

                    self._pinch_frames += 1

                    if (
                        self._pinch_frames
                        >= max(
                            1,
                            self.config.activation_frames,
                        )
                    ):

                        self._state = (
                            DragState.STARTED
                        )

                        self._start_position = (
                            current
                        )

                        self._start_timestamp = (
                            now
                        )

                        started = True

                else:

                    self._pinch_frames = 0

                    self._state = (
                        DragState.IDLE
                    )

            elif self._state in {
                DragState.STARTED,
                DragState.DRAGGING,
            }:

                if release_detected:

                    self._release_frames += 1

                    if (
                        self._release_frames
                        >= max(
                            1,
                            self.config.release_frames,
                        )
                    ):

                        self._state = (
                            DragState.ENDED
                        )

                        ended = True

                else:

                    self._release_frames = 0

                    if self._state == DragState.STARTED:

                        movement = 0.0

                        if (
                            self._start_position
                            is not None
                        ):

                            movement = (
                                current.distance_to(
                                    self._start_position
                                )
                            )

                        if (
                            movement
                            >= self.config.movement_threshold
                        ):

                            self._state = (
                                DragState.DRAGGING
                            )

            elif self._state == DragState.ENDED:

                self._state = (
                    DragState.IDLE
                )

                self._pinch_frames = 0

                self._release_frames = 0

            # --------------------------------------------------------------
            # DURATION
            # --------------------------------------------------------------

            duration = 0.0

            if (
                self._start_timestamp
                is not None
            ):

                duration = max(
                    0.0,
                    now
                    - self._start_timestamp,
                )

            # --------------------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------------------

            pinch_confidence = (
                self._calculate_pinch_confidence(
                    pinch_distance
                )
            )

            final_confidence = (
                confidence
                * pinch_confidence
                if self._state
                in {
                    DragState.POSSIBLE,
                    DragState.STARTED,
                    DragState.DRAGGING,
                }
                else confidence
            )

            final_confidence = self._clamp(
                final_confidence,
                0.0,
                1.0,
            )

            # --------------------------------------------------------------
            # EVENT
            # --------------------------------------------------------------

            active = (
                self._state
                in {
                    DragState.STARTED,
                    DragState.DRAGGING,
                }
            )

            event = DragEvent(
                state=self._state,
                active=active,
                started=started,
                ended=ended,
                x=current.x,
                y=current.y,
                z=current.z,
                delta_x=delta_x,
                delta_y=delta_y,
                delta_z=delta_z,
                distance=pinch_distance,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                speed=speed,
                confidence=final_confidence,
                duration=duration,
                timestamp=now,
                metadata={
                    "pinch_active": pinch_active,
                    "release_detected": (
                        release_detected
                    ),
                    "valid_timing": valid_timing,
                    "pinch_frames": (
                        self._pinch_frames
                    ),
                    "release_frames": (
                        self._release_frames
                    ),
                },
            )

            self._last_event = event

            self._position = current

            self._previous_position = current

            self._last_timestamp = now

            self._last_confidence = (
                final_confidence
            )

            if started or active or ended:

                self._emit(event)

            # After reporting ENDED, transition to IDLE
            # on the following frame.
            if ended:

                self._state = (
                    DragState.ENDED
                )

            return event

    # ========================================================================
    # TRACKING LOSS
    # ========================================================================

    def _handle_tracking_loss(
        self,
        timestamp: float,
        confidence: float,
    ) -> DragEvent:

        if self._state in {
            DragState.STARTED,
            DragState.DRAGGING,
            DragState.POSSIBLE,
        }:

            self._state = (
                DragState.ENDED
            )

            event = DragEvent(
                state=DragState.ENDED,
                active=False,
                started=False,
                ended=True,
                x=(
                    self._position.x
                    if self._position
                    else 0.0
                ),
                y=(
                    self._position.y
                    if self._position
                    else 0.0
                ),
                z=(
                    self._position.z
                    if self._position
                    else 0.0
                ),
                confidence=confidence,
                timestamp=timestamp,
                metadata={
                    "reason": (
                        "tracking_lost"
                    )
                },
            )

            self._last_event = event

            self._emit(event)

            return event

        return DragEvent(
            state=DragState.IDLE,
            active=False,
            confidence=confidence,
            timestamp=timestamp,
            metadata={
                "reason": (
                    "tracking_lost"
                )
            },
        )

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _calculate_pinch_confidence(
        self,
        distance: float,
    ) -> float:

        threshold = max(
            self.config.pinch_threshold,
            1e-6,
        )

        if distance <= threshold:

            return 1.0

        release = max(
            self.config.release_threshold,
            threshold + 1e-6,
        )

        if distance >= release:

            return 0.0

        ratio = (
            release - distance
        ) / (
            release - threshold
        )

        return self._clamp(
            ratio,
            0.0,
            1.0,
        )

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def subscribe(
        self,
        callback: Callable[
            [DragEvent],
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
            [DragEvent],
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
        event: DragEvent,
    ) -> None:

        callbacks = list(
            self._callbacks
        )

        for callback in callbacks:

            try:

                callback(event)

            except Exception:

                logger.exception(
                    "Drag callback failed"
                )

    # ========================================================================
    # STATE API
    # ========================================================================

    def get_state(self) -> DragState:

        with self._lock:

            return self._state

    def is_active(self) -> bool:

        with self._lock:

            return self._state in {
                DragState.STARTED,
                DragState.DRAGGING,
            }

    def is_dragging(self) -> bool:

        with self._lock:

            return (
                self._state
                == DragState.DRAGGING
            )

    def get_position(self) -> DragPoint:

        with self._lock:

            return (
                self._position
                or DragPoint(
                    0.0,
                    0.0,
                    0.0,
                )
            )

    def get_velocity(self) -> DragPoint:

        with self._lock:

            return DragPoint(
                self._last_event.velocity_x,
                self._last_event.velocity_y,
                0.0,
            )

    def get_speed(self) -> float:

        with self._lock:

            return self._last_event.speed

    def get_duration(self) -> float:

        with self._lock:

            return self._last_event.duration

    def get_last_event(self) -> DragEvent:

        with self._lock:

            return self._last_event

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

    def set_pinch_threshold(
        self,
        threshold: float,
    ) -> None:

        with self._lock:

            self.config.pinch_threshold = max(
                0.001,
                float(threshold),
            )

            if (
                self.config.release_threshold
                <= self.config.pinch_threshold
            ):

                self.config.release_threshold = (
                    self.config.pinch_threshold
                    * 1.35
                )

    def set_release_threshold(
        self,
        threshold: float,
    ) -> None:

        with self._lock:

            self.config.release_threshold = max(
                self.config.pinch_threshold
                + 0.001,
                float(threshold),
            )

    def get_config(self) -> dict[str, Any]:

        with self._lock:

            return {
                "pinch_threshold": (
                    self.config.pinch_threshold
                ),
                "release_threshold": (
                    self.config.release_threshold
                ),
                "activation_frames": (
                    self.config.activation_frames
                ),
                "release_frames": (
                    self.config.release_frames
                ),
                "minimum_confidence": (
                    self.config.minimum_confidence
                ),
                "movement_threshold": (
                    self.config.movement_threshold
                ),
                "smoothing": (
                    self.config.smoothing
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

            self._state = DragState.IDLE

            self._position = None

            self._smoothed_position = None

            self._previous_position = None

            self._start_position = None

            self._last_timestamp = 0.0

            self._start_timestamp = None

            self._last_event = DragEvent()

            self._pinch_frames = 0

            self._release_frames = 0

            self._last_confidence = 0.0

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _normalize_landmarks(
        landmarks: Sequence[
            DragPoint | Sequence[float]
        ],
    ) -> Optional[list[DragPoint]]:

        if len(landmarks) < 21:

            return None

        result: list[
            DragPoint
        ] = []

        try:

            for landmark in landmarks:

                if isinstance(
                    landmark,
                    DragPoint,
                ):

                    result.append(
                        landmark
                    )

                else:

                    if len(landmark) < 2:

                        return None

                    result.append(
                        DragPoint(
                            x=float(
                                landmark[0]
                            ),
                            y=float(
                                landmark[1]
                            ),
                            z=(
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

            logger.exception(
                "Invalid drag landmarks"
            )

            return None

        return result

    @staticmethod
    def _lerp(
        a: DragPoint,
        b: DragPoint,
        amount: float,
    ) -> DragPoint:

        amount = max(
            0.0,
            min(
                1.0,
                amount,
            ),
        )

        return DragPoint(
            x=a.x
            + (
                b.x - a.x
            ) * amount,

            y=a.y
            + (
                b.y - a.y
            ) * amount,

            z=a.z
            + (
                b.z - a.z
            ) * amount,
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
    DragDetector
] = None

_default_lock = threading.RLock()


def get_drag_detector() -> DragDetector:

    global _default_detector

    with _default_lock:

        if _default_detector is None:

            _default_detector = (
                DragDetector()
            )

        return _default_detector


def detect_drag(
    landmarks: Sequence[
        DragPoint | Sequence[float]
    ],
    *,
    confidence: float = 1.0,
    timestamp: Optional[float] = None,
) -> DragEvent:

    return get_drag_detector().update(
        landmarks,
        confidence=confidence,
        timestamp=timestamp,
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "DragPoint",
    "DragState",
    "DragConfig",
    "DragEvent",
    "DragDetector",
    "get_drag_detector",
    "detect_drag",
]


