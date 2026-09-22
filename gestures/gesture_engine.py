"""
RENIX Gesture Engine
====================

Central orchestration layer for gesture recognition.

Pipeline:

    Vision / Hand Landmarks
            ↓
       Hand Tracker
            ↓
     Gesture Classifier
            ↓
       Gesture Engine
            ↓
      Gesture Event
            ↓
       Gesture Mapper
            ↓
    RENIX UI / Computer / Automation

The engine is intentionally independent from the holographic UI and
computer-control layers. It recognizes gestures and emits structured
events; consumers decide what those events should do.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional


logger = logging.getLogger("RENIX.gestures.engine")


# ============================================================================
# ENUMS
# ============================================================================


class GestureState(str, Enum):
    """Lifecycle state of a gesture."""

    IDLE = "idle"
    STARTED = "started"
    ACTIVE = "active"
    UPDATED = "updated"
    ENDED = "ended"
    CANCELLED = "cancelled"


class HandSide(str, Enum):
    """Detected hand side."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class GesturePoint:
    """Normalized 2D/3D point."""

    x: float
    y: float
    z: float = 0.0

    def distance_to(
        self,
        other: "GesturePoint",
    ) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def midpoint(
        self,
        other: "GesturePoint",
    ) -> "GesturePoint":
        return GesturePoint(
            x=(self.x + other.x) / 2.0,
            y=(self.y + other.y) / 2.0,
            z=(self.z + other.z) / 2.0,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass
class HandState:
    """Current state of one detected hand."""

    hand_id: str

    side: HandSide = HandSide.UNKNOWN

    confidence: float = 0.0

    landmarks: list[Any] = field(
        default_factory=list
    )

    palm: Optional[GesturePoint] = None

    wrist: Optional[GesturePoint] = None

    thumb_tip: Optional[GesturePoint] = None

    index_tip: Optional[GesturePoint] = None

    middle_tip: Optional[GesturePoint] = None

    ring_tip: Optional[GesturePoint] = None

    pinky_tip: Optional[GesturePoint] = None

    is_open: bool = False

    is_closed: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "side": self.side.value,
            "confidence": self.confidence,
            "landmarks": self.landmarks,
            "palm": (
                self.palm.to_dict()
                if self.palm
                else None
            ),
            "wrist": (
                self.wrist.to_dict()
                if self.wrist
                else None
            ),
            "thumb_tip": (
                self.thumb_tip.to_dict()
                if self.thumb_tip
                else None
            ),
            "index_tip": (
                self.index_tip.to_dict()
                if self.index_tip
                else None
            ),
            "middle_tip": (
                self.middle_tip.to_dict()
                if self.middle_tip
                else None
            ),
            "ring_tip": (
                self.ring_tip.to_dict()
                if self.ring_tip
                else None
            ),
            "pinky_tip": (
                self.pinky_tip.to_dict()
                if self.pinky_tip
                else None
            ),
            "is_open": self.is_open,
            "is_closed": self.is_closed,
            "metadata": dict(self.metadata),
        }


@dataclass
class GestureEvent:
    """Structured gesture event emitted by RENIX."""

    name: str

    state: GestureState

    timestamp: float

    confidence: float = 1.0

    hand: HandSide = HandSide.UNKNOWN

    hand_id: Optional[str] = None

    position: Optional[GesturePoint] = None

    delta: Optional[GesturePoint] = None

    scale: Optional[float] = None

    rotation: Optional[float] = None

    velocity: Optional[GesturePoint] = None

    duration: float = 0.0

    frame_number: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "hand": self.hand.value,
            "hand_id": self.hand_id,
            "position": (
                self.position.to_dict()
                if self.position
                else None
            ),
            "delta": (
                self.delta.to_dict()
                if self.delta
                else None
            ),
            "scale": self.scale,
            "rotation": self.rotation,
            "velocity": (
                self.velocity.to_dict()
                if self.velocity
                else None
            ),
            "duration": self.duration,
            "frame_number": self.frame_number,
            "metadata": dict(self.metadata),
        }


@dataclass
class GestureEngineConfig:
    """Runtime configuration."""

    enabled: bool = True

    confidence_threshold: float = 0.50

    gesture_timeout: float = 1.0

    smoothing_window: int = 5

    max_history: int = 120

    position_threshold: float = 0.015

    swipe_distance_threshold: float = 0.12

    swipe_velocity_threshold: float = 0.45

    pinch_threshold: float = 0.055

    pinch_release_threshold: float = 0.075

    grab_threshold: float = 0.12

    zoom_threshold: float = 0.025

    rotation_threshold: float = 0.08

    emit_updates: bool = True

    max_hands: int = 2


# ============================================================================
# GESTURE ENGINE
# ============================================================================


class GestureEngine:
    """
    RENIX central gesture recognition engine.

    It accepts hand information from the vision subsystem and produces
    high-level GestureEvent objects.

    The engine supports:
        - Pinch
        - Grab
        - Point
        - Palm
        - Swipe
        - Drag
        - Zoom
        - Rotate
        - Two-hand interaction
    """

    def __init__(
        self,
        config: Optional[
            GestureEngineConfig
        ] = None,
        *,
        classifier: Any = None,
        mapper: Any = None,
    ) -> None:

        self.config = (
            config
            or GestureEngineConfig()
        )

        self.classifier = classifier

        self.mapper = mapper

        self._lock = threading.RLock()

        self._running = False

        self._frame_number = 0

        self._hands: dict[
            str,
            HandState,
        ] = {}

        self._previous_hands: dict[
            str,
            HandState,
        ] = {}

        self._gesture_states: dict[
            tuple[str, str],
            dict[str, Any],
        ] = {}

        self._position_history: dict[
            str,
            deque[GesturePoint],
        ] = {}

        self._event_history: deque[
            GestureEvent
        ] = deque(
            maxlen=self.config.max_history
        )

        self._callbacks: list[
            Callable[
                [GestureEvent],
                None,
            ]
        ] = []

        self._last_update_time = (
            time.monotonic()
        )

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            self._running = True

            self._last_update_time = (
                time.monotonic()
            )

    def stop(self) -> None:

        with self._lock:

            self._running = False

            self._cancel_all_gestures()

    def reset(self) -> None:

        with self._lock:

            self._hands.clear()

            self._previous_hands.clear()

            self._gesture_states.clear()

            self._position_history.clear()

            self._event_history.clear()

            self._frame_number = 0

    def is_running(self) -> bool:

        with self._lock:

            return self._running

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def add_callback(
        self,
        callback: Callable[
            [GestureEvent],
            None,
        ],
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "Gesture callback must be callable."
            )

        with self._lock:

            if callback not in self._callbacks:

                self._callbacks.append(
                    callback
                )

    def remove_callback(
        self,
        callback: Callable[
            [GestureEvent],
            None,
        ],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    # ========================================================================
    # MAIN UPDATE
    # ========================================================================

    def update(
        self,
        hands: Any,
        *,
        timestamp: Optional[float] = None,
        frame_number: Optional[int] = None,
    ) -> list[GestureEvent]:

        if not self.config.enabled:

            return []

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        with self._lock:

            if frame_number is None:

                self._frame_number += 1

            else:

                self._frame_number = (
                    frame_number
                )

            current_frame = (
                self._frame_number
            )

            normalized = (
                self._normalize_hands(
                    hands
                )
            )

            self._previous_hands = dict(
                self._hands
            )

            self._hands = {
                hand.hand_id: hand
                for hand in normalized
            }

        events: list[
            GestureEvent
        ] = []

        # ------------------------------------------------------------
        # Process each hand.
        # ------------------------------------------------------------

        for hand in normalized:

            events.extend(
                self._process_hand(
                    hand,
                    now,
                    current_frame,
                )
            )

        # ------------------------------------------------------------
        # Detect hands that disappeared.
        # ------------------------------------------------------------

        previous_ids = set(
            self._previous_hands.keys()
        )

        current_ids = set(
            self._hands.keys()
        )

        disappeared = (
            previous_ids - current_ids
        )

        for hand_id in disappeared:

            events.extend(
                self._end_hand_gestures(
                    hand_id,
                    now,
                    current_frame,
                )
            )

        # ------------------------------------------------------------
        # Two-hand gestures.
        # ------------------------------------------------------------

        if len(normalized) >= 2:

            events.extend(
                self._process_two_hand(
                    normalized[:2],
                    now,
                    current_frame,
                )
            )

        # ------------------------------------------------------------
        # Publish events.
        # ------------------------------------------------------------

        for event in events:

            self._record_event(event)

            self._publish(event)

        with self._lock:

            self._last_update_time = (
                time.monotonic()
            )

        return events

    # ========================================================================
    # HAND NORMALIZATION
    # ========================================================================

    def _normalize_hands(
        self,
        hands: Any,
    ) -> list[HandState]:

        if hands is None:

            return []

        if isinstance(
            hands,
            dict,
        ):

            if "hands" in hands:

                hands = hands["hands"]

            else:

                hands = [hands]

        if not isinstance(
            hands,
            (list, tuple),
        ):

            hands = [hands]

        normalized: list[
            HandState
        ] = []

        for index, raw in enumerate(
            hands
        ):

            try:

                hand = self._normalize_hand(
                    raw,
                    index,
                )

                if (
                    hand.confidence
                    >= self.config.confidence_threshold
                ):

                    normalized.append(
                        hand
                    )

            except Exception:

                logger.exception(
                    "Failed to normalize hand."
                )

        return normalized[
            : self.config.max_hands
        ]

    def _normalize_hand(
        self,
        raw: Any,
        index: int,
    ) -> HandState:

        if isinstance(
            raw,
            HandState,
        ):

            return raw

        if isinstance(
            raw,
            dict,
        ):

            hand_id = str(
                raw.get(
                    "hand_id",
                    raw.get(
                        "id",
                        f"hand_{index}",
                    ),
                )
            )

            side = self._normalize_side(
                raw.get(
                    "side",
                    raw.get(
                        "handedness",
                        "unknown",
                    ),
                )
            )

            confidence = self._to_float(
                raw.get(
                    "confidence",
                    raw.get(
                        "score",
                        1.0,
                    ),
                ),
                1.0,
            )

            landmarks = raw.get(
                "landmarks",
                [],
            )

            return self._build_hand(
                hand_id,
                side,
                confidence,
                landmarks,
                raw,
            )

        hand_id = str(
            getattr(
                raw,
                "hand_id",
                getattr(
                    raw,
                    "id",
                    f"hand_{index}",
                ),
            )
        )

        side = self._normalize_side(
            getattr(
                raw,
                "side",
                getattr(
                    raw,
                    "handedness",
                    "unknown",
                ),
            )
        )

        confidence = self._to_float(
            getattr(
                raw,
                "confidence",
                getattr(
                    raw,
                    "score",
                    1.0,
                ),
            ),
            1.0,
        )

        landmarks = getattr(
            raw,
            "landmarks",
            [],
        )

        metadata = {}

        if hasattr(
            raw,
            "__dict__",
        ):

            metadata = dict(
                raw.__dict__
            )

        return self._build_hand(
            hand_id,
            side,
            confidence,
            landmarks,
            metadata,
        )

    def _build_hand(
        self,
        hand_id: str,
        side: HandSide,
        confidence: float,
        landmarks: Any,
        metadata: dict[str, Any],
    ) -> HandState:

        points = self._normalize_landmarks(
            landmarks
        )

        palm = (
            self._estimate_palm(points)
            if points
            else None
        )

        wrist = (
            points[0]
            if len(points) > 0
            else None
        )

        thumb = (
            points[4]
            if len(points) > 4
            else None
        )

        index_tip = (
            points[8]
            if len(points) > 8
            else None
        )

        middle_tip = (
            points[12]
            if len(points) > 12
            else None
        )

        ring_tip = (
            points[16]
            if len(points) > 16
            else None
        )

        pinky_tip = (
            points[20]
            if len(points) > 20
            else None
        )

        open_state = self._estimate_open_hand(
            points
        )

        closed_state = self._estimate_closed_hand(
            points
        )

        return HandState(
            hand_id=hand_id,
            side=side,
            confidence=confidence,
            landmarks=points,
            palm=palm,
            wrist=wrist,
            thumb_tip=thumb,
            index_tip=index_tip,
            middle_tip=middle_tip,
            ring_tip=ring_tip,
            pinky_tip=pinky_tip,
            is_open=open_state,
            is_closed=closed_state,
            metadata=metadata,
        )

    def _normalize_landmarks(
        self,
        landmarks: Any,
    ) -> list[GesturePoint]:

        if landmarks is None:
            return []

        if isinstance(
            landmarks,
            dict,
        ):

            landmarks = (
                landmarks.get(
                    "landmarks",
                    [],
                )
            )

        result: list[
            GesturePoint
        ] = []

        for point in landmarks:

            if isinstance(
                point,
                GesturePoint,
            ):

                result.append(point)

                continue

            if isinstance(
                point,
                dict,
            ):

                result.append(
                    GesturePoint(
                        x=self._to_float(
                            point.get(
                                "x",
                                0.0,
                            ),
                            0.0,
                        ),
                        y=self._to_float(
                            point.get(
                                "y",
                                0.0,
                            ),
                            0.0,
                        ),
                        z=self._to_float(
                            point.get(
                                "z",
                                0.0,
                            ),
                            0.0,
                        ),
                    )
                )

                continue

            if isinstance(
                point,
                (list, tuple),
            ):

                if len(point) >= 2:

                    result.append(
                        GesturePoint(
                            x=float(point[0]),
                            y=float(point[1]),
                            z=(
                                float(point[2])
                                if len(point) > 2
                                else 0.0
                            ),
                        )
                    )

                continue

            x = getattr(
                point,
                "x",
                None,
            )

            y = getattr(
                point,
                "y",
                None,
            )

            z = getattr(
                point,
                "z",
                0.0,
            )

            if (
                x is not None
                and y is not None
            ):

                result.append(
                    GesturePoint(
                        x=float(x),
                        y=float(y),
                        z=float(z),
                    )
                )

        return result

    # ========================================================================
    # HAND PROCESSING
    # ========================================================================

    def _process_hand(
        self,
        hand: HandState,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        events: list[
            GestureEvent
        ] = []

        if hand.palm:

            self._update_position_history(
                hand
            )

        # ------------------------------------------------------------
        # Classifier integration.
        # ------------------------------------------------------------

        classifier_result = (
            self._classify_with_external_classifier(
                hand
            )
        )

        if classifier_result:

            events.extend(
                self._process_classifier_result(
                    hand,
                    classifier_result,
                    timestamp,
                    frame_number,
                )
            )

        # ------------------------------------------------------------
        # Built-in gesture detection.
        # ------------------------------------------------------------

        events.extend(
            self._detect_pinch(
                hand,
                timestamp,
                frame_number,
            )
        )

        events.extend(
            self._detect_grab(
                hand,
                timestamp,
                frame_number,
            )
        )

        events.extend(
            self._detect_point(
                hand,
                timestamp,
                frame_number,
            )
        )

        events.extend(
            self._detect_palm(
                hand,
                timestamp,
                frame_number,
            )
        )

        events.extend(
            self._detect_swipe(
                hand,
                timestamp,
                frame_number,
            )
        )

        events.extend(
            self._detect_drag(
                hand,
                timestamp,
                frame_number,
            )
        )

        return events

    # ========================================================================
    # PINCH
    # ========================================================================

    def _detect_pinch(
        self,
        hand: HandState,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        if (
            hand.thumb_tip is None
            or hand.index_tip is None
        ):

            return []

        distance = (
            hand.thumb_tip.distance_to(
                hand.index_tip
            )
        )

        key = (
            hand.hand_id,
            "pinch",
        )

        state = self._gesture_states.get(
            key,
            {},
        )

        active = bool(
            state.get(
                "active",
                False,
            )
        )

        threshold = (
            self.config.pinch_threshold
            if not active
            else self.config.pinch_release_threshold
        )

        is_pinching = (
            distance <= threshold
        )

        if (
            is_pinching
            and not active
        ):

            self._set_gesture_state(
                key,
                True,
                timestamp,
            )

            return [
                self._event(
                    "pinch",
                    GestureState.STARTED,
                    hand,
                    timestamp,
                    frame_number,
                    confidence=hand.confidence,
                    position=(
                        hand.thumb_tip.midpoint(
                            hand.index_tip
                        )
                    ),
                    metadata={
                        "distance": distance,
                    },
                )
            ]

        if (
            is_pinching
            and active
        ):

            if not self.config.emit_updates:
                return []

            return [
                self._event(
                    "pinch",
                    GestureState.ACTIVE,
                    hand,
                    timestamp,
                    frame_number,
                    confidence=hand.confidence,
                    position=(
                        hand.thumb_tip.midpoint(
                            hand.index_tip
                        )
                    ),
                    metadata={
                        "distance": distance,
                    },
                )
            ]

        if (
            not is_pinching
            and active
        ):

            duration = (
                timestamp
                - self._gesture_started_at(
                    key,
                    timestamp,
                )
            )

            self._set_gesture_state(
                key,
                False,
                timestamp,
            )

            return [
                self._event(
                    "pinch",
                    GestureState.ENDED,
                    hand,
                    timestamp,
                    frame_number,
                    confidence=hand.confidence,
                    position=(
                        hand.thumb_tip.midpoint(
                            hand.index_tip
                        )
                    ),
                    duration=duration,
                    metadata={
                        "distance": distance,
                    },
                )
            ]

        return []

    # ========================================================================
    # GRAB
    # ========================================================================

    def _detect_grab(
        self,
        hand: HandState,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        if not hand.landmarks:

            return []

        key = (
            hand.hand_id,
            "grab",
        )

        active = self._is_gesture_active(
            key
        )

        grab = (
            hand.is_closed
            or self._finger_compaction(
                hand
            )
            <= self.config.grab_threshold
        )

        if grab and not active:

            self._set_gesture_state(
                key,
                True,
                timestamp,
            )

            return [
                self._event(
                    "grab",
                    GestureState.STARTED,
                    hand,
                    timestamp,
                    frame_number,
                )
            ]

        if grab and active:

            if not self.config.emit_updates:
                return []

            return [
                self._event(
                    "grab",
                    GestureState.ACTIVE,
                    hand,
                    timestamp,
                    frame_number,
                )
            ]

        if not grab and active:

            duration = (
                timestamp
                - self._gesture_started_at(
                    key,
                    timestamp,
                )
            )

            self._set_gesture_state(
                key,
                False,
                timestamp,
            )

            return [
                self._event(
                    "grab",
                    GestureState.ENDED,
                    hand,
                    timestamp,
                    frame_number,
                    duration=duration,
                )
            ]

        return []

    # ========================================================================
    # POINT
    # ========================================================================

    def _detect_point(
        self,
        hand: HandState,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        if len(hand.landmarks) < 21:
            return []

        index_extended = (
            self._finger_extended(
                hand.landmarks,
                8,
                6,
            )
        )

        middle_extended = (
            self._finger_extended(
                hand.landmarks,
                12,
                10,
            )
        )

        ring_extended = (
            self._finger_extended(
                hand.landmarks,
                16,
                14,
            )
        )

        pinky_extended = (
            self._finger_extended(
                hand.landmarks,
                20,
                18,
            )
        )

        pointing = (
            index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        )

        key = (
            hand.hand_id,
            "point",
        )

        active = self._is_gesture_active(
            key
        )

        if pointing and not active:

            self._set_gesture_state(
                key,
                True,
                timestamp,
            )

            return [
                self._event(
                    "point",
                    GestureState.STARTED,
                    hand,
                    timestamp,
                    frame_number,
                    position=(
                        hand.index_tip
                    ),
                )
            ]

        if pointing and active:

            if not self.config.emit_updates:
                return []

            return [
                self._event(
                    "point",
                    GestureState.ACTIVE,
                    hand,
                    timestamp,
                    frame_number,
                    position=(
                        hand.index_tip
                    ),
                )
            ]

        if not pointing and active:

            self._set_gesture_state(
                key,
                False,
                timestamp,
            )

            return [
                self._event(
                    "point",
                    GestureState.ENDED,
                    hand,
                    timestamp,
                    frame_number,
                )
            ]

        return []

    # ========================================================================
    # PALM
    # ========================================================================

    def _detect_palm(
        self,
        hand: HandState,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        if len(hand.landmarks) < 21:
            return []

        open_palm = (
            hand.is_open
        )

        key = (
            hand.hand_id,
            "palm",
        )

        active = self._is_gesture_active(
            key
        )

        if open_palm and not active:

            self._set_gesture_state(
                key,
                True,
                timestamp,
            )

            return [
                self._event(
                    "palm",
                    GestureState.STARTED,
                    hand,
                    timestamp,
                    frame_number,
                    position=hand.palm,
                )
            ]

        if open_palm and active:

            if not self.config.emit_updates:
                return []

            return [
                self._event(
                    "palm",
                    GestureState.ACTIVE,
                    hand,
                    timestamp,
                    frame_number,
                    position=hand.palm,
                )
            ]

        if not open_palm and active:

            self._set_gesture_state(
                key,
                False,
                timestamp,
            )

            return [
                self._event(
                    "palm",
                    GestureState.ENDED,
                    hand,
                    timestamp,
                    frame_number,
                )
            ]

        return []

    # ========================================================================
    # SWIPE
    # ========================================================================

    def _detect_swipe(
        self,
        hand: HandState,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        history = self._position_history.get(
            hand.hand_id
        )

        if not history or len(history) < 2:
            return []

        first = history[0]

        last = history[-1]

        delta = GesturePoint(
            last.x - first.x,
            last.y - first.y,
            last.z - first.z,
        )

        distance = math.sqrt(
            delta.x ** 2
            + delta.y ** 2
            + delta.z ** 2
        )

        if distance < (
            self.config.swipe_distance_threshold
        ):

            return []

        velocity = self._estimate_velocity(
            hand.hand_id
        )

        if velocity is None:

            return []

        speed = math.sqrt(
            velocity.x ** 2
            + velocity.y ** 2
            + velocity.z ** 2
        )

        if speed < (
            self.config.swipe_velocity_threshold
        ):

            return []

        direction = self._swipe_direction(
            delta
        )

        key = (
            hand.hand_id,
            "swipe",
        )

        now = self._last_gesture_emit(
            key
        )

        if (
            now is not None
            and timestamp - now < 0.35
        ):

            return []

        self._mark_gesture_emit(
            key,
            timestamp,
        )

        return [
            self._event(
                f"swipe_{direction}",
                GestureState.STARTED,
                hand,
                timestamp,
                frame_number,
                delta=delta,
                velocity=velocity,
                metadata={
                    "direction": direction,
                    "distance": distance,
                    "speed": speed,
                },
            )
        ]

    # ========================================================================
    # DRAG
    # ========================================================================

    def _detect_drag(
        self,
        hand: HandState,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        if hand.palm is None:
            return []

        grab_key = (
            hand.hand_id,
            "grab",
        )

        drag_key = (
            hand.hand_id,
            "drag",
        )

        grabbing = self._is_gesture_active(
            grab_key
        )

        dragging = self._is_gesture_active(
            drag_key
        )

        if grabbing and not dragging:

            self._set_gesture_state(
                drag_key,
                True,
                timestamp,
            )

            return [
                self._event(
                    "drag",
                    GestureState.STARTED,
                    hand,
                    timestamp,
                    frame_number,
                    position=hand.palm,
                )
            ]

        if grabbing and dragging:

            previous = (
                self._previous_hands.get(
                    hand.hand_id
                )
            )

            delta = None

            if (
                previous is not None
                and previous.palm is not None
            ):

                delta = GesturePoint(
                    hand.palm.x
                    - previous.palm.x,
                    hand.palm.y
                    - previous.palm.y,
                    hand.palm.z
                    - previous.palm.z,
                )

            if not self.config.emit_updates:
                return []

            return [
                self._event(
                    "drag",
                    GestureState.UPDATED,
                    hand,
                    timestamp,
                    frame_number,
                    position=hand.palm,
                    delta=delta,
                )
            ]

        if not grabbing and dragging:

            self._set_gesture_state(
                drag_key,
                False,
                timestamp,
            )

            return [
                self._event(
                    "drag",
                    GestureState.ENDED,
                    hand,
                    timestamp,
                    frame_number,
                    position=hand.palm,
                )
            ]

        return []

    # ========================================================================
    # TWO-HAND GESTURES
    # ========================================================================

    def _process_two_hand(
        self,
        hands: list[HandState],
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        if len(hands) < 2:
            return []

        first = hands[0]

        second = hands[1]

        if (
            first.palm is None
            or second.palm is None
        ):

            return []

        events: list[
            GestureEvent
        ] = []

        distance = (
            first.palm.distance_to(
                second.palm
            )
        )

        previous_first = (
            self._previous_hands.get(
                first.hand_id
            )
        )

        previous_second = (
            self._previous_hands.get(
                second.hand_id
            )
        )

        if (
            previous_first is not None
            and previous_second is not None
            and previous_first.palm is not None
            and previous_second.palm is not None
        ):

            previous_distance = (
                previous_first.palm.distance_to(
                    previous_second.palm
                )
            )

            scale_delta = (
                distance
                - previous_distance
            )

            if abs(scale_delta) >= (
                self.config.zoom_threshold
            ):

                scale = (
                    distance
                    / previous_distance
                    if previous_distance > 0
                    else 1.0
                )

                direction = (
                    "zoom_in"
                    if scale_delta > 0
                    else "zoom_out"
                )

                events.append(
                    self._event(
                        direction,
                        GestureState.UPDATED,
                        first,
                        timestamp,
                        frame_number,
                        position=(
                            first.palm.midpoint(
                                second.palm
                            )
                        ),
                        scale=scale,
                        metadata={
                            "distance": distance,
                            "previous_distance": (
                                previous_distance
                            ),
                        },
                    )
                )

        rotation = self._calculate_rotation(
            first.palm,
            second.palm,
            (
                previous_first.palm
                if (
                previous_first is not None
                and previous_first.palm is not None
                )
                else None
            ),
            (
                previous_second.palm
                if (
                previous_second is not None
                and previous_second.palm is not None
                )
                else None
            ),
        )

        if (
            rotation is not None
            and abs(rotation)
            >= self.config.rotation_threshold
        ):

            direction = (
                "rotate_right"
                if rotation > 0
                else "rotate_left"
            )

            events.append(
                self._event(
                    direction,
                    GestureState.UPDATED,
                    first,
                    timestamp,
                    frame_number,
                    position=(
                        first.palm.midpoint(
                            second.palm
                        )
                    ),
                    rotation=rotation,
                    metadata={
                        "two_hand": True,
                    },
                )
            )

        return events

    # ========================================================================
    # GEOMETRY
    # ========================================================================

    def _estimate_palm(
        self,
        points: list[GesturePoint],
    ) -> Optional[GesturePoint]:

        indices = [
            0,
            5,
            9,
            13,
            17,
        ]

        valid = [
            points[index]
            for index in indices
            if index < len(points)
        ]

        if not valid:
            return None

        return GesturePoint(
            x=sum(
                p.x for p in valid
            )
            / len(valid),
            y=sum(
                p.y for p in valid
            )
            / len(valid),
            z=sum(
                p.z for p in valid
            )
            / len(valid),
        )

    def _estimate_open_hand(
        self,
        points: list[GesturePoint],
    ) -> bool:

        if len(points) < 21:
            return False

        fingers = [
            (8, 6),
            (12, 10),
            (16, 14),
            (20, 18),
        ]

        extended = sum(
            1
            for tip, joint in fingers
            if self._finger_extended(
                points,
                tip,
                joint,
            )
        )

        return extended >= 3

    def _estimate_closed_hand(
        self,
        points: list[GesturePoint],
    ) -> bool:

        if len(points) < 21:
            return False

        fingers = [
            (8, 6),
            (12, 10),
            (16, 14),
            (20, 18),
        ]

        extended = sum(
            1
            for tip, joint in fingers
            if self._finger_extended(
                points,
                tip,
                joint,
            )
        )

        return extended <= 1

    def _finger_extended(
        self,
        points: list[GesturePoint],
        tip_index: int,
        joint_index: int,
    ) -> bool:

        if (
            tip_index >= len(points)
            or joint_index >= len(points)
        ):

            return False

        wrist = points[0]

        tip = points[tip_index]

        joint = points[joint_index]

        tip_distance = (
            wrist.distance_to(tip)
        )

        joint_distance = (
            wrist.distance_to(joint)
        )

        return (
            tip_distance
            > joint_distance * 1.12
        )

    def _finger_compaction(
        self,
        hand: HandState,
    ) -> float:

        if (
            hand.palm is None
            or len(hand.landmarks) < 21
        ):

            return 999.0

        tips = [
            hand.index_tip,
            hand.middle_tip,
            hand.ring_tip,
            hand.pinky_tip,
        ]

        distances = [
            hand.palm.distance_to(
                tip
            )
            for tip in tips
            if tip is not None
        ]

        if not distances:
            return 999.0

        return sum(
            distances
        ) / len(distances)

    def _calculate_rotation(
        self,
        first: GesturePoint,
        second: GesturePoint,
        previous_first: Optional[
            GesturePoint
        ],
        previous_second: Optional[
            GesturePoint
        ],
    ) -> Optional[float]:

        if (
            previous_first is None
            or previous_second is None
        ):

            return None

        current_angle = math.atan2(
            second.y - first.y,
            second.x - first.x,
        )

        previous_angle = math.atan2(
            previous_second.y
            - previous_first.y,
            previous_second.x
            - previous_first.x,
        )

        delta = (
            current_angle
            - previous_angle
        )

        while delta > math.pi:
            delta -= 2 * math.pi

        while delta < -math.pi:
            delta += 2 * math.pi

        return delta

    def _swipe_direction(
        self,
        delta: GesturePoint,
    ) -> str:

        if (
            abs(delta.x)
            >= abs(delta.y)
        ):

            return (
                "right"
                if delta.x > 0
                else "left"
            )

        return (
            "down"
            if delta.y > 0
            else "up"
        )

    # ========================================================================
    # POSITION HISTORY
    # ========================================================================

    def _update_position_history(
        self,
        hand: HandState,
    ) -> None:

        if hand.palm is None:
            return

        with self._lock:

            history = (
                self._position_history.get(
                    hand.hand_id
                )
            )

            if history is None:

                history = deque(
                    maxlen=20
                )

                self._position_history[
                    hand.hand_id
                ] = history

            history.append(
                hand.palm
            )

    def _estimate_velocity(
        self,
        hand_id: str,
    ) -> Optional[
        GesturePoint
    ]:

        history = (
            self._position_history.get(
                hand_id
            )
        )

        if (
            history is None
            or len(history) < 2
        ):

            return None

        previous = history[-2]

        current = history[-1]

        return GesturePoint(
            current.x - previous.x,
            current.y - previous.y,
            current.z - previous.z,
        )

    # ========================================================================
    # EXTERNAL CLASSIFIER
    # ========================================================================

    def _classify_with_external_classifier(
        self,
        hand: HandState,
    ) -> Any:

        if self.classifier is None:
            return None

        for method_name in (
            "classify",
            "predict",
            "process",
            "analyze",
        ):

            method = getattr(
                self.classifier,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                return method(hand)

            except Exception:

                logger.debug(
                    "External classifier failed.",
                    exc_info=True,
                )

                return None

        return None

    def _process_classifier_result(
        self,
        hand: HandState,
        result: Any,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        if result is None:
            return []

        if isinstance(
            result,
            str,
        ):

            name = result

            confidence = 1.0

        elif isinstance(
            result,
            dict,
        ):

            name = str(
                result.get(
                    "gesture",
                    result.get(
                        "name",
                        "",
                    ),
                )
            )

            confidence = self._to_float(
                result.get(
                    "confidence",
                    result.get(
                        "score",
                        1.0,
                    ),
                ),
                1.0,
            )

        else:

            name = str(
                getattr(
                    result,
                    "gesture",
                    getattr(
                        result,
                        "name",
                        "",
                    ),
                )
            )

            confidence = self._to_float(
                getattr(
                    result,
                    "confidence",
                    getattr(
                        result,
                        "score",
                        1.0,
                    ),
                ),
                1.0,
            )

        if not name:
            return []

        if (
            confidence
            < self.config.confidence_threshold
        ):

            return []

        return [
            self._event(
                name.lower(),
                GestureState.ACTIVE,
                hand,
                timestamp,
                frame_number,
                confidence=confidence,
                metadata={
                    "source": "external_classifier"
                },
            )
        ]

    # ========================================================================
    # GESTURE STATE
    # ========================================================================

    def _set_gesture_state(
        self,
        key: tuple[str, str],
        active: bool,
        timestamp: float,
    ) -> None:

        with self._lock:

            state = self._gesture_states.setdefault(
                key,
                {},
            )

            state["active"] = active

            if active:

                state[
                    "started_at"
                ] = timestamp

            else:

                state[
                    "ended_at"
                ] = timestamp

    def _is_gesture_active(
        self,
        key: tuple[str, str],
    ) -> bool:

        with self._lock:

            return bool(
                self._gesture_states.get(
                    key,
                    {},
                ).get(
                    "active",
                    False,
                )
            )

    def _gesture_started_at(
        self,
        key: tuple[str, str],
        default: float,
    ) -> float:

        with self._lock:

            return float(
                self._gesture_states.get(
                    key,
                    {},
                ).get(
                    "started_at",
                    default,
                )
            )

    def _last_gesture_emit(
        self,
        key: tuple[str, str],
    ) -> Optional[float]:

        with self._lock:

            value = (
                self._gesture_states.get(
                    key,
                    {},
                ).get(
                    "last_emit"
                )
            )

            if value is None:
                return None

            return float(value)

    def _mark_gesture_emit(
        self,
        key: tuple[str, str],
        timestamp: float,
    ) -> None:

        with self._lock:

            self._gesture_states.setdefault(
                key,
                {},
            )["last_emit"] = timestamp

    # ========================================================================
    # EVENT CREATION
    # ========================================================================

    def _event(
        self,
        name: str,
        state: GestureState,
        hand: HandState,
        timestamp: float,
        frame_number: int,
        *,
        confidence: Optional[float] = None,
        position: Optional[
            GesturePoint
        ] = None,
        delta: Optional[
            GesturePoint
        ] = None,
        scale: Optional[float] = None,
        rotation: Optional[float] = None,
        velocity: Optional[
            GesturePoint
        ] = None,
        duration: float = 0.0,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> GestureEvent:

        return GestureEvent(
            name=name,
            state=state,
            timestamp=timestamp,
            confidence=(
                hand.confidence
                if confidence is None
                else confidence
            ),
            hand=hand.side,
            hand_id=hand.hand_id,
            position=position,
            delta=delta,
            scale=scale,
            rotation=rotation,
            velocity=velocity,
            duration=duration,
            frame_number=frame_number,
            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )

    # ========================================================================
    # DISAPPEARING HANDS
    # ========================================================================

    def _end_hand_gestures(
        self,
        hand_id: str,
        timestamp: float,
        frame_number: int,
    ) -> list[GestureEvent]:

        events: list[
            GestureEvent
        ] = []

        previous = (
            self._previous_hands.get(
                hand_id
            )
        )

        if previous is None:
            return events

        for (
            state_key,
            state,
        ) in list(
            self._gesture_states.items()
        ):

            current_hand_id, name = (
                state_key
            )

            if current_hand_id != hand_id:
                continue

            if not state.get(
                "active",
                False,
            ):

                continue

            started = float(
                state.get(
                    "started_at",
                    timestamp,
                )
            )

            duration = max(
                0.0,
                timestamp - started,
            )

            self._set_gesture_state(
                state_key,
                False,
                timestamp,
            )

            events.append(
                self._event(
                    name,
                    GestureState.ENDED,
                    previous,
                    timestamp,
                    frame_number,
                    duration=duration,
                    metadata={
                        "reason": (
                            "hand_lost"
                        )
                    },
                )
            )

        self._position_history.pop(
            hand_id,
            None,
        )

        return events

    def _cancel_all_gestures(
        self,
    ) -> None:

        timestamp = time.time()

        for key, state in list(
            self._gesture_states.items()
        ):

            if state.get(
                "active",
                False,
            ):

                state["active"] = False

                state["ended_at"] = timestamp

    # ========================================================================
    # EVENT HISTORY / PUBLISHING
    # ========================================================================

    def _record_event(
        self,
        event: GestureEvent,
    ) -> None:

        with self._lock:

            self._event_history.append(
                event
            )

    def _publish(
        self,
        event: GestureEvent,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks
            )

        for callback in callbacks:

            try:

                callback(event)

            except Exception:

                logger.exception(
                    "Gesture callback failed."
                )

        if self.mapper is not None:

            try:

                method = getattr(
                    self.mapper,
                    "map",
                    None,
                )

                if callable(method):

                    method(event)

            except Exception:

                logger.exception(
                    "Gesture mapper failed."
                )

    # ========================================================================
    # PUBLIC STATE ACCESS
    # ========================================================================

    def get_hands(
        self,
    ) -> list[HandState]:

        with self._lock:

            return list(
                self._hands.values()
            )

    def get_hand(
        self,
        hand_id: str,
    ) -> Optional[HandState]:

        with self._lock:

            return self._hands.get(
                hand_id
            )

    def get_active_gestures(
        self,
    ) -> list[str]:

        with self._lock:

            return [
                name
                for (
                    _hand_id,
                    name,
                ), state in (
                    self._gesture_states.items()
                )
                if state.get(
                    "active",
                    False,
                )
            ]

    def get_event_history(
        self,
        limit: Optional[int] = None,
    ) -> list[GestureEvent]:

        with self._lock:

            events = list(
                self._event_history
            )

        if limit is None:
            return events

        return events[-max(0, limit):]

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            history = list(
                self._event_history
            )

        counts: dict[
            str,
            int,
        ] = {}

        for event in history:

            counts[event.name] = (
                counts.get(
                    event.name,
                    0,
                )
                + 1
            )

        return {
            "running": self._running,
            "frame_number": (
                self._frame_number
            ),
            "hands_detected": len(
                self._hands
            ),
            "active_gestures": (
                self.get_active_gestures()
            ),
            "event_count": len(history),
            "gesture_counts": counts,
        }

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _normalize_side(
        self,
        value: Any,
    ) -> HandSide:

        if isinstance(
            value,
            HandSide,
        ):

            return value

        text = str(
            value
        ).lower()

        if "left" in text:
            return HandSide.LEFT

        if "right" in text:
            return HandSide.RIGHT

        return HandSide.UNKNOWN

    def _to_float(
        self,
        value: Any,
        default: float,
    ) -> float:

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default


# ============================================================================
# DEFAULT ENGINE
# ============================================================================


_default_engine: Optional[
    GestureEngine
] = None

_default_engine_lock = (
    threading.RLock()
)


def get_gesture_engine() -> GestureEngine:

    global _default_engine

    with _default_engine_lock:

        if _default_engine is None:

            _default_engine = (
                GestureEngine()
            )

        return _default_engine


def process_gestures(
    hands: Any,
    **kwargs: Any,
) -> list[GestureEvent]:

    return (
        get_gesture_engine()
        .update(
            hands,
            **kwargs,
        )
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "GestureState",
    "HandSide",
    "GesturePoint",
    "HandState",
    "GestureEvent",
    "GestureEngineConfig",
    "GestureEngine",
    "get_gesture_engine",
    "process_gestures",
]


