"""
RENIX Gesture Classifier
========================

Converts low-level hand/finger tracking data into semantic gestures.

Supported gestures include:
    - OPEN_PALM
    - FIST
    - POINT
    - THUMBS_UP
    - THUMBS_DOWN
    - PEACE
    - THREE
    - FOUR
    - FIVE
    - PINCH
    - GRAB
    - TWO_HAND_ZOOM
    - TWO_HAND_ROTATE
    - SWIPE_LEFT
    - SWIPE_RIGHT
    - SWIPE_UP
    - SWIPE_DOWN
    - PALM_FORWARD
    - UNKNOWN

This module only CLASSIFIES gestures.
Mapping gestures to computer/UI actions belongs to gesture_mapper.py.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .finger_tracker import (
    FingerState,
    HandFingerState,
)


logger = logging.getLogger(
    "RENIX.gestures.gesture_classifier"
)


# ============================================================================
# ENUMS
# ============================================================================


class GestureType(str, Enum):
    UNKNOWN = "unknown"

    OPEN_PALM = "open_palm"
    FIST = "fist"

    POINT = "point"

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"

    PEACE = "peace"
    THREE = "three"
    FOUR = "four"
    FIVE = "five"

    PINCH = "pinch"
    GRAB = "grab"

    PALM_FORWARD = "palm_forward"

    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"

    TWO_HAND_ZOOM = "two_hand_zoom"
    TWO_HAND_ROTATE = "two_hand_rotate"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class Gesture:
    """Classified gesture."""

    gesture_type: GestureType

    confidence: float

    hand_id: Optional[str] = None

    side: str = "unknown"

    timestamp: float = field(
        default_factory=time.time
    )

    duration: float = 0.0

    magnitude: float = 0.0

    direction_x: float = 0.0
    direction_y: float = 0.0

    center_x: float = 0.0
    center_y: float = 0.0

    active_fingers: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def name(self) -> str:
        return self.gesture_type.value

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "gesture": self.gesture_type.value,
            "confidence": self.confidence,
            "hand_id": self.hand_id,
            "side": self.side,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "magnitude": self.magnitude,
            "direction": {
                "x": self.direction_x,
                "y": self.direction_y,
            },
            "center": {
                "x": self.center_x,
                "y": self.center_y,
            },
            "active_fingers": list(
                self.active_fingers
            ),
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class GestureClassifierConfig:
    """Classifier configuration."""

    enabled: bool = True

    minimum_confidence: float = 0.50

    pinch_distance_threshold: float = 0.055

    pinch_release_threshold: float = 0.085

    grab_distance_threshold: float = 0.13

    swipe_velocity_threshold: float = 0.65

    swipe_min_distance: float = 0.10

    swipe_history_size: int = 8

    gesture_stability_frames: int = 2

    gesture_switch_cooldown: float = 0.08

    two_hand_distance_change_threshold: float = 0.025

    two_hand_rotation_threshold: float = 0.12

    use_hysteresis: bool = True


# ============================================================================
# CLASSIFIER
# ============================================================================


class GestureClassifier:
    """
    High-level hand gesture classifier for RENIX.
    """

    def __init__(
        self,
        config: Optional[
            GestureClassifierConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or GestureClassifierConfig()
        )

        self._lock = threading.RLock()

        self._current: dict[
            str,
            Gesture,
        ] = {}

        self._candidate: dict[
            str,
            GestureType,
        ] = {}

        self._candidate_frames: dict[
            str,
            int,
        ] = {}

        self._last_switch: dict[
            str,
            float,
        ] = {}

        self._history: dict[
            str,
            deque[tuple[float, float, float]],
        ] = {}

        self._pinching: set[str] = set()

    # ========================================================================
    # MAIN UPDATE
    # ========================================================================

    def update(
        self,
        hands: list[
            HandFingerState
        ],
        *,
        timestamp: Optional[float] = None,
    ) -> list[Gesture]:

        if not self.config.enabled:

            return []

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        with self._lock:

            active_ids = {
                hand.hand_id
                for hand in hands
            }

            result: list[Gesture] = []

            for hand in hands:

                if (
                    hand.confidence
                    < self.config.minimum_confidence
                ):
                    continue

                self._update_history(
                    hand,
                    now,
                )

                gesture = (
                    self._classify_hand(
                        hand,
                        now,
                    )
                )

                stable = (
                    self._stabilize(
                        hand.hand_id,
                        gesture,
                        now,
                    )
                )

                result.append(stable)

            # Two-hand gestures are evaluated separately.
            if len(hands) >= 2:

                two_hand = (
                    self._classify_two_hand(
                        hands,
                        now,
                    )
                )

                if two_hand:

                    result.append(
                        two_hand
                    )

            # Clean stale state.
            stale = [
                hand_id
                for hand_id in self._current
                if hand_id not in active_ids
            ]

            for hand_id in stale:

                self._current.pop(
                    hand_id,
                    None,
                )

                self._candidate.pop(
                    hand_id,
                    None,
                )

                self._candidate_frames.pop(
                    hand_id,
                    None,
                )

                self._history.pop(
                    hand_id,
                    None,
                )

                self._pinching.discard(
                    hand_id
                )

            return result

    # ========================================================================
    # SINGLE HAND CLASSIFICATION
    # ========================================================================

    def _classify_hand(
        self,
        hand: HandFingerState,
        timestamp: float,
    ) -> Gesture:

        fingers = hand.fingers

        thumb = fingers.get("thumb")
        index = fingers.get("index")
        middle = fingers.get("middle")
        ring = fingers.get("ring")
        pinky = fingers.get("pinky")

        active = [
            name
            for name, finger in fingers.items()
            if finger.extended
        ]

        # ------------------------------------------------------------
        # Pinch has priority over ordinary finger patterns.
        # ------------------------------------------------------------

        pinch_confidence = (
            self._pinch_confidence(hand)
        )

        if pinch_confidence >= 0.50:

            self._pinching.add(
                hand.hand_id
            )

            return self._make_gesture(
                GestureType.PINCH,
                pinch_confidence,
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Open palm.
        # ------------------------------------------------------------

        if (
            thumb
            and index
            and middle
            and ring
            and pinky
            and all(
                finger.extended
                for finger in (
                    thumb,
                    index,
                    middle,
                    ring,
                    pinky,
                )
            )
        ):

            return self._make_gesture(
                GestureType.OPEN_PALM,
                self._pattern_confidence(
                    hand,
                    5,
                ),
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Fist.
        # ------------------------------------------------------------

        if (
            self._is_curled(index)
            and self._is_curled(middle)
            and self._is_curled(ring)
            and self._is_curled(pinky)
        ):

            return self._make_gesture(
                GestureType.FIST,
                self._pattern_confidence(
                    hand,
                    4,
                ),
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Thumbs up / down.
        # ------------------------------------------------------------

        if thumb and self._is_extended(thumb):

            others_curled = all(
                self._is_curled(finger)
                for finger in (
                    index,
                    middle,
                    ring,
                    pinky,
                )
            )

            if others_curled:

                if thumb.direction_y < -0.15:

                    return self._make_gesture(
                        GestureType.THUMBS_UP,
                        0.90,
                        hand,
                        timestamp,
                        active,
                    )

                if thumb.direction_y > 0.15:

                    return self._make_gesture(
                        GestureType.THUMBS_DOWN,
                        0.90,
                        hand,
                        timestamp,
                        active,
                    )

        # ------------------------------------------------------------
        # Point.
        # ------------------------------------------------------------

        if (
            index
            and self._is_extended(index)
            and middle
            and ring
            and pinky
            and self._is_curled(middle)
            and self._is_curled(ring)
            and self._is_curled(pinky)
        ):

            return self._make_gesture(
                GestureType.POINT,
                0.92,
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Peace.
        # ------------------------------------------------------------

        if (
            index
            and middle
            and ring
            and pinky
            and self._is_extended(index)
            and self._is_extended(middle)
            and self._is_curled(ring)
            and self._is_curled(pinky)
        ):

            return self._make_gesture(
                GestureType.PEACE,
                0.94,
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Three.
        # ------------------------------------------------------------

        if (
            index
            and middle
            and ring
            and self._is_extended(index)
            and self._is_extended(middle)
            and self._is_extended(ring)
            and pinky
            and self._is_curled(pinky)
        ):

            return self._make_gesture(
                GestureType.THREE,
                0.90,
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Four.
        # ------------------------------------------------------------

        if (
            index
            and middle
            and ring
            and pinky
            and self._is_extended(index)
            and self._is_extended(middle)
            and self._is_extended(ring)
            and self._is_extended(pinky)
            and thumb
            and self._is_curled(thumb)
        ):

            return self._make_gesture(
                GestureType.FOUR,
                0.90,
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Five.
        # ------------------------------------------------------------

        if (
            len(active) == 5
        ):

            return self._make_gesture(
                GestureType.FIVE,
                0.92,
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Grab.
        # ------------------------------------------------------------

        grab_confidence = (
            self._grab_confidence(hand)
        )

        if grab_confidence >= 0.55:

            return self._make_gesture(
                GestureType.GRAB,
                grab_confidence,
                hand,
                timestamp,
                active,
            )

        # ------------------------------------------------------------
        # Swipe.
        # ------------------------------------------------------------

        swipe = self._detect_swipe(
            hand,
            timestamp,
        )

        if swipe:

            return swipe

        # ------------------------------------------------------------
        # Palm forward fallback.
        # ------------------------------------------------------------

        if (
            len(active) >= 4
        ):

            return self._make_gesture(
                GestureType.PALM_FORWARD,
                0.65,
                hand,
                timestamp,
                active,
            )

        return self._make_gesture(
            GestureType.UNKNOWN,
            0.30,
            hand,
            timestamp,
            active,
        )

    # ========================================================================
    # PINCH
    # ========================================================================

    def _pinch_confidence(
        self,
        hand: HandFingerState,
    ) -> float:

        thumb = hand.get("thumb")
        index = hand.get("index")

        if (
            thumb is None
            or index is None
            or thumb.tip is None
            or index.tip is None
        ):

            return 0.0

        distance = (
            thumb.tip.distance_to(
                index.tip
            )
        )

        if (
            hand.hand_id in self._pinching
            and self.config.use_hysteresis
        ):

            threshold = (
                self.config.pinch_release_threshold
            )

        else:

            threshold = (
                self.config.pinch_distance_threshold
            )

        if distance >= threshold:

            if (
                distance
                >= self.config.pinch_release_threshold
            ):

                self._pinching.discard(
                    hand.hand_id
                )

            return 0.0

        confidence = 1.0 - (
            distance / threshold
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # ========================================================================
    # GRAB
    # ========================================================================

    def _grab_confidence(
        self,
        hand: HandFingerState,
    ) -> float:

        tips = []

        for name in (
            "index",
            "middle",
            "ring",
            "pinky",
        ):

            finger = hand.get(
                name
            )

            if (
                finger
                and finger.tip
            ):

                tips.append(
                    finger.tip
                )

        if len(tips) < 3:

            return 0.0

        palm = hand.palm_center

        if palm is None:

            return 0.0

        distances = [
            palm.distance_to(
                tip
            )
            for tip in tips
        ]

        average = sum(
            distances
        ) / len(distances)

        if (
            average
            >= self.config.grab_distance_threshold
        ):

            return 0.0

        confidence = 1.0 - (
            average
            / self.config.grab_distance_threshold
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # ========================================================================
    # SWIPES
    # ========================================================================

    def _update_history(
        self,
        hand: HandFingerState,
        timestamp: float,
    ) -> None:

        if hand.palm_center is None:

            return

        history = self._history.get(
            hand.hand_id
        )

        if history is None:

            history = deque(
                maxlen=(
                    self.config.swipe_history_size
                )
            )

            self._history[
                hand.hand_id
            ] = history

        history.append(
            (
                timestamp,
                hand.palm_center.x,
                hand.palm_center.y,
            )
        )

    def _detect_swipe(
        self,
        hand: HandFingerState,
        timestamp: float,
    ) -> Optional[Gesture]:

        history = self._history.get(
            hand.hand_id
        )

        if (
            history is None
            or len(history) < 3
        ):

            return None

        first_time, first_x, first_y = (
            history[0]
        )

        last_time, last_x, last_y = (
            history[-1]
        )

        delta_time = (
            last_time - first_time
        )

        if delta_time <= 0:

            return None

        dx = last_x - first_x
        dy = last_y - first_y

        distance = math.sqrt(
            dx * dx
            + dy * dy
        )

        velocity = (
            distance / delta_time
        )

        if (
            distance
            < self.config.swipe_min_distance
        ):

            return None

        if (
            velocity
            < self.config.swipe_velocity_threshold
        ):

            return None

        if abs(dx) >= abs(dy):

            if dx > 0:

                gesture_type = (
                    GestureType.SWIPE_RIGHT
                )

            else:

                gesture_type = (
                    GestureType.SWIPE_LEFT
                )

        else:

            if dy > 0:

                gesture_type = (
                    GestureType.SWIPE_DOWN
                )

            else:

                gesture_type = (
                    GestureType.SWIPE_UP
                )

        confidence = min(
            1.0,
            velocity
            / (
                self.config.swipe_velocity_threshold
                * 2.0
            ),
        )

        return Gesture(
            gesture_type=gesture_type,
            confidence=confidence,
            hand_id=hand.hand_id,
            side=hand.side,
            timestamp=timestamp,
            magnitude=distance,
            direction_x=(
                dx / distance
            ),
            direction_y=(
                dy / distance
            ),
            center_x=(
                hand.palm_center.x
                if hand.palm_center
                else 0.0
            ),
            center_y=(
                hand.palm_center.y
                if hand.palm_center
                else 0.0
            ),
            active_fingers=(
                hand.extended_fingers()
            ),
            metadata={
                "velocity": velocity,
                "distance": distance,
            },
        )

    # ========================================================================
    # TWO-HAND GESTURES
    # ========================================================================

    def _classify_two_hand(
        self,
        hands: list[
            HandFingerState
        ],
        timestamp: float,
    ) -> Optional[Gesture]:

        if len(hands) < 2:

            return None

        first = hands[0]
        second = hands[1]

        if (
            first.palm_center is None
            or second.palm_center is None
        ):

            return None

        current_distance = self._distance_2d(
            first.palm_center.x,
            first.palm_center.y,
            second.palm_center.x,
            second.palm_center.y,
        )

        previous_distance = self._previous_two_hand_distance(
            first.hand_id,
            second.hand_id,
        )

        if previous_distance is not None:

            delta = (
                current_distance
                - previous_distance
            )

            if abs(delta) >= (
                self.config
                .two_hand_distance_change_threshold
            ):

                direction = (
                    1.0
                    if delta > 0
                    else -1.0
                )

                confidence = min(
                    1.0,
                    abs(delta)
                    / (
                        self.config
                        .two_hand_distance_change_threshold
                        * 2.0
                    ),
                )

                return Gesture(
                    gesture_type=(
                        GestureType.TWO_HAND_ZOOM
                    ),
                    confidence=confidence,
                    hand_id=None,
                    side="both",
                    timestamp=timestamp,
                    magnitude=abs(delta),
                    direction_x=direction,
                    direction_y=0.0,
                    metadata={
                        "distance": current_distance,
                        "distance_delta": delta,
                        "hands": [
                            first.hand_id,
                            second.hand_id,
                        ],
                    },
                )

        previous_angle = (
            self._previous_two_hand_angle(
                first.hand_id,
                second.hand_id,
            )
        )

        current_angle = math.atan2(
            second.palm_center.y
            - first.palm_center.y,
            second.palm_center.x
            - first.palm_center.x,
        )

        if previous_angle is not None:

            angle_delta = (
                self._normalize_angle(
                    current_angle
                    - previous_angle
                )
            )

            if abs(angle_delta) >= (
                self.config
                .two_hand_rotation_threshold
            ):

                confidence = min(
                    1.0,
                    abs(angle_delta)
                    / (
                        self.config
                        .two_hand_rotation_threshold
                        * 2.0
                    ),
                )

                return Gesture(
                    gesture_type=(
                        GestureType.TWO_HAND_ROTATE
                    ),
                    confidence=confidence,
                    hand_id=None,
                    side="both",
                    timestamp=timestamp,
                    magnitude=abs(
                        angle_delta
                    ),
                    direction_x=(
                        1.0
                        if angle_delta > 0
                        else -1.0
                    ),
                    metadata={
                        "angle_delta": angle_delta,
                        "angle": current_angle,
                        "hands": [
                            first.hand_id,
                            second.hand_id,
                        ],
                    },
                )

        return None

    def _previous_two_hand_distance(
        self,
        first_id: str,
        second_id: str,
    ) -> Optional[float]:

        first_history = self._history.get(
            first_id
        )

        second_history = self._history.get(
            second_id
        )

        if (
            not first_history
            or not second_history
        ):

            return None

        _, x1, y1 = first_history[-1]
        _, x2, y2 = second_history[-1]

        return self._distance_2d(
            x1,
            y1,
            x2,
            y2,
        )

    def _previous_two_hand_angle(
        self,
        first_id: str,
        second_id: str,
    ) -> Optional[float]:

        first_history = self._history.get(
            first_id
        )

        second_history = self._history.get(
            second_id
        )

        if (
            not first_history
            or not second_history
        ):

            return None

        _, x1, y1 = first_history[-1]
        _, x2, y2 = second_history[-1]

        return math.atan2(
            y2 - y1,
            x2 - x1,
        )

    # ========================================================================
    # STABILITY
    # ========================================================================

    def _stabilize(
        self,
        hand_id: str,
        gesture: Gesture,
        timestamp: float,
    ) -> Gesture:

        gesture_type = (
            gesture.gesture_type
        )

        current = self._current.get(
            hand_id
        )

        # Unknown gestures don't replace a stable
        # meaningful gesture immediately.
        if (
            gesture_type
            == GestureType.UNKNOWN
            and current is not None
            and current.confidence >= 0.60
        ):

            return current

        candidate = self._candidate.get(
            hand_id
        )

        if candidate == gesture_type:

            self._candidate_frames[
                hand_id
            ] = (
                self._candidate_frames.get(
                    hand_id,
                    0,
                )
                + 1
            )

        else:

            self._candidate[
                hand_id
            ] = gesture_type

            self._candidate_frames[
                hand_id
            ] = 1

        frames = self._candidate_frames[
            hand_id
        ]

        if (
            current is None
            or frames
            >= self.config.gesture_stability_frames
        ):

            last_switch = (
                self._last_switch.get(
                    hand_id,
                    0.0,
                )
            )

            if (
                current is not None
                and timestamp - last_switch
                < self.config.gesture_switch_cooldown
            ):

                return current

            self._current[
                hand_id
            ] = gesture

            self._last_switch[
                hand_id
            ] = timestamp

            return gesture

        return current

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _make_gesture(
        self,
        gesture_type: GestureType,
        confidence: float,
        hand: HandFingerState,
        timestamp: float,
        active: list[str],
    ) -> Gesture:

        return Gesture(
            gesture_type=gesture_type,
            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            hand_id=hand.hand_id,
            side=hand.side,
            timestamp=timestamp,
            center_x=(
                hand.palm_center.x
                if hand.palm_center
                else 0.0
            ),
            center_y=(
                hand.palm_center.y
                if hand.palm_center
                else 0.0
            ),
            active_fingers=list(
                active
            ),
        )

    def _pattern_confidence(
        self,
        hand: HandFingerState,
        required_count: int,
    ) -> float:

        ratio = (
            min(
                hand.extended_count,
                required_count,
            )
            / required_count
        )

        return min(
            1.0,
            0.70 + ratio * 0.30,
        )

    @staticmethod
    def _is_extended(
        finger: Optional[FingerState],
    ) -> bool:

        return bool(
            finger
            and finger.extended
        )

    @staticmethod
    def _is_curled(
        finger: Optional[FingerState],
    ) -> bool:

        return bool(
            finger
            and finger.curled
        )

    @staticmethod
    def _distance_2d(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:

        return math.sqrt(
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        )

    @staticmethod
    def _normalize_angle(
        angle: float,
    ) -> float:

        while angle > math.pi:

            angle -= 2.0 * math.pi

        while angle < -math.pi:

            angle += 2.0 * math.pi

        return angle

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def get_current(
        self,
        hand_id: Optional[str] = None,
    ) -> Any:

        with self._lock:

            if hand_id is not None:

                gesture = self._current.get(
                    hand_id
                )

                return (
                    self._copy_gesture(
                        gesture
                    )
                    if gesture
                    else None
                )

            return [
                self._copy_gesture(
                    gesture
                )
                for gesture in (
                    self._current.values()
                )
            ]

    def get_gesture(
        self,
        hand_id: str,
    ) -> Optional[Gesture]:

        return self.get_current(
            hand_id
        )

    def classify(
        self,
        hands: list[
            HandFingerState
        ],
        **kwargs: Any,
    ) -> list[Gesture]:

        return self.update(
            hands,
            **kwargs,
        )

    def clear(self) -> None:

        with self._lock:

            self._current.clear()
            self._candidate.clear()
            self._candidate_frames.clear()
            self._last_switch.clear()
            self._history.clear()
            self._pinching.clear()

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "enabled": (
                    self.config.enabled
                ),
                "tracked_gestures": len(
                    self._current
                ),
                "active_hands": len(
                    self._history
                ),
                "pinching_hands": len(
                    self._pinching
                ),
            }

    # ========================================================================
    # COPY
    # ========================================================================

    @staticmethod
    def _copy_gesture(
        gesture: Gesture,
    ) -> Gesture:

        return Gesture(
            gesture_type=gesture.gesture_type,
            confidence=gesture.confidence,
            hand_id=gesture.hand_id,
            side=gesture.side,
            timestamp=gesture.timestamp,
            duration=gesture.duration,
            magnitude=gesture.magnitude,
            direction_x=gesture.direction_x,
            direction_y=gesture.direction_y,
            center_x=gesture.center_x,
            center_y=gesture.center_y,
            active_fingers=list(
                gesture.active_fingers
            ),
            metadata=dict(
                gesture.metadata
            ),
        )


# ============================================================================
# DEFAULT INSTANCE
# ============================================================================


_default_classifier: Optional[
    GestureClassifier
] = None

_default_lock = threading.RLock()


def get_gesture_classifier() -> GestureClassifier:

    global _default_classifier

    with _default_lock:

        if _default_classifier is None:

            _default_classifier = (
                GestureClassifier()
            )

        return _default_classifier


def classify_gestures(
    hands: list[
        HandFingerState
    ],
    **kwargs: Any,
) -> list[Gesture]:

    return (
        get_gesture_classifier()
        .update(
            hands,
            **kwargs,
        )
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "GestureType",
    "Gesture",
    "GestureClassifierConfig",
    "GestureClassifier",
    "get_gesture_classifier",
    "classify_gestures",
]


