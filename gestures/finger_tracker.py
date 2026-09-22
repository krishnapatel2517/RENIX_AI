"""
RENIX Finger Tracker
====================

Tracks individual finger positions from hand landmarks.

Responsibilities:
    - Extract finger joints from 21-point hand landmarks.
    - Track fingertips and joints.
    - Determine finger extension/flexion state.
    - Calculate finger direction, length, and velocity.
    - Provide stable data to gesture_classifier.py / gesture_mapper.py.

Expected landmark layout (MediaPipe-compatible):

    0   WRIST

    1-4   THUMB
    1 MCP
    2 PIP
    3 DIP
    4 TIP

    5-8   INDEX
    5 MCP
    6 PIP
    7 DIP
    8 TIP

    9-12  MIDDLE
    9 MCP
    10 PIP
    11 DIP
    12 TIP

    13-16 RING
    13 MCP
    14 PIP
    15 DIP
    16 TIP

    17-20 PINKY
    17 MCP
    18 PIP
    19 DIP
    20 TIP
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .hand_tracker import Landmark, TrackedHand


logger = logging.getLogger(
    "RENIX.gestures.finger_tracker"
)


# ============================================================================
# CONSTANTS
# ============================================================================

FINGER_NAMES = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
)

FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class FingerState:
    """Current state of one finger."""

    name: str

    mcp: Optional[Landmark] = None
    pip: Optional[Landmark] = None
    dip: Optional[Landmark] = None
    tip: Optional[Landmark] = None

    extended: bool = False
    curled: bool = False
    partially_extended: bool = False

    confidence: float = 0.0

    length: float = 0.0
    direction_x: float = 0.0
    direction_y: float = 0.0
    direction_z: float = 0.0

    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0

    speed: float = 0.0

    extension_ratio: float = 0.0

    flexion_angle: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "name": self.name,
            "mcp": (
                self.mcp.to_dict()
                if self.mcp
                else None
            ),
            "pip": (
                self.pip.to_dict()
                if self.pip
                else None
            ),
            "dip": (
                self.dip.to_dict()
                if self.dip
                else None
            ),
            "tip": (
                self.tip.to_dict()
                if self.tip
                else None
            ),
            "extended": self.extended,
            "curled": self.curled,
            "partially_extended": (
                self.partially_extended
            ),
            "confidence": self.confidence,
            "length": self.length,
            "direction": {
                "x": self.direction_x,
                "y": self.direction_y,
                "z": self.direction_z,
            },
            "velocity": {
                "x": self.velocity_x,
                "y": self.velocity_y,
                "z": self.velocity_z,
            },
            "speed": self.speed,
            "extension_ratio": (
                self.extension_ratio
            ),
            "flexion_angle": (
                self.flexion_angle
            ),
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class HandFingerState:
    """Finger states for an entire hand."""

    hand_id: str
    side: str

    confidence: float

    fingers: dict[
        str,
        FingerState,
    ] = field(
        default_factory=dict
    )

    extended_count: int = 0
    curled_count: int = 0

    palm_center: Optional[Landmark] = None

    timestamp: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def get(
        self,
        finger_name: str,
    ) -> Optional[FingerState]:

        return self.fingers.get(
            finger_name
        )

    def is_extended(
        self,
        finger_name: str,
    ) -> bool:

        finger = self.get(
            finger_name
        )

        return bool(
            finger and finger.extended
        )

    def is_curled(
        self,
        finger_name: str,
    ) -> bool:

        finger = self.get(
            finger_name
        )

        return bool(
            finger and finger.curled
        )

    def extended_fingers(
        self,
    ) -> list[str]:

        return [
            name
            for name, finger
            in self.fingers.items()
            if finger.extended
        ]

    def curled_fingers(
        self,
    ) -> list[str]:

        return [
            name
            for name, finger
            in self.fingers.items()
            if finger.curled
        ]

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "hand_id": self.hand_id,
            "side": self.side,
            "confidence": self.confidence,
            "fingers": {
                name: finger.to_dict()
                for name, finger
                in self.fingers.items()
            },
            "extended_count": (
                self.extended_count
            ),
            "curled_count": (
                self.curled_count
            ),
            "palm_center": (
                self.palm_center.to_dict()
                if self.palm_center
                else None
            ),
            "timestamp": self.timestamp,
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class FingerTrackerConfig:
    """Finger tracking configuration."""

    enabled: bool = True

    extended_angle_threshold: float = 160.0

    curled_angle_threshold: float = 100.0

    partial_angle_threshold: float = 135.0

    extension_ratio_threshold: float = 1.20

    curl_ratio_threshold: float = 0.90

    minimum_confidence: float = 0.35

    smoothing_alpha: float = 0.55

    velocity_alpha: float = 0.40

    thumb_angle_threshold: float = 145.0

    use_3d: bool = True

    calculate_velocity: bool = True


# ============================================================================
# FINGER TRACKER
# ============================================================================


class FingerTracker:
    """
    Converts tracked hand landmarks into detailed finger states.

    This class intentionally does not classify high-level gestures such as:

        pinch
        grab
        swipe
        point
        palm
        zoom

    Those belong to the gesture classification layer.
    """

    def __init__(
        self,
        config: Optional[
            FingerTrackerConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or FingerTrackerConfig()
        )

        self._lock = threading.RLock()

        self._states: dict[
            str,
            HandFingerState,
        ] = {}

        self._previous_states: dict[
            str,
            HandFingerState,
        ] = {}

        self._last_update: dict[
            str,
            float,
        ] = {}

    # ========================================================================
    # MAIN UPDATE
    # ========================================================================

    def update(
        self,
        hands: list[
            TrackedHand
        ],
        *,
        timestamp: Optional[float] = None,
    ) -> list[HandFingerState]:

        if not self.config.enabled:

            return []

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        with self._lock:

            self._previous_states = {
                key: self._copy_state(value)
                for key, value in (
                    self._states.items()
                )
            }

            result: list[
                HandFingerState
            ] = []

            active_ids: set[str] = set()

            for hand in hands:

                if not hand.is_visible:

                    continue

                if (
                    hand.confidence
                    < self.config.minimum_confidence
                ):

                    continue

                state = (
                    self._process_hand(
                        hand,
                        now,
                    )
                )

                self._states[
                    hand.hand_id
                ] = state

                self._last_update[
                    hand.hand_id
                ] = now

                active_ids.add(
                    hand.hand_id
                )

                result.append(
                    self._copy_state(
                        state
                    )
                )

            # Remove hands that disappeared.
            stale_ids = [
                hand_id
                for hand_id in self._states
                if hand_id not in active_ids
            ]

            for hand_id in stale_ids:

                self._states.pop(
                    hand_id,
                    None,
                )

                self._last_update.pop(
                    hand_id,
                    None,
                )

            return result

    # ========================================================================
    # HAND PROCESSING
    # ========================================================================

    def _process_hand(
        self,
        hand: TrackedHand,
        timestamp: float,
    ) -> HandFingerState:

        previous = self._previous_states.get(
            hand.hand_id
        )

        fingers: dict[
            str,
            FingerState,
        ] = {}

        for finger_name in FINGER_NAMES:

            finger = (
                self._calculate_finger(
                    hand,
                    finger_name,
                )
            )

            if previous:

                previous_finger = (
                    previous.fingers.get(
                        finger_name
                    )
                )

                if previous_finger:

                    finger = (
                        self._apply_smoothing(
                            previous_finger,
                            finger,
                        )
                    )

            fingers[
                finger_name
            ] = finger

        extended_count = sum(
            1
            for finger in fingers.values()
            if finger.extended
        )

        curled_count = sum(
            1
            for finger in fingers.values()
            if finger.curled
        )

        return HandFingerState(
            hand_id=hand.hand_id,
            side=hand.side,
            confidence=hand.confidence,
            fingers=fingers,
            extended_count=extended_count,
            curled_count=curled_count,
            palm_center=hand.palm,
            timestamp=timestamp,
            metadata={
                "hand_speed": hand.speed,
                "hand_velocity": {
                    "x": hand.velocity_x,
                    "y": hand.velocity_y,
                    "z": hand.velocity_z,
                },
            },
        )

    # ========================================================================
    # FINGER CALCULATION
    # ========================================================================

    def _calculate_finger(
        self,
        hand: TrackedHand,
        finger_name: str,
    ) -> FingerState:

        indices = FINGER_LANDMARKS[
            finger_name
        ]

        points: list[
            Optional[Landmark]
        ] = [
            (
                hand.landmarks[index]
                if index < len(
                    hand.landmarks
                )
                else None
            )
            for index in indices
        ]

        mcp, pip, dip, tip = points

        if not all(points):

            return FingerState(
                name=finger_name,
                mcp=mcp,
                pip=pip,
                dip=dip,
                tip=tip,
                confidence=0.0,
            )

        assert (
            mcp is not None
            and pip is not None
            and dip is not None
            and tip is not None
        )

        extension_ratio = (
            self._calculate_extension_ratio(
                mcp,
                pip,
                dip,
                tip,
            )
        )

        flexion_angle = (
            self._calculate_finger_angle(
                mcp,
                pip,
                dip,
                tip,
            )
        )

        length = (
            mcp.distance_to(tip)
        )

        direction = (
            self._direction(
                mcp,
                tip,
            )
        )

        extended = (
            self._is_extended(
                finger_name,
                flexion_angle,
                extension_ratio,
            )
        )

        curled = (
            self._is_curled(
                finger_name,
                flexion_angle,
                extension_ratio,
            )
        )

        partially_extended = (
            not extended
            and not curled
            and flexion_angle
            >= self.config.partial_angle_threshold
        )

        return FingerState(
            name=finger_name,
            mcp=mcp,
            pip=pip,
            dip=dip,
            tip=tip,
            extended=extended,
            curled=curled,
            partially_extended=(
                partially_extended
            ),
            confidence=hand.confidence,
            length=length,
            direction_x=direction[0],
            direction_y=direction[1],
            direction_z=direction[2],
            extension_ratio=extension_ratio,
            flexion_angle=flexion_angle,
        )

    # ========================================================================
    # EXTENSION / CURL
    # ========================================================================

    def _is_extended(
        self,
        finger_name: str,
        angle: float,
        ratio: float,
    ) -> bool:

        if finger_name == "thumb":

            angle_threshold = (
                self.config.thumb_angle_threshold
            )

        else:

            angle_threshold = (
                self.config.extended_angle_threshold
            )

        return (
            angle >= angle_threshold
            and ratio
            >= self.config.extension_ratio_threshold
        )

    def _is_curled(
        self,
        finger_name: str,
        angle: float,
        ratio: float,
    ) -> bool:

        return (
            angle
            <= self.config.curled_angle_threshold
            or ratio
            <= self.config.curl_ratio_threshold
        )

    # ========================================================================
    # GEOMETRY
    # ========================================================================

    def _calculate_extension_ratio(
        self,
        mcp: Landmark,
        pip: Landmark,
        dip: Landmark,
        tip: Landmark,
    ) -> float:

        base_distance = (
            mcp.distance_to(tip)
        )

        joint_distance = (
            mcp.distance_to(pip)
            + pip.distance_to(dip)
            + dip.distance_to(tip)
        )

        if joint_distance <= 0:

            return 0.0

        return (
            base_distance
            / joint_distance
        )

    def _calculate_finger_angle(
        self,
        mcp: Landmark,
        pip: Landmark,
        dip: Landmark,
        tip: Landmark,
    ) -> float:

        angle1 = self._angle_between(
            mcp,
            pip,
            dip,
        )

        angle2 = self._angle_between(
            pip,
            dip,
            tip,
        )

        return (
            angle1 * 0.55
            + angle2 * 0.45
        )

    def _angle_between(
        self,
        a: Landmark,
        b: Landmark,
        c: Landmark,
    ) -> float:

        vector1 = (
            a.x - b.x,
            a.y - b.y,
            a.z - b.z,
        )

        vector2 = (
            c.x - b.x,
            c.y - b.y,
            c.z - b.z,
        )

        dot = (
            vector1[0] * vector2[0]
            + vector1[1] * vector2[1]
            + vector1[2] * vector2[2]
        )

        mag1 = math.sqrt(
            vector1[0] ** 2
            + vector1[1] ** 2
            + vector1[2] ** 2
        )

        mag2 = math.sqrt(
            vector2[0] ** 2
            + vector2[1] ** 2
            + vector2[2] ** 2
        )

        if mag1 <= 1e-9 or mag2 <= 1e-9:

            return 0.0

        cosine = dot / (
            mag1 * mag2
        )

        cosine = max(
            -1.0,
            min(
                1.0,
                cosine,
            ),
        )

        return math.degrees(
            math.acos(cosine)
        )

    def _direction(
        self,
        start: Landmark,
        end: Landmark,
    ) -> tuple[
        float,
        float,
        float,
    ]:

        dx = end.x - start.x
        dy = end.y - start.y
        dz = end.z - start.z

        magnitude = math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )

        if magnitude <= 1e-9:

            return (
                0.0,
                0.0,
                0.0,
            )

        return (
            dx / magnitude,
            dy / magnitude,
            dz / magnitude,
        )

    # ========================================================================
    # SMOOTHING
    # ========================================================================

    def _apply_smoothing(
        self,
        previous: FingerState,
        current: FingerState,
    ) -> FingerState:

        alpha = (
            self.config.smoothing_alpha
        )

        if (
            previous.tip is not None
            and current.tip is not None
        ):

            current.tip = (
                previous.tip.lerp(
                    current.tip,
                    alpha,
                )
            )

        if (
            previous.mcp is not None
            and current.mcp is not None
        ):

            current.mcp = (
                previous.mcp.lerp(
                    current.mcp,
                    alpha,
                )
            )

        if (
            previous.pip is not None
            and current.pip is not None
        ):

            current.pip = (
                previous.pip.lerp(
                    current.pip,
                    alpha,
                )
            )

        if (
            previous.dip is not None
            and current.dip is not None
        ):

            current.dip = (
                previous.dip.lerp(
                    current.dip,
                    alpha,
                )
            )

        if (
            previous.tip is not None
            and current.tip is not None
        ):

            current.velocity_x = (
                (
                    current.tip.x
                    - previous.tip.x
                )
                * self.config.velocity_alpha
                + previous.velocity_x
                * (
                    1.0
                    - self.config.velocity_alpha
                )
            )

            current.velocity_y = (
                (
                    current.tip.y
                    - previous.tip.y
                )
                * self.config.velocity_alpha
                + previous.velocity_y
                * (
                    1.0
                    - self.config.velocity_alpha
                )
            )

            current.velocity_z = (
                (
                    current.tip.z
                    - previous.tip.z
                )
                * self.config.velocity_alpha
                + previous.velocity_z
                * (
                    1.0
                    - self.config.velocity_alpha
                )
            )

            current.speed = math.sqrt(
                current.velocity_x ** 2
                + current.velocity_y ** 2
                + current.velocity_z ** 2
            )

        return current

    # ========================================================================
    # PUBLIC ACCESS
    # ========================================================================

    def get_state(
        self,
        hand_id: str,
    ) -> Optional[
        HandFingerState
    ]:

        with self._lock:

            state = self._states.get(
                hand_id
            )

            if state is None:

                return None

            return self._copy_state(
                state
            )

    def get_states(
        self,
    ) -> list[HandFingerState]:

        with self._lock:

            return [
                self._copy_state(state)
                for state in (
                    self._states.values()
                )
            ]

    def get_left_hand(
        self,
    ) -> Optional[
        HandFingerState
    ]:

        for state in self.get_states():

            if state.side == "left":

                return state

        return None

    def get_right_hand(
        self,
    ) -> Optional[
        HandFingerState
    ]:

        for state in self.get_states():

            if state.side == "right":

                return state

        return None

    # ========================================================================
    # SIMPLE GESTURE HELPERS
    # ========================================================================

    def is_fist(
        self,
        hand_id: str,
    ) -> bool:

        state = self.get_state(
            hand_id
        )

        if state is None:

            return False

        return (
            state.curled_count >= 4
        )

    def is_open_palm(
        self,
        hand_id: str,
    ) -> bool:

        state = self.get_state(
            hand_id
        )

        if state is None:

            return False

        return (
            state.extended_count >= 4
        )

    def is_pointing(
        self,
        hand_id: str,
    ) -> bool:

        state = self.get_state(
            hand_id
        )

        if state is None:

            return False

        index = state.get(
            "index"
        )

        if index is None:

            return False

        other_fingers_curled = all(
            state.is_curled(name)
            for name in (
                "middle",
                "ring",
                "pinky",
            )
        )

        return (
            index.extended
            and other_fingers_curled
        )

    def get_finger_tip(
        self,
        hand_id: str,
        finger_name: str,
    ) -> Optional[Landmark]:

        state = self.get_state(
            hand_id
        )

        if state is None:

            return None

        finger = state.get(
            finger_name
        )

        if finger is None:

            return None

        return finger.tip

    # ========================================================================
    # COPY HELPERS
    # ========================================================================

    def _copy_state(
        self,
        state: HandFingerState,
    ) -> HandFingerState:

        copied_fingers: dict[
            str,
            FingerState,
        ] = {}

        for name, finger in (
            state.fingers.items()
        ):

            copied_fingers[
                name
            ] = FingerState(
                name=finger.name,
                mcp=finger.mcp,
                pip=finger.pip,
                dip=finger.dip,
                tip=finger.tip,
                extended=finger.extended,
                curled=finger.curled,
                partially_extended=(
                    finger.partially_extended
                ),
                confidence=finger.confidence,
                length=finger.length,
                direction_x=(
                    finger.direction_x
                ),
                direction_y=(
                    finger.direction_y
                ),
                direction_z=(
                    finger.direction_z
                ),
                velocity_x=(
                    finger.velocity_x
                ),
                velocity_y=(
                    finger.velocity_y
                ),
                velocity_z=(
                    finger.velocity_z
                ),
                speed=finger.speed,
                extension_ratio=(
                    finger.extension_ratio
                ),
                flexion_angle=(
                    finger.flexion_angle
                ),
                metadata=dict(
                    finger.metadata
                ),
            )

        return HandFingerState(
            hand_id=state.hand_id,
            side=state.side,
            confidence=state.confidence,
            fingers=copied_fingers,
            extended_count=(
                state.extended_count
            ),
            curled_count=(
                state.curled_count
            ),
            palm_center=state.palm_center,
            timestamp=state.timestamp,
            metadata=dict(
                state.metadata
            ),
        )


# ============================================================================
# DEFAULT INSTANCE
# ============================================================================


_default_finger_tracker: Optional[
    FingerTracker
] = None

_default_lock = threading.RLock()


def get_finger_tracker() -> FingerTracker:

    global _default_finger_tracker

    with _default_lock:

        if (
            _default_finger_tracker
            is None
        ):

            _default_finger_tracker = (
                FingerTracker()
            )

        return _default_finger_tracker


def track_fingers(
    hands: list[TrackedHand],
    **kwargs: Any,
) -> list[HandFingerState]:

    return (
        get_finger_tracker()
        .update(
            hands,
            **kwargs,
        )
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "FINGER_NAMES",
    "FINGER_LANDMARKS",
    "FingerState",
    "HandFingerState",
    "FingerTrackerConfig",
    "FingerTracker",
    "get_finger_tracker",
    "track_fingers",
]


