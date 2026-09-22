"""
RENIX - Grab Gesture Detector
=============================

Detects an intentional "grab" gesture from hand landmarks.

A grab is generally interpreted as:
    - fingers curled toward the palm
    - thumb folded/in contact with the fingers
    - hand remaining sufficiently stable

This module only detects the gesture.
It does NOT:
    - move files
    - drag windows
    - control the mouse
    - execute commands

Those actions belong to gesture_mapper.py / gesture_engine.py.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence


logger = logging.getLogger("RENIX.gestures.grab")


# ============================================================================
# ENUMS
# ============================================================================


class GrabState(str, Enum):
    OPEN = "open"
    CLOSING = "closing"
    GRABBED = "grabbed"
    RELEASING = "releasing"


# ============================================================================
# LANDMARK INDICES
# ============================================================================

# MediaPipe-style hand landmark indices.
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(frozen=True)
class GrabPoint:
    """3D normalized hand landmark."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class GrabConfig:
    """
    Configuration for grab detection.

    finger_curl_threshold:
        Minimum curl score required for a finger to count as curled.

    thumb_contact_threshold:
        Maximum normalized thumb-to-index distance for a strong grab.

    activation_frames:
        Number of consecutive frames required before a grab becomes active.

    release_frames:
        Number of consecutive open frames required before release.
    """

    finger_curl_threshold: float = 0.55

    thumb_contact_threshold: float = 0.24

    palm_distance_ratio: float = 1.20

    activation_frames: int = 3

    release_frames: int = 3

    confidence_threshold: float = 0.45

    smoothing: float = 0.35

    cooldown: float = 0.10

    enabled: bool = True

    require_thumb: bool = False

    minimum_curled_fingers: int = 3


@dataclass
class GrabStateResult:
    """Result returned by the grab detector."""

    state: GrabState = GrabState.OPEN

    grabbed: bool = False

    just_grabbed: bool = False

    just_released: bool = False

    confidence: float = 0.0

    grab_strength: float = 0.0

    curled_fingers: int = 0

    finger_scores: dict[str, float] = field(
        default_factory=dict
    )

    thumb_score: float = 0.0

    thumb_contact: float = 0.0

    palm_center_x: float = 0.0
    palm_center_y: float = 0.0
    palm_center_z: float = 0.0

    duration: float = 0.0

    timestamp: float = 0.0

    hand_id: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "grabbed": self.grabbed,
            "just_grabbed": self.just_grabbed,
            "just_released": self.just_released,
            "confidence": self.confidence,
            "grab_strength": self.grab_strength,
            "curled_fingers": self.curled_fingers,
            "finger_scores": dict(
                self.finger_scores
            ),
            "thumb_score": self.thumb_score,
            "thumb_contact": self.thumb_contact,
            "palm_center": {
                "x": self.palm_center_x,
                "y": self.palm_center_y,
                "z": self.palm_center_z,
            },
            "duration": self.duration,
            "timestamp": self.timestamp,
            "hand_id": self.hand_id,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# GRAB DETECTOR
# ============================================================================


class GrabDetector:
    """
    Detects a grab using hand landmarks.

    Expected landmark format:

        [
            (x, y, z),
            ...
        ]

    with 21 MediaPipe-style landmarks.

    The detector combines:

        - finger curl
        - fingertip-to-palm distance
        - thumb position
        - hand confidence
        - frame stability

    to produce a robust grab state.
    """

    FINGER_NAMES = (
        "index",
        "middle",
        "ring",
        "pinky",
    )

    FINGER_INDICES = {
        "index": (
            INDEX_MCP,
            INDEX_PIP,
            INDEX_DIP,
            INDEX_TIP,
        ),
        "middle": (
            MIDDLE_MCP,
            MIDDLE_PIP,
            MIDDLE_DIP,
            MIDDLE_TIP,
        ),
        "ring": (
            RING_MCP,
            RING_PIP,
            RING_DIP,
            RING_TIP,
        ),
        "pinky": (
            PINKY_MCP,
            PINKY_PIP,
            PINKY_DIP,
            PINKY_TIP,
        ),
    }

    def __init__(
        self,
        config: Optional[GrabConfig] = None,
        *,
        hand_id: Optional[str] = None,
    ) -> None:

        self.config = (
            config
            or GrabConfig()
        )

        self.hand_id = hand_id

        self._lock = threading.RLock()

        self._grabbed = False

        self._state = GrabState.OPEN

        self._closing_count = 0

        self._releasing_count = 0

        self._grab_start_time: Optional[
            float
        ] = None

        self._last_transition_time = 0.0

        self._last_result = (
            GrabStateResult()
        )

        self._last_finger_scores: dict[
            str,
            float,
        ] = {}

        self._smoothed_strength = 0.0

        self._callbacks: list[
            Callable[[GrabStateResult], None]
        ] = []

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        landmarks: Sequence[
            GrabPoint | Sequence[float]
        ],
        *,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        hand_id: Optional[str] = None,
    ) -> GrabStateResult:

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

        points = self._normalize_landmarks(
            landmarks
        )

        with self._lock:

            if hand_id is not None:

                self.hand_id = hand_id

            if not self.config.enabled:

                self._grabbed = False

                self._state = GrabState.OPEN

                self._closing_count = 0

                self._releasing_count = 0

                result = GrabStateResult(
                    state=GrabState.OPEN,
                    confidence=confidence,
                    timestamp=now,
                    hand_id=self.hand_id,
                )

                self._last_result = result

                return result

            if len(points) < 21:

                logger.debug(
                    "Grab detection requires 21 landmarks"
                )

                return self._invalid_result(
                    now,
                    confidence,
                )

            if (
                confidence
                < self.config.confidence_threshold
            ):

                return self._invalid_result(
                    now,
                    confidence,
                )

            finger_scores = (
                self._calculate_finger_scores(
                    points
                )
            )

            thumb_score = (
                self._calculate_thumb_score(
                    points
                )
            )

            thumb_contact = (
                self._calculate_thumb_contact(
                    points
                )
            )

            curled_fingers = sum(
                score
                >= self.config.finger_curl_threshold
                for score
                in finger_scores.values()
            )

            finger_strength = self._average(
                list(
                    finger_scores.values()
                )
            )

            if self.config.require_thumb:

                hand_strength = (
                    finger_strength
                    * 0.75
                    + thumb_score
                    * 0.25
                )

            else:

                hand_strength = (
                    finger_strength
                    * 0.85
                    + thumb_score
                    * 0.15
                )

            if (
                curled_fingers
                >= self.config.minimum_curled_fingers
            ):

                hand_strength = max(
                    hand_strength,
                    0.65,
                )

            if (
                thumb_contact
                <= self.config.thumb_contact_threshold
            ):

                hand_strength += 0.10

            hand_strength = self._clamp(
                hand_strength,
                0.0,
                1.0,
            )

            alpha = self._clamp(
                self.config.smoothing,
                0.0,
                1.0,
            )

            self._smoothed_strength = (
                self._smoothed_strength
                * (1.0 - alpha)
                + hand_strength * alpha
            )

            should_grab = (
                curled_fingers
                >= self.config.minimum_curled_fingers
                and (
                    not self.config.require_thumb
                    or thumb_score
                    >= self.config.finger_curl_threshold
                )
            )

            previous_grabbed = (
                self._grabbed
            )

            just_grabbed = False
            just_released = False

            # --------------------------------------------------------------
            # STATE MACHINE
            # --------------------------------------------------------------

            if not self._grabbed:

                if should_grab:

                    self._closing_count += 1

                    self._releasing_count = 0

                    self._state = (
                        GrabState.CLOSING
                    )

                    if (
                        self._closing_count
                        >= max(
                            1,
                            self.config.activation_frames,
                        )
                        and (
                            now
                            - self._last_transition_time
                            >= self.config.cooldown
                        )
                    ):

                        self._grabbed = True

                        self._state = (
                            GrabState.GRABBED
                        )

                        self._grab_start_time = now

                        self._last_transition_time = now

                        just_grabbed = True

                else:

                    self._closing_count = 0

                    self._releasing_count = 0

                    self._state = (
                        GrabState.OPEN
                    )

            else:

                if should_grab:

                    self._releasing_count = 0

                    self._state = (
                        GrabState.GRABBED
                    )

                else:

                    self._releasing_count += 1

                    self._closing_count = 0

                    self._state = (
                        GrabState.RELEASING
                    )

                    if (
                        self._releasing_count
                        >= max(
                            1,
                            self.config.release_frames,
                        )
                    ):

                        self._grabbed = False

                        self._state = (
                            GrabState.OPEN
                        )

                        self._grab_start_time = None

                        self._last_transition_time = now

                        just_released = True

            # --------------------------------------------------------------
            # PALM CENTER
            # --------------------------------------------------------------

            palm = self._calculate_palm_center(
                points
            )

            duration = 0.0

            if (
                self._grabbed
                and self._grab_start_time
                is not None
            ):

                duration = max(
                    0.0,
                    now
                    - self._grab_start_time,
                )

            # --------------------------------------------------------------
            # RESULT
            # --------------------------------------------------------------

            result = GrabStateResult(
                state=self._state,
                grabbed=self._grabbed,
                just_grabbed=just_grabbed,
                just_released=just_released,
                confidence=self._clamp(
                    confidence
                    * (
                        0.70
                        + 0.30
                        * self._smoothed_strength
                    ),
                    0.0,
                    1.0,
                ),
                grab_strength=(
                    self._smoothed_strength
                ),
                curled_fingers=curled_fingers,
                finger_scores=finger_scores,
                thumb_score=thumb_score,
                thumb_contact=thumb_contact,
                palm_center_x=palm.x,
                palm_center_y=palm.y,
                palm_center_z=palm.z,
                duration=duration,
                timestamp=now,
                hand_id=self.hand_id,
                metadata={
                    "input_confidence": confidence,
                    "previous_grabbed": previous_grabbed,
                },
            )

            self._last_finger_scores = (
                finger_scores
            )

            self._last_result = result

            if (
                just_grabbed
                or just_released
            ):

                self._emit(result)

            return result

    # ========================================================================
    # FINGER CURL
    # ========================================================================

    def _calculate_finger_scores(
        self,
        points: Sequence[GrabPoint],
    ) -> dict[str, float]:

        palm_center = (
            self._calculate_palm_center(
                points
            )
        )

        scores: dict[str, float] = {}

        for name, indices in (
            self.FINGER_INDICES.items()
        ):

            mcp_i, pip_i, dip_i, tip_i = (
                indices
            )

            mcp = points[mcp_i]
            pip = points[pip_i]
            dip = points[dip_i]
            tip = points[tip_i]

            palm_distance = self._distance(
                palm_center,
                mcp,
            )

            if palm_distance < 1e-6:

                scores[name] = 0.0

                continue

            tip_distance = self._distance(
                palm_center,
                tip,
            )

            # Angle-based curl.
            angle_1 = self._angle(
                mcp,
                pip,
                dip,
            )

            angle_2 = self._angle(
                pip,
                dip,
                tip,
            )

            # Straight finger -> approximately 180 degrees.
            # Curled finger -> substantially lower.
            angle_score_1 = self._normalize_angle_curl(
                angle_1
            )

            angle_score_2 = self._normalize_angle_curl(
                angle_2
            )

            # Distance-based curl.
            distance_ratio = (
                tip_distance
                / max(
                    palm_distance,
                    1e-6,
                )
            )

            distance_score = self._clamp(
                1.0
                - (
                    distance_ratio
                    / self.config.palm_distance_ratio
                ),
                0.0,
                1.0,
            )

            score = (
                angle_score_1 * 0.30
                + angle_score_2 * 0.30
                + distance_score * 0.40
            )

            scores[name] = self._clamp(
                score,
                0.0,
                1.0,
            )

        return scores

    def _calculate_thumb_score(
        self,
        points: Sequence[GrabPoint],
    ) -> float:

        thumb_mcp = points[
            THUMB_MCP
        ]

        thumb_ip = points[
            THUMB_IP
        ]

        thumb_tip = points[
            THUMB_TIP
        ]

        index_mcp = points[
            INDEX_MCP
        ]

        index_tip = points[
            INDEX_TIP
        ]

        angle = self._angle(
            thumb_mcp,
            thumb_ip,
            thumb_tip,
        )

        angle_score = (
            self._normalize_angle_curl(
                angle
            )
        )

        tip_to_index = self._distance(
            thumb_tip,
            index_tip,
        )

        palm_scale = self._distance(
            thumb_mcp,
            index_mcp,
        )

        ratio = (
            tip_to_index
            / max(
                palm_scale,
                1e-6,
            )
        )

        contact_score = self._clamp(
            1.0
            - ratio
            / max(
                self.config.thumb_contact_threshold,
                1e-6,
            ),
            0.0,
            1.0,
        )

        return self._clamp(
            angle_score * 0.45
            + contact_score * 0.55,
            0.0,
            1.0,
        )

    def _calculate_thumb_contact(
        self,
        points: Sequence[GrabPoint],
    ) -> float:

        thumb_tip = points[
            THUMB_TIP
        ]

        index_tip = points[
            INDEX_TIP
        ]

        palm_center = (
            self._calculate_palm_center(
                points
            )
        )

        scale = self._distance(
            palm_center,
            points[INDEX_MCP],
        )

        if scale <= 1e-6:

            return float(
                "inf"
            )

        return (
            self._distance(
                thumb_tip,
                index_tip,
            )
            / scale
        )

    # ========================================================================
    # PALM
    # ========================================================================

    @staticmethod
    def _calculate_palm_center(
        points: Sequence[GrabPoint],
    ) -> GrabPoint:

        indices = (
            WRIST,
            INDEX_MCP,
            MIDDLE_MCP,
            RING_MCP,
            PINKY_MCP,
        )

        selected = [
            points[i]
            for i in indices
        ]

        return GrabPoint(
            x=sum(
                p.x for p in selected
            )
            / len(selected),
            y=sum(
                p.y for p in selected
            )
            / len(selected),
            z=sum(
                p.z for p in selected
            )
            / len(selected),
        )

    # ========================================================================
    # GEOMETRY
    # ========================================================================

    @staticmethod
    def _distance(
        a: GrabPoint,
        b: GrabPoint,
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
    def _angle(
        a: GrabPoint,
        b: GrabPoint,
        c: GrabPoint,
    ) -> float:

        ba = (
            a.x - b.x,
            a.y - b.y,
            a.z - b.z,
        )

        bc = (
            c.x - b.x,
            c.y - b.y,
            c.z - b.z,
        )

        dot = (
            ba[0] * bc[0]
            + ba[1] * bc[1]
            + ba[2] * bc[2]
        )

        mag_ba = math.sqrt(
            ba[0] ** 2
            + ba[1] ** 2
            + ba[2] ** 2
        )

        mag_bc = math.sqrt(
            bc[0] ** 2
            + bc[1] ** 2
            + bc[2] ** 2
        )

        denominator = (
            mag_ba * mag_bc
        )

        if denominator <= 1e-9:

            return 180.0

        cosine = dot / denominator

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

    @staticmethod
    def _normalize_angle_curl(
        angle: float,
    ) -> float:

        # ~180° = straight
        # ~60°  = strongly curled
        straight = 180.0
        curled = 60.0

        if angle >= straight:
            return 0.0

        if angle <= curled:
            return 1.0

        return (
            straight - angle
        ) / (
            straight - curled
        )

    # ========================================================================
    # RESULT / STATE
    # ========================================================================

    def _invalid_result(
        self,
        timestamp: float,
        confidence: float,
    ) -> GrabStateResult:

        return GrabStateResult(
            state=(
                GrabState.GRABBED
                if self._grabbed
                else GrabState.OPEN
            ),
            grabbed=self._grabbed,
            confidence=confidence,
            grab_strength=(
                self._smoothed_strength
            ),
            timestamp=timestamp,
            hand_id=self.hand_id,
        )

    def is_grabbed(self) -> bool:

        with self._lock:

            return self._grabbed

    @property
    def grabbed(self) -> bool:

        return self.is_grabbed()

    def get_state(self) -> GrabState:

        with self._lock:

            return self._state

    def get_strength(self) -> float:

        with self._lock:

            return self._smoothed_strength

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
                not self._grabbed
                or self._grab_start_time
                is None
            ):

                return 0.0

            return max(
                0.0,
                now
                - self._grab_start_time,
            )

    def get_last_result(
        self,
    ) -> GrabStateResult:

        with self._lock:

            return self._last_result

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def subscribe(
        self,
        callback: Callable[
            [GrabStateResult],
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
            [GrabStateResult],
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
        result: GrabStateResult,
    ) -> None:

        callbacks = list(
            self._callbacks
        )

        for callback in callbacks:

            try:

                callback(result)

            except Exception:

                logger.exception(
                    "Grab callback failed"
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
        finger_threshold: Optional[
            float
        ] = None,
        minimum_curled_fingers: Optional[
            int
        ] = None,
    ) -> None:

        with self._lock:

            if finger_threshold is not None:

                self.config.finger_curl_threshold = (
                    self._clamp(
                        finger_threshold,
                        0.0,
                        1.0,
                    )
                )

            if (
                minimum_curled_fingers
                is not None
            ):

                self.config.minimum_curled_fingers = (
                    max(
                        1,
                        min(
                            4,
                            int(
                                minimum_curled_fingers
                            ),
                        ),
                    )
                )

    def get_config(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "finger_curl_threshold": (
                    self.config.finger_curl_threshold
                ),
                "thumb_contact_threshold": (
                    self.config.thumb_contact_threshold
                ),
                "palm_distance_ratio": (
                    self.config.palm_distance_ratio
                ),
                "activation_frames": (
                    self.config.activation_frames
                ),
                "release_frames": (
                    self.config.release_frames
                ),
                "confidence_threshold": (
                    self.config.confidence_threshold
                ),
                "smoothing": (
                    self.config.smoothing
                ),
                "cooldown": (
                    self.config.cooldown
                ),
                "enabled": (
                    self.config.enabled
                ),
                "require_thumb": (
                    self.config.require_thumb
                ),
                "minimum_curled_fingers": (
                    self.config.minimum_curled_fingers
                ),
                "hand_id": self.hand_id,
            }

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:

        with self._lock:

            self._grabbed = False

            self._state = GrabState.OPEN

            self._closing_count = 0

            self._releasing_count = 0

            self._grab_start_time = None

            self._last_transition_time = 0.0

            self._smoothed_strength = 0.0

            self._last_result = (
                GrabStateResult()
            )

            self._last_finger_scores = {}

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _normalize_landmarks(
        landmarks: Sequence[
            GrabPoint | Sequence[float]
        ],
    ) -> list[GrabPoint]:

        points: list[GrabPoint] = []

        for landmark in landmarks:

            if isinstance(
                landmark,
                GrabPoint,
            ):

                points.append(landmark)

                continue

            if not isinstance(
                landmark,
                (tuple, list),
            ):

                raise TypeError(
                    "Each landmark must be "
                    "GrabPoint, tuple, or list"
                )

            if len(landmark) < 2:

                raise ValueError(
                    "Each landmark requires x and y"
                )

            x = float(
                landmark[0]
            )

            y = float(
                landmark[1]
            )

            z = (
                float(landmark[2])
                if len(landmark) >= 3
                else 0.0
            )

            points.append(
                GrabPoint(
                    x=x,
                    y=y,
                    z=z,
                )
            )

        return points

    @staticmethod
    def _average(
        values: list[float],
    ) -> float:

        if not values:

            return 0.0

        return sum(values) / len(values)

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
# MULTI-HAND GRAB DETECTOR
# ============================================================================


class MultiHandGrabDetector:
    """
    Maintains one GrabDetector per hand.
    """

    def __init__(
        self,
        config: Optional[GrabConfig] = None,
    ) -> None:

        self.config = (
            config
            or GrabConfig()
        )

        self._lock = threading.RLock()

        self._detectors: dict[
            str,
            GrabDetector,
        ] = {}

    def get_detector(
        self,
        hand_id: str,
    ) -> GrabDetector:

        with self._lock:

            if hand_id not in self._detectors:

                self._detectors[
                    hand_id
                ] = GrabDetector(
                    self.config,
                    hand_id=hand_id,
                )

            return self._detectors[
                hand_id
            ]

    def update(
        self,
        hand_id: str,
        landmarks: Sequence[
            GrabPoint | Sequence[float]
        ],
        *,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> GrabStateResult:

        detector = self.get_detector(
            hand_id
        )

        return detector.update(
            landmarks,
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

    def get_grabbed_hands(
        self,
    ) -> list[str]:

        with self._lock:

            return [
                hand_id
                for (
                    hand_id,
                    detector,
                ) in self._detectors.items()
                if detector.is_grabbed()
            ]


# ============================================================================
# CONVENIENCE API
# ============================================================================


_default_detector: Optional[
    GrabDetector
] = None

_default_lock = threading.RLock()


def get_grab_detector() -> GrabDetector:

    global _default_detector

    with _default_lock:

        if _default_detector is None:

            _default_detector = (
                GrabDetector()
            )

        return _default_detector


def detect_grab(
    landmarks: Sequence[
        GrabPoint | Sequence[float]
    ],
    *,
    confidence: float = 1.0,
    timestamp: Optional[float] = None,
) -> GrabStateResult:

    return get_grab_detector().update(
        landmarks,
        confidence=confidence,
        timestamp=timestamp,
    )


def is_grab(
    landmarks: Sequence[
        GrabPoint | Sequence[float]
    ],
) -> bool:

    detector = get_grab_detector()

    result = detector.update(
        landmarks
    )

    return result.grabbed


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "GrabPoint",
    "GrabConfig",
    "GrabState",
    "GrabStateResult",
    "GrabDetector",
    "MultiHandGrabDetector",
    "get_grab_detector",
    "detect_grab",
    "is_grab",
]


