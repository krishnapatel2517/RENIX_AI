"""
RENIX - Palm Tracking & Gesture Utilities
=========================================

Provides palm-center, palm-size, orientation and stability calculations
for the RENIX gesture system.

Designed to work with MediaPipe-style 21 hand landmarks.

Landmark layout:
    0  = wrist
    1  = thumb CMC
    2  = thumb MCP
    3  = thumb IP
    4  = thumb tip

    5  = index MCP
    6  = index PIP
    7  = index DIP
    8  = index tip

    9  = middle MCP
    10 = middle PIP
    11 = middle DIP
    12 = middle tip

    13 = ring MCP
    14 = ring PIP
    15 = ring DIP
    16 = ring tip

    17 = pinky MCP
    18 = pinky PIP
    19 = pinky DIP
    20 = pinky tip
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence


logger = logging.getLogger("RENIX.gestures.palm")


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass(frozen=True)
class PalmPoint:
    """Normalized 3D point."""

    x: float
    y: float
    z: float = 0.0

    def distance_to(
        self,
        other: "PalmPoint",
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
class PalmMetrics:
    """Calculated palm measurements."""

    center: PalmPoint = field(
        default_factory=lambda: PalmPoint(0.0, 0.0, 0.0)
    )

    wrist: PalmPoint = field(
        default_factory=lambda: PalmPoint(0.0, 0.0, 0.0)
    )

    index_mcp: PalmPoint = field(
        default_factory=lambda: PalmPoint(0.0, 0.0, 0.0)
    )

    middle_mcp: PalmPoint = field(
        default_factory=lambda: PalmPoint(0.0, 0.0, 0.0)
    )

    ring_mcp: PalmPoint = field(
        default_factory=lambda: PalmPoint(0.0, 0.0, 0.0)
    )

    pinky_mcp: PalmPoint = field(
        default_factory=lambda: PalmPoint(0.0, 0.0, 0.0)
    )

    width: float = 0.0
    height: float = 0.0
    depth: float = 0.0

    area: float = 0.0

    orientation: float = 0.0

    tilt: float = 0.0

    spread: float = 0.0

    confidence: float = 0.0

    timestamp: float = 0.0

    valid: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": {
                "x": self.center.x,
                "y": self.center.y,
                "z": self.center.z,
            },
            "wrist": {
                "x": self.wrist.x,
                "y": self.wrist.y,
                "z": self.wrist.z,
            },
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "area": self.area,
            "orientation": self.orientation,
            "tilt": self.tilt,
            "spread": self.spread,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "valid": self.valid,
            "metadata": dict(self.metadata),
        }


@dataclass
class PalmConfig:
    """Configuration for palm calculations."""

    smoothing: float = 0.35

    minimum_confidence: float = 0.35

    minimum_landmarks: int = 21

    stability_threshold: float = 0.015

    orientation_smoothing: float = 0.30

    enabled: bool = True


# ============================================================================
# PALM TRACKER
# ============================================================================


class PalmTracker:
    """
    Tracks the palm of one hand.

    The tracker is intentionally independent from the camera or MediaPipe.
    Any tracker producing 21 compatible landmarks can feed this class.
    """

    # MediaPipe landmark indices.
    WRIST = 0

    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    THUMB_MCP = 2
    THUMB_TIP = 4

    def __init__(
        self,
        config: Optional[PalmConfig] = None,
    ) -> None:

        self.config = (
            config
            or PalmConfig()
        )

        self._lock = threading.RLock()

        self._previous_center: Optional[
            PalmPoint
        ] = None

        self._smoothed_center: Optional[
            PalmPoint
        ] = None

        self._previous_orientation: Optional[
            float
        ] = None

        self._smoothed_orientation: Optional[
            float
        ] = None

        self._last_metrics = PalmMetrics()

        self._last_timestamp = 0.0

        self._velocity = PalmPoint(
            0.0,
            0.0,
            0.0,
        )

        self._stable = False

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        landmarks: Sequence[
            PalmPoint | Sequence[float]
        ],
        *,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> PalmMetrics:

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

                return PalmMetrics(
                    timestamp=now
                )

            if (
                confidence
                < self.config.minimum_confidence
            ):

                return PalmMetrics(
                    timestamp=now,
                    confidence=confidence,
                )

            if (
                len(landmarks)
                < self.config.minimum_landmarks
            ):

                return PalmMetrics(
                    timestamp=now,
                    confidence=confidence,
                )

            points = self._normalize_landmarks(
                landmarks
            )

            if points is None:

                return PalmMetrics(
                    timestamp=now,
                    confidence=confidence,
                )

            wrist = points[self.WRIST]

            index_mcp = points[
                self.INDEX_MCP
            ]

            middle_mcp = points[
                self.MIDDLE_MCP
            ]

            ring_mcp = points[
                self.RING_MCP
            ]

            pinky_mcp = points[
                self.PINKY_MCP
            ]

            center = self._calculate_center(
                wrist,
                index_mcp,
                middle_mcp,
                ring_mcp,
                pinky_mcp,
            )

            # --------------------------------------------------------------
            # SMOOTH CENTER
            # --------------------------------------------------------------

            if self._smoothed_center is None:

                self._smoothed_center = center

            else:

                self._smoothed_center = (
                    self._lerp_point(
                        self._smoothed_center,
                        center,
                        self.config.smoothing,
                    )
                )

            # --------------------------------------------------------------
            # VELOCITY
            # --------------------------------------------------------------

            dt = max(
                now - self._last_timestamp,
                1e-6,
            )

            if self._previous_center is not None:

                self._velocity = PalmPoint(
                    (
                        self._smoothed_center.x
                        - self._previous_center.x
                    ) / dt,

                    (
                        self._smoothed_center.y
                        - self._previous_center.y
                    ) / dt,

                    (
                        self._smoothed_center.z
                        - self._previous_center.z
                    ) / dt,
                )

            self._previous_center = (
                self._smoothed_center
            )

            # --------------------------------------------------------------
            # PALM WIDTH
            # --------------------------------------------------------------

            width = self._distance(
                index_mcp,
                pinky_mcp,
            )

            # --------------------------------------------------------------
            # PALM HEIGHT
            # --------------------------------------------------------------

            palm_top = self._midpoint(
                index_mcp,
                pinky_mcp,
            )

            height = self._distance(
                wrist,
                palm_top,
            )

            # --------------------------------------------------------------
            # DEPTH
            # --------------------------------------------------------------

            depth = self._calculate_depth(
                wrist,
                index_mcp,
                middle_mcp,
                ring_mcp,
                pinky_mcp,
            )

            # --------------------------------------------------------------
            # AREA
            # --------------------------------------------------------------

            area = (
                width
                * height
            )

            # --------------------------------------------------------------
            # ORIENTATION
            # --------------------------------------------------------------

            raw_orientation = (
                self._calculate_orientation(
                    index_mcp,
                    pinky_mcp,
                )
            )

            if (
                self._smoothed_orientation
                is None
            ):

                self._smoothed_orientation = (
                    raw_orientation
                )

            else:

                orientation_delta = (
                    self._angle_difference(
                        raw_orientation,
                        self._smoothed_orientation,
                    )
                )

                self._smoothed_orientation = (
                    self._smoothed_orientation
                    + orientation_delta
                    * self.config.orientation_smoothing
                )

            self._previous_orientation = (
                raw_orientation
            )

            # --------------------------------------------------------------
            # TILT
            # --------------------------------------------------------------

            tilt = self._calculate_tilt(
                wrist,
                middle_mcp,
            )

            # --------------------------------------------------------------
            # FINGER SPREAD
            # --------------------------------------------------------------

            spread = self._calculate_spread(
                points
            )

            # --------------------------------------------------------------
            # STABILITY
            # --------------------------------------------------------------

            movement = 0.0

            if self._previous_center is not None:

                movement = self._distance(
                    center,
                    self._previous_center,
                )

            self._stable = (
                movement
                <= self.config.stability_threshold
            )

            metrics = PalmMetrics(
                center=self._smoothed_center,
                wrist=wrist,
                index_mcp=index_mcp,
                middle_mcp=middle_mcp,
                ring_mcp=ring_mcp,
                pinky_mcp=pinky_mcp,
                width=width,
                height=height,
                depth=depth,
                area=area,
                orientation=(
                    self._smoothed_orientation
                    or 0.0
                ),
                tilt=tilt,
                spread=spread,
                confidence=confidence,
                timestamp=now,
                valid=True,
                metadata={
                    "velocity": {
                        "x": self._velocity.x,
                        "y": self._velocity.y,
                        "z": self._velocity.z,
                    },
                    "stable": self._stable,
                    "movement": movement,
                },
            )

            self._last_metrics = metrics

            self._last_timestamp = now

            return metrics

    # ========================================================================
    # CENTER
    # ========================================================================

    @staticmethod
    def _calculate_center(
        wrist: PalmPoint,
        index_mcp: PalmPoint,
        middle_mcp: PalmPoint,
        ring_mcp: PalmPoint,
        pinky_mcp: PalmPoint,
    ) -> PalmPoint:

        points = (
            wrist,
            index_mcp,
            middle_mcp,
            ring_mcp,
            pinky_mcp,
        )

        return PalmPoint(
            x=sum(p.x for p in points)
            / len(points),

            y=sum(p.y for p in points)
            / len(points),

            z=sum(p.z for p in points)
            / len(points),
        )

    # ========================================================================
    # ORIENTATION
    # ========================================================================

    @staticmethod
    def _calculate_orientation(
        index_mcp: PalmPoint,
        pinky_mcp: PalmPoint,
    ) -> float:

        dx = (
            pinky_mcp.x
            - index_mcp.x
        )

        dy = (
            pinky_mcp.y
            - index_mcp.y
        )

        return math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )

    @staticmethod
    def _calculate_tilt(
        wrist: PalmPoint,
        middle_mcp: PalmPoint,
    ) -> float:

        dx = (
            middle_mcp.x
            - wrist.x
        )

        dy = (
            middle_mcp.y
            - wrist.y
        )

        return math.degrees(
            math.atan2(
                dx,
                -dy,
            )
        )

    # ========================================================================
    # DIMENSIONS
    # ========================================================================

    @staticmethod
    def _calculate_depth(
        *points: PalmPoint,
    ) -> float:

        if not points:

            return 0.0

        z_values = [
            point.z
            for point in points
        ]

        return (
            max(z_values)
            - min(z_values)
        )

    @staticmethod
    def _calculate_spread(
        points: Sequence[PalmPoint],
    ) -> float:

        if len(points) < 21:

            return 0.0

        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]

        index_mcp = points[5]
        middle_mcp = points[9]
        ring_mcp = points[13]
        pinky_mcp = points[17]

        finger_distances = (
            PalmTracker._distance(
                index_tip,
                index_mcp,
            ),
            PalmTracker._distance(
                middle_tip,
                middle_mcp,
            ),
            PalmTracker._distance(
                ring_tip,
                ring_mcp,
            ),
            PalmTracker._distance(
                pinky_tip,
                pinky_mcp,
            ),
        )

        return sum(
            finger_distances
        ) / len(
            finger_distances
        )

    # ========================================================================
    # VELOCITY / STABILITY
    # ========================================================================

    def get_velocity(self) -> PalmPoint:

        with self._lock:

            return self._velocity

    def get_speed(self) -> float:

        with self._lock:

            return math.sqrt(
                self._velocity.x ** 2
                + self._velocity.y ** 2
                + self._velocity.z ** 2
            )

    def is_stable(self) -> bool:

        with self._lock:

            return self._stable

    # ========================================================================
    # STATE
    # ========================================================================

    def get_metrics(self) -> PalmMetrics:

        with self._lock:

            return self._last_metrics

    def get_center(self) -> PalmPoint:

        with self._lock:

            return self._last_metrics.center

    def get_orientation(self) -> float:

        with self._lock:

            return (
                self._last_metrics.orientation
            )

    def get_size(self) -> float:

        with self._lock:

            return max(
                self._last_metrics.width,
                self._last_metrics.height,
            )

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self) -> None:

        with self._lock:

            self._previous_center = None

            self._smoothed_center = None

            self._previous_orientation = None

            self._smoothed_orientation = None

            self._last_metrics = (
                PalmMetrics()
            )

            self._last_timestamp = 0.0

            self._velocity = PalmPoint(
                0.0,
                0.0,
                0.0,
            )

            self._stable = False

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _normalize_landmarks(
        landmarks: Sequence[
            PalmPoint | Sequence[float]
        ],
    ) -> Optional[list[PalmPoint]]:

        result: list[PalmPoint] = []

        try:

            for landmark in landmarks:

                if isinstance(
                    landmark,
                    PalmPoint,
                ):

                    result.append(landmark)

                    continue

                if len(landmark) < 2:

                    return None

                result.append(
                    PalmPoint(
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
                "Invalid palm landmarks"
            )

            return None

        return result

    @staticmethod
    def _distance(
        a: PalmPoint,
        b: PalmPoint,
    ) -> float:

        return a.distance_to(b)

    @staticmethod
    def _midpoint(
        a: PalmPoint,
        b: PalmPoint,
    ) -> PalmPoint:

        return PalmPoint(
            x=(a.x + b.x) / 2.0,
            y=(a.y + b.y) / 2.0,
            z=(a.z + b.z) / 2.0,
        )

    @staticmethod
    def _lerp_point(
        a: PalmPoint,
        b: PalmPoint,
        amount: float,
    ) -> PalmPoint:

        amount = max(
            0.0,
            min(1.0, amount),
        )

        return PalmPoint(
            x=a.x + (b.x - a.x) * amount,
            y=a.y + (b.y - a.y) * amount,
            z=a.z + (b.z - a.z) * amount,
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
# CONVENIENCE FUNCTIONS
# ============================================================================


_default_tracker: Optional[
    PalmTracker
] = None

_default_lock = threading.RLock()


def get_palm_tracker() -> PalmTracker:

    global _default_tracker

    with _default_lock:

        if _default_tracker is None:

            _default_tracker = (
                PalmTracker()
            )

        return _default_tracker


def calculate_palm(
    landmarks: Sequence[
        PalmPoint | Sequence[float]
    ],
    *,
    confidence: float = 1.0,
    timestamp: Optional[float] = None,
) -> PalmMetrics:

    return get_palm_tracker().update(
        landmarks,
        confidence=confidence,
        timestamp=timestamp,
    )


def palm_center(
    landmarks: Sequence[
        PalmPoint | Sequence[float]
    ],
) -> PalmPoint:

    points = (
        PalmTracker._normalize_landmarks(
            landmarks
        )
    )

    if not points or len(points) < 21:

        raise ValueError(
            "21 valid hand landmarks are required"
        )

    return PalmTracker._calculate_center(
        points[0],
        points[5],
        points[9],
        points[13],
        points[17],
    )


def palm_distance(
    a: PalmPoint | Sequence[float],
    b: PalmPoint | Sequence[float],
) -> float:

    def normalize(
        point: PalmPoint | Sequence[float],
    ) -> PalmPoint:

        if isinstance(
            point,
            PalmPoint,
        ):

            return point

        if len(point) < 2:

            raise ValueError(
                "Point requires x and y"
            )

        return PalmPoint(
            float(point[0]),
            float(point[1]),
            (
                float(point[2])
                if len(point) >= 3
                else 0.0
            ),
        )

    return normalize(a).distance_to(
        normalize(b)
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "PalmPoint",
    "PalmMetrics",
    "PalmConfig",
    "PalmTracker",
    "get_palm_tracker",
    "calculate_palm",
    "palm_center",
    "palm_distance",
]


