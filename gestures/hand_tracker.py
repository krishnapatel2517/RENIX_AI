"""
RENIX Hand Tracker
==================

Tracks hands across video frames and provides stable hand landmarks
to the RENIX gesture engine.

Responsibilities:
    - Accept hand landmarks from a vision backend.
    - Normalize MediaPipe-style / generic landmark formats.
    - Assign stable hand IDs.
    - Track left/right hands.
    - Smooth landmark movement.
    - Estimate hand velocity.
    - Handle temporary detection loss.
    - Maintain tracking confidence.
    - Provide clean data to gesture_engine.py.

This module does NOT decide what a gesture means.
That responsibility belongs to gesture_engine.py and gesture_classifier.py.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger("RENIX.gestures.hand_tracker")


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class Landmark:
    """Normalized hand landmark."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def distance_to(
        self,
        other: "Landmark",
    ) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def lerp(
        self,
        other: "Landmark",
        alpha: float,
    ) -> "Landmark":
        alpha = max(
            0.0,
            min(1.0, alpha),
        )

        return Landmark(
            x=self.x
            + (other.x - self.x) * alpha,
            y=self.y
            + (other.y - self.y) * alpha,
            z=self.z
            + (other.z - self.z) * alpha,
            visibility=self.visibility
            + (
                other.visibility
                - self.visibility
            )
            * alpha,
        )

    def to_dict(
        self,
    ) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
        }


@dataclass
class TrackedHand:
    """Complete tracked hand state."""

    hand_id: str

    side: str = "unknown"

    confidence: float = 0.0

    landmarks: list[Landmark] = field(
        default_factory=list
    )

    raw_landmarks: list[Landmark] = field(
        default_factory=list
    )

    wrist: Optional[Landmark] = None

    palm: Optional[Landmark] = None

    center: Optional[Landmark] = None

    velocity_x: float = 0.0

    velocity_y: float = 0.0

    velocity_z: float = 0.0

    speed: float = 0.0

    age: int = 0

    missed_frames: int = 0

    first_seen: float = 0.0

    last_seen: float = 0.0

    is_visible: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "hand_id": self.hand_id,
            "side": self.side,
            "confidence": self.confidence,
            "landmarks": [
                point.to_dict()
                for point in self.landmarks
            ],
            "wrist": (
                self.wrist.to_dict()
                if self.wrist
                else None
            ),
            "palm": (
                self.palm.to_dict()
                if self.palm
                else None
            ),
            "center": (
                self.center.to_dict()
                if self.center
                else None
            ),
            "velocity": {
                "x": self.velocity_x,
                "y": self.velocity_y,
                "z": self.velocity_z,
            },
            "speed": self.speed,
            "age": self.age,
            "missed_frames": self.missed_frames,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "is_visible": self.is_visible,
            "metadata": dict(self.metadata),
        }


@dataclass
class HandTrackerConfig:
    """Configuration for hand tracking."""

    enabled: bool = True

    max_hands: int = 2

    min_confidence: float = 0.45

    smoothing_alpha: float = 0.55

    velocity_smoothing_alpha: float = 0.40

    max_match_distance: float = 0.22

    max_missed_frames: int = 8

    history_size: int = 20

    prediction_enabled: bool = True

    prediction_strength: float = 0.45

    normalize_coordinates: bool = True

    mirror_x: bool = False

    use_depth: bool = True

    stable_side_assignment: bool = True


# ============================================================================
# HAND TRACKER
# ============================================================================


class HandTracker:
    """
    Stable multi-hand tracker for RENIX.

    Typical usage:

        tracker = HandTracker()
        tracker.start()

        hands = tracker.update(
            detector_output
        )

        for hand in hands:
            print(hand.hand_id)

    The output can be converted directly into structures accepted by
    RENIX gesture_engine.py.
    """

    def __init__(
        self,
        config: Optional[
            HandTrackerConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or HandTrackerConfig()
        )

        self._lock = threading.RLock()

        self._running = False

        self._frame_number = 0

        self._next_hand_number = 1

        self._tracks: dict[
            str,
            TrackedHand,
        ] = {}

        self._previous_tracks: dict[
            str,
            TrackedHand,
        ] = {}

        self._history: dict[
            str,
            deque[Landmark],
        ] = {}

        self._last_update_time = (
            time.monotonic()
        )

        self._fps = 0.0

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

    def reset(self) -> None:

        with self._lock:

            self._tracks.clear()

            self._previous_tracks.clear()

            self._history.clear()

            self._frame_number = 0

            self._next_hand_number = 1

            self._fps = 0.0

    def is_running(self) -> bool:

        with self._lock:

            return self._running

    # ========================================================================
    # MAIN UPDATE
    # ========================================================================

    def update(
        self,
        detections: Any,
        *,
        timestamp: Optional[float] = None,
    ) -> list[TrackedHand]:

        if not self.config.enabled:

            return []

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        with self._lock:

            self._frame_number += 1

            self._update_fps()

            normalized = (
                self._normalize_detections(
                    detections
                )
            )

            self._previous_tracks = {
                key: self._copy_track(value)
                for key, value in (
                    self._tracks.items()
                )
            }

            matches, unmatched_detections, unmatched_tracks = (
                self._match_detections(
                    normalized
                )
            )

            updated_tracks: dict[
                str,
                TrackedHand,
            ] = {}

            # ------------------------------------------------------------
            # Update matched tracks.
            # ------------------------------------------------------------

            for (
                track_id,
                detection_index,
            ) in matches:

                detection = normalized[
                    detection_index
                ]

                previous = self._tracks[
                    track_id
                ]

                updated = (
                    self._update_track(
                        previous,
                        detection,
                        now,
                    )
                )

                updated_tracks[
                    track_id
                ] = updated

            # ------------------------------------------------------------
            # Create new tracks.
            # ------------------------------------------------------------

            for index in unmatched_detections:

                detection = normalized[
                    index
                ]

                new_track = (
                    self._create_track(
                        detection,
                        now,
                    )
                )

                updated_tracks[
                    new_track.hand_id
                ] = new_track

            # ------------------------------------------------------------
            # Handle lost tracks.
            # ------------------------------------------------------------

            for track_id in unmatched_tracks:

                previous = self._tracks[
                    track_id
                ]

                lost = (
                    self._update_lost_track(
                        previous,
                        now,
                    )
                )

                if lost is not None:

                    updated_tracks[
                        track_id
                    ] = lost

                else:

                    self._history.pop(
                        track_id,
                        None,
                    )

            self._tracks = updated_tracks

            return [
                self._copy_track(hand)
                for hand in self._tracks.values()
                if hand.is_visible
            ]

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    def _normalize_detections(
        self,
        detections: Any,
    ) -> list[dict[str, Any]]:

        if detections is None:

            return []

        # ------------------------------------------------------------
        # Common detector wrappers.
        # ------------------------------------------------------------

        if isinstance(
            detections,
            dict,
        ):

            if "hands" in detections:

                detections = detections[
                    "hands"
                ]

            elif "detections" in detections:

                detections = detections[
                    "detections"
                ]

            elif "landmarks" in detections:

                detections = [detections]

        if not isinstance(
            detections,
            (list, tuple),
        ):

            detections = [detections]

        result: list[
            dict[str, Any]
        ] = []

        for raw in detections:

            try:

                detection = (
                    self._normalize_detection(
                        raw
                    )
                )

                if detection[
                    "confidence"
                ] < self.config.min_confidence:

                    continue

                result.append(
                    detection
                )

            except Exception:

                logger.exception(
                    "Failed to normalize hand detection."
                )

        return result[
            : self.config.max_hands
        ]

    def _normalize_detection(
        self,
        raw: Any,
    ) -> dict[str, Any]:

        if isinstance(
            raw,
            TrackedHand,
        ):

            return {
                "landmarks": list(
                    raw.landmarks
                ),
                "side": raw.side,
                "confidence": raw.confidence,
                "metadata": dict(
                    raw.metadata
                ),
            }

        if isinstance(
            raw,
            dict,
        ):

            landmarks = raw.get(
                "landmarks",
                raw.get(
                    "points",
                    raw.get(
                        "keypoints",
                        [],
                    ),
                ),
            )

            side = raw.get(
                "side",
                raw.get(
                    "handedness",
                    raw.get(
                        "label",
                        "unknown",
                    ),
                ),
            )

            confidence = raw.get(
                "confidence",
                raw.get(
                    "score",
                    raw.get(
                        "handedness_score",
                        1.0,
                    ),
                ),
            )

            metadata = dict(raw)

        else:

            landmarks = getattr(
                raw,
                "landmarks",
                getattr(
                    raw,
                    "points",
                    [],
                ),
            )

            side = getattr(
                raw,
                "side",
                getattr(
                    raw,
                    "handedness",
                    "unknown",
                ),
            )

            confidence = getattr(
                raw,
                "confidence",
                getattr(
                    raw,
                    "score",
                    1.0,
                ),
            )

            metadata = {}

            if hasattr(
                raw,
                "__dict__",
            ):

                metadata = dict(
                    raw.__dict__
                )

        normalized_landmarks = (
            self._normalize_landmarks(
                landmarks
            )
        )

        if self.config.mirror_x:

            normalized_landmarks = [
                Landmark(
                    x=1.0 - point.x,
                    y=point.y,
                    z=point.z,
                    visibility=point.visibility,
                )
                for point in normalized_landmarks
            ]

        return {
            "landmarks": normalized_landmarks,
            "side": self._normalize_side(
                side
            ),
            "confidence": self._safe_float(
                confidence,
                1.0,
            ),
            "metadata": metadata,
        }

    def _normalize_landmarks(
        self,
        landmarks: Any,
    ) -> list[Landmark]:

        if landmarks is None:

            return []

        if isinstance(
            landmarks,
            dict,
        ):

            landmarks = landmarks.get(
                "landmarks",
                [],
            )

        result: list[
            Landmark
        ] = []

        for point in landmarks:

            if isinstance(
                point,
                Landmark,
            ):

                result.append(point)

                continue

            if isinstance(
                point,
                dict,
            ):

                result.append(
                    Landmark(
                        x=self._safe_float(
                            point.get(
                                "x",
                                0.0,
                            )
                        ),
                        y=self._safe_float(
                            point.get(
                                "y",
                                0.0,
                            )
                        ),
                        z=self._safe_float(
                            point.get(
                                "z",
                                0.0,
                            )
                        ),
                        visibility=self._safe_float(
                            point.get(
                                "visibility",
                                point.get(
                                    "confidence",
                                    1.0,
                                ),
                            ),
                            1.0,
                        ),
                    )
                )

                continue

            if isinstance(
                point,
                (list, tuple),
            ):

                if len(point) < 2:
                    continue

                result.append(
                    Landmark(
                        x=self._safe_float(
                            point[0]
                        ),
                        y=self._safe_float(
                            point[1]
                        ),
                        z=(
                            self._safe_float(
                                point[2]
                            )
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

            if x is None or y is None:
                continue

            result.append(
                Landmark(
                    x=self._safe_float(
                        x
                    ),
                    y=self._safe_float(
                        y
                    ),
                    z=self._safe_float(
                        getattr(
                            point,
                            "z",
                            0.0,
                        )
                    ),
                    visibility=self._safe_float(
                        getattr(
                            point,
                            "visibility",
                            1.0,
                        ),
                        1.0,
                    ),
                )
            )

        return result

    # ========================================================================
    # TRACK MATCHING
    # ========================================================================

    def _match_detections(
        self,
        detections: list[
            dict[str, Any]
        ],
    ) -> tuple[
        list[tuple[str, int]],
        list[int],
        list[str],
    ]:

        if not self._tracks:

            return (
                [],
                list(
                    range(
                        len(detections)
                    )
                ),
                [],
            )

        candidates: list[
            tuple[
                float,
                str,
                int,
            ]
        ] = []

        for track_id, track in (
            self._tracks.items()
        ):

            predicted = (
                self._predict_track_position(
                    track
                )
            )

            for index, detection in enumerate(
                detections
            ):

                landmarks = detection[
                    "landmarks"
                ]

                if not landmarks:
                    continue

                center = (
                    self._calculate_center(
                        landmarks
                    )
                )

                distance = (
                    predicted.distance_to(
                        center
                    )
                    if predicted
                    else self._landmark_distance(
                        track.landmarks,
                        landmarks,
                    )
                )

                side_penalty = (
                    self._side_penalty(
                        track.side,
                        detection["side"],
                    )
                )

                total_cost = (
                    distance
                    + side_penalty
                )

                if total_cost <= (
                    self.config.max_match_distance
                ):

                    candidates.append(
                        (
                            total_cost,
                            track_id,
                            index,
                        )
                    )

        candidates.sort(
            key=lambda item: item[0]
        )

        used_tracks: set[
            str
        ] = set()

        used_detections: set[
            int
        ] = set()

        matches: list[
            tuple[str, int]
        ] = []

        for (
            _cost,
            track_id,
            index,
        ) in candidates:

            if track_id in used_tracks:
                continue

            if index in used_detections:
                continue

            used_tracks.add(
                track_id
            )

            used_detections.add(
                index
            )

            matches.append(
                (
                    track_id,
                    index,
                )
            )

        unmatched_detections = [
            index
            for index in range(
                len(detections)
            )
            if index not in used_detections
        ]

        unmatched_tracks = [
            track_id
            for track_id in self._tracks
            if track_id not in used_tracks
        ]

        return (
            matches,
            unmatched_detections,
            unmatched_tracks,
        )

    def _side_penalty(
        self,
        track_side: str,
        detection_side: str,
    ) -> float:

        if (
            not self.config.stable_side_assignment
        ):

            return 0.0

        if (
            track_side == "unknown"
            or detection_side == "unknown"
        ):

            return 0.0

        if track_side == detection_side:

            return 0.0

        return 0.08

    # ========================================================================
    # TRACK CREATION / UPDATE
    # ========================================================================

    def _create_track(
        self,
        detection: dict[str, Any],
        timestamp: float,
    ) -> TrackedHand:

        hand_id = (
            f"hand_{self._next_hand_number}"
        )

        self._next_hand_number += 1

        landmarks = list(
            detection["landmarks"]
        )

        center = (
            self._calculate_center(
                landmarks
            )
            if landmarks
            else None
        )

        palm = (
            self._calculate_palm(
                landmarks
            )
            if landmarks
            else None
        )

        wrist = (
            landmarks[0]
            if landmarks
            else None
        )

        track = TrackedHand(
            hand_id=hand_id,
            side=detection["side"],
            confidence=detection[
                "confidence"
            ],
            landmarks=landmarks,
            raw_landmarks=list(
                landmarks
            ),
            wrist=wrist,
            palm=palm,
            center=center,
            first_seen=timestamp,
            last_seen=timestamp,
            age=1,
            missed_frames=0,
            is_visible=True,
            metadata=detection[
                "metadata"
            ],
        )

        if center:

            self._history[
                hand_id
            ] = deque(
                [center],
                maxlen=self.config.history_size,
            )

        return track

    def _update_track(
        self,
        previous: TrackedHand,
        detection: dict[str, Any],
        timestamp: float,
    ) -> TrackedHand:

        raw_landmarks = list(
            detection["landmarks"]
        )

        smoothed_landmarks = (
            self._smooth_landmarks(
                previous.landmarks,
                raw_landmarks,
            )
        )

        previous_center = (
            previous.center
        )

        center = (
            self._calculate_center(
                smoothed_landmarks
            )
            if smoothed_landmarks
            else previous_center
        )

        palm = (
            self._calculate_palm(
                smoothed_landmarks
            )
            if smoothed_landmarks
            else previous.palm
        )

        wrist = (
            smoothed_landmarks[0]
            if smoothed_landmarks
            else previous.wrist
        )

        velocity = (
            self._calculate_velocity(
                previous_center,
                center,
                timestamp
                - previous.last_seen,
            )
        )

        velocity_x = (
            previous.velocity_x
            + (
                velocity[0]
                - previous.velocity_x
            )
            * self.config.velocity_smoothing_alpha
        )

        velocity_y = (
            previous.velocity_y
            + (
                velocity[1]
                - previous.velocity_y
            )
            * self.config.velocity_smoothing_alpha
        )

        velocity_z = (
            previous.velocity_z
            + (
                velocity[2]
                - previous.velocity_z
            )
            * self.config.velocity_smoothing_alpha
        )

        speed = math.sqrt(
            velocity_x**2
            + velocity_y**2
            + velocity_z**2
        )

        side = (
            detection["side"]
            if detection["side"]
            != "unknown"
            else previous.side
        )

        confidence = (
            previous.confidence
            * 0.25
            + detection[
                "confidence"
            ]
            * 0.75
        )

        updated = TrackedHand(
            hand_id=previous.hand_id,
            side=side,
            confidence=confidence,
            landmarks=smoothed_landmarks,
            raw_landmarks=raw_landmarks,
            wrist=wrist,
            palm=palm,
            center=center,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            velocity_z=velocity_z,
            speed=speed,
            age=previous.age + 1,
            missed_frames=0,
            first_seen=previous.first_seen,
            last_seen=timestamp,
            is_visible=True,
            metadata={
                **previous.metadata,
                **detection[
                    "metadata"
                ],
            },
        )

        if center:

            history = self._history.get(
                previous.hand_id
            )

            if history is None:

                history = deque(
                    maxlen=self.config.history_size
                )

                self._history[
                    previous.hand_id
                ] = history

            history.append(center)

        return updated

    def _update_lost_track(
        self,
        previous: TrackedHand,
        timestamp: float,
    ) -> Optional[
        TrackedHand
    ]:

        missed = (
            previous.missed_frames + 1
        )

        if (
            missed
            > self.config.max_missed_frames
        ):

            return None

        predicted_center = (
            self._predict_track_position(
                previous
            )
        )

        predicted_landmarks = (
            self._predict_landmarks(
                previous
            )
            if self.config.prediction_enabled
            else list(
                previous.landmarks
            )
        )

        confidence = (
            previous.confidence
            * (
                0.80
                ** missed
            )
        )

        return TrackedHand(
            hand_id=previous.hand_id,
            side=previous.side,
            confidence=confidence,
            landmarks=predicted_landmarks,
            raw_landmarks=list(
                previous.raw_landmarks
            ),
            wrist=(
                predicted_landmarks[0]
                if predicted_landmarks
                else previous.wrist
            ),
            palm=(
                self._calculate_palm(
                    predicted_landmarks
                )
                if predicted_landmarks
                else previous.palm
            ),
            center=predicted_center,
            velocity_x=previous.velocity_x,
            velocity_y=previous.velocity_y,
            velocity_z=previous.velocity_z,
            speed=previous.speed,
            age=previous.age + 1,
            missed_frames=missed,
            first_seen=previous.first_seen,
            last_seen=timestamp,
            is_visible=False,
            metadata={
                **previous.metadata,
                "predicted": True,
            },
        )

    # ========================================================================
    # SMOOTHING
    # ========================================================================

    def _smooth_landmarks(
        self,
        previous: list[Landmark],
        current: list[Landmark],
    ) -> list[Landmark]:

        if not previous:

            return list(current)

        if len(previous) != len(current):

            return list(current)

        alpha = (
            self.config.smoothing_alpha
        )

        return [
            old.lerp(
                new,
                alpha,
            )
            for old, new in zip(
                previous,
                current,
            )
        ]

    # ========================================================================
    # PREDICTION
    # ========================================================================

    def _predict_track_position(
        self,
        track: TrackedHand,
    ) -> Optional[Landmark]:

        if track.center is None:

            return None

        if not self.config.prediction_enabled:

            return track.center

        strength = (
            self.config.prediction_strength
        )

        return Landmark(
            x=track.center.x
            + track.velocity_x * strength,
            y=track.center.y
            + track.velocity_y * strength,
            z=track.center.z
            + track.velocity_z * strength,
        )

    def _predict_landmarks(
        self,
        track: TrackedHand,
    ) -> list[Landmark]:

        if not track.landmarks:

            return []

        strength = (
            self.config.prediction_strength
        )

        return [
            Landmark(
                x=point.x
                + track.velocity_x * strength,
                y=point.y
                + track.velocity_y * strength,
                z=point.z
                + track.velocity_z * strength,
                visibility=point.visibility
                * 0.95,
            )
            for point in track.landmarks
        ]

    # ========================================================================
    # GEOMETRY
    # ========================================================================

    def _calculate_center(
        self,
        landmarks: list[Landmark],
    ) -> Landmark:

        if not landmarks:

            return Landmark(
                0.0,
                0.0,
                0.0,
            )

        return Landmark(
            x=sum(
                point.x
                for point in landmarks
            )
            / len(landmarks),
            y=sum(
                point.y
                for point in landmarks
            )
            / len(landmarks),
            z=sum(
                point.z
                for point in landmarks
            )
            / len(landmarks),
        )

    def _calculate_palm(
        self,
        landmarks: list[Landmark],
    ) -> Landmark:

        indices = [
            0,
            5,
            9,
            13,
            17,
        ]

        points = [
            landmarks[index]
            for index in indices
            if index < len(landmarks)
        ]

        if not points:

            return self._calculate_center(
                landmarks
            )

        return Landmark(
            x=sum(
                point.x
                for point in points
            )
            / len(points),
            y=sum(
                point.y
                for point in points
            )
            / len(points),
            z=sum(
                point.z
                for point in points
            )
            / len(points),
        )

    def _landmark_distance(
        self,
        first: list[Landmark],
        second: list[Landmark],
    ) -> float:

        if not first or not second:

            return float("inf")

        count = min(
            len(first),
            len(second),
        )

        if count == 0:

            return float("inf")

        total = 0.0

        for index in range(count):

            total += first[
                index
            ].distance_to(
                second[index]
            )

        return total / count

    def _calculate_velocity(
        self,
        previous: Optional[Landmark],
        current: Optional[Landmark],
        delta_time: float,
    ) -> tuple[
        float,
        float,
        float,
    ]:

        if (
            previous is None
            or current is None
            or delta_time <= 0
        ):

            return (
                0.0,
                0.0,
                0.0,
            )

        return (
            (
                current.x - previous.x
            )
            / delta_time,
            (
                current.y - previous.y
            )
            / delta_time,
            (
                current.z - previous.z
            )
            / delta_time,
        )

    # ========================================================================
    # FPS
    # ========================================================================

    def _update_fps(self) -> None:

        now = time.monotonic()

        delta = (
            now
            - self._last_update_time
        )

        self._last_update_time = now

        if delta <= 0:

            return

        current_fps = 1.0 / delta

        if self._fps <= 0:

            self._fps = current_fps

        else:

            self._fps = (
                self._fps * 0.85
                + current_fps * 0.15
            )

    # ========================================================================
    # PUBLIC ACCESSORS
    # ========================================================================

    def get_hands(
        self,
        *,
        visible_only: bool = True,
    ) -> list[TrackedHand]:

        with self._lock:

            hands = list(
                self._tracks.values()
            )

        if visible_only:

            hands = [
                hand
                for hand in hands
                if hand.is_visible
            ]

        return [
            self._copy_track(hand)
            for hand in hands
        ]

    def get_hand(
        self,
        hand_id: str,
    ) -> Optional[TrackedHand]:

        with self._lock:

            hand = self._tracks.get(
                hand_id
            )

            if hand is None:

                return None

            return self._copy_track(
                hand
            )

    def get_left_hand(
        self,
    ) -> Optional[TrackedHand]:

        for hand in self.get_hands():

            if hand.side == "left":

                return hand

        return None

    def get_right_hand(
        self,
    ) -> Optional[TrackedHand]:

        for hand in self.get_hands():

            if hand.side == "right":

                return hand

        return None

    def get_fps(
        self,
    ) -> float:

        with self._lock:

            return self._fps

    def get_frame_number(
        self,
    ) -> int:

        with self._lock:

            return self._frame_number

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "frame_number": (
                    self._frame_number
                ),
                "fps": self._fps,
                "tracked_hands": len(
                    self._tracks
                ),
                "visible_hands": sum(
                    1
                    for hand in (
                        self._tracks.values()
                    )
                    if hand.is_visible
                ),
                "max_hands": (
                    self.config.max_hands
                ),
            }

    # ========================================================================
    # CONVERSION FOR GESTURE ENGINE
    # ========================================================================

    def to_gesture_engine_input(
        self,
        hands: Optional[
            list[TrackedHand]
        ] = None,
    ) -> list[dict[str, Any]]:

        if hands is None:

            hands = self.get_hands()

        result: list[
            dict[str, Any]
        ] = []

        for hand in hands:

            if not hand.is_visible:

                continue

            result.append(
                {
                    "hand_id": hand.hand_id,
                    "side": hand.side,
                    "confidence": hand.confidence,
                    "landmarks": [
                        {
                            "x": point.x,
                            "y": point.y,
                            "z": point.z,
                        }
                        for point in hand.landmarks
                    ],
                    "palm": (
                        hand.palm.to_dict()
                        if hand.palm
                        else None
                    ),
                    "wrist": (
                        hand.wrist.to_dict()
                        if hand.wrist
                        else None
                    ),
                    "velocity": {
                        "x": hand.velocity_x,
                        "y": hand.velocity_y,
                        "z": hand.velocity_z,
                    },
                    "speed": hand.speed,
                    "metadata": dict(
                        hand.metadata
                    ),
                }
            )

        return result

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _normalize_side(
        self,
        value: Any,
    ) -> str:

        text = str(
            value
        ).lower()

        if "left" in text:

            return "left"

        if "right" in text:

            return "right"

        return "unknown"

    def _safe_float(
        self,
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            number = float(value)

            if math.isnan(number):

                return default

            if math.isinf(number):

                return default

            return number

        except (
            TypeError,
            ValueError,
        ):

            return default

    def _copy_track(
        self,
        track: TrackedHand,
    ) -> TrackedHand:

        return TrackedHand(
            hand_id=track.hand_id,
            side=track.side,
            confidence=track.confidence,
            landmarks=list(
                track.landmarks
            ),
            raw_landmarks=list(
                track.raw_landmarks
            ),
            wrist=track.wrist,
            palm=track.palm,
            center=track.center,
            velocity_x=track.velocity_x,
            velocity_y=track.velocity_y,
            velocity_z=track.velocity_z,
            speed=track.speed,
            age=track.age,
            missed_frames=track.missed_frames,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            is_visible=track.is_visible,
            metadata=dict(
                track.metadata
            ),
        )


# ============================================================================
# DEFAULT TRACKER
# ============================================================================


_default_tracker: Optional[
    HandTracker
] = None

_default_tracker_lock = (
    threading.RLock()
)


def get_hand_tracker() -> HandTracker:

    global _default_tracker

    with _default_tracker_lock:

        if _default_tracker is None:

            _default_tracker = (
                HandTracker()
            )

        return _default_tracker


def track_hands(
    detections: Any,
    **kwargs: Any,
) -> list[TrackedHand]:

    return (
        get_hand_tracker()
        .update(
            detections,
            **kwargs,
        )
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "Landmark",
    "TrackedHand",
    "HandTrackerConfig",
    "HandTracker",
    "get_hand_tracker",
    "track_hands",
]


