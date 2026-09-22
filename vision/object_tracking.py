"""
RENIX Vision — Object Tracking

Tracks detected objects across consecutive frames.

Responsibilities:
- Maintain stable tracking IDs.
- Match detections between frames.
- Track object positions and movement.
- Estimate velocity and direction.
- Handle objects entering/leaving the frame.
- Support configurable IoU-based matching.
- Provide track history.
- Work with the DetectedObject / BoundingBox classes from
  object_detection.py.

This module does not perform object detection itself.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .object_detection import (
    BoundingBox,
    DetectedObject,
)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class TrackPoint:
    """One historical position of a tracked object."""

    x: float
    y: float
    timestamp: float

    def to_tuple(self) -> tuple[float, float]:
        return self.x, self.y

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "timestamp": self.timestamp,
        }


@dataclass
class ObjectTrack:
    """
    Persistent state for one tracked object.
    """

    track_id: int

    label: str

    confidence: float

    box: BoundingBox

    first_seen: float

    last_seen: float

    hits: int = 1

    missed_frames: int = 0

    age: int = 1

    history: list[TrackPoint] = field(
        default_factory=list
    )

    velocity_x: float = 0.0

    velocity_y: float = 0.0

    speed: float = 0.0

    direction: float = 0.0

    active: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def center(self) -> tuple[int, int]:
        return self.box.center

    @property
    def movement_vector(
        self,
    ) -> tuple[float, float]:
        return (
            self.velocity_x,
            self.velocity_y,
        )

    @property
    def lifetime(self) -> float:
        return max(
            0.0,
            self.last_seen - self.first_seen,
        )

    def update(
        self,
        detection: DetectedObject,
        timestamp: float,
        max_history: int,
    ) -> None:

        old_center = self.box.center

        new_center = detection.box.center

        dt = max(
            0.0001,
            timestamp - self.last_seen,
        )

        self.velocity_x = (
            new_center[0] - old_center[0]
        ) / dt

        self.velocity_y = (
            new_center[1] - old_center[1]
        ) / dt

        self.speed = math.sqrt(
            self.velocity_x ** 2
            + self.velocity_y ** 2
        )

        if self.speed > 0.001:

            self.direction = math.degrees(
                math.atan2(
                    self.velocity_y,
                    self.velocity_x,
                )
            )

        self.box = detection.box

        self.confidence = (
            detection.confidence
        )

        self.label = detection.label

        self.last_seen = timestamp

        self.hits += 1

        self.missed_frames = 0

        self.age += 1

        self.active = True

        self.metadata.update(
            detection.metadata
        )

        self.history.append(
            TrackPoint(
                x=float(new_center[0]),
                y=float(new_center[1]),
                timestamp=timestamp,
            )
        )

        if len(self.history) > max_history:

            del self.history[
                : len(self.history) - max_history
            ]

    def mark_missed(self) -> None:

        self.missed_frames += 1

        self.age += 1

    def deactivate(self) -> None:

        self.active = False

    def predict_position(
        self,
        seconds: float = 0.1,
    ) -> tuple[float, float]:

        cx, cy = self.center

        return (
            cx + self.velocity_x * seconds,
            cy + self.velocity_y * seconds,
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": self.confidence,
            "box": self.box.to_dict(),
            "center": self.center,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "lifetime": self.lifetime,
            "hits": self.hits,
            "missed_frames": self.missed_frames,
            "age": self.age,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "speed": self.speed,
            "direction": self.direction,
            "active": self.active,
            "history": [
                point.to_dict()
                for point in self.history
            ],
            "metadata": dict(self.metadata),
        }


@dataclass
class TrackingResult:
    """
    Result returned after tracking one frame.
    """

    tracks: list[ObjectTrack]

    timestamp: float

    frame_number: int

    processing_time: float

    active_count: int

    created_count: int

    lost_count: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def objects(self) -> list[ObjectTrack]:
        return self.tracks

    def get(
        self,
        track_id: int,
    ) -> Optional[ObjectTrack]:

        for track in self.tracks:

            if track.track_id == track_id:

                return track

        return None

    def by_label(
        self,
        label: str,
    ) -> list[ObjectTrack]:

        normalized = label.lower().strip()

        return [
            track
            for track in self.tracks
            if track.label.lower()
            == normalized
        ]

    def to_dict(self) -> dict[str, Any]:

        return {
            "tracks": [
                track.to_dict()
                for track in self.tracks
            ],
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "processing_time": self.processing_time,
            "active_count": self.active_count,
            "created_count": self.created_count,
            "lost_count": self.lost_count,
            "metadata": dict(self.metadata),
        }


@dataclass
class TrackingConfig:
    """
    Runtime tracking configuration.
    """

    iou_threshold: float = 0.30

    max_missed_frames: int = 10

    max_history: int = 30

    min_hits: int = 1

    enable_prediction: bool = True

    prediction_seconds: float = 0.10

    same_label_only: bool = True

    max_track_distance: float = 300.0

    smoothing_factor: float = 0.50


# ============================================================================
# OBJECT TRACKER
# ============================================================================


class ObjectTracker:
    """
    RENIX multi-object tracker.

    Matching strategy:

        1. Label compatibility
        2. IoU similarity
        3. Center distance
        4. Greedy best-match assignment

    The tracker intentionally does not depend on a specific detector.
    """

    def __init__(
        self,
        config: Optional[
            TrackingConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or TrackingConfig()
        )

        self._lock = threading.RLock()

        self._tracks: dict[
            int,
            ObjectTrack,
        ] = {}

        self._next_track_id = 1

        self._frame_number = 0

        self._last_timestamp: Optional[
            float
        ] = None

        self._total_created = 0

        self._total_lost = 0

    # ========================================================================
    # MAIN UPDATE
    # ========================================================================

    def update(
        self,
        detections: Sequence[
            DetectedObject
        ],
        timestamp: Optional[
            float
        ] = None,
    ) -> TrackingResult:

        started = time.perf_counter()

        now = (
            time.time()
            if timestamp is None
            else float(timestamp)
        )

        with self._lock:

            self._frame_number += 1

            active_before = {
                track_id: track
                for track_id, track
                in self._tracks.items()
                if track.active
            }

            detections = list(
                detections or []
            )

            matched_track_ids: set[
                int
            ] = set()

            matched_detection_indices: set[
                int
            ] = set()

            matches = self._match(
                active_before,
                detections,
            )

            for (
                track_id,
                detection_index,
            ) in matches:

                track = active_before[
                    track_id
                ]

                detection = detections[
                    detection_index
                ]

                self._update_track(
                    track,
                    detection,
                    now,
                )

                matched_track_ids.add(
                    track_id
                )

                matched_detection_indices.add(
                    detection_index
                )

            lost_count = 0

            for track_id, track in (
                active_before.items()
            ):

                if track_id in matched_track_ids:
                    continue

                track.mark_missed()

                if (
                    track.missed_frames
                    > self.config.max_missed_frames
                ):

                    track.deactivate()

                    lost_count += 1

                    self._total_lost += 1

            created_count = 0

            for index, detection in enumerate(
                detections
            ):

                if (
                    index
                    in matched_detection_indices
                ):
                    continue

                track = self._create_track(
                    detection,
                    now,
                )

                self._tracks[
                    track.track_id
                ] = track

                created_count += 1

                self._total_created += 1

            self._last_timestamp = now

            active_tracks = [
                track
                for track in self._tracks.values()
                if track.active
                and track.missed_frames
                <= self.config.max_missed_frames
            ]

            return TrackingResult(
                tracks=active_tracks,
                timestamp=now,
                frame_number=self._frame_number,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                active_count=len(
                    active_tracks
                ),
                created_count=created_count,
                lost_count=lost_count,
            )

    # ========================================================================
    # MATCHING
    # ========================================================================

    def _match(
        self,
        tracks: dict[
            int,
            ObjectTrack,
        ],
        detections: list[
            DetectedObject
        ],
    ) -> list[
        tuple[int, int]
    ]:

        candidates: list[
            tuple[
                float,
                int,
                int,
            ]
        ] = []

        for track_id, track in (
            tracks.items()
        ):

            for index, detection in enumerate(
                detections
            ):

                if (
                    self.config.same_label_only
                    and track.label.lower()
                    != detection.label.lower()
                ):
                    continue

                iou = track.box.iou(
                    detection.box
                )

                distance = self._center_distance(
                    track.box,
                    detection.box,
                )

                if (
                    iou
                    < self.config.iou_threshold
                    and distance
                    > self.config.max_track_distance
                ):
                    continue

                score = self._match_score(
                    iou,
                    distance,
                )

                candidates.append(
                    (
                        score,
                        track_id,
                        index,
                    )
                )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        used_tracks: set[int] = set()

        used_detections: set[int] = set()

        matches: list[
            tuple[int, int]
        ] = []

        for (
            score,
            track_id,
            detection_index,
        ) in candidates:

            if track_id in used_tracks:
                continue

            if (
                detection_index
                in used_detections
            ):
                continue

            used_tracks.add(
                track_id
            )

            used_detections.add(
                detection_index
            )

            matches.append(
                (
                    track_id,
                    detection_index,
                )
            )

        return matches

    def _match_score(
        self,
        iou: float,
        distance: float,
    ) -> float:

        distance_score = 1.0 / (
            1.0 + distance
        )

        return (
            iou * 0.85
            + distance_score * 0.15
        )

    @staticmethod
    def _center_distance(
        first: BoundingBox,
        second: BoundingBox,
    ) -> float:

        x1, y1 = first.center

        x2, y2 = second.center

        return math.sqrt(
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        )

    # ========================================================================
    # TRACK CREATION / UPDATE
    # ========================================================================

    def _create_track(
        self,
        detection: DetectedObject,
        timestamp: float,
    ) -> ObjectTrack:

        track_id = (
            detection.track_id
            if detection.track_id is not None
            else self._next_track_id
        )

        if (
            detection.track_id is None
            or track_id
            >= self._next_track_id
        ):

            self._next_track_id = (
                track_id + 1
            )

        center = detection.box.center

        return ObjectTrack(
            track_id=track_id,
            label=detection.label,
            confidence=detection.confidence,
            box=detection.box,
            first_seen=timestamp,
            last_seen=timestamp,
            history=[
                TrackPoint(
                    x=float(center[0]),
                    y=float(center[1]),
                    timestamp=timestamp,
                )
            ],
            metadata=dict(
                detection.metadata
            ),
        )

    def _update_track(
        self,
        track: ObjectTrack,
        detection: DetectedObject,
        timestamp: float,
    ) -> None:

        previous_vx = (
            track.velocity_x
        )

        previous_vy = (
            track.velocity_y
        )

        track.update(
            detection,
            timestamp,
            self.config.max_history,
        )

        alpha = max(
            0.0,
            min(
                1.0,
                self.config.smoothing_factor,
            ),
        )

        track.velocity_x = (
            previous_vx * (1.0 - alpha)
            + track.velocity_x * alpha
        )

        track.velocity_y = (
            previous_vy * (1.0 - alpha)
            + track.velocity_y * alpha
        )

        track.speed = math.sqrt(
            track.velocity_x ** 2
            + track.velocity_y ** 2
        )

        if track.speed > 0.001:

            track.direction = math.degrees(
                math.atan2(
                    track.velocity_y,
                    track.velocity_x,
                )
            )

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    def get_track(
        self,
        track_id: int,
    ) -> Optional[ObjectTrack]:

        with self._lock:

            return self._tracks.get(
                int(track_id)
            )

    def get_active_tracks(
        self,
    ) -> list[ObjectTrack]:

        with self._lock:

            return [
                track
                for track
                in self._tracks.values()
                if track.active
            ]

    def get_tracks_by_label(
        self,
        label: str,
    ) -> list[ObjectTrack]:

        normalized = label.lower().strip()

        with self._lock:

            return [
                track
                for track
                in self._tracks.values()
                if (
                    track.active
                    and track.label.lower()
                    == normalized
                )
            ]

    def get_nearest_track(
        self,
        x: float,
        y: float,
        label: Optional[str] = None,
    ) -> Optional[ObjectTrack]:

        candidates = (
            self.get_tracks_by_label(
                label
            )
            if label is not None
            else self.get_active_tracks()
        )

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda track:
            math.sqrt(
                (
                    track.center[0]
                    - x
                ) ** 2
                + (
                    track.center[1]
                    - y
                ) ** 2
            ),
        )

    def get_moving_tracks(
        self,
        minimum_speed: float = 5.0,
    ) -> list[ObjectTrack]:

        return [
            track
            for track in self.get_active_tracks()
            if track.speed
            >= minimum_speed
        ]

    # ========================================================================
    # PREDICTION
    # ========================================================================

    def predict(
        self,
        track_id: int,
        seconds: Optional[float] = None,
    ) -> Optional[
        tuple[float, float]
    ]:

        track = self.get_track(
            track_id
        )

        if track is None:
            return None

        if not self.config.enable_prediction:
            return track.center

        duration = (
            self.config.prediction_seconds
            if seconds is None
            else max(
                0.0,
                float(seconds),
            )
        )

        return track.predict_position(
            duration
        )

    def predict_all(
        self,
        seconds: Optional[float] = None,
    ) -> dict[
        int,
        tuple[float, float]
    ]:

        result: dict[
            int,
            tuple[float, float]
        ] = {}

        for track in (
            self.get_active_tracks()
        ):

            prediction = self.predict(
                track.track_id,
                seconds,
            )

            if prediction is not None:

                result[
                    track.track_id
                ] = prediction

        return result

    # ========================================================================
    # HISTORY
    # ========================================================================

    def get_history(
        self,
        track_id: int,
    ) -> list[TrackPoint]:

        track = self.get_track(
            track_id
        )

        if track is None:
            return []

        return list(
            track.history
        )

    def clear_history(
        self,
        track_id: Optional[int] = None,
    ) -> None:

        with self._lock:

            if track_id is None:

                for track in (
                    self._tracks.values()
                ):

                    track.history.clear()

                return

            track = self._tracks.get(
                int(track_id)
            )

            if track is not None:

                track.history.clear()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def remove_track(
        self,
        track_id: int,
    ) -> bool:

        with self._lock:

            track = self._tracks.pop(
                int(track_id),
                None,
            )

            if track is None:
                return False

            track.deactivate()

            return True

    def clear_inactive(
        self,
    ) -> int:

        with self._lock:

            inactive = [
                track_id
                for track_id, track
                in self._tracks.items()
                if not track.active
            ]

            for track_id in inactive:

                del self._tracks[
                    track_id
                ]

            return len(inactive)

    def reset(
        self,
    ) -> None:

        with self._lock:

            self._tracks.clear()

            self._next_track_id = 1

            self._frame_number = 0

            self._last_timestamp = None

            self._total_created = 0

            self._total_lost = 0

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            active = sum(
                1
                for track
                in self._tracks.values()
                if track.active
            )

            return {
                "frame_number": (
                    self._frame_number
                ),
                "total_tracks": len(
                    self._tracks
                ),
                "active_tracks": active,
                "total_created": (
                    self._total_created
                ),
                "total_lost": (
                    self._total_lost
                ),
                "next_track_id": (
                    self._next_track_id
                ),
                "last_timestamp": (
                    self._last_timestamp
                ),
                "iou_threshold": (
                    self.config.iou_threshold
                ),
                "max_missed_frames": (
                    self.config.max_missed_frames
                ),
                "history_length": (
                    self.config.max_history
                ),
            }


# ============================================================================
# SHARED TRACKER
# ============================================================================


_default_tracker: Optional[
    ObjectTracker
] = None

_default_tracker_lock = (
    threading.RLock()
)


def get_object_tracker() -> ObjectTracker:

    global _default_tracker

    with _default_tracker_lock:

        if _default_tracker is None:

            _default_tracker = (
                ObjectTracker()
            )

        return _default_tracker


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def track_objects(
    detections: Sequence[
        DetectedObject
    ],
    timestamp: Optional[
        float
    ] = None,
) -> TrackingResult:

    return get_object_tracker().update(
        detections,
        timestamp,
    )


def get_track(
    track_id: int,
) -> Optional[ObjectTrack]:

    return get_object_tracker().get_track(
        track_id
    )


def get_active_tracks() -> list[
    ObjectTrack
]:

    return (
        get_object_tracker()
        .get_active_tracks()
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "TrackPoint",
    "ObjectTrack",
    "TrackingResult",
    "TrackingConfig",
    "ObjectTracker",
    "get_object_tracker",
    "track_objects",
    "get_track",
    "get_active_tracks",
]


