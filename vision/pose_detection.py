"""
RENIX Vision — Pose Detection

Detects human body pose landmarks from camera frames.

Responsibilities:
- Pose landmark representation
- Body-joint coordinate handling
- Pose detection backend abstraction
- Landmark normalization
- Pixel-coordinate conversion
- Basic body geometry calculations
- Posture/orientation hints
- Graceful fallback when no pose backend is configured

Gesture interpretation belongs in:
    RENIX/gestures/

Person identity belongs in:
    RENIX/vision/face_recognition.py

Object detection belongs in:
    RENIX/vision/object_detection.py
"""

from __future__ import annotations

import logging
import math
import threading
import time

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from .object_detection import BoundingBox


logger = logging.getLogger(
    "RENIX.vision.pose_detection"
)


# ============================================================================
# POSE LANDMARK INDICES
# ============================================================================
#
# MediaPipe-style pose landmark numbering.
#

NOSE = 0

LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3

RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6

LEFT_EAR = 7
RIGHT_EAR = 8

MOUTH_LEFT = 9
MOUTH_RIGHT = 10

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_PINKY = 17
RIGHT_PINKY = 18

LEFT_INDEX = 19
RIGHT_INDEX = 20

LEFT_THUMB = 21
RIGHT_THUMB = 22

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

LEFT_HEEL = 29
RIGHT_HEEL = 30

LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


POSE_LANDMARK_NAMES = {
    NOSE: "nose",
    LEFT_EYE_INNER: "left_eye_inner",
    LEFT_EYE: "left_eye",
    LEFT_EYE_OUTER: "left_eye_outer",
    RIGHT_EYE_INNER: "right_eye_inner",
    RIGHT_EYE: "right_eye",
    RIGHT_EYE_OUTER: "right_eye_outer",
    LEFT_EAR: "left_ear",
    RIGHT_EAR: "right_ear",
    MOUTH_LEFT: "mouth_left",
    MOUTH_RIGHT: "mouth_right",
    LEFT_SHOULDER: "left_shoulder",
    RIGHT_SHOULDER: "right_shoulder",
    LEFT_ELBOW: "left_elbow",
    RIGHT_ELBOW: "right_elbow",
    LEFT_WRIST: "left_wrist",
    RIGHT_WRIST: "right_wrist",
    LEFT_PINKY: "left_pinky",
    RIGHT_PINKY: "right_pinky",
    LEFT_INDEX: "left_index",
    RIGHT_INDEX: "right_index",
    LEFT_THUMB: "left_thumb",
    RIGHT_THUMB: "right_thumb",
    LEFT_HIP: "left_hip",
    RIGHT_HIP: "right_hip",
    LEFT_KNEE: "left_knee",
    RIGHT_KNEE: "right_knee",
    LEFT_ANKLE: "left_ankle",
    RIGHT_ANKLE: "right_ankle",
    LEFT_HEEL: "left_heel",
    RIGHT_HEEL: "right_heel",
    LEFT_FOOT_INDEX: "left_foot_index",
    RIGHT_FOOT_INDEX: "right_foot_index",
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class PoseLandmark:
    """
    Represents one body landmark.

    x/y are normally normalized coordinates.
    z is relative depth when provided by the backend.
    """

    index: int

    name: str

    x: float

    y: float

    z: float = 0.0

    visibility: float = 1.0

    presence: float = 1.0

    pixel_x: Optional[int] = None

    pixel_y: Optional[int] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
            "presence": self.presence,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "metadata": dict(self.metadata),
        }


@dataclass
class PoseDetection:
    """
    Complete pose detected for one person.
    """

    pose_id: int

    confidence: float

    landmarks: list[PoseLandmark] = field(
        default_factory=list
    )

    box: Optional[BoundingBox] = None

    track_id: Optional[int] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)

    @property
    def center(self) -> Optional[tuple[int, int]]:
        if self.box is not None:
            return self.box.center

        landmark = self.get_landmark(
            NOSE
        )

        if landmark is not None:
            if (
                landmark.pixel_x is not None
                and landmark.pixel_y is not None
            ):
                return (
                    landmark.pixel_x,
                    landmark.pixel_y,
                )

        return None

    def get_landmark(
        self,
        index_or_name: int | str,
    ) -> Optional[PoseLandmark]:

        if isinstance(
            index_or_name,
            str,
        ):
            target = (
                index_or_name
                .strip()
                .lower()
            )

            for landmark in self.landmarks:
                if landmark.name.lower() == target:
                    return landmark

            return None

        for landmark in self.landmarks:
            if landmark.index == index_or_name:
                return landmark

        return None

    def get_pixel_position(
        self,
        index_or_name: int | str,
    ) -> Optional[tuple[int, int]]:

        landmark = self.get_landmark(
            index_or_name
        )

        if landmark is None:
            return None

        if (
            landmark.pixel_x is None
            or landmark.pixel_y is None
        ):
            return None

        return (
            landmark.pixel_x,
            landmark.pixel_y,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose_id": self.pose_id,
            "confidence": self.confidence,
            "landmarks": [
                landmark.to_dict()
                for landmark in self.landmarks
            ],
            "box": (
                self.box.to_dict()
                if self.box is not None
                else None
            ),
            "track_id": self.track_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class PoseDetectionResult:
    """
    Result generated for one camera frame.
    """

    poses: list[PoseDetection]

    timestamp: float

    processing_time: float

    image_width: int

    image_height: int

    success: bool = True

    error: Optional[str] = None

    backend: str = "unavailable"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def count(self) -> int:
        return len(self.poses)

    @property
    def primary_pose(
        self,
    ) -> Optional[PoseDetection]:

        if not self.poses:
            return None

        return max(
            self.poses,
            key=lambda pose: pose.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "poses": [
                pose.to_dict()
                for pose in self.poses
            ],
            "count": self.count,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "success": self.success,
            "error": self.error,
            "backend": self.backend,
            "metadata": dict(self.metadata),
        }


@dataclass
class PoseDetectionConfig:
    """
    Configuration for pose detection.
    """

    confidence_threshold: float = 0.50

    tracking_threshold: float = 0.50

    max_poses: int = 1

    backend: str = "auto"

    model_path: Optional[str] = None

    static_image_mode: bool = False

    enable_segmentation: bool = False

    enable_world_landmarks: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# BACKEND PROTOCOL
# ============================================================================


class PoseDetectionBackend(Protocol):
    """
    Interface for MediaPipe, YOLO, ONNX or custom
    pose-estimation implementations.
    """

    @property
    def name(self) -> str:
        ...

    def is_available(self) -> bool:
        ...

    def detect(
        self,
        image: Any,
        confidence_threshold: float,
        max_poses: int,
    ) -> list[PoseDetection]:
        ...


# ============================================================================
# GENERIC BACKEND ADAPTER
# ============================================================================


class GenericPoseBackend:
    """
    Adapter around a custom pose detector.

    The custom detector may return:
        PoseDetection
        list[PoseDetection]
        dictionaries
    """

    def __init__(
        self,
        detector: Any,
        name: str = "custom",
    ) -> None:

        self.detector = detector
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self.detector is not None

    def detect(
        self,
        image: Any,
        confidence_threshold: float,
        max_poses: int,
    ) -> list[PoseDetection]:

        if self.detector is None:
            return []

        raw = self.detector(
            image,
            confidence_threshold=confidence_threshold,
            max_poses=max_poses,
        )

        return self._normalize(raw)

    def _normalize(
        self,
        raw: Any,
    ) -> list[PoseDetection]:

        if raw is None:
            return []

        if isinstance(
            raw,
            PoseDetection,
        ):
            return [raw]

        if isinstance(raw, dict):
            raw = [raw]

        result: list[PoseDetection] = []

        for index, item in enumerate(raw):

            if isinstance(
                item,
                PoseDetection,
            ):
                result.append(item)
                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            try:
                result.append(
                    self._dict_to_pose(
                        item,
                        index,
                    )
                )

            except Exception:
                logger.debug(
                    "Could not normalize "
                    "pose detection.",
                    exc_info=True,
                )

        return result

    @staticmethod
    def _dict_to_pose(
        item: dict[str, Any],
        default_id: int,
    ) -> PoseDetection:

        pose_id = int(
            item.get(
                "pose_id",
                item.get(
                    "id",
                    default_id,
                ),
            )
        )

        confidence = float(
            item.get(
                "confidence",
                item.get(
                    "score",
                    0.0,
                ),
            )
        )

        landmarks = (
            GenericPoseBackend
            ._parse_landmarks(
                item.get(
                    "landmarks",
                    [],
                )
            )
        )

        box = (
            GenericPoseBackend
            ._parse_box(
                item.get(
                    "box"
                )
            )
        )

        return PoseDetection(
            pose_id=pose_id,
            confidence=confidence,
            landmarks=landmarks,
            box=box,
            track_id=item.get(
                "track_id"
            ),
            metadata=dict(
                item.get(
                    "metadata",
                    {},
                )
            ),
        )

    @staticmethod
    def _parse_box(
        value: Any,
    ) -> Optional[BoundingBox]:

        if value is None:
            return None

        if isinstance(
            value,
            BoundingBox,
        ):
            return value

        if isinstance(
            value,
            dict,
        ):

            if {
                "x",
                "y",
                "width",
                "height",
            }.issubset(value):

                return BoundingBox(
                    x=value["x"],
                    y=value["y"],
                    width=value["width"],
                    height=value["height"],
                )

            if {
                "x1",
                "y1",
                "x2",
                "y2",
            }.issubset(value):

                return BoundingBox(
                    x=value["x1"],
                    y=value["y1"],
                    width=(
                        value["x2"]
                        - value["x1"]
                    ),
                    height=(
                        value["y2"]
                        - value["y1"]
                    ),
                )

        try:
            values = list(value)

            if len(values) == 4:

                x1, y1, x2, y2 = values

                return BoundingBox(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                )

        except Exception:
            pass

        return None

    @staticmethod
    def _parse_landmarks(
        value: Any,
    ) -> list[PoseLandmark]:

        if value is None:
            return []

        result: list[PoseLandmark] = []

        for index, item in enumerate(value):

            if isinstance(
                item,
                PoseLandmark,
            ):
                result.append(item)
                continue

            if isinstance(item, dict):

                landmark_index = int(
                    item.get(
                        "index",
                        index,
                    )
                )

                result.append(
                    PoseLandmark(
                        index=landmark_index,
                        name=str(
                            item.get(
                                "name",
                                POSE_LANDMARK_NAMES.get(
                                    landmark_index,
                                    f"landmark_{landmark_index}",
                                ),
                            )
                        ),
                        x=float(
                            item.get(
                                "x",
                                0.0,
                            )
                        ),
                        y=float(
                            item.get(
                                "y",
                                0.0,
                            )
                        ),
                        z=float(
                            item.get(
                                "z",
                                0.0,
                            )
                        ),
                        visibility=float(
                            item.get(
                                "visibility",
                                1.0,
                            )
                        ),
                        presence=float(
                            item.get(
                                "presence",
                                1.0,
                            )
                        ),
                        pixel_x=item.get(
                            "pixel_x"
                        ),
                        pixel_y=item.get(
                            "pixel_y"
                        ),
                        metadata=dict(
                            item.get(
                                "metadata",
                                {},
                            )
                        ),
                    )
                )

                continue

            try:

                values = list(item)

                if len(values) >= 2:

                    result.append(
                        PoseLandmark(
                            index=index,
                            name=POSE_LANDMARK_NAMES.get(
                                index,
                                f"landmark_{index}",
                            ),
                            x=float(
                                values[0]
                            ),
                            y=float(
                                values[1]
                            ),
                            z=(
                                float(
                                    values[2]
                                )
                                if len(values) >= 3
                                else 0.0
                            ),
                        )
                    )

            except Exception:
                continue

        return result


# ============================================================================
# POSE DETECTOR
# ============================================================================


class PoseDetector:
    """
    Main RENIX pose-detection service.
    """

    def __init__(
        self,
        config: Optional[
            PoseDetectionConfig
        ] = None,
        backend: Optional[
            PoseDetectionBackend
        ] = None,
    ) -> None:

        self.config = (
            config
            or PoseDetectionConfig()
        )

        self._backend = backend

        self._lock = threading.RLock()

        self._initialized = False

        self._last_result: Optional[
            PoseDetectionResult
        ] = None

        self._frame_counter = 0

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:

        self._initialized = True

        if self._backend is not None:

            logger.info(
                "RENIX pose detector initialized "
                "with backend: %s",
                self._backend.name,
            )

        else:

            logger.info(
                "No pose-detection backend configured. "
                "Running in fallback mode."
            )

    # ========================================================================
    # STATUS
    # ========================================================================

    @property
    def backend_name(self) -> str:

        if self._backend is None:
            return "unavailable"

        return self._backend.name

    def is_available(self) -> bool:

        return (
            self._backend is not None
            and self._backend.is_available()
        )

    # ========================================================================
    # DETECTION
    # ========================================================================

    def detect(
        self,
        image: Any,
    ) -> PoseDetectionResult:

        started = time.perf_counter()

        timestamp = time.time()

        with self._lock:
            self._frame_counter += 1

        width, height = (
            self._get_dimensions(image)
        )

        if image is None:

            return self._failure(
                timestamp,
                started,
                "Input image is None.",
                width,
                height,
            )

        if width <= 0 or height <= 0:

            return self._failure(
                timestamp,
                started,
                "Invalid image dimensions.",
                width,
                height,
            )

        if not self.is_available():

            result = PoseDetectionResult(
                poses=[],
                timestamp=timestamp,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                image_width=width,
                image_height=height,
                success=True,
                backend="unavailable",
                metadata={
                    "fallback": True,
                    "reason": (
                        "No pose-detection "
                        "backend configured."
                    ),
                },
            )

            self._store_result(result)

            return result

        try:

            poses = self._backend.detect(
                image,
                self.config.confidence_threshold,
                self.config.max_poses,
            )

            poses = self._sanitize(
                poses
            )

            self._convert_landmarks_to_pixels(
                poses,
                width,
                height,
            )

            if (
                self.config.max_poses > 0
            ):
                poses = poses[
                    : self.config.max_poses
                ]

            result = PoseDetectionResult(
                poses=poses,
                timestamp=timestamp,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                image_width=width,
                image_height=height,
                success=True,
                backend=self.backend_name,
                metadata={
                    "frame_number": (
                        self._frame_counter
                    ),
                },
            )

            self._store_result(result)

            return result

        except Exception as exc:

            logger.exception(
                "Pose detection failed."
            )

            return self._failure(
                timestamp,
                started,
                str(exc),
                width,
                height,
            )

    # ========================================================================
    # SANITIZATION
    # ========================================================================

    def _sanitize(
        self,
        poses: Sequence[
            PoseDetection
        ],
    ) -> list[PoseDetection]:

        result: list[PoseDetection] = []

        threshold = (
            self.config.confidence_threshold
        )

        for pose in poses:

            if not isinstance(
                pose,
                PoseDetection,
            ):
                continue

            pose.confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        pose.confidence
                    ),
                ),
            )

            if pose.confidence < threshold:
                continue

            for landmark in pose.landmarks:

                landmark.visibility = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            landmark.visibility
                        ),
                    ),
                )

                landmark.presence = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            landmark.presence
                        ),
                    ),
                )

            result.append(pose)

        return result

    # ========================================================================
    # COORDINATES
    # ========================================================================

    @staticmethod
    def _convert_landmarks_to_pixels(
        poses: list[PoseDetection],
        width: int,
        height: int,
    ) -> None:

        for pose in poses:

            for landmark in pose.landmarks:

                if (
                    0.0 <= landmark.x <= 1.0
                    and 0.0 <= landmark.y <= 1.0
                ):

                    landmark.pixel_x = int(
                        landmark.x * width
                    )

                    landmark.pixel_y = int(
                        landmark.y * height
                    )

                else:

                    landmark.pixel_x = int(
                        landmark.x
                    )

                    landmark.pixel_y = int(
                        landmark.y
                    )

    @staticmethod
    def _get_dimensions(
        image: Any,
    ) -> tuple[int, int]:

        if image is None:
            return 0, 0

        try:

            height, width = (
                image.shape[:2]
            )

            return (
                int(width),
                int(height),
            )

        except Exception:

            return 0, 0

    # ========================================================================
    # RESULT MANAGEMENT
    # ========================================================================

    def _store_result(
        self,
        result: PoseDetectionResult,
    ) -> None:

        with self._lock:
            self._last_result = result

    def _failure(
        self,
        timestamp: float,
        started: float,
        error: str,
        width: int,
        height: int,
    ) -> PoseDetectionResult:

        result = PoseDetectionResult(
            poses=[],
            timestamp=timestamp,
            processing_time=(
                time.perf_counter()
                - started
            ),
            image_width=width,
            image_height=height,
            success=False,
            error=error,
            backend=self.backend_name,
        )

        self._store_result(result)

        return result

    def get_last_result(
        self,
    ) -> Optional[
        PoseDetectionResult
    ]:

        with self._lock:
            return self._last_result

    # ========================================================================
    # BACKEND MANAGEMENT
    # ========================================================================

    def set_backend(
        self,
        backend: Optional[
            PoseDetectionBackend
        ],
    ) -> None:

        with self._lock:
            self._backend = backend

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_confidence_threshold(
        self,
        threshold: float,
    ) -> None:

        threshold = float(threshold)

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "Confidence threshold must "
                "be between 0 and 1."
            )

        with self._lock:
            self.config.confidence_threshold = (
                threshold
            )

    def set_max_poses(
        self,
        max_poses: int,
    ) -> None:

        max_poses = int(max_poses)

        if max_poses <= 0:

            raise ValueError(
                "max_poses must be positive."
            )

        with self._lock:
            self.config.max_poses = max_poses

    # ========================================================================
    # LANDMARK GEOMETRY
    # ========================================================================

    @staticmethod
    def distance(
        pose: PoseDetection,
        first: int | str,
        second: int | str,
    ) -> Optional[float]:

        point_a = pose.get_landmark(first)
        point_b = pose.get_landmark(second)

        if (
            point_a is None
            or point_b is None
        ):
            return None

        return _distance_3d(
            point_a.x,
            point_a.y,
            point_a.z,
            point_b.x,
            point_b.y,
            point_b.z,
        )

    @staticmethod
    def pixel_distance(
        pose: PoseDetection,
        first: int | str,
        second: int | str,
    ) -> Optional[float]:

        point_a = pose.get_landmark(first)
        point_b = pose.get_landmark(second)

        if (
            point_a is None
            or point_b is None
            or point_a.pixel_x is None
            or point_a.pixel_y is None
            or point_b.pixel_x is None
            or point_b.pixel_y is None
        ):
            return None

        return _distance_2d(
            point_a.pixel_x,
            point_a.pixel_y,
            point_b.pixel_x,
            point_b.pixel_y,
        )

    @staticmethod
    def angle(
        pose: PoseDetection,
        first: int | str,
        middle: int | str,
        third: int | str,
    ) -> Optional[float]:

        point_a = pose.get_landmark(first)
        point_b = pose.get_landmark(middle)
        point_c = pose.get_landmark(third)

        if (
            point_a is None
            or point_b is None
            or point_c is None
        ):
            return None

        vector_ba = (
            point_a.x - point_b.x,
            point_a.y - point_b.y,
            point_a.z - point_b.z,
        )

        vector_bc = (
            point_c.x - point_b.x,
            point_c.y - point_b.y,
            point_c.z - point_b.z,
        )

        return _angle_between_vectors(
            vector_ba,
            vector_bc,
        )

    # ========================================================================
    # BODY HELPERS
    # ========================================================================

    @staticmethod
    def shoulder_width(
        pose: PoseDetection,
    ) -> Optional[float]:

        return PoseDetector.distance(
            pose,
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
        )

    @staticmethod
    def hip_width(
        pose: PoseDetection,
    ) -> Optional[float]:

        return PoseDetector.distance(
            pose,
            LEFT_HIP,
            RIGHT_HIP,
        )

    @staticmethod
    def torso_length(
        pose: PoseDetection,
    ) -> Optional[float]:

        left_shoulder = pose.get_landmark(
            LEFT_SHOULDER
        )

        right_shoulder = pose.get_landmark(
            RIGHT_SHOULDER
        )

        left_hip = pose.get_landmark(
            LEFT_HIP
        )

        right_hip = pose.get_landmark(
            RIGHT_HIP
        )

        if (
            left_shoulder is None
            or right_shoulder is None
            or left_hip is None
            or right_hip is None
        ):
            return None

        shoulder_center = (
            (
                left_shoulder.x
                + right_shoulder.x
            )
            / 2,
            (
                left_shoulder.y
                + right_shoulder.y
            )
            / 2,
            (
                left_shoulder.z
                + right_shoulder.z
            )
            / 2,
        )

        hip_center = (
            (
                left_hip.x
                + right_hip.x
            )
            / 2,
            (
                left_hip.y
                + right_hip.y
            )
            / 2,
            (
                left_hip.z
                + right_hip.z
            )
            / 2,
        )

        return _distance_3d(
            *shoulder_center,
            *hip_center,
        )

    @staticmethod
    def body_center(
        pose: PoseDetection,
    ) -> Optional[tuple[float, float]]:

        points = []

        for index in (
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP,
        ):

            landmark = pose.get_landmark(index)

            if landmark is not None:

                points.append(
                    (
                        landmark.x,
                        landmark.y,
                    )
                )

        if not points:
            return None

        return (
            sum(point[0] for point in points)
            / len(points),
            sum(point[1] for point in points)
            / len(points),
        )

    # ========================================================================
    # POSTURE HELPERS
    # ========================================================================

    @staticmethod
    def arm_angles(
        pose: PoseDetection,
    ) -> dict[str, Optional[float]]:

        return {
            "left": PoseDetector.angle(
                pose,
                LEFT_SHOULDER,
                LEFT_ELBOW,
                LEFT_WRIST,
            ),
            "right": PoseDetector.angle(
                pose,
                RIGHT_SHOULDER,
                RIGHT_ELBOW,
                RIGHT_WRIST,
            ),
        }

    @staticmethod
    def knee_angles(
        pose: PoseDetection,
    ) -> dict[str, Optional[float]]:

        return {
            "left": PoseDetector.angle(
                pose,
                LEFT_HIP,
                LEFT_KNEE,
                LEFT_ANKLE,
            ),
            "right": PoseDetector.angle(
                pose,
                RIGHT_HIP,
                RIGHT_KNEE,
                RIGHT_ANKLE,
            ),
        }

    @staticmethod
    def shoulder_tilt(
        pose: PoseDetection,
    ) -> Optional[float]:

        left = pose.get_landmark(
            LEFT_SHOULDER
        )

        right = pose.get_landmark(
            RIGHT_SHOULDER
        )

        if left is None or right is None:
            return None

        dx = right.x - left.x
        dy = right.y - left.y

        if abs(dx) < 1e-9:
            return None

        return math.degrees(
            math.atan2(dy, dx)
        )

    @staticmethod
    def is_facing_camera(
        pose: PoseDetection,
        tolerance: float = 0.12,
    ) -> bool:

        left_shoulder = pose.get_landmark(
            LEFT_SHOULDER
        )

        right_shoulder = pose.get_landmark(
            RIGHT_SHOULDER
        )

        left_hip = pose.get_landmark(
            LEFT_HIP
        )

        right_hip = pose.get_landmark(
            RIGHT_HIP
        )

        if any(
            point is None
            for point in (
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip,
            )
        ):
            return False

        shoulder_difference = abs(
            left_shoulder.z
            - right_shoulder.z
        )

        hip_difference = abs(
            left_hip.z
            - right_hip.z
        )

        return (
            shoulder_difference
            <= tolerance
            and hip_difference
            <= tolerance
        )

    # ========================================================================
    # MOVEMENT HELPERS
    # ========================================================================

    @staticmethod
    def wrist_positions(
        pose: PoseDetection,
    ) -> dict[
        str,
        Optional[tuple[float, float]],
    ]:

        result = {}

        for name, index in (
            ("left", LEFT_WRIST),
            ("right", RIGHT_WRIST),
        ):

            landmark = pose.get_landmark(index)

            if landmark is None:

                result[name] = None

            else:

                result[name] = (
                    landmark.x,
                    landmark.y,
                )

        return result

    @staticmethod
    def hand_height(
        pose: PoseDetection,
        side: str,
    ) -> Optional[float]:

        side = side.lower().strip()

        index = (
            LEFT_WRIST
            if side == "left"
            else RIGHT_WRIST
            if side == "right"
            else None
        )

        if index is None:
            raise ValueError(
                "side must be 'left' or 'right'."
            )

        landmark = pose.get_landmark(index)

        if landmark is None:
            return None

        return landmark.y

    @staticmethod
    def hand_above_head(
        pose: PoseDetection,
        side: str,
    ) -> bool:

        wrist = pose.get_landmark(
            LEFT_WRIST
            if side.lower() == "left"
            else RIGHT_WRIST
        )

        nose = pose.get_landmark(
            NOSE
        )

        if wrist is None or nose is None:
            return False

        return wrist.y < nose.y

    # ========================================================================
    # BACKEND MANAGEMENT
    # ========================================================================

    def close(self) -> None:

        backend = self._backend

        if backend is None:
            return

        close_method = getattr(
            backend,
            "close",
            None,
        )

        if callable(close_method):

            try:
                close_method()

            except Exception:
                logger.exception(
                    "Error while closing "
                    "pose backend."
                )

        with self._lock:
            self._backend = None


# ============================================================================
# SHARED DETECTOR
# ============================================================================


_default_detector: Optional[
    PoseDetector
] = None

_default_detector_lock = (
    threading.RLock()
)


def get_pose_detector() -> PoseDetector:

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                PoseDetector()
            )

        return _default_detector


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def detect_pose(
    image: Any,
) -> PoseDetectionResult:

    return get_pose_detector().detect(
        image
    )


def get_poses(
    image: Any,
) -> list[PoseDetection]:

    return detect_pose(image).poses


def get_primary_pose(
    image: Any,
) -> Optional[PoseDetection]:

    return detect_pose(
        image
    ).primary_pose


def landmark_distance(
    pose: PoseDetection,
    first: int | str,
    second: int | str,
) -> Optional[float]:

    return PoseDetector.distance(
        pose,
        first,
        second,
    )


def landmark_angle(
    pose: PoseDetection,
    first: int | str,
    middle: int | str,
    third: int | str,
) -> Optional[float]:

    return PoseDetector.angle(
        pose,
        first,
        middle,
        third,
    )


# ============================================================================
# GEOMETRY
# ============================================================================


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


def _distance_3d(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
) -> float:

    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
        + (z2 - z1) ** 2
    )


def _angle_between_vectors(
    vector_a: tuple[float, float, float],
    vector_b: tuple[float, float, float],
) -> float:

    dot = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(
            value * value
            for value in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            value * value
            for value in vector_b
        )
    )

    if (
        magnitude_a <= 1e-12
        or magnitude_b <= 1e-12
    ):
        return 0.0

    cosine = dot / (
        magnitude_a
        * magnitude_b
    )

    cosine = max(
        -1.0,
        min(1.0, cosine),
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
    "POSE_LANDMARK_NAMES",
    "PoseLandmark",
    "PoseDetection",
    "PoseDetectionResult",
    "PoseDetectionConfig",
    "PoseDetectionBackend",
    "GenericPoseBackend",
    "PoseDetector",
    "get_pose_detector",
    "detect_pose",
    "get_poses",
    "get_primary_pose",
    "landmark_distance",
    "landmark_angle",
]


