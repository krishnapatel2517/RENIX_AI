"""
RENIX Vision — Person Detection

Detects people in camera frames, screenshots, and images.

Responsibilities:
- Person detection
- Multiple-person detection
- Bounding boxes
- Confidence scores
- Person tracking IDs when available
- Region-of-interest filtering
- Detection statistics
- Graceful fallback when optional vision dependencies
  are unavailable

This module is intentionally independent from:
    face_detection.py
    face_recognition.py
    object_detection.py

The higher-level vision pipeline can combine all three.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


logger = logging.getLogger(
    "RENIX.vision.person_detection"
)


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class PersonBoundingBox:
    """
    Bounding box around a detected person.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        self.x = max(0, int(self.x))
        self.y = max(0, int(self.y))
        self.width = max(0, int(self.width))
        self.height = max(0, int(self.height))

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0

        return self.width / self.height

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:
        return (
            self.x <= x <= self.x2
            and self.y <= y <= self.y2
        )

    def intersection(
        self,
        other: "PersonBoundingBox",
    ) -> Optional["PersonBoundingBox"]:

        left = max(
            self.x,
            other.x,
        )

        top = max(
            self.y,
            other.y,
        )

        right = min(
            self.x2,
            other.x2,
        )

        bottom = min(
            self.y2,
            other.y2,
        )

        if right <= left or bottom <= top:
            return None

        return PersonBoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )

    def iou(
        self,
        other: "PersonBoundingBox",
    ) -> float:

        intersection = self.intersection(
            other
        )

        if intersection is None:
            return 0.0

        intersection_area = (
            intersection.area
        )

        union_area = (
            self.area
            + other.area
            - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return (
            intersection_area
            / union_area
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "x2": self.x2,
            "y2": self.y2,
            "center": self.center,
            "area": self.area,
            "aspect_ratio": self.aspect_ratio,
        }


@dataclass
class PersonDetection:
    """
    One detected person.
    """

    box: PersonBoundingBox

    confidence: float

    class_name: str = "person"

    class_id: int = 0

    track_id: Optional[int] = None

    visible_ratio: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def center(self) -> tuple[int, int]:
        return self.box.center

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "confidence": self.confidence,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "track_id": self.track_id,
            "visible_ratio": self.visible_ratio,
            "metadata": dict(self.metadata),
        }


@dataclass
class PersonDetectionResult:
    """
    Result returned by the person detector.
    """

    people: list[PersonDetection]

    timestamp: float

    processing_time: float

    image_width: int

    image_height: int

    success: bool = True

    error: Optional[str] = None

    backend: str = "opencv_hog"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def count(self) -> int:
        return len(self.people)

    @property
    def primary_person(
        self,
    ) -> Optional[PersonDetection]:

        if not self.people:
            return None

        return max(
            self.people,
            key=lambda person:
            person.box.area
            * person.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "people": [
                person.to_dict()
                for person in self.people
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
class PersonDetectionConfig:
    """
    Configuration for person detection.
    """

    confidence_threshold: float = 0.50

    nms_threshold: float = 0.40

    scale: float = 1.05

    min_neighbors: int = 4

    min_width: int = 30

    min_height: int = 60

    max_people: int = 20

    enable_tracking: bool = True

    roi: Optional[
        tuple[int, int, int, int]
    ] = None

    backend: str = "auto"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# PERSON DETECTOR
# ============================================================================


class PersonDetector:
    """
    RENIX person detection engine.

    Example
    -------

    detector = PersonDetector()

    result = detector.detect(frame)

    for person in result.people:
        print(
            person.confidence,
            person.box,
        )
    """

    def __init__(
        self,
        config: Optional[
            PersonDetectionConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or PersonDetectionConfig()
        )

        self._lock = threading.RLock()

        self._hog = None

        self._backend = "unavailable"

        self._initialized = False

        self._last_result: Optional[
            PersonDetectionResult
        ] = None

        self._next_track_id = 1

        self._previous_detections: list[
            PersonDetection
        ] = []

        self._detection_calls = 0

        self._total_people = 0

        self._total_processing_time = 0.0

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:
        """
        Initialize the best available lightweight backend.
        """

        if not CV2_AVAILABLE:

            logger.warning(
                "OpenCV is unavailable. "
                "Person detection will run in "
                "graceful fallback mode."
            )

            self._backend = "unavailable"

            self._initialized = True

            return

        try:

            self._hog = (
                cv2.HOGDescriptor()
            )

            self._hog.setSVMDetector(
                cv2.HOGDescriptor_getDefaultPeopleDetector()
            )

            self._backend = "opencv_hog"

            self._initialized = True

            logger.info(
                "RENIX person detector initialized "
                "using OpenCV HOG."
            )

        except Exception:

            logger.exception(
                "Failed to initialize person detector."
            )

            self._hog = None

            self._backend = "unavailable"

            self._initialized = True

    # ========================================================================
    # AVAILABILITY
    # ========================================================================

    def is_available(self) -> bool:
        return (
            self._hog is not None
            and CV2_AVAILABLE
        )

    @property
    def backend(self) -> str:
        return self._backend

    # ========================================================================
    # DETECTION
    # ========================================================================

    def detect(
        self,
        image: Any,
    ) -> PersonDetectionResult:
        """
        Detect people in an image.

        Parameters
        ----------
        image:
            OpenCV/Numpy image.

        Returns
        -------
        PersonDetectionResult
        """

        started = time.perf_counter()

        timestamp = time.time()

        if image is None:

            return self._failure(
                timestamp,
                started,
                "Input image is None.",
            )

        try:

            width, height = (
                self._get_dimensions(
                    image
                )
            )

            if (
                width <= 0
                or height <= 0
            ):

                return self._failure(
                    timestamp,
                    started,
                    "Invalid image dimensions.",
                )

            if not self.is_available():

                result = PersonDetectionResult(
                    people=[],
                    timestamp=timestamp,
                    processing_time=(
                        time.perf_counter()
                        - started
                    ),
                    image_width=width,
                    image_height=height,
                    success=True,
                    backend=self._backend,
                    metadata={
                        "fallback": True,
                        "reason":
                        "Person detection backend unavailable.",
                    },
                )

                self._record(
                    result
                )

                return result

            processed_image = (
                self._prepare_image(
                    image
                )
            )

            if processed_image is None:

                return self._failure(
                    timestamp,
                    started,
                    "Unable to prepare image.",
                    width,
                    height,
                )

            boxes, weights = (
                self._hog.detectMultiScale(
                    processed_image,
                    winStride=(8, 8),
                    padding=(16, 16),
                    scale=self.config.scale,
                )
            )

            detections = (
                self._build_detections(
                    boxes,
                    weights,
                    width,
                    height,
                )
            )

            detections = (
                self._apply_roi(
                    detections
                )
            )

            detections = (
                self._sort_detections(
                    detections
                )
            )

            if (
                self.config.max_people > 0
            ):

                detections = detections[
                    : self.config.max_people
                ]

            if self.config.enable_tracking:

                self._assign_track_ids(
                    detections
                )

            result = PersonDetectionResult(
                people=detections,
                timestamp=timestamp,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                image_width=width,
                image_height=height,
                success=True,
                backend=self._backend,
            )

            self._record(
                result
            )

            return result

        except Exception as exc:

            logger.exception(
                "Person detection failed."
            )

            return self._failure(
                timestamp,
                started,
                str(exc),
            )

    # ========================================================================
    # IMAGE PREPARATION
    # ========================================================================

    @staticmethod
    def _get_dimensions(
        image: Any,
    ) -> tuple[int, int]:

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

    @staticmethod
    def _prepare_image(
        image: Any,
    ) -> Any:

        if not NUMPY_AVAILABLE:
            return None

        try:

            array = np.asarray(
                image
            )

            if array.size == 0:
                return None

            if array.ndim == 2:

                if CV2_AVAILABLE:

                    array = cv2.cvtColor(
                        array,
                        cv2.COLOR_GRAY2BGR,
                    )

            elif (
                array.ndim == 3
                and array.shape[2] == 4
            ):

                if CV2_AVAILABLE:

                    array = cv2.cvtColor(
                        array,
                        cv2.COLOR_BGRA2BGR,
                    )

            return array

        except Exception:

            logger.exception(
                "Failed to prepare person "
                "detection image."
            )

            return None

    # ========================================================================
    # BUILD DETECTIONS
    # ========================================================================

    def _build_detections(
        self,
        boxes: Any,
        weights: Any,
        image_width: int,
        image_height: int,
    ) -> list[PersonDetection]:

        detections: list[
            PersonDetection
        ] = []

        for index, box in enumerate(
            boxes
        ):

            try:

                x, y, width, height = (
                    map(
                        int,
                        box,
                    )
                )

                if width < self.config.min_width:
                    continue

                if height < self.config.min_height:
                    continue

                confidence = (
                    self._extract_confidence(
                        weights,
                        index,
                    )
                )

                confidence = (
                    self._normalize_confidence(
                        confidence
                    )
                )

                if (
                    confidence
                    < self.config.confidence_threshold
                ):
                    continue

                bbox = PersonBoundingBox(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )

                visible_ratio = (
                    self._calculate_visible_ratio(
                        bbox,
                        image_width,
                        image_height,
                    )
                )

                detections.append(
                    PersonDetection(
                        box=bbox,
                        confidence=confidence,
                        visible_ratio=(
                            visible_ratio
                        ),
                    )
                )

            except Exception:

                logger.debug(
                    "Invalid person detection.",
                    exc_info=True,
                )

        return self._non_max_suppression(
            detections
        )

    @staticmethod
    def _extract_confidence(
        weights: Any,
        index: int,
    ) -> float:

        try:

            if weights is None:
                return 1.0

            value = weights[index]

            if hasattr(
                value,
                "item",
            ):

                value = value.item()

            if isinstance(
                value,
                (list, tuple),
            ):

                value = value[0]

            return float(
                value
            )

        except Exception:

            return 1.0

    @staticmethod
    def _normalize_confidence(
        confidence: float,
    ) -> float:
        """
        HOG's SVM score is not a probability.

        Convert it into a stable 0..1 confidence-like
        value for RENIX consumers.
        """

        try:

            confidence = float(
                confidence
            )

            # Logistic conversion.
            score = (
                1.0
                / (
                    1.0
                    + math.exp(
                        -confidence
                    )
                )
            )

            return max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

        except Exception:

            return 0.0

    # ========================================================================
    # NON-MAXIMUM SUPPRESSION
    # ========================================================================

    def _non_max_suppression(
        self,
        detections: list[
            PersonDetection
        ],
    ) -> list[PersonDetection]:

        if len(detections) <= 1:
            return detections

        ordered = sorted(
            detections,
            key=lambda item:
            item.confidence,
            reverse=True,
        )

        selected: list[
            PersonDetection
        ] = []

        for candidate in ordered:

            keep = True

            for existing in selected:

                if (
                    candidate.box.iou(
                        existing.box
                    )
                    > self.config.nms_threshold
                ):

                    keep = False
                    break

            if keep:
                selected.append(
                    candidate
                )

        return selected

    # ========================================================================
    # ROI
    # ========================================================================

    def _apply_roi(
        self,
        detections: list[
            PersonDetection
        ],
    ) -> list[PersonDetection]:

        roi = self.config.roi

        if roi is None:
            return detections

        rx, ry, rw, rh = roi

        roi_box = PersonBoundingBox(
            rx,
            ry,
            rw,
            rh,
        )

        filtered: list[
            PersonDetection
        ] = []

        for detection in detections:

            intersection = (
                detection.box.intersection(
                    roi_box
                )
            )

            if intersection is None:
                continue

            ratio = (
                intersection.area
                / max(
                    1,
                    detection.box.area,
                )
            )

            if ratio >= 0.25:

                detection.metadata[
                    "roi_overlap"
                ] = ratio

                filtered.append(
                    detection
                )

        return filtered

    # ========================================================================
    # SORTING
    # ========================================================================

    @staticmethod
    def _sort_detections(
        detections: list[
            PersonDetection
        ],
    ) -> list[PersonDetection]:

        return sorted(
            detections,
            key=lambda person: (
                person.box.area
                * person.confidence
            ),
            reverse=True,
        )

    # ========================================================================
    # TRACKING
    # ========================================================================

    def _assign_track_ids(
        self,
        detections: list[
            PersonDetection
        ],
    ) -> None:

        with self._lock:

            previous = (
                self._previous_detections
            )

            used_previous: set[
                int
            ] = set()

            for detection in detections:

                best_index = None

                best_iou = 0.0

                for index, old in enumerate(
                    previous
                ):

                    if index in used_previous:
                        continue

                    iou = detection.box.iou(
                        old.box
                    )

                    if iou > best_iou:

                        best_iou = iou

                        best_index = index

                if (
                    best_index is not None
                    and best_iou >= 0.25
                ):

                    old_detection = (
                        previous[
                            best_index
                        ]
                    )

                    detection.track_id = (
                        old_detection.track_id
                    )

                    used_previous.add(
                        best_index
                    )

                else:

                    detection.track_id = (
                        self._next_track_id
                    )

                    self._next_track_id += 1

            self._previous_detections = [
                PersonDetection(
                    box=PersonBoundingBox(
                        person.box.x,
                        person.box.y,
                        person.box.width,
                        person.box.height,
                    ),
                    confidence=person.confidence,
                    class_name=person.class_name,
                    class_id=person.class_id,
                    track_id=person.track_id,
                    visible_ratio=person.visible_ratio,
                    metadata=dict(
                        person.metadata
                    ),
                )
                for person in detections
            ]

    def reset_tracking(
        self,
    ) -> None:

        with self._lock:

            self._previous_detections.clear()

            self._next_track_id = 1

    # ========================================================================
    # GEOMETRY HELPERS
    # ========================================================================

    @staticmethod
    def _calculate_visible_ratio(
        box: PersonBoundingBox,
        image_width: int,
        image_height: int,
    ) -> float:

        if box.area <= 0:
            return 0.0

        image_box = PersonBoundingBox(
            0,
            0,
            image_width,
            image_height,
        )

        intersection = (
            box.intersection(
                image_box
            )
        )

        if intersection is None:
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                intersection.area
                / box.area,
            ),
        )

    # ========================================================================
    # PERSON LOCATION
    # ========================================================================

    def get_primary_person(
        self,
        image: Any,
    ) -> Optional[
        PersonDetection
    ]:

        result = self.detect(
            image
        )

        return result.primary_person

    def person_count(
        self,
        image: Any,
    ) -> int:

        return self.detect(
            image
        ).count

    def is_person_present(
        self,
        image: Any,
    ) -> bool:

        return self.person_count(
            image
        ) > 0

    # ========================================================================
    # RESULT / STATISTICS
    # ========================================================================

    def _record(
        self,
        result: PersonDetectionResult,
    ) -> None:

        with self._lock:

            self._last_result = result

            self._detection_calls += 1

            self._total_people += (
                result.count
            )

            self._total_processing_time += (
                result.processing_time
            )

    def _failure(
        self,
        timestamp: float,
        started: float,
        error: str,
        width: int = 0,
        height: int = 0,
    ) -> PersonDetectionResult:

        result = PersonDetectionResult(
            people=[],
            timestamp=timestamp,
            processing_time=(
                time.perf_counter()
                - started
            ),
            image_width=width,
            image_height=height,
            success=False,
            error=error,
            backend=self._backend,
        )

        self._record(
            result
        )

        return result

    def get_last_result(
        self,
    ) -> Optional[
        PersonDetectionResult
    ]:

        with self._lock:

            return self._last_result

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            calls = (
                self._detection_calls
            )

            average_time = (
                self._total_processing_time
                / calls
                if calls
                else 0.0
            )

            average_people = (
                self._total_people
                / calls
                if calls
                else 0.0
            )

            return {
                "backend": self._backend,
                "available": self.is_available(),
                "detection_calls": calls,
                "total_people": (
                    self._total_people
                ),
                "average_people_per_frame": (
                    average_people
                ),
                "average_processing_time": (
                    average_time
                ),
                "confidence_threshold": (
                    self.config.confidence_threshold
                ),
                "tracking_enabled": (
                    self.config.enable_tracking
                ),
            }

    def reset_statistics(
        self,
    ) -> None:

        with self._lock:

            self._detection_calls = 0

            self._total_people = 0

            self._total_processing_time = 0.0

    def reset(
        self,
    ) -> None:

        with self._lock:

            self.reset_tracking()

            self.reset_statistics()

            self._last_result = None

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_confidence_threshold(
        self,
        threshold: float,
    ) -> None:

        threshold = float(
            threshold
        )

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "Confidence threshold must "
                "be between 0 and 1."
            )

        with self._lock:

            self.config.confidence_threshold = (
                threshold
            )

    def set_max_people(
        self,
        maximum: int,
    ) -> None:

        maximum = int(
            maximum
        )

        if maximum < 1:

            raise ValueError(
                "Maximum people must be at least 1."
            )

        with self._lock:

            self.config.max_people = (
                maximum
            )

    def set_roi(
        self,
        roi: Optional[
            tuple[int, int, int, int]
        ],
    ) -> None:

        if roi is not None:

            if len(roi) != 4:

                raise ValueError(
                    "ROI must contain "
                    "(x, y, width, height)."
                )

            x, y, width, height = (
                map(
                    int,
                    roi,
                )
            )

            if width <= 0 or height <= 0:

                raise ValueError(
                    "ROI width and height "
                    "must be positive."
                )

            roi = (
                x,
                y,
                width,
                height,
            )

        with self._lock:

            self.config.roi = roi


# ============================================================================
# SHARED INSTANCE
# ============================================================================


_default_detector: Optional[
    PersonDetector
] = None

_default_detector_lock = (
    threading.RLock()
)


def get_person_detector() -> PersonDetector:
    """
    Return the shared RENIX person detector.
    """

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                PersonDetector()
            )

        return _default_detector


# ============================================================================
# CONVENIENCE API
# ============================================================================


def detect_people(
    image: Any,
) -> PersonDetectionResult:
    """
    Detect people using the shared RENIX detector.
    """

    return get_person_detector().detect(
        image
    )


def person_present(
    image: Any,
) -> bool:
    """
    Return True if at least one person is detected.
    """

    return get_person_detector().is_person_present(
        image
    )


def get_primary_person(
    image: Any,
) -> Optional[
    PersonDetection
]:
    """
    Return the most prominent detected person.
    """

    return get_person_detector().get_primary_person(
        image
    )


def get_person_count(
    image: Any,
) -> int:
    """
    Return number of detected people.
    """

    return get_person_detector().person_count(
        image
    )


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "PersonBoundingBox",
    "PersonDetection",
    "PersonDetectionResult",
    "PersonDetectionConfig",
    "PersonDetector",
    "get_person_detector",
    "detect_people",
    "person_present",
    "get_primary_person",
    "get_person_count",
    "CV2_AVAILABLE",
    "NUMPY_AVAILABLE",
]


