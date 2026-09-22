"""
RENIX Vision — Object Detection

General-purpose object detection layer for RENIX.

Responsibilities:
- Detect objects in camera frames, screenshots, and images.
- Return bounding boxes, labels, confidence scores, and class IDs.
- Support configurable confidence thresholds.
- Support region-of-interest filtering.
- Support non-maximum suppression.
- Support tracking IDs across consecutive frames.
- Provide a clean backend abstraction so a YOLO/ONNX/TensorRT
  backend can be plugged in later without changing RENIX callers.
- Fail gracefully when an optional detection backend is unavailable.

The module does NOT perform:
- Face recognition
- Person-specific identity recognition
- OCR
- Gesture recognition

Those responsibilities belong to their dedicated RENIX modules.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

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
    "RENIX.vision.object_detection"
)


# ============================================================================
# TYPES
# ============================================================================


@dataclass
class BoundingBox:
    """
    Generic object bounding box.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        self.x = int(self.x)
        self.y = int(self.y)
        self.width = max(0, int(self.width))
        self.height = max(0, int(self.height))

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0

        return self.width / self.height

    def intersection(
        self,
        other: "BoundingBox",
    ) -> Optional["BoundingBox"]:

        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)

        if right <= left or bottom <= top:
            return None

        return BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )

    def iou(
        self,
        other: "BoundingBox",
    ) -> float:

        intersection = self.intersection(
            other
        )

        if intersection is None:
            return 0.0

        union = (
            self.area
            + other.area
            - intersection.area
        )

        if union <= 0:
            return 0.0

        return intersection.area / union

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:

        return (
            self.x <= x <= self.x2
            and self.y <= y <= self.y2
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
class DetectedObject:
    """
    One detected object.
    """

    label: str
    confidence: float
    box: BoundingBox

    class_id: int = -1

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
            "label": self.label,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "track_id": self.track_id,
            "box": self.box.to_dict(),
            "visible_ratio": self.visible_ratio,
            "metadata": dict(self.metadata),
        }


@dataclass
class DetectionResult:
    """
    Complete object detection result.
    """

    objects: list[DetectedObject]

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
        return len(self.objects)

    @property
    def labels(self) -> list[str]:
        return [
            obj.label
            for obj in self.objects
        ]

    def get(
        self,
        label: str,
    ) -> list[DetectedObject]:

        normalized = label.strip().lower()

        return [
            obj
            for obj in self.objects
            if obj.label.lower()
            == normalized
        ]

    def contains(
        self,
        label: str,
    ) -> bool:

        normalized = label.strip().lower()

        return any(
            obj.label.lower()
            == normalized
            for obj in self.objects
        )

    def primary_object(
        self,
    ) -> Optional[DetectedObject]:

        if not self.objects:
            return None

        return max(
            self.objects,
            key=lambda obj:
            obj.confidence * obj.box.area,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [
                obj.to_dict()
                for obj in self.objects
            ],
            "count": self.count,
            "labels": self.labels,
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
class ObjectDetectionConfig:
    """
    Runtime configuration.
    """

    confidence_threshold: float = 0.50

    nms_threshold: float = 0.45

    max_objects: int = 100

    enable_tracking: bool = True

    tracking_iou_threshold: float = 0.30

    roi: Optional[
        tuple[int, int, int, int]
    ] = None

    allowed_labels: Optional[
        set[str]
    ] = None

    blocked_labels: set[str] = field(
        default_factory=set
    )

    backend: str = "auto"

    model_path: Optional[str] = None

    class_names_path: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# BACKEND INTERFACE
# ============================================================================


class DetectionBackend(Protocol):
    """
    Protocol implemented by object detection backends.
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
    ) -> list[DetectedObject]:
        ...


# ============================================================================
# GENERIC BACKEND
# ============================================================================


class GenericBackend:
    """
    Adapter backend.

    This allows an external detector to be plugged into RENIX without
    changing ObjectDetector.

    Example:

        backend = GenericBackend(
            detector=my_yolo_detector,
            name="yolo"
        )
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
    ) -> list[DetectedObject]:

        if self.detector is None:
            return []

        raw = self.detector(
            image,
            confidence_threshold=confidence_threshold,
        )

        return self._normalize(raw)

    def _normalize(
        self,
        raw: Any,
    ) -> list[DetectedObject]:

        if raw is None:
            return []

        if isinstance(raw, DetectedObject):
            return [raw]

        if isinstance(raw, dict):
            raw = [raw]

        result: list[
            DetectedObject
        ] = []

        for item in raw:

            try:

                if isinstance(
                    item,
                    DetectedObject,
                ):

                    result.append(item)
                    continue

                if not isinstance(
                    item,
                    dict,
                ):

                    continue

                label = str(
                    item.get(
                        "label",
                        item.get(
                            "class_name",
                            "unknown",
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

                box_data = item.get(
                    "box",
                    item,
                )

                box = self._parse_box(
                    box_data
                )

                if box is None:
                    continue

                class_id = int(
                    item.get(
                        "class_id",
                        item.get(
                            "class",
                            -1,
                        ),
                    )
                )

                result.append(
                    DetectedObject(
                        label=label,
                        confidence=confidence,
                        class_id=class_id,
                        box=box,
                    )
                )

            except Exception:

                logger.debug(
                    "Unable to normalize "
                    "custom detection.",
                    exc_info=True,
                )

        return result

    @staticmethod
    def _parse_box(
        value: Any,
    ) -> Optional[BoundingBox]:

        if isinstance(
            value,
            BoundingBox,
        ):

            return value

        if isinstance(
            value,
            dict,
        ):

            try:

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

            except Exception:
                return None

        try:

            values = list(value)

            if len(values) == 4:

                x1, y1, x2, y2 = values

                return BoundingBox(
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                )

        except Exception:
            pass

        return None


# ============================================================================
# OBJECT DETECTOR
# ============================================================================


class ObjectDetector:
    """
    RENIX general object detector.

    A custom detector can be injected:

        detector = ObjectDetector(
            backend=GenericBackend(my_detector, "yolo")
        )

    Without an injected backend, this class remains operational in
    graceful fallback mode and returns an empty result instead of crashing
    the complete RENIX system.
    """

    def __init__(
        self,
        config: Optional[
            ObjectDetectionConfig
        ] = None,
        backend: Optional[
            DetectionBackend
        ] = None,
    ) -> None:

        self.config = (
            config
            or ObjectDetectionConfig()
        )

        self._lock = threading.RLock()

        self._backend: Optional[
            DetectionBackend
        ] = backend

        self._initialized = False

        self._last_result: Optional[
            DetectionResult
        ] = None

        self._previous_objects: list[
            DetectedObject
        ] = []

        self._next_track_id = 1

        self._detection_calls = 0

        self._total_objects = 0

        self._total_processing_time = 0.0

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:

        if self._backend is not None:

            self._initialized = True

            logger.info(
                "RENIX object detector initialized "
                "with backend: %s",
                self._backend.name,
            )

            return

        backend_name = (
            self.config.backend.lower().strip()
        )

        if backend_name in {
            "none",
            "disabled",
            "unavailable",
        }:

            self._initialized = True

            return

        # No heavyweight model is silently downloaded.
        #
        # A real YOLO/ONNX backend can be injected later.
        #
        # This keeps RENIX startup deterministic and avoids downloading
        # multi-hundred-MB models automatically.
        self._backend = None

        self._initialized = True

        logger.info(
            "No object detection model backend "
            "configured. Running in fallback mode."
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
    # DETECT
    # ========================================================================

    def detect(
        self,
        image: Any,
    ) -> DetectionResult:

        started = time.perf_counter()

        timestamp = time.time()

        if image is None:

            return self._failure(
                timestamp=timestamp,
                started=started,
                error="Input image is None.",
            )

        width, height = (
            self._get_dimensions(
                image
            )
        )

        if width <= 0 or height <= 0:

            return self._failure(
                timestamp=timestamp,
                started=started,
                error="Invalid image dimensions.",
                width=width,
                height=height,
            )

        if not self.is_available():

            result = DetectionResult(
                objects=[],
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
                        "No object detection "
                        "backend is configured."
                    ),
                },
            )

            self._record(result)

            return result

        try:

            objects = self._backend.detect(
                image,
                self.config.confidence_threshold,
            )

            objects = self._sanitize_objects(
                objects
            )

            objects = self._apply_confidence_filter(
                objects
            )

            objects = self._apply_label_filter(
                objects
            )

            objects = self._apply_roi(
                objects
            )

            objects = self._non_max_suppression(
                objects
            )

            objects = self._sort_objects(
                objects
            )

            if self.config.max_objects > 0:

                objects = objects[
                    : self.config.max_objects
                ]

            self._calculate_visibility(
                objects,
                width,
                height,
            )

            if self.config.enable_tracking:

                self._assign_track_ids(
                    objects
                )

            result = DetectionResult(
                objects=objects,
                timestamp=timestamp,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                image_width=width,
                image_height=height,
                success=True,
                backend=self.backend_name,
            )

            self._record(result)

            return result

        except Exception as exc:

            logger.exception(
                "Object detection failed."
            )

            return self._failure(
                timestamp=timestamp,
                started=started,
                error=str(exc),
                width=width,
                height=height,
            )

    # ========================================================================
    # VALIDATION
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
    def _sanitize_objects(
        objects: Sequence[
            DetectedObject
        ],
    ) -> list[DetectedObject]:

        sanitized: list[
            DetectedObject
        ] = []

        for obj in objects:

            if not isinstance(
                obj,
                DetectedObject,
            ):

                continue

            if obj.box.width <= 0:
                continue

            if obj.box.height <= 0:
                continue

            obj.confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        obj.confidence
                    ),
                ),
            )

            obj.label = (
                str(obj.label).strip()
                or "unknown"
            )

            sanitized.append(obj)

        return sanitized

    def _apply_confidence_filter(
        self,
        objects: list[
            DetectedObject
        ],
    ) -> list[DetectedObject]:

        threshold = (
            self.config.confidence_threshold
        )

        return [
            obj
            for obj in objects
            if obj.confidence >= threshold
        ]

    # ========================================================================
    # LABEL FILTER
    # ========================================================================

    def _apply_label_filter(
        self,
        objects: list[
            DetectedObject
        ],
    ) -> list[DetectedObject]:

        allowed = (
            self.config.allowed_labels
        )

        blocked = {
            label.lower()
            for label in
            self.config.blocked_labels
        }

        result: list[
            DetectedObject
        ] = []

        for obj in objects:

            label = obj.label.lower()

            if allowed is not None:

                normalized_allowed = {
                    item.lower()
                    for item in allowed
                }

                if label not in normalized_allowed:
                    continue

            if label in blocked:
                continue

            result.append(obj)

        return result

    # ========================================================================
    # ROI
    # ========================================================================

    def _apply_roi(
        self,
        objects: list[
            DetectedObject
        ],
    ) -> list[DetectedObject]:

        roi = self.config.roi

        if roi is None:
            return objects

        rx, ry, rw, rh = roi

        roi_box = BoundingBox(
            rx,
            ry,
            rw,
            rh,
        )

        result: list[
            DetectedObject
        ] = []

        for obj in objects:

            intersection = (
                obj.box.intersection(
                    roi_box
                )
            )

            if intersection is None:
                continue

            overlap = (
                intersection.area
                / max(
                    1,
                    obj.box.area,
                )
            )

            if overlap >= 0.25:

                obj.metadata[
                    "roi_overlap"
                ] = overlap

                result.append(obj)

        return result

    # ========================================================================
    # NMS
    # ========================================================================

    def _non_max_suppression(
        self,
        objects: list[
            DetectedObject
        ],
    ) -> list[DetectedObject]:

        if len(objects) <= 1:
            return objects

        grouped: dict[
            str,
            list[DetectedObject]
        ] = {}

        for obj in objects:

            key = obj.label.lower()

            grouped.setdefault(
                key,
                [],
            ).append(obj)

        result: list[
            DetectedObject
        ] = []

        for group in grouped.values():

            ordered = sorted(
                group,
                key=lambda obj:
                obj.confidence,
                reverse=True,
            )

            selected: list[
                DetectedObject
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

            result.extend(
                selected
            )

        return result

    # ========================================================================
    # SORTING
    # ========================================================================

    @staticmethod
    def _sort_objects(
        objects: list[
            DetectedObject
        ],
    ) -> list[DetectedObject]:

        return sorted(
            objects,
            key=lambda obj:
            obj.confidence * obj.box.area,
            reverse=True,
        )

    # ========================================================================
    # VISIBILITY
    # ========================================================================

    @staticmethod
    def _calculate_visibility(
        objects: list[
            DetectedObject
        ],
        image_width: int,
        image_height: int,
    ) -> None:

        image_box = BoundingBox(
            0,
            0,
            image_width,
            image_height,
        )

        for obj in objects:

            if obj.box.area <= 0:

                obj.visible_ratio = 0.0

                continue

            intersection = (
                obj.box.intersection(
                    image_box
                )
            )

            if intersection is None:

                obj.visible_ratio = 0.0

            else:

                obj.visible_ratio = max(
                    0.0,
                    min(
                        1.0,
                        intersection.area
                        / obj.box.area,
                    ),
                )

    # ========================================================================
    # TRACKING
    # ========================================================================

    def _assign_track_ids(
        self,
        objects: list[
            DetectedObject
        ],
    ) -> None:

        with self._lock:

            previous = (
                self._previous_objects
            )

            used_previous: set[
                int
            ] = set()

            for obj in objects:

                best_index: Optional[
                    int
                ] = None

                best_iou = 0.0

                for index, old in enumerate(
                    previous
                ):

                    if index in used_previous:
                        continue

                    if (
                        old.label.lower()
                        != obj.label.lower()
                    ):
                        continue

                    score = obj.box.iou(
                        old.box
                    )

                    if score > best_iou:

                        best_iou = score

                        best_index = index

                if (
                    best_index is not None
                    and best_iou
                    >= self.config.tracking_iou_threshold
                ):

                    old = previous[
                        best_index
                    ]

                    obj.track_id = (
                        old.track_id
                    )

                    used_previous.add(
                        best_index
                    )

                else:

                    obj.track_id = (
                        self._next_track_id
                    )

                    self._next_track_id += 1

            self._previous_objects = [
                DetectedObject(
                    label=obj.label,
                    confidence=obj.confidence,
                    box=BoundingBox(
                        obj.box.x,
                        obj.box.y,
                        obj.box.width,
                        obj.box.height,
                    ),
                    class_id=obj.class_id,
                    track_id=obj.track_id,
                    visible_ratio=obj.visible_ratio,
                    metadata=dict(
                        obj.metadata
                    ),
                )
                for obj in objects
            ]

    def reset_tracking(
        self,
    ) -> None:

        with self._lock:

            self._previous_objects.clear()

            self._next_track_id = 1

    # ========================================================================
    # RECORDING / STATISTICS
    # ========================================================================

    def _record(
        self,
        result: DetectionResult,
    ) -> None:

        with self._lock:

            self._last_result = result

            self._detection_calls += 1

            self._total_objects += (
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
    ) -> DetectionResult:

        result = DetectionResult(
            objects=[],
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

        self._record(result)

        return result

    def get_last_result(
        self,
    ) -> Optional[
        DetectionResult
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

            average_objects = (
                self._total_objects
                / calls
                if calls
                else 0.0
            )

            return {
                "backend": self.backend_name,
                "available": self.is_available(),
                "detection_calls": calls,
                "total_objects": (
                    self._total_objects
                ),
                "average_objects_per_frame": (
                    average_objects
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

            self._total_objects = 0

            self._total_processing_time = 0.0

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def set_backend(
        self,
        backend: Optional[
            DetectionBackend
        ],
    ) -> None:

        with self._lock:

            self._backend = backend

            self.reset_tracking()

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

    def set_nms_threshold(
        self,
        threshold: float,
    ) -> None:

        threshold = float(
            threshold
        )

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                "NMS threshold must "
                "be between 0 and 1."
            )

        with self._lock:

            self.config.nms_threshold = (
                threshold
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
                    "ROI must be "
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
                    "ROI dimensions "
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

    def set_allowed_labels(
        self,
        labels: Optional[
            Iterable[str]
        ],
    ) -> None:

        with self._lock:

            if labels is None:

                self.config.allowed_labels = None

            else:

                self.config.allowed_labels = {
                    str(label).strip()
                    for label in labels
                    if str(label).strip()
                }

    def set_blocked_labels(
        self,
        labels: Iterable[str],
    ) -> None:

        with self._lock:

            self.config.blocked_labels = {
                str(label).strip()
                for label in labels
                if str(label).strip()
            }

    def reset(
        self,
    ) -> None:

        with self._lock:

            self.reset_tracking()

            self.reset_statistics()

            self._last_result = None


# ============================================================================
# SHARED DETECTOR
# ============================================================================


_default_detector: Optional[
    ObjectDetector
] = None

_default_detector_lock = (
    threading.RLock()
)


def get_object_detector() -> ObjectDetector:
    """
    Return RENIX's shared object detector.
    """

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                ObjectDetector()
            )

        return _default_detector


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def detect_objects(
    image: Any,
) -> DetectionResult:
    """
    Detect all supported objects.
    """

    return get_object_detector().detect(
        image
    )


def object_present(
    image: Any,
    label: str,
) -> bool:
    """
    Check whether a particular object exists.
    """

    result = detect_objects(
        image
    )

    return result.contains(
        label
    )


def get_objects(
    image: Any,
    label: Optional[str] = None,
) -> list[DetectedObject]:

    result = detect_objects(
        image
    )

    if label is None:
        return result.objects

    return result.get(
        label
    )


def get_primary_object(
    image: Any,
) -> Optional[DetectedObject]:

    return detect_objects(
        image
    ).primary_object()


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "BoundingBox",
    "DetectedObject",
    "DetectionResult",
    "ObjectDetectionConfig",
    "DetectionBackend",
    "GenericBackend",
    "ObjectDetector",
    "ObjectDetection",
    "get_object_detector",
    "detect_objects",
    "object_present",
    "get_objects",
    "get_primary_object",
    "CV2_AVAILABLE",
    "NUMPY_AVAILABLE",
]


ObjectDetection = ObjectDetector
