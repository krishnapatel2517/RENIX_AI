"""
RENIX Vision — Face Detection

Production-oriented face detection layer for RENIX.

Responsibilities:
- Detect faces from camera frames/images
- Return bounding boxes
- Calculate confidence scores
- Calculate face landmarks when supported
- Track detection timestamps
- Support multiple detection backends
- Provide normalized coordinates
- Provide face crops
- Thread-safe detector configuration
- Graceful fallback when optional vision libraries
  are unavailable

The module is intentionally designed so that the rest of
RENIX can use a stable API even when a particular backend
is not installed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(
    "RENIX.vision.face_detection"
)


# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================

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


# ============================================================================
# DATA TYPES
# ============================================================================


@dataclass
class BoundingBox:
    """Pixel-space bounding box."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> int:
        return max(0, self.width) * max(
            0,
            self.height,
        )

    def clamp(
        self,
        image_width: int,
        image_height: int,
    ) -> "BoundingBox":
        """Clamp the box to image boundaries."""

        x1 = max(
            0,
            min(self.x, image_width),
        )

        y1 = max(
            0,
            min(self.y, image_height),
        )

        x2 = max(
            x1,
            min(self.x2, image_width),
        )

        y2 = max(
            y1,
            min(self.y2, image_height),
        )

        return BoundingBox(
            x=int(x1),
            y=int(y1),
            width=int(x2 - x1),
            height=int(y2 - y1),
        )

    def normalized(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        """Return x, y, width, height normalized to 0..1."""

        if image_width <= 0 or image_height <= 0:
            return (0.0, 0.0, 0.0, 0.0)

        return (
            self.x / image_width,
            self.y / image_height,
            self.width / image_width,
            self.height / image_height,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "x2": self.x2,
            "y2": self.y2,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": self.area,
        }


@dataclass
class FaceLandmark:
    """A facial landmark point."""

    x: float
    y: float
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
        }


@dataclass
class FaceDetection:
    """One detected face."""

    box: BoundingBox

    confidence: float

    landmarks: list[
        FaceLandmark
    ] = field(default_factory=list)

    face_id: Optional[str] = None

    timestamp: float = field(
        default_factory=time.time
    )

    image_width: int = 0
    image_height: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def center(self) -> tuple[float, float]:
        return (
            self.box.center_x,
            self.box.center_y,
        )

    @property
    def normalized_box(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        return self.box.normalized(
            self.image_width,
            self.image_height,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "confidence": self.confidence,
            "face_id": self.face_id,
            "timestamp": self.timestamp,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "normalized_box": self.normalized_box,
            "landmarks": [
                landmark.to_dict()
                for landmark in self.landmarks
            ],
            "metadata": dict(self.metadata),
        }


@dataclass
class FaceDetectionResult:
    """Result returned by the face detector."""

    faces: list[
        FaceDetection
    ]

    timestamp: float

    image_width: int

    image_height: int

    processing_time: float

    backend: str

    success: bool = True

    error: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def count(self) -> int:
        return len(self.faces)

    @property
    def largest_face(
        self,
    ) -> Optional[FaceDetection]:

        if not self.faces:
            return None

        return max(
            self.faces,
            key=lambda face: face.box.area,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "faces": [
                face.to_dict()
                for face in self.faces
            ],
            "count": self.count,
            "timestamp": self.timestamp,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "processing_time": self.processing_time,
            "backend": self.backend,
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass
class FaceDetectorConfig:
    """Configuration for face detection."""

    backend: str = "auto"

    scale_factor: float = 1.1

    min_neighbors: int = 5

    min_width: int = 30

    min_height: int = 30

    max_faces: int = 50

    confidence_threshold: float = 0.0

    grayscale: bool = True

    equalize_histogram: bool = False

    detect_landmarks: bool = False

    return_crops: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# FACE DETECTOR
# ============================================================================


class FaceDetector:
    """
    RENIX face detector.

    The default implementation uses OpenCV's Haar cascade
    when available. The architecture is deliberately
    backend-independent so more advanced detectors can be
    added later without changing the RENIX vision API.

    Example
    -------
    ::

        detector = FaceDetector()

        result = detector.detect(frame)

        for face in result.faces:
            print(face.box)
    """

    def __init__(
        self,
        config: Optional[
            FaceDetectorConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or FaceDetectorConfig()
        )

        self._lock = threading.RLock()

        self._classifier: Any = None

        self._backend = "unavailable"

        self._initialized = False

        self._last_result: Optional[
            FaceDetectionResult
        ] = None

        self._total_detections = 0

        self._detection_calls = 0

        self._total_processing_time = 0.0

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:
        """Initialize the selected detection backend."""

        with self._lock:

            self._classifier = None

            self._backend = "unavailable"

            if not CV2_AVAILABLE:

                logger.warning(
                    "OpenCV is unavailable; "
                    "face detection is disabled."
                )

                self._initialized = False

                return

            backend = (
                self.config.backend
                or "auto"
            ).lower()

            if backend in {
                "auto",
                "opencv",
                "haar",
                "haarcascade",
            }:

                if self._initialize_opencv():

                    self._backend = (
                        "opencv_haar"
                    )

                    self._initialized = True

                    return

            logger.warning(
                "No usable face detection "
                "backend is available."
            )

            self._initialized = False

    def _initialize_opencv(
        self,
    ) -> bool:

        if not CV2_AVAILABLE:
            return False

        try:

            cascade_path = (
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )

            classifier = (
                cv2.CascadeClassifier(
                    cascade_path
                )
            )

            if classifier.empty():

                logger.error(
                    "OpenCV face cascade "
                    "could not be loaded."
                )

                return False

            self._classifier = classifier

            return True

        except Exception:

            logger.exception(
                "Failed to initialize "
                "OpenCV face detector."
            )

            return False

    # ========================================================================
    # AVAILABILITY
    # ========================================================================

    @staticmethod
    def library_available() -> bool:
        return CV2_AVAILABLE

    def is_available(self) -> bool:

        with self._lock:

            return bool(
                self._initialized
                and self._classifier is not None
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
    ) -> FaceDetectionResult:
        """
        Detect faces in an image.

        Parameters
        ----------
        image:
            OpenCV/Numpy image.

        Returns
        -------
        FaceDetectionResult
        """

        started = time.perf_counter()

        timestamp = time.time()

        if image is None:

            return self._failure_result(
                timestamp,
                started,
                "Input image is None.",
            )

        dimensions = self._get_dimensions(
            image
        )

        if dimensions is None:

            return self._failure_result(
                timestamp,
                started,
                "Unable to determine image dimensions.",
            )

        width, height = dimensions

        if not self.is_available():

            return self._failure_result(
                timestamp,
                started,
                "Face detector backend is unavailable.",
                width=width,
                height=height,
            )

        try:

            prepared = (
                self._prepare_image(
                    image
                )
            )

            detections = (
                self._detect_opencv(
                    prepared
                )
            )

            faces: list[
                FaceDetection
            ] = []

            for index, (
                x,
                y,
                w,
                h,
            ) in enumerate(
                detections
            ):

                box = BoundingBox(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                ).clamp(
                    width,
                    height,
                )

                if (
                    box.width <= 0
                    or box.height <= 0
                ):
                    continue

                confidence = (
                    self._estimate_confidence(
                        box,
                        width,
                        height,
                    )
                )

                if (
                    confidence
                    < self.config.confidence_threshold
                ):
                    continue

                landmarks: list[
                    FaceLandmark
                ] = []

                if (
                    self.config.detect_landmarks
                ):

                    landmarks = (
                        self._estimate_landmarks(
                            box
                        )
                    )

                face = FaceDetection(
                    box=box,
                    confidence=confidence,
                    landmarks=landmarks,
                    face_id=f"face_{index}",
                    timestamp=timestamp,
                    image_width=width,
                    image_height=height,
                    metadata={
                        "backend": self._backend,
                    },
                )

                if (
                    self.config.return_crops
                ):

                    crop = self._crop_face(
                        image,
                        box,
                    )

                    if crop is not None:

                        face.metadata[
                            "crop"
                        ] = crop

                faces.append(face)

                if (
                    len(faces)
                    >= self.config.max_faces
                ):
                    break

            processing_time = (
                time.perf_counter()
                - started
            )

            result = FaceDetectionResult(
                faces=faces,
                timestamp=timestamp,
                image_width=width,
                image_height=height,
                processing_time=processing_time,
                backend=self._backend,
                success=True,
            )

            self._record_result(
                result
            )

            return result

        except Exception as exc:

            logger.exception(
                "Face detection failed."
            )

            return self._failure_result(
                timestamp,
                started,
                str(exc),
                width=width,
                height=height,
            )

    def detect_faces(
        self,
        image: Any,
    ) -> list[FaceDetection]:
        """Convenience method returning only detected faces."""

        return self.detect(
            image
        ).faces

    def count_faces(
        self,
        image: Any,
    ) -> int:
        """Return the number of detected faces."""

        return len(
            self.detect_faces(
                image
            )
        )

    # ========================================================================
    # OPENCV BACKEND
    # ========================================================================

    def _detect_opencv(
        self,
        image: Any,
    ) -> list[Any]:

        if self._classifier is None:

            return []

        faces = (
            self._classifier.detectMultiScale(
                image,
                scaleFactor=max(
                    1.01,
                    float(
                        self.config.scale_factor
                    ),
                ),
                minNeighbors=max(
                    0,
                    int(
                        self.config.min_neighbors
                    ),
                ),
                minSize=(
                    max(
                        1,
                        self.config.min_width,
                    ),
                    max(
                        1,
                        self.config.min_height,
                    ),
                ),
            )
        )

        if faces is None:
            return []

        return list(faces)

    # ========================================================================
    # IMAGE PREPARATION
    # ========================================================================

    def _prepare_image(
        self,
        image: Any,
    ) -> Any:

        if not CV2_AVAILABLE:
            return image

        prepared = image

        if (
            self.config.grayscale
        ):

            if (
                hasattr(
                    prepared,
                    "ndim",
                )
                and prepared.ndim == 3
            ):

                prepared = cv2.cvtColor(
                    prepared,
                    cv2.COLOR_BGR2GRAY,
                )

        if (
            self.config.equalize_histogram
        ):

            if (
                hasattr(
                    prepared,
                    "ndim",
                )
                and prepared.ndim == 2
            ):

                prepared = cv2.equalizeHist(
                    prepared
                )

        return prepared

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _estimate_confidence(
        self,
        box: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Estimate a normalized confidence score.

        Haar cascades do not provide a true probability
        score. This value is therefore a heuristic score
        intended for ranking/filtering rather than a
        calibrated probability.
        """

        if (
            image_width <= 0
            or image_height <= 0
        ):
            return 0.0

        area_ratio = (
            box.area
            / float(
                image_width
                * image_height
            )
        )

        size_score = min(
            1.0,
            math.sqrt(
                max(
                    0.0,
                    area_ratio,
                )
            )
            * 3.0,
        )

        center_x = box.center_x
        center_y = box.center_y

        normalized_x = (
            center_x
            / image_width
        )

        normalized_y = (
            center_y
            / image_height
        )

        center_distance = math.sqrt(
            (
                normalized_x
                - 0.5
            )
            ** 2
            + (
                normalized_y
                - 0.5
            )
            ** 2
        )

        center_score = max(
            0.0,
            1.0
            - center_distance
            * 1.5,
        )

        confidence = (
            0.55 * size_score
            + 0.45 * center_score
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # ========================================================================
    # LANDMARKS
    # ========================================================================

    def _estimate_landmarks(
        self,
        box: BoundingBox,
    ) -> list[FaceLandmark]:
        """
        Provide coarse geometric facial landmarks.

        This fallback does not claim true facial landmark
        detection. It supplies useful anchor points until
        a dedicated landmark backend is installed.
        """

        x = float(box.x)
        y = float(box.y)

        w = float(box.width)
        h = float(box.height)

        return [
            FaceLandmark(
                x=x + w * 0.30,
                y=y + h * 0.38,
                name="left_eye",
            ),
            FaceLandmark(
                x=x + w * 0.70,
                y=y + h * 0.38,
                name="right_eye",
            ),
            FaceLandmark(
                x=x + w * 0.50,
                y=y + h * 0.53,
                name="nose",
            ),
            FaceLandmark(
                x=x + w * 0.35,
                y=y + h * 0.72,
                name="left_mouth",
            ),
            FaceLandmark(
                x=x + w * 0.65,
                y=y + h * 0.72,
                name="right_mouth",
            ),
            FaceLandmark(
                x=x + w * 0.50,
                y=y + h * 0.74,
                name="mouth_center",
            ),
        ]

    # ========================================================================
    # FACE CROPS
    # ========================================================================

    def _crop_face(
        self,
        image: Any,
        box: BoundingBox,
    ) -> Any:

        try:

            return image[
                box.y : box.y2,
                box.x : box.x2,
            ].copy()

        except Exception:

            logger.debug(
                "Unable to crop detected face.",
                exc_info=True,
            )

            return None

    def crop_faces(
        self,
        image: Any,
    ) -> list[Any]:
        """Return image crops for every detected face."""

        result = self.detect(
            image
        )

        crops: list[Any] = []

        for face in result.faces:

            crop = self._crop_face(
                image,
                face.box,
            )

            if crop is not None:
                crops.append(crop)

        return crops

    # ========================================================================
    # DIMENSIONS
    # ========================================================================

    @staticmethod
    def _get_dimensions(
        image: Any,
    ) -> Optional[
        tuple[int, int]
    ]:

        try:

            if not hasattr(
                image,
                "shape",
            ):
                return None

            shape = image.shape

            if len(shape) < 2:
                return None

            height = int(
                shape[0]
            )

            width = int(
                shape[1]
            )

            if (
                width <= 0
                or height <= 0
            ):
                return None

            return width, height

        except Exception:

            return None

    # ========================================================================
    # RESULT MANAGEMENT
    # ========================================================================

    def _record_result(
        self,
        result: FaceDetectionResult,
    ) -> None:

        with self._lock:

            self._last_result = result

            self._detection_calls += 1

            self._total_detections += (
                result.count
            )

            self._total_processing_time += (
                result.processing_time
            )

    def _failure_result(
        self,
        timestamp: float,
        started: float,
        error: str,
        *,
        width: int = 0,
        height: int = 0,
    ) -> FaceDetectionResult:

        processing_time = (
            time.perf_counter()
            - started
        )

        result = FaceDetectionResult(
            faces=[],
            timestamp=timestamp,
            image_width=width,
            image_height=height,
            processing_time=processing_time,
            backend=self._backend,
            success=False,
            error=error,
        )

        with self._lock:

            self._last_result = result

            self._detection_calls += 1

            self._total_processing_time += (
                processing_time
            )

        return result

    def get_last_result(
        self,
    ) -> Optional[FaceDetectionResult]:

        with self._lock:

            return self._last_result

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            calls = self._detection_calls

            average_time = (
                self._total_processing_time
                / calls
                if calls > 0
                else 0.0
            )

            average_faces = (
                self._total_detections
                / calls
                if calls > 0
                else 0.0
            )

            return {
                "backend": self._backend,
                "available": self.is_available(),
                "detection_calls": calls,
                "total_faces": (
                    self._total_detections
                ),
                "average_faces_per_call": (
                    average_faces
                ),
                "average_processing_time": (
                    average_time
                ),
                "last_result": (
                    self._last_result.to_dict()
                    if self._last_result
                    else None
                ),
            }

    def reset_statistics(
        self,
    ) -> None:

        with self._lock:

            self._detection_calls = 0

            self._total_detections = 0

            self._total_processing_time = 0.0

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def update_config(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Update detector configuration.

        Example:

            detector.update_config(
                scale_factor=1.15,
                min_neighbors=6,
            )
        """

        with self._lock:

            for key, value in kwargs.items():

                if not hasattr(
                    self.config,
                    key,
                ):

                    raise AttributeError(
                        f"Unknown face detector "
                        f"configuration: {key}"
                    )

                setattr(
                    self.config,
                    key,
                    value,
                )

            self._initialize()

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(
        self,
    ) -> None:

        with self._lock:

            self._last_result = None

            self.reset_statistics()

            self._initialize()


# ============================================================================
# DEFAULT DETECTOR
# ============================================================================


_default_detector: Optional[
    FaceDetector
] = None

_default_detector_lock = (
    threading.RLock()
)


def get_face_detector() -> FaceDetector:
    """Return the shared RENIX face detector."""

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                FaceDetector()
            )

        return _default_detector


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def detect_faces(
    image: Any,
) -> list[FaceDetection]:
    """Detect faces using the default detector."""

    return get_face_detector().detect_faces(
        image
    )


def detect(
    image: Any,
) -> FaceDetectionResult:
    """Run face detection using the default detector."""

    return get_face_detector().detect(
        image
    )


def count_faces(
    image: Any,
) -> int:
    """Count faces using the default detector."""

    return get_face_detector().count_faces(
        image
    )


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "BoundingBox",
    "FaceLandmark",
    "FaceDetection",
    "FaceDetectionResult",
    "FaceDetectorConfig",
    "FaceDetector",
    "get_face_detector",
    "detect_faces",
    "detect",
    "count_faces",
    "CV2_AVAILABLE",
    "NUMPY_AVAILABLE",
]


