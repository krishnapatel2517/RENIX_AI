"""
RENIX Vision — Document Detection

Detects document-like regions in camera frames and screenshots.

Responsibilities:
- Detect rectangular document surfaces
- Find paper/document contours
- Estimate document corners
- Estimate document confidence
- Identify document orientation
- Support perspective correction
- Detect document-like layouts
- Provide structured results to OCR and scene understanding
- Gracefully degrade when OpenCV is unavailable

This module does NOT perform OCR.
OCR belongs to vision/ocr.py.
"""

from __future__ import annotations

import logging
import math
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

from .object_detection import BoundingBox


logger = logging.getLogger(
    "RENIX.vision.document_detection"
)


# ============================================================================
# ENUMS
# ============================================================================


class DocumentType(str, Enum):
    UNKNOWN = "unknown"
    PAPER = "paper"
    PAGE = "page"
    BOOK = "book"
    RECEIPT = "receipt"
    CARD = "card"
    ID_CARD = "id_card"
    NOTE = "note"
    SCREEN = "screen"
    WHITEBOARD = "whiteboard"


class DocumentOrientation(str, Enum):
    UNKNOWN = "unknown"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SQUARE = "square"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class DocumentCorners:
    """
    Four document corners.

    Order:
        top-left
        top-right
        bottom-right
        bottom-left
    """

    top_left: tuple[float, float]

    top_right: tuple[float, float]

    bottom_right: tuple[float, float]

    bottom_left: tuple[float, float]

    def as_array(self):
        if np is None:
            return [
                self.top_left,
                self.top_right,
                self.bottom_right,
                self.bottom_left,
            ]

        return np.array(
            [
                self.top_left,
                self.top_right,
                self.bottom_right,
                self.bottom_left,
            ],
            dtype="float32",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_left": self.top_left,
            "top_right": self.top_right,
            "bottom_right": self.bottom_right,
            "bottom_left": self.bottom_left,
        }


@dataclass
class DocumentDetection:
    """
    One detected document.
    """

    document_id: str

    document_type: DocumentType = (
        DocumentType.UNKNOWN
    )

    confidence: float = 0.0

    box: Optional[BoundingBox] = None

    corners: Optional[
        DocumentCorners
    ] = None

    orientation: DocumentOrientation = (
        DocumentOrientation.UNKNOWN
    )

    area_ratio: float = 0.0

    rectangularity: float = 0.0

    aspect_ratio: float = 0.0

    skew_angle: float = 0.0

    text_likelihood: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_type": (
                self.document_type.value
            ),
            "confidence": self.confidence,
            "box": (
                self.box.to_dict()
                if self.box is not None
                else None
            ),
            "corners": (
                self.corners.to_dict()
                if self.corners is not None
                else None
            ),
            "orientation": (
                self.orientation.value
            ),
            "area_ratio": self.area_ratio,
            "rectangularity": self.rectangularity,
            "aspect_ratio": self.aspect_ratio,
            "skew_angle": self.skew_angle,
            "text_likelihood": (
                self.text_likelihood
            ),
            "metadata": dict(self.metadata),
        }


@dataclass
class DocumentDetectionResult:
    """
    Result from document detection.
    """

    documents: list[
        DocumentDetection
    ] = field(default_factory=list)

    image_width: int = 0

    image_height: int = 0

    processing_time: float = 0.0

    success: bool = True

    error: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def detected(self) -> bool:
        return bool(self.documents)

    @property
    def count(self) -> int:
        return len(self.documents)

    @property
    def primary(
        self,
    ) -> Optional[DocumentDetection]:

        if not self.documents:
            return None

        return max(
            self.documents,
            key=lambda item: item.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": [
                document.to_dict()
                for document in self.documents
            ],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "processing_time": self.processing_time,
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass
class DocumentDetectionConfig:
    """
    Configuration for document detection.
    """

    min_area_ratio: float = 0.05

    max_area_ratio: float = 0.98

    min_rectangularity: float = 0.65

    min_confidence: float = 0.40

    max_documents: int = 10

    blur_kernel: int = 5

    canny_low: int = 50

    canny_high: int = 150

    approximation_epsilon: float = 0.02

    detect_nested_documents: bool = False

    use_adaptive_threshold: bool = True

    classify_document_type: bool = True

    estimate_corners: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# DOCUMENT DETECTOR
# ============================================================================


class DocumentDetector:
    """
    RENIX document detection engine.
    """

    def __init__(
        self,
        config: Optional[
            DocumentDetectionConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or DocumentDetectionConfig()
        )

        self._lock = threading.RLock()

        self._last_result: Optional[
            DocumentDetectionResult
        ] = None

        self._frame_counter = 0

        logger.info(
            "RENIX document detector initialized."
        )

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def detect(
        self,
        image: Any,
    ) -> DocumentDetectionResult:

        started = time.perf_counter()

        width, height = (
            self._get_dimensions(image)
        )

        with self._lock:
            self._frame_counter += 1

        if image is None:

            return self._failure(
                started,
                width,
                height,
                "Image is None.",
            )

        if cv2 is None:

            return self._failure(
                started,
                width,
                height,
                "OpenCV is not installed.",
            )

        try:

            processed = self._preprocess(
                image
            )

            contours = (
                self._find_contours(
                    processed
                )
            )

            documents: list[
                DocumentDetection
            ] = []

            image_area = (
                width * height
            )

            if image_area <= 0:
                image_area = 1

            for index, contour in enumerate(
                contours
            ):

                detection = (
                    self._analyze_contour(
                        contour,
                        image_area,
                        width,
                        height,
                        index,
                    )
                )

                if detection is None:
                    continue

                if (
                    detection.confidence
                    < self.config.min_confidence
                ):
                    continue

                documents.append(
                    detection
                )

                if (
                    self.config.max_documents
                    > 0
                    and len(documents)
                    >= self.config.max_documents
                ):
                    break

            documents = (
                self._remove_duplicates(
                    documents
                )
            )

            documents.sort(
                key=lambda item: item.confidence,
                reverse=True,
            )

            if (
                self.config.max_documents
                > 0
            ):
                documents = documents[
                    : self.config.max_documents
                ]

            result = (
                DocumentDetectionResult(
                    documents=documents,
                    image_width=width,
                    image_height=height,
                    processing_time=(
                        time.perf_counter()
                        - started
                    ),
                    success=True,
                    metadata={
                        "frame_number": (
                            self._frame_counter
                        ),
                        "contour_count": len(
                            contours
                        ),
                    },
                )
            )

            self._store_result(result)

            return result

        except Exception as exc:

            logger.exception(
                "Document detection failed."
            )

            return self._failure(
                started,
                width,
                height,
                str(exc),
            )

    # ========================================================================
    # DOCUMENT PRESENCE
    # ========================================================================

    def is_document_present(
        self,
        image: Any,
    ) -> bool:

        result = self.detect(image)

        return result.detected

    # ========================================================================
    # PRIMARY DOCUMENT
    # ========================================================================

    def detect_primary(
        self,
        image: Any,
    ) -> Optional[DocumentDetection]:

        result = self.detect(image)

        return result.primary

    # ========================================================================
    # PREPROCESSING
    # ========================================================================

    def _preprocess(
        self,
        image: Any,
    ) -> Any:

        if cv2 is None:
            return image

        processed = image

        if len(
            processed.shape
        ) >= 3:

            gray = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2GRAY,
            )

        else:

            gray = processed

        kernel = int(
            self.config.blur_kernel
        )

        if kernel < 3:
            kernel = 3

        if kernel % 2 == 0:
            kernel += 1

        gray = cv2.GaussianBlur(
            gray,
            (kernel, kernel),
            0,
        )

        if (
            self.config.use_adaptive_threshold
        ):

            thresholded = (
                cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    21,
                    5,
                )
            )

            return thresholded

        return cv2.Canny(
            gray,
            self.config.canny_low,
            self.config.canny_high,
        )

    # ========================================================================
    # CONTOURS
    # ========================================================================

    def _find_contours(
        self,
        image: Any,
    ) -> list[Any]:

        if cv2 is None:
            return []

        contours, _ = cv2.findContours(
            image,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

    # ========================================================================
    # CONTOUR ANALYSIS
    # ========================================================================

    def _analyze_contour(
        self,
        contour: Any,
        image_area: int,
        image_width: int,
        image_height: int,
        index: int,
    ) -> Optional[
        DocumentDetection
    ]:

        if cv2 is None:
            return None

        contour_area = float(
            cv2.contourArea(contour)
        )

        if contour_area <= 0:
            return None

        area_ratio = (
            contour_area
            / float(image_area)
        )

        if (
            area_ratio
            < self.config.min_area_ratio
        ):
            return None

        if (
            area_ratio
            > self.config.max_area_ratio
        ):
            return None

        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        if perimeter <= 0:
            return None

        epsilon = (
            self.config.approximation_epsilon
            * perimeter
        )

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        if len(polygon) != 4:
            return None

        corners = self._order_corners(
            polygon
        )

        if corners is None:
            return None

        box = self._corners_to_box(
            corners
        )

        if box is None:
            return None

        box_area = (
            box.width
            * box.height
        )

        if box_area <= 0:
            return None

        rectangularity = (
            contour_area
            / box_area
        )

        rectangularity = max(
            0.0,
            min(
                1.0,
                rectangularity,
            ),
        )

        if (
            rectangularity
            < self.config.min_rectangularity
        ):
            return None

        aspect_ratio = (
            max(
                box.width,
                box.height,
            )
            / max(
                1.0,
                min(
                    box.width,
                    box.height,
                ),
            )
        )

        orientation = (
            self._orientation(
                box
            )
        )

        skew_angle = (
            self._estimate_skew(
                corners
            )
        )

        text_likelihood = (
            self._estimate_text_likelihood(
                contour,
                image_width,
                image_height,
            )
        )

        document_type = (
            self._classify_document_type(
                box=box,
                aspect_ratio=aspect_ratio,
                rectangularity=(
                    rectangularity
                ),
                text_likelihood=(
                    text_likelihood
                ),
            )
        )

        confidence = (
            self._calculate_confidence(
                area_ratio=area_ratio,
                rectangularity=(
                    rectangularity
                ),
                text_likelihood=(
                    text_likelihood
                ),
                skew_angle=skew_angle,
            )
        )

        return DocumentDetection(
            document_id=(
                f"document_{index}"
            ),
            document_type=document_type,
            confidence=confidence,
            box=box,
            corners=(
                corners
                if self.config.estimate_corners
                else None
            ),
            orientation=orientation,
            area_ratio=area_ratio,
            rectangularity=rectangularity,
            aspect_ratio=aspect_ratio,
            skew_angle=skew_angle,
            text_likelihood=text_likelihood,
            metadata={
                "contour_area": contour_area,
                "perimeter": perimeter,
                "polygon_points": len(
                    polygon
                ),
            },
        )

    # ========================================================================
    # CORNER PROCESSING
    # ========================================================================

    @staticmethod
    def _order_corners(
        polygon: Any,
    ) -> Optional[
        DocumentCorners
    ]:

        if polygon is None:
            return None

        try:

            points = polygon.reshape(
                4,
                2,
            ).astype(float)

        except Exception:
            return None

        if len(points) != 4:
            return None

        sums = (
            points[:, 0]
            + points[:, 1]
        )

        differences = (
            points[:, 0]
            - points[:, 1]
        )

        top_left = points[
            differences.argmin()
        ]

        bottom_right = points[
            differences.argmax()
        ]

        top_right = points[
            sums.argmin()
        ]

        bottom_left = points[
            sums.argmax()
        ]

        return DocumentCorners(
            top_left=(
                float(top_left[0]),
                float(top_left[1]),
            ),
            top_right=(
                float(top_right[0]),
                float(top_right[1]),
            ),
            bottom_right=(
                float(bottom_right[0]),
                float(bottom_right[1]),
            ),
            bottom_left=(
                float(bottom_left[0]),
                float(bottom_left[1]),
            ),
        )

    @staticmethod
    def _corners_to_box(
        corners: DocumentCorners,
    ) -> Optional[BoundingBox]:

        points = [
            corners.top_left,
            corners.top_right,
            corners.bottom_right,
            corners.bottom_left,
        ]

        xs = [
            point[0]
            for point in points
        ]

        ys = [
            point[1]
            for point in points
        ]

        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

        return BoundingBox(
            x=x1,
            y=y1,
            width=width,
            height=height,
        )

    # ========================================================================
    # ORIENTATION
    # ========================================================================

    @staticmethod
    def _orientation(
        box: BoundingBox,
    ) -> DocumentOrientation:

        ratio = (
            box.width
            / max(
                1.0,
                box.height,
            )
        )

        if 0.90 <= ratio <= 1.10:
            return DocumentOrientation.SQUARE

        if ratio > 1.0:
            return DocumentOrientation.LANDSCAPE

        return DocumentOrientation.PORTRAIT

    # ========================================================================
    # SKEW
    # ========================================================================

    @staticmethod
    def _estimate_skew(
        corners: DocumentCorners,
    ) -> float:

        dx = (
            corners.top_right[0]
            - corners.top_left[0]
        )

        dy = (
            corners.top_right[1]
            - corners.top_left[1]
        )

        if abs(dx) < 1e-6:
            return 90.0

        angle = math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )

        return angle

    # ========================================================================
    # TEXT LIKELIHOOD
    # ========================================================================

    def _estimate_text_likelihood(
        self,
        contour: Any,
        image_width: int,
        image_height: int,
    ) -> float:

        if cv2 is None:
            return 0.0

        try:

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0:
                return 0.0

            area = cv2.contourArea(
                contour
            )

            if area <= 0:
                return 0.0

            compactness = (
                4.0
                * math.pi
                * area
                / (perimeter**2)
            )

            # A perfectly compact rectangle has
            # relatively high compactness. Documents
            # with text usually contain additional
            # internal visual complexity, so this
            # factor remains intentionally weak.
            score = 1.0 - compactness

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
    # TYPE CLASSIFICATION
    # ========================================================================

    def _classify_document_type(
        self,
        *,
        box: BoundingBox,
        aspect_ratio: float,
        rectangularity: float,
        text_likelihood: float,
    ) -> DocumentType:

        if not self.config.classify_document_type:
            return DocumentType.UNKNOWN

        ratio = (
            max(
                box.width,
                box.height,
            )
            / max(
                1.0,
                min(
                    box.width,
                    box.height,
                ),
            )
        )

        if (
            ratio >= 1.6
            and text_likelihood > 0.35
        ):

            return DocumentType.RECEIPT

        if (
            1.2 <= ratio <= 1.8
            and rectangularity > 0.75
        ):

            return DocumentType.PAGE

        if (
            1.4 <= ratio <= 1.9
            and text_likelihood < 0.30
        ):

            return DocumentType.CARD

        if (
            0.85 <= aspect_ratio <= 1.20
        ):

            return DocumentType.NOTE

        return DocumentType.PAPER

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _calculate_confidence(
        *,
        area_ratio: float,
        rectangularity: float,
        text_likelihood: float,
        skew_angle: float,
    ) -> float:

        area_score = min(
            1.0,
            max(
                0.0,
                area_ratio / 0.35,
            ),
        )

        rectangular_score = (
            rectangularity
        )

        skew_score = max(
            0.0,
            1.0
            - min(
                abs(skew_angle),
                45.0,
            )
            / 45.0,
        )

        confidence = (
            area_score * 0.25
            + rectangular_score * 0.45
            + text_likelihood * 0.10
            + skew_score * 0.20
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # ========================================================================
    # DUPLICATES
    # ========================================================================

    def _remove_duplicates(
        self,
        documents: Sequence[
            DocumentDetection
        ],
    ) -> list[DocumentDetection]:

        result: list[
            DocumentDetection
        ] = []

        for document in documents:

            duplicate = False

            for existing in result:

                if (
                    document.box is None
                    or existing.box is None
                ):
                    continue

                if self._box_iou(
                    document.box,
                    existing.box,
                ) > 0.75:

                    duplicate = True

                    if (
                        document.confidence
                        > existing.confidence
                    ):

                        result.remove(
                            existing
                        )
                        result.append(
                            document
                        )

                    break

            if not duplicate:
                result.append(document)

        return result

    @staticmethod
    def _box_iou(
        first: BoundingBox,
        second: BoundingBox,
    ) -> float:

        first_x1 = first.x
        first_y1 = first.y
        first_x2 = (
            first.x + first.width
        )
        first_y2 = (
            first.y + first.height
        )

        second_x1 = second.x
        second_y1 = second.y
        second_x2 = (
            second.x + second.width
        )
        second_y2 = (
            second.y + second.height
        )

        intersection_width = max(
            0.0,
            min(
                first_x2,
                second_x2,
            )
            - max(
                first_x1,
                second_x1,
            ),
        )

        intersection_height = max(
            0.0,
            min(
                first_y2,
                second_y2,
            )
            - max(
                first_y1,
                second_y1,
            ),
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        if intersection_area <= 0:
            return 0.0

        first_area = (
            first.width
            * first.height
        )

        second_area = (
            second.width
            * second.height
        )

        union = (
            first_area
            + second_area
            - intersection_area
        )

        if union <= 0:
            return 0.0

        return (
            intersection_area
            / union
        )

    # ========================================================================
    # PERSPECTIVE CORRECTION
    # ========================================================================

    def rectify(
        self,
        image: Any,
        document: DocumentDetection,
        *,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
    ) -> Any:

        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required for document "
                "rectification."
            )

        if document.corners is None:
            raise ValueError(
                "Document does not contain corners."
            )

        corners = document.corners

        source = corners.as_array()

        width = (
            output_width
            or self._distance(
                corners.top_left,
                corners.top_right,
            )
        )

        height = (
            output_height
            or max(
                self._distance(
                    corners.top_left,
                    corners.bottom_left,
                ),
                self._distance(
                    corners.top_right,
                    corners.bottom_right,
                ),
            )
        )

        width = max(
            1,
            int(width),
        )

        height = max(
            1,
            int(height),
        )

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(
            source,
            destination,
        )

        return cv2.warpPerspective(
            image,
            matrix,
            (width, height),
        )

    @staticmethod
    def _distance(
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:

        return math.sqrt(
            (
                first[0]
                - second[0]
            )
            ** 2
            + (
                first[1]
                - second[1]
            )
            ** 2
        )

    # ========================================================================
    # LAST RESULT
    # ========================================================================

    def _store_result(
        self,
        result: DocumentDetectionResult,
    ) -> None:

        with self._lock:
            self._last_result = result

    def get_last_result(
        self,
    ) -> Optional[
        DocumentDetectionResult
    ]:

        with self._lock:
            return self._last_result

    # ========================================================================
    # DIMENSIONS
    # ========================================================================

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
    # FAILURE
    # ========================================================================

    def _failure(
        self,
        started: float,
        width: int,
        height: int,
        error: str,
    ) -> DocumentDetectionResult:

        result = (
            DocumentDetectionResult(
                documents=[],
                image_width=width,
                image_height=height,
                processing_time=(
                    time.perf_counter()
                    - started
                ),
                success=False,
                error=error,
            )
        )

        self._store_result(result)

        return result


# ============================================================================
# SHARED INSTANCE
# ============================================================================


_default_detector: Optional[
    DocumentDetector
] = None

_default_detector_lock = (
    threading.RLock()
)


def get_document_detector(
) -> DocumentDetector:

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                DocumentDetector()
            )

        return _default_detector


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def detect_documents(
    image: Any,
) -> DocumentDetectionResult:

    return get_document_detector().detect(
        image
    )


def detect_primary_document(
    image: Any,
) -> Optional[DocumentDetection]:

    return (
        get_document_detector()
        .detect_primary(image)
    )


def is_document_present(
    image: Any,
) -> bool:

    return (
        get_document_detector()
        .is_document_present(image)
    )


def rectify_document(
    image: Any,
    document: DocumentDetection,
    *,
    output_width: Optional[int] = None,
    output_height: Optional[int] = None,
) -> Any:

    return (
        get_document_detector()
        .rectify(
            image,
            document,
            output_width=output_width,
            output_height=output_height,
        )
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "DocumentType",
    "DocumentOrientation",
    "DocumentCorners",
    "DocumentDetection",
    "DocumentDetectionResult",
    "DocumentDetectionConfig",
    "DocumentDetector",
    "get_document_detector",
    "detect_documents",
    "detect_primary_document",
    "is_document_present",
    "rectify_document",
]


