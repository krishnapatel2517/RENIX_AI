"""
RENIX Vision — QR / Barcode Detection

Detects QR codes and supported barcodes from:
- Camera frames
- Screenshots
- Images
- Video frames

Responsibilities:
- QR code detection
- Barcode detection when supported by OpenCV
- Decode payloads
- Return bounding boxes
- Return polygon points
- Track multiple codes
- Normalize decoded data
- Gracefully handle unavailable OpenCV functionality

This module does NOT execute decoded URLs, commands, or payloads.
It only detects and returns them. Any action based on a QR/barcode
must be handled by RENIX security/command layers.
"""

from __future__ import annotations

import logging
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
    "RENIX.vision.qr_detection"
)


# ============================================================================
# ENUMS
# ============================================================================


class CodeType(str, Enum):
    UNKNOWN = "unknown"
    QR = "qr"
    BARCODE = "barcode"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class CodeCorners:
    """
    Four corner points of a detected code.

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

    def as_list(
        self,
    ) -> list[tuple[float, float]]:

        return [
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        ]

    def as_array(self):

        points = self.as_list()

        if np is None:
            return points

        return np.asarray(
            points,
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
class CodeDetection:
    """
    One QR/barcode detection.
    """

    code_id: str

    code_type: CodeType = (
        CodeType.UNKNOWN
    )

    data: str = ""

    raw_data: Optional[bytes] = None

    confidence: float = 0.0

    box: Optional[BoundingBox] = None

    corners: Optional[CodeCorners] = None

    polygon: list[
        tuple[float, float]
    ] = field(default_factory=list)

    valid: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "code_id": self.code_id,
            "code_type": (
                self.code_type.value
            ),
            "data": self.data,
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
            "polygon": self.polygon,
            "valid": self.valid,
            "metadata": dict(self.metadata),
        }


@dataclass
class CodeDetectionResult:
    """
    Result of QR/barcode detection.
    """

    codes: list[
        CodeDetection
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
        return bool(self.codes)

    @property
    def count(self) -> int:
        return len(self.codes)

    @property
    def qr_codes(
        self,
    ) -> list[CodeDetection]:

        return [
            code
            for code in self.codes
            if code.code_type
            == CodeType.QR
        ]

    @property
    def barcodes(
        self,
    ) -> list[CodeDetection]:

        return [
            code
            for code in self.codes
            if code.code_type
            == CodeType.BARCODE
        ]

    @property
    def primary(
        self,
    ) -> Optional[CodeDetection]:

        if not self.codes:
            return None

        return max(
            self.codes,
            key=lambda code: (
                code.confidence
            ),
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "codes": [
                code.to_dict()
                for code in self.codes
            ],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "processing_time": (
                self.processing_time
            ),
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass
class QRDetectionConfig:
    """
    QR/barcode detector configuration.
    """

    detect_qr: bool = True

    detect_barcodes: bool = True

    min_confidence: float = 0.25

    max_codes: int = 50

    preprocess: bool = True

    grayscale: bool = True

    enhance_contrast: bool = True

    try_rotations: bool = True

    rotation_angles: tuple[
        int, ...
    ] = (
        90,
        180,
        270,
    )

    return_invalid_detections: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# CODE DETECTOR
# ============================================================================


class QRCodeDetector:
    """
    RENIX QR/barcode detection engine.

    Example:

        detector = QRCodeDetector()

        result = detector.detect(frame)

        for code in result.codes:
            print(code.data)
    """

    def __init__(
        self,
        config: Optional[
            QRDetectionConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or QRDetectionConfig()
        )

        self._lock = threading.RLock()

        self._last_result: Optional[
            CodeDetectionResult
        ] = None

        self._frame_counter = 0

        self._qr_detector = None

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:

        if cv2 is None:

            logger.warning(
                "OpenCV is not installed. "
                "QR detection is unavailable."
            )

            return

        try:

            if hasattr(
                cv2,
                "QRCodeDetector",
            ):

                self._qr_detector = (
                    cv2.QRCodeDetector()
                )

            else:

                logger.warning(
                    "OpenCV QRCodeDetector "
                    "is unavailable."
                )

        except Exception as exc:

            logger.warning(
                "Failed to initialize QR detector: %s",
                exc,
            )

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def detect(
        self,
        image: Any,
    ) -> CodeDetectionResult:

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

            processed = (
                self._preprocess(image)
            )

            detections: list[
                CodeDetection
            ] = []

            # ---------------------------------------------------------------
            # QR
            # ---------------------------------------------------------------

            if self.config.detect_qr:

                detections.extend(
                    self._detect_qr(
                        processed
                    )
                )

            # ---------------------------------------------------------------
            # Barcode
            # ---------------------------------------------------------------

            if self.config.detect_barcodes:

                detections.extend(
                    self._detect_barcodes(
                        processed
                    )
                )

            # ---------------------------------------------------------------
            # Rotated QR fallback
            # ---------------------------------------------------------------

            if (
                self.config.try_rotations
                and self.config.detect_qr
                and len(detections) == 0
            ):

                detections.extend(
                    self._detect_rotated_qr(
                        image
                    )
                )

            # ---------------------------------------------------------------
            # Cleanup
            # ---------------------------------------------------------------

            detections = (
                self._normalize_detections(
                    detections
                )
            )

            detections = (
                self._remove_duplicates(
                    detections
                )
            )

            detections.sort(
                key=lambda code: (
                    code.confidence
                ),
                reverse=True,
            )

            if (
                self.config.max_codes
                > 0
            ):

                detections = detections[
                    : self.config.max_codes
                ]

            result = (
                CodeDetectionResult(
                    codes=detections,
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
                    },
                )
            )

            self._store_result(result)

            return result

        except Exception as exc:

            logger.exception(
                "QR/barcode detection failed."
            )

            return self._failure(
                started,
                width,
                height,
                str(exc),
            )

    # ========================================================================
    # QR DETECTION
    # ========================================================================

    def _detect_qr(
        self,
        image: Any,
    ) -> list[CodeDetection]:

        if (
            self._qr_detector is None
        ):
            return []

        detections: list[
            CodeDetection
        ] = []

        # ---------------------------------------------------------------
        # Multi-code detection
        # ---------------------------------------------------------------

        try:

            result = (
                self._qr_detector.detectAndDecodeMulti(
                    image
                )
            )

            if isinstance(
                result,
                tuple,
            ):

                if len(result) == 4:

                    found, decoded_info, points, _ = (
                        result
                    )

                elif len(result) == 3:

                    found, decoded_info, points = (
                        result
                    )

                else:

                    found = False
                    decoded_info = []
                    points = None

                if found:

                    for index, data in enumerate(
                        decoded_info or []
                    ):

                        polygon = (
                            self._points_to_polygon(
                                points,
                                index,
                            )
                        )

                        detection = (
                            self._create_qr_detection(
                                data=data,
                                polygon=polygon,
                                index=index,
                            )
                        )

                        if detection is not None:

                            detections.append(
                                detection
                            )

        except Exception as exc:

            logger.debug(
                "QR multi-code detection failed: %s",
                exc,
            )

        # ---------------------------------------------------------------
        # Single QR fallback
        # ---------------------------------------------------------------

        if not detections:

            try:

                data, points, _ = (
                    self._qr_detector.detectAndDecode(
                        image
                    )
                )

                polygon = (
                    self._points_to_polygon(
                        points,
                        0,
                    )
                )

                detection = (
                    self._create_qr_detection(
                        data=data,
                        polygon=polygon,
                        index=0,
                    )
                )

                if detection is not None:

                    detections.append(
                        detection
                    )

            except Exception as exc:

                logger.debug(
                    "QR single-code detection failed: %s",
                    exc,
                )

        return detections

    # ========================================================================
    # ROTATED QR DETECTION
    # ========================================================================

    def _detect_rotated_qr(
        self,
        image: Any,
    ) -> list[CodeDetection]:

        if cv2 is None:
            return []

        detections: list[
            CodeDetection
        ] = []

        for angle in (
            self.config.rotation_angles
        ):

            try:

                rotated = (
                    self._rotate_image(
                        image,
                        angle,
                    )
                )

                found = self._detect_qr(
                    rotated
                )

                for detection in found:

                    detection.metadata[
                        "rotation"
                    ] = angle

                    detections.append(
                        detection
                    )

            except Exception as exc:

                logger.debug(
                    "Rotated QR detection "
                    "failed at %s°: %s",
                    angle,
                    exc,
                )

        return detections

    # ========================================================================
    # BARCODE DETECTION
    # ========================================================================

    def _detect_barcodes(
        self,
        image: Any,
    ) -> list[CodeDetection]:

        if cv2 is None:
            return []

        barcode_class = getattr(
            cv2,
            "barcode_BarcodeDetector",
            None,
        )

        if barcode_class is None:

            # Older OpenCV installations may
            # expose BarcodeDetector differently.
            barcode_class = getattr(
                cv2,
                "BarcodeDetector",
                None,
            )

        if barcode_class is None:
            return []

        try:

            detector = (
                barcode_class()
            )

        except Exception as exc:

            logger.debug(
                "Barcode detector initialization "
                "failed: %s",
                exc,
            )

            return []

        detections: list[
            CodeDetection
        ] = []

        # Newer OpenCV variants commonly expose
        # detectAndDecodeWithType.
        try:

            method = getattr(
                detector,
                "detectAndDecodeWithType",
                None,
            )

            if callable(method):

                result = method(image)

                decoded = None
                decoded_type = None
                points = None

                if isinstance(
                    result,
                    tuple,
                ):

                    if len(result) >= 3:

                        decoded = result[
                            0
                        ]

                        decoded_type = result[
                            1
                        ]

                        points = result[
                            2
                        ]

                if decoded is not None:

                    values = (
                        decoded
                        if isinstance(
                            decoded,
                            (list, tuple),
                        )
                        else [decoded]
                    )

                    types = (
                        decoded_type
                        if isinstance(
                            decoded_type,
                            (list, tuple),
                        )
                        else [decoded_type]
                    )

                    for index, data in enumerate(
                        values
                    ):

                        if data is None:
                            continue

                        polygon = (
                            self._points_to_polygon(
                                points,
                                index,
                            )
                        )

                        if not str(data).strip():
                            continue

                        code = (
                            self._create_barcode_detection(
                                data=str(data),
                                polygon=polygon,
                                index=index,
                                barcode_type=(
                                    types[index]
                                    if index
                                    < len(types)
                                    else None
                                ),
                            )
                        )

                        if code is not None:
                            detections.append(
                                code
                            )

                    if detections:
                        return detections

        except Exception as exc:

            logger.debug(
                "Barcode decode-with-type "
                "failed: %s",
                exc,
            )

        # ---------------------------------------------------------------
        # Generic barcode detection fallback
        # ---------------------------------------------------------------

        try:

            detect_method = getattr(
                detector,
                "detect",
                None,
            )

            if callable(detect_method):

                result = detect_method(
                    image
                )

                found = False
                points = None

                if isinstance(
                    result,
                    tuple,
                ):

                    if len(result) >= 2:

                        found = result[
                            0
                        ]

                        points = result[
                            1
                        ]

                else:

                    found = bool(result)

                if found and points is not None:

                    polygon = (
                        self._points_to_polygon(
                            points,
                            0,
                        )
                    )

                    detections.append(
                        CodeDetection(
                            code_id="barcode_0",
                            code_type=(
                                CodeType.BARCODE
                            ),
                            data="",
                            confidence=0.30,
                            box=(
                                self._polygon_to_box(
                                    polygon
                                )
                            ),
                            corners=(
                                self._polygon_to_corners(
                                    polygon
                                )
                            ),
                            polygon=polygon,
                            valid=False,
                            metadata={
                                "decoded": False
                            },
                        )
                    )

        except Exception as exc:

            logger.debug(
                "Generic barcode detection "
                "failed: %s",
                exc,
            )

        return detections

    # ========================================================================
    # CREATE QR RESULT
    # ========================================================================

    def _create_qr_detection(
        self,
        *,
        data: Any,
        polygon: list[
            tuple[float, float]
        ],
        index: int,
    ) -> Optional[
        CodeDetection
    ]:

        normalized = (
            self._normalize_data(data)
        )

        valid = bool(normalized)

        if (
            not valid
            and not self.config.return_invalid_detections
        ):
            return None

        box = (
            self._polygon_to_box(
                polygon
            )
        )

        confidence = (
            self._estimate_confidence(
                polygon=polygon,
                valid=valid,
            )
        )

        if (
            confidence
            < self.config.min_confidence
            and not valid
        ):
            return None

        return CodeDetection(
            code_id=(
                f"qr_{index}"
            ),
            code_type=CodeType.QR,
            data=normalized,
            raw_data=(
                normalized.encode(
                    "utf-8",
                    errors="replace",
                )
                if normalized
                else None
            ),
            confidence=confidence,
            box=box,
            corners=(
                self._polygon_to_corners(
                    polygon
                )
            ),
            polygon=polygon,
            valid=valid,
            metadata={
                "payload_length": len(
                    normalized
                )
            },
        )

    # ========================================================================
    # CREATE BARCODE RESULT
    # ========================================================================

    def _create_barcode_detection(
        self,
        *,
        data: str,
        polygon: list[
            tuple[float, float]
        ],
        index: int,
        barcode_type: Any = None,
    ) -> Optional[
        CodeDetection
    ]:

        normalized = (
            self._normalize_data(data)
        )

        if not normalized:
            return None

        return CodeDetection(
            code_id=(
                f"barcode_{index}"
            ),
            code_type=CodeType.BARCODE,
            data=normalized,
            raw_data=(
                normalized.encode(
                    "utf-8",
                    errors="replace",
                )
            ),
            confidence=(
                self._estimate_confidence(
                    polygon=polygon,
                    valid=True,
                )
            ),
            box=(
                self._polygon_to_box(
                    polygon
                )
            ),
            corners=(
                self._polygon_to_corners(
                    polygon
                )
            ),
            polygon=polygon,
            valid=True,
            metadata={
                "barcode_type": (
                    str(barcode_type)
                    if barcode_type is not None
                    else None
                )
            },
        )

    # ========================================================================
    # PREPROCESSING
    # ========================================================================

    def _preprocess(
        self,
        image: Any,
    ) -> Any:

        if not self.config.preprocess:
            return image

        if cv2 is None:
            return image

        try:

            processed = image

            if (
                self.config.grayscale
                and len(
                    processed.shape
                ) >= 3
            ):

                processed = (
                    cv2.cvtColor(
                        processed,
                        cv2.COLOR_BGR2GRAY,
                    )
                )

            if (
                self.config.enhance_contrast
            ):

                processed = (
                    cv2.equalizeHist(
                        processed
                    )
                    if len(
                        processed.shape
                    ) == 2
                    else processed
                )

            return processed

        except Exception as exc:

            logger.debug(
                "QR preprocessing failed: %s",
                exc,
            )

            return image

    # ========================================================================
    # POINT HELPERS
    # ========================================================================

    @staticmethod
    def _points_to_polygon(
        points: Any,
        index: int = 0,
    ) -> list[
        tuple[float, float]
    ]:

        if points is None:
            return []

        try:

            array = np.asarray(
                points
            )

            # Typical OpenCV output:
            # shape = (N, 4, 2)
            if array.ndim == 3:

                if (
                    index >= array.shape[0]
                ):
                    return []

                array = array[index]

            # Shape = (4, 2)
            if array.ndim == 2:

                return [
                    (
                        float(point[0]),
                        float(point[1]),
                    )
                    for point in array
                    if len(point) >= 2
                ]

        except Exception:
            pass

        return []

    @staticmethod
    def _polygon_to_box(
        polygon: Sequence[
            tuple[float, float]
        ],
    ) -> Optional[BoundingBox]:

        if not polygon:
            return None

        xs = [
            point[0]
            for point in polygon
        ]

        ys = [
            point[1]
            for point in polygon
        ]

        if not xs or not ys:
            return None

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
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

    @staticmethod
    def _polygon_to_corners(
        polygon: Sequence[
            tuple[float, float]
        ],
    ) -> Optional[CodeCorners]:

        if len(polygon) < 4:
            return None

        points = list(
            polygon[:4]
        )

        if len(points) != 4:
            return None

        # Order points using sums/differences.
        try:

            if np is None:
                return CodeCorners(
                    top_left=points[0],
                    top_right=points[1],
                    bottom_right=points[2],
                    bottom_left=points[3],
                )

            array = np.asarray(
                points,
                dtype="float32",
            )

            sums = (
                array[:, 0]
                + array[:, 1]
            )

            differences = (
                array[:, 0]
                - array[:, 1]
            )

            tl = array[
                differences.argmin()
            ]

            br = array[
                differences.argmax()
            ]

            tr = array[
                sums.argmin()
            ]

            bl = array[
                sums.argmax()
            ]

            return CodeCorners(
                top_left=(
                    float(tl[0]),
                    float(tl[1]),
                ),
                top_right=(
                    float(tr[0]),
                    float(tr[1]),
                ),
                bottom_right=(
                    float(br[0]),
                    float(br[1]),
                ),
                bottom_left=(
                    float(bl[0]),
                    float(bl[1]),
                ),
            )

        except Exception:
            return None

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _estimate_confidence(
        *,
        polygon: Sequence[
            tuple[float, float]
        ],
        valid: bool,
    ) -> float:

        if not polygon:
            return 0.30 if valid else 0.0

        if len(polygon) < 4:
            return 0.35 if valid else 0.0

        # OpenCV has already performed the actual
        # detection. The confidence here is a
        # conservative structural estimate.
        confidence = 0.60

        if valid:
            confidence += 0.30

        if len(polygon) == 4:
            confidence += 0.10

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_data(
        data: Any,
    ) -> str:

        if data is None:
            return ""

        if isinstance(
            data,
            bytes,
        ):

            try:

                data = data.decode(
                    "utf-8",
                    errors="replace",
                )

            except Exception:
                data = str(data)

        value = str(data)

        return value.strip()

    def _normalize_detections(
        self,
        detections: Sequence[
            CodeDetection
        ],
    ) -> list[CodeDetection]:

        result: list[
            CodeDetection
        ] = []

        for index, detection in enumerate(
            detections
        ):

            detection.code_id = (
                f"{detection.code_type.value}_"
                f"{index}"
            )

            detection.data = (
                self._normalize_data(
                    detection.data
                )
            )

            detection.valid = bool(
                detection.data
            )

            result.append(
                detection
            )

        return result

    # ========================================================================
    # DUPLICATE REMOVAL
    # ========================================================================

    def _remove_duplicates(
        self,
        detections: Sequence[
            CodeDetection
        ],
    ) -> list[CodeDetection]:

        unique: list[
            CodeDetection
        ] = []

        for detection in detections:

            duplicate = False

            for existing in unique:

                # Same payload and same type.
                if (
                    detection.code_type
                    == existing.code_type
                    and detection.data
                    and detection.data
                    == existing.data
                ):

                    duplicate = True

                    if (
                        detection.confidence
                        > existing.confidence
                    ):

                        unique.remove(
                            existing
                        )

                        unique.append(
                            detection
                        )

                    break

                # Same physical region.
                if (
                    detection.box is not None
                    and existing.box is not None
                    and self._box_iou(
                        detection.box,
                        existing.box,
                    )
                    > 0.80
                    and (
                        detection.code_type
                        == existing.code_type
                    )
                ):

                    duplicate = True

                    if (
                        detection.confidence
                        > existing.confidence
                    ):

                        unique.remove(
                            existing
                        )

                        unique.append(
                            detection
                        )

                    break

            if not duplicate:
                unique.append(
                    detection
                )

        return unique

    @staticmethod
    def _box_iou(
        first: BoundingBox,
        second: BoundingBox,
    ) -> float:

        x1 = max(
            first.x,
            second.x,
        )

        y1 = max(
            first.y,
            second.y,
        )

        x2 = min(
            first.x + first.width,
            second.x + second.width,
        )

        y2 = min(
            first.y + first.height,
            second.y + second.height,
        )

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        intersection = (
            width * height
        )

        if intersection <= 0:
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
            - intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection
            / union
        )

    # ========================================================================
    # ROTATION
    # ========================================================================

    @staticmethod
    def _rotate_image(
        image: Any,
        angle: int,
    ) -> Any:

        if cv2 is None:
            return image

        angle = angle % 360

        if angle == 90:

            return cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )

        if angle == 180:

            return cv2.rotate(
                image,
                cv2.ROTATE_180,
            )

        if angle == 270:

            return cv2.rotate(
                image,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

        return image

    # ========================================================================
    # AVAILABILITY
    # ========================================================================

    def is_available(self) -> bool:

        return (
            cv2 is not None
            and self._qr_detector is not None
        )

    def supports_barcodes(
        self,
    ) -> bool:

        if cv2 is None:
            return False

        return bool(
            hasattr(
                cv2,
                "BarcodeDetector",
            )
            or hasattr(
                cv2,
                "barcode_BarcodeDetector",
            )
        )

    # ========================================================================
    # LAST RESULT
    # ========================================================================

    def _store_result(
        self,
        result: CodeDetectionResult,
    ) -> None:

        with self._lock:
            self._last_result = result

    def get_last_result(
        self,
    ) -> Optional[
        CodeDetectionResult
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
    ) -> CodeDetectionResult:

        result = (
            CodeDetectionResult(
                codes=[],
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
    QRCodeDetector
] = None

_default_detector_lock = (
    threading.RLock()
)


def get_qr_detector(
) -> QRCodeDetector:

    global _default_detector

    with _default_detector_lock:

        if _default_detector is None:

            _default_detector = (
                QRCodeDetector()
            )

        return _default_detector


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def detect_codes(
    image: Any,
) -> CodeDetectionResult:

    return get_qr_detector().detect(
        image
    )


def detect_qr_codes(
    image: Any,
) -> list[CodeDetection]:

    result = get_qr_detector().detect(
        image
    )

    return result.qr_codes


def detect_barcodes(
    image: Any,
) -> list[CodeDetection]:

    result = get_qr_detector().detect(
        image
    )

    return result.barcodes


def decode_qr(
    image: Any,
) -> Optional[str]:

    codes = detect_qr_codes(
        image
    )

    if not codes:
        return None

    primary = max(
        codes,
        key=lambda code: (
            code.confidence
        ),
    )

    if not primary.data:
        return None

    return primary.data


def is_qr_available() -> bool:

    return get_qr_detector().is_available()


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "CodeType",
    "CodeCorners",
    "CodeDetection",
    "CodeDetectionResult",
    "QRDetectionConfig",
    "QRCodeDetector",
    "get_qr_detector",
    "detect_codes",
    "detect_qr_codes",
    "detect_barcodes",
    "decode_qr",
    "is_qr_available",
]


