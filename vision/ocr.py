"""
RENIX Vision — OCR Engine

Handles optical character recognition for RENIX.

Responsibilities:
- Extract text from images/frames
- Detect text regions
- Normalize OCR output
- Support screenshots, documents and camera frames
- Provide confidence scores and bounding boxes
- Support multiple OCR backends
- Gracefully operate when optional OCR packages are unavailable
- Expose a clean interface to the vision pipeline

The engine is intentionally backend-agnostic.
"""

from __future__ import annotations

import logging
import re
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

try:
    import pytesseract
except ImportError:
    pytesseract = None


from .object_detection import BoundingBox


logger = logging.getLogger("RENIX.vision.ocr")


# ============================================================================
# ENUMS
# ============================================================================


class OCRBackend(str, Enum):
    AUTO = "auto"
    TESSERACT = "tesseract"
    NONE = "none"


class TextOrientation(str, Enum):
    UNKNOWN = "unknown"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class OCRTextRegion:
    """
    One detected text region.
    """

    text: str

    confidence: float = 0.0

    box: Optional[BoundingBox] = None

    line_number: Optional[int] = None

    word_number: Optional[int] = None

    block_number: Optional[int] = None

    paragraph_number: Optional[int] = None

    orientation: TextOrientation = (
        TextOrientation.UNKNOWN
    )

    language: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": (
                self.box.to_dict()
                if self.box is not None
                else None
            ),
            "line_number": self.line_number,
            "word_number": self.word_number,
            "block_number": self.block_number,
            "paragraph_number": self.paragraph_number,
            "orientation": (
                self.orientation.value
            ),
            "language": self.language,
            "metadata": dict(self.metadata),
        }


@dataclass
class OCRResult:
    """
    Complete OCR result for one image.
    """

    text: str = ""

    regions: list[OCRTextRegion] = field(
        default_factory=list
    )

    confidence: float = 0.0

    language: Optional[str] = None

    backend: OCRBackend = OCRBackend.NONE

    processing_time: float = 0.0

    image_width: int = 0

    image_height: int = 0

    success: bool = True

    error: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def region_count(self) -> int:
        return len(self.regions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "regions": [
                region.to_dict()
                for region in self.regions
            ],
            "confidence": self.confidence,
            "language": self.language,
            "backend": self.backend.value,
            "processing_time": self.processing_time,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass
class OCRConfig:
    """
    OCR configuration.
    """

    backend: OCRBackend = OCRBackend.AUTO

    language: str = "eng"

    confidence_threshold: float = 0.30

    detect_regions: bool = True

    preprocess: bool = True

    grayscale: bool = True

    denoise: bool = True

    threshold: bool = True

    deskew: bool = True

    upscale: float = 1.5

    psm: int = 6

    oem: int = 3

    max_regions: int = 500

    preserve_interword_spaces: bool = True

    remove_empty_regions: bool = True

    normalize_whitespace: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# OCR ENGINE
# ============================================================================


class OCREngine:
    """
    RENIX OCR service.

    Example:

        engine = OCREngine()

        result = engine.read(image)

        print(result.text)

    Optional Tesseract support is automatically detected.
    """

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
    ) -> None:

        self.config = (
            config
            or OCRConfig()
        )

        self._lock = threading.RLock()

        self._last_result: Optional[
            OCRResult
        ] = None

        self._initialized = False

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:

        with self._lock:

            if self._initialized:
                return

            if (
                self.config.backend
                in (
                    OCRBackend.AUTO,
                    OCRBackend.TESSERACT,
                )
            ):

                if pytesseract is None:

                    logger.warning(
                        "pytesseract is not installed. "
                        "OCR will operate in unavailable mode."
                    )

                else:

                    logger.info(
                        "Tesseract OCR backend detected."
                    )

            self._initialized = True

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def read(
        self,
        image: Any,
        *,
        language: Optional[str] = None,
    ) -> OCRResult:

        started = time.perf_counter()

        width, height = (
            self._get_dimensions(image)
        )

        selected_language = (
            language
            or self.config.language
        )

        try:

            if image is None:

                return self._failure(
                    started,
                    width,
                    height,
                    "Image is None.",
                )

            backend = self._select_backend()

            if backend == OCRBackend.NONE:

                return self._failure(
                    started,
                    width,
                    height,
                    "No OCR backend is available.",
                )

            processed = self._preprocess(image)

            if backend == OCRBackend.TESSERACT:

                result = self._read_tesseract(
                    processed,
                    selected_language,
                )

            else:

                return self._failure(
                    started,
                    width,
                    height,
                    "Unsupported OCR backend.",
                )

            result.processing_time = (
                time.perf_counter()
                - started
            )

            result.image_width = width
            result.image_height = height
            result.language = (
                selected_language
            )

            self._last_result = result

            return result

        except Exception as exc:

            logger.exception(
                "OCR processing failed."
            )

            return self._failure(
                started,
                width,
                height,
                str(exc),
            )

    # ========================================================================
    # SIMPLE TEXT API
    # ========================================================================

    def extract_text(
        self,
        image: Any,
        *,
        language: Optional[str] = None,
    ) -> str:

        result = self.read(
            image,
            language=language,
        )

        return result.text

    # ========================================================================
    # REGION API
    # ========================================================================

    def detect_text_regions(
        self,
        image: Any,
        *,
        language: Optional[str] = None,
    ) -> list[OCRTextRegion]:

        result = self.read(
            image,
            language=language,
        )

        return result.regions

    # ========================================================================
    # TESSERACT
    # ========================================================================

    def _read_tesseract(
        self,
        image: Any,
        language: str,
    ) -> OCRResult:

        if pytesseract is None:

            return OCRResult(
                success=False,
                backend=OCRBackend.NONE,
                error=(
                    "pytesseract is not installed."
                ),
            )

        config = self._tesseract_config()

        text = pytesseract.image_to_string(
            image,
            lang=language,
            config=config,
        )

        text = self._normalize_text(text)

        regions: list[
            OCRTextRegion
        ] = []

        if self.config.detect_regions:

            regions = (
                self._extract_tesseract_regions(
                    image,
                    language,
                    config,
                )
            )

        confidence = (
            self._calculate_confidence(
                regions
            )
        )

        if not regions and text:

            regions = [
                OCRTextRegion(
                    text=text,
                    confidence=confidence,
                    box=None,
                    language=language,
                )
            ]

        return OCRResult(
            text=text,
            regions=regions,
            confidence=confidence,
            language=language,
            backend=OCRBackend.TESSERACT,
            success=True,
        )

    def _extract_tesseract_regions(
        self,
        image: Any,
        language: str,
        config: str,
    ) -> list[OCRTextRegion]:

        if pytesseract is None:
            return []

        try:

            data = (
                pytesseract.image_to_data(
                    image,
                    lang=language,
                    config=config,
                    output_type=(
                        pytesseract.Output.DICT
                    ),
                )
            )

        except Exception as exc:

            logger.warning(
                "Tesseract region extraction failed: %s",
                exc,
            )

            return []

        regions: list[
            OCRTextRegion
        ] = []

        total = len(
            data.get(
                "text",
                [],
            )
        )

        for index in range(total):

            raw_text = str(
                data["text"][index]
            ).strip()

            if (
                self.config.remove_empty_regions
                and not raw_text
            ):
                continue

            try:

                confidence = float(
                    data["conf"][index]
                )

            except (
                ValueError,
                TypeError,
            ):

                confidence = 0.0

            confidence = (
                confidence / 100.0
            )

            if (
                confidence
                < self.config.confidence_threshold
            ):
                continue

            x = int(
                data["left"][index]
            )

            y = int(
                data["top"][index]
            )

            width = int(
                data["width"][index]
            )

            height = int(
                data["height"][index]
            )

            box = BoundingBox(
                x=x,
                y=y,
                width=width,
                height=height,
            )

            regions.append(
                OCRTextRegion(
                    text=raw_text,
                    confidence=confidence,
                    box=box,
                    line_number=(
                        self._safe_int(
                            data.get(
                                "line_num"
                            ),
                            index,
                        )
                    ),
                    word_number=(
                        self._safe_int(
                            data.get(
                                "word_num"
                            ),
                            index,
                        )
                    ),
                    block_number=(
                        self._safe_int(
                            data.get(
                                "block_num"
                            ),
                            index,
                        )
                    ),
                    paragraph_number=(
                        self._safe_int(
                            data.get(
                                "par_num"
                            ),
                            index,
                        )
                    ),
                    orientation=(
                        TextOrientation.HORIZONTAL
                    ),
                    language=language,
                )
            )

            if (
                self.config.max_regions
                > 0
                and len(regions)
                >= self.config.max_regions
            ):
                break

        return self._merge_regions(
            regions
        )

    # ========================================================================
    # REGION MERGING
    # ========================================================================

    def _merge_regions(
        self,
        regions: Sequence[
            OCRTextRegion
        ],
    ) -> list[OCRTextRegion]:

        if not regions:
            return []

        grouped: dict[
            tuple[int, int, int],
            list[OCRTextRegion],
        ] = {}

        for region in regions:

            key = (
                region.block_number or 0,
                region.paragraph_number or 0,
                region.line_number or 0,
            )

            grouped.setdefault(
                key,
                [],
            ).append(region)

        merged: list[
            OCRTextRegion
        ] = []

        for _, group in sorted(
            grouped.items()
        ):

            if len(group) == 1:

                merged.append(group[0])
                continue

            ordered = sorted(
                group,
                key=lambda item: (
                    item.box.x
                    if item.box
                    else 0
                ),
            )

            text = " ".join(
                item.text
                for item in ordered
                if item.text
            ).strip()

            boxes = [
                item.box
                for item in ordered
                if item.box is not None
            ]

            box = self._union_boxes(
                boxes
            )

            confidence = sum(
                item.confidence
                for item in ordered
            ) / len(ordered)

            first = ordered[0]

            merged.append(
                OCRTextRegion(
                    text=text,
                    confidence=confidence,
                    box=box,
                    line_number=(
                        first.line_number
                    ),
                    word_number=(
                        first.word_number
                    ),
                    block_number=(
                        first.block_number
                    ),
                    paragraph_number=(
                        first.paragraph_number
                    ),
                    orientation=(
                        first.orientation
                    ),
                    language=first.language,
                )
            )

        return merged

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

        processed = image

        try:

            if self.config.grayscale:

                if (
                    hasattr(processed, "shape")
                    and len(
                        processed.shape
                    ) >= 3
                    and processed.shape[2] >= 3
                ):

                    processed = (
                        cv2.cvtColor(
                            processed,
                            cv2.COLOR_BGR2GRAY,
                        )
                    )

            if self.config.denoise:

                processed = (
                    cv2.GaussianBlur(
                        processed,
                        (3, 3),
                        0,
                    )
                )

            if self.config.threshold:

                if len(
                    processed.shape
                ) == 2:

                    processed = (
                        cv2.adaptiveThreshold(
                            processed,
                            255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY,
                            31,
                            11,
                        )
                    )

            if (
                self.config.upscale
                and self.config.upscale
                != 1.0
            ):

                scale = float(
                    self.config.upscale
                )

                processed = (
                    cv2.resize(
                        processed,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=(
                            cv2.INTER_CUBIC
                        ),
                    )
                )

            return processed

        except Exception as exc:

            logger.warning(
                "OCR preprocessing failed: %s",
                exc,
            )

            return image

    # ========================================================================
    # DESKEW
    # ========================================================================

    def _deskew(
        self,
        image: Any,
    ) -> Any:

        if cv2 is None:
            return image

        try:

            if len(
                image.shape
            ) != 2:
                return image

            coords = np.column_stack(
                np.where(image < 255)
            )

            if len(coords) < 20:
                return image

            angle = cv2.minAreaRect(
                coords
            )[-1]

            if angle < -45:
                angle = -(90 + angle)

            else:
                angle = -angle

            height, width = (
                image.shape[:2]
            )

            center = (
                width // 2,
                height // 2,
            )

            matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0,
            )

            return cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

        except Exception:
            return image

    # ========================================================================
    # TESSERACT CONFIG
    # ========================================================================

    def _tesseract_config(
        self,
    ) -> str:

        parts = [
            f"--psm {self.config.psm}",
            f"--oem {self.config.oem}",
        ]

        if (
            self.config.preserve_interword_spaces
        ):

            parts.append(
                "-c preserve_interword_spaces=1"
            )

        return " ".join(parts)

    # ========================================================================
    # TEXT NORMALIZATION
    # ========================================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        lines = [
            line.strip()
            for line in text.split("\n")
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        if self.config.normalize_whitespace:

            normalized: list[str] = []

            for line in lines:

                line = re.sub(
                    r"[ \t]+",
                    " ",
                    line,
                )

                normalized.append(
                    line.strip()
                )

            lines = normalized

        return "\n".join(lines).strip()

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _calculate_confidence(
        regions: Sequence[
            OCRTextRegion
        ],
    ) -> float:

        if not regions:
            return 0.0

        total_weight = 0.0

        weighted_sum = 0.0

        for region in regions:

            weight = max(
                1.0,
                float(
                    len(
                        region.text
                    )
                ),
            )

            weighted_sum += (
                region.confidence
                * weight
            )

            total_weight += weight

        if total_weight <= 0:
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                weighted_sum
                / total_weight,
            ),
        )

    # ========================================================================
    # BACKEND SELECTION
    # ========================================================================

    def _select_backend(
        self,
    ) -> OCRBackend:

        requested = self.config.backend

        if requested == OCRBackend.NONE:
            return OCRBackend.NONE

        if (
            requested
            == OCRBackend.TESSERACT
        ):

            if pytesseract is not None:
                return OCRBackend.TESSERACT

            return OCRBackend.NONE

        if requested == OCRBackend.AUTO:

            if pytesseract is not None:
                return OCRBackend.TESSERACT

            return OCRBackend.NONE

        return OCRBackend.NONE

    # ========================================================================
    # LAST RESULT
    # ========================================================================

    def get_last_result(
        self,
    ) -> Optional[OCRResult]:

        with self._lock:
            return self._last_result

    # ========================================================================
    # AVAILABILITY
    # ========================================================================

    def is_available(self) -> bool:

        backend = self._select_backend()

        return backend != OCRBackend.NONE

    # ========================================================================
    # LANGUAGE SUPPORT
    # ========================================================================

    def get_available_languages(
        self,
    ) -> list[str]:

        if pytesseract is None:
            return []

        try:

            return sorted(
                pytesseract.get_languages(
                    config=""
                )
            )

        except Exception as exc:

            logger.warning(
                "Unable to determine OCR languages: %s",
                exc,
            )

            return []

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _safe_int(
        values: Any,
        index: int,
    ) -> Optional[int]:

        if values is None:
            return None

        try:
            return int(values[index])

        except (
            IndexError,
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _union_boxes(
        boxes: Sequence[
            Optional[BoundingBox]
        ],
    ) -> Optional[BoundingBox]:

        valid = [
            box
            for box in boxes
            if box is not None
        ]

        if not valid:
            return None

        x1 = min(
            box.x
            for box in valid
        )

        y1 = min(
            box.y
            for box in valid
        )

        x2 = max(
            box.x + box.width
            for box in valid
        )

        y2 = max(
            box.y + box.height
            for box in valid
        )

        return BoundingBox(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
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
    # FAILURE
    # ========================================================================

    def _failure(
        self,
        started: float,
        width: int,
        height: int,
        error: str,
    ) -> OCRResult:

        result = OCRResult(
            text="",
            regions=[],
            confidence=0.0,
            backend=OCRBackend.NONE,
            processing_time=(
                time.perf_counter()
                - started
            ),
            image_width=width,
            image_height=height,
            success=False,
            error=error,
        )

        with self._lock:
            self._last_result = result

        return result


# ============================================================================
# SHARED ENGINE
# ============================================================================


_default_engine: Optional[
    OCREngine
] = None

_default_engine_lock = (
    threading.RLock()
)


def get_ocr_engine() -> OCREngine:

    global _default_engine

    with _default_engine_lock:

        if _default_engine is None:

            _default_engine = OCREngine()

        return _default_engine


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def read_text(
    image: Any,
    *,
    language: Optional[str] = None,
) -> OCRResult:

    return get_ocr_engine().read(
        image,
        language=language,
    )


def extract_text(
    image: Any,
    *,
    language: Optional[str] = None,
) -> str:

    return get_ocr_engine().extract_text(
        image,
        language=language,
    )


def detect_text_regions(
    image: Any,
    *,
    language: Optional[str] = None,
) -> list[OCRTextRegion]:

    return get_ocr_engine().detect_text_regions(
        image,
        language=language,
    )


def is_ocr_available() -> bool:

    return get_ocr_engine().is_available()


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "OCRBackend",
    "TextOrientation",
    "OCRTextRegion",
    "OCRResult",
    "OCRConfig",
    "OCREngine",
    "get_ocr_engine",
    "read_text",
    "extract_text",
    "detect_text_regions",
    "is_ocr_available",
]


