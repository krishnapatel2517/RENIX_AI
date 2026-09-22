"""
RENIX Vision — Screen Understanding Engine

Purpose:
    Converts captured screen frames into structured information that
    RENIX can reason about.

Responsibilities:
    - Screen/frame analysis
    - OCR
    - UI/text region detection
    - Basic object/region detection
    - Screen classification
    - Coordinate mapping
    - Change detection
    - Visual summaries
    - Searchable structured screen state

This module DOES NOT:
    - click
    - type
    - press keys
    - execute commands
    - modify files
    - control applications

Those responsibilities belong to RENIX computer/control modules.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

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


logger = logging.getLogger(
    "RENIX.vision.screen_understanding"
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class BoundingBox:
    """Rectangle describing a detected screen region."""

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
    def center(self) -> tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )

    @property
    def area(self) -> int:
        return max(0, self.width) * max(
            0, self.height
        )

    def contains(
        self,
        x: int,
        y: int,
    ) -> bool:
        return (
            self.x <= x <= self.x2
            and self.y <= y <= self.y2
        )

    def intersects(
        self,
        other: "BoundingBox",
    ) -> bool:

        return not (
            self.x2 < other.x
            or other.x2 < self.x
            or self.y2 < other.y
            or other.y2 < self.y
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class TextRegion:
    """OCR result for one piece of screen text."""

    text: str

    confidence: float

    box: BoundingBox

    line_number: int = 0

    word_number: int = 0

    block_number: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": self.box.to_dict(),
            "line_number": self.line_number,
            "word_number": self.word_number,
            "block_number": self.block_number,
            "metadata": dict(self.metadata),
        }


@dataclass
class VisualRegion:
    """Generic visually detected region."""

    region_type: str

    box: BoundingBox

    confidence: float = 0.0

    label: Optional[str] = None

    text: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_type": self.region_type,
            "box": self.box.to_dict(),
            "confidence": self.confidence,
            "label": self.label,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass
class ScreenChange:
    """Describes a visual change between two frames."""

    changed: bool

    score: float

    changed_pixels: int

    total_pixels: int

    percentage: float

    box: Optional[BoundingBox] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "score": self.score,
            "changed_pixels": self.changed_pixels,
            "total_pixels": self.total_pixels,
            "percentage": self.percentage,
            "box": (
                self.box.to_dict()
                if self.box
                else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass
class ScreenAnalysis:
    """Complete structured representation of a screen frame."""

    timestamp: float

    width: int

    height: int

    frame_number: Optional[int] = None

    monitor_index: Optional[int] = None

    screen_type: str = "unknown"

    application_hint: Optional[str] = None

    text: str = ""

    text_regions: list[TextRegion] = field(
        default_factory=list
    )

    visual_regions: list[VisualRegion] = field(
        default_factory=list
    )

    change: Optional[ScreenChange] = None

    screen_hash: str = ""

    confidence: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "frame_number": self.frame_number,
            "monitor_index": self.monitor_index,
            "screen_type": self.screen_type,
            "application_hint": self.application_hint,
            "text": self.text,
            "text_regions": [
                item.to_dict()
                for item in self.text_regions
            ],
            "visual_regions": [
                item.to_dict()
                for item in self.visual_regions
            ],
            "change": (
                self.change.to_dict()
                if self.change
                else None
            ),
            "screen_hash": self.screen_hash,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass
class ScreenUnderstandingConfig:
    """Configuration for screen analysis."""

    enable_ocr: bool = True

    minimum_ocr_confidence: float = 35.0

    detect_visual_regions: bool = True

    detect_buttons: bool = True

    detect_input_fields: bool = True

    detect_panels: bool = True

    detect_lines: bool = False

    detect_screen_changes: bool = True

    change_threshold: float = 12.0

    change_percentage_threshold: float = 1.0

    resize_width: int = 1600

    ocr_language: str = "eng"

    use_preprocessing: bool = True

    cache_results: bool = True

    max_cached_results: int = 20


# ============================================================================
# SCREEN UNDERSTANDING ENGINE
# ============================================================================


class ScreenUnderstanding:
    """
    RENIX screen-understanding engine.

    Example:

        engine = ScreenUnderstanding()

        result = engine.analyze(frame.image)

        print(result.text)
        print(result.screen_type)
    """

    def __init__(
        self,
        config: Optional[
            ScreenUnderstandingConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or ScreenUnderstandingConfig()
        )

        self._lock = threading.RLock()

        self._previous_image: Any = None

        self._previous_hash: Optional[
            str
        ] = None

        self._last_analysis: Optional[
            ScreenAnalysis
        ] = None

        self._cache: dict[
            str,
            ScreenAnalysis,
        ] = {}

        self._callbacks: list[
            Callable[
                [ScreenAnalysis],
                None,
            ]
        ] = []

    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================

    def analyze(
        self,
        image: Any,
        *,
        frame_number: Optional[int] = None,
        monitor_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> ScreenAnalysis:

        if image is None:

            raise ValueError(
                "image cannot be None."
            )

        if np is None:

            raise RuntimeError(
                "NumPy is required for "
                "screen understanding."
            )

        timestamp = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        image = np.asarray(image)

        if image.size == 0:

            raise ValueError(
                "image cannot be empty."
            )

        height, width = image.shape[:2]

        screen_hash = self._hash_image(
            image
        )

        if self.config.cache_results:

            cached = self._get_cached(
                screen_hash
            )

            if cached is not None:

                # Return a fresh timestamp/
                # frame context while preserving
                # analysis content.
                cached.timestamp = timestamp
                cached.frame_number = (
                    frame_number
                )
                cached.monitor_index = (
                    monitor_index
                )

                return cached

        text_regions: list[
            TextRegion
        ] = []

        text = ""

        if self.config.enable_ocr:

            try:

                text_regions = (
                    self.extract_text(
                        image
                    )
                )

                text = self._combine_text(
                    text_regions
                )

            except Exception as exc:

                logger.exception(
                    "OCR analysis failed: %s",
                    exc,
                )

        visual_regions: list[
            VisualRegion
        ] = []

        if self.config.detect_visual_regions:

            try:

                visual_regions = (
                    self.detect_visual_regions(
                        image
                    )
                )

            except Exception as exc:

                logger.exception(
                    "Visual region detection failed: %s",
                    exc,
                )

        change: Optional[
            ScreenChange
        ] = None

        if self.config.detect_screen_changes:

            try:

                change = (
                    self.compare_with_previous(
                        image
                    )
                )

            except Exception as exc:

                logger.exception(
                    "Screen change detection failed: %s",
                    exc,
                )

        screen_type = (
            self.classify_screen(
                image=image,
                text=text,
                regions=visual_regions,
            )
        )

        application_hint = (
            self.detect_application_hint(
                text=text
            )
        )

        confidence = (
            self._calculate_analysis_confidence(
                text_regions=text_regions,
                visual_regions=visual_regions,
                screen_type=screen_type,
            )
        )

        analysis = ScreenAnalysis(
            timestamp=timestamp,
            width=int(width),
            height=int(height),
            frame_number=frame_number,
            monitor_index=monitor_index,
            screen_type=screen_type,
            application_hint=application_hint,
            text=text,
            text_regions=text_regions,
            visual_regions=visual_regions,
            change=change,
            screen_hash=screen_hash,
            confidence=confidence,
        )

        with self._lock:

            self._last_analysis = analysis

            if self.config.cache_results:

                self._store_cache(
                    screen_hash,
                    analysis,
                )

        self._notify_callbacks(
            analysis
        )

        return analysis

    # ========================================================================
    # OCR
    # ========================================================================

    def extract_text(
        self,
        image: Any,
    ) -> list[TextRegion]:

        if pytesseract is None:

            logger.warning(
                "pytesseract is unavailable. "
                "OCR skipped."
            )

            return []

        if np is None:

            return []

        processed = image

        if self.config.use_preprocessing:

            processed = (
                self._preprocess_for_ocr(
                    image
                )
            )

        try:

            data = (
                pytesseract.image_to_data(
                    processed,
                    lang=self.config.ocr_language,
                    output_type=(
                        pytesseract.Output.DICT
                    ),
                    config="--psm 6",
                )
            )

        except Exception as exc:

            logger.exception(
                "Tesseract OCR failed: %s",
                exc,
            )

            return []

        results: list[
            TextRegion
        ] = []

        count = len(
            data.get("text", [])
        )

        for index in range(count):

            raw_text = str(
                data["text"][index]
            ).strip()

            if not raw_text:
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

            if (
                confidence
                < self.config.minimum_ocr_confidence
            ):
                continue

            try:

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

            except (
                ValueError,
                TypeError,
            ):

                continue

            if width <= 0 or height <= 0:
                continue

            region = TextRegion(
                text=raw_text,
                confidence=confidence,
                box=BoundingBox(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                ),
                line_number=self._safe_int(
                    data.get(
                        "line_num",
                        [],
                    ),
                    index,
                ),
                word_number=self._safe_int(
                    data.get(
                        "word_num",
                        [],
                    ),
                    index,
                ),
                block_number=self._safe_int(
                    data.get(
                        "block_num",
                        [],
                    ),
                    index,
                ),
            )

            results.append(region)

        return self._merge_adjacent_text(
            results
        )

    def _preprocess_for_ocr(
        self,
        image: Any,
    ) -> Any:

        if cv2 is None:

            return image

        try:

            processed = image

            if (
                processed.ndim == 3
                and processed.shape[2] >= 3
            ):

                processed = cv2.cvtColor(
                    processed,
                    cv2.COLOR_BGR2GRAY,
                )

            # Upscale smaller screens for OCR.
            height, width = (
                processed.shape[:2]
            )

            if width < 1200:

                scale = (
                    1200 / width
                )

                processed = cv2.resize(
                    processed,
                    (
                        int(
                            width * scale
                        ),
                        int(
                            height * scale
                        ),
                    ),
                    interpolation=(
                        cv2.INTER_CUBIC
                    ),
                )

            processed = cv2.GaussianBlur(
                processed,
                (3, 3),
                0,
            )

            processed = cv2.threshold(
                processed,
                0,
                255,
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU,
            )[1]

            return processed

        except Exception:

            logger.debug(
                "OCR preprocessing failed.",
                exc_info=True,
            )

            return image

    # ========================================================================
    # TEXT HELPERS
    # ========================================================================

    def _combine_text(
        self,
        regions: Iterable[
            TextRegion
        ],
    ) -> str:

        ordered = sorted(
            regions,
            key=lambda item: (
                item.box.y,
                item.box.x,
            ),
        )

        lines: list[str] = []

        current_line: list[
            TextRegion
        ] = []

        current_y: Optional[int] = None

        for region in ordered:

            y = region.box.y

            if (
                current_y is None
                or abs(y - current_y) <= 15
            ):

                current_line.append(
                    region
                )

            else:

                if current_line:

                    current_line.sort(
                        key=lambda item:
                        item.box.x
                    )

                    lines.append(
                        " ".join(
                            item.text
                            for item
                            in current_line
                        )
                    )

                current_line = [
                    region
                ]

            current_y = y

        if current_line:

            current_line.sort(
                key=lambda item:
                item.box.x
            )

            lines.append(
                " ".join(
                    item.text
                    for item
                    in current_line
                )
            )

        return "\n".join(
            line.strip()
            for line in lines
            if line.strip()
        )

    def _merge_adjacent_text(
        self,
        regions: list[TextRegion],
    ) -> list[TextRegion]:

        if len(regions) < 2:
            return regions

        regions = sorted(
            regions,
            key=lambda item: (
                item.box.y,
                item.box.x,
            ),
        )

        merged: list[
            TextRegion
        ] = []

        for region in regions:

            if not merged:

                merged.append(region)
                continue

            previous = merged[-1]

            same_line = (
                abs(
                    region.box.y
                    - previous.box.y
                )
                <= max(
                    10,
                    int(
                        previous.box.height
                        * 0.6
                    ),
                )
            )

            gap = (
                region.box.x
                - previous.box.x2
            )

            should_merge = (
                same_line
                and 0 <= gap <= 20
            )

            if should_merge:

                new_x = min(
                    previous.box.x,
                    region.box.x,
                )

                new_y = min(
                    previous.box.y,
                    region.box.y,
                )

                new_x2 = max(
                    previous.box.x2,
                    region.box.x2,
                )

                new_y2 = max(
                    previous.box.y2,
                    region.box.y2,
                )

                previous.text = (
                    previous.text
                    + " "
                    + region.text
                )

                previous.box = BoundingBox(
                    x=new_x,
                    y=new_y,
                    width=(
                        new_x2 - new_x
                    ),
                    height=(
                        new_y2 - new_y
                    ),
                )

                previous.confidence = (
                    previous.confidence
                    + region.confidence
                ) / 2

            else:

                merged.append(region)

        return merged

    # ========================================================================
    # VISUAL REGION DETECTION
    # ========================================================================

    def detect_visual_regions(
        self,
        image: Any,
    ) -> list[VisualRegion]:

        if cv2 is None or np is None:

            return []

        regions: list[
            VisualRegion
        ] = []

        if self.config.detect_buttons:

            regions.extend(
                self._detect_buttons(
                    image
                )
            )

        if self.config.detect_input_fields:

            regions.extend(
                self._detect_input_fields(
                    image
                )
            )

        if self.config.detect_panels:

            regions.extend(
                self._detect_panels(
                    image
                )
            )

        if self.config.detect_lines:

            regions.extend(
                self._detect_lines(
                    image
                )
            )

        return self._remove_overlapping_regions(
            regions
        )

    # ========================================================================
    # BUTTON DETECTION
    # ========================================================================

    def _detect_buttons(
        self,
        image: Any,
    ) -> list[VisualRegion]:

        if cv2 is None:

            return []

        results: list[
            VisualRegion
        ] = []

        try:

            gray = self._to_gray(
                image
            )

            edges = cv2.Canny(
                gray,
                50,
                150,
            )

            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            image_height, image_width = (
                gray.shape[:2]
            )

            image_area = (
                image_width
                * image_height
            )

            for contour in contours:

                x, y, w, h = (
                    cv2.boundingRect(
                        contour
                    )
                )

                if w < 35 or h < 18:
                    continue

                if w > image_width * 0.7:
                    continue

                if h > image_height * 0.3:
                    continue

                area = w * h

                if area < image_area * 0.00005:
                    continue

                aspect = (
                    w / max(h, 1)
                )

                if aspect < 1.2 or aspect > 12:
                    continue

                perimeter = cv2.arcLength(
                    contour,
                    True,
                )

                if perimeter <= 0:
                    continue

                approx = cv2.approxPolyDP(
                    contour,
                    0.04 * perimeter,
                    True,
                )

                if len(approx) not in {
                    4,
                    5,
                    6,
                }:
                    continue

                results.append(
                    VisualRegion(
                        region_type="button",
                        box=BoundingBox(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                        ),
                        confidence=0.55,
                    )
                )

        except Exception:

            logger.debug(
                "Button detection failed.",
                exc_info=True,
            )

        return results

    # ========================================================================
    # INPUT FIELD DETECTION
    # ========================================================================

    def _detect_input_fields(
        self,
        image: Any,
    ) -> list[VisualRegion]:

        if cv2 is None:

            return []

        results: list[
            VisualRegion
        ] = []

        try:

            gray = self._to_gray(
                image
            )

            edges = cv2.Canny(
                gray,
                40,
                120,
            )

            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:

                x, y, w, h = (
                    cv2.boundingRect(
                        contour
                    )
                )

                if w < 100:
                    continue

                if h < 20:
                    continue

                if h > 120:
                    continue

                aspect = (
                    w / max(h, 1)
                )

                if aspect < 3:
                    continue

                if aspect > 40:
                    continue

                area = w * h

                if area < 2000:
                    continue

                perimeter = cv2.arcLength(
                    contour,
                    True,
                )

                if perimeter <= 0:
                    continue

                approx = cv2.approxPolyDP(
                    contour,
                    0.03 * perimeter,
                    True,
                )

                if len(approx) != 4:
                    continue

                results.append(
                    VisualRegion(
                        region_type="input_field",
                        box=BoundingBox(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                        ),
                        confidence=0.58,
                    )
                )

        except Exception:

            logger.debug(
                "Input field detection failed.",
                exc_info=True,
            )

        return results

    # ========================================================================
    # PANEL DETECTION
    # ========================================================================

    def _detect_panels(
        self,
        image: Any,
    ) -> list[VisualRegion]:

        if cv2 is None:

            return []

        results: list[
            VisualRegion
        ] = []

        try:

            gray = self._to_gray(
                image
            )

            edges = cv2.Canny(
                gray,
                30,
                100,
            )

            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            height, width = (
                gray.shape[:2]
            )

            for contour in contours:

                x, y, w, h = (
                    cv2.boundingRect(
                        contour
                    )
                )

                if w < width * 0.08:
                    continue

                if h < height * 0.05:
                    continue

                if w > width * 0.98:
                    continue

                if h > height * 0.95:
                    continue

                aspect = (
                    w / max(h, 1)
                )

                if (
                    aspect < 0.25
                    or aspect > 10
                ):
                    continue

                results.append(
                    VisualRegion(
                        region_type="panel",
                        box=BoundingBox(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                        ),
                        confidence=0.45,
                    )
                )

        except Exception:

            logger.debug(
                "Panel detection failed.",
                exc_info=True,
            )

        return results

    # ========================================================================
    # LINE DETECTION
    # ========================================================================

    def _detect_lines(
        self,
        image: Any,
    ) -> list[VisualRegion]:

        if cv2 is None:

            return []

        results: list[
            VisualRegion
        ] = []

        try:

            gray = self._to_gray(
                image
            )

            edges = cv2.Canny(
                gray,
                50,
                150,
            )

            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10,
            )

            if lines is None:
                return []

            for line in lines:

                x1, y1, x2, y2 = (
                    line[0]
                )

                length = (
                    (
                        (x2 - x1) ** 2
                        + (y2 - y1) ** 2
                    )
                    ** 0.5
                )

                if length < 100:
                    continue

                x = min(x1, x2)
                y = min(y1, y2)

                width = abs(
                    x2 - x1
                )

                height = abs(
                    y2 - y1
                )

                results.append(
                    VisualRegion(
                        region_type="line",
                        box=BoundingBox(
                            x=x,
                            y=y,
                            width=max(
                                1,
                                width,
                            ),
                            height=max(
                                1,
                                height,
                            ),
                        ),
                        confidence=0.65,
                        metadata={
                            "length": length
                        },
                    )
                )

        except Exception:

            logger.debug(
                "Line detection failed.",
                exc_info=True,
            )

        return results

    # ========================================================================
    # SCREEN CLASSIFICATION
    # ========================================================================

    def classify_screen(
        self,
        image: Any,
        text: str = "",
        regions: Optional[
            list[VisualRegion]
        ] = None,
    ) -> str:

        del image

        regions = regions or []

        normalized = text.lower()

        # Browser/web pages.
        browser_keywords = [
            "http",
            "www",
            "search",
            "sign in",
            "login",
            "address bar",
            "bookmarks",
            "google",
            "youtube",
            "github",
            "reddit",
        ]

        browser_score = sum(
            1
            for keyword
            in browser_keywords
            if keyword in normalized
        )

        if browser_score >= 2:

            return "browser"

        # Terminal / command prompt.
        terminal_keywords = [
            "powershell",
            "command prompt",
            "cmd.exe",
            "terminal",
            "python.exe",
            "pip install",
            "git ",
            "npm ",
            "traceback",
            "directory",
        ]

        terminal_score = sum(
            1
            for keyword
            in terminal_keywords
            if keyword in normalized
        )

        if terminal_score >= 2:

            return "terminal"

        # Code editor / IDE.
        coding_keywords = [
            "def ",
            "class ",
            "import ",
            "return ",
            "function",
            "variable",
            "debug",
            "source control",
            "explorer",
            "editor",
        ]

        coding_score = sum(
            1
            for keyword
            in coding_keywords
            if keyword in normalized
        )

        if coding_score >= 3:

            return "code_editor"

        # File manager.
        file_keywords = [
            "this pc",
            "documents",
            "downloads",
            "desktop",
            "pictures",
            "videos",
            "file name",
            "folder",
            "date modified",
            "size",
        ]

        file_score = sum(
            1
            for keyword
            in file_keywords
            if keyword in normalized
        )

        if file_score >= 2:

            return "file_manager"

        # Settings.
        settings_keywords = [
            "settings",
            "privacy",
            "bluetooth",
            "network",
            "display",
            "system",
            "accounts",
            "windows update",
        ]

        settings_score = sum(
            1
            for keyword
            in settings_keywords
            if keyword in normalized
        )

        if settings_score >= 2:

            return "settings"

        # Media.
        media_keywords = [
            "play",
            "pause",
            "volume",
            "playlist",
            "music",
            "video",
            "next track",
            "previous track",
        ]

        media_score = sum(
            1
            for keyword
            in media_keywords
            if keyword in normalized
        )

        if media_score >= 3:

            return "media"

        # Presence of many button-like regions.
        button_count = sum(
            1
            for region in regions
            if region.region_type == "button"
        )

        if button_count >= 10:

            return "application_ui"

        return "desktop"

    # ========================================================================
    # APPLICATION HINT
    # ========================================================================

    def detect_application_hint(
        self,
        text: str,
    ) -> Optional[str]:

        normalized = text.lower()

        known_apps = {
            "google chrome": [
                "chrome",
                "google chrome",
            ],
            "microsoft edge": [
                "edge",
                "microsoft edge",
            ],
            "firefox": [
                "firefox",
            ],
            "visual studio code": [
                "visual studio code",
                "vs code",
                "vscode",
            ],
            "pycharm": [
                "pycharm",
            ],
            "windows terminal": [
                "windows terminal",
            ],
            "powershell": [
                "powershell",
            ],
            "command prompt": [
                "command prompt",
                "cmd.exe",
            ],
            "file explorer": [
                "file explorer",
                "this pc",
            ],
            "spotify": [
                "spotify",
            ],
            "youtube": [
                "youtube",
            ],
            "discord": [
                "discord",
            ],
            "steam": [
                "steam",
            ],
            "github": [
                "github",
            ],
        }

        for application, keywords in (
            known_apps.items()
        ):

            for keyword in keywords:

                if keyword in normalized:

                    return application

        return None

    # ========================================================================
    # SCREEN CHANGE DETECTION
    # ========================================================================

    def compare_with_previous(
        self,
        image: Any,
    ) -> ScreenChange:

        if np is None:

            raise RuntimeError(
                "NumPy is required."
            )

        current = np.asarray(
            image
        )

        with self._lock:

            previous = self._previous_image

            self._previous_image = (
                current.copy()
            )

        if previous is None:

            return ScreenChange(
                changed=True,
                score=100.0,
                changed_pixels=(
                    current.shape[0]
                    * current.shape[1]
                ),
                total_pixels=(
                    current.shape[0]
                    * current.shape[1]
                ),
                percentage=100.0,
                metadata={
                    "reason": "first_frame"
                },
            )

        try:

            current_gray = self._to_gray(
                current
            )

            previous_gray = self._to_gray(
                previous
            )

            if (
                current_gray.shape
                != previous_gray.shape
            ):

                return ScreenChange(
                    changed=True,
                    score=100.0,
                    changed_pixels=(
                        current_gray.size
                    ),
                    total_pixels=(
                        current_gray.size
                    ),
                    percentage=100.0,
                    metadata={
                        "reason": (
                            "resolution_changed"
                        )
                    },
                )

            diff = cv2.absdiff(
                current_gray,
                previous_gray,
            ) if cv2 is not None else (
                np.abs(
                    current_gray.astype(
                        np.int16
                    )
                    - previous_gray.astype(
                        np.int16
                    )
                ).astype(np.uint8)
            )

            score = float(
                np.mean(diff)
            )

            threshold = max(
                1,
                int(
                    self.config.change_threshold
                ),
            )

            changed_mask = (
                diff >= threshold
            )

            changed_pixels = int(
                np.count_nonzero(
                    changed_mask
                )
            )

            total_pixels = int(
                diff.size
            )

            percentage = (
                (
                    changed_pixels
                    / max(
                        total_pixels,
                        1,
                    )
                )
                * 100.0
            )

            changed = (
                score
                >= self.config.change_threshold
                or percentage
                >= self.config.change_percentage_threshold
            )

            box = (
                self._get_change_box(
                    changed_mask
                )
                if changed
                else None
            )

            return ScreenChange(
                changed=changed,
                score=score,
                changed_pixels=changed_pixels,
                total_pixels=total_pixels,
                percentage=percentage,
                box=box,
            )

        except Exception as exc:

            logger.exception(
                "Screen comparison failed: %s",
                exc,
            )

            return ScreenChange(
                changed=True,
                score=100.0,
                changed_pixels=0,
                total_pixels=0,
                percentage=100.0,
                metadata={
                    "error": str(exc)
                },
            )

    def _get_change_box(
        self,
        mask: Any,
    ) -> Optional[BoundingBox]:

        if np is None:
            return None

        try:

            ys, xs = np.where(
                mask
            )

            if len(xs) == 0:

                return None

            x1 = int(
                np.min(xs)
            )

            y1 = int(
                np.min(ys)
            )

            x2 = int(
                np.max(xs)
            )

            y2 = int(
                np.max(ys)
            )

            return BoundingBox(
                x=x1,
                y=y1,
                width=x2 - x1 + 1,
                height=y2 - y1 + 1,
            )

        except Exception:

            return None

    # ========================================================================
    # IMAGE HASH
    # ========================================================================

    def _hash_image(
        self,
        image: Any,
    ) -> str:

        if np is None:

            return ""

        try:

            # Resize before hashing so that the
            # hash is inexpensive for large screens.
            sample = image

            if (
                cv2 is not None
                and image.shape[1] > 256
            ):

                scale = (
                    256
                    / image.shape[1]
                )

                sample = cv2.resize(
                    image,
                    (
                        256,
                        max(
                            1,
                            int(
                                image.shape[0]
                                * scale
                            ),
                        ),
                    ),
                )

            return hashlib.sha256(
                np.ascontiguousarray(
                    sample
                ).tobytes()
            ).hexdigest()

        except Exception:

            return ""

    # ========================================================================
    # UTILITY
    # ========================================================================

    def _to_gray(
        self,
        image: Any,
    ) -> Any:

        if np is None:

            raise RuntimeError(
                "NumPy is unavailable."
            )

        if image.ndim == 2:

            return image

        if cv2 is not None:

            if image.shape[2] >= 3:

                return cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )

        # Fallback RGB/BGR weighted grayscale.
        return np.mean(
            image[:, :, :3],
            axis=2,
        ).astype(np.uint8)

    def _safe_int(
        self,
        values: Any,
        index: int,
    ) -> int:

        try:

            return int(
                values[index]
            )

        except Exception:

            return 0

    def _calculate_analysis_confidence(
        self,
        text_regions: list[
            TextRegion
        ],
        visual_regions: list[
            VisualRegion
        ],
        screen_type: str,
    ) -> float:

        confidence = 0.20

        if text_regions:

            average_ocr = sum(
                item.confidence
                for item in text_regions
            ) / len(text_regions)

            confidence += min(
                0.40,
                average_ocr / 250,
            )

        if visual_regions:

            average_visual = sum(
                item.confidence
                for item in visual_regions
            ) / len(visual_regions)

            confidence += min(
                0.25,
                average_visual * 0.25,
            )

        if screen_type != "unknown":

            confidence += 0.15

        return min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

    # ========================================================================
    # REGION FILTERING
    # ========================================================================

    def _remove_overlapping_regions(
        self,
        regions: list[
            VisualRegion
        ],
    ) -> list[VisualRegion]:

        if not regions:
            return []

        # Prefer higher-confidence regions.
        ordered = sorted(
            regions,
            key=lambda item: (
                item.confidence,
                item.box.area,
            ),
            reverse=True,
        )

        selected: list[
            VisualRegion
        ] = []

        for region in ordered:

            overlap = False

            for existing in selected:

                if not region.box.intersects(
                    existing.box
                ):
                    continue

                intersection = (
                    self._intersection_area(
                        region.box,
                        existing.box,
                    )
                )

                smaller = min(
                    region.box.area,
                    existing.box.area,
                )

                if (
                    smaller > 0
                    and intersection
                    / smaller
                    > 0.75
                ):

                    overlap = True
                    break

            if not overlap:

                selected.append(
                    region
                )

        return selected

    def _intersection_area(
        self,
        a: BoundingBox,
        b: BoundingBox,
    ) -> int:

        left = max(
            a.x,
            b.x,
        )

        top = max(
            a.y,
            b.y,
        )

        right = min(
            a.x2,
            b.x2,
        )

        bottom = min(
            a.y2,
            b.y2,
        )

        if right <= left or bottom <= top:

            return 0

        return (
            right - left
        ) * (
            bottom - top
        )

    # ========================================================================
    # SEARCH
    # ========================================================================

    def find_text(
        self,
        analysis: ScreenAnalysis,
        query: str,
        *,
        case_sensitive: bool = False,
    ) -> list[TextRegion]:

        if not query:

            return []

        flags = (
            0
            if case_sensitive
            else re.IGNORECASE
        )

        pattern = re.compile(
            re.escape(query),
            flags,
        )

        return [
            region
            for region
            in analysis.text_regions
            if pattern.search(
                region.text
            )
        ]

    def find_text_containing(
        self,
        analysis: ScreenAnalysis,
        query: str,
    ) -> list[TextRegion]:

        normalized = (
            query.lower().strip()
        )

        if not normalized:

            return []

        return [
            region
            for region
            in analysis.text_regions
            if normalized
            in region.text.lower()
        ]

    def get_text_at(
        self,
        analysis: ScreenAnalysis,
        x: int,
        y: int,
    ) -> Optional[TextRegion]:

        candidates = [
            region
            for region
            in analysis.text_regions
            if region.box.contains(
                x,
                y,
            )
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda region:
            region.box.area,
        )

    def get_region_at(
        self,
        analysis: ScreenAnalysis,
        x: int,
        y: int,
    ) -> Optional[VisualRegion]:

        candidates = [
            region
            for region
            in analysis.visual_regions
            if region.box.contains(
                x,
                y,
            )
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda region:
            region.box.area,
        )

    # ========================================================================
    # CACHE
    # ========================================================================

    def _get_cached(
        self,
        screen_hash: str,
    ) -> Optional[
        ScreenAnalysis
    ]:

        if not screen_hash:
            return None

        with self._lock:

            return self._cache.get(
                screen_hash
            )

    def _store_cache(
        self,
        screen_hash: str,
        analysis: ScreenAnalysis,
    ) -> None:

        if not screen_hash:
            return

        with self._lock:

            self._cache[
                screen_hash
            ] = analysis

            while (
                len(self._cache)
                > self.config.max_cached_results
            ):

                oldest_key = next(
                    iter(self._cache)
                )

                del self._cache[
                    oldest_key
                ]

    def clear_cache(
        self,
    ) -> None:

        with self._lock:

            self._cache.clear()

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def add_callback(
        self,
        callback: Callable[
            [ScreenAnalysis],
            None,
        ],
    ) -> None:

        if not callable(callback):

            raise TypeError(
                "callback must be callable."
            )

        with self._lock:

            if callback not in self._callbacks:

                self._callbacks.append(
                    callback
                )

    def remove_callback(
        self,
        callback: Callable[
            [ScreenAnalysis],
            None,
        ],
    ) -> None:

        with self._lock:

            if callback in self._callbacks:

                self._callbacks.remove(
                    callback
                )

    def _notify_callbacks(
        self,
        analysis: ScreenAnalysis,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks
            )

        for callback in callbacks:

            try:

                callback(analysis)

            except Exception:

                logger.exception(
                    "Screen analysis callback failed."
                )

    # ========================================================================
    # LAST ANALYSIS
    # ========================================================================

    def get_last_analysis(
        self,
    ) -> Optional[ScreenAnalysis]:

        with self._lock:

            return self._last_analysis

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(
        self,
    ) -> None:

        with self._lock:

            self._previous_image = None

            self._previous_hash = None

            self._last_analysis = None

            self._cache.clear()


# ============================================================================
# DEFAULT ENGINE
# ============================================================================


_default_engine: Optional[
    ScreenUnderstanding
] = None

_default_engine_lock = (
    threading.RLock()
)


def get_screen_understanding(
) -> ScreenUnderstanding:

    global _default_engine

    with _default_engine_lock:

        if _default_engine is None:

            _default_engine = (
                ScreenUnderstanding()
            )

        return _default_engine


# ============================================================================
# CONVENIENCE API
# ============================================================================


def analyze_screen(
    image: Any,
    **kwargs: Any,
) -> ScreenAnalysis:

    return (
        get_screen_understanding()
        .analyze(
            image,
            **kwargs,
        )
    )


def extract_screen_text(
    image: Any,
) -> list[TextRegion]:

    return (
        get_screen_understanding()
        .extract_text(image)
    )


def classify_screen(
    image: Any,
    text: str = "",
) -> str:

    return (
        get_screen_understanding()
        .classify_screen(
            image=image,
            text=text,
        )
    )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "BoundingBox",
    "TextRegion",
    "VisualRegion",
    "ScreenChange",
    "ScreenAnalysis",
    "ScreenUnderstandingConfig",
    "ScreenUnderstanding",
    "get_screen_understanding",
    "analyze_screen",
    "extract_screen_text",
    "classify_screen",
]


