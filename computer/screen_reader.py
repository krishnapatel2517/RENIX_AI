"""
RENIX Screen Reader
===================

Reads visible text from the screen and provides basic
screen-understanding utilities for RENIX.

Features:
- OCR from full screen
- OCR from screen regions
- Read text from screenshots
- Find text on screen
- Return text with bounding boxes
- Capture screen automatically
- Optional Tesseract backend
- Optional EasyOCR backend
- Safe fallback handling

Recommended dependencies:
    pip install pillow
    pip install pytesseract

Optional:
    pip install easyocr
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A detected text region on the screen."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 0.0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.left + self.width // 2,
            self.top + self.height // 2,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["right"] = self.right
        data["bottom"] = self.bottom
        data["center"] = self.center

        return data


class ScreenReader:
    """
    High-level screen text reader for RENIX.

    Example:

        reader = ScreenReader()

        text = reader.read_screen()

        results = reader.find_text(
            "Settings"
        )
    """

    def __init__(
        self,
        enabled: bool = True,
        language: str = "eng",
        confidence_threshold: float = 30.0,
        tesseract_path: Optional[str] = None,
    ) -> None:

        self.enabled = enabled

        self.language = language

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.tesseract_path = (
            tesseract_path
        )

        self._pytesseract = None
        self._easyocr = None
        self._reader = None

        self._pyautogui = None

        self._lock = threading.RLock()

        self.last_results: list[
            TextRegion
        ] = []

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:

        try:

            import pytesseract

            self._pytesseract = pytesseract

            if self.tesseract_path:

                pytesseract.pytesseract.tesseract_cmd = (
                    self.tesseract_path
                )

        except Exception as exc:

            logger.debug(
                "pytesseract unavailable: %s",
                exc,
            )

        try:

            import pyautogui

            self._pyautogui = pyautogui

        except Exception as exc:

            logger.debug(
                "PyAutoGUI unavailable: %s",
                exc,
            )

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX screen reader is disabled."
            )

        if (
            self._pytesseract is None
            and self._easyocr is None
        ):
            raise RuntimeError(
                "No OCR backend is available."
            )

    # ==========================================================
    # STATE
    # ==========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    # ==========================================================
    # SCREEN CAPTURE
    # ==========================================================

    def capture_screen(self) -> Any:

        if self._pyautogui is None:

            raise RuntimeError(
                "PyAutoGUI is required for "
                "screen capture."
            )

        return self._pyautogui.screenshot()

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Any:

        if self._pyautogui is None:

            raise RuntimeError(
                "PyAutoGUI is required for "
                "screen capture."
            )

        return self._pyautogui.screenshot(
            region=(
                int(x),
                int(y),
                int(width),
                int(height),
            )
        )

    # ==========================================================
    # OCR TEXT
    # ==========================================================

    def read_image(
        self,
        image: Any,
    ) -> str:
        """
        Extract plain text from a PIL image.
        """

        self._check()

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        with self._lock:

            if self._pytesseract is not None:

                try:

                    text = (
                        self._pytesseract.image_to_string(
                            image,
                            lang=self.language,
                        )
                    )

                    return text.strip()

                except Exception as exc:

                    logger.warning(
                        "Tesseract OCR failed: %s",
                        exc,
                    )

            if self._easyocr is not None:

                try:

                    results = (
                        self._easyocr.readtext(
                            image
                        )
                    )

                    return "\n".join(
                        result[1]
                        for result in results
                    ).strip()

                except Exception as exc:

                    logger.warning(
                        "EasyOCR failed: %s",
                        exc,
                    )

        return ""

    def read_file(
        self,
        image_path: str,
    ) -> str:
        """
        Read text from an image file.
        """

        self._check()

        path = (
            Path(image_path)
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        try:

            from PIL import Image

            image = Image.open(path)

            return self.read_image(
                image
            )

        except Exception as exc:

            raise RuntimeError(
                f"Unable to read image: {exc}"
            ) from exc

    # ==========================================================
    # READ SCREEN
    # ==========================================================

    def read_screen(self) -> str:

        image = self.capture_screen()

        return self.read_image(
            image
        )

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> str:

        image = self.capture_region(
            x,
            y,
            width,
            height,
        )

        return self.read_image(
            image
        )

    # ==========================================================
    # OCR WITH BOUNDING BOXES
    # ==========================================================

    def detect_text(
        self,
        image: Any,
        min_confidence: Optional[
            float
        ] = None,
    ) -> list[TextRegion]:
        """
        Detect text and return bounding boxes.
        """

        self._check()

        threshold = (
            self.confidence_threshold
            if min_confidence is None
            else float(min_confidence)
        )

        results: list[
            TextRegion
        ] = []

        with self._lock:

            if self._pytesseract is None:

                return results

            try:

                data = (
                    self._pytesseract.image_to_data(
                        image,
                        lang=self.language,
                        output_type=(
                            self._pytesseract
                            .Output
                            .DICT
                        ),
                    )
                )

            except Exception as exc:

                logger.warning(
                    "OCR bounding-box detection failed: %s",
                    exc,
                )

                return results

        count = len(
            data.get(
                "text",
                [],
            )
        )

        for index in range(count):

            text = str(
                data["text"][index]
            ).strip()

            if not text:
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

            if confidence < threshold:
                continue

            region = TextRegion(
                text=text,
                left=int(
                    data["left"][index]
                ),
                top=int(
                    data["top"][index]
                ),
                width=int(
                    data["width"][index]
                ),
                height=int(
                    data["height"][index]
                ),
                confidence=confidence,
            )

            results.append(region)

        self.last_results = results

        return results

    def detect_screen_text(
        self,
        min_confidence: Optional[
            float
        ] = None,
    ) -> list[TextRegion]:

        image = self.capture_screen()

        return self.detect_text(
            image,
            min_confidence=min_confidence,
        )

    def detect_region_text(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        min_confidence: Optional[
            float
        ] = None,
    ) -> list[TextRegion]:

        image = self.capture_region(
            x,
            y,
            width,
            height,
        )

        regions = self.detect_text(
            image,
            min_confidence=min_confidence,
        )

        # Convert region coordinates from
        # local-region coordinates to
        # screen coordinates.
        for region in regions:

            region.left += int(x)
            region.top += int(y)

        return regions

    # ==========================================================
    # FIND TEXT
    # ==========================================================

    def find_text(
        self,
        query: str,
        exact: bool = False,
        case_sensitive: bool = False,
        min_confidence: Optional[
            float
        ] = None,
    ) -> list[TextRegion]:

        query = str(query).strip()

        if not query:
            return []

        regions = self.detect_screen_text(
            min_confidence=min_confidence
        )

        if not case_sensitive:

            query_compare = (
                query.lower()
            )

        else:

            query_compare = query

        matches: list[
            TextRegion
        ] = []

        for region in regions:

            text = region.text

            if not case_sensitive:
                text_compare = text.lower()
            else:
                text_compare = text

            if exact:

                if text_compare == query_compare:
                    matches.append(region)

            else:

                if query_compare in text_compare:
                    matches.append(region)

        return matches

    # ==========================================================
    # FIND FIRST
    # ==========================================================

    def find_first(
        self,
        query: str,
        exact: bool = False,
    ) -> Optional[TextRegion]:

        matches = self.find_text(
            query,
            exact=exact,
        )

        if not matches:
            return None

        return matches[0]

    # ==========================================================
    # TEXT POSITION
    # ==========================================================

    def find_text_position(
        self,
        query: str,
        exact: bool = False,
    ) -> Optional[tuple[int, int]]:

        match = self.find_first(
            query,
            exact=exact,
        )

        if match is None:
            return None

        return match.center

    # ==========================================================
    # SCREEN SEARCH
    # ==========================================================

    def contains_text(
        self,
        query: str,
        exact: bool = False,
    ) -> bool:

        return (
            self.find_first(
                query,
                exact=exact,
            )
            is not None
        )

    # ==========================================================
    # READ LINE STRUCTURE
    # ==========================================================

    def get_lines(
        self,
        regions: Optional[
            list[TextRegion]
        ] = None,
    ) -> list[str]:

        if regions is None:
            regions = (
                self.detect_screen_text()
            )

        if not regions:
            return []

        sorted_regions = sorted(
            regions,
            key=lambda item: (
                item.top,
                item.left,
            ),
        )

        lines: list[
            list[TextRegion]
        ] = []

        tolerance = 12

        for region in sorted_regions:

            placed = False

            for line in lines:

                average_y = sum(
                    item.top
                    for item in line
                ) / len(line)

                if abs(
                    region.top
                    - average_y
                ) <= tolerance:

                    line.append(region)

                    placed = True
                    break

            if not placed:

                lines.append(
                    [region]
                )

        output = []

        for line in lines:

            line.sort(
                key=lambda item: item.left
            )

            output.append(
                " ".join(
                    item.text
                    for item in line
                )
            )

        return output

    # ==========================================================
    # CLEAN OCR TEXT
    # ==========================================================

    def clean_text(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    def read_clean_screen(self) -> str:

        return self.clean_text(
            self.read_screen()
        )

    # ==========================================================
    # CLICK TEXT
    # ==========================================================

    def click_text(
        self,
        query: str,
        exact: bool = False,
        clicks: int = 1,
    ) -> bool:
        """
        Locate text on screen and click its center.
        """

        match = self.find_first(
            query,
            exact=exact,
        )

        if match is None:
            return False

        if self._pyautogui is None:

            raise RuntimeError(
                "PyAutoGUI is required "
                "for click operations."
            )

        x, y = match.center

        self._pyautogui.click(
            x=x,
            y=y,
            clicks=int(clicks),
        )

        return True

    # ==========================================================
    # TYPE INTO TEXT FIELD
    # ==========================================================

    def click_and_type(
        self,
        query: str,
        text: str,
        exact: bool = False,
    ) -> bool:

        if not self.click_text(
            query,
            exact=exact,
        ):
            return False

        if self._pyautogui is None:

            raise RuntimeError(
                "PyAutoGUI is required "
                "for typing operations."
            )

        self._pyautogui.write(
            str(text),
            interval=0.01,
        )

        return True

    # ==========================================================
    # LAST RESULTS
    # ==========================================================

    def get_last_results(
        self,
    ) -> list[dict[str, Any]]:

        return [
            result.to_dict()
            for result in self.last_results
        ]

    def clear_results(self) -> None:

        self.last_results.clear()

    # ==========================================================
    # CONFIGURATION
    # ==========================================================

    def set_language(
        self,
        language: str,
    ) -> None:

        language = str(
            language
        ).strip()

        if not language:
            raise ValueError(
                "Language cannot be empty."
            )

        self.language = language

    def set_confidence_threshold(
        self,
        threshold: float,
    ) -> None:

        threshold = float(threshold)

        if not (
            0 <= threshold <= 100
        ):
            raise ValueError(
                "Confidence threshold "
                "must be between 0 and 100."
            )

        self.confidence_threshold = (
            threshold
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "ocr_backend": (
                "pytesseract"
                if self._pytesseract is not None
                else (
                    "easyocr"
                    if self._easyocr is not None
                    else None
                )
            ),
            "pyautogui_available": (
                self._pyautogui is not None
            ),
            "language": self.language,
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "last_result_count": len(
                self.last_results
            ),
        }

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        self.last_results.clear()

        self._pytesseract = None
        self._easyocr = None
        self._reader = None
        self._pyautogui = None

        logger.info(
            "RENIX screen reader shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_reader: Optional[
    ScreenReader
] = None


def get_screen_reader() -> ScreenReader:
    """Return the shared RENIX screen reader."""

    global _default_reader

    if _default_reader is None:

        _default_reader = (
            ScreenReader()
        )

    return _default_reader


