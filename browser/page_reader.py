"""
RENIX Page Reader
=================

Extracts readable information from browser pages.

Responsibilities:
    - Read visible page text
    - Extract headings
    - Extract links
    - Extract images
    - Extract metadata
    - Return structured page content
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class PageReader:
    """Read and extract structured information from browser pages."""

    def __init__(
        self,
        browser_manager: Any = None,
    ) -> None:
        self.browser_manager = browser_manager

    # ========================================================
    # PAGE
    # ========================================================

    def get_page(
        self,
        page_id: str | None = None,
    ) -> Any:
        if self.browser_manager is None:
            raise RuntimeError(
                "BrowserManager is not configured."
            )

        return self.browser_manager.get_page(
            page_id
        )

    # ========================================================
    # TEXT
    # ========================================================

    def read_text(
        self,
        *,
        page_id: str | None = None,
        clean: bool = True,
    ) -> str:
        """Return the visible text of the current page."""

        page = self.get_page(
            page_id
        )

        text = self._extract_text(
            page
        )

        if clean:
            return self.clean_text(
                text
            )

        return text

    def _extract_text(
        self,
        page: Any,
    ) -> str:

        # Preferred browser automation API.
        locator = getattr(
            page,
            "locator",
            None,
        )

        if callable(locator):

            try:
                body = locator(
                    "body"
                )

                inner_text = getattr(
                    body,
                    "inner_text",
                    None,
                )

                if callable(inner_text):
                    return str(
                        inner_text()
                    )

            except Exception:
                logger.debug(
                    "Locator text extraction failed.",
                    exc_info=True,
                )

        # JavaScript fallback.
        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if callable(evaluator):

            try:
                result = evaluator(
                    """
                    () => {
                        if (!document.body) {
                            return "";
                        }
                        return document.body.innerText || "";
                    }
                    """
                )

                return str(
                    result or ""
                )

            except Exception:
                logger.debug(
                    "JavaScript text extraction failed.",
                    exc_info=True,
                )

        # Generic page text fallback.
        text = getattr(
            page,
            "text",
            None,
        )

        if text is not None:
            return str(
                text
            )

        raise RuntimeError(
            "Unable to extract page text."
        )

    # ========================================================
    # CLEANING
    # ========================================================

    @staticmethod
    def clean_text(
        text: str,
    ) -> str:
        """Normalize unnecessary whitespace."""

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

        lines = []

        for line in text.split(
            "\n"
        ):

            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            ).strip()

            if line:
                lines.append(
                    line
                )

        return "\n".join(
            lines
        )

    # ========================================================
    # HEADINGS
    # ========================================================

    def get_headings(
        self,
        *,
        page_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract h1-h6 headings."""

        page = self.get_page(
            page_id
        )

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(evaluator):
            raise RuntimeError(
                "Browser page does not support evaluation."
            )

        result = evaluator(
            """
            () => Array.from(
                document.querySelectorAll(
                    "h1, h2, h3, h4, h5, h6"
                )
            ).map((element) => ({
                level: Number(element.tagName.substring(1)),
                text: (element.innerText || "").trim()
            })).filter(item => item.text)
            """
        )

        return result or []

    # ========================================================
    # LINKS
    # ========================================================

    def get_links(
        self,
        *,
        page_id: str | None = None,
        include_empty: bool = False,
    ) -> list[dict[str, Any]]:
        """Extract links from the current page."""

        page = self.get_page(
            page_id
        )

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(evaluator):
            raise RuntimeError(
                "Browser page does not support evaluation."
            )

        result = evaluator(
            """
            () => Array.from(
                document.querySelectorAll("a")
            ).map((element) => ({
                text: (element.innerText || "").trim(),
                href: element.href || "",
                title: element.title || ""
            }))
            """
        ) or []

        if include_empty:
            return result

        return [
            link
            for link in result
            if link.get("href")
        ]

    # ========================================================
    # IMAGES
    # ========================================================

    def get_images(
        self,
        *,
        page_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract image information."""

        page = self.get_page(
            page_id
        )

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(evaluator):
            raise RuntimeError(
                "Browser page does not support evaluation."
            )

        result = evaluator(
            """
            () => Array.from(
                document.querySelectorAll("img")
            ).map((element) => ({
                src: element.src || "",
                alt: element.alt || "",
                title: element.title || "",
                width: element.naturalWidth || 0,
                height: element.naturalHeight || 0
            }))
            """
        ) or []

        return result

    # ========================================================
    # METADATA
    # ========================================================

    def get_metadata(
        self,
        *,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        """Extract common HTML metadata."""

        page = self.get_page(
            page_id
        )

        evaluator = getattr(
            page,
            "evaluate",
            None,
        )

        if not callable(evaluator):
            raise RuntimeError(
                "Browser page does not support evaluation."
            )

        result = evaluator(
            """
            () => {
                const getMeta = (selector) => {
                    const element =
                        document.querySelector(selector);

                    return element
                        ? element.getAttribute("content") || ""
                        : "";
                };

                return {
                    title: document.title || "",
                    description: getMeta(
                        'meta[name="description"]'
                    ),
                    keywords: getMeta(
                        'meta[name="keywords"]'
                    ),
                    author: getMeta(
                        'meta[name="author"]'
                    ),
                    canonical: (
                        document.querySelector(
                            'link[rel="canonical"]'
                        )?.href || ""
                    ),
                    language:
                        document.documentElement.lang || ""
                };
            }
            """
        )

        return result or {}

    # ========================================================
    # PAGE STRUCTURE
    # ========================================================

    def get_structure(
        self,
        *,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a structured representation of a page."""

        return {
            "metadata": self.get_metadata(
                page_id=page_id
            ),
            "headings": self.get_headings(
                page_id=page_id
            ),
            "links": self.get_links(
                page_id=page_id
            ),
            "images": self.get_images(
                page_id=page_id
            ),
            "text": self.read_text(
                page_id=page_id
            ),
        }

    # ========================================================
    # SEARCH WITHIN PAGE
    # ========================================================

    def find_text(
        self,
        query: str,
        *,
        page_id: str | None = None,
        case_sensitive: bool = False,
    ) -> list[str]:
        """Find lines containing a specific piece of text."""

        if not query.strip():
            return []

        text = self.read_text(
            page_id=page_id,
            clean=False,
        )

        if not case_sensitive:
            query_compare = query.lower()
        else:
            query_compare = query

        matches = []

        for line in text.splitlines():

            comparison = (
                line
                if case_sensitive
                else line.lower()
            )

            if query_compare in comparison:
                matches.append(
                    line.strip()
                )

        return [
            line
            for line in matches
            if line
        ]

    # ========================================================
    # WORD COUNT
    # ========================================================

    def word_count(
        self,
        *,
        page_id: str | None = None,
    ) -> int:
        """Count words in readable page text."""

        text = self.read_text(
            page_id=page_id
        )

        if not text:
            return 0

        return len(
            re.findall(
                r"\b[\w'-]+\b",
                text,
                flags=re.UNICODE,
            )
        )

    # ========================================================
    # SUMMARY INPUT
    # ========================================================

    def get_readable_content(
        self,
        *,
        page_id: str | None = None,
        max_characters: int | None = None,
    ) -> str:
        """
        Return cleaned content suitable for passing to
        RENIX's research/summarization systems.
        """

        text = self.read_text(
            page_id=page_id
        )

        if (
            max_characters is not None
            and max_characters > 0
        ):
            return text[
                :max_characters
            ]

        return text


__all__ = [
    "PageReader",
]


