"""
RENIX Webpage Parser
====================

Parses HTML/webpage content into structured information.

Responsibilities:
    - Extract page title
    - Extract headings
    - Extract paragraphs
    - Extract links
    - Extract images
    - Extract metadata
    - Convert HTML into readable text
    - Build a structured page representation
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class WebpageParser:
    """Parse webpage HTML into structured data."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(
        self,
        *,
        base_url: str | None = None,
    ) -> None:

        self.base_url = (
            base_url.strip()
            if base_url
            else None
        )

    # ========================================================
    # MAIN PARSER
    # ========================================================

    def parse(
        self,
        content: str,
        *,
        url: str | None = None,
    ) -> dict[str, Any]:
        """
        Parse HTML into a structured dictionary.
        """

        if not isinstance(
            content,
            str,
        ):
            raise TypeError(
                "content must be a string."
            )

        effective_url = (
            url
            or self.base_url
        )

        return {
            "url": effective_url,
            "title": self.extract_title(
                content
            ),
            "metadata": self.extract_metadata(
                content
            ),
            "headings": self.extract_headings(
                content
            ),
            "paragraphs": self.extract_paragraphs(
                content
            ),
            "links": self.extract_links(
                content,
                base_url=effective_url,
            ),
            "images": self.extract_images(
                content,
                base_url=effective_url,
            ),
            "text": self.to_text(
                content
            ),
        }

    # ========================================================
    # TITLE
    # ========================================================

    def extract_title(
        self,
        content: str,
    ) -> str:

        match = re.search(
            r"<title\b[^>]*>(.*?)</title\s*>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not match:
            return ""

        return self.clean_text(
            self._strip_tags(
                match.group(1)
            )
        )

    # ========================================================
    # METADATA
    # ========================================================

    def extract_metadata(
        self,
        content: str,
    ) -> dict[str, Any]:

        metadata: dict[str, Any] = {}

        # Meta tags.
        meta_pattern = re.compile(
            r"<meta\b([^>]*)>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in meta_pattern.finditer(
            content
        ):

            attributes = self._parse_attributes(
                match.group(1)
            )

            name = (
                attributes.get("name")
                or attributes.get("property")
                or attributes.get("http-equiv")
            )

            value = attributes.get(
                "content"
            )

            if name and value:
                metadata[
                    name.lower()
                ] = html.unescape(
                    value.strip()
                )

        # Canonical URL.
        link_pattern = re.compile(
            r"<link\b([^>]*)>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in link_pattern.finditer(
            content
        ):

            attributes = self._parse_attributes(
                match.group(1)
            )

            rel = attributes.get(
                "rel",
                "",
            ).lower()

            if rel == "canonical":
                href = attributes.get(
                    "href"
                )

                if href:
                    metadata[
                        "canonical"
                    ] = self.resolve_url(
                        href
                    )

        # Language.
        html_match = re.search(
            r"<html\b([^>]*)>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if html_match:

            attributes = self._parse_attributes(
                html_match.group(1)
            )

            if attributes.get("lang"):
                metadata[
                    "language"
                ] = attributes[
                    "lang"
                ]

        return metadata

    # ========================================================
    # HEADINGS
    # ========================================================

    def extract_headings(
        self,
        content: str,
    ) -> list[dict[str, Any]]:

        headings = []

        pattern = re.compile(
            r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(
            content
        ):

            text = self.clean_text(
                self._strip_tags(
                    match.group(2)
                )
            )

            if not text:
                continue

            headings.append(
                {
                    "level": int(
                        match.group(1)
                    ),
                    "text": text,
                }
            )

        return headings

    # ========================================================
    # PARAGRAPHS
    # ========================================================

    def extract_paragraphs(
        self,
        content: str,
    ) -> list[str]:

        paragraphs = []

        pattern = re.compile(
            r"<p\b[^>]*>(.*?)</p\s*>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(
            content
        ):

            text = self.clean_text(
                self._strip_tags(
                    match.group(1)
                )
            )

            if text:
                paragraphs.append(
                    text
                )

        return paragraphs

    # ========================================================
    # LINKS
    # ========================================================

    def extract_links(
        self,
        content: str,
        *,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:

        links = []

        pattern = re.compile(
            r"<a\b([^>]*)>(.*?)</a\s*>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(
            content
        ):

            attributes = self._parse_attributes(
                match.group(1)
            )

            href = attributes.get(
                "href",
                "",
            ).strip()

            text = self.clean_text(
                self._strip_tags(
                    match.group(2)
                )
            )

            if href:
                resolved = self.resolve_url(
                    href,
                    base_url=base_url,
                )
            else:
                resolved = ""

            links.append(
                {
                    "text": text,
                    "url": resolved,
                    "href": href,
                    "title": attributes.get(
                        "title",
                        "",
                    ),
                    "rel": attributes.get(
                        "rel",
                        "",
                    ),
                }
            )

        return links

    # ========================================================
    # IMAGES
    # ========================================================

    def extract_images(
        self,
        content: str,
        *,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:

        images = []

        pattern = re.compile(
            r"<img\b([^>]*)>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(
            content
        ):

            attributes = self._parse_attributes(
                match.group(1)
            )

            src = (
                attributes.get(
                    "src"
                )
                or attributes.get(
                    "data-src"
                )
                or ""
            ).strip()

            if src:
                resolved = self.resolve_url(
                    src,
                    base_url=base_url,
                )
            else:
                resolved = ""

            images.append(
                {
                    "src": resolved,
                    "alt": attributes.get(
                        "alt",
                        "",
                    ),
                    "title": attributes.get(
                        "title",
                        "",
                    ),
                    "width": attributes.get(
                        "width"
                    ),
                    "height": attributes.get(
                        "height"
                    ),
                }
            )

        return images

    # ========================================================
    # TEXT EXTRACTION
    # ========================================================

    def to_text(
        self,
        content: str,
    ) -> str:
        """
        Convert HTML into readable text.

        Script/style/noscript/template content is removed.
        """

        if not content:
            return ""

        text = content

        # Remove non-readable sections.
        text = re.sub(
            r"<(script|style|noscript|template)\b[^>]*>.*?</\1\s*>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Convert block tags into line breaks.
        for tag in self.BLOCK_TAGS:

            text = re.sub(
                rf"</?{tag}\b[^>]*>",
                "\n",
                text,
                flags=re.IGNORECASE,
            )

        # Remove remaining HTML tags.
        text = self._strip_tags(
            text
        )

        # Decode HTML entities.
        text = html.unescape(
            text
        )

        return self.clean_text(
            text
        )

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    @staticmethod
    def clean_text(
        text: str,
    ) -> str:

        text = html.unescape(
            text or ""
        )

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # Normalize spaces while preserving newlines.
        text = re.sub(
            r"[ \t\f\v]+",
            " ",
            text,
        )

        # Remove spaces around line breaks.
        text = re.sub(
            r" *\n *",
            "\n",
            text,
        )

        # Collapse excessive blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ========================================================
    # URL RESOLUTION
    # ========================================================

    def resolve_url(
        self,
        url: str,
        *,
        base_url: str | None = None,
    ) -> str:

        url = (
            url or ""
        ).strip()

        if not url:
            return ""

        effective_base = (
            base_url
            or self.base_url
        )

        if effective_base:
            return urljoin(
                effective_base,
                url,
            )

        return url

    # ========================================================
    # URL HELPERS
    # ========================================================

    @staticmethod
    def is_absolute_url(
        url: str,
    ) -> bool:

        parsed = urlparse(
            url
        )

        return bool(
            parsed.scheme
            and parsed.netloc
        )

    @staticmethod
    def is_external_url(
        url: str,
        base_url: str,
    ) -> bool:

        try:
            target = urlparse(
                url
            )

            base = urlparse(
                base_url
            )

            return (
                target.netloc.lower()
                != base.netloc.lower()
            )

        except Exception:
            return False

    # ========================================================
    # ATTRIBUTE PARSER
    # ========================================================

    @staticmethod
    def _parse_attributes(
        source: str,
    ) -> dict[str, str]:

        attributes: dict[
            str,
            str,
        ] = {}

        pattern = re.compile(
            r"""
            ([^\s=/>]+)
            \s*=\s*
            (?:
                "([^"]*)"
                |
                '([^']*)'
                |
                ([^\s"'=<>`]+)
            )
            """,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        for match in pattern.finditer(
            source
        ):

            name = match.group(
                1
            ).lower()

            value = (
                match.group(2)
                if match.group(2) is not None
                else match.group(3)
                if match.group(3) is not None
                else match.group(4)
                or ""
            )

            attributes[
                name
            ] = html.unescape(
                value
            )

        return attributes

    # ========================================================
    # TAG STRIPPING
    # ========================================================

    @staticmethod
    def _strip_tags(
        text: str,
    ) -> str:

        return re.sub(
            r"<[^>]+>",
            " ",
            text,
            flags=re.DOTALL,
        )

    # ========================================================
    # CONTENT FILTERING
    # ========================================================

    def extract_main_content(
        self,
        content: str,
    ) -> str:
        """
        Attempt to extract the main readable content.

        Preference order:
            article -> main -> body
        """

        patterns = [
            r"<article\b[^>]*>(.*?)</article\s*>",
            r"<main\b[^>]*>(.*?)</main\s*>",
            r"<body\b[^>]*>(.*?)</body\s*>",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if match:

                text = self.to_text(
                    match.group(1)
                )

                if text:
                    return text

        return self.to_text(
            content
        )

    # ========================================================
    # WORD COUNT
    # ========================================================

    def word_count(
        self,
        content: str,
    ) -> int:

        text = self.to_text(
            content
        )

        return len(
            re.findall(
                r"\b[\w'-]+\b",
                text,
                flags=re.UNICODE,
            )
        )

    # ========================================================
    # LINK FILTERING
    # ========================================================

    def get_internal_links(
        self,
        content: str,
        *,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:

        effective_base = (
            base_url
            or self.base_url
        )

        links = self.extract_links(
            content,
            base_url=effective_base,
        )

        if not effective_base:
            return links

        return [
            link
            for link in links
            if not self.is_external_url(
                link["url"],
                effective_base,
            )
        ]

    def get_external_links(
        self,
        content: str,
        *,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:

        effective_base = (
            base_url
            or self.base_url
        )

        if not effective_base:
            return []

        links = self.extract_links(
            content,
            base_url=effective_base,
        )

        return [
            link
            for link in links
            if self.is_external_url(
                link["url"],
                effective_base,
            )
        ]


__all__ = [
    "WebpageParser",
]


