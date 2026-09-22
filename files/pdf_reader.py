"""
RENIX PDF Reader
================

Dedicated PDF reading and analysis service for RENIX.

Features:
- PDF metadata extraction
- Page-by-page text extraction
- Full-document text extraction
- Page range reading
- Keyword search
- Regular-expression search
- Context extraction
- Page statistics
- Table-aware text extraction where supported
- PDF structure inspection
- Safe file-size limits
- Optional PyMuPDF dependency

Dependency:
    pip install PyMuPDF
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class PDFReaderError(Exception):
    """Base exception for RENIX PDF reader errors."""


class PDFDependencyError(PDFReaderError):
    """Raised when PyMuPDF is unavailable."""


class PDFSecurityError(PDFReaderError):
    """Raised when a PDF violates configured safety limits."""


class PDFTooLargeError(PDFReaderError):
    """Raised when a PDF exceeds the configured size limit."""


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class PDFPage:
    page_number: int
    text: str
    width: float = 0.0
    height: float = 0.0
    rotation: int = 0
    character_count: int = 0
    word_count: int = 0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.character_count = len(self.text)
        self.word_count = len(
            re.findall(r"\S+", self.text)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }


@dataclass
class PDFMatch:
    page_number: int
    position: int
    matched_text: str
    context: str
    line_number: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "position": self.position,
            "matched_text": self.matched_text,
            "context": self.context,
            "line_number": self.line_number,
        }


@dataclass
class PDFMetadata:
    path: str
    filename: str
    size_bytes: int
    page_count: int
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    encrypted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "page_count": self.page_count,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "encrypted": self.encrypted,
        }


@dataclass
class PDFResult:
    success: bool
    text: str = ""
    pages: list[PDFPage] = field(
        default_factory=list
    )
    metadata: Optional[PDFMetadata] = None
    matches: list[PDFMatch] = field(
        default_factory=list
    )
    data: Any = None
    error: Optional[str] = None
    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "pages": [
                page.to_dict()
                for page in self.pages
            ],
            "metadata": (
                self.metadata.to_dict()
                if self.metadata
                else None
            ),
            "matches": [
                match.to_dict()
                for match in self.matches
            ],
            "data": self.data,
            "error": self.error,
            "warnings": self.warnings,
        }


# ============================================================
# PDF READER
# ============================================================


class PDFReader:
    """
    RENIX's dedicated PDF reading engine.

    PyMuPDF is imported lazily so RENIX can start even when
    PDF support has not been installed.
    """

    DEFAULT_MAX_FILE_SIZE = 250 * 1024 * 1024

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        preserve_whitespace: bool = False,
    ) -> None:

        self.enabled = enabled
        self.max_file_size = max_file_size
        self.preserve_whitespace = (
            preserve_whitespace
        )

        logger.info(
            "RENIX PDFReader initialized."
        )

    # ========================================================
    # STATE
    # ========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def _check_enabled(self) -> None:
        if not self.enabled:
            raise PDFReaderError(
                "PDFReader is disabled."
            )

    # ========================================================
    # DEPENDENCY
    # ========================================================

    @staticmethod
    def _load_fitz():
        try:
            import fitz

            return fitz

        except ImportError as exc:

            raise PDFDependencyError(
                "PyMuPDF is required for PDF support. "
                "Install it with: pip install PyMuPDF"
            ) from exc

    # ========================================================
    # PATH VALIDATION
    # ========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise PDFReaderError(
                "PDF path cannot be None."
            )

        return Path(path).expanduser()

    def validate_pdf(
        self,
        path: Path,
    ) -> None:

        if not path.exists():
            raise FileNotFoundError(
                f"PDF does not exist: {path}"
            )

        if not path.is_file():
            raise PDFReaderError(
                f"Path is not a file: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise PDFReaderError(
                f"Not a PDF file: {path}"
            )

        size = path.stat().st_size

        if size > self.max_file_size:

            raise PDFTooLargeError(
                f"PDF size ({size} bytes) exceeds "
                f"configured limit "
                f"({self.max_file_size} bytes)."
            )

    # ========================================================
    # OPEN
    # ========================================================

    def _open(
        self,
        path: Path,
    ):

        fitz = self._load_fitz()

        try:

            return fitz.open(
                str(path)
            )

        except Exception as exc:

            raise PDFReaderError(
                f"Unable to open PDF: {exc}"
            ) from exc

    # ========================================================
    # METADATA
    # ========================================================

    def get_metadata(
        self,
        path: str | os.PathLike[str],
    ) -> PDFMetadata:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        self.validate_pdf(pdf_path)

        document = self._open(pdf_path)

        try:

            metadata = document.metadata or {}

            return PDFMetadata(
                path=str(
                    pdf_path.resolve()
                ),
                filename=pdf_path.name,
                size_bytes=pdf_path.stat().st_size,
                page_count=document.page_count,
                title=metadata.get("title"),
                author=metadata.get("author"),
                subject=metadata.get("subject"),
                keywords=metadata.get("keywords"),
                creator=metadata.get("creator"),
                producer=metadata.get("producer"),
                creation_date=metadata.get(
                    "creationDate"
                ),
                modification_date=metadata.get(
                    "modDate"
                ),
                encrypted=bool(
                    document.is_encrypted
                ),
            )

        finally:

            document.close()

    # ========================================================
    # TEXT NORMALIZATION
    # ========================================================

    def normalize_text(
        self,
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        if self.preserve_whitespace:
            return text

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ========================================================
    # PAGE READER
    # ========================================================

    def read_page(
        self,
        path: str | os.PathLike[str],
        page_number: int,
    ) -> PDFPage:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        self.validate_pdf(pdf_path)

        if page_number < 1:
            raise ValueError(
                "page_number must be >= 1."
            )

        document = self._open(pdf_path)

        try:

            if page_number > document.page_count:

                raise IndexError(
                    f"PDF has only "
                    f"{document.page_count} pages."
                )

            page = document.load_page(
                page_number - 1
            )

            raw_text = page.get_text(
                "text"
            )

            text = self.normalize_text(
                raw_text
            )

            return PDFPage(
                page_number=page_number,
                text=text,
                width=float(page.rect.width),
                height=float(page.rect.height),
                rotation=int(page.rotation),
                metadata={
                    "has_images": bool(
                        page.get_images(
                            full=True
                        )
                    ),
                    "image_count": len(
                        page.get_images(
                            full=True
                        )
                    ),
                },
            )

        finally:

            document.close()

    # ========================================================
    # PAGE RANGE
    # ========================================================

    def read_pages(
        self,
        path: str | os.PathLike[str],
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> list[PDFPage]:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        self.validate_pdf(pdf_path)

        if start_page < 1:
            raise ValueError(
                "start_page must be >= 1."
            )

        document = self._open(pdf_path)

        try:

            total_pages = document.page_count

            if end_page is None:
                end_page = total_pages

            if end_page < start_page:
                raise ValueError(
                    "end_page must be >= start_page."
                )

            if end_page > total_pages:
                raise IndexError(
                    f"PDF has only {total_pages} pages."
                )

            pages = []

            for index in range(
                start_page - 1,
                end_page,
            ):

                page = document.load_page(
                    index
                )

                text = self.normalize_text(
                    page.get_text("text")
                )

                pages.append(
                    PDFPage(
                        page_number=index + 1,
                        text=text,
                        width=float(
                            page.rect.width
                        ),
                        height=float(
                            page.rect.height
                        ),
                        rotation=int(
                            page.rotation
                        ),
                        metadata={
                            "has_images": bool(
                                page.get_images(
                                    full=True
                                )
                            ),
                            "image_count": len(
                                page.get_images(
                                    full=True
                                )
                            ),
                        },
                    )
                )

            return pages

        finally:

            document.close()

    # ========================================================
    # FULL DOCUMENT
    # ========================================================

    def read(
        self,
        path: str | os.PathLike[str],
        *,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> PDFResult:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        try:

            metadata = self.get_metadata(
                pdf_path
            )

            pages = self.read_pages(
                pdf_path,
                start_page=start_page,
                end_page=end_page,
            )

            text = "\n\n".join(
                page.text
                for page in pages
            )

            return PDFResult(
                success=True,
                text=text,
                pages=pages,
                metadata=metadata,
            )

        except Exception as exc:

            logger.exception(
                "Failed to read PDF."
            )

            return PDFResult(
                success=False,
                error=str(exc),
            )

    # ========================================================
    # ITERATOR
    # ========================================================

    def iter_pages(
        self,
        path: str | os.PathLike[str],
    ) -> Iterator[PDFPage]:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        self.validate_pdf(pdf_path)

        document = self._open(pdf_path)

        try:

            for index in range(
                document.page_count
            ):

                page = document.load_page(
                    index
                )

                text = self.normalize_text(
                    page.get_text("text")
                )

                yield PDFPage(
                    page_number=index + 1,
                    text=text,
                    width=float(
                        page.rect.width
                    ),
                    height=float(
                        page.rect.height
                    ),
                    rotation=int(
                        page.rotation
                    ),
                )

        finally:

            document.close()

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        path: str | os.PathLike[str],
        query: str,
        *,
        case_sensitive: bool = False,
        context_chars: int = 120,
    ) -> list[PDFMatch]:

        self._check_enabled()

        if not query:
            return []

        if context_chars < 0:
            raise ValueError(
                "context_chars cannot be negative."
            )

        result = self.read(path)

        if not result.success:

            raise PDFReaderError(
                result.error
                or "Unable to read PDF."
            )

        matches: list[PDFMatch] = []

        search_query = (
            query
            if case_sensitive
            else query.lower()
        )

        for page in result.pages:

            searchable = (
                page.text
                if case_sensitive
                else page.text.lower()
            )

            position = 0

            while True:

                position = searchable.find(
                    search_query,
                    position,
                )

                if position == -1:
                    break

                start = max(
                    0,
                    position - context_chars,
                )

                end = min(
                    len(page.text),
                    position
                    + len(query)
                    + context_chars,
                )

                context = page.text[
                    start:end
                ]

                line_number = (
                    page.text[:position].count(
                        "\n"
                    )
                    + 1
                )

                matches.append(
                    PDFMatch(
                        page_number=page.page_number,
                        position=position,
                        matched_text=page.text[
                            position:
                            position + len(query)
                        ],
                        context=context,
                        line_number=line_number,
                    )
                )

                position += max(
                    1,
                    len(query),
                )

        return matches

    # ========================================================
    # REGEX SEARCH
    # ========================================================

    def regex_search(
        self,
        path: str | os.PathLike[str],
        pattern: str,
        *,
        flags: int = 0,
        context_chars: int = 120,
    ) -> list[PDFMatch]:

        self._check_enabled()

        if not pattern:
            return []

        try:
            compiled = re.compile(
                pattern,
                flags,
            )

        except re.error as exc:

            raise PDFReaderError(
                f"Invalid regular expression: {exc}"
            ) from exc

        result = self.read(path)

        if not result.success:

            raise PDFReaderError(
                result.error
                or "Unable to read PDF."
            )

        matches: list[PDFMatch] = []

        for page in result.pages:

            for match in compiled.finditer(
                page.text
            ):

                start = max(
                    0,
                    match.start()
                    - context_chars,
                )

                end = min(
                    len(page.text),
                    match.end()
                    + context_chars,
                )

                line_number = (
                    page.text[:match.start()]
                    .count("\n")
                    + 1
                )

                matches.append(
                    PDFMatch(
                        page_number=page.page_number,
                        position=match.start(),
                        matched_text=match.group(0),
                        context=page.text[
                            start:end
                        ],
                        line_number=line_number,
                    )
                )

        return matches

    # ========================================================
    # PAGE STATISTICS
    # ========================================================

    def page_statistics(
        self,
        path: str | os.PathLike[str],
    ) -> list[dict[str, Any]]:

        pages = self.read_pages(path)

        statistics = []

        for page in pages:

            statistics.append(
                {
                    "page": page.page_number,
                    "characters": page.character_count,
                    "words": page.word_count,
                    "lines": (
                        page.text.count("\n") + 1
                        if page.text
                        else 0
                    ),
                    "width": page.width,
                    "height": page.height,
                    "rotation": page.rotation,
                }
            )

        return statistics

    # ========================================================
    # IMAGES
    # ========================================================

    def get_image_count(
        self,
        path: str | os.PathLike[str],
    ) -> int:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        self.validate_pdf(pdf_path)

        document = self._open(pdf_path)

        try:

            total = 0

            for index in range(
                document.page_count
            ):

                page = document.load_page(
                    index
                )

                total += len(
                    page.get_images(
                        full=True
                    )
                )

            return total

        finally:

            document.close()

    # ========================================================
    # LINKS
    # ========================================================

    def get_links(
        self,
        path: str | os.PathLike[str],
    ) -> list[dict[str, Any]]:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        self.validate_pdf(pdf_path)

        document = self._open(pdf_path)

        links: list[dict[str, Any]] = []

        try:

            for index in range(
                document.page_count
            ):

                page = document.load_page(
                    index
                )

                for link in page.get_links():

                    item = dict(link)

                    item["page_number"] = (
                        index + 1
                    )

                    links.append(item)

            return links

        finally:

            document.close()

    # ========================================================
    # TABLE-STYLE EXTRACTION
    # ========================================================

    def extract_blocks(
        self,
        path: str | os.PathLike[str],
        page_number: int,
    ) -> list[dict[str, Any]]:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        self.validate_pdf(pdf_path)

        if page_number < 1:
            raise ValueError(
                "page_number must be >= 1."
            )

        document = self._open(pdf_path)

        try:

            if page_number > document.page_count:
                raise IndexError(
                    "Page number exceeds document length."
                )

            page = document.load_page(
                page_number - 1
            )

            blocks = page.get_text(
                "blocks"
            )

            results = []

            for block in blocks:

                if len(block) < 5:
                    continue

                results.append(
                    {
                        "x0": block[0],
                        "y0": block[1],
                        "x1": block[2],
                        "y1": block[3],
                        "text": block[4],
                        "block_number": (
                            block[5]
                            if len(block) > 5
                            else None
                        ),
                    }
                )

            return results

        finally:

            document.close()

    # ========================================================
    # OUTLINE / TOC
    # ========================================================

    def get_toc(
        self,
        path: str | os.PathLike[str],
    ) -> list[dict[str, Any]]:

        self._check_enabled()

        pdf_path = self.normalize_path(path)

        self.validate_pdf(pdf_path)

        document = self._open(pdf_path)

        try:

            toc = document.get_toc()

            results = []

            for item in toc:

                if len(item) >= 3:

                    results.append(
                        {
                            "level": item[0],
                            "title": item[1],
                            "page": item[2],
                        }
                    )

            return results

        finally:

            document.close()

    # ========================================================
    # PAGE PREVIEW
    # ========================================================

    def preview(
        self,
        path: str | os.PathLike[str],
        *,
        max_characters: int = 5000,
    ) -> PDFResult:

        if max_characters <= 0:
            raise ValueError(
                "max_characters must be positive."
            )

        result = self.read(path)

        if not result.success:
            return result

        if len(result.text) > max_characters:

            result.text = (
                result.text[:max_characters]
                + "\n\n"
                "[RENIX: PDF preview truncated]"
            )

        return result

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> dict[str, Any]:

        dependency_available = True

        try:
            self._load_fitz()

        except PDFDependencyError:
            dependency_available = False

        return {
            "enabled": self.enabled,
            "max_file_size": self.max_file_size,
            "preserve_whitespace": (
                self.preserve_whitespace
            ),
            "pymupdf_available": (
                dependency_available
            ),
            "capabilities": [
                "metadata",
                "full_text",
                "page_text",
                "page_ranges",
                "page_iterator",
                "keyword_search",
                "regex_search",
                "page_statistics",
                "image_count",
                "link_extraction",
                "text_blocks",
                "table_style_extraction",
                "table_of_contents",
                "preview",
            ],
        }

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX PDFReader shut down."
        )


# ============================================================
# DEFAULT INSTANCE
# ============================================================

_default_pdf_reader: Optional[
    PDFReader
] = None


def get_pdf_reader() -> PDFReader:

    global _default_pdf_reader

    if _default_pdf_reader is None:
        _default_pdf_reader = PDFReader()

    return _default_pdf_reader


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def read_pdf(
    path: str | os.PathLike[str],
) -> PDFResult:

    return get_pdf_reader().read(path)


def read_pdf_page(
    path: str | os.PathLike[str],
    page_number: int,
) -> PDFPage:

    return get_pdf_reader().read_page(
        path,
        page_number,
    )


def search_pdf(
    path: str | os.PathLike[str],
    query: str,
) -> list[PDFMatch]:

    return get_pdf_reader().search(
        path,
        query,
    )


def pdf_metadata(
    path: str | os.PathLike[str],
) -> PDFMetadata:

    return get_pdf_reader().get_metadata(
        path
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "PDFReaderError",
    "PDFDependencyError",
    "PDFSecurityError",
    "PDFTooLargeError",
    "PDFPage",
    "PDFMatch",
    "PDFMetadata",
    "PDFResult",
    "PDFReader",
    "get_pdf_reader",
    "read_pdf",
    "read_pdf_page",
    "search_pdf",
    "pdf_metadata",
]


