"""
RENIX File Preview Engine
=========================

Generates lightweight previews for files without modifying them.

Supported preview types:
    - text
    - code
    - JSON/YAML/config
    - CSV/TSV
    - images
    - audio
    - video
    - PDF
    - documents
    - archives
    - unknown files

The preview engine is intentionally safe:
    - read-only
    - size-limited
    - no file execution
    - no archive extraction
    - graceful fallback on unsupported formats
"""

from __future__ import annotations

import csv
import json
import logging
import mimetypes
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# CONSTANTS
# ==============================================================

DEFAULT_MAX_PREVIEW_SIZE = 5 * 1024 * 1024
DEFAULT_MAX_TEXT_CHARS = 12_000
DEFAULT_MAX_LINES = 120
DEFAULT_MAX_CSV_ROWS = 20
DEFAULT_MAX_CSV_COLUMNS = 20

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".tex",
}

CODE_EXTENSIONS = {
    ".py",
    ".pyw",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".cxx",
    ".cs",
    ".go",
    ".rs",
    ".swift",
    ".kt",
    ".kts",
    ".dart",
    ".php",
    ".rb",
    ".r",
    ".lua",
    ".pl",
    ".pm",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".bat",
    ".cmd",
    ".ps1",
    ".sql",
}

CONFIG_EXTENSIONS = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".config",
    ".env",
    ".properties",
    ".xml",
}

CSV_EXTENSIONS = {
    ".csv",
    ".tsv",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".svg",
    ".ico",
    ".heic",
    ".heif",
    ".avif",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".oga",
    ".m4a",
    ".wma",
    ".opus",
    ".aiff",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".webm",
    ".flv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp",
}

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
}

ARCHIVE_EXTENSIONS = {
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".tgz",
    ".tbz2",
    ".iso",
    ".cab",
}


# ==============================================================
# DATA MODEL
# ==============================================================


@dataclass
class FilePreview:
    """Represents a RENIX file preview."""

    path: str
    name: str
    extension: str
    preview_type: str
    mime_type: Optional[str]
    size: int
    truncated: bool
    content: Any
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================
# FILE PREVIEW ENGINE
# ==============================================================


class FilePreviewEngine:
    """
    Read-only preview generator for RENIX.

    This class does not execute, modify, move, copy, or delete
    files.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_preview_size: int = DEFAULT_MAX_PREVIEW_SIZE,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        max_lines: int = DEFAULT_MAX_LINES,
    ) -> None:

        self.enabled = enabled
        self.max_preview_size = max_preview_size
        self.max_text_chars = max_text_chars
        self.max_lines = max_lines

        logger.info(
            "RENIX FilePreviewEngine initialized."
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

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX FilePreviewEngine is disabled."
            )

    # ==========================================================
    # BASIC INFORMATION
    # ==========================================================

    @staticmethod
    def detect_mime_type(
        path: Path,
    ) -> Optional[str]:

        mime_type, _ = mimetypes.guess_type(
            str(path)
        )

        return mime_type

    @staticmethod
    def get_extension(
        path: Path,
    ) -> str:

        return path.suffix.lower()

    # ==========================================================
    # FILE TYPE DETECTION
    # ==========================================================

    def detect_preview_type(
        self,
        path: Path,
    ) -> str:

        extension = self.get_extension(
            path
        )

        if extension in TEXT_EXTENSIONS:
            return "text"

        if extension in CODE_EXTENSIONS:
            return "code"

        if extension in CONFIG_EXTENSIONS:
            return "configuration"

        if extension in CSV_EXTENSIONS:
            return "table"

        if extension in IMAGE_EXTENSIONS:
            return "image"

        if extension in AUDIO_EXTENSIONS:
            return "audio"

        if extension in VIDEO_EXTENSIONS:
            return "video"

        if extension in DOCUMENT_EXTENSIONS:

            if extension == ".pdf":
                return "pdf"

            return "document"

        if extension in ARCHIVE_EXTENSIONS:
            return "archive"

        mime_type = self.detect_mime_type(
            path
        )

        if mime_type:

            if mime_type.startswith("image/"):
                return "image"

            if mime_type.startswith("audio/"):
                return "audio"

            if mime_type.startswith("video/"):
                return "video"

            if mime_type.startswith("text/"):
                return "text"

            if mime_type == "application/pdf":
                return "pdf"

        return "unknown"

    # ==========================================================
    # SIZE VALIDATION
    # ==========================================================

    def validate_file(
        self,
        path: Path,
    ) -> int:

        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

        if not path.is_file():
            raise IsADirectoryError(
                str(path)
            )

        try:
            size = path.stat().st_size
        except OSError as exc:
            raise OSError(
                f"Unable to inspect file: {path}"
            ) from exc

        if size > self.max_preview_size:

            raise ValueError(
                f"File is too large for preview: "
                f"{size} bytes > "
                f"{self.max_preview_size} bytes"
            )

        return size

    # ==========================================================
    # TEXT READING
    # ==========================================================

    @staticmethod
    def _read_text_with_fallback(
        path: Path,
        max_chars: int,
    ) -> tuple[str, bool]:

        encodings = (
            "utf-8",
            "utf-8-sig",
            "utf-16",
            "cp1252",
            "latin-1",
        )

        for encoding in encodings:

            try:

                with path.open(
                    "r",
                    encoding=encoding,
                    errors="strict",
                ) as file:

                    text = file.read(
                        max_chars + 1
                    )

                truncated = (
                    len(text)
                    > max_chars
                )

                return (
                    text[:max_chars],
                    truncated,
                )

            except UnicodeDecodeError:
                continue

        # Final safe fallback.
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            text = file.read(
                max_chars + 1
            )

        truncated = (
            len(text)
            > max_chars
        )

        return (
            text[:max_chars],
            truncated,
        )

    # ==========================================================
    # LINE LIMITING
    # ==========================================================

    def _limit_lines(
        self,
        text: str,
    ) -> tuple[str, bool]:

        lines = text.splitlines()

        if len(lines) <= self.max_lines:
            return text, False

        limited = lines[
            : self.max_lines
        ]

        return (
            "\n".join(limited),
            True,
        )

    # ==========================================================
    # TEXT PREVIEW
    # ==========================================================

    def preview_text(
        self,
        path: Path,
    ) -> tuple[Any, bool, dict[str, Any]]:

        text, truncated_chars = (
            self._read_text_with_fallback(
                path,
                self.max_text_chars,
            )
        )

        text, truncated_lines = (
            self._limit_lines(text)
        )

        metadata = {
            "line_count": len(
                text.splitlines()
            ),
            "character_count": len(text),
        }

        return (
            text,
            truncated_chars or truncated_lines,
            metadata,
        )

    # ==========================================================
    # JSON PREVIEW
    # ==========================================================

    def preview_json(
        self,
        path: Path,
    ) -> tuple[Any, bool, dict[str, Any]]:

        text, truncated = (
            self._read_text_with_fallback(
                path,
                self.max_text_chars,
            )
        )

        try:

            data = json.loads(text)

            if isinstance(data, dict):

                metadata = {
                    "data_type": "object",
                    "keys": list(data.keys())[
                        :50
                    ],
                }

            elif isinstance(data, list):

                metadata = {
                    "data_type": "array",
                    "items": len(data),
                }

            else:

                metadata = {
                    "data_type": type(data).__name__
                }

            return (
                data,
                truncated,
                metadata,
            )

        except json.JSONDecodeError:

            return (
                text,
                truncated,
                {
                    "data_type": "invalid_json",
                },
            )

    # ==========================================================
    # CONFIG PREVIEW
    # ==========================================================

    def preview_configuration(
        self,
        path: Path,
    ) -> tuple[Any, bool, dict[str, Any]]:

        extension = self.get_extension(
            path
        )

        if extension == ".json":

            return self.preview_json(
                path
            )

        text, truncated = (
            self._read_text_with_fallback(
                path,
                self.max_text_chars,
            )
        )

        text, line_truncated = (
            self._limit_lines(text)
        )

        return (
            text,
            truncated or line_truncated,
            {
                "format": extension.lstrip("."),
                "line_count": len(
                    text.splitlines()
                ),
            },
        )

    # ==========================================================
    # CSV / TSV PREVIEW
    # ==========================================================

    def preview_table(
        self,
        path: Path,
    ) -> tuple[Any, bool, dict[str, Any]]:

        extension = self.get_extension(
            path
        )

        delimiter = "\t" if extension == ".tsv" else ","

        rows: list[list[str]] = []
        truncated = False

        try:

            with path.open(
                "r",
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as file:

                reader = csv.reader(
                    file,
                    delimiter=delimiter,
                )

                for row_index, row in enumerate(
                    reader
                ):

                    if (
                        row_index
                        >= DEFAULT_MAX_CSV_ROWS
                    ):
                        truncated = True
                        break

                    rows.append(
                        row[
                            :DEFAULT_MAX_CSV_COLUMNS
                        ]
                    )

                    if len(row) > DEFAULT_MAX_CSV_COLUMNS:
                        truncated = True

        except OSError as exc:

            raise OSError(
                f"Unable to read table: {path}"
            ) from exc

        column_count = max(
            (
                len(row)
                for row in rows
            ),
            default=0,
        )

        return (
            rows,
            truncated,
            {
                "format": (
                    "tsv"
                    if delimiter == "\t"
                    else "csv"
                ),
                "rows": len(rows),
                "columns": column_count,
            },
        )

    # ==========================================================
    # IMAGE PREVIEW
    # ==========================================================

    def preview_image(
        self,
        path: Path,
    ) -> tuple[Any, bool, dict[str, Any]]:

        metadata: dict[str, Any] = {
            "extension": self.get_extension(
                path
            ),
            "mime_type": self.detect_mime_type(
                path
            ),
        }

        # Optional Pillow integration.
        try:

            from PIL import Image

            with Image.open(path) as image:

                metadata.update(
                    {
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                        "format": image.format,
                    }
                )

        except ImportError:

            metadata["image_library"] = (
                "Pillow not installed"
            )

        except Exception as exc:

            metadata["inspection_error"] = str(
                exc
            )

        # The UI can use the original path as the
        # image source. No image bytes are loaded here.
        return (
            None,
            False,
            metadata,
        )

    # ==========================================================
    # MEDIA PREVIEW
    # ==========================================================

    def preview_media(
        self,
        path: Path,
        media_type: str,
    ) -> tuple[Any, bool, dict[str, Any]]:

        return (
            None,
            False,
            {
                "media_type": media_type,
                "mime_type": self.detect_mime_type(
                    path
                ),
                "extension": self.get_extension(
                    path
                ),
                "source": str(
                    path.resolve()
                ),
            },
        )

    # ==========================================================
    # DOCUMENT PREVIEW
    # ==========================================================

    def preview_document(
        self,
        path: Path,
    ) -> tuple[Any, bool, dict[str, Any]]:

        extension = self.get_extension(
            path
        )

        metadata = {
            "format": extension.lstrip("."),
            "mime_type": self.detect_mime_type(
                path
            ),
        }

        # PDF text extraction is intentionally optional.
        if extension == ".pdf":

            try:

                from pypdf import PdfReader

                reader = PdfReader(
                    str(path)
                )

                pages = []

                for page in reader.pages[:10]:

                    try:
                        pages.append(
                            page.extract_text()
                            or ""
                        )
                    except Exception:
                        pages.append("")

                text = "\n\n".join(
                    pages
                )

                text = text[
                    : self.max_text_chars
                ]

                metadata[
                    "pages"
                ] = len(reader.pages)

                return (
                    text,
                    len(text)
                    >= self.max_text_chars,
                    metadata,
                )

            except ImportError:

                metadata[
                    "reader"
                ] = "pypdf not installed"

            except Exception as exc:

                metadata[
                    "extraction_error"
                ] = str(exc)

        return (
            None,
            False,
            metadata,
        )

    # ==========================================================
    # ARCHIVE PREVIEW
    # ==========================================================

    def preview_archive(
        self,
        path: Path,
    ) -> tuple[Any, bool, dict[str, Any]]:

        metadata = {
            "archive_type": self.get_extension(
                path
            ).lstrip("."),
            "mime_type": self.detect_mime_type(
                path
            ),
        }

        # Intentionally do not extract archives.
        try:

            import zipfile

            if zipfile.is_zipfile(path):

                with zipfile.ZipFile(
                    path,
                    "r",
                ) as archive:

                    names = archive.namelist()

                metadata[
                    "file_count"
                ] = len(names)

                return (
                    names[:100],
                    len(names) > 100,
                    metadata,
                )

        except Exception as exc:

            metadata[
                "inspection_error"
            ] = str(exc)

        return (
            None,
            False,
            metadata,
        )

    # ==========================================================
    # UNKNOWN FILE
    # ==========================================================

    def preview_unknown(
        self,
        path: Path,
    ) -> tuple[Any, bool, dict[str, Any]]:

        return (
            None,
            False,
            {
                "mime_type": self.detect_mime_type(
                    path
                ),
                "extension": self.get_extension(
                    path
                ),
                "message": (
                    "No specialized preview "
                    "handler is available."
                ),
            },
        )

    # ==========================================================
    # MAIN PREVIEW
    # ==========================================================

    def preview(
        self,
        path: str | os.PathLike[str],
    ) -> FilePreview:

        self._check()

        file_path = Path(
            path
        ).expanduser()

        size = self.validate_file(
            file_path
        )

        extension = self.get_extension(
            file_path
        )

        mime_type = self.detect_mime_type(
            file_path
        )

        preview_type = (
            self.detect_preview_type(
                file_path
            )
        )

        if preview_type in {
            "text",
            "code",
        }:

            content, truncated, metadata = (
                self.preview_text(
                    file_path
                )
            )

        elif preview_type == "configuration":

            content, truncated, metadata = (
                self.preview_configuration(
                    file_path
                )
            )

        elif preview_type == "table":

            content, truncated, metadata = (
                self.preview_table(
                    file_path
                )
            )

        elif preview_type == "image":

            content, truncated, metadata = (
                self.preview_image(
                    file_path
                )
            )

        elif preview_type in {
            "audio",
            "video",
        }:

            content, truncated, metadata = (
                self.preview_media(
                    file_path,
                    preview_type,
                )
            )

        elif preview_type in {
            "pdf",
            "document",
        }:

            content, truncated, metadata = (
                self.preview_document(
                    file_path
                )
            )

        elif preview_type == "archive":

            content, truncated, metadata = (
                self.preview_archive(
                    file_path
                )
            )

        else:

            content, truncated, metadata = (
                self.preview_unknown(
                    file_path
                )
            )

        return FilePreview(
            path=str(
                file_path.resolve()
            ),
            name=file_path.name,
            extension=extension,
            preview_type=preview_type,
            mime_type=mime_type,
            size=size,
            truncated=truncated,
            content=content,
            metadata=metadata,
        )

    # ==========================================================
    # SAFE PREVIEW
    # ==========================================================

    def preview_safe(
        self,
        path: str | os.PathLike[str],
    ) -> Optional[dict[str, Any]]:

        try:

            return self.preview(
                path
            ).to_dict()

        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:

            logger.warning(
                "Unable to preview %s: %s",
                path,
                exc,
            )

            return None

    # ==========================================================
    # QUICK PREVIEW
    # ==========================================================

    def quick_preview(
        self,
        path: str | os.PathLike[str],
    ) -> str:

        result = self.preview_safe(
            path
        )

        if not result:
            return "Preview unavailable."

        preview_type = result[
            "preview_type"
        ]

        content = result.get(
            "content"
        )

        if preview_type in {
            "text",
            "code",
            "configuration",
            "pdf",
            "document",
        }:

            if content is None:
                return (
                    f"{preview_type.title()} "
                    "preview unavailable."
                )

            return str(content)

        if preview_type == "table":

            if not content:
                return "Empty table."

            return "\n".join(
                " | ".join(
                    str(cell)
                    for cell in row
                )
                for row in content
            )

        if preview_type == "image":

            metadata = result.get(
                "metadata",
                {},
            )

            width = metadata.get(
                "width"
            )

            height = metadata.get(
                "height"
            )

            if width and height:

                return (
                    f"Image: {width} × "
                    f"{height}"
                )

            return "Image file."

        if preview_type in {
            "audio",
            "video",
        }:

            return (
                f"{preview_type.title()} file: "
                f"{result['name']}"
            )

        if preview_type == "archive":

            count = len(
                content or []
            )

            return (
                f"Archive containing "
                f"{count} listed items."
            )

        return (
            f"File: {result['name']}"
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "max_preview_size": (
                self.max_preview_size
            ),
            "max_text_chars": (
                self.max_text_chars
            ),
            "max_lines": self.max_lines,
            "supported_preview_types": [
                "text",
                "code",
                "configuration",
                "table",
                "image",
                "audio",
                "video",
                "pdf",
                "document",
                "archive",
                "unknown",
            ],
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX FilePreviewEngine shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_preview_engine: Optional[
    FilePreviewEngine
] = None


def get_file_preview_engine() -> FilePreviewEngine:

    global _default_preview_engine

    if _default_preview_engine is None:

        _default_preview_engine = (
            FilePreviewEngine()
        )

    return _default_preview_engine


# ==============================================================
# MODULE-LEVEL HELPERS
# ==============================================================


def preview_file(
    path: str | os.PathLike[str],
) -> dict[str, Any]:

    return (
        get_file_preview_engine()
        .preview(path)
        .to_dict()
    )


def quick_preview(
    path: str | os.PathLike[str],
) -> str:

    return (
        get_file_preview_engine()
        .quick_preview(path)
    )


# ==============================================================
# PUBLIC API
# ==============================================================

__all__ = [
    "FilePreview",
    "FilePreviewEngine",
    "get_file_preview_engine",
    "preview_file",
    "quick_preview",
]


