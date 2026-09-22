"""
RENIX File Classifier
=====================

Classifies files into logical categories for RENIX.

Categories:
    document
    image
    video
    audio
    code
    archive
    spreadsheet
    presentation
    database
    configuration
    executable
    font
    subtitle
    model
    text
    unknown

The classifier is intentionally dependency-light so it can run
before specialized readers are initialized.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# FILE CATEGORY DEFINITIONS
# ==============================================================

CATEGORY_DOCUMENT = "document"
CATEGORY_IMAGE = "image"
CATEGORY_VIDEO = "video"
CATEGORY_AUDIO = "audio"
CATEGORY_CODE = "code"
CATEGORY_ARCHIVE = "archive"
CATEGORY_SPREADSHEET = "spreadsheet"
CATEGORY_PRESENTATION = "presentation"
CATEGORY_DATABASE = "database"
CATEGORY_CONFIGURATION = "configuration"
CATEGORY_EXECUTABLE = "executable"
CATEGORY_FONT = "font"
CATEGORY_SUBTITLE = "subtitle"
CATEGORY_MODEL = "model"
CATEGORY_TEXT = "text"
CATEGORY_UNKNOWN = "unknown"


# ==============================================================
# EXTENSION MAPS
# ==============================================================

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
    ".pages",
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
    ".asm",
    ".v",
    ".vhd",
    ".vhdl",
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

SPREADSHEET_EXTENSIONS = {
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".ods",
    ".csv",
    ".tsv",
}

PRESENTATION_EXTENSIONS = {
    ".ppt",
    ".pptx",
    ".pptm",
    ".odp",
    ".key",
}

DATABASE_EXTENSIONS = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".mdb",
    ".accdb",
    ".db3",
}

CONFIGURATION_EXTENSIONS = {
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

EXECUTABLE_EXTENSIONS = {
    ".exe",
    ".msi",
    ".com",
    ".scr",
    ".dll",
    ".sys",
    ".bin",
    ".app",
    ".deb",
    ".rpm",
    ".apk",
}

FONT_EXTENSIONS = {
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".eot",
}

SUBTITLE_EXTENSIONS = {
    ".srt",
    ".vtt",
    ".ass",
    ".ssa",
    ".sub",
}

MODEL_EXTENSIONS = {
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".gguf",
    ".ggml",
    ".tflite",
    ".pb",
    ".h5",
    ".keras",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".log",
    ".rst",
    ".tex",
}


EXTENSION_CATEGORY_MAP: dict[str, str] = {}

for _extension in DOCUMENT_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_DOCUMENT

for _extension in IMAGE_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_IMAGE

for _extension in VIDEO_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_VIDEO

for _extension in AUDIO_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_AUDIO

for _extension in CODE_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_CODE

for _extension in ARCHIVE_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_ARCHIVE

for _extension in SPREADSHEET_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_SPREADSHEET

for _extension in PRESENTATION_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_PRESENTATION

for _extension in DATABASE_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_DATABASE

for _extension in CONFIGURATION_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_CONFIGURATION

for _extension in EXECUTABLE_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_EXECUTABLE

for _extension in FONT_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_FONT

for _extension in SUBTITLE_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_SUBTITLE

for _extension in MODEL_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_MODEL

for _extension in TEXT_EXTENSIONS:
    EXTENSION_CATEGORY_MAP[_extension] = CATEGORY_TEXT


# ==============================================================
# CLASSIFICATION RESULT
# ==============================================================


@dataclass
class FileClassification:
    """Classification information for a filesystem entry."""

    path: str
    name: str
    extension: str
    category: str
    mime_type: Optional[str]
    is_file: bool
    is_directory: bool
    size: int
    confidence: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================
# FILE CLASSIFIER
# ==============================================================


class FileClassifier:
    """
    RENIX file classification engine.

    Classification priority:

        1. filesystem type
        2. known extension
        3. MIME type
        4. filename hints
        5. unknown
    """

    def __init__(
        self,
        enabled: bool = True,
    ) -> None:

        self.enabled = enabled

        logger.info(
            "RENIX FileClassifier initialized."
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
                "RENIX FileClassifier is disabled."
            )

    # ==========================================================
    # MIME
    # ==========================================================

    @staticmethod
    def detect_mime_type(
        path: Path,
    ) -> Optional[str]:

        mime_type, _ = mimetypes.guess_type(
            str(path)
        )

        return mime_type

    # ==========================================================
    # EXTENSION
    # ==========================================================

    @staticmethod
    def get_extension(
        path: Path,
    ) -> str:

        return path.suffix.lower()

    # ==========================================================
    # CATEGORY BY EXTENSION
    # ==========================================================

    def category_from_extension(
        self,
        extension: str,
    ) -> Optional[str]:

        extension = extension.lower()

        if extension and not extension.startswith("."):
            extension = "." + extension

        return EXTENSION_CATEGORY_MAP.get(
            extension
        )

    # ==========================================================
    # CATEGORY BY MIME
    # ==========================================================

    @staticmethod
    def category_from_mime(
        mime_type: Optional[str],
    ) -> Optional[str]:

        if not mime_type:
            return None

        mime_type = mime_type.lower()

        if mime_type.startswith("image/"):
            return CATEGORY_IMAGE

        if mime_type.startswith("video/"):
            return CATEGORY_VIDEO

        if mime_type.startswith("audio/"):
            return CATEGORY_AUDIO

        if mime_type.startswith("text/"):
            return CATEGORY_TEXT

        if mime_type.startswith(
            "application/pdf"
        ):
            return CATEGORY_DOCUMENT

        if mime_type in {
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }:
            return CATEGORY_DOCUMENT

        if mime_type in {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }:
            return CATEGORY_SPREADSHEET

        if mime_type in {
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }:
            return CATEGORY_PRESENTATION

        if mime_type in {
            "application/zip",
            "application/x-rar-compressed",
            "application/x-7z-compressed",
            "application/gzip",
            "application/x-tar",
        }:
            return CATEGORY_ARCHIVE

        if mime_type in {
            "application/x-sql",
            "application/vnd.sqlite3",
        }:
            return CATEGORY_DATABASE

        return None

    # ==========================================================
    # FILENAME HINTS
    # ==========================================================

    @staticmethod
    def category_from_filename(
        filename: str,
    ) -> Optional[str]:

        name = filename.lower()

        executable_names = {
            "makefile",
            "dockerfile",
            "cmakelists.txt",
        }

        if name in executable_names:
            return CATEGORY_CODE

        if (
            name.endswith(".env")
            or name.startswith(".env.")
        ):
            return CATEGORY_CONFIGURATION

        return None

    # ==========================================================
    # DESCRIPTION
    # ==========================================================

    @staticmethod
    def category_description(
        category: str,
    ) -> str:

        descriptions = {
            CATEGORY_DOCUMENT:
                "Document file",

            CATEGORY_IMAGE:
                "Image or graphic file",

            CATEGORY_VIDEO:
                "Video file",

            CATEGORY_AUDIO:
                "Audio or music file",

            CATEGORY_CODE:
                "Source-code or script file",

            CATEGORY_ARCHIVE:
                "Compressed or archive file",

            CATEGORY_SPREADSHEET:
                "Spreadsheet or tabular data file",

            CATEGORY_PRESENTATION:
                "Presentation file",

            CATEGORY_DATABASE:
                "Database file",

            CATEGORY_CONFIGURATION:
                "Configuration or structured data file",

            CATEGORY_EXECUTABLE:
                "Executable or system binary",

            CATEGORY_FONT:
                "Font file",

            CATEGORY_SUBTITLE:
                "Subtitle or caption file",

            CATEGORY_MODEL:
                "Machine-learning model file",

            CATEGORY_TEXT:
                "Plain-text or markup file",

            CATEGORY_UNKNOWN:
                "Unknown file type",
        }

        return descriptions.get(
            category,
            "Unknown file type",
        )

    # ==========================================================
    # CLASSIFY
    # ==========================================================

    def classify(
        self,
        path: str | os.PathLike[str],
    ) -> FileClassification:

        self._check()

        file_path = Path(
            path
        ).expanduser()

        if not file_path.exists():

            raise FileNotFoundError(
                str(file_path)
            )

        is_directory = file_path.is_dir()
        is_file = file_path.is_file()

        if is_directory:

            return FileClassification(
                path=str(
                    file_path.resolve()
                ),
                name=file_path.name,
                extension="",
                category="directory",
                mime_type=None,
                is_file=False,
                is_directory=True,
                size=0,
                confidence=100.0,
                description="Directory",
            )

        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0

        extension = self.get_extension(
            file_path
        )

        mime_type = self.detect_mime_type(
            file_path
        )

        # ------------------------------------------------------
        # Extension classification
        # ------------------------------------------------------

        category = (
            self.category_from_extension(
                extension
            )
        )

        if category:

            confidence = 98.0

        else:

            # --------------------------------------------------
            # MIME classification
            # --------------------------------------------------

            category = (
                self.category_from_mime(
                    mime_type
                )
            )

            if category:

                confidence = 85.0

            else:

                # ----------------------------------------------
                # Filename classification
                # ----------------------------------------------

                category = (
                    self.category_from_filename(
                        file_path.name
                    )
                )

                if category:
                    confidence = 75.0
                else:
                    category = CATEGORY_UNKNOWN
                    confidence = 20.0

        return FileClassification(
            path=str(
                file_path.resolve()
            ),
            name=file_path.name,
            extension=extension,
            category=category,
            mime_type=mime_type,
            is_file=is_file,
            is_directory=is_directory,
            size=size,
            confidence=confidence,
            description=self.category_description(
                category
            ),
        )

    # ==========================================================
    # SAFE CLASSIFICATION
    # ==========================================================

    def classify_safe(
        self,
        path: str | os.PathLike[str],
    ) -> Optional[dict[str, Any]]:

        try:

            return self.classify(
                path
            ).to_dict()

        except (
            OSError,
            PermissionError,
            RuntimeError,
        ) as exc:

            logger.warning(
                "Unable to classify %s: %s",
                path,
                exc,
            )

            return None

    # ==========================================================
    # BATCH CLASSIFICATION
    # ==========================================================

    def classify_many(
        self,
        paths: list[
            str | os.PathLike[str]
        ],
    ) -> list[dict[str, Any]]:

        self._check()

        results: list[
            dict[str, Any]
        ] = []

        for path in paths:

            result = self.classify_safe(
                path
            )

            if result:
                results.append(result)

        return results

    # ==========================================================
    # DIRECTORY CLASSIFICATION
    # ==========================================================

    def classify_directory(
        self,
        root: str | os.PathLike[str],
        *,
        recursive: bool = True,
        include_directories: bool = False,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:

        self._check()

        root_path = Path(
            root
        ).expanduser()

        if not root_path.exists():
            raise FileNotFoundError(
                str(root_path)
            )

        if not root_path.is_dir():
            raise NotADirectoryError(
                str(root_path)
            )

        iterator = (
            root_path.rglob("*")
            if recursive
            else root_path.iterdir()
        )

        results: list[
            dict[str, Any]
        ] = []

        for path in iterator:

            if (
                path.is_dir()
                and not include_directories
            ):
                continue

            result = self.classify_safe(
                path
            )

            if result:
                results.append(result)

            if (
                limit is not None
                and len(results) >= limit
            ):
                break

        return results

    # ==========================================================
    # FILTER BY CATEGORY
    # ==========================================================

    def filter_category(
        self,
        paths: list[
            str | os.PathLike[str]
        ],
        category: str,
    ) -> list[dict[str, Any]]:

        self._check()

        category = category.lower()

        results = []

        for path in paths:

            result = self.classify_safe(
                path
            )

            if not result:
                continue

            if (
                result["category"]
                == category
            ):
                results.append(result)

        return results

    # ==========================================================
    # DIRECTORY SUMMARY
    # ==========================================================

    def directory_summary(
        self,
        root: str | os.PathLike[str],
        *,
        recursive: bool = True,
    ) -> dict[str, Any]:

        results = self.classify_directory(
            root,
            recursive=recursive,
        )

        counts: dict[str, int] = {}
        total_size = 0

        for result in results:

            category = result[
                "category"
            ]

            counts[category] = (
                counts.get(category, 0)
                + 1
            )

            total_size += int(
                result.get(
                    "size",
                    0,
                )
            )

        return {
            "root": str(
                Path(root).expanduser()
            ),
            "total_files": len(results),
            "total_size": total_size,
            "categories": counts,
        }

    # ==========================================================
    # CATEGORY HELPERS
    # ==========================================================

    def is_document(
        self,
        path: str | Path,
    ) -> bool:

        return (
            self.classify_safe(path)
            or {}
        ).get("category") == CATEGORY_DOCUMENT

    def is_image(
        self,
        path: str | Path,
    ) -> bool:

        return (
            self.classify_safe(path)
            or {}
        ).get("category") == CATEGORY_IMAGE

    def is_video(
        self,
        path: str | Path,
    ) -> bool:

        return (
            self.classify_safe(path)
            or {}
        ).get("category") == CATEGORY_VIDEO

    def is_audio(
        self,
        path: str | Path,
    ) -> bool:

        return (
            self.classify_safe(path)
            or {}
        ).get("category") == CATEGORY_AUDIO

    def is_code(
        self,
        path: str | Path,
    ) -> bool:

        return (
            self.classify_safe(path)
            or {}
        ).get("category") == CATEGORY_CODE

    def is_archive(
        self,
        path: str | Path,
    ) -> bool:

        return (
            self.classify_safe(path)
            or {}
        ).get("category") == CATEGORY_ARCHIVE

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "known_extensions": len(
                EXTENSION_CATEGORY_MAP
            ),
            "categories": [
                CATEGORY_DOCUMENT,
                CATEGORY_IMAGE,
                CATEGORY_VIDEO,
                CATEGORY_AUDIO,
                CATEGORY_CODE,
                CATEGORY_ARCHIVE,
                CATEGORY_SPREADSHEET,
                CATEGORY_PRESENTATION,
                CATEGORY_DATABASE,
                CATEGORY_CONFIGURATION,
                CATEGORY_EXECUTABLE,
                CATEGORY_FONT,
                CATEGORY_SUBTITLE,
                CATEGORY_MODEL,
                CATEGORY_TEXT,
                CATEGORY_UNKNOWN,
            ],
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX FileClassifier shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_classifier: Optional[
    FileClassifier
] = None


def get_file_classifier() -> FileClassifier:

    global _default_classifier

    if _default_classifier is None:

        _default_classifier = (
            FileClassifier()
        )

    return _default_classifier


# ==============================================================
# MODULE-LEVEL HELPER
# ==============================================================


def classify_file(
    path: str | os.PathLike[str],
) -> dict[str, Any]:

    return (
        get_file_classifier()
        .classify(path)
        .to_dict()
    )


# ==============================================================
# PUBLIC API
# ==============================================================


__all__ = [
    "FileClassification",
    "FileClassifier",
    "classify_file",
    "get_file_classifier",
    "CATEGORY_DOCUMENT",
    "CATEGORY_IMAGE",
    "CATEGORY_VIDEO",
    "CATEGORY_AUDIO",
    "CATEGORY_CODE",
    "CATEGORY_ARCHIVE",
    "CATEGORY_SPREADSHEET",
    "CATEGORY_PRESENTATION",
    "CATEGORY_DATABASE",
    "CATEGORY_CONFIGURATION",
    "CATEGORY_EXECUTABLE",
    "CATEGORY_FONT",
    "CATEGORY_SUBTITLE",
    "CATEGORY_MODEL",
    "CATEGORY_TEXT",
    "CATEGORY_UNKNOWN",
]


