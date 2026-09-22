"""
RENIX File Metadata
===================

Read-only metadata inspection for files and directories.

Provides:
- File size
- Creation/modification/access times
- Extension
- MIME type
- Permissions
- Hidden-file detection
- Read/write/execute access
- Hashes
- Basic media/image metadata when optional libraries exist
- Directory metadata
- JSON-serializable output

This module never modifies files.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import stat
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# CONSTANTS
# ==============================================================

DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024

SUPPORTED_HASH_ALGORITHMS = {
    "md5",
    "sha1",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
}


# ==============================================================
# DATA MODEL
# ==============================================================


@dataclass
class FileMetadata:
    """Complete metadata representation for a filesystem entry."""

    path: str
    name: str
    stem: str
    extension: str
    absolute_path: str

    is_file: bool
    is_directory: bool
    is_symlink: bool
    is_hidden: bool

    size: int
    size_human: str

    created_at: Optional[str]
    modified_at: Optional[str]
    accessed_at: Optional[str]

    mime_type: Optional[str]

    mode: Optional[str]
    permissions: Optional[str]

    readable: bool
    writable: bool
    executable: bool

    owner_id: Optional[int]
    group_id: Optional[int]

    inode: Optional[int]

    file_hash: Optional[str]

    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================
# METADATA ENGINE
# ==============================================================


class FileMetadataEngine:
    """
    RENIX read-only filesystem metadata engine.
    """

    def __init__(
        self,
        enabled: bool = True,
        hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
        hash_chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
    ) -> None:

        self.enabled = enabled
        self.hash_algorithm = hash_algorithm.lower()
        self.hash_chunk_size = hash_chunk_size

        if (
            self.hash_algorithm
            not in SUPPORTED_HASH_ALGORITHMS
        ):
            raise ValueError(
                f"Unsupported hash algorithm: "
                f"{hash_algorithm}"
            )

        logger.info(
            "RENIX FileMetadataEngine initialized."
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
                "RENIX FileMetadataEngine is disabled."
            )

    # ==========================================================
    # PATH HELPERS
    # ==========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        return Path(path).expanduser()

    @staticmethod
    def is_hidden(
        path: Path,
    ) -> bool:

        if path.name.startswith("."):
            return True

        try:
            return bool(
                getattr(
                    path.stat(),
                    "st_file_attributes",
                    0,
                )
                & getattr(
                    stat,
                    "FILE_ATTRIBUTE_HIDDEN",
                    0,
                )
            )

        except OSError:
            return False

    # ==========================================================
    # SIZE
    # ==========================================================

    @staticmethod
    def format_size(
        size: int,
    ) -> str:

        if size < 1024:
            return f"{size} B"

        units = [
            "KB",
            "MB",
            "GB",
            "TB",
            "PB",
        ]

        value = float(size)

        for unit in units:

            value /= 1024

            if value < 1024:
                return f"{value:.2f} {unit}"

        return f"{value:.2f} EB"

    # ==========================================================
    # TIME
    # ==========================================================

    @staticmethod
    def timestamp_to_iso(
        timestamp: float,
    ) -> str:

        return datetime.fromtimestamp(
            timestamp
        ).astimezone().isoformat()

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
    # PERMISSIONS
    # ==========================================================

    @staticmethod
    def permission_string(
        mode: int,
    ) -> str:

        return stat.filemode(mode)

    @staticmethod
    def mode_string(
        mode: int,
    ) -> str:

        return oct(
            stat.S_IMODE(mode)
        )

    # ==========================================================
    # HASHING
    # ==========================================================

    def calculate_hash(
        self,
        path: str | os.PathLike[str],
        algorithm: Optional[str] = None,
    ) -> str:

        self._check()

        file_path = self.normalize_path(
            path
        )

        if not file_path.exists():
            raise FileNotFoundError(
                str(file_path)
            )

        if not file_path.is_file():
            raise IsADirectoryError(
                str(file_path)
            )

        selected_algorithm = (
            algorithm or self.hash_algorithm
        ).lower()

        if (
            selected_algorithm
            not in SUPPORTED_HASH_ALGORITHMS
        ):
            raise ValueError(
                f"Unsupported hash algorithm: "
                f"{selected_algorithm}"
            )

        hasher = hashlib.new(
            selected_algorithm
        )

        with file_path.open(
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    self.hash_chunk_size
                )

                if not chunk:
                    break

                hasher.update(chunk)

        return hasher.hexdigest()

    # ==========================================================
    # ACCESS
    # ==========================================================

    @staticmethod
    def access_info(
        path: Path,
    ) -> dict[str, bool]:

        return {
            "readable": os.access(
                path,
                os.R_OK,
            ),
            "writable": os.access(
                path,
                os.W_OK,
            ),
            "executable": os.access(
                path,
                os.X_OK,
            ),
        }

    # ==========================================================
    # OPTIONAL IMAGE METADATA
    # ==========================================================

    @staticmethod
    def image_metadata(
        path: Path,
    ) -> dict[str, Any]:

        metadata: dict[str, Any] = {}

        try:

            from PIL import Image

            with Image.open(path) as image:

                metadata.update(
                    {
                        "image_width": image.width,
                        "image_height": image.height,
                        "image_mode": image.mode,
                        "image_format": image.format,
                    }
                )

                if image.info:
                    metadata[
                        "image_info_keys"
                    ] = list(
                        image.info.keys()
                    )[:50]

        except ImportError:

            metadata[
                "image_library"
            ] = "Pillow not installed"

        except Exception as exc:

            metadata[
                "image_metadata_error"
            ] = str(exc)

        return metadata

    # ==========================================================
    # OPTIONAL PDF METADATA
    # ==========================================================

    @staticmethod
    def pdf_metadata(
        path: Path,
    ) -> dict[str, Any]:

        metadata: dict[str, Any] = {}

        try:

            from pypdf import PdfReader

            reader = PdfReader(
                str(path)
            )

            metadata[
                "page_count"
            ] = len(reader.pages)

            if reader.metadata:

                pdf_meta = {}

                for key, value in (
                    reader.metadata.items()
                ):

                    clean_key = str(
                        key
                    ).lstrip("/")

                    pdf_meta[
                        clean_key
                    ] = str(value)

                metadata[
                    "pdf_metadata"
                ] = pdf_meta

        except ImportError:

            metadata[
                "pdf_library"
            ] = "pypdf not installed"

        except Exception as exc:

            metadata[
                "pdf_metadata_error"
            ] = str(exc)

        return metadata

    # ==========================================================
    # MEDIA METADATA
    # ==========================================================

    @staticmethod
    def media_metadata(
        path: Path,
    ) -> dict[str, Any]:

        metadata: dict[str, Any] = {}

        try:

            import mutagen

            audio = mutagen.File(
                str(path),
                easy=False,
            )

            if audio is not None:

                if getattr(
                    audio,
                    "info",
                    None,
                ):

                    info = audio.info

                    if hasattr(
                        info,
                        "length",
                    ):
                        metadata[
                            "duration_seconds"
                        ] = float(
                            info.length
                        )

                    if hasattr(
                        info,
                        "bitrate",
                    ):
                        metadata[
                            "bitrate"
                        ] = int(
                            info.bitrate
                        )

                    if hasattr(
                        info,
                        "sample_rate",
                    ):
                        metadata[
                            "sample_rate"
                        ] = int(
                            info.sample_rate
                        )

        except ImportError:

            metadata[
                "media_library"
            ] = "mutagen not installed"

        except Exception as exc:

            metadata[
                "media_metadata_error"
            ] = str(exc)

        return metadata

    # ==========================================================
    # DIRECTORY METADATA
    # ==========================================================

    def directory_metadata(
        self,
        path: Path,
    ) -> dict[str, Any]:

        file_count = 0
        directory_count = 0
        total_size = 0

        try:

            for item in path.iterdir():

                try:

                    if item.is_file():

                        file_count += 1

                        try:
                            total_size += (
                                item.stat().st_size
                            )
                        except OSError:
                            pass

                    elif item.is_dir():

                        directory_count += 1

                except OSError:
                    continue

        except OSError as exc:

            return {
                "directory_error": str(exc)
            }

        return {
            "file_count": file_count,
            "directory_count": directory_count,
            "direct_size": total_size,
        }

    # ==========================================================
    # MAIN METADATA
    # ==========================================================

    def get_metadata(
        self,
        path: str | os.PathLike[str],
        *,
        include_hash: bool = False,
        hash_algorithm: Optional[str] = None,
    ) -> FileMetadata:

        self._check()

        file_path = self.normalize_path(
            path
        )

        if not file_path.exists():
            raise FileNotFoundError(
                str(file_path)
            )

        try:
            file_stat = file_path.stat()
        except OSError as exc:
            raise OSError(
                f"Unable to read metadata: "
                f"{file_path}"
            ) from exc

        is_file = file_path.is_file()
        is_directory = file_path.is_dir()
        is_symlink = file_path.is_symlink()

        extension = (
            file_path.suffix.lower()
        )

        mime_type = (
            self.detect_mime_type(
                file_path
            )
        )

        access = self.access_info(
            file_path
        )

        file_hash: Optional[str] = None

        if include_hash and is_file:

            try:

                file_hash = (
                    self.calculate_hash(
                        file_path,
                        hash_algorithm,
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Unable to calculate hash "
                    "for %s: %s",
                    file_path,
                    exc,
                )

        extra: dict[str, Any] = {}

        if is_directory:

            extra.update(
                self.directory_metadata(
                    file_path
                )
            )

        elif extension in {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tif",
            ".tiff",
            ".svg",
            ".heic",
            ".heif",
        }:

            extra.update(
                self.image_metadata(
                    file_path
                )
            )

        elif extension == ".pdf":

            extra.update(
                self.pdf_metadata(
                    file_path
                )
            )

        elif extension in {
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            ".ogg",
            ".m4a",
            ".opus",
            ".wma",
        }:

            extra.update(
                self.media_metadata(
                    file_path
                )
            )

        return FileMetadata(
            path=str(file_path),
            name=file_path.name,
            stem=file_path.stem,
            extension=extension,
            absolute_path=str(
                file_path.resolve()
            ),
            is_file=is_file,
            is_directory=is_directory,
            is_symlink=is_symlink,
            is_hidden=self.is_hidden(
                file_path
            ),
            size=(
                file_stat.st_size
                if is_file
                else 0
            ),
            size_human=self.format_size(
                file_stat.st_size
                if is_file
                else 0
            ),
            created_at=self.timestamp_to_iso(
                file_stat.st_ctime
            ),
            modified_at=self.timestamp_to_iso(
                file_stat.st_mtime
            ),
            accessed_at=self.timestamp_to_iso(
                file_stat.st_atime
            ),
            mime_type=mime_type,
            mode=self.mode_string(
                file_stat.st_mode
            ),
            permissions=self.permission_string(
                file_stat.st_mode
            ),
            readable=access["readable"],
            writable=access["writable"],
            executable=access["executable"],
            owner_id=getattr(
                file_stat,
                "st_uid",
                None,
            ),
            group_id=getattr(
                file_stat,
                "st_gid",
                None,
            ),
            inode=getattr(
                file_stat,
                "st_ino",
                None,
            ),
            file_hash=file_hash,
            extra=extra,
        )

    # ==========================================================
    # SAFE METADATA
    # ==========================================================

    def get_metadata_safe(
        self,
        path: str | os.PathLike[str],
        *,
        include_hash: bool = False,
        hash_algorithm: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:

        try:

            return self.get_metadata(
                path,
                include_hash=include_hash,
                hash_algorithm=hash_algorithm,
            ).to_dict()

        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:

            logger.warning(
                "Unable to get metadata for "
                "%s: %s",
                path,
                exc,
            )

            return None

    # ==========================================================
    # BATCH METADATA
    # ==========================================================

    def get_metadata_many(
        self,
        paths: list[
            str | os.PathLike[str]
        ],
        *,
        include_hash: bool = False,
    ) -> list[dict[str, Any]]:

        self._check()

        results = []

        for path in paths:

            result = self.get_metadata_safe(
                path,
                include_hash=include_hash,
            )

            if result is not None:
                results.append(result)

        return results

    # ==========================================================
    # QUICK INFO
    # ==========================================================

    def quick_info(
        self,
        path: str | os.PathLike[str],
    ) -> dict[str, Any]:

        metadata = self.get_metadata(
            path
        )

        return {
            "name": metadata.name,
            "path": metadata.absolute_path,
            "type": (
                metadata.mime_type
                or metadata.extension
                or "unknown"
            ),
            "size": metadata.size_human,
            "modified": metadata.modified_at,
            "hidden": metadata.is_hidden,
            "readable": metadata.readable,
            "writable": metadata.writable,
            "executable": metadata.executable,
        }

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "hash_algorithm": self.hash_algorithm,
            "hash_chunk_size": self.hash_chunk_size,
            "supported_hash_algorithms": sorted(
                SUPPORTED_HASH_ALGORITHMS
            ),
            "optional_metadata": {
                "images": "Pillow",
                "pdf": "pypdf",
                "media": "mutagen",
            },
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX FileMetadataEngine shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_metadata_engine: Optional[
    FileMetadataEngine
] = None


def get_file_metadata_engine() -> FileMetadataEngine:

    global _default_metadata_engine

    if _default_metadata_engine is None:

        _default_metadata_engine = (
            FileMetadataEngine()
        )

    return _default_metadata_engine


# ==============================================================
# MODULE-LEVEL HELPERS
# ==============================================================


def get_file_metadata(
    path: str | os.PathLike[str],
    *,
    include_hash: bool = False,
) -> dict[str, Any]:

    return (
        get_file_metadata_engine()
        .get_metadata(
            path,
            include_hash=include_hash,
        )
        .to_dict()
    )


def get_file_hash(
    path: str | os.PathLike[str],
    algorithm: str = DEFAULT_HASH_ALGORITHM,
) -> str:

    return (
        get_file_metadata_engine()
        .calculate_hash(
            path,
            algorithm,
        )
    )


# ==============================================================
# PUBLIC API
# ==============================================================

__all__ = [
    "FileMetadata",
    "FileMetadataEngine",
    "get_file_metadata_engine",
    "get_file_metadata",
    "get_file_hash",
]


