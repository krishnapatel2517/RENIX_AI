"""
RENIX Copy Manager
==================

High-level file and folder copying service.

Responsibilities:
- Copy individual files
- Copy directories
- Batch copy
- Recursive directory copying
- Progress callbacks
- Collision handling
- Dry-run support
- Verification
- Copy statistics
- Operation history

This module performs COPY operations only.
It does not delete or move the source.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .file_operations import (
    FileOperations,
    OperationResult,
    get_file_operations,
)

logger = logging.getLogger(__name__)


# ==============================================================
# CONSTANTS
# ==============================================================

DEFAULT_BUFFER_SIZE = 1024 * 1024
DEFAULT_HASH_ALGORITHM = "sha256"

ProgressCallback = Callable[
    [int, int, str],
    None,
]


# ==============================================================
# EXCEPTIONS
# ==============================================================


class CopyManagerError(Exception):
    """Base exception for copy operations."""


class CopySourceNotFoundError(CopyManagerError):
    """Raised when the source does not exist."""


class CopyDestinationExistsError(CopyManagerError):
    """Raised when the destination already exists."""


class CopyVerificationError(CopyManagerError):
    """Raised when copied content fails verification."""


# ==============================================================
# DATA MODELS
# ==============================================================


@dataclass
class CopyStats:
    """Statistics for a copy operation."""

    files_copied: int = 0
    directories_copied: int = 0
    bytes_copied: int = 0
    elapsed_seconds: float = 0.0
    failed: int = 0

    @property
    def average_speed_bytes_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0

        return (
            self.bytes_copied
            / self.elapsed_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_copied": self.files_copied,
            "directories_copied": (
                self.directories_copied
            ),
            "bytes_copied": self.bytes_copied,
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "failed": self.failed,
            "average_speed_bytes_per_second": (
                self.average_speed_bytes_per_second
            ),
        }


@dataclass
class CopyResult:
    """Detailed result of a copy operation."""

    success: bool
    source: Optional[str] = None
    destination: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    dry_run: bool = False
    verified: bool = False
    stats: CopyStats = field(
        default_factory=CopyStats
    )
    files: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "destination": self.destination,
            "message": self.message,
            "error": self.error,
            "dry_run": self.dry_run,
            "verified": self.verified,
            "stats": self.stats.to_dict(),
            "files": self.files,
        }


@dataclass
class CopyOperation:
    """Historical copy operation."""

    source: Optional[str]
    destination: Optional[str]
    success: bool
    timestamp: float
    bytes_copied: int
    files_copied: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "success": self.success,
            "timestamp": self.timestamp,
            "bytes_copied": self.bytes_copied,
            "files_copied": self.files_copied,
            "message": self.message,
        }


# ==============================================================
# COPY MANAGER
# ==============================================================


class CopyManager:
    """
    RENIX high-level copy engine.

    Source files are never modified by this class.
    """

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
        file_operations: Optional[
            FileOperations
        ] = None,
    ) -> None:

        self.enabled = enabled
        self.dry_run = dry_run
        self.buffer_size = buffer_size
        self.hash_algorithm = hash_algorithm

        self.file_operations = (
            file_operations
            or get_file_operations()
        )

        self._history: list[
            CopyOperation
        ] = []

        logger.info(
            "RENIX CopyManager initialized."
        )

    # ==========================================================
    # STATE
    # ==========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def set_dry_run(
        self,
        enabled: bool,
    ) -> None:
        self.dry_run = enabled

    def _check(self) -> None:
        if not self.enabled:
            raise CopyManagerError(
                "RENIX CopyManager is disabled."
            )

    # ==========================================================
    # PATH HELPERS
    # ==========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise CopyManagerError(
                "Path cannot be None."
            )

        return Path(path).expanduser()

    @staticmethod
    def absolute_path(
        path: str | os.PathLike[str],
    ) -> Path:

        return (
            CopyManager.normalize_path(
                path
            ).resolve()
        )

    # ==========================================================
    # HISTORY
    # ==========================================================

    def _record(
        self,
        source: Optional[Path],
        destination: Optional[Path],
        success: bool,
        stats: CopyStats,
        message: str,
    ) -> None:

        self._history.append(
            CopyOperation(
                source=(
                    str(source)
                    if source
                    else None
                ),
                destination=(
                    str(destination)
                    if destination
                    else None
                ),
                success=success,
                timestamp=time.time(),
                bytes_copied=(
                    stats.bytes_copied
                ),
                files_copied=(
                    stats.files_copied
                ),
                message=message,
            )
        )

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:

        history = self._history

        if limit is not None:
            history = history[-limit:]

        return [
            item.to_dict()
            for item in history
        ]

    def clear_history(self) -> None:
        self._history.clear()

    # ==========================================================
    # SIZE
    # ==========================================================

    @staticmethod
    def format_size(
        size: int,
    ) -> str:

        if size < 1024:
            return f"{size} B"

        value = float(size)

        for unit in (
            "KB",
            "MB",
            "GB",
            "TB",
            "PB",
        ):

            value /= 1024

            if value < 1024:
                return f"{value:.2f} {unit}"

        return f"{value:.2f} EB"

    # ==========================================================
    # HASHING
    # ==========================================================

    def calculate_hash(
        self,
        path: str | os.PathLike[str],
    ) -> str:

        self._check()

        file_path = self.normalize_path(
            path
        )

        if not file_path.is_file():
            raise FileNotFoundError(
                str(file_path)
            )

        hasher = hashlib.new(
            self.hash_algorithm
        )

        with file_path.open("rb") as file:

            while True:

                chunk = file.read(
                    self.buffer_size
                )

                if not chunk:
                    break

                hasher.update(chunk)

        return hasher.hexdigest()

    def verify_copy(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> bool:

        self._check()

        source_path = self.normalize_path(
            source
        )

        destination_path = (
            self.normalize_path(
                destination
            )
        )

        if not source_path.is_file():
            return False

        if not destination_path.is_file():
            return False

        try:

            if (
                source_path.stat().st_size
                != destination_path.stat().st_size
            ):
                return False

            return (
                self.calculate_hash(
                    source_path
                )
                == self.calculate_hash(
                    destination_path
                )
            )

        except OSError:
            return False

    # ==========================================================
    # FILE COPY
    # ==========================================================

    def copy_file(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = False,
        progress_callback: Optional[
            ProgressCallback
        ] = None,
    ) -> CopyResult:

        self._check()

        source_path = self.normalize_path(
            source
        )

        destination_path = (
            self.normalize_path(
                destination
            )
        )

        stats = CopyStats()
        start_time = time.perf_counter()

        if not source_path.exists():

            error = CopySourceNotFoundError(
                str(source_path)
            )

            result = CopyResult(
                success=False,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message="Source file not found.",
                error=str(error),
            )

            self._record(
                source_path,
                destination_path,
                False,
                stats,
                result.message,
            )

            return result

        if not source_path.is_file():

            result = CopyResult(
                success=False,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message=(
                    "Source is not a file."
                ),
                error="Source is not a file.",
            )

            return result

        if (
            destination_path.exists()
            and not overwrite
        ):

            error = (
                CopyDestinationExistsError(
                    str(destination_path)
                )
            )

            result = CopyResult(
                success=False,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message=(
                    "Destination already exists."
                ),
                error=str(error),
            )

            self._record(
                source_path,
                destination_path,
                False,
                stats,
                result.message,
            )

            return result

        file_size = source_path.stat().st_size

        if self.dry_run:

            stats.files_copied = 1
            stats.bytes_copied = file_size
            stats.elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            return CopyResult(
                success=True,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message=(
                    "Dry-run: file would "
                    "be copied."
                ),
                dry_run=True,
                stats=stats,
            )

        try:

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            copied = 0

            with (
                source_path.open("rb") as source_file,
                destination_path.open(
                    "wb"
                ) as destination_file,
            ):

                while True:

                    chunk = source_file.read(
                        self.buffer_size
                    )

                    if not chunk:
                        break

                    destination_file.write(
                        chunk
                    )

                    copied += len(chunk)

                    if progress_callback:
                        progress_callback(
                            copied,
                            file_size,
                            str(source_path),
                        )

            shutil.copystat(
                source_path,
                destination_path,
            )

            stats.files_copied = 1
            stats.bytes_copied = copied
            stats.elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            verified = False

            if verify:

                verified = (
                    self.verify_copy(
                        source_path,
                        destination_path,
                    )
                )

                if not verified:

                    raise CopyVerificationError(
                        "Copied file failed "
                        "verification."
                    )

            result = CopyResult(
                success=True,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message="File copied successfully.",
                verified=verified,
                stats=stats,
                files=[
                    str(destination_path)
                ],
            )

            self._record(
                source_path,
                destination_path,
                True,
                stats,
                result.message,
            )

            return result

        except (
            OSError,
            CopyVerificationError,
        ) as exc:

            stats.failed = 1
            stats.elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            result = CopyResult(
                success=False,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message="File copy failed.",
                error=str(exc),
                stats=stats,
            )

            self._record(
                source_path,
                destination_path,
                False,
                stats,
                result.message,
            )

            return result

    # ==========================================================
    # DIRECTORY COPY
    # ==========================================================

    def copy_directory(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = False,
        progress_callback: Optional[
            ProgressCallback
        ] = None,
    ) -> CopyResult:

        self._check()

        source_path = self.normalize_path(
            source
        )

        destination_path = (
            self.normalize_path(
                destination
            )
        )

        stats = CopyStats()
        start_time = time.perf_counter()

        if not source_path.exists():

            return CopyResult(
                success=False,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message="Source directory not found.",
                error=str(
                    CopySourceNotFoundError(
                        str(source_path)
                    )
                ),
            )

        if not source_path.is_dir():

            return CopyResult(
                success=False,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message=(
                    "Source is not a directory."
                ),
                error="Source is not a directory.",
            )

        if (
            destination_path.exists()
            and not overwrite
        ):

            return CopyResult(
                success=False,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message=(
                    "Destination already exists."
                ),
                error=str(
                    CopyDestinationExistsError(
                        str(destination_path)
                    )
                ),
            )

        files = []

        for current_root, _, filenames in os.walk(
            source_path
        ):

            current_path = Path(
                current_root
            )

            for filename in filenames:

                file_path = (
                    current_path / filename
                )

                try:

                    relative = (
                        file_path.relative_to(
                            source_path
                        )
                    )

                    target = (
                        destination_path
                        / relative
                    )

                    files.append(
                        (
                            file_path,
                            target,
                        )
                    )

                except ValueError:
                    continue

        total_bytes = 0

        for source_file, _ in files:

            try:
                total_bytes += (
                    source_file.stat().st_size
                )
            except OSError:
                pass

        if self.dry_run:

            stats.files_copied = len(files)
            stats.bytes_copied = total_bytes
            stats.directories_copied = 1
            stats.elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            return CopyResult(
                success=True,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message=(
                    "Dry-run: directory would "
                    "be copied."
                ),
                dry_run=True,
                stats=stats,
            )

        try:

            destination_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            copied_files = []

            for (
                source_file,
                target_file,
            ) in files:

                target_file.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                result = self.copy_file(
                    source_file,
                    target_file,
                    overwrite=True,
                    verify=verify,
                    progress_callback=(
                        self._directory_progress_callback(
                            progress_callback,
                            stats,
                            total_bytes,
                        )
                        if progress_callback
                        else None
                    ),
                )

                if not result.success:

                    stats.failed += 1
                    continue

                stats.files_copied += 1
                stats.bytes_copied += (
                    result.stats.bytes_copied
                )

                copied_files.extend(
                    result.files
                )

            stats.directories_copied = 1
            stats.elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            success = stats.failed == 0

            result = CopyResult(
                success=success,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message=(
                    "Directory copied successfully."
                    if success
                    else "Directory copied with errors."
                ),
                error=(
                    None
                    if success
                    else f"{stats.failed} file(s) failed."
                ),
                verified=(
                    verify and success
                ),
                stats=stats,
                files=copied_files,
            )

            self._record(
                source_path,
                destination_path,
                success,
                stats,
                result.message,
            )

            return result

        except OSError as exc:

            stats.failed += 1
            stats.elapsed_seconds = (
                time.perf_counter()
                - start_time
            )

            result = CopyResult(
                success=False,
                source=str(source_path),
                destination=str(
                    destination_path
                ),
                message="Directory copy failed.",
                error=str(exc),
                stats=stats,
            )

            self._record(
                source_path,
                destination_path,
                False,
                stats,
                result.message,
            )

            return result

    # ==========================================================
    # PROGRESS HELPER
    # ==========================================================

    @staticmethod
    def _directory_progress_callback(
        callback: ProgressCallback,
        stats: CopyStats,
        total_bytes: int,
    ) -> ProgressCallback:

        def wrapped(
            current: int,
            file_size: int,
            path: str,
        ) -> None:

            completed = (
                stats.bytes_copied
                + current
            )

            callback(
                completed,
                total_bytes,
                path,
            )

        return wrapped

    # ==========================================================
    # BATCH COPY
    # ==========================================================

    def copy_many(
        self,
        sources: Iterable[
            str | os.PathLike[str]
        ],
        destination_directory: str
        | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = False,
    ) -> list[CopyResult]:

        self._check()

        destination = self.normalize_path(
            destination_directory
        )

        if not self.dry_run:

            destination.mkdir(
                parents=True,
                exist_ok=True,
            )

        results = []

        for source in sources:

            source_path = (
                self.normalize_path(
                    source
                )
            )

            target = (
                destination
                / source_path.name
            )

            if source_path.is_dir():

                result = self.copy_directory(
                    source_path,
                    target,
                    overwrite=overwrite,
                    verify=verify,
                )

            else:

                result = self.copy_file(
                    source_path,
                    target,
                    overwrite=overwrite,
                    verify=verify,
                )

            results.append(result)

        return results

    # ==========================================================
    # COPY WITH AUTO-RENAME
    # ==========================================================

    def copy_unique(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        verify: bool = False,
    ) -> CopyResult:

        self._check()

        source_path = self.normalize_path(
            source
        )

        destination_path = (
            self.normalize_path(
                destination
            )
        )

        if not destination_path.exists():

            return (
                self.copy_directory(
                    source_path,
                    destination_path,
                    verify=verify,
                )
                if source_path.is_dir()
                else self.copy_file(
                    source_path,
                    destination_path,
                    verify=verify,
                )
            )

        stem = destination_path.stem
        suffix = destination_path.suffix
        parent = destination_path.parent

        counter = 1

        while True:

            candidate = (
                parent
                / f"{stem} ({counter}){suffix}"
            )

            if not candidate.exists():
                break

            counter += 1

        if source_path.is_dir():

            return self.copy_directory(
                source_path,
                candidate,
                verify=verify,
            )

        return self.copy_file(
            source_path,
            candidate,
            verify=verify,
        )

    # ==========================================================
    # COPY CONTENTS
    # ==========================================================

    def copy_contents(
        self,
        source_directory: str
        | os.PathLike[str],
        destination_directory: str
        | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = False,
    ) -> CopyResult:

        self._check()

        source = self.normalize_path(
            source_directory
        )

        destination = self.normalize_path(
            destination_directory
        )

        if not source.is_dir():

            return CopyResult(
                success=False,
                source=str(source),
                destination=str(
                    destination
                ),
                message=(
                    "Source is not a directory."
                ),
            )

        if not self.dry_run:

            destination.mkdir(
                parents=True,
                exist_ok=True,
            )

        combined_stats = CopyStats()
        copied_files = []
        failed = 0

        for item in source.iterdir():

            target = (
                destination / item.name
            )

            if item.is_dir():

                result = self.copy_directory(
                    item,
                    target,
                    overwrite=overwrite,
                    verify=verify,
                )

            else:

                result = self.copy_file(
                    item,
                    target,
                    overwrite=overwrite,
                    verify=verify,
                )

            combined_stats.files_copied += (
                result.stats.files_copied
            )

            combined_stats.directories_copied += (
                result.stats.directories_copied
            )

            combined_stats.bytes_copied += (
                result.stats.bytes_copied
            )

            copied_files.extend(
                result.files
            )

            if not result.success:
                failed += 1

        combined_stats.failed = failed

        return CopyResult(
            success=failed == 0,
            source=str(source),
            destination=str(destination),
            message=(
                "Directory contents copied."
                if failed == 0
                else "Some items failed to copy."
            ),
            error=(
                None
                if failed == 0
                else f"{failed} item(s) failed."
            ),
            verified=(
                verify and failed == 0
            ),
            stats=combined_stats,
            files=copied_files,
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "buffer_size": self.buffer_size,
            "hash_algorithm": (
                self.hash_algorithm
            ),
            "history_entries": len(
                self._history
            ),
            "capabilities": [
                "copy_file",
                "copy_directory",
                "copy_many",
                "copy_unique",
                "copy_contents",
                "verify_copy",
                "calculate_hash",
            ],
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX CopyManager shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_copy_manager: Optional[
    CopyManager
] = None


def get_copy_manager() -> CopyManager:

    global _default_copy_manager

    if _default_copy_manager is None:

        _default_copy_manager = (
            CopyManager()
        )

    return _default_copy_manager


# ==============================================================
# MODULE HELPERS
# ==============================================================


def copy_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> CopyResult:

    return (
        get_copy_manager()
        .copy_file(
            source,
            destination,
            overwrite=overwrite,
        )
    )


def copy_directory(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> CopyResult:

    return (
        get_copy_manager()
        .copy_directory(
            source,
            destination,
            overwrite=overwrite,
        )
    )


def copy_unique(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> CopyResult:

    return (
        get_copy_manager()
        .copy_unique(
            source,
            destination,
        )
    )


# ==============================================================
# PUBLIC API
# ==============================================================

__all__ = [
    "CopyManagerError",
    "CopySourceNotFoundError",
    "CopyDestinationExistsError",
    "CopyVerificationError",
    "CopyStats",
    "CopyResult",
    "CopyOperation",
    "CopyManager",
    "get_copy_manager",
    "copy_file",
    "copy_directory",
    "copy_unique",
]


