"""
RENIX Move Manager
==================

High-level file and directory moving service.

Responsibilities:
- Move files
- Move directories
- Batch moves
- Collision handling
- Automatic unique naming
- Dry-run support
- Progress reporting
- Move history
- Safe source/destination validation

The manager prefers an atomic filesystem move when possible.
Cross-device moves automatically fall back to copy + source removal.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# CONSTANTS
# ==============================================================

DEFAULT_BUFFER_SIZE = 1024 * 1024

ProgressCallback = Callable[[int, int, str], None]


# ==============================================================
# EXCEPTIONS
# ==============================================================


class MoveManagerError(Exception):
    """Base exception for move operations."""


class MoveSourceNotFoundError(MoveManagerError):
    """Raised when the source does not exist."""


class MoveDestinationExistsError(MoveManagerError):
    """Raised when destination already exists."""


class MoveInvalidDestinationError(MoveManagerError):
    """Raised when destination is invalid."""


class MoveVerificationError(MoveManagerError):
    """Raised when the moved item cannot be verified."""


# ==============================================================
# DATA MODELS
# ==============================================================


@dataclass
class MoveStats:
    """Statistics for a move operation."""

    files_moved: int = 0
    directories_moved: int = 0
    bytes_moved: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def average_speed_bytes_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0

        return self.bytes_moved / self.elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_moved": self.files_moved,
            "directories_moved": self.directories_moved,
            "bytes_moved": self.bytes_moved,
            "failed": self.failed,
            "elapsed_seconds": self.elapsed_seconds,
            "average_speed_bytes_per_second": (
                self.average_speed_bytes_per_second
            ),
        }


@dataclass
class MoveResult:
    """Detailed result of a move operation."""

    success: bool
    source: Optional[str] = None
    destination: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    dry_run: bool = False
    verified: bool = False
    cross_device: bool = False
    stats: MoveStats = field(default_factory=MoveStats)
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "destination": self.destination,
            "message": self.message,
            "error": self.error,
            "dry_run": self.dry_run,
            "verified": self.verified,
            "cross_device": self.cross_device,
            "stats": self.stats.to_dict(),
            "files": self.files,
        }


@dataclass
class MoveOperation:
    """Historical record of a move operation."""

    source: Optional[str]
    destination: Optional[str]
    success: bool
    timestamp: float
    files_moved: int
    bytes_moved: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "success": self.success,
            "timestamp": self.timestamp,
            "files_moved": self.files_moved,
            "bytes_moved": self.bytes_moved,
            "message": self.message,
        }


# ==============================================================
# MOVE MANAGER
# ==============================================================


class MoveManager:
    """
    RENIX high-level move engine.

    A move is treated as a filesystem operation, not a copy
    operation. The source is removed from its original location
    after a successful move.

    Use dry_run=True when RENIX needs to preview an operation.
    """

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:

        self.enabled = enabled
        self.dry_run = dry_run
        self.buffer_size = buffer_size

        self._history: list[MoveOperation] = []

        logger.info("RENIX MoveManager initialized.")

    # ==========================================================
    # STATE
    # ==========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def set_dry_run(self, enabled: bool) -> None:
        self.dry_run = enabled

    def _check_enabled(self) -> None:
        if not self.enabled:
            raise MoveManagerError(
                "RENIX MoveManager is disabled."
            )

    # ==========================================================
    # PATH HELPERS
    # ==========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise MoveManagerError(
                "Path cannot be None."
            )

        return Path(path).expanduser()

    @staticmethod
    def absolute_path(
        path: str | os.PathLike[str],
    ) -> Path:

        return MoveManager.normalize_path(path).resolve()

    @staticmethod
    def _is_same_path(
        source: Path,
        destination: Path,
    ) -> bool:

        try:
            return source.resolve() == destination.resolve()
        except OSError:
            return source.absolute() == destination.absolute()

    @staticmethod
    def _is_inside(
        child: Path,
        parent: Path,
    ) -> bool:

        try:
            child.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    # ==========================================================
    # HISTORY
    # ==========================================================

    def _record(
        self,
        source: Optional[Path],
        destination: Optional[Path],
        success: bool,
        stats: MoveStats,
        message: str,
    ) -> None:

        self._history.append(
            MoveOperation(
                source=str(source) if source else None,
                destination=(
                    str(destination)
                    if destination
                    else None
                ),
                success=success,
                timestamp=time.time(),
                files_moved=stats.files_moved,
                bytes_moved=stats.bytes_moved,
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

        return [item.to_dict() for item in history]

    def clear_history(self) -> None:
        self._history.clear()

    # ==========================================================
    # SIZE
    # ==========================================================

    @staticmethod
    def calculate_size(path: Path) -> int:

        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0

        if path.is_dir():

            total = 0

            for root, _, files in os.walk(path):

                for filename in files:

                    file_path = Path(root) / filename

                    try:
                        total += file_path.stat().st_size
                    except OSError:
                        continue

            return total

        return 0

    @staticmethod
    def format_size(size: int) -> str:

        if size < 1024:
            return f"{size} B"

        value = float(size)

        for unit in ("KB", "MB", "GB", "TB", "PB"):

            value /= 1024

            if value < 1024:
                return f"{value:.2f} {unit}"

        return f"{value:.2f} EB"

    # ==========================================================
    # DESTINATION HELPERS
    # ==========================================================

    def generate_unique_destination(
        self,
        destination: Path,
    ) -> Path:

        if not destination.exists():
            return destination

        if destination.is_dir():
            base = destination.name
            parent = destination.parent
            suffix = ""
        else:
            base = destination.stem
            parent = destination.parent
            suffix = destination.suffix

        counter = 1

        while True:

            candidate = (
                parent
                / f"{base} ({counter}){suffix}"
            )

            if not candidate.exists():
                return candidate

            counter += 1

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def _validate(
        self,
        source: Path,
        destination: Path,
        overwrite: bool,
    ) -> Optional[str]:

        if not source.exists():
            return "Source does not exist."

        if self._is_same_path(source, destination):
            return "Source and destination are identical."

        if source.is_dir():

            if self._is_inside(destination, source):
                return (
                    "Destination cannot be inside "
                    "the source directory."
                )

        if destination.exists() and not overwrite:
            return "Destination already exists."

        if destination.exists():

            if source.is_file() and destination.is_dir():
                return (
                    "Cannot overwrite a directory "
                    "with a file."
                )

            if source.is_dir() and destination.is_file():
                return (
                    "Cannot overwrite a file "
                    "with a directory."
                )

        return None

    # ==========================================================
    # FILE MOVE
    # ==========================================================

    def move_file(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> MoveResult:

        self._check_enabled()

        source_path = self.normalize_path(source)
        destination_path = self.normalize_path(destination)

        stats = MoveStats()
        start_time = time.perf_counter()

        error = self._validate(
            source_path,
            destination_path,
            overwrite,
        )

        if error:

            result = MoveResult(
                success=False,
                source=str(source_path),
                destination=str(destination_path),
                message=error,
                error=error,
                stats=stats,
            )

            self._record(
                source_path,
                destination_path,
                False,
                stats,
                error,
            )

            return result

        if not source_path.is_file():

            message = "Source is not a file."

            result = MoveResult(
                success=False,
                source=str(source_path),
                destination=str(destination_path),
                message=message,
                error=message,
                stats=stats,
            )

            return result

        size = self.calculate_size(source_path)

        if self.dry_run:

            stats.files_moved = 1
            stats.bytes_moved = size
            stats.elapsed_seconds = (
                time.perf_counter() - start_time
            )

            return MoveResult(
                success=True,
                source=str(source_path),
                destination=str(destination_path),
                message="Dry-run: file would be moved.",
                dry_run=True,
                stats=stats,
            )

        try:

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cross_device = False

            try:

                shutil.move(
                    str(source_path),
                    str(destination_path),
                )

            except OSError as exc:

                if getattr(exc, "errno", None) != 18:
                    raise

                # EXDEV = cross-device move.
                cross_device = True

                self._copy_file_cross_device(
                    source_path,
                    destination_path,
                    size,
                    progress_callback,
                )

                source_path.unlink()

            stats.files_moved = 1
            stats.bytes_moved = size
            stats.elapsed_seconds = (
                time.perf_counter() - start_time
            )

            verified = False

            if verify:

                verified = (
                    destination_path.exists()
                    and not source_path.exists()
                )

                if not verified:
                    raise MoveVerificationError(
                        "Move verification failed."
                    )

            if progress_callback:
                progress_callback(
                    size,
                    size,
                    str(destination_path),
                )

            result = MoveResult(
                success=True,
                source=str(source_path),
                destination=str(destination_path),
                message="File moved successfully.",
                verified=verified,
                cross_device=cross_device,
                stats=stats,
                files=[str(destination_path)],
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
            MoveVerificationError,
        ) as exc:

            stats.failed = 1
            stats.elapsed_seconds = (
                time.perf_counter() - start_time
            )

            result = MoveResult(
                success=False,
                source=str(source_path),
                destination=str(destination_path),
                message="File move failed.",
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
    # CROSS DEVICE COPY
    # ==========================================================

    def _copy_file_cross_device(
        self,
        source: Path,
        destination: Path,
        total_size: int,
        progress_callback: Optional[ProgressCallback],
    ) -> None:

        copied = 0

        with (
            source.open("rb") as source_file,
            destination.open("wb") as destination_file,
        ):

            while True:

                chunk = source_file.read(
                    self.buffer_size
                )

                if not chunk:
                    break

                destination_file.write(chunk)

                copied += len(chunk)

                if progress_callback:
                    progress_callback(
                        copied,
                        total_size,
                        str(source),
                    )

        shutil.copystat(
            source,
            destination,
        )

    # ==========================================================
    # DIRECTORY MOVE
    # ==========================================================

    def move_directory(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> MoveResult:

        self._check_enabled()

        source_path = self.normalize_path(source)
        destination_path = self.normalize_path(destination)

        stats = MoveStats()
        start_time = time.perf_counter()

        error = self._validate(
            source_path,
            destination_path,
            overwrite,
        )

        if error:

            result = MoveResult(
                success=False,
                source=str(source_path),
                destination=str(destination_path),
                message=error,
                error=error,
                stats=stats,
            )

            self._record(
                source_path,
                destination_path,
                False,
                stats,
                error,
            )

            return result

        if not source_path.is_dir():

            message = "Source is not a directory."

            return MoveResult(
                success=False,
                source=str(source_path),
                destination=str(destination_path),
                message=message,
                error=message,
                stats=stats,
            )

        total_size = self.calculate_size(source_path)
        file_count = self._count_files(source_path)

        if self.dry_run:

            stats.files_moved = file_count
            stats.directories_moved = 1
            stats.bytes_moved = total_size
            stats.elapsed_seconds = (
                time.perf_counter() - start_time
            )

            return MoveResult(
                success=True,
                source=str(source_path),
                destination=str(destination_path),
                message=(
                    "Dry-run: directory would be moved."
                ),
                dry_run=True,
                stats=stats,
            )

        try:

            cross_device = False

            try:

                shutil.move(
                    str(source_path),
                    str(destination_path),
                )

            except OSError as exc:

                if getattr(exc, "errno", None) != 18:
                    raise

                cross_device = True

                self._move_directory_cross_device(
                    source_path,
                    destination_path,
                    overwrite=overwrite,
                    progress_callback=progress_callback,
                )

            stats.files_moved = file_count
            stats.directories_moved = 1
            stats.bytes_moved = total_size
            stats.elapsed_seconds = (
                time.perf_counter() - start_time
            )

            verified = False

            if verify:

                verified = (
                    destination_path.exists()
                    and not source_path.exists()
                )

                if not verified:
                    raise MoveVerificationError(
                        "Directory move verification failed."
                    )

            if progress_callback:
                progress_callback(
                    total_size,
                    total_size,
                    str(destination_path),
                )

            result = MoveResult(
                success=True,
                source=str(source_path),
                destination=str(destination_path),
                message="Directory moved successfully.",
                verified=verified,
                cross_device=cross_device,
                stats=stats,
                files=[str(destination_path)],
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
            MoveVerificationError,
        ) as exc:

            stats.failed = 1
            stats.elapsed_seconds = (
                time.perf_counter() - start_time
            )

            result = MoveResult(
                success=False,
                source=str(source_path),
                destination=str(destination_path),
                message="Directory move failed.",
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
    # CROSS DEVICE DIRECTORY MOVE
    # ==========================================================

    def _move_directory_cross_device(
        self,
        source: Path,
        destination: Path,
        *,
        overwrite: bool,
        progress_callback: Optional[ProgressCallback],
    ) -> None:

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_size = self.calculate_size(source)
        copied_size = 0

        for root, dirs, files in os.walk(source):

            current_root = Path(root)

            relative_root = current_root.relative_to(source)
            destination_root = (
                destination / relative_root
            )

            destination_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            for directory in dirs:

                (
                    destination_root / directory
                ).mkdir(
                    parents=True,
                    exist_ok=True,
                )

            for filename in files:

                source_file = (
                    current_root / filename
                )

                destination_file = (
                    destination_root / filename
                )

                if (
                    destination_file.exists()
                    and not overwrite
                ):
                    raise MoveDestinationExistsError(
                        str(destination_file)
                    )

                file_size = self.calculate_size(
                    source_file
                )

                self._copy_file_cross_device(
                    source_file,
                    destination_file,
                    file_size,
                    lambda current, total, path: (
                        progress_callback(
                            copied_size + current,
                            total_size,
                            path,
                        )
                        if progress_callback
                        else None
                    ),
                )

                copied_size += file_size

        shutil.copystat(
            source,
            destination,
        )

        shutil.rmtree(source)

    # ==========================================================
    # COUNT FILES
    # ==========================================================

    @staticmethod
    def _count_files(path: Path) -> int:

        count = 0

        for _, _, files in os.walk(path):
            count += len(files)

        return count

    # ==========================================================
    # MOVE ANY
    # ==========================================================

    def move(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> MoveResult:

        self._check_enabled()

        source_path = self.normalize_path(source)

        if source_path.is_dir():

            return self.move_directory(
                source_path,
                destination,
                overwrite=overwrite,
                verify=verify,
                progress_callback=progress_callback,
            )

        return self.move_file(
            source_path,
            destination,
            overwrite=overwrite,
            verify=verify,
            progress_callback=progress_callback,
        )

    # ==========================================================
    # MOVE MANY
    # ==========================================================

    def move_many(
        self,
        sources: Iterable[str | os.PathLike[str]],
        destination_directory: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = False,
        unique_names: bool = False,
    ) -> list[MoveResult]:

        self._check_enabled()

        destination = self.normalize_path(
            destination_directory
        )

        if not self.dry_run:
            destination.mkdir(
                parents=True,
                exist_ok=True,
            )

        results: list[MoveResult] = []

        for source in sources:

            source_path = self.normalize_path(source)

            target = destination / source_path.name

            if unique_names:
                target = self.generate_unique_destination(
                    target
                )

            result = self.move(
                source_path,
                target,
                overwrite=overwrite,
                verify=verify,
            )

            results.append(result)

        return results

    # ==========================================================
    # MOVE UNIQUE
    # ==============================================================

    def move_unique(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        verify: bool = False,
    ) -> MoveResult:

        self._check_enabled()

        source_path = self.normalize_path(source)
        destination_path = self.normalize_path(destination)

        destination_path = (
            self.generate_unique_destination(
                destination_path
            )
        )

        return self.move(
            source_path,
            destination_path,
            overwrite=False,
            verify=verify,
        )

    # ==========================================================
    # MOVE INTO DIRECTORY
    # ==============================================================

    def move_into(
        self,
        source: str | os.PathLike[str],
        directory: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        unique_names: bool = False,
        verify: bool = False,
    ) -> MoveResult:

        self._check_enabled()

        source_path = self.normalize_path(source)
        directory_path = self.normalize_path(directory)

        if not self.dry_run:

            directory_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        destination = (
            directory_path / source_path.name
        )

        if unique_names:

            destination = (
                self.generate_unique_destination(
                    destination
                )
            )

        return self.move(
            source_path,
            destination,
            overwrite=overwrite,
            verify=verify,
        )

    # ==========================================================
    # MOVE CONTENTS
    # ==========================================================

    def move_contents(
        self,
        source_directory: str | os.PathLike[str],
        destination_directory: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        unique_names: bool = False,
        verify: bool = False,
    ) -> list[MoveResult]:

        self._check_enabled()

        source = self.normalize_path(
            source_directory
        )

        destination = self.normalize_path(
            destination_directory
        )

        if not source.is_dir():

            raise MoveManagerError(
                "Source directory does not exist."
            )

        if not self.dry_run:

            destination.mkdir(
                parents=True,
                exist_ok=True,
            )

        results: list[MoveResult] = []

        # Make a list first because moving items while
        # iterating over it changes the directory contents.
        items = list(source.iterdir())

        for item in items:

            target = destination / item.name

            if unique_names:
                target = (
                    self.generate_unique_destination(
                        target
                    )
                )

            results.append(
                self.move(
                    item,
                    target,
                    overwrite=overwrite,
                    verify=verify,
                )
            )

        return results

    # ==========================================================
    # PREVIEW
    # ==========================================================

    def preview(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:

        source_path = self.normalize_path(source)
        destination_path = self.normalize_path(
            destination
        )

        if not source_path.exists():

            return {
                "valid": False,
                "reason": "Source does not exist.",
            }

        validation_error = self._validate(
            source_path,
            destination_path,
            overwrite,
        )

        return {
            "valid": validation_error is None,
            "source": str(source_path),
            "destination": str(destination_path),
            "source_type": (
                "directory"
                if source_path.is_dir()
                else "file"
            ),
            "size_bytes": self.calculate_size(
                source_path
            ),
            "size_human": self.format_size(
                self.calculate_size(source_path)
            ),
            "destination_exists": (
                destination_path.exists()
            ),
            "reason": validation_error,
        }

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "buffer_size": self.buffer_size,
            "history_entries": len(self._history),
            "capabilities": [
                "move",
                "move_file",
                "move_directory",
                "move_many",
                "move_unique",
                "move_into",
                "move_contents",
                "preview",
                "cross_device_move",
                "verification",
            ],
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX MoveManager shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================

_default_move_manager: Optional[MoveManager] = None


def get_move_manager() -> MoveManager:

    global _default_move_manager

    if _default_move_manager is None:
        _default_move_manager = MoveManager()

    return _default_move_manager


# ==============================================================
# CONVENIENCE FUNCTIONS
# ==============================================================


def move_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> MoveResult:

    return get_move_manager().move_file(
        source,
        destination,
        overwrite=overwrite,
    )


def move_directory(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> MoveResult:

    return get_move_manager().move_directory(
        source,
        destination,
        overwrite=overwrite,
    )


def move(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> MoveResult:

    return get_move_manager().move(
        source,
        destination,
        overwrite=overwrite,
    )


def move_unique(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> MoveResult:

    return get_move_manager().move_unique(
        source,
        destination,
    )


# ==============================================================
# PUBLIC API
# ==============================================================

__all__ = [
    "MoveManagerError",
    "MoveSourceNotFoundError",
    "MoveDestinationExistsError",
    "MoveInvalidDestinationError",
    "MoveVerificationError",
    "MoveStats",
    "MoveResult",
    "MoveOperation",
    "MoveManager",
    "get_move_manager",
    "move_file",
    "move_directory",
    "move",
    "move_unique",
]


