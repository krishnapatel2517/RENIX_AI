"""
RENIX Rename Manager
====================

Safe, high-level file and directory renaming service.

Features:
- Rename files and folders
- Batch rename
- Sequential numbering
- Find/replace
- Prefix/suffix operations
- Extension changes
- Case conversion
- Collision protection
- Dry-run previews
- Rename history
- Verification
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class RenameManagerError(Exception):
    """Base exception for rename operations."""


class RenameSourceNotFoundError(RenameManagerError):
    """Raised when the source does not exist."""


class RenameDestinationExistsError(RenameManagerError):
    """Raised when the target name already exists."""


class RenameInvalidNameError(RenameManagerError):
    """Raised when a filename is invalid."""


class RenameVerificationError(RenameManagerError):
    """Raised when rename verification fails."""


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class RenameResult:
    success: bool
    source: Optional[str] = None
    destination: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    dry_run: bool = False
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "destination": self.destination,
            "message": self.message,
            "error": self.error,
            "dry_run": self.dry_run,
            "verified": self.verified,
        }


@dataclass
class RenameOperation:
    source: str
    destination: str
    success: bool
    timestamp: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "success": self.success,
            "timestamp": self.timestamp,
            "message": self.message,
        }


@dataclass
class BatchRenameResult:
    success: bool
    results: list[RenameResult] = field(default_factory=list)
    total: int = 0
    successful: int = 0
    failed: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "message": self.message,
            "results": [
                result.to_dict()
                for result in self.results
            ],
        }


# ============================================================
# RENAME MANAGER
# ============================================================


class RenameManager:
    """
    RENIX filesystem rename engine.

    All operations use Path.rename() so the filesystem performs
    a normal rename instead of unnecessarily copying data.
    """

    INVALID_WINDOWS_CHARS = '<>:"/\\|?*'

    RESERVED_WINDOWS_NAMES = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
    ) -> None:

        self.enabled = enabled
        self.dry_run = dry_run
        self._history: list[RenameOperation] = []

        logger.info(
            "RENIX RenameManager initialized."
        )

    # ========================================================
    # STATE
    # ========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def set_dry_run(self, enabled: bool) -> None:
        self.dry_run = enabled

    def _check_enabled(self) -> None:
        if not self.enabled:
            raise RenameManagerError(
                "RENIX RenameManager is disabled."
            )

    # ========================================================
    # PATH HELPERS
    # ========================================================

    @staticmethod
    def normalize_path(
        path: str | Path,
    ) -> Path:

        if path is None:
            raise RenameManagerError(
                "Path cannot be None."
            )

        return Path(path).expanduser()

    @staticmethod
    def absolute_path(path: str | Path) -> Path:
        return RenameManager.normalize_path(path).resolve()

    # ========================================================
    # NAME VALIDATION
    # ========================================================

    @classmethod
    def validate_name(
        cls,
        name: str,
        *,
        platform_specific: bool = True,
    ) -> tuple[bool, str]:

        if not isinstance(name, str):
            return False, "Name must be a string."

        if not name:
            return False, "Name cannot be empty."

        if name in {".", ".."}:
            return False, "Invalid special directory name."

        if "\x00" in name:
            return False, "Name contains a null character."

        if "/" in name or "\\" in name:
            return False, (
                "Name cannot contain path separators."
            )

        if platform_specific and Path().anchor == "":
            # Windows-specific validation is useful when
            # RENIX is running on Windows.
            for char in cls.INVALID_WINDOWS_CHARS:
                if char in name:
                    return False, (
                        f"Name contains invalid character: {char}"
                    )

            if name.endswith(" ") or name.endswith("."):
                return False, (
                    "Windows names cannot end with "
                    "a space or period."
                )

            stem = Path(name).stem.upper()

            if stem in cls.RESERVED_WINDOWS_NAMES:
                return False, (
                    f"'{name}' is a reserved Windows name."
                )

        return True, ""

    # ========================================================
    # UNIQUE NAME
    # ========================================================

    @staticmethod
    def generate_unique_name(
        path: str | Path,
        *,
        separator: str = " ",
        start: int = 1,
    ) -> Path:

        path = Path(path)

        if not path.exists():
            return path

        parent = path.parent

        if path.is_dir():
            stem = path.name
            suffix = ""
        else:
            stem = path.stem
            suffix = path.suffix

        counter = start

        while True:

            candidate = (
                parent
                / f"{stem}{separator}({counter}){suffix}"
            )

            if not candidate.exists():
                return candidate

            counter += 1

    # ========================================================
    # HISTORY
    # ========================================================

    def _record(
        self,
        source: Path,
        destination: Path,
        success: bool,
        message: str,
    ) -> None:

        self._history.append(
            RenameOperation(
                source=str(source),
                destination=str(destination),
                success=success,
                timestamp=time.time(),
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
            operation.to_dict()
            for operation in history
        ]

    def clear_history(self) -> None:
        self._history.clear()

    # ========================================================
    # DESTINATION
    # ========================================================

    def build_destination(
        self,
        source: Path,
        new_name: str,
    ) -> Path:

        valid, reason = self.validate_name(new_name)

        if not valid:
            raise RenameInvalidNameError(reason)

        return source.parent / new_name

    # ========================================================
    # CORE RENAME
    # ========================================================

    def rename(
        self,
        source: str | Path,
        new_name: str,
        *,
        overwrite: bool = False,
        unique_name: bool = False,
        verify: bool = False,
    ) -> RenameResult:

        self._check_enabled()

        source_path = self.normalize_path(source)

        if not source_path.exists():

            message = "Source does not exist."

            result = RenameResult(
                success=False,
                source=str(source_path),
                message=message,
                error=message,
            )

            return result

        try:
            destination = self.build_destination(
                source_path,
                new_name,
            )
        except RenameInvalidNameError as exc:

            return RenameResult(
                success=False,
                source=str(source_path),
                message="Invalid new name.",
                error=str(exc),
            )

        if destination.exists():

            if unique_name:
                destination = (
                    self.generate_unique_name(
                        destination
                    )
                )

            elif not overwrite:

                message = (
                    "Destination name already exists."
                )

                result = RenameResult(
                    success=False,
                    source=str(source_path),
                    destination=str(destination),
                    message=message,
                    error=message,
                )

                self._record(
                    source_path,
                    destination,
                    False,
                    message,
                )

                return result

        if self.dry_run:

            return RenameResult(
                success=True,
                source=str(source_path),
                destination=str(destination),
                message=(
                    "Dry-run: item would be renamed."
                ),
                dry_run=True,
            )

        try:

            if overwrite and destination.exists():
                self._remove_destination(
                    destination
                )

            source_path.rename(destination)

            verified = False

            if verify:

                verified = (
                    destination.exists()
                    and not source_path.exists()
                )

                if not verified:
                    raise RenameVerificationError(
                        "Rename verification failed."
                    )

            result = RenameResult(
                success=True,
                source=str(source_path),
                destination=str(destination),
                message="Renamed successfully.",
                verified=verified,
            )

            self._record(
                source_path,
                destination,
                True,
                result.message,
            )

            return result

        except (
            OSError,
            RenameVerificationError,
        ) as exc:

            message = "Rename failed."

            result = RenameResult(
                success=False,
                source=str(source_path),
                destination=str(destination),
                message=message,
                error=str(exc),
            )

            self._record(
                source_path,
                destination,
                False,
                message,
            )

            logger.exception(
                "RENIX rename operation failed."
            )

            return result

    # ========================================================
    # DESTINATION REMOVAL
    # ========================================================

    @staticmethod
    def _remove_destination(
        destination: Path,
    ) -> None:

        if destination.is_dir():
            import shutil

            shutil.rmtree(destination)
        else:
            destination.unlink()

    # ========================================================
    # RENAME TO EXACT PATH
    # ========================================================

    def rename_to(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
        unique_name: bool = False,
        verify: bool = False,
    ) -> RenameResult:

        self._check_enabled()

        source_path = self.normalize_path(source)
        destination_path = self.normalize_path(
            destination
        )

        if not source_path.exists():

            return RenameResult(
                success=False,
                source=str(source_path),
                destination=str(destination_path),
                message="Source does not exist.",
                error="Source does not exist.",
            )

        if destination_path.exists():

            if unique_name:

                destination_path = (
                    self.generate_unique_name(
                        destination_path
                    )
                )

            elif not overwrite:

                return RenameResult(
                    success=False,
                    source=str(source_path),
                    destination=str(destination_path),
                    message=(
                        "Destination already exists."
                    ),
                    error=(
                        "Destination already exists."
                    ),
                )

        if self.dry_run:

            return RenameResult(
                success=True,
                source=str(source_path),
                destination=str(destination_path),
                message=(
                    "Dry-run: item would be renamed."
                ),
                dry_run=True,
            )

        try:

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if overwrite and destination_path.exists():
                self._remove_destination(
                    destination_path
                )

            source_path.rename(destination_path)

            verified = False

            if verify:

                verified = (
                    destination_path.exists()
                    and not source_path.exists()
                )

                if not verified:
                    raise RenameVerificationError(
                        "Rename verification failed."
                    )

            result = RenameResult(
                success=True,
                source=str(source_path),
                destination=str(destination_path),
                message="Renamed successfully.",
                verified=verified,
            )

            self._record(
                source_path,
                destination_path,
                True,
                result.message,
            )

            return result

        except (
            OSError,
            RenameVerificationError,
        ) as exc:

            message = "Rename failed."

            self._record(
                source_path,
                destination_path,
                False,
                message,
            )

            return RenameResult(
                success=False,
                source=str(source_path),
                destination=str(destination_path),
                message=message,
                error=str(exc),
            )

    # ========================================================
    # PREFIX
    # ========================================================

    def add_prefix(
        self,
        source: str | Path,
        prefix: str,
        *,
        verify: bool = False,
    ) -> RenameResult:

        source_path = self.normalize_path(source)

        new_name = prefix + source_path.name

        return self.rename(
            source_path,
            new_name,
            verify=verify,
        )

    # ========================================================
    # SUFFIX
    # ========================================================

    def add_suffix(
        self,
        source: str | Path,
        suffix: str,
        *,
        verify: bool = False,
    ) -> RenameResult:

        source_path = self.normalize_path(source)

        if source_path.is_file():

            new_name = (
                source_path.stem
                + suffix
                + source_path.suffix
            )

        else:

            new_name = source_path.name + suffix

        return self.rename(
            source_path,
            new_name,
            verify=verify,
        )

    # ========================================================
    # FIND / REPLACE
    # ========================================================

    def replace_text(
        self,
        source: str | Path,
        old: str,
        new: str,
        *,
        count: int = -1,
        verify: bool = False,
    ) -> RenameResult:

        source_path = self.normalize_path(source)

        new_name = source_path.name.replace(
            old,
            new,
            count,
        )

        if new_name == source_path.name:

            return RenameResult(
                success=False,
                source=str(source_path),
                destination=str(source_path),
                message="No matching text found.",
                error="No matching text found.",
            )

        return self.rename(
            source_path,
            new_name,
            verify=verify,
        )

    # ========================================================
    # REGEX REPLACE
    # ========================================================

    def regex_replace(
        self,
        source: str | Path,
        pattern: str,
        replacement: str,
        *,
        flags: int = 0,
        verify: bool = False,
    ) -> RenameResult:

        source_path = self.normalize_path(source)

        try:

            new_name = re.sub(
                pattern,
                replacement,
                source_path.name,
                flags=flags,
            )

        except re.error as exc:

            return RenameResult(
                success=False,
                source=str(source_path),
                message="Invalid regular expression.",
                error=str(exc),
            )

        if new_name == source_path.name:

            return RenameResult(
                success=False,
                source=str(source_path),
                destination=str(source_path),
                message="Pattern produced no change.",
                error="Pattern produced no change.",
            )

        return self.rename(
            source_path,
            new_name,
            verify=verify,
        )

    # ========================================================
    # EXTENSION
    # ========================================================

    def change_extension(
        self,
        source: str | Path,
        extension: str,
        *,
        verify: bool = False,
    ) -> RenameResult:

        source_path = self.normalize_path(source)

        if source_path.is_dir():

            return RenameResult(
                success=False,
                source=str(source_path),
                message=(
                    "Directories do not have file extensions."
                ),
                error=(
                    "Directories do not have file extensions."
                ),
            )

        extension = extension.strip()

        if extension and not extension.startswith("."):
            extension = "." + extension

        new_name = source_path.stem + extension

        return self.rename(
            source_path,
            new_name,
            verify=verify,
        )

    # ========================================================
    # CASE CONVERSION
    # ========================================================

    def lowercase(
        self,
        source: str | Path,
        *,
        verify: bool = False,
    ) -> RenameResult:

        source_path = self.normalize_path(source)

        return self.rename(
            source_path,
            source_path.name.lower(),
            verify=verify,
        )

    def uppercase(
        self,
        source: str | Path,
        *,
        verify: bool = False,
    ) -> RenameResult:

        source_path = self.normalize_path(source)

        return self.rename(
            source_path,
            source_path.name.upper(),
            verify=verify,
        )

    # ========================================================
    # BATCH RENAME
    # ========================================================

    def batch_rename(
        self,
        sources: Iterable[str | Path],
        name_generator,
        *,
        verify: bool = False,
        unique_name: bool = False,
    ) -> BatchRenameResult:

        self._check_enabled()

        sources = list(sources)

        results: list[RenameResult] = []

        for index, source in enumerate(sources):

            source_path = self.normalize_path(source)

            try:
                new_name = name_generator(
                    source_path,
                    index,
                )
            except Exception as exc:

                results.append(
                    RenameResult(
                        success=False,
                        source=str(source_path),
                        message=(
                            "Name generator failed."
                        ),
                        error=str(exc),
                    )
                )

                continue

            result = self.rename(
                source_path,
                str(new_name),
                unique_name=unique_name,
                verify=verify,
            )

            results.append(result)

        successful = sum(
            result.success
            for result in results
        )

        failed = len(results) - successful

        return BatchRenameResult(
            success=failed == 0,
            results=results,
            total=len(results),
            successful=successful,
            failed=failed,
            message=(
                f"{successful}/{len(results)} "
                "items renamed successfully."
            ),
        )

    # ========================================================
    # SEQUENTIAL RENAME
    # ========================================================

    def sequential_rename(
        self,
        sources: Iterable[str | Path],
        base_name: str,
        *,
        start: int = 1,
        digits: int = 0,
        keep_extension: bool = True,
        separator: str = "_",
        verify: bool = False,
    ) -> BatchRenameResult:

        sources = list(sources)

        def generator(
            path: Path,
            index: int,
        ) -> str:

            number = start + index

            if digits > 0:
                number_text = str(number).zfill(digits)
            else:
                number_text = str(number)

            stem = (
                f"{base_name}"
                f"{separator}"
                f"{number_text}"
            )

            if keep_extension and path.is_file():
                return stem + path.suffix

            return stem

        return self.batch_rename(
            sources,
            generator,
            verify=verify,
        )

    # ========================================================
    # PREVIEW
    # ========================================================

    def preview(
        self,
        source: str | Path,
        new_name: str,
        *,
        unique_name: bool = False,
    ) -> dict[str, Any]:

        source_path = self.normalize_path(source)

        if not source_path.exists():

            return {
                "valid": False,
                "reason": "Source does not exist.",
            }

        valid, reason = self.validate_name(
            new_name
        )

        if not valid:

            return {
                "valid": False,
                "source": str(source_path),
                "new_name": new_name,
                "reason": reason,
            }

        destination = (
            source_path.parent / new_name
        )

        if destination.exists():

            if unique_name:

                destination = (
                    self.generate_unique_name(
                        destination
                    )
                )

            else:

                return {
                    "valid": False,
                    "source": str(source_path),
                    "destination": str(destination),
                    "reason": (
                        "Destination already exists."
                    ),
                }

        return {
            "valid": True,
            "source": str(source_path),
            "destination": str(destination),
            "old_name": source_path.name,
            "new_name": destination.name,
        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "history_entries": len(self._history),
            "capabilities": [
                "rename",
                "rename_to",
                "prefix",
                "suffix",
                "find_replace",
                "regex_replace",
                "extension_change",
                "lowercase",
                "uppercase",
                "batch_rename",
                "sequential_rename",
                "unique_names",
                "preview",
                "verification",
            ],
        }

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX RenameManager shut down."
        )


# ============================================================
# SHARED INSTANCE
# ============================================================

_default_rename_manager: Optional[RenameManager] = None


def get_rename_manager() -> RenameManager:

    global _default_rename_manager

    if _default_rename_manager is None:
        _default_rename_manager = RenameManager()

    return _default_rename_manager


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def rename(
    source: str | Path,
    new_name: str,
    overwrite: bool = False,
) -> RenameResult:

    return get_rename_manager().rename(
        source,
        new_name,
        overwrite=overwrite,
    )


def rename_to(
    source: str | Path,
    destination: str | Path,
    overwrite: bool = False,
) -> RenameResult:

    return get_rename_manager().rename_to(
        source,
        destination,
        overwrite=overwrite,
    )


def add_prefix(
    source: str | Path,
    prefix: str,
) -> RenameResult:

    return get_rename_manager().add_prefix(
        source,
        prefix,
    )


def add_suffix(
    source: str | Path,
    suffix: str,
) -> RenameResult:

    return get_rename_manager().add_suffix(
        source,
        suffix,
    )


def replace_text(
    source: str | Path,
    old: str,
    new: str,
) -> RenameResult:

    return get_rename_manager().replace_text(
        source,
        old,
        new,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "RenameManagerError",
    "RenameSourceNotFoundError",
    "RenameDestinationExistsError",
    "RenameInvalidNameError",
    "RenameVerificationError",
    "RenameResult",
    "RenameOperation",
    "BatchRenameResult",
    "RenameManager",
    "get_rename_manager",
    "rename",
    "rename_to",
    "add_prefix",
    "add_suffix",
    "replace_text",
]


