"""
RENIX Delete Manager
====================

Safe, high-level deletion service for RENIX.

Features:
- Delete files
- Delete directories
- Recursive deletion
- Trash/recycle-bin support when available
- Permanent deletion
- Dry-run mode
- Verification
- Batch deletion
- Preview
- Deletion history
- Safety checks
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class DeleteManagerError(Exception):
    """Base exception for deletion operations."""


class DeleteSourceNotFoundError(DeleteManagerError):
    """Raised when the target does not exist."""


class DeletePermissionError(DeleteManagerError):
    """Raised when deletion is denied."""


class DeleteSafetyError(DeleteManagerError):
    """Raised when a deletion is considered unsafe."""


class DeleteVerificationError(DeleteManagerError):
    """Raised when deletion verification fails."""


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class DeleteStats:
    files_deleted: int = 0
    directories_deleted: int = 0
    bytes_deleted: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_deleted": self.files_deleted,
            "directories_deleted": self.directories_deleted,
            "bytes_deleted": self.bytes_deleted,
            "failed": self.failed,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class DeleteResult:
    success: bool
    target: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    dry_run: bool = False
    permanent: bool = False
    sent_to_trash: bool = False
    verified: bool = False
    stats: DeleteStats = field(default_factory=DeleteStats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "target": self.target,
            "message": self.message,
            "error": self.error,
            "dry_run": self.dry_run,
            "permanent": self.permanent,
            "sent_to_trash": self.sent_to_trash,
            "verified": self.verified,
            "stats": self.stats.to_dict(),
        }


@dataclass
class DeleteOperation:
    target: str
    success: bool
    timestamp: float
    permanent: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "success": self.success,
            "timestamp": self.timestamp,
            "permanent": self.permanent,
            "message": self.message,
        }


# ============================================================
# DELETE MANAGER
# ============================================================


class DeleteManager:
    """
    RENIX deletion engine.

    By default, deletion is conservative.

    `permanent=False`
        Attempts to move the item to the operating system's
        recycle bin/trash.

    `permanent=True`
        Permanently removes the item.

    The manager never permanently deletes an item merely because
    a normal delete was requested.
    """

    PROTECTED_NAMES = {
        "",
        ".",
        "..",
    }

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
        allow_permanent: bool = False,
    ) -> None:

        self.enabled = enabled
        self.dry_run = dry_run
        self.allow_permanent = allow_permanent

        self._history: list[DeleteOperation] = []

        logger.info(
            "RENIX DeleteManager initialized."
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

    def set_permanent_delete(
        self,
        enabled: bool,
    ) -> None:

        self.allow_permanent = enabled

    def _check_enabled(self) -> None:

        if not self.enabled:
            raise DeleteManagerError(
                "RENIX DeleteManager is disabled."
            )

    # ========================================================
    # PATH HELPERS
    # ========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise DeleteManagerError(
                "Path cannot be None."
            )

        return Path(path).expanduser()

    @staticmethod
    def absolute_path(
        path: str | os.PathLike[str],
    ) -> Path:

        return (
            DeleteManager
            .normalize_path(path)
            .resolve()
        )

    # ========================================================
    # SIZE
    # ========================================================

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

                    file_path = (
                        Path(root) / filename
                    )

                    try:
                        total += file_path.stat().st_size
                    except OSError:
                        continue

            return total

        return 0

    @staticmethod
    def count_items(
        path: Path,
    ) -> tuple[int, int]:

        files = 0
        directories = 0

        if path.is_file():
            return 1, 0

        if not path.is_dir():
            return 0, 0

        for root, dirs, filenames in os.walk(path):

            directories += len(dirs)
            files += len(filenames)

        directories += 1

        return files, directories

    # ========================================================
    # SAFETY
    # ========================================================

    def _validate_target(
        self,
        target: Path,
    ) -> Optional[str]:

        if not target.exists() and not target.is_symlink():
            return "Target does not exist."

        try:
            resolved = target.resolve()
        except OSError:
            resolved = target.absolute()

        # Prevent obviously dangerous root deletion.
        if resolved == resolved.anchor:
            return "Deleting a filesystem root is not allowed."

        # Prevent deleting current working directory.
        try:
            cwd = Path.cwd().resolve()

            if resolved == cwd:
                return (
                    "Deleting the current working directory "
                    "is not allowed."
                )
        except OSError:
            pass

        # Prevent deleting parent of current working directory.
        try:
            cwd = Path.cwd().resolve()

            if cwd.is_relative_to(resolved):
                return (
                    "Target contains the current working "
                    "directory and is therefore protected."
                )
        except (OSError, ValueError):
            pass

        return None

    def _check_permanent_permission(
        self,
        permanent: bool,
    ) -> None:

        if permanent and not self.allow_permanent:

            raise DeleteSafetyError(
                "Permanent deletion is disabled. "
                "Enable allow_permanent=True first."
            )

    # ========================================================
    # HISTORY
    # ========================================================

    def _record(
        self,
        target: Path,
        success: bool,
        permanent: bool,
        message: str,
    ) -> None:

        self._history.append(
            DeleteOperation(
                target=str(target),
                success=success,
                timestamp=time.time(),
                permanent=permanent,
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
    # READ-ONLY PREVIEW
    # ========================================================

    def preview(
        self,
        target: str | os.PathLike[str],
        *,
        permanent: bool = False,
    ) -> dict[str, Any]:

        target_path = self.normalize_path(target)

        validation_error = self._validate_target(
            target_path
        )

        if validation_error:

            return {
                "valid": False,
                "target": str(target_path),
                "reason": validation_error,
            }

        files, directories = self.count_items(
            target_path
        )

        size = self.calculate_size(target_path)

        return {
            "valid": True,
            "target": str(target_path),
            "type": (
                "directory"
                if target_path.is_dir()
                else "file"
            ),
            "permanent": permanent,
            "files": files,
            "directories": directories,
            "size_bytes": size,
            "exists": target_path.exists(),
        }

    # ========================================================
    # DELETE FILE
    # ========================================================

    def delete_file(
        self,
        target: str | os.PathLike[str],
        *,
        permanent: bool = False,
        verify: bool = True,
    ) -> DeleteResult:

        return self.delete(
            target,
            permanent=permanent,
            verify=verify,
        )

    # ========================================================
    # DELETE DIRECTORY
    # ========================================================

    def delete_directory(
        self,
        target: str | os.PathLike[str],
        *,
        permanent: bool = False,
        recursive: bool = True,
        verify: bool = True,
    ) -> DeleteResult:

        target_path = self.normalize_path(target)

        if target_path.exists() and target_path.is_dir():

            if not recursive and any(
                target_path.iterdir()
            ):
                message = (
                    "Directory is not empty and "
                    "recursive deletion is disabled."
                )

                return DeleteResult(
                    success=False,
                    target=str(target_path),
                    message=message,
                    error=message,
                )

        return self.delete(
            target_path,
            permanent=permanent,
            verify=verify,
        )

    # ========================================================
    # CORE DELETE
    # ========================================================

    def delete(
        self,
        target: str | os.PathLike[str],
        *,
        permanent: bool = False,
        verify: bool = True,
    ) -> DeleteResult:

        self._check_enabled()
        self._check_permanent_permission(permanent)

        target_path = self.normalize_path(target)

        stats = DeleteStats()
        start = time.perf_counter()

        validation_error = self._validate_target(
            target_path
        )

        if validation_error:

            return DeleteResult(
                success=False,
                target=str(target_path),
                message=validation_error,
                error=validation_error,
                permanent=permanent,
                stats=stats,
            )

        files, directories = self.count_items(
            target_path
        )

        size = self.calculate_size(target_path)

        stats.files_deleted = files
        stats.directories_deleted = directories
        stats.bytes_deleted = size

        # ----------------------------------------------------
        # DRY RUN
        # ----------------------------------------------------

        if self.dry_run:

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return DeleteResult(
                success=True,
                target=str(target_path),
                message=(
                    "Dry-run: target would be deleted."
                ),
                dry_run=True,
                permanent=permanent,
                stats=stats,
            )

        # ----------------------------------------------------
        # PERMANENT DELETE
        # ----------------------------------------------------

        if permanent:

            try:

                self._permanent_delete(
                    target_path
                )

                verified = False

                if verify:
                    verified = not target_path.exists()

                    if not verified:
                        raise DeleteVerificationError(
                            "Permanent deletion "
                            "verification failed."
                        )

                stats.elapsed_seconds = (
                    time.perf_counter() - start
                )

                message = (
                    "Target permanently deleted."
                )

                result = DeleteResult(
                    success=True,
                    target=str(target_path),
                    message=message,
                    permanent=True,
                    verified=verified,
                    stats=stats,
                )

                self._record(
                    target_path,
                    True,
                    True,
                    message,
                )

                return result

            except (
                OSError,
                DeleteVerificationError,
            ) as exc:

                stats.failed = 1
                stats.elapsed_seconds = (
                    time.perf_counter() - start
                )

                message = (
                    "Permanent deletion failed."
                )

                result = DeleteResult(
                    success=False,
                    target=str(target_path),
                    message=message,
                    error=str(exc),
                    permanent=True,
                    stats=stats,
                )

                self._record(
                    target_path,
                    False,
                    True,
                    message,
                )

                return result

        # ----------------------------------------------------
        # TRASH / RECYCLE BIN
        # ----------------------------------------------------

        try:

            sent_to_trash = self._send_to_trash(
                target_path
            )

            verified = False

            if verify:

                verified = not target_path.exists()

                if not verified:
                    raise DeleteVerificationError(
                        "Trash operation could not be "
                        "verified."
                    )

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            message = (
                "Target moved to the recycle bin/trash."
            )

            result = DeleteResult(
                success=True,
                target=str(target_path),
                message=message,
                sent_to_trash=sent_to_trash,
                verified=verified,
                stats=stats,
            )

            self._record(
                target_path,
                True,
                False,
                message,
            )

            return result

        except (
            OSError,
            DeleteVerificationError,
            DeleteManagerError,
        ) as exc:

            stats.failed = 1
            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            message = "Deletion failed."

            result = DeleteResult(
                success=False,
                target=str(target_path),
                message=message,
                error=str(exc),
                stats=stats,
            )

            self._record(
                target_path,
                False,
                False,
                message,
            )

            return result

    # ========================================================
    # PERMANENT DELETE IMPLEMENTATION
    # ========================================================

    @staticmethod
    def _permanent_delete(
        target: Path,
    ) -> None:

        # Symlinks must never be recursively traversed.
        if target.is_symlink():

            target.unlink()
            return

        if target.is_file():

            try:
                target.unlink()
            except PermissionError:

                DeleteManager._make_writable(
                    target
                )

                target.unlink()

            return

        if target.is_dir():

            shutil.rmtree(
                target,
                onerror=DeleteManager._remove_error,
            )

            return

        raise DeleteManagerError(
            f"Unsupported target type: {target}"
        )

    # ========================================================
    # READ-ONLY / PERMISSION HANDLING
    # ========================================================

    @staticmethod
    def _make_writable(
        path: Path,
    ) -> None:

        try:

            current_mode = path.stat().st_mode

            path.chmod(
                current_mode
                | stat.S_IWRITE
            )

        except OSError as exc:

            raise DeletePermissionError(
                f"Unable to make '{path}' writable."
            ) from exc

    @staticmethod
    def _remove_error(
        function,
        path,
        exc_info,
    ) -> None:

        path_obj = Path(path)

        try:

            DeleteManager._make_writable(
                path_obj
            )

            function(
                path
            )

        except Exception as exc:

            raise DeletePermissionError(
                f"Unable to delete '{path}'."
            ) from exc

    # ========================================================
    # TRASH SUPPORT
    # ========================================================

    @staticmethod
    def _send_to_trash(
        target: Path,
    ) -> bool:

        # RENIX deliberately keeps the default implementation
        # dependency-free. If send2trash is installed, use it.
        try:

            import send2trash

            send2trash.send2trash(
                str(target)
            )

            return True

        except ImportError:

            # Without a trash library we refuse to silently
            # convert a safe delete into a permanent delete.
            raise DeleteManagerError(
                "Trash support is unavailable. "
                "Install 'send2trash' or explicitly enable "
                "permanent deletion."
            )

    # ========================================================
    # BATCH DELETE
    # ========================================================

    def delete_many(
        self,
        targets: Iterable[str | os.PathLike[str]],
        *,
        permanent: bool = False,
        verify: bool = True,
    ) -> list[DeleteResult]:

        self._check_enabled()

        results: list[DeleteResult] = []

        for target in targets:

            result = self.delete(
                target,
                permanent=permanent,
                verify=verify,
            )

            results.append(result)

        return results

    # ========================================================
    # DELETE EMPTY DIRECTORIES
    # ========================================================

    def delete_empty_directories(
        self,
        root: str | os.PathLike[str],
        *,
        include_root: bool = False,
        permanent: bool = False,
    ) -> list[DeleteResult]:

        self._check_enabled()

        root_path = self.normalize_path(root)

        if not root_path.is_dir():
            raise DeleteManagerError(
                "Root must be a directory."
            )

        directories = [
            Path(current_root)
            for current_root, dirs, _ in os.walk(
                root_path,
                topdown=False,
            )
            if not dirs
        ]

        # Rebuild list bottom-up so nested empty folders
        # are handled before their parents.
        all_directories: list[Path] = []

        for current_root, dirs, _ in os.walk(
            root_path,
            topdown=False,
        ):

            all_directories.append(
                Path(current_root)
            )

        results: list[DeleteResult] = []

        for directory in all_directories:

            if directory == root_path and not include_root:
                continue

            try:

                if not any(directory.iterdir()):

                    results.append(
                        self.delete_directory(
                            directory,
                            permanent=permanent,
                            recursive=False,
                        )
                    )

            except OSError:
                continue

        return results

    # ========================================================
    # RECYCLE BIN STATUS
    # ========================================================

    @staticmethod
    def is_trash_available() -> bool:

        try:
            import send2trash  # noqa: F401

            return True
        except ImportError:
            return False

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "allow_permanent": self.allow_permanent,
            "trash_available": (
                self.is_trash_available()
            ),
            "history_entries": len(self._history),
            "capabilities": [
                "delete_file",
                "delete_directory",
                "delete",
                "delete_many",
                "delete_empty_directories",
                "trash_support",
                "permanent_delete",
                "dry_run",
                "verification",
                "preview",
            ],
        }

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX DeleteManager shut down."
        )


# ============================================================
# SHARED INSTANCE
# ============================================================

_default_delete_manager: Optional[DeleteManager] = None


def get_delete_manager() -> DeleteManager:

    global _default_delete_manager

    if _default_delete_manager is None:
        _default_delete_manager = DeleteManager()

    return _default_delete_manager


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def delete(
    target: str | os.PathLike[str],
    *,
    permanent: bool = False,
) -> DeleteResult:

    return get_delete_manager().delete(
        target,
        permanent=permanent,
    )


def delete_file(
    target: str | os.PathLike[str],
    *,
    permanent: bool = False,
) -> DeleteResult:

    return get_delete_manager().delete_file(
        target,
        permanent=permanent,
    )


def delete_directory(
    target: str | os.PathLike[str],
    *,
    permanent: bool = False,
) -> DeleteResult:

    return get_delete_manager().delete_directory(
        target,
        permanent=permanent,
    )


def delete_many(
    targets: Iterable[str | os.PathLike[str]],
    *,
    permanent: bool = False,
) -> list[DeleteResult]:

    return get_delete_manager().delete_many(
        targets,
        permanent=permanent,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "DeleteManagerError",
    "DeleteSourceNotFoundError",
    "DeletePermissionError",
    "DeleteSafetyError",
    "DeleteVerificationError",
    "DeleteStats",
    "DeleteResult",
    "DeleteOperation",
    "DeleteManager",
    "get_delete_manager",
    "delete",
    "delete_file",
    "delete_directory",
    "delete_many",
]


