"""
RENIX File Operations
=====================

Central, safe filesystem operation layer.

Responsibilities:
- Create files
- Read files
- Write files
- Append files
- Create directories
- Copy files/directories
- Move files/directories
- Rename files/directories
- Delete files/directories
- Existence checks
- Size information
- Safe path handling
- Dry-run support
- Operation logging

Important:
This module is designed as the low-level filesystem layer.
Higher-level RENIX agents should use this module instead of
performing filesystem mutations directly.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# CONSTANTS
# ==============================================================

DEFAULT_ENCODING = "utf-8"
DEFAULT_BUFFER_SIZE = 1024 * 1024


# ==============================================================
# EXCEPTIONS
# ==============================================================


class FileOperationError(Exception):
    """Base exception for RENIX file operations."""


class InvalidPathError(FileOperationError):
    """Raised when a filesystem path is invalid."""


class DestinationExistsError(FileOperationError):
    """Raised when a destination already exists."""


class OperationCancelledError(FileOperationError):
    """Raised when an operation is cancelled."""


# ==============================================================
# DATA MODELS
# ==============================================================


@dataclass
class OperationResult:
    """Result returned by filesystem operations."""

    success: bool
    operation: str
    source: Optional[str] = None
    destination: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    dry_run: bool = False
    details: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "source": self.source,
            "destination": self.destination,
            "message": self.message,
            "error": self.error,
            "dry_run": self.dry_run,
            "details": self.details,
        }


# ==============================================================
# FILE OPERATIONS ENGINE
# ==============================================================


class FileOperations:
    """
    RENIX central filesystem operation engine.

    All mutation methods support dry_run mode and return
    OperationResult objects.
    """

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
        encoding: str = DEFAULT_ENCODING,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:

        self.enabled = enabled
        self.dry_run = dry_run
        self.encoding = encoding
        self.buffer_size = buffer_size

        logger.info(
            "RENIX FileOperations initialized."
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

    def is_enabled(self) -> bool:
        return self.enabled

    def _check(self) -> None:

        if not self.enabled:
            raise FileOperationError(
                "RENIX FileOperations is disabled."
            )

    # ==========================================================
    # PATH HELPERS
    # ==========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise InvalidPathError(
                "Path cannot be None."
            )

        try:
            return Path(path).expanduser()
        except (TypeError, ValueError) as exc:
            raise InvalidPathError(
                f"Invalid path: {path}"
            ) from exc

    @staticmethod
    def absolute_path(
        path: str | os.PathLike[str],
    ) -> Path:

        return (
            FileOperations.normalize_path(
                path
            ).resolve()
        )

    @staticmethod
    def ensure_parent_directory(
        path: Path,
    ) -> None:

        parent = path.parent

        if parent and not parent.exists():
            parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def exists(
        path: str | os.PathLike[str],
    ) -> bool:

        return Path(path).expanduser().exists()

    @staticmethod
    def is_file(
        path: str | os.PathLike[str],
    ) -> bool:

        return Path(path).expanduser().is_file()

    @staticmethod
    def is_directory(
        path: str | os.PathLike[str],
    ) -> bool:

        return Path(path).expanduser().is_dir()

    # ==========================================================
    # INTERNAL RESULT HELPERS
    # ==========================================================

    @staticmethod
    def _success(
        operation: str,
        source: Optional[Path] = None,
        destination: Optional[Path] = None,
        message: str = "",
        dry_run: bool = False,
        details: Optional[dict[str, Any]] = None,
    ) -> OperationResult:

        return OperationResult(
            success=True,
            operation=operation,
            source=(
                str(source)
                if source is not None
                else None
            ),
            destination=(
                str(destination)
                if destination is not None
                else None
            ),
            message=message,
            dry_run=dry_run,
            details=details or {},
        )

    @staticmethod
    def _failure(
        operation: str,
        error: Exception | str,
        source: Optional[Path] = None,
        destination: Optional[Path] = None,
        dry_run: bool = False,
    ) -> OperationResult:

        return OperationResult(
            success=False,
            operation=operation,
            source=(
                str(source)
                if source is not None
                else None
            ),
            destination=(
                str(destination)
                if destination is not None
                else None
            ),
            message="Operation failed.",
            error=str(error),
            dry_run=dry_run,
        )

    # ==========================================================
    # CREATE DIRECTORY
    # ==========================================================

    def create_directory(
        self,
        path: str | os.PathLike[str],
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> OperationResult:

        self._check()

        target = self.normalize_path(path)

        if self.dry_run:

            return self._success(
                "create_directory",
                destination=target,
                message=(
                    "Dry-run: directory would "
                    "be created."
                ),
                dry_run=True,
            )

        try:

            target.mkdir(
                parents=parents,
                exist_ok=exist_ok,
            )

            return self._success(
                "create_directory",
                destination=target,
                message="Directory created.",
            )

        except OSError as exc:

            logger.error(
                "Directory creation failed: %s",
                exc,
            )

            return self._failure(
                "create_directory",
                exc,
                destination=target,
            )

    # ==========================================================
    # CREATE EMPTY FILE
    # ==========================================================

    def create_file(
        self,
        path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        self._check()

        target = self.normalize_path(path)

        if target.exists() and not overwrite:

            return self._failure(
                "create_file",
                DestinationExistsError(
                    f"File already exists: {target}"
                ),
                destination=target,
            )

        if self.dry_run:

            return self._success(
                "create_file",
                destination=target,
                message=(
                    "Dry-run: file would "
                    "be created."
                ),
                dry_run=True,
            )

        try:

            self.ensure_parent_directory(
                target
            )

            if overwrite:

                target.write_bytes(
                    b""
                )

            else:

                target.touch(
                    exist_ok=False
                )

            return self._success(
                "create_file",
                destination=target,
                message="File created.",
            )

        except OSError as exc:

            return self._failure(
                "create_file",
                exc,
                destination=target,
            )

    # ==========================================================
    # READ TEXT
    # ==========================================================

    def read_text(
        self,
        path: str | os.PathLike[str],
        *,
        encoding: Optional[str] = None,
    ) -> str:

        self._check()

        target = self.normalize_path(path)

        if not target.exists():
            raise FileNotFoundError(
                str(target)
            )

        if not target.is_file():
            raise IsADirectoryError(
                str(target)
            )

        return target.read_text(
            encoding=encoding or self.encoding
        )

    # ==========================================================
    # WRITE TEXT
    # ==========================================================

    def write_text(
        self,
        path: str | os.PathLike[str],
        content: str,
        *,
        encoding: Optional[str] = None,
        overwrite: bool = True,
        atomic: bool = True,
    ) -> OperationResult:

        self._check()

        target = self.normalize_path(path)

        if (
            target.exists()
            and not overwrite
        ):

            return self._failure(
                "write_text",
                DestinationExistsError(
                    f"File already exists: {target}"
                ),
                destination=target,
            )

        if self.dry_run:

            return self._success(
                "write_text",
                destination=target,
                message=(
                    "Dry-run: text would "
                    "be written."
                ),
                dry_run=True,
                details={
                    "characters": len(content),
                    "bytes": len(
                        content.encode(
                            encoding
                            or self.encoding
                        )
                    ),
                },
            )

        try:

            self.ensure_parent_directory(
                target
            )

            selected_encoding = (
                encoding or self.encoding
            )

            if atomic:

                self._atomic_write_text(
                    target,
                    content,
                    selected_encoding,
                )

            else:

                target.write_text(
                    content,
                    encoding=selected_encoding,
                )

            return self._success(
                "write_text",
                destination=target,
                message="Text written.",
                details={
                    "characters": len(content),
                },
            )

        except OSError as exc:

            return self._failure(
                "write_text",
                exc,
                destination=target,
            )

    def _atomic_write_text(
        self,
        target: Path,
        content: str,
        encoding: str,
    ) -> None:

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
            text=True,
        )

        try:

            with os.fdopen(
                fd,
                "w",
                encoding=encoding,
            ) as file:

                file.write(content)
                file.flush()
                os.fsync(file.fileno())

            os.replace(
                temp_name,
                target,
            )

        except Exception:

            try:
                os.unlink(temp_name)
            except OSError:
                pass

            raise

    # ==========================================================
    # APPEND TEXT
    # ==========================================================

    def append_text(
        self,
        path: str | os.PathLike[str],
        content: str,
        *,
        encoding: Optional[str] = None,
    ) -> OperationResult:

        self._check()

        target = self.normalize_path(path)

        if self.dry_run:

            return self._success(
                "append_text",
                destination=target,
                message=(
                    "Dry-run: text would "
                    "be appended."
                ),
                dry_run=True,
            )

        try:

            self.ensure_parent_directory(
                target
            )

            with target.open(
                "a",
                encoding=(
                    encoding or self.encoding
                ),
            ) as file:

                file.write(content)

            return self._success(
                "append_text",
                destination=target,
                message="Text appended.",
            )

        except OSError as exc:

            return self._failure(
                "append_text",
                exc,
                destination=target,
            )

    # ==========================================================
    # READ BYTES
    # ==========================================================

    def read_bytes(
        self,
        path: str | os.PathLike[str],
    ) -> bytes:

        self._check()

        target = self.normalize_path(path)

        if not target.is_file():
            raise FileNotFoundError(
                str(target)
            )

        return target.read_bytes()

    # ==========================================================
    # WRITE BYTES
    # ==========================================================

    def write_bytes(
        self,
        path: str | os.PathLike[str],
        content: bytes,
        *,
        overwrite: bool = True,
    ) -> OperationResult:

        self._check()

        target = self.normalize_path(path)

        if (
            target.exists()
            and not overwrite
        ):

            return self._failure(
                "write_bytes",
                DestinationExistsError(
                    f"File already exists: {target}"
                ),
                destination=target,
            )

        if self.dry_run:

            return self._success(
                "write_bytes",
                destination=target,
                message=(
                    "Dry-run: bytes would "
                    "be written."
                ),
                dry_run=True,
                details={
                    "bytes": len(content),
                },
            )

        try:

            self.ensure_parent_directory(
                target
            )

            target.write_bytes(
                content
            )

            return self._success(
                "write_bytes",
                destination=target,
                message="Binary data written.",
            )

        except OSError as exc:

            return self._failure(
                "write_bytes",
                exc,
                destination=target,
            )

    # ==========================================================
    # COPY
    # ==========================================================

    def copy(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        self._check()

        source_path = self.normalize_path(
            source
        )

        destination_path = (
            self.normalize_path(
                destination
            )
        )

        if not source_path.exists():

            return self._failure(
                "copy",
                FileNotFoundError(
                    str(source_path)
                ),
                source=source_path,
                destination=destination_path,
            )

        if (
            destination_path.exists()
            and not overwrite
        ):

            return self._failure(
                "copy",
                DestinationExistsError(
                    f"Destination exists: "
                    f"{destination_path}"
                ),
                source=source_path,
                destination=destination_path,
            )

        if self.dry_run:

            return self._success(
                "copy",
                source=source_path,
                destination=destination_path,
                message=(
                    "Dry-run: item would "
                    "be copied."
                ),
                dry_run=True,
            )

        try:

            self.ensure_parent_directory(
                destination_path
            )

            if source_path.is_dir():

                if (
                    destination_path.exists()
                    and overwrite
                ):

                    shutil.copytree(
                        source_path,
                        destination_path,
                        dirs_exist_ok=True,
                    )

                else:

                    shutil.copytree(
                        source_path,
                        destination_path,
                    )

            else:

                shutil.copy2(
                    source_path,
                    destination_path,
                )

            return self._success(
                "copy",
                source=source_path,
                destination=destination_path,
                message="Item copied.",
            )

        except OSError as exc:

            return self._failure(
                "copy",
                exc,
                source=source_path,
                destination=destination_path,
            )

    # ==========================================================
    # MOVE
    # ==========================================================

    def move(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        self._check()

        source_path = self.normalize_path(
            source
        )

        destination_path = (
            self.normalize_path(
                destination
            )
        )

        if not source_path.exists():

            return self._failure(
                "move",
                FileNotFoundError(
                    str(source_path)
                ),
                source=source_path,
                destination=destination_path,
            )

        if (
            destination_path.exists()
            and not overwrite
        ):

            return self._failure(
                "move",
                DestinationExistsError(
                    f"Destination exists: "
                    f"{destination_path}"
                ),
                source=source_path,
                destination=destination_path,
            )

        if self.dry_run:

            return self._success(
                "move",
                source=source_path,
                destination=destination_path,
                message=(
                    "Dry-run: item would "
                    "be moved."
                ),
                dry_run=True,
            )

        try:

            self.ensure_parent_directory(
                destination_path
            )

            if (
                destination_path.exists()
                and overwrite
            ):

                self.delete(
                    destination_path,
                    recursive=True,
                )

            shutil.move(
                str(source_path),
                str(destination_path),
            )

            return self._success(
                "move",
                source=source_path,
                destination=destination_path,
                message="Item moved.",
            )

        except OSError as exc:

            return self._failure(
                "move",
                exc,
                source=source_path,
                destination=destination_path,
            )

    # ==========================================================
    # RENAME
    # ==========================================================

    def rename(
        self,
        source: str | os.PathLike[str],
        new_name: str,
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        self._check()

        source_path = self.normalize_path(
            source
        )

        if not source_path.exists():

            return self._failure(
                "rename",
                FileNotFoundError(
                    str(source_path)
                ),
                source=source_path,
            )

        if not new_name:
            return self._failure(
                "rename",
                InvalidPathError(
                    "New name cannot be empty."
                ),
                source=source_path,
            )

        destination_path = (
            source_path.parent / new_name
        )

        if (
            destination_path.exists()
            and not overwrite
        ):

            return self._failure(
                "rename",
                DestinationExistsError(
                    f"Destination exists: "
                    f"{destination_path}"
                ),
                source=source_path,
                destination=destination_path,
            )

        if self.dry_run:

            return self._success(
                "rename",
                source=source_path,
                destination=destination_path,
                message=(
                    "Dry-run: item would "
                    "be renamed."
                ),
                dry_run=True,
            )

        try:

            if (
                destination_path.exists()
                and overwrite
            ):

                self.delete(
                    destination_path,
                    recursive=True,
                )

            source_path.rename(
                destination_path
            )

            return self._success(
                "rename",
                source=source_path,
                destination=destination_path,
                message="Item renamed.",
            )

        except OSError as exc:

            return self._failure(
                "rename",
                exc,
                source=source_path,
                destination=destination_path,
            )

    # ==========================================================
    # DELETE
    # ==========================================================

    def delete(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> OperationResult:

        self._check()

        target = self.normalize_path(path)

        if not target.exists():

            return self._failure(
                "delete",
                FileNotFoundError(
                    str(target)
                ),
                source=target,
            )

        if self.dry_run:

            return self._success(
                "delete",
                source=target,
                message=(
                    "Dry-run: item would "
                    "be deleted."
                ),
                dry_run=True,
            )

        try:

            if target.is_dir():

                if not recursive:

                    return self._failure(
                        "delete",
                        FileOperationError(
                            "Directory deletion requires "
                            "recursive=True."
                        ),
                        source=target,
                    )

                shutil.rmtree(
                    target
                )

            else:

                target.unlink()

            return self._success(
                "delete",
                source=target,
                message="Item deleted.",
            )

        except OSError as exc:

            return self._failure(
                "delete",
                exc,
                source=target,
            )

    # ==========================================================
    # DIRECTORY LISTING
    # ==========================================================

    def list_directory(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> list[str]:

        self._check()

        target = self.normalize_path(path)

        if not target.is_dir():
            raise NotADirectoryError(
                str(target)
            )

        if not recursive:

            return [
                str(item)
                for item in target.iterdir()
            ]

        results = []

        for current_root, _, files in os.walk(
            target
        ):

            current_path = Path(
                current_root
            )

            for name in files:

                results.append(
                    str(
                        current_path / name
                    )
                )

        return results

    # ==========================================================
    # COPY FILE ONLY
    # ==========================================================

    def copy_file(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        source_path = self.normalize_path(
            source
        )

        if not source_path.is_file():

            return self._failure(
                "copy_file",
                FileOperationError(
                    "Source is not a file."
                ),
                source=source_path,
            )

        return self.copy(
            source_path,
            destination,
            overwrite=overwrite,
        )

    # ==========================================================
    # COPY DIRECTORY ONLY
    # ==========================================================

    def copy_directory(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        source_path = self.normalize_path(
            source
        )

        if not source_path.is_dir():

            return self._failure(
                "copy_directory",
                FileOperationError(
                    "Source is not a directory."
                ),
                source=source_path,
            )

        return self.copy(
            source_path,
            destination,
            overwrite=overwrite,
        )

    # ==========================================================
    # GET SIZE
    # ==========================================================

    def get_size(
        self,
        path: str | os.PathLike[str],
    ) -> int:

        self._check()

        target = self.normalize_path(path)

        if target.is_file():
            return target.stat().st_size

        if target.is_dir():

            total = 0

            for current_root, _, files in os.walk(
                target
            ):

                current_path = Path(
                    current_root
                )

                for name in files:

                    file_path = (
                        current_path / name
                    )

                    try:
                        total += (
                            file_path.stat().st_size
                        )
                    except OSError:
                        continue

            return total

        raise FileNotFoundError(
            str(target)
        )

    # ==========================================================
    # TEMPORARY FILE
    # ==========================================================

    def create_temp_file(
        self,
        *,
        suffix: str = "",
        prefix: str = "renix_",
        directory: Optional[
            str | os.PathLike[str]
        ] = None,
    ) -> Path:

        self._check()

        fd, path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=(
                str(directory)
                if directory is not None
                else None
            ),
        )

        os.close(fd)

        return Path(path)

    # ==========================================================
    # OPERATION DISPATCHER
    # ==========================================================

    def execute(
        self,
        operation: str,
        **kwargs: Any,
    ) -> Any:

        self._check()

        operation = operation.lower().strip()

        handlers = {
            "create_file": self.create_file,
            "create_directory": self.create_directory,
            "read_text": self.read_text,
            "write_text": self.write_text,
            "append_text": self.append_text,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "copy": self.copy,
            "move": self.move,
            "rename": self.rename,
            "delete": self.delete,
            "list_directory": self.list_directory,
            "copy_file": self.copy_file,
            "copy_directory": self.copy_directory,
            "get_size": self.get_size,
        }

        handler = handlers.get(
            operation
        )

        if handler is None:

            raise FileOperationError(
                f"Unknown file operation: "
                f"{operation}"
            )

        return handler(**kwargs)

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "encoding": self.encoding,
            "buffer_size": self.buffer_size,
            "operations": [
                "create_file",
                "create_directory",
                "read_text",
                "write_text",
                "append_text",
                "read_bytes",
                "write_bytes",
                "copy",
                "move",
                "rename",
                "delete",
                "list_directory",
                "copy_file",
                "copy_directory",
                "get_size",
            ],
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX FileOperations shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_file_operations: Optional[
    FileOperations
] = None


def get_file_operations() -> FileOperations:

    global _default_file_operations

    if _default_file_operations is None:

        _default_file_operations = (
            FileOperations()
        )

    return _default_file_operations


# ==============================================================
# MODULE-LEVEL HELPERS
# ==============================================================


def read_file(
    path: str | os.PathLike[str],
    encoding: str = DEFAULT_ENCODING,
) -> str:

    return (
        get_file_operations()
        .read_text(
            path,
            encoding=encoding,
        )
    )


def write_file(
    path: str | os.PathLike[str],
    content: str,
    encoding: str = DEFAULT_ENCODING,
) -> OperationResult:

    return (
        get_file_operations()
        .write_text(
            path,
            content,
            encoding=encoding,
        )
    )


def copy_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> OperationResult:

    return (
        get_file_operations()
        .copy_file(
            source,
            destination,
            overwrite=overwrite,
        )
    )


def move_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> OperationResult:

    return (
        get_file_operations()
        .move(
            source,
            destination,
            overwrite=overwrite,
        )
    )


def delete_file(
    path: str | os.PathLike[str],
) -> OperationResult:

    return (
        get_file_operations()
        .delete(
            path
        )
    )


# ==============================================================
# PUBLIC API
# ==============================================================

__all__ = [
    "FileOperationError",
    "InvalidPathError",
    "DestinationExistsError",
    "OperationCancelledError",
    "OperationResult",
    "FileOperations",
    "get_file_operations",
    "read_file",
    "write_file",
    "copy_file",
    "move_file",
    "delete_file",
]


