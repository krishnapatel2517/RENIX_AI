"""
RENIX Folder Manager
====================

High-level folder management layer for RENIX.

Responsibilities:
- Create folders
- Delete folders
- Rename folders
- Move folders
- Copy folders
- List folder contents
- Recursive traversal
- Search folders
- Calculate folder size
- Check folder information
- Safe dry-run mode
- Operation history

This module does not perform AI reasoning. It provides a clean
filesystem API that can be used by RENIX agents, automation,
file UI, and command routing.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .file_operations import (
    FileOperations,
    OperationResult,
    get_file_operations,
)

logger = logging.getLogger(__name__)


# ==============================================================
# CONSTANTS
# ==============================================================

DEFAULT_ENCODING = "utf-8"


# ==============================================================
# EXCEPTIONS
# ==============================================================


class FolderManagerError(Exception):
    """Base exception for folder management."""


class FolderNotFoundError(FolderManagerError):
    """Raised when a folder does not exist."""


class NotAFolderError(FolderManagerError):
    """Raised when a path is not a directory."""


class FolderAlreadyExistsError(FolderManagerError):
    """Raised when a folder already exists."""


# ==============================================================
# DATA MODELS
# ==============================================================


@dataclass
class FolderInfo:
    """Information about a folder."""

    path: str
    name: str
    exists: bool
    is_directory: bool
    parent: Optional[str]
    files: int = 0
    directories: int = 0
    total_size: int = 0
    total_size_human: str = "0 B"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "exists": self.exists,
            "is_directory": self.is_directory,
            "parent": self.parent,
            "files": self.files,
            "directories": self.directories,
            "total_size": self.total_size,
            "total_size_human": self.total_size_human,
        }


@dataclass
class FolderItem:
    """Represents an item inside a folder."""

    path: str
    name: str
    item_type: str
    size: int = 0
    size_human: str = "0 B"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "item_type": self.item_type,
            "size": self.size,
            "size_human": self.size_human,
        }


@dataclass
class FolderOperation:
    """History record for a folder operation."""

    operation: str
    source: Optional[str]
    destination: Optional[str]
    success: bool
    message: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "source": self.source,
            "destination": self.destination,
            "success": self.success,
            "message": self.message,
            "timestamp": self.timestamp,
        }


# ==============================================================
# FOLDER MANAGER
# ==============================================================


class FolderManager:
    """
    RENIX high-level folder management engine.
    """

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
        file_operations: Optional[
            FileOperations
        ] = None,
    ) -> None:

        self.enabled = enabled
        self.dry_run = dry_run

        self.file_operations = (
            file_operations
            or get_file_operations()
        )

        self._history: list[
            FolderOperation
        ] = []

        logger.info(
            "RENIX FolderManager initialized."
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
            raise FolderManagerError(
                "RENIX FolderManager is disabled."
            )

    # ==========================================================
    # PATH HELPERS
    # ==========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise FolderManagerError(
                "Folder path cannot be None."
            )

        return Path(path).expanduser()

    @staticmethod
    def absolute_path(
        path: str | os.PathLike[str],
    ) -> Path:

        return (
            FolderManager.normalize_path(
                path
            ).resolve()
        )

    def _require_folder(
        self,
        path: str | os.PathLike[str],
    ) -> Path:

        folder = self.normalize_path(path)

        if not folder.exists():
            raise FolderNotFoundError(
                str(folder)
            )

        if not folder.is_dir():
            raise NotAFolderError(
                str(folder)
            )

        return folder

    # ==========================================================
    # SIZE FORMAT
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
    # HISTORY
    # ==========================================================

    def _record(
        self,
        operation: str,
        source: Optional[Path],
        destination: Optional[Path],
        success: bool,
        message: str,
    ) -> None:

        import time

        self._history.append(
            FolderOperation(
                operation=operation,
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
                message=message,
                timestamp=time.time(),
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
    # CREATE
    # ==========================================================

    def create_folder(
        self,
        path: str | os.PathLike[str],
        *,
        parents: bool = True,
        exist_ok: bool = False,
    ) -> OperationResult:

        self._check()

        folder = self.normalize_path(path)

        if folder.exists():

            if folder.is_dir() and exist_ok:

                return OperationResult(
                    success=True,
                    operation="create_folder",
                    destination=str(folder),
                    message=(
                        "Folder already exists."
                    ),
                )

            result = OperationResult(
                success=False,
                operation="create_folder",
                destination=str(folder),
                message="Folder already exists.",
                error=str(
                    FolderAlreadyExistsError(
                        str(folder)
                    )
                ),
            )

            self._record(
                "create_folder",
                None,
                folder,
                False,
                result.message,
            )

            return result

        if self.dry_run:

            return OperationResult(
                success=True,
                operation="create_folder",
                destination=str(folder),
                message=(
                    "Dry-run: folder would "
                    "be created."
                ),
                dry_run=True,
            )

        try:

            folder.mkdir(
                parents=parents,
                exist_ok=exist_ok,
            )

            result = OperationResult(
                success=True,
                operation="create_folder",
                destination=str(folder),
                message="Folder created.",
            )

            self._record(
                "create_folder",
                None,
                folder,
                True,
                result.message,
            )

            return result

        except OSError as exc:

            result = OperationResult(
                success=False,
                operation="create_folder",
                destination=str(folder),
                message="Folder creation failed.",
                error=str(exc),
            )

            self._record(
                "create_folder",
                None,
                folder,
                False,
                result.message,
            )

            return result

    # ==========================================================
    # CREATE MULTIPLE FOLDERS
    # ==========================================================

    def create_folders(
        self,
        paths: list[
            str | os.PathLike[str]
        ],
        *,
        parents: bool = True,
    ) -> list[OperationResult]:

        self._check()

        results = []

        for path in paths:

            results.append(
                self.create_folder(
                    path,
                    parents=parents,
                )
            )

        return results

    # ==========================================================
    # DELETE
    # ==========================================================

    def delete_folder(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> OperationResult:

        self._check()

        folder = self.normalize_path(path)

        if not folder.exists():

            result = OperationResult(
                success=False,
                operation="delete_folder",
                source=str(folder),
                message="Folder not found.",
                error=str(
                    FolderNotFoundError(
                        str(folder)
                    )
                ),
            )

            self._record(
                "delete_folder",
                folder,
                None,
                False,
                result.message,
            )

            return result

        if not folder.is_dir():

            result = OperationResult(
                success=False,
                operation="delete_folder",
                source=str(folder),
                message="Path is not a folder.",
                error=str(
                    NotAFolderError(
                        str(folder)
                    )
                ),
            )

            return result

        if self.dry_run:

            return OperationResult(
                success=True,
                operation="delete_folder",
                source=str(folder),
                message=(
                    "Dry-run: folder would "
                    "be deleted."
                ),
                dry_run=True,
            )

        try:

            if recursive:

                shutil.rmtree(folder)

            else:

                folder.rmdir()

            result = OperationResult(
                success=True,
                operation="delete_folder",
                source=str(folder),
                message="Folder deleted.",
            )

            self._record(
                "delete_folder",
                folder,
                None,
                True,
                result.message,
            )

            return result

        except OSError as exc:

            result = OperationResult(
                success=False,
                operation="delete_folder",
                source=str(folder),
                message=(
                    "Folder deletion failed."
                ),
                error=str(exc),
            )

            self._record(
                "delete_folder",
                folder,
                None,
                False,
                result.message,
            )

            return result

    # ==========================================================
    # RENAME
    # ==========================================================

    def rename_folder(
        self,
        path: str | os.PathLike[str],
        new_name: str,
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        self._check()

        folder = self._require_folder(
            path
        )

        if not new_name:
            raise FolderManagerError(
                "New folder name cannot be empty."
            )

        if (
            Path(new_name).name
            != new_name
        ):
            raise FolderManagerError(
                "new_name must contain only "
                "a folder name, not a path."
            )

        destination = (
            folder.parent / new_name
        )

        if (
            destination.exists()
            and not overwrite
        ):

            return OperationResult(
                success=False,
                operation="rename_folder",
                source=str(folder),
                destination=str(
                    destination
                ),
                message=(
                    "Destination folder already exists."
                ),
                error=str(
                    FolderAlreadyExistsError(
                        str(destination)
                    )
                ),
            )

        if self.dry_run:

            return OperationResult(
                success=True,
                operation="rename_folder",
                source=str(folder),
                destination=str(
                    destination
                ),
                message=(
                    "Dry-run: folder would "
                    "be renamed."
                ),
                dry_run=True,
            )

        try:

            if (
                destination.exists()
                and overwrite
            ):

                if destination.is_dir():
                    shutil.rmtree(
                        destination
                    )
                else:
                    destination.unlink()

            folder.rename(
                destination
            )

            result = OperationResult(
                success=True,
                operation="rename_folder",
                source=str(folder),
                destination=str(
                    destination
                ),
                message="Folder renamed.",
            )

            self._record(
                "rename_folder",
                folder,
                destination,
                True,
                result.message,
            )

            return result

        except OSError as exc:

            return OperationResult(
                success=False,
                operation="rename_folder",
                source=str(folder),
                destination=str(
                    destination
                ),
                message=(
                    "Folder rename failed."
                ),
                error=str(exc),
            )

    # ==========================================================
    # COPY
    # ==========================================================

    def copy_folder(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        self._check()

        source_folder = self._require_folder(
            source
        )

        destination_folder = (
            self.normalize_path(
                destination
            )
        )

        if (
            destination_folder.exists()
            and not overwrite
        ):

            return OperationResult(
                success=False,
                operation="copy_folder",
                source=str(
                    source_folder
                ),
                destination=str(
                    destination_folder
                ),
                message=(
                    "Destination already exists."
                ),
                error=str(
                    FolderAlreadyExistsError(
                        str(destination_folder)
                    )
                ),
            )

        if self.dry_run:

            return OperationResult(
                success=True,
                operation="copy_folder",
                source=str(
                    source_folder
                ),
                destination=str(
                    destination_folder
                ),
                message=(
                    "Dry-run: folder would "
                    "be copied."
                ),
                dry_run=True,
            )

        try:

            shutil.copytree(
                source_folder,
                destination_folder,
                dirs_exist_ok=overwrite,
            )

            result = OperationResult(
                success=True,
                operation="copy_folder",
                source=str(
                    source_folder
                ),
                destination=str(
                    destination_folder
                ),
                message="Folder copied.",
            )

            self._record(
                "copy_folder",
                source_folder,
                destination_folder,
                True,
                result.message,
            )

            return result

        except OSError as exc:

            return OperationResult(
                success=False,
                operation="copy_folder",
                source=str(
                    source_folder
                ),
                destination=str(
                    destination_folder
                ),
                message=(
                    "Folder copy failed."
                ),
                error=str(exc),
            )

    # ==========================================================
    # MOVE
    # ==========================================================

    def move_folder(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> OperationResult:

        self._check()

        source_folder = self._require_folder(
            source
        )

        destination_folder = (
            self.normalize_path(
                destination
            )
        )

        if (
            destination_folder.exists()
            and not overwrite
        ):

            return OperationResult(
                success=False,
                operation="move_folder",
                source=str(
                    source_folder
                ),
                destination=str(
                    destination_folder
                ),
                message=(
                    "Destination already exists."
                ),
                error=str(
                    FolderAlreadyExistsError(
                        str(destination_folder)
                    )
                ),
            )

        if self.dry_run:

            return OperationResult(
                success=True,
                operation="move_folder",
                source=str(
                    source_folder
                ),
                destination=str(
                    destination_folder
                ),
                message=(
                    "Dry-run: folder would "
                    "be moved."
                ),
                dry_run=True,
            )

        try:

            if (
                destination_folder.exists()
                and overwrite
            ):

                shutil.rmtree(
                    destination_folder
                )

            destination_folder.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.move(
                str(source_folder),
                str(destination_folder),
            )

            result = OperationResult(
                success=True,
                operation="move_folder",
                source=str(
                    source_folder
                ),
                destination=str(
                    destination_folder
                ),
                message="Folder moved.",
            )

            self._record(
                "move_folder",
                source_folder,
                destination_folder,
                True,
                result.message,
            )

            return result

        except OSError as exc:

            return OperationResult(
                success=False,
                operation="move_folder",
                source=str(
                    source_folder
                ),
                destination=str(
                    destination_folder
                ),
                message=(
                    "Folder move failed."
                ),
                error=str(exc),
            )

    # ==========================================================
    # LIST CONTENTS
    # ==========================================================

    def list_contents(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
        include_files: bool = True,
        include_directories: bool = True,
    ) -> list[FolderItem]:

        self._check()

        folder = self._require_folder(
            path
        )

        results: list[FolderItem] = []

        if recursive:

            iterator = folder.rglob("*")

        else:

            iterator = folder.iterdir()

        for item in iterator:

            try:

                if item.is_dir():

                    if not include_directories:
                        continue

                    results.append(
                        FolderItem(
                            path=str(
                                item.resolve()
                            ),
                            name=item.name,
                            item_type="directory",
                        )
                    )

                elif item.is_file():

                    if not include_files:
                        continue

                    size = item.stat().st_size

                    results.append(
                        FolderItem(
                            path=str(
                                item.resolve()
                            ),
                            name=item.name,
                            item_type="file",
                            size=size,
                            size_human=(
                                self.format_size(
                                    size
                                )
                            ),
                        )
                    )

            except OSError:

                continue

        return results

    # ==========================================================
    # SUBDIRECTORIES
    # ==========================================================

    def list_subdirectories(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> list[str]:

        folder = self._require_folder(
            path
        )

        iterator = (
            folder.rglob("*")
            if recursive
            else folder.iterdir()
        )

        return [
            str(item.resolve())
            for item in iterator
            if item.is_dir()
        ]

    # ==========================================================
    # FILES
    # ==========================================================

    def list_files(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> list[str]:

        folder = self._require_folder(
            path
        )

        iterator = (
            folder.rglob("*")
            if recursive
            else folder.iterdir()
        )

        return [
            str(item.resolve())
            for item in iterator
            if item.is_file()
        ]

    # ==========================================================
    # SEARCH FOLDERS
    # ==========================================================

    def search_folders(
        self,
        root: str | os.PathLike[str],
        query: str,
        *,
        case_sensitive: bool = False,
    ) -> list[str]:

        self._check()

        root_folder = self._require_folder(
            root
        )

        if not query:
            return []

        search_query = (
            query
            if case_sensitive
            else query.lower()
        )

        results = []

        for folder in root_folder.rglob("*"):

            if not folder.is_dir():
                continue

            name = (
                folder.name
                if case_sensitive
                else folder.name.lower()
            )

            if search_query in name:
                results.append(
                    str(folder.resolve())
                )

        return results

    # ==========================================================
    # FOLDER SIZE
    # ==========================================================

    def get_folder_size(
        self,
        path: str | os.PathLike[str],
    ) -> int:

        self._check()

        folder = self._require_folder(
            path
        )

        total = 0

        for current_root, _, files in os.walk(
            folder
        ):

            current_path = Path(
                current_root
            )

            for filename in files:

                file_path = (
                    current_path / filename
                )

                try:
                    total += (
                        file_path.stat().st_size
                    )
                except OSError:
                    continue

        return total

    def get_folder_size_human(
        self,
        path: str | os.PathLike[str],
    ) -> str:

        return self.format_size(
            self.get_folder_size(path)
        )

    # ==========================================================
    # FOLDER INFORMATION
    # ==========================================================

    def get_folder_info(
        self,
        path: str | os.PathLike[str],
    ) -> FolderInfo:

        self._check()

        folder = self.normalize_path(
            path
        )

        if not folder.exists():

            return FolderInfo(
                path=str(
                    folder.resolve()
                ),
                name=folder.name,
                exists=False,
                is_directory=False,
                parent=(
                    str(
                        folder.parent.resolve()
                    )
                    if folder.parent
                    else None
                ),
            )

        if not folder.is_dir():

            return FolderInfo(
                path=str(
                    folder.resolve()
                ),
                name=folder.name,
                exists=True,
                is_directory=False,
                parent=str(
                    folder.parent.resolve()
                ),
            )

        file_count = 0
        directory_count = 0
        total_size = 0

        for current_root, directories, files in os.walk(
            folder
        ):

            directory_count += len(
                directories
            )

            file_count += len(files)

            current_path = Path(
                current_root
            )

            for filename in files:

                file_path = (
                    current_path / filename
                )

                try:
                    total_size += (
                        file_path.stat().st_size
                    )
                except OSError:
                    continue

        return FolderInfo(
            path=str(
                folder.resolve()
            ),
            name=folder.name,
            exists=True,
            is_directory=True,
            parent=str(
                folder.parent.resolve()
            ),
            files=file_count,
            directories=directory_count,
            total_size=total_size,
            total_size_human=(
                self.format_size(
                    total_size
                )
            ),
        )

    # ==========================================================
    # EMPTY CHECK
    # ==========================================================

    def is_empty(
        self,
        path: str | os.PathLike[str],
    ) -> bool:

        folder = self._require_folder(
            path
        )

        try:
            next(folder.iterdir())
            return False
        except StopIteration:
            return True

    # ==========================================================
    # COUNT
    # ==========================================================

    def count_items(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> dict[str, int]:

        contents = self.list_contents(
            path,
            recursive=recursive,
        )

        return {
            "files": sum(
                item.item_type == "file"
                for item in contents
            ),
            "directories": sum(
                item.item_type == "directory"
                for item in contents
            ),
            "total": len(contents),
        }

    # ==========================================================
    # EXISTS
    # ==========================================================

    def folder_exists(
        self,
        path: str | os.PathLike[str],
    ) -> bool:

        folder = self.normalize_path(
            path
        )

        return (
            folder.exists()
            and folder.is_dir()
        )

    # ==========================================================
    # PARENT
    # ==========================================================

    def get_parent(
        self,
        path: str | os.PathLike[str],
    ) -> Optional[str]:

        folder = self.normalize_path(
            path
        )

        if folder.parent == folder:
            return None

        return str(
            folder.parent.resolve()
        )

    # ==========================================================
    # CHILD PATH
    # ==========================================================

    def child_path(
        self,
        folder: str | os.PathLike[str],
        child_name: str,
    ) -> str:

        parent = self._require_folder(
            folder
        )

        if not child_name:
            raise FolderManagerError(
                "Child name cannot be empty."
            )

        return str(
            (parent / child_name).resolve()
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "history_entries": len(
                self._history
            ),
            "capabilities": [
                "create_folder",
                "create_folders",
                "delete_folder",
                "rename_folder",
                "copy_folder",
                "move_folder",
                "list_contents",
                "list_files",
                "list_subdirectories",
                "search_folders",
                "get_folder_size",
                "get_folder_info",
                "is_empty",
                "count_items",
                "folder_exists",
                "get_parent",
                "child_path",
            ],
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX FolderManager shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_folder_manager: Optional[
    FolderManager
] = None


def get_folder_manager() -> FolderManager:

    global _default_folder_manager

    if _default_folder_manager is None:

        _default_folder_manager = (
            FolderManager()
        )

    return _default_folder_manager


# ==============================================================
# MODULE-LEVEL HELPERS
# ==============================================================


def create_folder(
    path: str | os.PathLike[str],
    parents: bool = True,
) -> OperationResult:

    return (
        get_folder_manager()
        .create_folder(
            path,
            parents=parents,
        )
    )


def delete_folder(
    path: str | os.PathLike[str],
    recursive: bool = False,
) -> OperationResult:

    return (
        get_folder_manager()
        .delete_folder(
            path,
            recursive=recursive,
        )
    )


def rename_folder(
    path: str | os.PathLike[str],
    new_name: str,
) -> OperationResult:

    return (
        get_folder_manager()
        .rename_folder(
            path,
            new_name,
        )
    )


def copy_folder(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> OperationResult:

    return (
        get_folder_manager()
        .copy_folder(
            source,
            destination,
            overwrite=overwrite,
        )
    )


def move_folder(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    overwrite: bool = False,
) -> OperationResult:

    return (
        get_folder_manager()
        .move_folder(
            source,
            destination,
            overwrite=overwrite,
        )
    )


# ==============================================================
# PUBLIC API
# ==============================================================

__all__ = [
    "FolderManagerError",
    "FolderNotFoundError",
    "NotAFolderError",
    "FolderAlreadyExistsError",
    "FolderInfo",
    "FolderItem",
    "FolderOperation",
    "FolderManager",
    "get_folder_manager",
    "create_folder",
    "delete_folder",
    "rename_folder",
    "copy_folder",
    "move_folder",
]


