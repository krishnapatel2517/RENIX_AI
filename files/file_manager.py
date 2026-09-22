"""
RENIX File Manager
==================

Central file-management service for RENIX.

Responsibilities:
- Inspect files and folders
- Create files and directories
- Read/write text files
- Copy/move/rename/delete files
- Calculate file sizes
- Search basic filesystem paths
- Create safe temporary files
- Provide metadata used by other RENIX modules

This module is intentionally filesystem-focused.
Higher-level semantic search, classification, preview,
archive handling, and permissions belong to their
dedicated modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# DATA MODELS
# ==============================================================


@dataclass
class FileInfo:
    """
    Standard RENIX representation of a filesystem item.
    """

    path: str
    name: str
    extension: str
    size: int
    is_file: bool
    is_directory: bool
    exists: bool
    readable: bool
    writable: bool
    created_at: Optional[float]
    modified_at: Optional[float]
    accessed_at: Optional[float]
    mime_type: Optional[str]

    @property
    def size_mb(self) -> float:
        return self.size / (1024 * 1024)

    @property
    def size_kb(self) -> float:
        return self.size / 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================
# FILE MANAGER
# ==============================================================


class FileManager:
    """
    Main filesystem manager used by RENIX.

    Example:

        manager = FileManager()

        info = manager.get_info("example.txt")

        manager.write_text(
            "example.txt",
            "Hello RENIX!"
        )

        manager.copy(
            "example.txt",
            "backup/example.txt"
        )
    """

    def __init__(
        self,
        enabled: bool = True,
        allow_delete: bool = True,
        allow_write: bool = True,
        allow_overwrite: bool = False,
    ) -> None:

        self.enabled = enabled
        self.allow_delete = allow_delete
        self.allow_write = allow_write
        self.allow_overwrite = allow_overwrite

        self._initialize()

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def _initialize(self) -> None:

        logger.info(
            "RENIX FileManager initialized."
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
                "RENIX FileManager is disabled."
            )

    def _check_write(self) -> None:

        self._check()

        if not self.allow_write:

            raise PermissionError(
                "RENIX file writing is disabled."
            )

    def _check_delete(self) -> None:

        self._check()

        if not self.allow_delete:

            raise PermissionError(
                "RENIX file deletion is disabled."
            )

    # ==========================================================
    # PATH HELPERS
    # ==========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        return Path(path).expanduser().resolve()

    @staticmethod
    def path_exists(
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
    # FILE INFORMATION
    # ==========================================================

    def get_info(
        self,
        path: str | os.PathLike[str],
    ) -> FileInfo:

        self._check()

        file_path = self.normalize_path(path)

        exists = file_path.exists()

        if not exists:

            return FileInfo(
                path=str(file_path),
                name=file_path.name,
                extension=file_path.suffix,
                size=0,
                is_file=False,
                is_directory=False,
                exists=False,
                readable=False,
                writable=False,
                created_at=None,
                modified_at=None,
                accessed_at=None,
                mime_type=None,
            )

        try:

            stat = file_path.stat()

            size = (
                stat.st_size
                if file_path.is_file()
                else 0
            )

            created_at = getattr(
                stat,
                "st_ctime",
                None,
            )

            modified_at = getattr(
                stat,
                "st_mtime",
                None,
            )

            accessed_at = getattr(
                stat,
                "st_atime",
                None,
            )

        except OSError:

            size = 0
            created_at = None
            modified_at = None
            accessed_at = None

        try:
            readable = os.access(
                file_path,
                os.R_OK,
            )
        except Exception:
            readable = False

        try:
            writable = os.access(
                file_path,
                os.W_OK,
            )
        except Exception:
            writable = False

        mime_type, _ = mimetypes.guess_type(
            str(file_path)
        )

        return FileInfo(
            path=str(file_path),
            name=file_path.name,
            extension=file_path.suffix,
            size=size,
            is_file=file_path.is_file(),
            is_directory=file_path.is_dir(),
            exists=True,
            readable=readable,
            writable=writable,
            created_at=created_at,
            modified_at=modified_at,
            accessed_at=accessed_at,
            mime_type=mime_type,
        )

    # ==========================================================
    # DIRECTORY OPERATIONS
    # ==========================================================

    def create_directory(
        self,
        path: str | os.PathLike[str],
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> str:

        self._check_write()

        directory = self.normalize_path(path)

        directory.mkdir(
            parents=parents,
            exist_ok=exist_ok,
        )

        return str(directory)

    def list_directory(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
        include_files: bool = True,
        include_directories: bool = True,
    ) -> list[dict[str, Any]]:

        self._check()

        directory = self.normalize_path(path)

        if not directory.exists():

            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        if not directory.is_dir():

            raise NotADirectoryError(
                str(directory)
            )

        iterator: Iterable[Path]

        if recursive:

            iterator = directory.rglob("*")

        else:

            iterator = directory.iterdir()

        results: list[
            dict[str, Any]
        ] = []

        for item in iterator:

            if (
                item.is_file()
                and not include_files
            ):
                continue

            if (
                item.is_dir()
                and not include_directories
            ):
                continue

            try:

                results.append(
                    self.get_info(
                        item
                    ).to_dict()
                )

            except OSError as exc:

                logger.debug(
                    "Unable to inspect %s: %s",
                    item,
                    exc,
                )

        return results

    # ==========================================================
    # FILE CREATION
    # ==========================================================

    def create_file(
        self,
        path: str | os.PathLike[str],
        *,
        overwrite: Optional[bool] = None,
    ) -> str:

        self._check_write()

        file_path = self.normalize_path(path)

        if overwrite is None:
            overwrite = self.allow_overwrite

        if (
            file_path.exists()
            and not overwrite
        ):

            raise FileExistsError(
                f"File already exists: {file_path}"
            )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if overwrite:

            file_path.write_bytes(
                b""
            )

        else:

            file_path.touch(
                exist_ok=False
            )

        return str(file_path)

    # ==========================================================
    # TEXT READ
    # ==========================================================

    def read_text(
        self,
        path: str | os.PathLike[str],
        *,
        encoding: str = "utf-8",
    ) -> str:

        self._check()

        file_path = self.normalize_path(path)

        if not file_path.exists():

            raise FileNotFoundError(
                str(file_path)
            )

        if not file_path.is_file():

            raise IsADirectoryError(
                str(file_path)
            )

        return file_path.read_text(
            encoding=encoding
        )

    # ==========================================================
    # TEXT WRITE
    # ==========================================================

    def write_text(
        self,
        path: str | os.PathLike[str],
        content: str,
        *,
        encoding: str = "utf-8",
        overwrite: Optional[bool] = None,
    ) -> str:

        self._check_write()

        file_path = self.normalize_path(path)

        if overwrite is None:
            overwrite = self.allow_overwrite

        if (
            file_path.exists()
            and not overwrite
        ):

            raise FileExistsError(
                f"File already exists: {file_path}"
            )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_text(
            content,
            encoding=encoding,
        )

        return str(file_path)

    # ==========================================================
    # BYTE READ
    # ==========================================================

    def read_bytes(
        self,
        path: str | os.PathLike[str],
    ) -> bytes:

        self._check()

        file_path = self.normalize_path(path)

        if not file_path.exists():

            raise FileNotFoundError(
                str(file_path)
            )

        return file_path.read_bytes()

    # ==========================================================
    # BYTE WRITE
    # ==========================================================

    def write_bytes(
        self,
        path: str | os.PathLike[str],
        content: bytes,
        *,
        overwrite: Optional[bool] = None,
    ) -> str:

        self._check_write()

        file_path = self.normalize_path(path)

        if overwrite is None:
            overwrite = self.allow_overwrite

        if (
            file_path.exists()
            and not overwrite
        ):

            raise FileExistsError(
                f"File already exists: {file_path}"
            )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_bytes(
            content
        )

        return str(file_path)

    # ==========================================================
    # JSON
    # ==========================================================

    def read_json(
        self,
        path: str | os.PathLike[str],
        *,
        encoding: str = "utf-8",
    ) -> Any:

        content = self.read_text(
            path,
            encoding=encoding,
        )

        return json.loads(content)

    def write_json(
        self,
        path: str | os.PathLike[str],
        data: Any,
        *,
        encoding: str = "utf-8",
        overwrite: Optional[bool] = None,
        indent: int = 4,
    ) -> str:

        content = json.dumps(
            data,
            indent=indent,
            ensure_ascii=False,
        )

        return self.write_text(
            path,
            content,
            encoding=encoding,
            overwrite=overwrite,
        )

    # ==========================================================
    # COPY
    # ==========================================================

    def copy(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: Optional[bool] = None,
    ) -> str:

        self._check_write()

        source_path = self.normalize_path(
            source
        )

        destination_path = (
            self.normalize_path(
                destination
            )
        )

        if not source_path.exists():

            raise FileNotFoundError(
                str(source_path)
            )

        if overwrite is None:
            overwrite = self.allow_overwrite

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            destination_path.exists()
            and not overwrite
        ):

            raise FileExistsError(
                f"Destination already exists: "
                f"{destination_path}"
            )

        if source_path.is_dir():

            if destination_path.exists():

                if not overwrite:

                    raise FileExistsError(
                        str(destination_path)
                    )

                shutil.rmtree(
                    destination_path
                )

            shutil.copytree(
                source_path,
                destination_path,
            )

        else:

            shutil.copy2(
                source_path,
                destination_path,
            )

        return str(destination_path)

    # ==========================================================
    # MOVE
    # ==========================================================

    def move(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: Optional[bool] = None,
    ) -> str:

        self._check_write()

        source_path = self.normalize_path(
            source
        )

        destination_path = (
            self.normalize_path(
                destination
            )
        )

        if not source_path.exists():

            raise FileNotFoundError(
                str(source_path)
            )

        if overwrite is None:
            overwrite = self.allow_overwrite

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            destination_path.exists()
            and not overwrite
        ):

            raise FileExistsError(
                f"Destination already exists: "
                f"{destination_path}"
            )

        if (
            destination_path.exists()
            and overwrite
        ):

            if destination_path.is_dir():

                shutil.rmtree(
                    destination_path
                )

            else:

                destination_path.unlink()

        result = shutil.move(
            str(source_path),
            str(destination_path),
        )

        return str(
            Path(result).resolve()
        )

    # ==========================================================
    # RENAME
    # ==========================================================

    def rename(
        self,
        path: str | os.PathLike[str],
        new_name: str,
        *,
        overwrite: Optional[bool] = None,
    ) -> str:

        self._check_write()

        source = self.normalize_path(
            path
        )

        if not source.exists():

            raise FileNotFoundError(
                str(source)
            )

        if not new_name.strip():

            raise ValueError(
                "New name cannot be empty."
            )

        if (
            Path(new_name).name
            != new_name
        ):

            raise ValueError(
                "new_name must contain only "
                "the new file/folder name."
            )

        destination = (
            source.parent / new_name
        )

        if overwrite is None:
            overwrite = self.allow_overwrite

        if (
            destination.exists()
            and not overwrite
        ):

            raise FileExistsError(
                str(destination)
            )

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

        source.rename(
            destination
        )

        return str(
            destination.resolve()
        )

    # ==========================================================
    # DELETE
    # ==========================================================

    def delete(
        self,
        path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> bool:

        self._check_delete()

        target = self.normalize_path(
            path
        )

        if not target.exists():

            return False

        if target.is_dir():

            if not recursive:

                # rmdir only works for empty directories.
                target.rmdir()

            else:

                shutil.rmtree(
                    target
                )

        else:

            target.unlink()

        return True

    # ==========================================================
    # FILE SIZE
    # ==========================================================

    def get_size(
        self,
        path: str | os.PathLike[str],
    ) -> int:

        self._check()

        target = self.normalize_path(
            path
        )

        if not target.exists():

            raise FileNotFoundError(
                str(target)
            )

        if target.is_file():

            return target.stat().st_size

        total = 0

        for item in target.rglob("*"):

            if item.is_file():

                try:
                    total += item.stat().st_size
                except OSError:
                    pass

        return total

    # ==========================================================
    # HASH
    # ==========================================================

    def calculate_hash(
        self,
        path: str | os.PathLike[str],
        *,
        algorithm: str = "sha256",
        chunk_size: int = 1024 * 1024,
    ) -> str:

        self._check()

        file_path = self.normalize_path(
            path
        )

        if not file_path.is_file():

            raise ValueError(
                "Hash calculation requires a file."
            )

        if chunk_size <= 0:

            raise ValueError(
                "chunk_size must be positive."
            )

        try:

            digest = hashlib.new(
                algorithm
            )

        except ValueError as exc:

            raise ValueError(
                f"Unsupported hash algorithm: "
                f"{algorithm}"
            ) from exc

        with file_path.open(
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    chunk_size
                )

                if not chunk:
                    break

                digest.update(
                    chunk
                )

        return digest.hexdigest()

    # ==========================================================
    # TEMPORARY FILE
    # ==========================================================

    def create_temp_file(
        self,
        *,
        suffix: str = "",
        prefix: str = "renix_",
        content: Optional[str] = None,
    ) -> str:

        self._check_write()

        fd, path = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
        )

        try:

            if content is not None:

                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8",
                ) as file:

                    file.write(
                        content
                    )

            else:

                os.close(fd)

        except Exception:

            try:
                os.close(fd)
            except OSError:
                pass

            try:
                os.unlink(path)
            except OSError:
                pass

            raise

        return path

    # ==========================================================
    # TEMPORARY DIRECTORY
    # ==========================================================

    def create_temp_directory(
        self,
        *,
        prefix: str = "renix_",
    ) -> str:

        self._check_write()

        return tempfile.mkdtemp(
            prefix=prefix
        )

    # ==========================================================
    # FIND
    # ==========================================================

    def find(
        self,
        root: str | os.PathLike[str],
        pattern: str = "*",
        *,
        files_only: bool = False,
        directories_only: bool = False,
    ) -> list[str]:

        self._check()

        root_path = self.normalize_path(
            root
        )

        if not root_path.exists():

            raise FileNotFoundError(
                str(root_path)
            )

        if not root_path.is_dir():

            raise NotADirectoryError(
                str(root_path)
            )

        results: list[str] = []

        for item in root_path.rglob(
            pattern
        ):

            if files_only and not item.is_file():
                continue

            if (
                directories_only
                and not item.is_dir()
            ):
                continue

            results.append(
                str(item)
            )

        return results

    # ==========================================================
    # EXTENSION SEARCH
    # ==========================================================

    def find_by_extension(
        self,
        root: str | os.PathLike[str],
        extension: str,
    ) -> list[str]:

        self._check()

        extension = extension.strip()

        if extension and not extension.startswith("."):

            extension = "." + extension

        pattern = (
            f"*{extension}"
        )

        return self.find(
            root,
            pattern,
            files_only=True,
        )

    # ==========================================================
    # RECENT FILES
    # ==========================================================

    def get_recent_files(
        self,
        root: str | os.PathLike[str],
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        self._check()

        files = self.find(
            root,
            "*",
            files_only=True,
        )

        results = []

        for path in files:

            try:

                info = self.get_info(
                    path
                )

                results.append(
                    info.to_dict()
                )

            except OSError:

                continue

        results.sort(
            key=lambda item: (
                item.get(
                    "modified_at"
                )
                or 0
            ),
            reverse=True,
        )

        return results[
            :max(1, int(limit))
        ]

    # ==========================================================
    # EXTENSION
    # ==========================================================

    @staticmethod
    def get_extension(
        path: str | os.PathLike[str],
    ) -> str:

        return Path(path).suffix

    # ==========================================================
    # MIME TYPE
    # ==========================================================

    @staticmethod
    def get_mime_type(
        path: str | os.PathLike[str],
    ) -> Optional[str]:

        mime_type, _ = mimetypes.guess_type(
            str(path)
        )

        return mime_type

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "allow_delete": self.allow_delete,
            "allow_write": self.allow_write,
            "allow_overwrite": self.allow_overwrite,
            "platform": os.name,
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX FileManager shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_file_manager: Optional[
    FileManager
] = None


def get_file_manager() -> FileManager:
    """
    Return the shared RENIX FileManager instance.
    """

    global _default_file_manager

    if _default_file_manager is None:

        _default_file_manager = (
            FileManager()
        )

    return _default_file_manager


# ==============================================================
# SIMPLE MODULE-LEVEL HELPERS
# ==============================================================


def read_text(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
) -> str:

    return get_file_manager().read_text(
        path,
        encoding=encoding,
    )


def write_text(
    path: str | os.PathLike[str],
    content: str,
    *,
    encoding: str = "utf-8",
    overwrite: Optional[bool] = None,
) -> str:

    return get_file_manager().write_text(
        path,
        content,
        encoding=encoding,
        overwrite=overwrite,
    )


def copy(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    overwrite: Optional[bool] = None,
) -> str:

    return get_file_manager().copy(
        source,
        destination,
        overwrite=overwrite,
    )


def move(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    overwrite: Optional[bool] = None,
) -> str:

    return get_file_manager().move(
        source,
        destination,
        overwrite=overwrite,
    )


def delete(
    path: str | os.PathLike[str],
    *,
    recursive: bool = False,
) -> bool:

    return get_file_manager().delete(
        path,
        recursive=recursive,
    )


def get_info(
    path: str | os.PathLike[str],
) -> FileInfo:

    return get_file_manager().get_info(
        path
    )


__all__ = [
    "FileInfo",
    "FileManager",
    "get_file_manager",
    "read_text",
    "write_text",
    "copy",
    "move",
    "delete",
    "get_info",
]


