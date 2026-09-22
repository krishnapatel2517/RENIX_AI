"""
RENIX Code Writer
=================

Responsible for safely creating and writing source-code files.

Responsibilities:
    - Create new source files
    - Write complete files
    - Append content
    - Insert content at specific lines
    - Create parent directories when required
    - Prevent accidental overwrites
    - Create backups before overwriting
    - Validate paths
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CodeWriter:
    """Safe source-code writer for RENIX."""

    def __init__(
        self,
        *,
        allowed_root: str | os.PathLike[str] | None = None,
        backup_directory: str | os.PathLike[str] | None = None,
        create_backups: bool = True,
        max_file_size: int = 10 * 1024 * 1024,
    ) -> None:

        self.allowed_root = (
            Path(allowed_root)
            .expanduser()
            .resolve()
            if allowed_root is not None
            else None
        )

        self.create_backups = create_backups
        self.max_file_size = max_file_size

        self.backup_directory = (
            Path(backup_directory)
            .expanduser()
            .resolve()
            if backup_directory is not None
            else None
        )

        if self.backup_directory is not None:
            self.backup_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # =========================================================
    # WRITE
    # =========================================================

    def write(
        self,
        file_path: str | os.PathLike[str],
        content: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = True,
        create_directories: bool = True,
        backup: bool | None = None,
    ) -> Path:
        """
        Write complete content to a source file.
        """

        if not isinstance(content, str):
            raise TypeError(
                "content must be a string."
            )

        if len(content.encode(encoding)) > self.max_file_size:
            raise ValueError(
                "Content exceeds the configured maximum file size."
            )

        path = self._validate_path(
            file_path
        )

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {path}"
            )

        if create_directories:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
        elif not path.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {path.parent}"
            )

        should_backup = (
            self.create_backups
            if backup is None
            else backup
        )

        if (
            path.exists()
            and should_backup
        ):
            self.create_backup(
                path
            )

        temporary_path = path.with_name(
            f".{path.name}.renix.tmp"
        )

        try:
            temporary_path.write_text(
                content,
                encoding=encoding,
                newline="",
            )

            temporary_path.replace(
                path
            )

        except Exception:

            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

            raise

        logger.info(
            "Wrote source file: %s",
            path,
        )

        return path

    def write_file(
        self,
        file_path: str | os.PathLike[str],
        content: str,
        **kwargs: Any,
    ) -> Path:

        return self.write(
            file_path,
            content,
            **kwargs,
        )

    # =========================================================
    # CREATE
    # =========================================================

    def create(
        self,
        file_path: str | os.PathLike[str],
        content: str = "",
        *,
        encoding: str = "utf-8",
    ) -> Path:
        """
        Create a new file.

        Existing files are never overwritten.
        """

        return self.write(
            file_path,
            content,
            encoding=encoding,
            overwrite=False,
            backup=False,
        )

    def create_file(
        self,
        file_path: str | os.PathLike[str],
        content: str = "",
        **kwargs: Any,
    ) -> Path:

        return self.create(
            file_path,
            content,
            **kwargs,
        )

    # =========================================================
    # APPEND
    # =========================================================

    def append(
        self,
        file_path: str | os.PathLike[str],
        content: str,
        *,
        encoding: str = "utf-8",
        add_newline: bool = False,
        backup: bool | None = None,
    ) -> Path:

        path = self._validate_path(
            file_path
        )

        if path.exists():

            existing = path.read_text(
                encoding=encoding,
                errors="replace",
            )

        else:

            existing = ""

        if (
            add_newline
            and existing
            and not existing.endswith("\n")
        ):
            existing += "\n"

        existing += content

        return self.write(
            path,
            existing,
            encoding=encoding,
            overwrite=True,
            backup=backup,
        )

    # =========================================================
    # INSERT
    # =========================================================

    def insert_at_line(
        self,
        file_path: str | os.PathLike[str],
        line_number: int,
        content: str,
        *,
        encoding: str = "utf-8",
        backup: bool | None = None,
    ) -> Path:

        if line_number < 1:
            raise ValueError(
                "line_number must be >= 1."
            )

        path = self._validate_path(
            file_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        existing = path.read_text(
            encoding=encoding,
            errors="replace",
        )

        lines = existing.splitlines(
            keepends=True
        )

        insertion = content

        if insertion and not insertion.endswith(
            "\n"
        ):
            insertion += "\n"

        index = min(
            line_number - 1,
            len(lines),
        )

        lines.insert(
            index,
            insertion,
        )

        return self.write(
            path,
            "".join(lines),
            encoding=encoding,
            overwrite=True,
            backup=backup,
        )

    # =========================================================
    # REPLACE
    # =========================================================

    def replace_text(
        self,
        file_path: str | os.PathLike[str],
        old_text: str,
        new_text: str,
        *,
        encoding: str = "utf-8",
        replace_all: bool = True,
        backup: bool | None = None,
    ) -> Path:

        if not old_text:
            raise ValueError(
                "old_text cannot be empty."
            )

        path = self._validate_path(
            file_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        content = path.read_text(
            encoding=encoding,
            errors="replace",
        )

        if old_text not in content:
            raise ValueError(
                "old_text was not found in the file."
            )

        if replace_all:

            updated = content.replace(
                old_text,
                new_text,
            )

        else:

            updated = content.replace(
                old_text,
                new_text,
                1,
            )

        return self.write(
            path,
            updated,
            encoding=encoding,
            overwrite=True,
            backup=backup,
        )

    # =========================================================
    # DELETE CONTENT
    # =========================================================

    def remove_text(
        self,
        file_path: str | os.PathLike[str],
        text: str,
        *,
        encoding: str = "utf-8",
        remove_all: bool = True,
        backup: bool | None = None,
    ) -> Path:

        if not text:
            raise ValueError(
                "text cannot be empty."
            )

        path = self._validate_path(
            file_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        content = path.read_text(
            encoding=encoding,
            errors="replace",
        )

        if text not in content:
            raise ValueError(
                "text was not found in the file."
            )

        if remove_all:

            updated = content.replace(
                text,
                "",
            )

        else:

            updated = content.replace(
                text,
                "",
                1,
            )

        return self.write(
            path,
            updated,
            encoding=encoding,
            overwrite=True,
            backup=backup,
        )

    # =========================================================
    # BACKUPS
    # =========================================================

    def create_backup(
        self,
        file_path: str | os.PathLike[str],
    ) -> Path:

        path = self._validate_path(
            file_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Cannot back up missing file: {path}"
            )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        if self.backup_directory is not None:

            backup_root = (
                self.backup_directory
            )

            backup_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            backup_path = (
                backup_root
                / f"{path.name}.{timestamp}.bak"
            )

        else:

            backup_path = path.with_name(
                f"{path.name}.{timestamp}.bak"
            )

        shutil.copy2(
            path,
            backup_path,
        )

        logger.info(
            "Created backup: %s",
            backup_path,
        )

        return backup_path

    # =========================================================
    # COPY
    # =========================================================

    def copy(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> Path:

        source_path = self._validate_path(
            source
        )

        destination_path = self._validate_path(
            destination
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file does not exist: {source_path}"
            )

        if not source_path.is_file():
            raise IsADirectoryError(
                f"Source is not a file: {source_path}"
            )

        if (
            destination_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Destination already exists: {destination_path}"
            )

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        return destination_path

    # =========================================================
    # ENCODING
    # =========================================================

    def write_lines(
        self,
        file_path: str | os.PathLike[str],
        lines: list[str],
        *,
        encoding: str = "utf-8",
        backup: bool | None = None,
    ) -> Path:

        content = "\n".join(
            lines
        )

        if lines:
            content += "\n"

        return self.write(
            file_path,
            content,
            encoding=encoding,
            overwrite=True,
            backup=backup,
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    def validate_content(
        self,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:

        if not isinstance(
            content,
            str,
        ):
            return {
                "valid": False,
                "error": "Content must be a string.",
            }

        try:

            encoded = content.encode(
                encoding
            )

        except UnicodeEncodeError as exc:

            return {
                "valid": False,
                "error": str(exc),
            }

        if len(encoded) > self.max_file_size:

            return {
                "valid": False,
                "error": (
                    "Content exceeds maximum file size."
                ),
            }

        return {
            "valid": True,
            "bytes": len(encoded),
            "lines": len(
                content.splitlines()
            ),
        }

    # =========================================================
    # PATH SAFETY
    # =========================================================

    def _validate_path(
        self,
        file_path: str | os.PathLike[str],
    ) -> Path:

        path = (
            Path(file_path)
            .expanduser()
            .resolve()
        )

        if self.allowed_root is not None:

            try:

                path.relative_to(
                    self.allowed_root
                )

            except ValueError as exc:

                raise PermissionError(
                    "File is outside the allowed coding workspace."
                ) from exc

        return path


__all__ = [
    "CodeWriter",
]


