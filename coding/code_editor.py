"""
RENIX Code Editor
=================

Provides safe programmatic editing of source-code files.

Responsibilities:
    - Replace exact text
    - Insert text
    - Delete text
    - Replace lines
    - Insert lines
    - Delete lines
    - Apply multiple edits atomically
    - Create backups before modifications
    - Verify expected matches
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EditOperation:
    """Represents one source-code edit."""

    operation: str
    old_text: str | None = None
    new_text: str | None = None
    line_number: int | None = None
    end_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "line_number": self.line_number,
            "end_line": self.end_line,
        }


@dataclass
class EditResult:
    """Result of a code-edit operation."""

    path: str
    operation: str
    changed: bool
    matches: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "operation": self.operation,
            "changed": self.changed,
            "matches": self.matches,
            "message": self.message,
        }


class CodeEditor:
    """Safe source-code editing engine."""

    def __init__(
        self,
        *,
        allowed_root: str | os.PathLike[str] | None = None,
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

    # =========================================================
    # GENERIC EDIT
    # =========================================================

    def edit(
        self,
        file_path: str | os.PathLike[str],
        *,
        old_text: str | None = None,
        new_text: str | None = None,
        operation: str = "replace",
        line_number: int | None = None,
        end_line: int | None = None,
        backup: bool | None = None,
    ) -> EditResult:

        operation = operation.lower().strip()

        if operation == "replace":
            return self.replace(
                file_path,
                old_text,
                new_text,
                backup=backup,
            )

        if operation == "insert":
            return self.insert(
                file_path,
                new_text or "",
                line_number=line_number,
                backup=backup,
            )

        if operation == "delete":
            return self.delete(
                file_path,
                old_text,
                backup=backup,
            )

        if operation == "replace_line":
            return self.replace_line(
                file_path,
                line_number=line_number,
                new_text=new_text or "",
                backup=backup,
            )

        if operation == "delete_lines":
            return self.delete_lines(
                file_path,
                line_number=line_number,
                end_line=end_line,
                backup=backup,
            )

        raise ValueError(
            f"Unsupported edit operation: {operation}"
        )

    # =========================================================
    # REPLACE TEXT
    # =========================================================

    def replace(
        self,
        file_path: str | os.PathLike[str],
        old_text: str | None,
        new_text: str | None,
        *,
        expected_matches: int | None = None,
        replace_all: bool = True,
        backup: bool | None = None,
    ) -> EditResult:

        if old_text is None:
            raise ValueError(
                "old_text is required for replacement."
            )

        if new_text is None:
            new_text = ""

        if not old_text:
            raise ValueError(
                "old_text cannot be empty."
            )

        path = self._validate_file(
            file_path
        )

        content = self._read(
            path
        )

        matches = content.count(
            old_text
        )

        if matches == 0:
            return EditResult(
                path=str(path),
                operation="replace",
                changed=False,
                matches=0,
                message="Target text was not found.",
            )

        if (
            expected_matches is not None
            and matches != expected_matches
        ):
            raise ValueError(
                f"Expected {expected_matches} matches, "
                f"but found {matches}."
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

        if updated == content:
            return EditResult(
                path=str(path),
                operation="replace",
                changed=False,
                matches=matches,
                message="No changes were required.",
            )

        self._write(
            path,
            updated,
            backup=backup,
        )

        return EditResult(
            path=str(path),
            operation="replace",
            changed=True,
            matches=matches,
            message="Replacement completed.",
        )

    # =========================================================
    # INSERT TEXT
    # =========================================================

    def insert(
        self,
        file_path: str | os.PathLike[str],
        text: str,
        *,
        line_number: int | None = None,
        backup: bool | None = None,
    ) -> EditResult:

        path = self._validate_file(
            file_path
        )

        content = self._read(
            path
        )

        if not text:
            return EditResult(
                path=str(path),
                operation="insert",
                changed=False,
                message="No text supplied.",
            )

        if line_number is None:

            if content and not content.endswith("\n"):
                content += "\n"

            updated = content + text

            if not updated.endswith("\n"):
                updated += "\n"

        else:

            if line_number < 1:
                raise ValueError(
                    "line_number must be >= 1."
                )

            lines = content.splitlines(
                keepends=True
            )

            insertion = text

            if not insertion.endswith("\n"):
                insertion += "\n"

            index = min(
                line_number - 1,
                len(lines),
            )

            lines.insert(
                index,
                insertion,
            )

            updated = "".join(lines)

        self._write(
            path,
            updated,
            backup=backup,
        )

        return EditResult(
            path=str(path),
            operation="insert",
            changed=True,
            matches=1,
            message="Text inserted.",
        )

    # =========================================================
    # DELETE TEXT
    # =========================================================

    def delete(
        self,
        file_path: str | os.PathLike[str],
        text: str | None,
        *,
        expected_matches: int | None = None,
        remove_all: bool = True,
        backup: bool | None = None,
    ) -> EditResult:

        if not text:
            raise ValueError(
                "text is required."
            )

        path = self._validate_file(
            file_path
        )

        content = self._read(
            path
        )

        matches = content.count(
            text
        )

        if matches == 0:
            return EditResult(
                path=str(path),
                operation="delete",
                changed=False,
                matches=0,
                message="Target text was not found.",
            )

        if (
            expected_matches is not None
            and matches != expected_matches
        ):
            raise ValueError(
                f"Expected {expected_matches} matches, "
                f"but found {matches}."
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

        self._write(
            path,
            updated,
            backup=backup,
        )

        return EditResult(
            path=str(path),
            operation="delete",
            changed=True,
            matches=matches,
            message="Text deleted.",
        )

    # =========================================================
    # LINE REPLACEMENT
    # =========================================================

    def replace_line(
        self,
        file_path: str | os.PathLike[str],
        *,
        line_number: int | None,
        new_text: str,
        backup: bool | None = None,
    ) -> EditResult:

        if line_number is None:
            raise ValueError(
                "line_number is required."
            )

        if line_number < 1:
            raise ValueError(
                "line_number must be >= 1."
            )

        path = self._validate_file(
            file_path
        )

        content = self._read(
            path
        )

        lines = content.splitlines(
            keepends=True
        )

        if line_number > len(lines):
            raise IndexError(
                f"Line {line_number} does not exist."
            )

        replacement = new_text

        if not replacement.endswith("\n"):
            replacement += "\n"

        lines[line_number - 1] = replacement

        updated = "".join(lines)

        self._write(
            path,
            updated,
            backup=backup,
        )

        return EditResult(
            path=str(path),
            operation="replace_line",
            changed=True,
            matches=1,
            message=f"Line {line_number} replaced.",
        )

    # =========================================================
    # LINE INSERTION
    # =========================================================

    def insert_line(
        self,
        file_path: str | os.PathLike[str],
        line_number: int,
        text: str,
        *,
        backup: bool | None = None,
    ) -> EditResult:

        return self.insert(
            file_path,
            text,
            line_number=line_number,
            backup=backup,
        )

    # =========================================================
    # LINE DELETION
    # =========================================================

    def delete_line(
        self,
        file_path: str | os.PathLike[str],
        line_number: int,
        *,
        backup: bool | None = None,
    ) -> EditResult:

        return self.delete_lines(
            file_path,
            line_number=line_number,
            end_line=line_number,
            backup=backup,
        )

    def delete_lines(
        self,
        file_path: str | os.PathLike[str],
        *,
        line_number: int | None,
        end_line: int | None = None,
        backup: bool | None = None,
    ) -> EditResult:

        if line_number is None:
            raise ValueError(
                "line_number is required."
            )

        if line_number < 1:
            raise ValueError(
                "line_number must be >= 1."
            )

        if end_line is None:
            end_line = line_number

        if end_line < line_number:
            raise ValueError(
                "end_line cannot be smaller than line_number."
            )

        path = self._validate_file(
            file_path
        )

        content = self._read(
            path
        )

        lines = content.splitlines(
            keepends=True
        )

        if line_number > len(lines):
            raise IndexError(
                f"Line {line_number} does not exist."
            )

        end_line = min(
            end_line,
            len(lines),
        )

        del lines[
            line_number - 1:end_line
        ]

        updated = "".join(lines)

        self._write(
            path,
            updated,
            backup=backup,
        )

        deleted_count = (
            end_line
            - line_number
            + 1
        )

        return EditResult(
            path=str(path),
            operation="delete_lines",
            changed=True,
            matches=deleted_count,
            message=(
                f"Deleted lines "
                f"{line_number}-{end_line}."
            ),
        )

    # =========================================================
    # MULTIPLE EDITS
    # =========================================================

    def apply_operations(
        self,
        file_path: str | os.PathLike[str],
        operations: list[EditOperation | dict[str, Any]],
        *,
        backup: bool | None = None,
    ) -> list[EditResult]:

        path = self._validate_file(
            file_path
        )

        original = self._read(
            path
        )

        working = original
        results: list[EditResult] = []

        try:

            for operation in operations:

                if isinstance(
                    operation,
                    EditOperation,
                ):
                    op = operation
                else:
                    op = EditOperation(
                        **operation
                    )

                working = self._apply_to_content(
                    working,
                    op,
                )

                results.append(
                    EditResult(
                        path=str(path),
                        operation=op.operation,
                        changed=True,
                        matches=1,
                        message="Operation applied.",
                    )
                )

            if working != original:

                self._write(
                    path,
                    working,
                    backup=backup,
                )

            else:

                for result in results:
                    result.changed = False

            return results

        except Exception:

            logger.exception(
                "Atomic edit failed for %s",
                path,
            )

            raise

    # =========================================================
    # CONTENT OPERATIONS
    # =========================================================

    def _apply_to_content(
        self,
        content: str,
        operation: EditOperation,
    ) -> str:

        name = operation.operation.lower()

        if name == "replace":

            if not operation.old_text:
                raise ValueError(
                    "replace requires old_text."
                )

            return content.replace(
                operation.old_text,
                operation.new_text or "",
                1,
            )

        if name == "replace_all":

            if not operation.old_text:
                raise ValueError(
                    "replace_all requires old_text."
                )

            return content.replace(
                operation.old_text,
                operation.new_text or "",
            )

        if name == "delete":

            if not operation.old_text:
                raise ValueError(
                    "delete requires old_text."
                )

            return content.replace(
                operation.old_text,
                "",
            )

        if name == "insert":

            lines = content.splitlines(
                keepends=True
            )

            if operation.line_number is None:
                lines.append(
                    operation.new_text or ""
                )
            else:

                index = max(
                    0,
                    operation.line_number - 1,
                )

                text = (
                    operation.new_text or ""
                )

                if text and not text.endswith("\n"):
                    text += "\n"

                lines.insert(
                    index,
                    text,
                )

            return "".join(lines)

        if name == "replace_line":

            if operation.line_number is None:
                raise ValueError(
                    "replace_line requires line_number."
                )

            lines = content.splitlines(
                keepends=True
            )

            index = operation.line_number - 1

            if index < 0 or index >= len(lines):
                raise IndexError(
                    "Line does not exist."
                )

            replacement = (
                operation.new_text or ""
            )

            if not replacement.endswith("\n"):
                replacement += "\n"

            lines[index] = replacement

            return "".join(lines)

        if name == "delete_lines":

            if operation.line_number is None:
                raise ValueError(
                    "delete_lines requires line_number."
                )

            lines = content.splitlines(
                keepends=True
            )

            start = operation.line_number - 1

            end = (
                operation.end_line
                if operation.end_line is not None
                else operation.line_number
            )

            del lines[
                start:end
            ]

            return "".join(lines)

        raise ValueError(
            f"Unsupported operation: {operation.operation}"
        )

    # =========================================================
    # FILE I/O
    # =========================================================

    def _read(
        self,
        path: Path,
    ) -> str:

        if path.stat().st_size > self.max_file_size:
            raise ValueError(
                f"File exceeds maximum size: {path}"
            )

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    def _write(
        self,
        path: Path,
        content: str,
        *,
        backup: bool | None,
    ) -> None:

        should_backup = (
            self.create_backups
            if backup is None
            else backup
        )

        if (
            path.exists()
            and should_backup
        ):
            self._create_backup(
                path
            )

        temporary = path.with_name(
            f".{path.name}.renix.tmp"
        )

        temporary.write_text(
            content,
            encoding="utf-8",
            newline="",
        )

        temporary.replace(
            path
        )

        logger.info(
            "Edited source file: %s",
            path,
        )

    def _create_backup(
        self,
        path: Path,
    ) -> Path:

        timestamp = (
            __import__("datetime")
            .datetime.datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        backup = path.with_name(
            f"{path.name}.{timestamp}.bak"
        )

        backup.write_bytes(
            path.read_bytes()
        )

        return backup

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_file(
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

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        if not path.is_file():
            raise IsADirectoryError(
                f"Expected a file: {path}"
            )

        return path


__all__ = [
    "CodeEditor",
    "EditOperation",
    "EditResult",
]


