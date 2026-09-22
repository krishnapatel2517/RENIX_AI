"""
RENIX File Permissions
======================

Safe permission and access-control utilities for RENIX.

Responsibilities:
- Inspect file permissions
- Check read/write/execute access
- Detect read-only files
- Inspect directory permissions
- Change permissions where supported
- Windows-aware handling
- Permission summaries
- Safe validation before file operations

This module does NOT bypass operating-system permissions.
It only reports or changes permissions through normal OS APIs.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class FilePermissionError(Exception):
    """Base permission-management exception."""


class PermissionInspectionError(FilePermissionError):
    """Raised when permissions cannot be inspected."""


class PermissionChangeError(FilePermissionError):
    """Raised when permissions cannot be changed."""


class InvalidPermissionError(FilePermissionError):
    """Raised when an invalid permission value is supplied."""


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class PermissionInfo:
    path: str
    exists: bool
    is_file: bool
    is_directory: bool
    is_symlink: bool

    readable: bool
    writable: bool
    executable: bool

    owner_read: bool
    owner_write: bool
    owner_execute: bool

    group_read: bool
    group_write: bool
    group_execute: bool

    others_read: bool
    others_write: bool
    others_execute: bool

    mode: Optional[int] = None
    mode_string: Optional[str] = None
    read_only: bool = False

    platform: str = sys.platform

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "is_file": self.is_file,
            "is_directory": self.is_directory,
            "is_symlink": self.is_symlink,
            "readable": self.readable,
            "writable": self.writable,
            "executable": self.executable,
            "owner_read": self.owner_read,
            "owner_write": self.owner_write,
            "owner_execute": self.owner_execute,
            "group_read": self.group_read,
            "group_write": self.group_write,
            "group_execute": self.group_execute,
            "others_read": self.others_read,
            "others_write": self.others_write,
            "others_execute": self.others_execute,
            "mode": self.mode,
            "mode_string": self.mode_string,
            "read_only": self.read_only,
            "platform": self.platform,
        }


@dataclass
class PermissionChangeResult:
    success: bool
    path: str
    previous_mode: Optional[int] = None
    new_mode: Optional[int] = None
    error: Optional[str] = None
    warnings: list[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "path": self.path,
            "previous_mode": self.previous_mode,
            "new_mode": self.new_mode,
            "error": self.error,
            "warnings": self.warnings,
        }


# ============================================================
# FILE PERMISSIONS MANAGER
# ============================================================


class FilePermissions:
    """
    RENIX file-permission inspection and management service.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        allow_changes: bool = True,
    ) -> None:

        self.enabled = enabled
        self.allow_changes = allow_changes

        logger.info(
            "RENIX FilePermissions initialized."
        )

    # ========================================================
    # STATE
    # ========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def _check_enabled(self) -> None:
        if not self.enabled:
            raise FilePermissionError(
                "FilePermissions is disabled."
            )

    def _check_changes_enabled(self) -> None:
        if not self.allow_changes:
            raise PermissionChangeError(
                "Permission changes are disabled."
            )

    # ========================================================
    # PATH
    # ========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise FilePermissionError(
                "Path cannot be None."
            )

        return Path(path).expanduser()

    # ========================================================
    # EXISTENCE
    # ========================================================

    def exists(
        self,
        path: str | os.PathLike[str],
    ) -> bool:

        self._check_enabled()

        return self.normalize_path(path).exists()

    # ========================================================
    # ACCESS CHECKS
    # ========================================================

    def can_read(
        self,
        path: str | os.PathLike[str],
    ) -> bool:

        self._check_enabled()

        return os.access(
            self.normalize_path(path),
            os.R_OK,
        )

    def can_write(
        self,
        path: str | os.PathLike[str],
    ) -> bool:

        self._check_enabled()

        return os.access(
            self.normalize_path(path),
            os.W_OK,
        )

    def can_execute(
        self,
        path: str | os.PathLike[str],
    ) -> bool:

        self._check_enabled()

        return os.access(
            self.normalize_path(path),
            os.X_OK,
        )

    def can_access(
        self,
        path: str | os.PathLike[str],
        *,
        read: bool = False,
        write: bool = False,
        execute: bool = False,
    ) -> bool:

        self._check_enabled()

        flags = 0

        if read:
            flags |= os.R_OK

        if write:
            flags |= os.W_OK

        if execute:
            flags |= os.X_OK

        if flags == 0:
            return True

        return os.access(
            self.normalize_path(path),
            flags,
        )

    # ========================================================
    # MODE
    # ========================================================

    def get_mode(
        self,
        path: str | os.PathLike[str],
    ) -> int:

        self._check_enabled()

        target = self.normalize_path(path)

        try:

            return stat.S_IMODE(
                target.stat().st_mode
            )

        except OSError as exc:

            raise PermissionInspectionError(
                f"Unable to inspect permissions: "
                f"{target}"
            ) from exc

    def get_mode_string(
        self,
        path: str | os.PathLike[str],
    ) -> str:

        self._check_enabled()

        target = self.normalize_path(path)

        try:

            return stat.filemode(
                target.stat().st_mode
            )

        except OSError as exc:

            raise PermissionInspectionError(
                f"Unable to inspect permission mode: "
                f"{target}"
            ) from exc

    # ========================================================
    # DETAILED INSPECTION
    # ========================================================

    def inspect(
        self,
        path: str | os.PathLike[str],
    ) -> PermissionInfo:

        self._check_enabled()

        target = self.normalize_path(path)

        if not target.exists():

            return PermissionInfo(
                path=str(target),
                exists=False,
                is_file=False,
                is_directory=False,
                is_symlink=target.is_symlink(),
                readable=False,
                writable=False,
                executable=False,
                owner_read=False,
                owner_write=False,
                owner_execute=False,
                group_read=False,
                group_write=False,
                group_execute=False,
                others_read=False,
                others_write=False,
                others_execute=False,
                read_only=False,
            )

        try:

            raw_mode = target.stat().st_mode

            mode = stat.S_IMODE(
                raw_mode
            )

            return PermissionInfo(
                path=str(
                    target.resolve()
                ),
                exists=True,
                is_file=target.is_file(),
                is_directory=target.is_dir(),
                is_symlink=target.is_symlink(),

                readable=os.access(
                    target,
                    os.R_OK,
                ),

                writable=os.access(
                    target,
                    os.W_OK,
                ),

                executable=os.access(
                    target,
                    os.X_OK,
                ),

                owner_read=bool(
                    mode & stat.S_IRUSR
                ),

                owner_write=bool(
                    mode & stat.S_IWUSR
                ),

                owner_execute=bool(
                    mode & stat.S_IXUSR
                ),

                group_read=bool(
                    mode & stat.S_IRGRP
                ),

                group_write=bool(
                    mode & stat.S_IWGRP
                ),

                group_execute=bool(
                    mode & stat.S_IXGRP
                ),

                others_read=bool(
                    mode & stat.S_IROTH
                ),

                others_write=bool(
                    mode & stat.S_IWOTH
                ),

                others_execute=bool(
                    mode & stat.S_IXOTH
                ),

                mode=mode,

                mode_string=stat.filemode(
                    raw_mode
                ),

                read_only=(
                    not os.access(
                        target,
                        os.W_OK,
                    )
                ),
            )

        except OSError as exc:

            raise PermissionInspectionError(
                f"Unable to inspect permissions: "
                f"{target}"
            ) from exc

    # ========================================================
    # READ-ONLY
    # ========================================================

    def is_read_only(
        self,
        path: str | os.PathLike[str],
    ) -> bool:

        info = self.inspect(path)

        if not info.exists:
            return False

        return info.read_only

    def set_read_only(
        self,
        path: str | os.PathLike[str],
    ) -> PermissionChangeResult:

        self._check_enabled()
        self._check_changes_enabled()

        target = self.normalize_path(path)

        try:

            previous = self.get_mode(
                target
            )

            if os.name == "nt":

                # Windows read-only attribute.
                import ctypes

                FILE_ATTRIBUTE_READONLY = 0x01
                FILE_ATTRIBUTE_NORMAL = 0x80

                attributes = ctypes.windll.kernel32.GetFileAttributesW(
                    str(target)
                )

                if attributes == -1:
                    raise OSError(
                        "Unable to read Windows file attributes."
                    )

                new_attributes = (
                    attributes
                    | FILE_ATTRIBUTE_READONLY
                )

                if (
                    not ctypes.windll.kernel32.SetFileAttributesW(
                        str(target),
                        new_attributes,
                    )
                ):
                    raise OSError(
                        "Unable to set read-only attribute."
                    )

                return PermissionChangeResult(
                    success=True,
                    path=str(target),
                    previous_mode=previous,
                    new_mode=previous,
                )

            new_mode = (
                previous
                & ~(
                    stat.S_IWUSR
                    | stat.S_IWGRP
                    | stat.S_IWOTH
                )
            )

            os.chmod(
                target,
                new_mode,
            )

            return PermissionChangeResult(
                success=True,
                path=str(target),
                previous_mode=previous,
                new_mode=new_mode,
            )

        except Exception as exc:

            logger.exception(
                "Unable to make file read-only."
            )

            return PermissionChangeResult(
                success=False,
                path=str(target),
                error=str(exc),
            )

    # ========================================================
    # MAKE WRITABLE
    # ========================================================

    def set_writable(
        self,
        path: str | os.PathLike[str],
    ) -> PermissionChangeResult:

        self._check_enabled()
        self._check_changes_enabled()

        target = self.normalize_path(path)

        try:

            previous = self.get_mode(
                target
            )

            if os.name == "nt":

                import ctypes

                FILE_ATTRIBUTE_READONLY = 0x01

                attributes = ctypes.windll.kernel32.GetFileAttributesW(
                    str(target)
                )

                if attributes == -1:
                    raise OSError(
                        "Unable to read Windows file attributes."
                    )

                new_attributes = (
                    attributes
                    & ~FILE_ATTRIBUTE_READONLY
                )

                if (
                    not ctypes.windll.kernel32.SetFileAttributesW(
                        str(target),
                        new_attributes,
                    )
                ):
                    raise OSError(
                        "Unable to remove read-only attribute."
                    )

                return PermissionChangeResult(
                    success=True,
                    path=str(target),
                    previous_mode=previous,
                    new_mode=previous,
                )

            new_mode = (
                previous
                | stat.S_IWUSR
            )

            os.chmod(
                target,
                new_mode,
            )

            return PermissionChangeResult(
                success=True,
                path=str(target),
                previous_mode=previous,
                new_mode=new_mode,
            )

        except Exception as exc:

            return PermissionChangeResult(
                success=False,
                path=str(target),
                error=str(exc),
            )

    # ========================================================
    # CHMOD
    # ========================================================

    @staticmethod
    def parse_mode(
        mode: int | str,
    ) -> int:

        if isinstance(mode, int):

            if mode < 0 or mode > 0o7777:
                raise InvalidPermissionError(
                    "Permission mode must be between "
                    "0 and 0o7777."
                )

            return mode

        if isinstance(mode, str):

            value = mode.strip()

            # Numeric form: 755, 0644, 0o755
            try:

                if value.startswith(
                    "0o"
                ):
                    return int(
                        value,
                        8,
                    )

                if value.isdigit():
                    return int(
                        value,
                        8,
                    )

            except ValueError:
                pass

            # Symbolic form: rwxr-xr-x
            if len(value) == 9:

                mapping = {
                    "r": 4,
                    "w": 2,
                    "x": 1,
                    "-": 0,
                }

                positions = [
                    value[0:3],
                    value[3:6],
                    value[6:9],
                ]

                result = 0

                for index, group in enumerate(
                    positions
                ):

                    if len(group) != 3:
                        break

                    group_value = 0

                    for character in group:

                        if character not in mapping:
                            raise InvalidPermissionError(
                                f"Invalid permission string: "
                                f"{mode}"
                            )

                        group_value += mapping[
                            character
                        ]

                    result |= (
                        group_value
                        << (
                            6
                            - index * 3
                        )
                    )

                else:
                    return result

        raise InvalidPermissionError(
            f"Invalid permission mode: {mode}"
        )

    def chmod(
        self,
        path: str | os.PathLike[str],
        mode: int | str,
    ) -> PermissionChangeResult:

        self._check_enabled()
        self._check_changes_enabled()

        target = self.normalize_path(path)

        try:

            previous = self.get_mode(
                target
            )

            parsed_mode = self.parse_mode(
                mode
            )

            os.chmod(
                target,
                parsed_mode,
            )

            return PermissionChangeResult(
                success=True,
                path=str(target),
                previous_mode=previous,
                new_mode=parsed_mode,
            )

        except Exception as exc:

            logger.exception(
                "Permission change failed."
            )

            return PermissionChangeResult(
                success=False,
                path=str(target),
                error=str(exc),
            )

    # ========================================================
    # CONVENIENCE MODES
    # ========================================================

    def owner_read_only(
        self,
        path: str | os.PathLike[str],
    ) -> PermissionChangeResult:

        return self.chmod(
            path,
            0o400,
        )

    def owner_read_write(
        self,
        path: str | os.PathLike[str],
    ) -> PermissionChangeResult:

        return self.chmod(
            path,
            0o600,
        )

    def owner_read_write_execute(
        self,
        path: str | os.PathLike[str],
    ) -> PermissionChangeResult:

        return self.chmod(
            path,
            0o700,
        )

    def standard_file(
        self,
        path: str | os.PathLike[str],
    ) -> PermissionChangeResult:

        return self.chmod(
            path,
            0o644,
        )

    def standard_directory(
        self,
        path: str | os.PathLike[str],
    ) -> PermissionChangeResult:

        return self.chmod(
            path,
            0o755,
        )

    # ========================================================
    # DIRECTORY PERMISSION CHECK
    # ========================================================

    def can_create_in_directory(
        self,
        directory: str | os.PathLike[str],
    ) -> bool:

        self._check_enabled()

        target = self.normalize_path(
            directory
        )

        if not target.exists():
            return False

        if not target.is_dir():
            return False

        return os.access(
            target,
            os.W_OK | os.X_OK,
        )

    # ========================================================
    # OPERATION CHECK
    # ========================================================

    def check_operation(
        self,
        path: str | os.PathLike[str],
        operation: str,
    ) -> dict[str, Any]:

        self._check_enabled()

        operation = operation.lower().strip()

        info = self.inspect(path)

        if not info.exists:

            return {
                "allowed": False,
                "operation": operation,
                "reason": "Path does not exist.",
                "permissions": info.to_dict(),
            }

        requirements = {
            "read": info.readable,
            "write": info.writable,
            "execute": info.executable,
            "delete": info.writable,
            "rename": info.writable,
            "move": info.writable,
            "copy": info.readable,
        }

        allowed = requirements.get(
            operation,
            False,
        )

        reason = (
            "Permission available."
            if allowed
            else "Required permission unavailable."
        )

        return {
            "allowed": allowed,
            "operation": operation,
            "reason": reason,
            "permissions": info.to_dict(),
        }

    # ========================================================
    # BATCH INSPECTION
    # ========================================================

    def inspect_many(
        self,
        paths: list[
            str | os.PathLike[str]
        ],
    ) -> list[PermissionInfo]:

        return [
            self.inspect(path)
            for path in paths
        ]

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "allow_changes": self.allow_changes,
            "platform": sys.platform,
            "os_name": os.name,
            "capabilities": [
                "permission_inspection",
                "read_check",
                "write_check",
                "execute_check",
                "read_only_detection",
                "read_only_toggle",
                "chmod",
                "directory_access_check",
                "operation_validation",
                "batch_inspection",
            ],
        }

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX FilePermissions shut down."
        )


# ============================================================
# DEFAULT INSTANCE
# ============================================================

_default_permissions: Optional[
    FilePermissions
] = None


def get_file_permissions() -> FilePermissions:

    global _default_permissions

    if _default_permissions is None:
        _default_permissions = FilePermissions()

    return _default_permissions


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def inspect_permissions(
    path: str | os.PathLike[str],
) -> PermissionInfo:

    return get_file_permissions().inspect(
        path
    )


def can_read(
    path: str | os.PathLike[str],
) -> bool:

    return get_file_permissions().can_read(
        path
    )


def can_write(
    path: str | os.PathLike[str],
) -> bool:

    return get_file_permissions().can_write(
        path
    )


def can_execute(
    path: str | os.PathLike[str],
) -> bool:

    return get_file_permissions().can_execute(
        path
    )


def set_read_only(
    path: str | os.PathLike[str],
) -> PermissionChangeResult:

    return get_file_permissions().set_read_only(
        path
    )


def set_writable(
    path: str | os.PathLike[str],
) -> PermissionChangeResult:

    return get_file_permissions().set_writable(
        path
    )


def chmod(
    path: str | os.PathLike[str],
    mode: int | str,
) -> PermissionChangeResult:

    return get_file_permissions().chmod(
        path,
        mode,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "FilePermissionError",
    "PermissionInspectionError",
    "PermissionChangeError",
    "InvalidPermissionError",
    "PermissionInfo",
    "PermissionChangeResult",
    "FilePermissions",
    "get_file_permissions",
    "inspect_permissions",
    "can_read",
    "can_write",
    "can_execute",
    "set_read_only",
    "set_writable",
    "chmod",
]


