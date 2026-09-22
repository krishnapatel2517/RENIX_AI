"""
RENIX Archive Manager
=====================

High-level archive management service for RENIX.

Supports:
- ZIP creation
- ZIP extraction
- TAR/TAR.GZ/TAR.BZ2/TAR.XZ creation
- TAR extraction
- Listing archive contents
- Adding files to ZIP archives
- Safe extraction against path traversal
- Overwrite protection
- Dry-run mode
- Verification
- Archive statistics
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class ArchiveManagerError(Exception):
    """Base exception for archive operations."""


class UnsupportedArchiveError(ArchiveManagerError):
    """Raised when the archive format is unsupported."""


class ArchiveSecurityError(ArchiveManagerError):
    """Raised when an archive operation is unsafe."""


class ArchiveVerificationError(ArchiveManagerError):
    """Raised when archive verification fails."""


class ArchiveExistsError(ArchiveManagerError):
    """Raised when an archive already exists."""


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class ArchiveStats:
    files: int = 0
    directories: int = 0
    bytes_processed: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "directories": self.directories,
            "bytes_processed": self.bytes_processed,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class ArchiveResult:
    success: bool
    operation: str
    source: Optional[str] = None
    destination: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    dry_run: bool = False
    verified: bool = False
    stats: ArchiveStats = field(
        default_factory=ArchiveStats
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
            "verified": self.verified,
            "stats": self.stats.to_dict(),
        }


@dataclass
class ArchiveEntry:
    name: str
    size: int
    compressed_size: int
    is_directory: bool
    modified_time: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "compressed_size": self.compressed_size,
            "is_directory": self.is_directory,
            "modified_time": self.modified_time,
        }


# ============================================================
# ARCHIVE MANAGER
# ============================================================


class ArchiveManager:
    """
    RENIX archive engine.

    Uses Python's standard-library archive modules so that the
    basic archive subsystem does not require external packages.
    """

    ZIP_EXTENSIONS = {
        ".zip",
    }

    TAR_EXTENSIONS = {
        ".tar",
        ".tgz",
        ".gz",
        ".bz2",
        ".xz",
    }

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
    ) -> None:

        self.enabled = enabled
        self.dry_run = dry_run

        logger.info(
            "RENIX ArchiveManager initialized."
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
            raise ArchiveManagerError(
                "RENIX ArchiveManager is disabled."
            )

    # ========================================================
    # PATH HELPERS
    # ========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise ArchiveManagerError(
                "Path cannot be None."
            )

        return Path(path).expanduser()

    @staticmethod
    def ensure_parent(path: Path) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # FORMAT DETECTION
    # ========================================================

    @staticmethod
    def detect_format(
        archive: str | os.PathLike[str],
    ) -> str:

        name = str(archive).lower()

        if name.endswith(".tar.gz"):
            return "gztar"

        if name.endswith(".tar.bz2"):
            return "bztar"

        if name.endswith(".tar.xz"):
            return "xztar"

        if name.endswith(".tgz"):
            return "gztar"

        if name.endswith(".tar"):
            return "tar"

        if name.endswith(".zip"):
            return "zip"

        raise UnsupportedArchiveError(
            f"Unsupported archive format: {archive}"
        )

    # ========================================================
    # SIZE / STATISTICS
    # ========================================================

    @staticmethod
    def calculate_size(
        path: Path,
    ) -> int:

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

        for _, dirs, filenames in os.walk(path):
            directories += len(dirs)
            files += len(filenames)

        return files, directories

    # ========================================================
    # CREATE ZIP
    # ========================================================

    def create_zip(
        self,
        sources: Iterable[str | os.PathLike[str]],
        archive_path: str | os.PathLike[str],
        *,
        compression: int = zipfile.ZIP_DEFLATED,
        overwrite: bool = False,
        verify: bool = True,
    ) -> ArchiveResult:

        self._check_enabled()

        sources = [
            self.normalize_path(source)
            for source in sources
        ]

        archive = self.normalize_path(
            archive_path
        )

        stats = ArchiveStats()
        start = time.perf_counter()

        for source in sources:

            if not source.exists():

                return ArchiveResult(
                    success=False,
                    operation="create_zip",
                    source=str(source),
                    destination=str(archive),
                    message="Source does not exist.",
                    error="Source does not exist.",
                    stats=stats,
                )

        if archive.exists() and not overwrite:

            return ArchiveResult(
                success=False,
                operation="create_zip",
                destination=str(archive),
                message="Archive already exists.",
                error="Archive already exists.",
                stats=stats,
            )

        if self.dry_run:

            for source in sources:
                files, directories = self.count_items(
                    source
                )
                stats.files += files
                stats.directories += directories
                stats.bytes_processed += (
                    self.calculate_size(source)
                )

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=True,
                operation="create_zip",
                destination=str(archive),
                message=(
                    "Dry-run: ZIP archive would be created."
                ),
                dry_run=True,
                stats=stats,
            )

        try:

            self.ensure_parent(archive)

            with zipfile.ZipFile(
                archive,
                mode="w",
                compression=compression,
                allowZip64=True,
            ) as zip_file:

                for source in sources:

                    if source.is_file():

                        zip_file.write(
                            source,
                            arcname=source.name,
                        )

                        stats.files += 1
                        stats.bytes_processed += (
                            self.calculate_size(source)
                        )

                    elif source.is_dir():

                        for root, dirs, files in os.walk(
                            source
                        ):

                            root_path = Path(root)

                            # Preserve empty directories.
                            if not files and not dirs:

                                archive_name = (
                                    root_path
                                    .relative_to(
                                        source.parent
                                    )
                                    .as_posix()
                                    + "/"
                                )

                                zip_file.writestr(
                                    archive_name,
                                    "",
                                )

                                stats.directories += 1

                            for directory in dirs:

                                directory_path = (
                                    root_path
                                    / directory
                                )

                                relative = (
                                    directory_path
                                    .relative_to(
                                        source.parent
                                    )
                                    .as_posix()
                                    + "/"
                                )

                                zip_file.writestr(
                                    relative,
                                    "",
                                )

                                stats.directories += 1

                            for filename in files:

                                file_path = (
                                    root_path
                                    / filename
                                )

                                archive_name = (
                                    file_path
                                    .relative_to(
                                        source.parent
                                    )
                                    .as_posix()
                                )

                                zip_file.write(
                                    file_path,
                                    arcname=archive_name,
                                )

                                stats.files += 1
                                stats.bytes_processed += (
                                    self.calculate_size(
                                        file_path
                                    )
                                )

            verified = False

            if verify:
                verified = self.verify_archive(
                    archive
                )

                if not verified:
                    raise ArchiveVerificationError(
                        "ZIP verification failed."
                    )

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=True,
                operation="create_zip",
                destination=str(archive),
                message="ZIP archive created successfully.",
                verified=verified,
                stats=stats,
            )

        except (
            OSError,
            zipfile.BadZipFile,
            ArchiveVerificationError,
        ) as exc:

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            logger.exception(
                "Failed to create ZIP archive."
            )

            return ArchiveResult(
                success=False,
                operation="create_zip",
                destination=str(archive),
                message="ZIP creation failed.",
                error=str(exc),
                stats=stats,
            )

    # ========================================================
    # CREATE TAR
    # ========================================================

    def create_tar(
        self,
        sources: Iterable[str | os.PathLike[str]],
        archive_path: str | os.PathLike[str],
        *,
        mode: Optional[str] = None,
        overwrite: bool = False,
        verify: bool = True,
    ) -> ArchiveResult:

        self._check_enabled()

        sources = [
            self.normalize_path(source)
            for source in sources
        ]

        archive = self.normalize_path(
            archive_path
        )

        stats = ArchiveStats()
        start = time.perf_counter()

        for source in sources:

            if not source.exists():

                return ArchiveResult(
                    success=False,
                    operation="create_tar",
                    source=str(source),
                    destination=str(archive),
                    message="Source does not exist.",
                    error="Source does not exist.",
                    stats=stats,
                )

        if archive.exists() and not overwrite:

            return ArchiveResult(
                success=False,
                operation="create_tar",
                destination=str(archive),
                message="Archive already exists.",
                error="Archive already exists.",
                stats=stats,
            )

        if mode is None:

            detected = self.detect_format(
                archive
            )

            mode_map = {
                "tar": "w",
                "gztar": "w:gz",
                "bztar": "w:bz2",
                "xztar": "w:xz",
            }

            mode = mode_map.get(
                detected,
                "w",
            )

        if self.dry_run:

            for source in sources:
                files, directories = self.count_items(
                    source
                )
                stats.files += files
                stats.directories += directories
                stats.bytes_processed += (
                    self.calculate_size(source)
                )

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=True,
                operation="create_tar",
                destination=str(archive),
                message=(
                    "Dry-run: TAR archive would be created."
                ),
                dry_run=True,
                stats=stats,
            )

        try:

            self.ensure_parent(archive)

            with tarfile.open(
                archive,
                mode=mode,
            ) as tar:

                for source in sources:

                    arcname = source.name

                    tar.add(
                        source,
                        arcname=arcname,
                        recursive=True,
                    )

                    files, directories = (
                        self.count_items(source)
                    )

                    stats.files += files
                    stats.directories += directories
                    stats.bytes_processed += (
                        self.calculate_size(source)
                    )

            verified = False

            if verify:

                verified = self.verify_archive(
                    archive
                )

                if not verified:
                    raise ArchiveVerificationError(
                        "TAR verification failed."
                    )

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=True,
                operation="create_tar",
                destination=str(archive),
                message="TAR archive created successfully.",
                verified=verified,
                stats=stats,
            )

        except (
            OSError,
            tarfile.TarError,
            ArchiveVerificationError,
        ) as exc:

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=False,
                operation="create_tar",
                destination=str(archive),
                message="TAR creation failed.",
                error=str(exc),
                stats=stats,
            )

    # ========================================================
    # CREATE AUTO
    # ========================================================

    def create(
        self,
        sources: Iterable[str | os.PathLike[str]],
        archive_path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = True,
    ) -> ArchiveResult:

        archive = self.normalize_path(
            archive_path
        )

        archive_format = self.detect_format(
            archive
        )

        if archive_format == "zip":

            return self.create_zip(
                sources,
                archive,
                overwrite=overwrite,
                verify=verify,
            )

        return self.create_tar(
            sources,
            archive,
            overwrite=overwrite,
            verify=verify,
        )

    # ========================================================
    # SAFE ZIP EXTRACTION
    # ========================================================

    @staticmethod
    def _safe_extract_path(
        destination: Path,
        member_name: str,
    ) -> Path:

        destination = destination.resolve()

        # Normalize archive separators.
        member_name = member_name.replace(
            "\\",
            "/",
        )

        member_path = Path(member_name)

        # Absolute paths are never allowed.
        if member_path.is_absolute():
            raise ArchiveSecurityError(
                f"Unsafe absolute archive path: {member_name}"
            )

        # Windows drive paths.
        if (
            len(member_name) >= 2
            and member_name[1] == ":"
        ):
            raise ArchiveSecurityError(
                f"Unsafe drive archive path: {member_name}"
            )

        target = (
            destination
            / member_name
        ).resolve()

        try:
            target.relative_to(destination)
        except ValueError as exc:

            raise ArchiveSecurityError(
                f"Path traversal detected: {member_name}"
            ) from exc

        return target

    def extract_zip(
        self,
        archive_path: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = True,
    ) -> ArchiveResult:

        self._check_enabled()

        archive = self.normalize_path(
            archive_path
        )

        destination_path = self.normalize_path(
            destination
        )

        stats = ArchiveStats()
        start = time.perf_counter()

        if not archive.exists():

            return ArchiveResult(
                success=False,
                operation="extract_zip",
                source=str(archive),
                destination=str(destination_path),
                message="Archive does not exist.",
                error="Archive does not exist.",
                stats=stats,
            )

        if self.dry_run:

            try:

                with zipfile.ZipFile(
                    archive,
                    "r",
                ) as zip_file:

                    for info in zip_file.infolist():

                        self._safe_extract_path(
                            destination_path,
                            info.filename,
                        )

                        if info.is_dir():
                            stats.directories += 1
                        else:
                            stats.files += 1
                            stats.bytes_processed += (
                                info.file_size
                            )

                return ArchiveResult(
                    success=True,
                    operation="extract_zip",
                    source=str(archive),
                    destination=str(destination_path),
                    message=(
                        "Dry-run: ZIP would be extracted."
                    ),
                    dry_run=True,
                    stats=stats,
                )

            except (
                OSError,
                zipfile.BadZipFile,
                ArchiveSecurityError,
            ) as exc:

                return ArchiveResult(
                    success=False,
                    operation="extract_zip",
                    source=str(archive),
                    destination=str(destination_path),
                    message="ZIP preview failed.",
                    error=str(exc),
                    stats=stats,
                )

        try:

            destination_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            with zipfile.ZipFile(
                archive,
                "r",
            ) as zip_file:

                members = zip_file.infolist()

                for info in members:

                    target = self._safe_extract_path(
                        destination_path,
                        info.filename,
                    )

                    if info.is_dir():

                        target.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        stats.directories += 1
                        continue

                    if target.exists() and not overwrite:

                        raise ArchiveManagerError(
                            f"Destination exists: {target}"
                        )

                    target.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    with zip_file.open(
                        info,
                        "r",
                    ) as source_file, target.open(
                        "wb"
                    ) as target_file:

                        shutil.copyfileobj(
                            source_file,
                            target_file,
                        )

                    stats.files += 1
                    stats.bytes_processed += (
                        info.file_size
                    )

            verified = False

            if verify:

                verified = self._verify_extraction(
                    destination_path,
                    stats.files,
                )

                if not verified:
                    raise ArchiveVerificationError(
                        "ZIP extraction verification failed."
                    )

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=True,
                operation="extract_zip",
                source=str(archive),
                destination=str(destination_path),
                message="ZIP extracted successfully.",
                verified=verified,
                stats=stats,
            )

        except (
            OSError,
            zipfile.BadZipFile,
            ArchiveManagerError,
            ArchiveSecurityError,
            ArchiveVerificationError,
        ) as exc:

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=False,
                operation="extract_zip",
                source=str(archive),
                destination=str(destination_path),
                message="ZIP extraction failed.",
                error=str(exc),
                stats=stats,
            )

    # ========================================================
    # SAFE TAR EXTRACTION
    # ========================================================

    def extract_tar(
        self,
        archive_path: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = True,
    ) -> ArchiveResult:

        self._check_enabled()

        archive = self.normalize_path(
            archive_path
        )

        destination_path = self.normalize_path(
            destination
        )

        stats = ArchiveStats()
        start = time.perf_counter()

        if not archive.exists():

            return ArchiveResult(
                success=False,
                operation="extract_tar",
                source=str(archive),
                destination=str(destination_path),
                message="Archive does not exist.",
                error="Archive does not exist.",
                stats=stats,
            )

        try:

            mode = self._tar_read_mode(
                archive
            )

            with tarfile.open(
                archive,
                mode=mode,
            ) as tar:

                members = tar.getmembers()

                for member in members:

                    self._safe_extract_path(
                        destination_path,
                        member.name,
                    )

                    if member.issym() or member.islnk():

                        raise ArchiveSecurityError(
                            "Symbolic/hard links are not "
                            "extracted for safety."
                        )

                    if member.isdir():
                        stats.directories += 1
                    elif member.isfile():
                        stats.files += 1
                        stats.bytes_processed += (
                            member.size
                        )

            if self.dry_run:

                return ArchiveResult(
                    success=True,
                    operation="extract_tar",
                    source=str(archive),
                    destination=str(destination_path),
                    message=(
                        "Dry-run: TAR would be extracted."
                    ),
                    dry_run=True,
                    stats=stats,
                )

            destination_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            with tarfile.open(
                archive,
                mode=mode,
            ) as tar:

                for member in members:

                    target = self._safe_extract_path(
                        destination_path,
                        member.name,
                    )

                    if member.isdir():

                        target.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        continue

                    if not member.isfile():
                        continue

                    if target.exists() and not overwrite:

                        raise ArchiveManagerError(
                            f"Destination exists: {target}"
                        )

                    target.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    source_file = tar.extractfile(
                        member
                    )

                    if source_file is None:
                        continue

                    with source_file, target.open(
                        "wb"
                    ) as output:

                        shutil.copyfileobj(
                            source_file,
                            output,
                        )

            verified = False

            if verify:

                verified = self._verify_extraction(
                    destination_path,
                    stats.files,
                )

                if not verified:
                    raise ArchiveVerificationError(
                        "TAR extraction verification failed."
                    )

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=True,
                operation="extract_tar",
                source=str(archive),
                destination=str(destination_path),
                message="TAR extracted successfully.",
                verified=verified,
                stats=stats,
            )

        except (
            OSError,
            tarfile.TarError,
            ArchiveManagerError,
            ArchiveSecurityError,
            ArchiveVerificationError,
        ) as exc:

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=False,
                operation="extract_tar",
                source=str(archive),
                destination=str(destination_path),
                message="TAR extraction failed.",
                error=str(exc),
                stats=stats,
            )

    # ========================================================
    # EXTRACT AUTO
    # ========================================================

    def extract(
        self,
        archive_path: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        verify: bool = True,
    ) -> ArchiveResult:

        archive = self.normalize_path(
            archive_path
        )

        archive_format = self.detect_format(
            archive
        )

        if archive_format == "zip":

            return self.extract_zip(
                archive,
                destination,
                overwrite=overwrite,
                verify=verify,
            )

        return self.extract_tar(
            archive,
            destination,
            overwrite=overwrite,
            verify=verify,
        )

    # ========================================================
    # LIST CONTENTS
    # ========================================================

    def list_contents(
        self,
        archive_path: str | os.PathLike[str],
    ) -> list[ArchiveEntry]:

        self._check_enabled()

        archive = self.normalize_path(
            archive_path
        )

        archive_format = self.detect_format(
            archive
        )

        entries: list[ArchiveEntry] = []

        if archive_format == "zip":

            with zipfile.ZipFile(
                archive,
                "r",
            ) as zip_file:

                for info in zip_file.infolist():

                    modified = None

                    if info.date_time:
                        modified = (
                            f"{info.date_time[0]:04d}-"
                            f"{info.date_time[1]:02d}-"
                            f"{info.date_time[2]:02d} "
                            f"{info.date_time[3]:02d}:"
                            f"{info.date_time[4]:02d}:"
                            f"{info.date_time[5]:02d}"
                        )

                    entries.append(
                        ArchiveEntry(
                            name=info.filename,
                            size=info.file_size,
                            compressed_size=(
                                info.compress_size
                            ),
                            is_directory=info.is_dir(),
                            modified_time=modified,
                        )
                    )

            return entries

        mode = self._tar_read_mode(
            archive
        )

        with tarfile.open(
            archive,
            mode=mode,
        ) as tar:

            for member in tar.getmembers():

                modified = None

                if member.mtime:
                    modified = time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(
                            member.mtime
                        ),
                    )

                entries.append(
                    ArchiveEntry(
                        name=member.name,
                        size=member.size,
                        compressed_size=0,
                        is_directory=member.isdir(),
                        modified_time=modified,
                    )
                )

        return entries

    # ========================================================
    # ADD TO ZIP
    # ========================================================

    def add_to_zip(
        self,
        archive_path: str | os.PathLike[str],
        sources: Iterable[str | os.PathLike[str]],
        *,
        verify: bool = True,
    ) -> ArchiveResult:

        self._check_enabled()

        archive = self.normalize_path(
            archive_path
        )

        if not archive.exists():

            return ArchiveResult(
                success=False,
                operation="add_to_zip",
                destination=str(archive),
                message="Archive does not exist.",
                error="Archive does not exist.",
            )

        if self.detect_format(archive) != "zip":

            raise UnsupportedArchiveError(
                "add_to_zip requires a ZIP archive."
            )

        sources = [
            self.normalize_path(source)
            for source in sources
        ]

        stats = ArchiveStats()
        start = time.perf_counter()

        try:

            if self.dry_run:

                for source in sources:

                    files, directories = (
                        self.count_items(source)
                    )

                    stats.files += files
                    stats.directories += directories
                    stats.bytes_processed += (
                        self.calculate_size(source)
                    )

                return ArchiveResult(
                    success=True,
                    operation="add_to_zip",
                    destination=str(archive),
                    message=(
                        "Dry-run: items would be added."
                    ),
                    dry_run=True,
                    stats=stats,
                )

            with zipfile.ZipFile(
                archive,
                mode="a",
                compression=zipfile.ZIP_DEFLATED,
            ) as zip_file:

                for source in sources:

                    if source.is_file():

                        zip_file.write(
                            source,
                            arcname=source.name,
                        )

                        stats.files += 1
                        stats.bytes_processed += (
                            self.calculate_size(source)
                        )

                    elif source.is_dir():

                        for root, _, files in os.walk(
                            source
                        ):

                            root_path = Path(root)

                            for filename in files:

                                file_path = (
                                    root_path
                                    / filename
                                )

                                archive_name = (
                                    file_path
                                    .relative_to(
                                        source.parent
                                    )
                                    .as_posix()
                                )

                                zip_file.write(
                                    file_path,
                                    arcname=archive_name,
                                )

                                stats.files += 1
                                stats.bytes_processed += (
                                    self.calculate_size(
                                        file_path
                                    )
                                )

            verified = False

            if verify:

                verified = self.verify_archive(
                    archive
                )

            stats.elapsed_seconds = (
                time.perf_counter() - start
            )

            return ArchiveResult(
                success=True,
                operation="add_to_zip",
                destination=str(archive),
                message="Items added to ZIP successfully.",
                verified=verified,
                stats=stats,
            )

        except (
            OSError,
            zipfile.BadZipFile,
        ) as exc:

            return ArchiveResult(
                success=False,
                operation="add_to_zip",
                destination=str(archive),
                message="Failed to add items to ZIP.",
                error=str(exc),
                stats=stats,
            )

    # ========================================================
    # VERIFY ARCHIVE
    # ========================================================

    def verify_archive(
        self,
        archive_path: str | os.PathLike[str],
    ) -> bool:

        archive = self.normalize_path(
            archive_path
        )

        archive_format = self.detect_format(
            archive
        )

        if archive_format == "zip":

            with zipfile.ZipFile(
                archive,
                "r",
            ) as zip_file:

                return zip_file.testzip() is None

        mode = self._tar_read_mode(
            archive
        )

        with tarfile.open(
            archive,
            mode=mode,
        ) as tar:

            for member in tar.getmembers():

                if member.isfile():

                    source = tar.extractfile(
                        member
                    )

                    if source is not None:
                        while source.read(
                            1024 * 1024
                        ):
                            pass

            return True

    # ========================================================
    # TAR MODE
    # ========================================================

    @staticmethod
    def _tar_read_mode(
        archive: Path,
    ) -> str:

        name = archive.name.lower()

        if name.endswith(".tar.gz") or name.endswith(
            ".tgz"
        ):
            return "r:gz"

        if name.endswith(".tar.bz2"):
            return "r:bz2"

        if name.endswith(".tar.xz"):
            return "r:xz"

        if name.endswith(".tar"):
            return "r"

        raise UnsupportedArchiveError(
            f"Unsupported TAR archive: {archive}"
        )

    # ========================================================
    # EXTRACTION VERIFICATION
    # ========================================================

    @staticmethod
    def _verify_extraction(
        destination: Path,
        expected_files: int,
    ) -> bool:

        if not destination.exists():
            return False

        actual_files = 0

        for _, _, files in os.walk(destination):
            actual_files += len(files)

        return actual_files >= expected_files

    # ========================================================
    # ARCHIVE INFORMATION
    # ========================================================

    def get_info(
        self,
        archive_path: str | os.PathLike[str],
    ) -> dict[str, Any]:

        archive = self.normalize_path(
            archive_path
        )

        entries = self.list_contents(
            archive
        )

        files = sum(
            not entry.is_directory
            for entry in entries
        )

        directories = sum(
            entry.is_directory
            for entry in entries
        )

        uncompressed = sum(
            entry.size
            for entry in entries
        )

        compressed = sum(
            entry.compressed_size
            for entry in entries
        )

        return {
            "path": str(archive),
            "format": self.detect_format(
                archive
            ),
            "size_bytes": (
                archive.stat().st_size
                if archive.exists()
                else 0
            ),
            "files": files,
            "directories": directories,
            "uncompressed_bytes": uncompressed,
            "compressed_bytes": compressed,
            "compression_ratio": (
                compressed / uncompressed
                if uncompressed
                else 0
            ),
        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "supported_formats": [
                "zip",
                "tar",
                "tar.gz",
                "tar.bz2",
                "tar.xz",
            ],
            "capabilities": [
                "create",
                "extract",
                "create_zip",
                "create_tar",
                "extract_zip",
                "extract_tar",
                "list_contents",
                "add_to_zip",
                "verify_archive",
                "safe_extraction",
                "archive_info",
            ],
        }

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX ArchiveManager shut down."
        )


# ============================================================
# SHARED INSTANCE
# ============================================================

_default_archive_manager: Optional[
    ArchiveManager
] = None


def get_archive_manager() -> ArchiveManager:

    global _default_archive_manager

    if _default_archive_manager is None:
        _default_archive_manager = ArchiveManager()

    return _default_archive_manager


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def create_archive(
    sources: Iterable[str | os.PathLike[str]],
    archive_path: str | os.PathLike[str],
) -> ArchiveResult:

    return get_archive_manager().create(
        sources,
        archive_path,
    )


def extract_archive(
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> ArchiveResult:

    return get_archive_manager().extract(
        archive_path,
        destination,
    )


def list_archive(
    archive_path: str | os.PathLike[str],
) -> list[ArchiveEntry]:

    return get_archive_manager().list_contents(
        archive_path
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "ArchiveManagerError",
    "UnsupportedArchiveError",
    "ArchiveSecurityError",
    "ArchiveVerificationError",
    "ArchiveExistsError",
    "ArchiveStats",
    "ArchiveResult",
    "ArchiveEntry",
    "ArchiveManager",
    "get_archive_manager",
    "create_archive",
    "extract_archive",
    "list_archive",
]


