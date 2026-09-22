"""
RENIX Duplicate Detector
========================

Detects duplicate files using a staged comparison:

1. File size
2. Partial hash (optional)
3. Full cryptographic hash

Features:
- Exact duplicate detection
- Duplicate groups
- Directory scanning
- Recursive scanning
- Safe read-only operation
- Configurable ignored extensions/directories
- Human-readable reports
- JSON-serializable results

This module NEVER deletes or modifies files.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# CONSTANTS
# ==============================================================

DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_PARTIAL_HASH_SIZE = 64 * 1024

SUPPORTED_HASH_ALGORITHMS = {
    "md5",
    "sha1",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
}


# ==============================================================
# DATA MODELS
# ==============================================================


@dataclass
class DuplicateFile:
    """Represents one file inside a duplicate group."""

    path: str
    name: str
    size: int
    size_human: str
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateGroup:
    """A collection of files with identical content."""

    hash: str
    size: int
    size_human: str
    files: list[DuplicateFile]

    @property
    def count(self) -> int:
        return len(self.files)

    @property
    def wasted_space(self) -> int:
        if len(self.files) <= 1:
            return 0

        return self.size * (
            len(self.files) - 1
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["count"] = self.count
        data["wasted_space"] = self.wasted_space
        return data


# ==============================================================
# DUPLICATE DETECTOR
# ==============================================================


class DuplicateDetector:
    """
    RENIX duplicate detection engine.

    The detector is completely read-only.
    """

    def __init__(
        self,
        enabled: bool = True,
        hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        partial_hash_size: int = DEFAULT_PARTIAL_HASH_SIZE,
        ignored_extensions: Optional[
            set[str]
        ] = None,
        ignored_directories: Optional[
            set[str]
        ] = None,
    ) -> None:

        self.enabled = enabled

        self.hash_algorithm = (
            hash_algorithm.lower()
        )

        self.chunk_size = chunk_size
        self.partial_hash_size = (
            partial_hash_size
        )

        if (
            self.hash_algorithm
            not in SUPPORTED_HASH_ALGORITHMS
        ):
            raise ValueError(
                f"Unsupported hash algorithm: "
                f"{hash_algorithm}"
            )

        self.ignored_extensions = {
            extension.lower()
            for extension in (
                ignored_extensions or set()
            )
        }

        self.ignored_directories = {
            name.lower()
            for name in (
                ignored_directories or set()
            )
        }

        logger.info(
            "RENIX DuplicateDetector initialized."
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
                "RENIX DuplicateDetector is disabled."
            )

    # ==========================================================
    # SIZE
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
    # FILE FILTERING
    # ==========================================================

    def should_ignore(
        self,
        path: Path,
    ) -> bool:

        if (
            path.suffix.lower()
            in self.ignored_extensions
        ):
            return True

        for part in path.parts:

            if (
                part.lower()
                in self.ignored_directories
            ):
                return True

        return False

    # ==========================================================
    # FILE ITERATION
    # ==========================================================

    def iter_files(
        self,
        root: str | os.PathLike[str],
        recursive: bool = True,
    ) -> Iterable[Path]:

        root_path = Path(
            root
        ).expanduser()

        if not root_path.exists():
            raise FileNotFoundError(
                str(root_path)
            )

        if root_path.is_file():

            if not self.should_ignore(
                root_path
            ):
                yield root_path

            return

        if not root_path.is_dir():
            return

        if recursive:

            for current_root, directories, files in os.walk(
                root_path
            ):

                directories[:] = [
                    directory
                    for directory in directories
                    if directory.lower()
                    not in self.ignored_directories
                ]

                current_path = Path(
                    current_root
                )

                for filename in files:

                    path = (
                        current_path
                        / filename
                    )

                    if self.should_ignore(
                        path
                    ):
                        continue

                    try:

                        if path.is_file():
                            yield path

                    except OSError:
                        continue

        else:

            try:

                for path in root_path.iterdir():

                    if self.should_ignore(
                        path
                    ):
                        continue

                    try:

                        if path.is_file():
                            yield path

                    except OSError:
                        continue

            except OSError:
                return

    # ==========================================================
    # HASHING
    # ==========================================================

    def _new_hasher(
        self,
        algorithm: Optional[str] = None,
    ):
        selected = (
            algorithm or self.hash_algorithm
        ).lower()

        if (
            selected
            not in SUPPORTED_HASH_ALGORITHMS
        ):
            raise ValueError(
                f"Unsupported hash algorithm: "
                f"{selected}"
            )

        return hashlib.new(selected)

    def partial_hash(
        self,
        path: str | os.PathLike[str],
    ) -> str:

        self._check()

        file_path = Path(path)

        hasher = self._new_hasher()

        with file_path.open("rb") as file:

            first_chunk = file.read(
                self.partial_hash_size
            )

            hasher.update(
                first_chunk
            )

            try:
                file.seek(
                    -self.partial_hash_size,
                    os.SEEK_END,
                )

                last_chunk = file.read(
                    self.partial_hash_size
                )

                hasher.update(
                    last_chunk
                )

            except OSError:

                # Small files may not have enough
                # bytes for a backward seek.
                pass

        return hasher.hexdigest()

    def full_hash(
        self,
        path: str | os.PathLike[str],
    ) -> str:

        self._check()

        file_path = Path(path)

        hasher = self._new_hasher()

        with file_path.open("rb") as file:

            while True:

                chunk = file.read(
                    self.chunk_size
                )

                if not chunk:
                    break

                hasher.update(chunk)

        return hasher.hexdigest()

    # ==========================================================
    # SIZE GROUPING
    # ==========================================================

    def group_by_size(
        self,
        files: Iterable[Path],
    ) -> dict[int, list[Path]]:

        groups: dict[
            int,
            list[Path],
        ] = defaultdict(list)

        for path in files:

            try:

                size = path.stat().st_size

            except OSError:

                continue

            groups[size].append(
                path
            )

        return groups

    # ==========================================================
    # PARTIAL HASH GROUPING
    # ==========================================================

    def group_by_partial_hash(
        self,
        files: Iterable[Path],
    ) -> dict[str, list[Path]]:

        groups: dict[
            str,
            list[Path],
        ] = defaultdict(list)

        for path in files:

            try:

                file_hash = (
                    self.partial_hash(
                        path
                    )
                )

            except (
                OSError,
                ValueError,
            ) as exc:

                logger.warning(
                    "Unable to partially hash "
                    "%s: %s",
                    path,
                    exc,
                )

                continue

            groups[file_hash].append(
                path
            )

        return groups

    # ==========================================================
    # FULL HASH GROUPING
    # ==========================================================

    def group_by_hash(
        self,
        files: Iterable[Path],
    ) -> dict[str, list[Path]]:

        groups: dict[
            str,
            list[Path],
        ] = defaultdict(list)

        for path in files:

            try:

                file_hash = self.full_hash(
                    path
                )

            except (
                OSError,
                ValueError,
            ) as exc:

                logger.warning(
                    "Unable to hash %s: %s",
                    path,
                    exc,
                )

                continue

            groups[file_hash].append(
                path
            )

        return groups

    # ==========================================================
    # DUPLICATE GROUP CREATION
    # ==========================================================

    def create_group(
        self,
        file_hash: str,
        files: list[Path],
    ) -> DuplicateGroup:

        if not files:
            raise ValueError(
                "Cannot create a duplicate group "
                "from an empty file list."
            )

        first = files[0]

        try:
            size = first.stat().st_size
        except OSError:
            size = 0

        duplicate_files = []

        for path in files:

            try:
                file_size = (
                    path.stat().st_size
                )
            except OSError:
                continue

            duplicate_files.append(
                DuplicateFile(
                    path=str(
                        path.resolve()
                    ),
                    name=path.name,
                    size=file_size,
                    size_human=self.format_size(
                        file_size
                    ),
                    hash=file_hash,
                )
            )

        return DuplicateGroup(
            hash=file_hash,
            size=size,
            size_human=self.format_size(
                size
            ),
            files=duplicate_files,
        )

    # ==========================================================
    # DETECTION
    # ==========================================================

    def find_duplicates(
        self,
        root: str | os.PathLike[str],
        recursive: bool = True,
    ) -> list[DuplicateGroup]:

        self._check()

        files = list(
            self.iter_files(
                root,
                recursive=recursive,
            )
        )

        if len(files) < 2:
            return []

        # Stage 1:
        # Only files with equal sizes can be exact
        # duplicates.
        size_groups = (
            self.group_by_size(
                files
            )
        )

        same_size_files = [
            path
            for group in size_groups.values()
            if len(group) > 1
            for path in group
        ]

        if len(same_size_files) < 2:
            return []

        # Stage 2:
        # Partial hashes reduce the amount of full
        # file hashing required.
        partial_groups: dict[
            str,
            list[Path],
        ] = defaultdict(list)

        for path in same_size_files:

            try:

                partial = (
                    self.partial_hash(
                        path
                    )
                )

            except OSError as exc:

                logger.warning(
                    "Partial hash failed for %s: %s",
                    path,
                    exc,
                )

                continue

            partial_groups[
                partial
            ].append(path)

        candidates = [
            path
            for group in partial_groups.values()
            if len(group) > 1
            for path in group
        ]

        if len(candidates) < 2:
            return []

        # Stage 3:
        # Full cryptographic hash.
        full_groups = self.group_by_hash(
            candidates
        )

        duplicate_groups = []

        for file_hash, group in (
            full_groups.items()
        ):

            if len(group) < 2:
                continue

            duplicate_groups.append(
                self.create_group(
                    file_hash,
                    group,
                )
            )

        duplicate_groups.sort(
            key=lambda group: (
                -group.wasted_space,
                -group.size,
            )
        )

        return duplicate_groups

    # ==========================================================
    # FIND DUPLICATES IN MULTIPLE ROOTS
    # ==========================================================

    def find_duplicates_many(
        self,
        roots: list[
            str | os.PathLike[str]
        ],
        recursive: bool = True,
    ) -> list[DuplicateGroup]:

        self._check()

        all_files: list[Path] = []

        for root in roots:

            try:

                all_files.extend(
                    self.iter_files(
                        root,
                        recursive=recursive,
                    )
                )

            except (
                OSError,
                ValueError,
            ) as exc:

                logger.warning(
                    "Unable to scan %s: %s",
                    root,
                    exc,
                )

        # Remove duplicate paths.
        unique_files = list(
            {
                str(path.resolve()): path
                for path in all_files
            }.values()
        )

        if len(unique_files) < 2:
            return []

        size_groups = (
            self.group_by_size(
                unique_files
            )
        )

        candidates = [
            path
            for group in size_groups.values()
            if len(group) > 1
            for path in group
        ]

        if len(candidates) < 2:
            return []

        partial_groups = (
            self.group_by_partial_hash(
                candidates
            )
        )

        candidates = [
            path
            for group in partial_groups.values()
            if len(group) > 1
            for path in group
        ]

        if len(candidates) < 2:
            return []

        full_groups = self.group_by_hash(
            candidates
        )

        results = []

        for file_hash, group in (
            full_groups.items()
        ):

            if len(group) >= 2:

                results.append(
                    self.create_group(
                        file_hash,
                        group,
                    )
                )

        results.sort(
            key=lambda group: (
                -group.wasted_space,
                -group.size,
            )
        )

        return results

    # ==========================================================
    # QUICK CHECK
    # ==========================================================

    def are_duplicates(
        self,
        first: str | os.PathLike[str],
        second: str | os.PathLike[str],
    ) -> bool:

        self._check()

        first_path = Path(first)
        second_path = Path(second)

        try:

            first_size = (
                first_path.stat().st_size
            )

            second_size = (
                second_path.stat().st_size
            )

        except OSError:

            return False

        if first_size != second_size:
            return False

        try:

            first_partial = (
                self.partial_hash(
                    first_path
                )
            )

            second_partial = (
                self.partial_hash(
                    second_path
                )
            )

        except OSError:

            return False

        if first_partial != second_partial:
            return False

        try:

            return (
                self.full_hash(
                    first_path
                )
                == self.full_hash(
                    second_path
                )
            )

        except OSError:

            return False

    # ==========================================================
    # REPORTING
    # ==========================================================

    @staticmethod
    def total_wasted_space(
        groups: list[DuplicateGroup],
    ) -> int:

        return sum(
            group.wasted_space
            for group in groups
        )

    def build_report(
        self,
        groups: list[DuplicateGroup],
    ) -> dict[str, Any]:

        total_files = sum(
            group.count
            for group in groups
        )

        wasted_space = (
            self.total_wasted_space(
                groups
            )
        )

        return {
            "duplicate_groups": len(
                groups
            ),
            "duplicate_files": total_files,
            "wasted_space": wasted_space,
            "wasted_space_human": (
                self.format_size(
                    wasted_space
                )
            ),
            "groups": [
                group.to_dict()
                for group in groups
            ],
        }

    def find_duplicates_report(
        self,
        root: str | os.PathLike[str],
        recursive: bool = True,
    ) -> dict[str, Any]:

        groups = self.find_duplicates(
            root,
            recursive=recursive,
        )

        return self.build_report(
            groups
        )

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "hash_algorithm": (
                self.hash_algorithm
            ),
            "chunk_size": self.chunk_size,
            "partial_hash_size": (
                self.partial_hash_size
            ),
            "ignored_extensions": sorted(
                self.ignored_extensions
            ),
            "ignored_directories": sorted(
                self.ignored_directories
            ),
            "read_only": True,
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX DuplicateDetector shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_duplicate_detector: Optional[
    DuplicateDetector
] = None


def get_duplicate_detector() -> DuplicateDetector:

    global _default_duplicate_detector

    if _default_duplicate_detector is None:

        _default_duplicate_detector = (
            DuplicateDetector()
        )

    return _default_duplicate_detector


# ==============================================================
# MODULE-LEVEL HELPERS
# ==============================================================


def find_duplicates(
    root: str | os.PathLike[str],
    recursive: bool = True,
) -> list[dict[str, Any]]:

    groups = (
        get_duplicate_detector()
        .find_duplicates(
            root,
            recursive=recursive,
        )
    )

    return [
        group.to_dict()
        for group in groups
    ]


def are_duplicates(
    first: str | os.PathLike[str],
    second: str | os.PathLike[str],
) -> bool:

    return (
        get_duplicate_detector()
        .are_duplicates(
            first,
            second,
        )
    )


# ==============================================================
# PUBLIC API
# ==============================================================

__all__ = [
    "DuplicateFile",
    "DuplicateGroup",
    "DuplicateDetector",
    "get_duplicate_detector",
    "find_duplicates",
    "are_duplicates",
]


