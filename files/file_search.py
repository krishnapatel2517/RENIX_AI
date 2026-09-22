"""
RENIX File Search
=================

Fast filesystem search layer for RENIX.

Features:
- Filename search
- Case-insensitive search
- Exact filename matching
- Extension filtering
- File/folder filtering
- Recursive search
- Hidden-file filtering
- Size filtering
- Modified-time filtering
- Result ranking
- Search cancellation
- Safe permission/error handling
"""

from __future__ import annotations

import fnmatch
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# DATA MODEL
# ==============================================================


@dataclass
class SearchResult:
    """One RENIX filesystem search result."""

    path: str
    name: str
    is_file: bool
    is_directory: bool
    size: int
    extension: str
    modified_at: Optional[float]
    score: float
    matched_by: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================
# SEARCH CONTROLLER
# ==============================================================


class FileSearch:
    """
    RENIX filename/path search engine.

    Example:

        search = FileSearch()

        results = search.search(
            "report",
            "C:/Users/Krishna/Documents"
        )
    """

    def __init__(
        self,
        enabled: bool = True,
        case_sensitive: bool = False,
    ) -> None:

        self.enabled = enabled
        self.case_sensitive = case_sensitive

        logger.info(
            "RENIX FileSearch initialized."
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
                "RENIX FileSearch is disabled."
            )

    # ==========================================================
    # NORMALIZATION
    # ==========================================================

    def _normalize(
        self,
        value: str,
    ) -> str:

        value = str(value).strip()

        if self.case_sensitive:
            return value

        return value.lower()

    # ==========================================================
    # HIDDEN FILE CHECK
    # ==========================================================

    @staticmethod
    def _is_hidden(
        path: Path,
    ) -> bool:

        # Unix-style hidden file.
        if path.name.startswith("."):
            return True

        # Windows hidden attribute.
        if os.name == "nt":

            try:

                import ctypes

                FILE_ATTRIBUTE_HIDDEN = 0x2

                attributes = (
                    ctypes.windll.kernel32.GetFileAttributesW(
                        str(path)
                    )
                )

                if attributes == -1:
                    return False

                return bool(
                    attributes
                    & FILE_ATTRIBUTE_HIDDEN
                )

            except Exception:
                return False

        return False

    # ==========================================================
    # RESULT SCORING
    # ==========================================================

    def _score(
        self,
        name: str,
        query: str,
    ) -> tuple[float, str]:

        normalized_name = self._normalize(
            name
        )

        normalized_query = self._normalize(
            query
        )

        if normalized_name == normalized_query:

            return 100.0, "exact_name"

        if normalized_name.startswith(
            normalized_query
        ):

            return 90.0, "name_prefix"

        if normalized_query in normalized_name:

            return 75.0, "name_contains"

        # Token-based matching.
        tokens = [
            token
            for token in normalized_query.split()
            if token
        ]

        if tokens:

            matched = sum(
                token in normalized_name
                for token in tokens
            )

            if matched == len(tokens):

                return 65.0, "name_tokens"

            if matched > 0:

                return (
                    40.0 + matched * 5.0,
                    "partial_tokens",
                )

        return 0.0, "none"

    # ==========================================================
    # MATCH
    # ==========================================================

    def _matches(
        self,
        path: Path,
        query: str,
        *,
        exact: bool = False,
        pattern: Optional[str] = None,
    ) -> tuple[bool, float, str]:

        name = path.name

        normalized_name = self._normalize(
            name
        )

        normalized_query = self._normalize(
            query
        )

        if exact:

            if normalized_name == normalized_query:

                return True, 100.0, "exact_name"

            return False, 0.0, "none"

        if pattern is not None:

            pattern_normalized = (
                self._normalize(pattern)
            )

            if fnmatch.fnmatch(
                normalized_name,
                pattern_normalized,
            ):

                return True, 80.0, "pattern"

            return False, 0.0, "none"

        score, matched_by = (
            self._score(
                name,
                query,
            )
        )

        return (
            score > 0,
            score,
            matched_by,
        )

    # ==========================================================
    # BUILD RESULT
    # ==========================================================

    def _build_result(
        self,
        path: Path,
        score: float,
        matched_by: str,
    ) -> Optional[SearchResult]:

        try:

            stat = path.stat()

            size = (
                stat.st_size
                if path.is_file()
                else 0
            )

            modified_at = stat.st_mtime

            return SearchResult(
                path=str(path),
                name=path.name,
                is_file=path.is_file(),
                is_directory=path.is_dir(),
                size=size,
                extension=path.suffix,
                modified_at=modified_at,
                score=score,
                matched_by=matched_by,
            )

        except (
            OSError,
            PermissionError,
        ):

            return None

    # ==========================================================
    # MAIN SEARCH
    # ==========================================================

    def search(
        self,
        query: str,
        root: str | os.PathLike[str] = ".",
        *,
        recursive: bool = True,
        files_only: bool = False,
        directories_only: bool = False,
        include_hidden: bool = False,
        exact: bool = False,
        pattern: Optional[str] = None,
        extension: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        modified_after: Optional[float] = None,
        modified_before: Optional[float] = None,
        limit: Optional[int] = 100,
    ) -> list[dict[str, Any]]:
        """
        Search filesystem entries by filename.
        """

        self._check()

        query = str(query).strip()

        if not query and pattern is None:

            raise ValueError(
                "query cannot be empty unless "
                "pattern is provided."
            )

        root_path = Path(
            root
        ).expanduser()

        if not root_path.exists():

            raise FileNotFoundError(
                str(root_path)
            )

        if not root_path.is_dir():

            raise NotADirectoryError(
                str(root_path)
            )

        if extension:

            extension = extension.strip()

            if not extension.startswith("."):
                extension = "." + extension

            extension = self._normalize(
                extension
            )

        if limit is not None:

            limit = max(
                1,
                int(limit)
            )

        if recursive:

            iterator: Iterable[Path] = (
                root_path.rglob("*")
            )

        else:

            iterator = root_path.iterdir()

        results: list[
            SearchResult
        ] = []

        for path in iterator:

            try:

                if (
                    not include_hidden
                    and self._is_hidden(path)
                ):
                    continue

                if (
                    files_only
                    and not path.is_file()
                ):
                    continue

                if (
                    directories_only
                    and not path.is_dir()
                ):
                    continue

                if (
                    files_only
                    and directories_only
                ):
                    continue

                if (
                    extension
                    and path.is_file()
                    and self._normalize(
                        path.suffix
                    ) != extension
                ):
                    continue

                if (
                    extension
                    and not path.is_file()
                ):
                    continue

                matched, score, matched_by = (
                    self._matches(
                        path,
                        query,
                        exact=exact,
                        pattern=pattern,
                    )
                )

                if not matched:
                    continue

                stat = path.stat()

                if (
                    min_size is not None
                    and path.is_file()
                    and stat.st_size
                    < min_size
                ):
                    continue

                if (
                    max_size is not None
                    and path.is_file()
                    and stat.st_size
                    > max_size
                ):
                    continue

                if (
                    modified_after is not None
                    and stat.st_mtime
                    < modified_after
                ):
                    continue

                if (
                    modified_before is not None
                    and stat.st_mtime
                    > modified_before
                ):
                    continue

                result = self._build_result(
                    path,
                    score,
                    matched_by,
                )

                if result:

                    results.append(
                        result
                    )

            except (
                OSError,
                PermissionError,
            ):

                continue

        # Best matches first.
        results.sort(
            key=lambda item: (
                -item.score,
                -(item.modified_at or 0),
                item.name.lower(),
            )
        )

        if limit is not None:

            results = results[:limit]

        return [
            result.to_dict()
            for result in results
        ]

    # ==========================================================
    # EXACT NAME SEARCH
    # ==========================================================

    def find_exact(
        self,
        filename: str,
        root: str | os.PathLike[str] = ".",
        *,
        recursive: bool = True,
        include_hidden: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        return self.search(
            filename,
            root,
            recursive=recursive,
            include_hidden=include_hidden,
            exact=True,
            limit=limit,
        )

    # ==========================================================
    # PATTERN SEARCH
    # ==========================================================

    def find_pattern(
        self,
        pattern: str,
        root: str | os.PathLike[str] = ".",
        *,
        recursive: bool = True,
        files_only: bool = True,
        include_hidden: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        return self.search(
            "",
            root,
            recursive=recursive,
            files_only=files_only,
            include_hidden=include_hidden,
            pattern=pattern,
            limit=limit,
        )

    # ==========================================================
    # EXTENSION SEARCH
    # ==========================================================

    def find_extension(
        self,
        extension: str,
        root: str | os.PathLike[str] = ".",
        *,
        recursive: bool = True,
        include_hidden: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        return self.search(
            extension,
            root,
            recursive=recursive,
            files_only=True,
            include_hidden=include_hidden,
            extension=extension,
            limit=limit,
        )

    # ==========================================================
    # RECENT FILE SEARCH
    # ==========================================================

    def recent_files(
        self,
        root: str | os.PathLike[str] = ".",
        *,
        hours: float = 24,
        limit: int = 50,
        include_hidden: bool = False,
    ) -> list[dict[str, Any]]:

        self._check()

        if hours < 0:
            raise ValueError(
                "hours cannot be negative."
            )

        cutoff = (
            time.time()
            - hours * 60 * 60
        )

        return self.search(
            "",
            root,
            recursive=True,
            files_only=True,
            include_hidden=include_hidden,
            pattern="*",
            modified_after=cutoff,
            limit=limit,
        )

    # ==========================================================
    # LARGE FILE SEARCH
    # ==========================================================

    def large_files(
        self,
        root: str | os.PathLike[str] = ".",
        *,
        minimum_size_mb: float = 100,
        limit: int = 50,
    ) -> list[dict[str, Any]]:

        self._check()

        if minimum_size_mb < 0:

            raise ValueError(
                "minimum_size_mb cannot be negative."
            )

        minimum_size = int(
            minimum_size_mb
            * 1024
            * 1024
        )

        return self.search(
            "",
            root,
            recursive=True,
            files_only=True,
            pattern="*",
            min_size=minimum_size,
            limit=limit,
        )

    # ==========================================================
    # FILE TYPE SEARCH
    # ==========================================================

    def find_documents(
        self,
        root: str | os.PathLike[str] = ".",
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        extensions = {
            ".pdf",
            ".doc",
            ".docx",
            ".txt",
            ".rtf",
            ".odt",
            ".md",
        }

        return self._find_extensions(
            root,
            extensions,
            limit,
        )

    def find_images(
        self,
        root: str | os.PathLike[str] = ".",
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".svg",
        }

        return self._find_extensions(
            root,
            extensions,
            limit,
        )

    def find_videos(
        self,
        root: str | os.PathLike[str] = ".",
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        extensions = {
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".webm",
            ".wmv",
            ".flv",
        }

        return self._find_extensions(
            root,
            extensions,
            limit,
        )

    def find_audio(
        self,
        root: str | os.PathLike[str] = ".",
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        extensions = {
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            ".ogg",
            ".m4a",
            ".wma",
        }

        return self._find_extensions(
            root,
            extensions,
            limit,
        )

    # ==========================================================
    # MULTI-EXTENSION SEARCH
    # ==========================================================

    def _find_extensions(
        self,
        root: str | os.PathLike[str],
        extensions: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:

        self._check()

        root_path = Path(
            root
        ).expanduser()

        if not root_path.is_dir():

            raise NotADirectoryError(
                str(root_path)
            )

        normalized_extensions = {
            self._normalize(
                extension
            )
            for extension in extensions
        }

        results: list[
            dict[str, Any]
        ] = []

        for path in root_path.rglob("*"):

            if not path.is_file():
                continue

            if (
                not self._is_hidden(path)
            ):

                extension = (
                    self._normalize(
                        path.suffix
                    )
                )

                if (
                    extension
                    in normalized_extensions
                ):

                    result = (
                        self._build_result(
                            path,
                            100.0,
                            "extension",
                        )
                    )

                    if result:
                        results.append(
                            result.to_dict()
                        )

        results.sort(
            key=lambda item: (
                -(item.get(
                    "modified_at"
                ) or 0)
            )
        )

        return results[:limit]

    # ==========================================================
    # DUPLICATE NAME SEARCH
    # ==========================================================

    def find_duplicate_names(
        self,
        root: str | os.PathLike[str] = ".",
    ) -> dict[str, list[str]]:

        self._check()

        root_path = Path(
            root
        ).expanduser()

        if not root_path.is_dir():

            raise NotADirectoryError(
                str(root_path)
            )

        names: dict[
            str,
            list[str]
        ] = {}

        for path in root_path.rglob("*"):

            if not path.is_file():
                continue

            key = self._normalize(
                path.name
            )

            names.setdefault(
                key,
                []
            ).append(
                str(path)
            )

        return {
            name: paths
            for name, paths in names.items()
            if len(paths) > 1
        }

    # ==========================================================
    # SEARCH BY PATH
    # ==========================================================

    def search_path(
        self,
        query: str,
        root: str | os.PathLike[str] = ".",
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:

        self._check()

        query_normalized = self._normalize(
            query
        )

        root_path = Path(
            root
        ).expanduser()

        results: list[
            SearchResult
        ] = []

        for path in root_path.rglob("*"):

            path_string = self._normalize(
                str(path)
            )

            if (
                query_normalized
                not in path_string
            ):
                continue

            result = self._build_result(
                path,
                70.0,
                "path_contains",
            )

            if result:
                results.append(
                    result
                )

        results.sort(
            key=lambda item: -item.score
        )

        return [
            item.to_dict()
            for item in results[:limit]
        ]

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "case_sensitive": self.case_sensitive,
            "platform": os.name,
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX FileSearch shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_file_search: Optional[
    FileSearch
] = None


def get_file_search() -> FileSearch:
    """
    Return the shared RENIX file-search instance.
    """

    global _default_file_search

    if _default_file_search is None:

        _default_file_search = FileSearch()

    return _default_file_search


# ==============================================================
# MODULE-LEVEL HELPER
# ==============================================================


def search_files(
    query: str,
    root: str | os.PathLike[str] = ".",
    *,
    recursive: bool = True,
    limit: int = 100,
) -> list[dict[str, Any]]:

    return get_file_search().search(
        query,
        root,
        recursive=recursive,
        limit=limit,
    )


__all__ = [
    "SearchResult",
    "FileSearch",
    "get_file_search",
    "search_files",
]


