"""
RENIX Code Reader
=================

Reads source-code files safely and provides structured information
to the RENIX coding subsystem.

Responsibilities:
    - Read source files
    - Detect encoding
    - Return line-based content
    - Read selected line ranges
    - Search source code
    - Provide file metadata
    - Detect programming language
    - Prevent accidental binary-file reads
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class CodeFile:
    """Structured representation of a source-code file."""

    path: str
    language: str
    encoding: str
    size: int
    lines: int
    content: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "encoding": self.encoding,
            "size": self.size,
            "lines": self.lines,
            "content": self.content,
        }


@dataclass
class CodeMatch:
    """Represents a search match inside a source file."""

    path: str
    line_number: int
    line: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "line_number": self.line_number,
            "line": self.line,
        }


class CodeReader:
    """Safe source-code reader for RENIX."""

    EXTENSIONS = {
        ".py": "Python",
        ".pyw": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".mjs": "JavaScript",
        ".cjs": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".java": "Java",
        ".c": "C",
        ".h": "C/C++",
        ".cc": "C++",
        ".cpp": "C++",
        ".cxx": "C++",
        ".hpp": "C++",
        ".cs": "C#",
        ".go": "Go",
        ".rs": "Rust",
        ".php": "PHP",
        ".rb": "Ruby",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".kts": "Kotlin",
        ".dart": "Dart",
        ".lua": "Lua",
        ".r": "R",
        ".R": "R",
        ".scala": "Scala",
        ".sh": "Shell",
        ".bash": "Shell",
        ".zsh": "Shell",
        ".ps1": "PowerShell",
        ".bat": "Batch",
        ".cmd": "Batch",
        ".html": "HTML",
        ".htm": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
        ".sass": "Sass",
        ".less": "Less",
        ".sql": "SQL",
        ".xml": "XML",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".ini": "INI",
        ".cfg": "Config",
        ".conf": "Config",
        ".md": "Markdown",
        ".txt": "Text",
    }

    BINARY_EXTENSIONS = {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".iso",
        ".zip",
        ".7z",
        ".rar",
        ".tar",
        ".gz",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".ico",
        ".mp3",
        ".wav",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
    }

    DEFAULT_ENCODINGS = (
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp1252",
        "latin-1",
    )

    def __init__(
        self,
        *,
        max_file_size: int = 10 * 1024 * 1024,
        allowed_root: str | os.PathLike[str] | None = None,
        allow_binary: bool = False,
    ) -> None:

        self.max_file_size = max_file_size
        self.allow_binary = allow_binary

        self.allowed_root = (
            Path(allowed_root)
            .expanduser()
            .resolve()
            if allowed_root is not None
            else None
        )

    # =========================================================
    # READ
    # =========================================================

    def read(
        self,
        file_path: str | os.PathLike[str],
    ) -> str:
        """Read a complete source file."""

        path = self._validate_path(
            file_path
        )

        self._validate_file(
            path
        )

        encoding = self.detect_encoding(
            path
        )

        try:
            return path.read_text(
                encoding=encoding,
                errors="replace",
            )

        except OSError as exc:

            logger.error(
                "Unable to read file %s: %s",
                path,
                exc,
            )

            raise

    def read_file(
        self,
        file_path: str | os.PathLike[str],
    ) -> str:

        return self.read(
            file_path
        )

    # =========================================================
    # STRUCTURED READ
    # =========================================================

    def read_code_file(
        self,
        file_path: str | os.PathLike[str],
    ) -> CodeFile:

        path = self._validate_path(
            file_path
        )

        self._validate_file(
            path
        )

        encoding = self.detect_encoding(
            path
        )

        content = path.read_text(
            encoding=encoding,
            errors="replace",
        )

        language = self.detect_language(
            path
        )

        return CodeFile(
            path=str(path),
            language=language,
            encoding=encoding,
            size=path.stat().st_size,
            lines=len(content.splitlines()),
            content=content,
        )

    # =========================================================
    # LINE READING
    # =========================================================

    def read_lines(
        self,
        file_path: str | os.PathLike[str],
        *,
        start: int = 1,
        end: int | None = None,
    ) -> list[str]:

        if start < 1:
            raise ValueError(
                "start must be >= 1"
            )

        content = self.read(
            file_path
        )

        lines = content.splitlines()

        if end is None:
            end = len(lines)

        if end < start:
            return []

        return lines[
            start - 1:end
        ]

    def read_line(
        self,
        file_path: str | os.PathLike[str],
        line_number: int,
    ) -> str:

        lines = self.read_lines(
            file_path,
            start=line_number,
            end=line_number,
        )

        if not lines:
            raise IndexError(
                f"Line does not exist: {line_number}"
            )

        return lines[0]

    # =========================================================
    # CONTEXT
    # =========================================================

    def read_context(
        self,
        file_path: str | os.PathLike[str],
        line_number: int,
        *,
        before: int = 5,
        after: int = 5,
    ) -> list[tuple[int, str]]:

        if line_number < 1:
            raise ValueError(
                "line_number must be >= 1"
            )

        if before < 0 or after < 0:
            raise ValueError(
                "before and after must be >= 0"
            )

        content = self.read(
            file_path
        )

        lines = content.splitlines()

        start = max(
            1,
            line_number - before,
        )

        end = min(
            len(lines),
            line_number + after,
        )

        return [
            (
                index,
                lines[index - 1],
            )
            for index in range(
                start,
                end + 1,
            )
        ]

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        file_path: str | os.PathLike[str],
        query: str,
        *,
        case_sensitive: bool = True,
    ) -> list[CodeMatch]:

        if not query:
            return []

        content = self.read(
            file_path
        )

        if case_sensitive:
            search_query = query
        else:
            search_query = query.lower()

        matches: list[CodeMatch] = []

        for number, line in enumerate(
            content.splitlines(),
            start=1,
        ):

            target = (
                line
                if case_sensitive
                else line.lower()
            )

            if search_query in target:

                matches.append(
                    CodeMatch(
                        path=str(
                            Path(file_path)
                            .expanduser()
                            .resolve()
                        ),
                        line_number=number,
                        line=line,
                    )
                )

        return matches

    def search_directory(
        self,
        directory: str | os.PathLike[str],
        query: str,
        *,
        recursive: bool = True,
        case_sensitive: bool = True,
    ) -> list[CodeMatch]:

        root = Path(
            directory
        ).expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {root}"
            )

        if not root.is_dir():
            raise NotADirectoryError(
                str(root)
            )

        matches: list[CodeMatch] = []

        iterator: Iterable[Path]

        if recursive:
            iterator = root.rglob("*")
        else:
            iterator = root.glob("*")

        for path in iterator:

            if not path.is_file():
                continue

            if not self.is_source_file(
                path
            ):
                continue

            try:

                matches.extend(
                    self.search(
                        path,
                        query,
                        case_sensitive=case_sensitive,
                    )
                )

            except (OSError, UnicodeError):

                logger.warning(
                    "Skipping unreadable file: %s",
                    path,
                )

        return matches

    # =========================================================
    # LANGUAGE
    # =========================================================

    def detect_language(
        self,
        file_path: str | os.PathLike[str],
    ) -> str:

        path = Path(
            file_path
        )

        extension = path.suffix

        language = self.EXTENSIONS.get(
            extension
        )

        if language:
            return language

        # Filename-based detection.
        if path.name == "Dockerfile":
            return "Dockerfile"

        if path.name == "Makefile":
            return "Makefile"

        if path.name == "Jenkinsfile":
            return "Groovy"

        if path.name == "Gemfile":
            return "Ruby"

        if path.name == "Rakefile":
            return "Ruby"

        return "Unknown"

    def is_source_file(
        self,
        file_path: str | os.PathLike[str],
    ) -> bool:

        path = Path(
            file_path
        )

        if path.name in {
            "Dockerfile",
            "Makefile",
            "Jenkinsfile",
            "Gemfile",
            "Rakefile",
        }:
            return True

        return path.suffix.lower() in {
            extension.lower()
            for extension
            in self.EXTENSIONS
        }

    # =========================================================
    # ENCODING
    # =========================================================

    def detect_encoding(
        self,
        file_path: str | os.PathLike[str],
    ) -> str:

        path = self._validate_path(
            file_path
        )

        raw = path.read_bytes()

        # UTF-8 BOM.
        if raw.startswith(
            b"\xef\xbb\xbf"
        ):
            return "utf-8-sig"

        # UTF-16 BOM.
        if raw.startswith(
            b"\xff\xfe"
        ):
            return "utf-16-le"

        if raw.startswith(
            b"\xfe\xff"
        ):
            return "utf-16-be"

        for encoding in self.DEFAULT_ENCODINGS:

            try:

                raw.decode(
                    encoding
                )

                return encoding

            except UnicodeDecodeError:
                continue

        return "utf-8"

    # =========================================================
    # METADATA
    # =========================================================

    def metadata(
        self,
        file_path: str | os.PathLike[str],
    ) -> dict:

        path = self._validate_path(
            file_path
        )

        self._validate_file(
            path
        )

        content = self.read(
            path
        )

        stat = path.stat()

        return {
            "path": str(path),
            "name": path.name,
            "extension": path.suffix,
            "language": self.detect_language(
                path
            ),
            "encoding": self.detect_encoding(
                path
            ),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "lines": len(
                content.splitlines()
            ),
            "sha256": self.hash_file(
                path
            ),
        }

    # =========================================================
    # HASH
    # =========================================================

    def hash_file(
        self,
        file_path: str | os.PathLike[str],
        *,
        algorithm: str = "sha256",
    ) -> str:

        path = self._validate_path(
            file_path
        )

        self._validate_file(
            path
        )

        try:
            digest = hashlib.new(
                algorithm
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported hash algorithm: {algorithm}"
            ) from exc

        with path.open(
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                digest.update(
                    chunk
                )

        return digest.hexdigest()

    # =========================================================
    # FILE VALIDATION
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

    def _validate_file(
        self,
        path: Path,
    ) -> None:

        if not path.exists():

            raise FileNotFoundError(
                f"File does not exist: {path}"
            )

        if not path.is_file():

            raise IsADirectoryError(
                f"Expected a file: {path}"
            )

        size = path.stat().st_size

        if size > self.max_file_size:

            raise ValueError(
                f"File exceeds maximum size of "
                f"{self.max_file_size} bytes."
            )

        if (
            not self.allow_binary
            and path.suffix.lower()
            in self.BINARY_EXTENSIONS
        ):

            raise ValueError(
                f"Binary file reading is disabled: {path.name}"
            )

        if self._looks_binary(
            path
        ):

            if not self.allow_binary:

                raise ValueError(
                    f"File appears to be binary: {path.name}"
                )

    @staticmethod
    def _looks_binary(
        path: Path,
    ) -> bool:

        try:

            sample = path.read_bytes()[
                :4096
            ]

        except OSError:
            return False

        if not sample:
            return False

        if b"\x00" in sample:
            return True

        # Count non-text bytes.
        suspicious = 0

        for byte in sample:

            if byte in {
                7,
                8,
                9,
                10,
                12,
                13,
                27,
            }:
                continue

            if byte < 32:
                suspicious += 1

        return (
            suspicious
            / len(sample)
            > 0.10
        )


__all__ = [
    "CodeFile",
    "CodeMatch",
    "CodeReader",
]


