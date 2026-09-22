"""
RENIX Error Analyzer
====================

Analyzes errors produced by RENIX coding tools.

Responsibilities:
    - Parse Python tracebacks
    - Classify errors
    - Extract file/line information
    - Identify likely causes
    - Generate debugging suggestions
    - Detect common dependency/import problems
    - Detect common Windows/Python errors
    - Produce structured results for the coding agent

This module analyzes errors only.
It does not automatically modify user code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ErrorLocation:
    """Location where an error occurred."""

    file_path: str | None = None
    line: int | None = None
    column: int | None = None
    function: str | None = None
    source_line: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "function": self.function,
            "source_line": self.source_line,
        }


@dataclass
class ErrorSuggestion:
    """Suggested action for fixing an error."""

    title: str
    description: str
    confidence: float = 0.5
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "command": self.command,
        }


@dataclass
class ErrorAnalysis:
    """Complete analysis of an error."""

    error_type: str
    message: str
    category: str
    severity: str
    location: ErrorLocation = field(
        default_factory=ErrorLocation
    )
    likely_causes: list[str] = field(
        default_factory=list
    )
    suggestions: list[ErrorSuggestion] = field(
        default_factory=list
    )
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "category": self.category,
            "severity": self.severity,
            "location": self.location.to_dict(),
            "likely_causes": self.likely_causes,
            "suggestions": [
                suggestion.to_dict()
                for suggestion in self.suggestions
            ],
            "traceback": self.traceback,
        }


class ErrorAnalyzer:
    """RENIX error-analysis engine."""

    TRACEBACK_FRAME = re.compile(
        r'File "(.+?)", line (\d+)(?:, in (.+))?'
    )

    ERROR_LINE = re.compile(
        r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Interrupt))"
        r"(?::\s*(?P<message>.*))?$",
        re.MULTILINE,
    )

    IMPORT_ERROR = re.compile(
        r"(?:No module named|cannot import name)"
        r"(?:\s+['\"]?([^'\"\s]+)['\"]?)?",
        re.IGNORECASE,
    )

    FILE_NOT_FOUND = re.compile(
        r"(?:No such file or directory|FileNotFoundError)",
        re.IGNORECASE,
    )

    PERMISSION_ERROR = re.compile(
        r"(?:PermissionError|access is denied|permission denied)",
        re.IGNORECASE,
    )

    CONNECTION_ERROR = re.compile(
        r"(?:ConnectionError|ConnectionRefusedError|"
        r"ConnectionResetError|Connection timed out|"
        r"Max retries exceeded)",
        re.IGNORECASE,
    )

    TIMEOUT_ERROR = re.compile(
        r"(?:TimeoutError|timed out|timeout)",
        re.IGNORECASE,
    )

    MEMORY_ERROR = re.compile(
        r"(?:MemoryError|out of memory|"
        r"cannot allocate memory)",
        re.IGNORECASE,
    )

    KEY_ERROR = re.compile(
        r"(?:KeyError)",
        re.IGNORECASE,
    )

    ATTRIBUTE_ERROR = re.compile(
        r"(?:AttributeError)",
        re.IGNORECASE,
    )

    TYPE_ERROR = re.compile(
        r"(?:TypeError)",
        re.IGNORECASE,
    )

    VALUE_ERROR = re.compile(
        r"(?:ValueError)",
        re.IGNORECASE,
    )

    INDEX_ERROR = re.compile(
        r"(?:IndexError)",
        re.IGNORECASE,
    )

    NAME_ERROR = re.compile(
        r"(?:NameError)",
        re.IGNORECASE,
    )

    SYNTAX_ERROR = re.compile(
        r"(?:SyntaxError|IndentationError|"
        r"TabError)",
        re.IGNORECASE,
    )

    MODULE_NOT_FOUND = re.compile(
        r"(?:ModuleNotFoundError)",
        re.IGNORECASE,
    )

    JSON_ERROR = re.compile(
        r"(?:JSONDecodeError|"
        r"Expecting property name|"
        r"Expecting value|"
        r"Invalid JSON)",
        re.IGNORECASE,
    )

    NETWORK_STATUS = re.compile(
        r"\b(4\d\d|5\d\d)\b"
    )

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
    ) -> None:

        self.project_root = (
            Path(project_root)
            .expanduser()
            .resolve()
            if project_root
            else None
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
        error_text: str,
        *,
        source_file: str | Path | None = None,
    ) -> ErrorAnalysis:

        if not error_text:
            return ErrorAnalysis(
                error_type="UnknownError",
                message="No error information was provided.",
                category="unknown",
                severity="info",
            )

        text = str(error_text).strip()

        error_type, message = (
            self._extract_error_type_and_message(
                text
            )
        )

        location = self._extract_location(
            text,
            source_file=source_file,
        )

        category = self.classify(
            error_type,
            message,
        )

        severity = self.determine_severity(
            error_type,
            message,
        )

        causes = self.find_likely_causes(
            error_type,
            message,
            text,
        )

        suggestions = self.generate_suggestions(
            error_type,
            message,
            category,
        )

        return ErrorAnalysis(
            error_type=error_type,
            message=message,
            category=category,
            severity=severity,
            location=location,
            likely_causes=causes,
            suggestions=suggestions,
            traceback=text,
        )

    # =========================================================
    # ERROR EXTRACTION
    # =========================================================

    def _extract_error_type_and_message(
        self,
        text: str,
    ) -> tuple[str, str]:

        matches = list(
            self.ERROR_LINE.finditer(
                text
            )
        )

        if matches:

            match = matches[-1]

            error_type = (
                match.group(
                    "type"
                )
                or "UnknownError"
            )

            message = (
                match.group(
                    "message"
                )
                or ""
            ).strip()

            return (
                error_type,
                message,
            )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if lines:
            return (
                "UnknownError",
                lines[-1],
            )

        return (
            "UnknownError",
            text,
        )

    def _extract_location(
        self,
        text: str,
        *,
        source_file: str | Path | None = None,
    ) -> ErrorLocation:

        matches = list(
            self.TRACEBACK_FRAME.finditer(
                text
            )
        )

        if matches:

            match = matches[-1]

            file_path = match.group(
                1
            )

            line = int(
                match.group(
                    2
                )
            )

            function = match.group(
                3
            )

            source_line = self._read_source_line(
                file_path,
                line,
            )

            return ErrorLocation(
                file_path=file_path,
                line=line,
                function=function,
                source_line=source_line,
            )

        if source_file is not None:

            path = Path(
                source_file
            ).expanduser()

            return ErrorLocation(
                file_path=str(path)
            )

        return ErrorLocation()

    # =========================================================
    # CLASSIFICATION
    # =========================================================

    def classify(
        self,
        error_type: str,
        message: str,
    ) -> str:

        combined = (
            f"{error_type} {message}"
        ).lower()

        if self.SYNTAX_ERROR.search(
            combined
        ):
            return "syntax"

        if (
            self.MODULE_NOT_FOUND.search(
                combined
            )
            or self.IMPORT_ERROR.search(
                combined
            )
        ):
            return "dependency"

        if self.FILE_NOT_FOUND.search(
            combined
        ):
            return "filesystem"

        if self.PERMISSION_ERROR.search(
            combined
        ):
            return "permissions"

        if self.CONNECTION_ERROR.search(
            combined
        ):
            return "network"

        if self.TIMEOUT_ERROR.search(
            combined
        ):
            return "timeout"

        if self.MEMORY_ERROR.search(
            combined
        ):
            return "resource"

        if self.KEY_ERROR.search(
            combined
        ):
            return "data"

        if self.ATTRIBUTE_ERROR.search(
            combined
        ):
            return "object"

        if self.TYPE_ERROR.search(
            combined
        ):
            return "type"

        if self.VALUE_ERROR.search(
            combined
        ):
            return "value"

        if self.INDEX_ERROR.search(
            combined
        ):
            return "index"

        if self.NAME_ERROR.search(
            combined
        ):
            return "name"

        if self.JSON_ERROR.search(
            combined
        ):
            return "data_format"

        if self.NETWORK_STATUS.search(
            combined
        ):
            return "network"

        return "runtime"

    # =========================================================
    # SEVERITY
    # =========================================================

    def determine_severity(
        self,
        error_type: str,
        message: str,
    ) -> str:

        combined = (
            f"{error_type} {message}"
        ).lower()

        if any(
            keyword in combined
            for keyword in (
                "syntaxerror",
                "indentationerror",
                "taberror",
            )
        ):
            return "high"

        if any(
            keyword in combined
            for keyword in (
                "permissionerror",
                "memoryerror",
                "security",
                "authentication",
                "authorization",
            )
        ):
            return "high"

        if any(
            keyword in combined
            for keyword in (
                "connection",
                "timeout",
                "filenotfound",
            )
        ):
            return "medium"

        return "medium"

    # =========================================================
    # LIKELY CAUSES
    # =========================================================

    def find_likely_causes(
        self,
        error_type: str,
        message: str,
        traceback_text: str,
    ) -> list[str]:

        combined = (
            f"{error_type} {message} "
            f"{traceback_text}"
        ).lower()

        causes: list[str] = []

        if self.SYNTAX_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "Invalid Python syntax.",
                    "Missing closing bracket, quote, or parenthesis.",
                    "Incorrect indentation.",
                    "Unexpected token or character.",
                ]
            )

        elif self.MODULE_NOT_FOUND.search(
            combined
        ):

            causes.extend(
                [
                    "The required Python package may not be installed.",
                    "The package may be installed in a different Python environment.",
                    "The import name may be incorrect.",
                    "The virtual environment may not be activated.",
                ]
            )

        elif self.IMPORT_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "The requested symbol may not exist in the module.",
                    "The installed package version may be incompatible.",
                    "A circular import may be occurring.",
                ]
            )

        elif self.FILE_NOT_FOUND.search(
            combined
        ):

            causes.extend(
                [
                    "The requested file or directory does not exist.",
                    "The path may be relative to the wrong working directory.",
                    "The file may have been renamed or moved.",
                ]
            )

        elif self.PERMISSION_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "The current process lacks required permissions.",
                    "The file may be locked by another application.",
                    "The target directory may require elevated privileges.",
                ]
            )

        elif self.TYPE_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "A value has an unexpected data type.",
                    "A function received incompatible arguments.",
                    "An object may not support the requested operation.",
                ]
            )

        elif self.ATTRIBUTE_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "The object does not contain the requested attribute.",
                    "The wrong object may have been passed.",
                    "The expected library API may have changed.",
                ]
            )

        elif self.KEY_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "A dictionary key does not exist.",
                    "Input data may have a different schema.",
                    "The key may contain unexpected capitalization or formatting.",
                ]
            )

        elif self.INDEX_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "The requested list or sequence index is out of range.",
                    "The sequence may contain fewer elements than expected.",
                    "An empty collection may not have been handled.",
                ]
            )

        elif self.NAME_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "A variable or function has not been defined.",
                    "The name may contain a typo.",
                    "The required import may be missing.",
                ]
            )

        elif self.VALUE_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "A value has an invalid format or range.",
                    "Input validation may be incomplete.",
                    "A conversion operation received unexpected input.",
                ]
            )

        elif self.JSON_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "The JSON document is malformed.",
                    "A trailing comma may be present.",
                    "A key may not use valid JSON quoting.",
                    "The input may not actually be JSON.",
                ]
            )

        elif self.CONNECTION_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "The target service may be offline.",
                    "The hostname or port may be incorrect.",
                    "A firewall may be blocking the connection.",
                    "The network connection may be unavailable.",
                ]
            )

        elif self.TIMEOUT_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "The operation took longer than the configured timeout.",
                    "The remote service may be slow or unavailable.",
                    "The program may be stuck waiting for an operation.",
                ]
            )

        elif self.MEMORY_ERROR.search(
            combined
        ):

            causes.extend(
                [
                    "The program may be using too much memory.",
                    "A large file or dataset may have been loaded entirely into memory.",
                    "There may be an unintended memory leak.",
                ]
            )

        if not causes:
            causes.append(
                "The error requires inspection of the traceback and surrounding code."
            )

        return self._unique(
            causes
        )

    # =========================================================
    # SUGGESTIONS
    # =========================================================

    def generate_suggestions(
        self,
        error_type: str,
        message: str,
        category: str,
    ) -> list[ErrorSuggestion]:

        combined = (
            f"{error_type} {message}"
        ).lower()

        suggestions: list[
            ErrorSuggestion
        ] = []

        if category == "syntax":

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Inspect the reported line",
                        description=(
                            "Check the reported line and the line immediately before it "
                            "for missing brackets, quotes, commas, or invalid syntax."
                        ),
                        confidence=0.95,
                    ),
                    ErrorSuggestion(
                        title="Run Python compilation",
                        description=(
                            "Compile the file without executing it to confirm whether "
                            "the syntax problem has been resolved."
                        ),
                        confidence=0.85,
                        command="python -m py_compile <file>",
                    ),
                ]
            )

        elif category == "dependency":

            package = self._extract_package_name(
                message
            )

            command = (
                f"python -m pip install {package}"
                if package
                else "python -m pip install <package>"
            )

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Check the package",
                        description=(
                            "Verify that the required package is installed "
                            "in the Python environment running RENIX."
                        ),
                        confidence=0.92,
                        command=command,
                    ),
                    ErrorSuggestion(
                        title="Check the Python environment",
                        description=(
                            "Confirm that RENIX and pip are using the same "
                            "Python interpreter."
                        ),
                        confidence=0.88,
                        command="python -m pip --version",
                    ),
                ]
            )

        elif category == "filesystem":

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Verify the path",
                        description=(
                            "Confirm that the requested file or directory exists "
                            "and that the path is correct."
                        ),
                        confidence=0.95,
                    ),
                    ErrorSuggestion(
                        title="Check the working directory",
                        description=(
                            "Relative paths depend on the process working directory. "
                            "Use an absolute path or verify the current directory."
                        ),
                        confidence=0.85,
                    ),
                ]
            )

        elif category == "permissions":

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Check permissions",
                        description=(
                            "Verify that the current user has access to the "
                            "requested file or directory."
                        ),
                        confidence=0.94,
                    ),
                    ErrorSuggestion(
                        title="Check file locks",
                        description=(
                            "Another application may currently have the file open."
                        ),
                        confidence=0.70,
                    ),
                ]
            )

        elif category == "type":

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Inspect the value type",
                        description=(
                            "Check the actual runtime type of each value involved "
                            "in the failing operation."
                        ),
                        confidence=0.90,
                    ),
                    ErrorSuggestion(
                        title="Validate function arguments",
                        description=(
                            "Compare the supplied arguments with the function's "
                            "expected parameter types."
                        ),
                        confidence=0.85,
                    ),
                ]
            )

        elif category == "object":

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Inspect the object",
                        description=(
                            "Check the object's actual class and available attributes."
                        ),
                        confidence=0.92,
                    ),
                    ErrorSuggestion(
                        title="Check the API",
                        description=(
                            "Verify that the attribute or method exists in the "
                            "installed version of the dependency."
                        ),
                        confidence=0.84,
                    ),
                ]
            )

        elif category == "network":

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Check connectivity",
                        description=(
                            "Verify that the target host and service are reachable."
                        ),
                        confidence=0.90,
                    ),
                    ErrorSuggestion(
                        title="Check configuration",
                        description=(
                            "Verify hostname, port, API endpoint, credentials, "
                            "and proxy settings."
                        ),
                        confidence=0.85,
                    ),
                ]
            )

        elif category == "timeout":

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Check the operation",
                        description=(
                            "Determine whether the operation is blocked or "
                            "taking longer than expected."
                        ),
                        confidence=0.90,
                    ),
                    ErrorSuggestion(
                        title="Review timeout settings",
                        description=(
                            "Increase the timeout only when the operation is "
                            "legitimately expected to take longer."
                        ),
                        confidence=0.75,
                    ),
                ]
            )

        elif category == "data_format":

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Validate the input",
                        description=(
                            "Inspect the input data around the reported failure "
                            "and verify that it follows the expected format."
                        ),
                        confidence=0.93,
                    ),
                    ErrorSuggestion(
                        title="Use a validator",
                        description=(
                            "Validate structured input before processing it."
                        ),
                        confidence=0.80,
                    ),
                ]
            )

        else:

            suggestions.extend(
                [
                    ErrorSuggestion(
                        title="Inspect the traceback",
                        description=(
                            "Start at the final traceback frame and inspect "
                            "the surrounding source code."
                        ),
                        confidence=0.88,
                    ),
                    ErrorSuggestion(
                        title="Reproduce the error",
                        description=(
                            "Run the smallest possible reproduction to isolate "
                            "the failing operation."
                        ),
                        confidence=0.82,
                    ),
                ]
            )

        if "deprecated" in combined:

            suggestions.append(
                ErrorSuggestion(
                    title="Check the current API",
                    description=(
                        "The error indicates that an API may have been deprecated. "
                        "Check the installed library version and current documentation."
                    ),
                    confidence=0.90,
                )
            )

        return suggestions

    # =========================================================
    # PACKAGE EXTRACTION
    # =========================================================

    def _extract_package_name(
        self,
        message: str,
    ) -> str | None:

        match = re.search(
            r"No module named ['\"]?([^'\"\s]+)",
            message,
            re.IGNORECASE,
        )

        if match:
            module = match.group(
                1
            )

            return module.split(
                "."
            )[0]

        return None

    # =========================================================
    # SOURCE READING
    # =========================================================

    def _read_source_line(
        self,
        file_path: str,
        line_number: int,
    ) -> str | None:

        try:

            path = Path(
                file_path
            ).expanduser()

            if not path.exists():
                return None

            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

            if (
                1
                <= line_number
                <= len(lines)
            ):
                return lines[
                    line_number - 1
                ].strip()

        except (
            OSError,
            UnicodeError,
        ):
            return None

        return None

    # =========================================================
    # BATCH ANALYSIS
    # =========================================================

    def analyze_many(
        self,
        errors: Iterable[str],
    ) -> list[ErrorAnalysis]:

        return [
            self.analyze(error)
            for error in errors
            if error
        ]

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _unique(
        values: Iterable[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for value in values:

            normalized = value.strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                normalized
            )

        return result


__all__ = [
    "ErrorAnalyzer",
    "ErrorAnalysis",
    "ErrorLocation",
    "ErrorSuggestion",
]


