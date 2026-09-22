"""
RENIX Code Analyzer
===================

Static analysis engine for the RENIX coding subsystem.

Responsibilities:
    - Analyze source files
    - Detect syntax errors
    - Detect common code problems
    - Count code statistics
    - Analyze Python AST when possible
    - Identify TODO/FIXME markers
    - Detect suspicious constructs
    - Produce structured analysis results

The analyzer does not modify source files.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    """A detected source-code issue."""

    severity: str
    message: str
    line: int | None = None
    column: int | None = None
    code: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "code": self.code,
            "source": self.source,
        }


@dataclass
class CodeStatistics:
    """Basic statistics about a source file."""

    total_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0
    characters: int = 0
    functions: int = 0
    classes: int = 0
    imports: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_lines": self.total_lines,
            "blank_lines": self.blank_lines,
            "comment_lines": self.comment_lines,
            "code_lines": self.code_lines,
            "characters": self.characters,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
        }


@dataclass
class AnalysisResult:
    """Complete analysis result."""

    path: str
    language: str
    valid: bool
    issues: list[Issue] = field(default_factory=list)
    statistics: CodeStatistics = field(
        default_factory=CodeStatistics
    )
    complexity: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "valid": self.valid,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "statistics": self.statistics.to_dict(),
            "complexity": self.complexity,
        }


class CodeAnalyzer:
    """Static analyzer used by RENIX."""

    LANGUAGE_EXTENSIONS = {
        ".py": "Python",
        ".pyw": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".java": "Java",
        ".c": "C",
        ".h": "C/C++",
        ".cpp": "C++",
        ".hpp": "C++",
        ".cs": "C#",
        ".go": "Go",
        ".rs": "Rust",
        ".php": "PHP",
        ".rb": "Ruby",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".dart": "Dart",
        ".sql": "SQL",
        ".sh": "Shell",
        ".ps1": "PowerShell",
        ".html": "HTML",
        ".css": "CSS",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
    }

    TODO_PATTERN = re.compile(
        r"\b(TODO|FIXME|XXX|HACK)\b",
        re.IGNORECASE,
    )

    SECRET_PATTERNS = [
        re.compile(
            r"(api[_-]?key|secret|password|token)"
            r"\s*[:=]\s*['\"][^'\"]+['\"]",
            re.IGNORECASE,
        ),
        re.compile(
            r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
    ]

    DANGEROUS_PYTHON_CALLS = {
        "eval",
        "exec",
        "compile",
        "__import__",
    }

    def __init__(
        self,
        *,
        max_file_size: int = 10 * 1024 * 1024,
        run_external_linters: bool = False,
    ) -> None:

        self.max_file_size = max_file_size
        self.run_external_linters = run_external_linters

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        target: str | Path,
    ) -> AnalysisResult | list[AnalysisResult]:

        path = Path(
            target
        ).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Target does not exist: {path}"
            )

        if path.is_dir():
            return self.analyze_directory(
                path
            )

        return self.analyze_file(
            path
        )

    def analyze_file(
        self,
        file_path: str | Path,
    ) -> AnalysisResult:

        path = Path(
            file_path
        ).expanduser().resolve()

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
                f"File exceeds maximum size: {self.max_file_size} bytes."
            )

        language = self.detect_language(
            path
        )

        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        statistics = self.calculate_statistics(
            content,
            language,
        )

        issues: list[Issue] = []

        if language == "Python":
            issues.extend(
                self._analyze_python(
                    content
                )
            )
        else:
            issues.extend(
                self._analyze_generic(
                    content,
                    language,
                )
            )

        issues.extend(
            self._find_todos(
                content
            )
        )

        issues.extend(
            self._find_possible_secrets(
                content
            )
        )

        complexity = self.calculate_complexity(
            content,
            language,
        )

        if self.run_external_linters:
            issues.extend(
                self._run_external_checks(
                    path,
                    language,
                )
            )

        valid = not any(
            issue.severity == "error"
            for issue in issues
        )

        return AnalysisResult(
            path=str(path),
            language=language,
            valid=valid,
            issues=issues,
            statistics=statistics,
            complexity=complexity,
        )

    def analyze_directory(
        self,
        directory: str | Path,
    ) -> list[AnalysisResult]:

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

        results: list[AnalysisResult] = []

        ignored = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".idea",
            ".vscode",
            "dist",
            "build",
        }

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            if any(
                part in ignored
                for part in path.parts
            ):
                continue

            if not self.is_source_file(
                path
            ):
                continue

            try:

                results.append(
                    self.analyze_file(
                        path
                    )
                )

            except Exception as exc:

                logger.warning(
                    "Could not analyze %s: %s",
                    path,
                    exc,
                )

                results.append(
                    AnalysisResult(
                        path=str(path),
                        language=self.detect_language(
                            path
                        ),
                        valid=False,
                        issues=[
                            Issue(
                                severity="error",
                                message=str(exc),
                            )
                        ],
                    )
                )

        return results

    # =========================================================
    # LANGUAGE
    # =========================================================

    def detect_language(
        self,
        file_path: str | Path,
    ) -> str:

        path = Path(
            file_path
        )

        if path.name == "Dockerfile":
            return "Dockerfile"

        if path.name == "Makefile":
            return "Makefile"

        return self.LANGUAGE_EXTENSIONS.get(
            path.suffix.lower(),
            "Unknown",
        )

    def is_source_file(
        self,
        file_path: str | Path,
    ) -> bool:

        path = Path(
            file_path
        )

        return (
            path.name in {
                "Dockerfile",
                "Makefile",
            }
            or path.suffix.lower()
            in self.LANGUAGE_EXTENSIONS
        )

    # =========================================================
    # PYTHON ANALYSIS
    # =========================================================

    def _analyze_python(
        self,
        content: str,
    ) -> list[Issue]:

        issues: list[Issue] = []

        try:

            tree = ast.parse(
                content
            )

        except SyntaxError as exc:

            issues.append(
                Issue(
                    severity="error",
                    message=exc.msg,
                    line=exc.lineno,
                    column=exc.offset,
                    code="PY-SYNTAX",
                    source="python",
                )
            )

            return issues

        for node in ast.walk(
            tree
        ):

            if isinstance(
                node,
                ast.Call,
            ):

                if isinstance(
                    node.func,
                    ast.Name,
                ):

                    function_name = (
                        node.func.id
                    )

                    if (
                        function_name
                        in self.DANGEROUS_PYTHON_CALLS
                    ):

                        issues.append(
                            Issue(
                                severity="warning",
                                message=(
                                    f"Use of potentially dangerous "
                                    f"built-in '{function_name}'."
                                ),
                                line=getattr(
                                    node,
                                    "lineno",
                                    None,
                                ),
                                column=getattr(
                                    node,
                                    "col_offset",
                                    None,
                                ),
                                code="PY-DANGEROUS",
                                source="python",
                            )
                        )

            if isinstance(
                node,
                ast.ExceptHandler,
            ):

                if node.type is None:

                    issues.append(
                        Issue(
                            severity="warning",
                            message=(
                                "Bare 'except' catches every exception."
                            ),
                            line=getattr(
                                node,
                                "lineno",
                                None,
                            ),
                            code="PY-BARE-EXCEPT",
                            source="python",
                        )
                    )

            if isinstance(
                node,
                ast.FunctionDef,
            ):

                if len(
                    node.body
                ) == 1 and isinstance(
                    node.body[0],
                    ast.Pass,
                ):

                    issues.append(
                        Issue(
                            severity="info",
                            message=(
                                f"Function '{node.name}' "
                                "contains only 'pass'."
                            ),
                            line=node.lineno,
                            code="PY-EMPTY-FUNCTION",
                            source="python",
                        )
                    )

        issues.extend(
            self._check_python_imports(
                tree
            )
        )

        issues.extend(
            self._check_python_mutable_defaults(
                tree
            )
        )

        return issues

    def _check_python_imports(
        self,
        tree: ast.AST,
    ) -> list[Issue]:

        issues: list[Issue] = []

        for node in ast.walk(
            tree
        ):

            if isinstance(
                node,
                ast.Import,
            ):

                for alias in node.names:

                    if alias.name == "*":

                        issues.append(
                            Issue(
                                severity="warning",
                                message=(
                                    "Wildcard import can make "
                                    "dependencies unclear."
                                ),
                                line=node.lineno,
                                code="PY-WILDCARD-IMPORT",
                                source="python",
                            )
                        )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):

                for alias in node.names:

                    if alias.name == "*":

                        issues.append(
                            Issue(
                                severity="warning",
                                message=(
                                    "Wildcard import can make "
                                    "dependencies unclear."
                                ),
                                line=node.lineno,
                                code="PY-WILDCARD-IMPORT",
                                source="python",
                            )
                        )

        return issues

    def _check_python_mutable_defaults(
        self,
        tree: ast.AST,
    ) -> list[Issue]:

        issues: list[Issue] = []

        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            defaults = list(
                node.args.defaults
            )

            defaults.extend(
                default
                for default in node.args.kw_defaults
                if default is not None
            )

            for default in defaults:

                if isinstance(
                    default,
                    (
                        ast.List,
                        ast.Dict,
                        ast.Set,
                    ),
                ):

                    issues.append(
                        Issue(
                            severity="warning",
                            message=(
                                f"Mutable default argument "
                                f"found in '{node.name}'."
                            ),
                            line=node.lineno,
                            code="PY-MUTABLE-DEFAULT",
                            source="python",
                        )
                    )

        return issues

    # =========================================================
    # GENERIC ANALYSIS
    # =========================================================

    def _analyze_generic(
        self,
        content: str,
        language: str,
    ) -> list[Issue]:

        issues: list[Issue] = []

        if language == "JSON":

            try:
                json.loads(
                    content
                )
            except json.JSONDecodeError as exc:

                issues.append(
                    Issue(
                        severity="error",
                        message=exc.msg,
                        line=exc.lineno,
                        column=exc.colno,
                        code="JSON-SYNTAX",
                        source="json",
                    )
                )

        # Generic suspicious patterns.
        lines = content.splitlines()

        for number, line in enumerate(
            lines,
            start=1,
        ):

            if "console.log(" in line:
                issues.append(
                    Issue(
                        severity="info",
                        message=(
                            "Debug logging detected."
                        ),
                        line=number,
                        code="DEBUG-LOG",
                        source=language,
                    )
                )

            if "TODO" in line.upper():
                # TODOs are separately reported,
                # so don't duplicate them here.
                pass

        return issues

    # =========================================================
    # TODO / FIXME
    # =========================================================

    def _find_todos(
        self,
        content: str,
    ) -> list[Issue]:

        issues: list[Issue] = []

        for number, line in enumerate(
            content.splitlines(),
            start=1,
        ):

            match = self.TODO_PATTERN.search(
                line
            )

            if not match:
                continue

            marker = match.group(
                1
            ).upper()

            issues.append(
                Issue(
                    severity="info",
                    message=(
                        f"{marker} marker found."
                    ),
                    line=number,
                    code=f"TODO-{marker}",
                    source="generic",
                )
            )

        return issues

    # =========================================================
    # SECRET DETECTION
    # =========================================================

    def _find_possible_secrets(
        self,
        content: str,
    ) -> list[Issue]:

        issues: list[Issue] = []

        for number, line in enumerate(
            content.splitlines(),
            start=1,
        ):

            for pattern in self.SECRET_PATTERNS:

                if pattern.search(
                    line
                ):

                    issues.append(
                        Issue(
                            severity="warning",
                            message=(
                                "Possible hard-coded secret "
                                "or credential detected."
                            ),
                            line=number,
                            code="SEC-HARDCODED-SECRET",
                            source="security",
                        )
                    )

                    break

        return issues

    # =========================================================
    # STATISTICS
    # =========================================================

    def calculate_statistics(
        self,
        content: str,
        language: str,
    ) -> CodeStatistics:

        lines = content.splitlines()

        statistics = CodeStatistics(
            total_lines=len(lines),
            characters=len(content),
        )

        for line in lines:

            stripped = line.strip()

            if not stripped:
                statistics.blank_lines += 1
                continue

            if self._is_comment(
                stripped,
                language,
            ):

                statistics.comment_lines += 1

            else:

                statistics.code_lines += 1

        if language == "Python":

            try:

                tree = ast.parse(
                    content
                )

                for node in ast.walk(
                    tree
                ):

                    if isinstance(
                        node,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    ):
                        statistics.functions += 1

                    elif isinstance(
                        node,
                        ast.ClassDef,
                    ):
                        statistics.classes += 1

                    elif isinstance(
                        node,
                        (
                            ast.Import,
                            ast.ImportFrom,
                        ),
                    ):
                        statistics.imports += 1

            except SyntaxError:
                pass

        return statistics

    @staticmethod
    def _is_comment(
        line: str,
        language: str,
    ) -> bool:

        if language in {
            "Python",
            "Shell",
            "Ruby",
            "YAML",
            "TOML",
            "Makefile",
        }:
            return line.startswith("#")

        if language in {
            "JavaScript",
            "TypeScript",
            "Java",
            "C",
            "C++",
            "C#",
            "Go",
            "Rust",
            "Kotlin",
            "Swift",
            "PHP",
            "CSS",
        }:
            return (
                line.startswith("//")
                or line.startswith("/*")
                or line.startswith("*")
            )

        if language == "SQL":
            return line.startswith("--")

        if language == "HTML":
            return line.startswith("<!--")

        return False

    # =========================================================
    # COMPLEXITY
    # =========================================================

    def calculate_complexity(
        self,
        content: str,
        language: str,
    ) -> int:

        if language == "Python":

            try:

                tree = ast.parse(
                    content
                )

            except SyntaxError:
                return 0

            complexity = 1

            decision_nodes = (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.Try,
                ast.IfExp,
                ast.BoolOp,
                ast.Match,
            )

            for node in ast.walk(
                tree
            ):

                if isinstance(
                    node,
                    decision_nodes,
                ):
                    complexity += 1

            return complexity

        # Generic approximation.
        patterns = (
            r"\bif\b",
            r"\belse\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"&&",
            r"\|\|",
        )

        complexity = 1

        for pattern in patterns:

            complexity += len(
                re.findall(
                    pattern,
                    content,
                    re.IGNORECASE,
                )
            )

        return complexity

    # =========================================================
    # EXTERNAL CHECKS
    # =========================================================

    def _run_external_checks(
        self,
        path: Path,
        language: str,
    ) -> list[Issue]:

        if language != "Python":
            return []

        try:

            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "py_compile",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        except (
            OSError,
            subprocess.SubprocessError,
        ) as exc:

            logger.warning(
                "External analysis failed: %s",
                exc,
            )

            return []

        if process.returncode == 0:
            return []

        return [
            Issue(
                severity="error",
                message=(
                    process.stderr.strip()
                    or "Python compilation failed."
                ),
                code="PY-COMPILE",
                source="py_compile",
            )
        ]


__all__ = [
    "Issue",
    "CodeStatistics",
    "AnalysisResult",
    "CodeAnalyzer",
]


