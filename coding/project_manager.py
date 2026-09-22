"""
RENIX Project Manager
=====================

Manages coding projects for RENIX.

Responsibilities:
    - Create projects
    - Open projects
    - Inspect project structure
    - Detect project type
    - Track the active project
    - Create common project files
    - Search project files
    - Validate project paths
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProjectInfo:
    """Information about a coding project."""

    name: str
    path: str
    project_type: str = "unknown"
    language: str | None = None
    files: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "project_type": self.project_type,
            "language": self.language,
            "files": list(self.files),
            "directories": list(self.directories),
        }


class ProjectManager:
    """Create, open and inspect RENIX coding projects."""

    PROJECT_MARKERS = {
        "pyproject.toml": (
            "python",
            "Python",
        ),
        "requirements.txt": (
            "python",
            "Python",
        ),
        "setup.py": (
            "python",
            "Python",
        ),
        "package.json": (
            "node",
            "JavaScript",
        ),
        "tsconfig.json": (
            "typescript",
            "TypeScript",
        ),
        "Cargo.toml": (
            "rust",
            "Rust",
        ),
        "go.mod": (
            "go",
            "Go",
        ),
        "pom.xml": (
            "java",
            "Java",
        ),
        "build.gradle": (
            "java",
            "Java",
        ),
        "CMakeLists.txt": (
            "cpp",
            "C/C++",
        ),
        "Makefile": (
            "make",
            None,
        ),
    }

    EXTENSIONS = {
        ".py": "Python",
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
        ".kts": "Kotlin",
        ".dart": "Dart",
        ".html": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
        ".sql": "SQL",
        ".sh": "Shell",
        ".ps1": "PowerShell",
    }

    DEFAULT_DIRECTORIES = (
        "src",
        "tests",
        "docs",
    )

    def __init__(
        self,
        workspace: str | os.PathLike[str] | None = None,
    ) -> None:

        if workspace is None:
            workspace = Path.cwd() / "projects"

        self.workspace = (
            Path(workspace)
            .expanduser()
            .resolve()
        )

        self.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.current_project: Path | None = None

    # =========================================================
    # PROJECT CREATION
    # =========================================================

    def create_project(
        self,
        project_path: str | os.PathLike[str],
        *,
        template: str | None = None,
        directories: list[str] | None = None,
        create_readme: bool = True,
    ) -> ProjectInfo:

        path = self._resolve_project_path(
            project_path
        )

        if path.exists() and any(
            path.iterdir()
        ):
            raise FileExistsError(
                f"Project directory is not empty: {path}"
            )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected_directories = (
            directories
            if directories is not None
            else list(self.DEFAULT_DIRECTORIES)
        )

        for directory in selected_directories:

            self._safe_project_path(
                path / directory
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        if create_readme:

            readme = path / "README.md"

            if not readme.exists():

                readme.write_text(
                    f"# {path.name}\n\n"
                    "Project created by RENIX.\n",
                    encoding="utf-8",
                )

        if template:
            self._apply_template(
                path,
                template,
            )

        self.current_project = path

        return self.inspect_project(
            path
        )

    # =========================================================
    # OPEN PROJECT
    # =========================================================

    def open_project(
        self,
        project_path: str | os.PathLike[str],
    ) -> ProjectInfo:

        path = (
            Path(project_path)
            .expanduser()
            .resolve()
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Project does not exist: {path}"
            )

        if not path.is_dir():

            raise NotADirectoryError(
                f"Project is not a directory: {path}"
            )

        self.current_project = path

        return self.inspect_project(
            path
        )

    def close_project(self) -> None:
        self.current_project = None

    # =========================================================
    # CURRENT PROJECT
    # =========================================================

    def get_current_project(
        self,
    ) -> Path | None:

        return self.current_project

    def require_current_project(
        self,
    ) -> Path:

        if self.current_project is None:

            raise RuntimeError(
                "No project is currently open."
            )

        return self.current_project

    # =========================================================
    # PROJECT INSPECTION
    # =========================================================

    def inspect_project(
        self,
        project_path: str | os.PathLike[str] | None = None,
        *,
        max_files: int = 5000,
    ) -> ProjectInfo:

        path = (
            self.require_current_project()
            if project_path is None
            else Path(project_path)
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Project does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"Project is not a directory: {path}"
            )

        files: list[str] = []
        directories: list[str] = []

        for item in self._walk_project(
            path
        ):

            relative = item.relative_to(
                path
            )

            relative_string = str(
                relative
            )

            if item.is_dir():

                directories.append(
                    relative_string
                )

            elif item.is_file():

                files.append(
                    relative_string
                )

                if len(files) >= max_files:
                    break

        project_type, language = (
            self.detect_project_type(
                path
            )
        )

        return ProjectInfo(
            name=path.name,
            path=str(path),
            project_type=project_type,
            language=language,
            files=sorted(files),
            directories=sorted(directories),
        )

    # =========================================================
    # PROJECT TYPE
    # =========================================================

    def detect_project_type(
        self,
        project_path: str | os.PathLike[str] | None = None,
    ) -> tuple[str, str | None]:

        path = (
            self.require_current_project()
            if project_path is None
            else Path(project_path)
            .expanduser()
            .resolve()
        )

        for marker, (
            project_type,
            language,
        ) in self.PROJECT_MARKERS.items():

            if (
                path / marker
            ).exists():

                return (
                    project_type,
                    language,
                )

        language_counts: dict[str, int] = {}

        for item in self._walk_project(
            path
        ):

            if not item.is_file():
                continue

            language = self.EXTENSIONS.get(
                item.suffix.lower()
            )

            if language:

                language_counts[
                    language
                ] = (
                    language_counts.get(
                        language,
                        0,
                    )
                    + 1
                )

        if language_counts:

            language = max(
                language_counts,
                key=language_counts.get,
            )

            return (
                language.lower(),
                language,
            )

        return (
            "unknown",
            None,
        )

    # =========================================================
    # FILES
    # =========================================================

    def list_files(
        self,
        project_path: str | os.PathLike[str] | None = None,
    ) -> list[Path]:

        path = (
            self.require_current_project()
            if project_path is None
            else Path(project_path)
            .expanduser()
            .resolve()
        )

        return [
            item
            for item in self._walk_project(
                path
            )
            if item.is_file()
        ]

    def list_directories(
        self,
        project_path: str | os.PathLike[str] | None = None,
    ) -> list[Path]:

        path = (
            self.require_current_project()
            if project_path is None
            else Path(project_path)
            .expanduser()
            .resolve()
        )

        return [
            item
            for item in self._walk_project(
                path
            )
            if item.is_dir()
        ]

    # =========================================================
    # SEARCH
    # =========================================================

    def search_files(
        self,
        pattern: str,
        project_path: str | os.PathLike[str] | None = None,
    ) -> list[Path]:

        path = (
            self.require_current_project()
            if project_path is None
            else Path(project_path)
            .expanduser()
            .resolve()
        )

        matches: list[Path] = []

        for item in self._walk_project(
            path
        ):

            if not item.is_file():
                continue

            if item.match(pattern):
                matches.append(item)

        return matches

    # =========================================================
    # CREATE FILE
    # =========================================================

    def create_file(
        self,
        relative_path: str | os.PathLike[str],
        *,
        content: str = "",
        overwrite: bool = False,
    ) -> Path:

        project = self.require_current_project()

        path = self._safe_project_path(
            project
            / relative_path
        )

        if path.exists() and not overwrite:

            raise FileExistsError(
                f"File already exists: {path}"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

        return path

    # =========================================================
    # CREATE DIRECTORY
    # =========================================================

    def create_directory(
        self,
        relative_path: str | os.PathLike[str],
    ) -> Path:

        project = self.require_current_project()

        path = self._safe_project_path(
            project
            / relative_path
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    # =========================================================
    # PROJECT CONFIGURATION
    # =========================================================

    def save_project_info(
        self,
        info: ProjectInfo,
        filename: str = ".renix-project.json",
    ) -> Path:

        project = Path(
            info.path
        ).resolve()

        path = self._safe_project_path(
            project / filename
        )

        path.write_text(
            json.dumps(
                info.to_dict(),
                indent=4,
            ),
            encoding="utf-8",
        )

        return path

    def load_project_info(
        self,
        project_path: str | os.PathLike[str] | None = None,
        filename: str = ".renix-project.json",
    ) -> ProjectInfo | None:

        project = (
            self.require_current_project()
            if project_path is None
            else Path(project_path)
            .expanduser()
            .resolve()
        )

        path = self._safe_project_path(
            project / filename
        )

        if not path.exists():
            return None

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return ProjectInfo(
            name=data.get(
                "name",
                project.name,
            ),
            path=data.get(
                "path",
                str(project),
            ),
            project_type=data.get(
                "project_type",
                "unknown",
            ),
            language=data.get(
                "language"
            ),
            files=data.get(
                "files",
                [],
            ),
            directories=data.get(
                "directories",
                [],
            ),
        )

    # =========================================================
    # PROJECT TEMPLATES
    # =========================================================

    def _apply_template(
        self,
        project_path: Path,
        template: str,
    ) -> None:

        template = template.lower().strip()

        if template == "python":

            (project_path / "src").mkdir(
                exist_ok=True
            )

            (project_path / "tests").mkdir(
                exist_ok=True
            )

            (project_path / "requirements.txt").touch()

            (project_path / "src" / "__init__.py").touch()

            (project_path / "tests" / "__init__.py").touch()

            (project_path / "main.py").write_text(
                'def main():\n'
                '    print("Hello from RENIX")\n'
                '\n'
                '\n'
                'if __name__ == "__main__":\n'
                '    main()\n',
                encoding="utf-8",
            )

        elif template == "node":

            (project_path / "src").mkdir(
                exist_ok=True
            )

            package = {
                "name": project_path.name.lower(),
                "version": "1.0.0",
                "private": True,
                "scripts": {
                    "start": "node src/index.js",
                },
            }

            (
                project_path / "package.json"
            ).write_text(
                json.dumps(
                    package,
                    indent=2,
                ),
                encoding="utf-8",
            )

            (
                project_path
                / "src"
                / "index.js"
            ).write_text(
                'console.log("Hello from RENIX");\n',
                encoding="utf-8",
            )

        elif template == "web":

            (project_path / "src").mkdir(
                exist_ok=True
            )

            (
                project_path
                / "src"
                / "index.html"
            ).write_text(
                "<!DOCTYPE html>\n"
                "<html>\n"
                "<head>\n"
                "    <meta charset=\"UTF-8\">\n"
                f"    <title>{project_path.name}</title>\n"
                "</head>\n"
                "<body>\n"
                "    <h1>Hello from RENIX</h1>\n"
                "</body>\n"
                "</html>\n",
                encoding="utf-8",
            )

            (
                project_path
                / "src"
                / "style.css"
            ).touch()

            (
                project_path
                / "src"
                / "script.js"
            ).touch()

        elif template in {
            "",
            "basic",
        }:

            return

        else:

            logger.warning(
                "Unknown project template: %s",
                template,
            )

    # =========================================================
    # PATH SAFETY
    # =========================================================

    def _resolve_project_path(
        self,
        project_path: str | os.PathLike[str],
    ) -> Path:

        path = (
            Path(project_path)
            .expanduser()
        )

        if not path.is_absolute():
            path = self.workspace / path

        return path.resolve()

    @staticmethod
    def _safe_project_path(
        path: Path,
    ) -> Path:

        return path.resolve()

    # =========================================================
    # WALK
    # =========================================================

    @staticmethod
    def _walk_project(
        root: Path,
    ):

        ignored = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            ".idea",
            ".vscode",
            ".pytest_cache",
            ".mypy_cache",
        }

        for current_root, directories, files in os.walk(
            root
        ):

            directories[:] = [
                directory
                for directory in directories
                if directory not in ignored
            ]

            current = Path(
                current_root
            )

            for directory in directories:

                yield current / directory

            for filename in files:

                yield current / filename


__all__ = [
    "ProjectInfo",
    "ProjectManager",
]


