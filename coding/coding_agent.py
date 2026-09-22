"""
RENIX Coding Agent
==================

High-level coding agent responsible for coordinating the
RENIX coding subsystem.

The agent can:
    - Inspect projects
    - Read source files
    - Create and modify code
    - Analyze code
    - Diagnose errors
    - Run tests
    - Build projects
    - Manage dependencies
    - Execute terminal commands through TerminalManager
    - Work with Git
    - Generate documentation

This module acts as an orchestration layer. Detailed operations
remain inside the specialized coding modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CodingAgent:
    """
    High-level interface for RENIX coding operations.
    """

    def __init__(
        self,
        project_manager: Any = None,
        code_reader: Any = None,
        code_writer: Any = None,
        code_editor: Any = None,
        code_analyzer: Any = None,
        debugger: Any = None,
        error_analyzer: Any = None,
        test_runner: Any = None,
        build_manager: Any = None,
        dependency_manager: Any = None,
        terminal_manager: Any = None,
        git_manager: Any = None,
        documentation: Any = None,
    ) -> None:

        self.project_manager = project_manager
        self.code_reader = code_reader
        self.code_writer = code_writer
        self.code_editor = code_editor
        self.code_analyzer = code_analyzer
        self.debugger = debugger
        self.error_analyzer = error_analyzer
        self.test_runner = test_runner
        self.build_manager = build_manager
        self.dependency_manager = dependency_manager
        self.terminal_manager = terminal_manager
        self.git_manager = git_manager
        self.documentation = documentation

        self.current_project: Path | None = None

    # =========================================================
    # PROJECT MANAGEMENT
    # =========================================================

    def open_project(
        self,
        project_path: str | Path,
    ) -> dict[str, Any]:

        path = Path(
            project_path
        ).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Project does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"Project path is not a directory: {path}"
            )

        self.current_project = path

        if self.project_manager is not None:

            method = getattr(
                self.project_manager,
                "open_project",
                None,
            )

            if callable(method):

                result = method(
                    path
                )

                return self._normalize_result(
                    result
                )

        return {
            "success": True,
            "project": str(path),
        }

    def create_project(
        self,
        project_path: str | Path,
        *,
        template: str | None = None,
    ) -> dict[str, Any]:

        path = Path(
            project_path
        ).expanduser().resolve()

        if self.project_manager is None:
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            return {
                "success": True,
                "project": str(path),
                "template": template,
            }

        method = getattr(
            self.project_manager,
            "create_project",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "ProjectManager does not support create_project()."
            )

        result = method(
            path,
            template=template,
        )

        self.current_project = path

        return self._normalize_result(
            result
        )

    def inspect_project(
        self,
    ) -> Any:

        manager = self._require(
            self.project_manager,
            "ProjectManager",
        )

        method = getattr(
            manager,
            "inspect_project",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "ProjectManager does not support inspect_project()."
            )

        return method(
            self.current_project
        )

    # =========================================================
    # FILE READING
    # =========================================================

    def read_file(
        self,
        file_path: str | Path,
    ) -> str:

        reader = self._require(
            self.code_reader,
            "CodeReader",
        )

        path = self._resolve_path(
            file_path
        )

        method = getattr(
            reader,
            "read",
            None,
        )

        if not callable(method):

            method = getattr(
                reader,
                "read_file",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "CodeReader does not provide a read method."
            )

        return method(
            path
        )

    # =========================================================
    # FILE WRITING
    # =========================================================

    def write_file(
        self,
        file_path: str | Path,
        content: str,
    ) -> Any:

        writer = self._require(
            self.code_writer,
            "CodeWriter",
        )

        path = self._resolve_path(
            file_path
        )

        method = getattr(
            writer,
            "write",
            None,
        )

        if not callable(method):

            method = getattr(
                writer,
                "write_file",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "CodeWriter does not provide a write method."
            )

        return method(
            path,
            content,
        )

    # =========================================================
    # CODE EDITING
    # =========================================================

    def edit_file(
        self,
        file_path: str | Path,
        *,
        old_text: str | None = None,
        new_text: str | None = None,
        operation: str = "replace",
    ) -> Any:

        editor = self._require(
            self.code_editor,
            "CodeEditor",
        )

        path = self._resolve_path(
            file_path
        )

        if hasattr(
            editor,
            "edit",
        ):

            return editor.edit(
                path,
                old_text=old_text,
                new_text=new_text,
                operation=operation,
            )

        if operation == "replace":

            method = getattr(
                editor,
                "replace",
                None,
            )

            if callable(method):

                return method(
                    path,
                    old_text,
                    new_text,
                )

        raise RuntimeError(
            "CodeEditor does not support the requested operation."
        )

    # =========================================================
    # CODE ANALYSIS
    # =========================================================

    def analyze(
        self,
        target: str | Path | None = None,
    ) -> Any:

        analyzer = self._require(
            self.code_analyzer,
            "CodeAnalyzer",
        )

        target_path = (
            self._resolve_path(target)
            if target is not None
            else self.current_project
        )

        if target_path is None:
            raise RuntimeError(
                "No project or target has been selected."
            )

        method = getattr(
            analyzer,
            "analyze",
            None,
        )

        if not callable(method):

            method = getattr(
                analyzer,
                "analyze_file",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "CodeAnalyzer does not provide an analyze method."
            )

        return method(
            target_path
        )

    # =========================================================
    # DEBUGGING
    # =========================================================

    def debug(
        self,
        target: str | Path | None = None,
        *,
        error: str | None = None,
    ) -> Any:

        debugger = self._require(
            self.debugger,
            "Debugger",
        )

        target_path = (
            self._resolve_path(target)
            if target is not None
            else self.current_project
        )

        method = getattr(
            debugger,
            "debug",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "Debugger does not provide debug()."
            )

        return method(
            target_path,
            error=error,
        )

    def analyze_error(
        self,
        error: str,
        *,
        context: str | None = None,
    ) -> Any:

        analyzer = self._require(
            self.error_analyzer,
            "ErrorAnalyzer",
        )

        method = getattr(
            analyzer,
            "analyze",
            None,
        )

        if not callable(method):

            method = getattr(
                analyzer,
                "analyze_error",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "ErrorAnalyzer does not provide an analysis method."
            )

        return method(
            error,
            context=context,
        )

    # =========================================================
    # TESTING
    # =========================================================

    def run_tests(
        self,
        *,
        target: str | Path | None = None,
        test_command: str | None = None,
    ) -> Any:

        runner = self._require(
            self.test_runner,
            "TestRunner",
        )

        target_path = (
            self._resolve_path(target)
            if target is not None
            else self.current_project
        )

        method = getattr(
            runner,
            "run",
            None,
        )

        if not callable(method):

            method = getattr(
                runner,
                "run_tests",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "TestRunner does not provide run_tests()."
            )

        if test_command is not None:

            return method(
                target_path,
                command=test_command,
            )

        return method(
            target_path
        )

    # =========================================================
    # BUILD
    # =========================================================

    def build(
        self,
        *,
        target: str | Path | None = None,
        command: str | None = None,
    ) -> Any:

        builder = self._require(
            self.build_manager,
            "BuildManager",
        )

        target_path = (
            self._resolve_path(target)
            if target is not None
            else self.current_project
        )

        method = getattr(
            builder,
            "build",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "BuildManager does not provide build()."
            )

        return method(
            target_path,
            command=command,
        )

    # =========================================================
    # DEPENDENCIES
    # =========================================================

    def install_dependency(
        self,
        package: str,
    ) -> Any:

        manager = self._require(
            self.dependency_manager,
            "DependencyManager",
        )

        method = getattr(
            manager,
            "install",
            None,
        )

        if not callable(method):

            method = getattr(
                manager,
                "install_dependency",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "DependencyManager does not provide install()."
            )

        return method(
            package,
            project_path=self.current_project,
        )

    def remove_dependency(
        self,
        package: str,
    ) -> Any:

        manager = self._require(
            self.dependency_manager,
            "DependencyManager",
        )

        method = getattr(
            manager,
            "remove",
            None,
        )

        if not callable(method):

            method = getattr(
                manager,
                "remove_dependency",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "DependencyManager does not provide remove()."
            )

        return method(
            package,
            project_path=self.current_project,
        )

    def list_dependencies(
        self,
    ) -> Any:

        manager = self._require(
            self.dependency_manager,
            "DependencyManager",
        )

        method = getattr(
            manager,
            "list",
            None,
        )

        if not callable(method):

            method = getattr(
                manager,
                "list_dependencies",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "DependencyManager does not provide list()."
            )

        return method(
            project_path=self.current_project
        )

    # =========================================================
    # TERMINAL
    # =========================================================

    def run_command(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> Any:

        terminal = self._require(
            self.terminal_manager,
            "TerminalManager",
        )

        working_directory = (
            self._resolve_path(cwd)
            if cwd is not None
            else self.current_project
        )

        method = getattr(
            terminal,
            "run",
            None,
        )

        if not callable(method):

            method = getattr(
                terminal,
                "execute",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "TerminalManager does not provide a command execution method."
            )

        kwargs: dict[str, Any] = {}

        if working_directory is not None:
            kwargs["cwd"] = working_directory

        if timeout is not None:
            kwargs["timeout"] = timeout

        return method(
            command,
            **kwargs,
        )

    # =========================================================
    # GIT
    # =========================================================

    def git_status(
        self,
    ) -> Any:

        manager = self._require(
            self.git_manager,
            "GitManager",
        )

        method = getattr(
            manager,
            "status",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "GitManager does not provide status()."
            )

        return method(
            self.current_project
        )

    def git_commit(
        self,
        message: str,
    ) -> Any:

        manager = self._require(
            self.git_manager,
            "GitManager",
        )

        method = getattr(
            manager,
            "commit",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "GitManager does not provide commit()."
            )

        return method(
            message,
            project_path=self.current_project,
        )

    def git_push(
        self,
    ) -> Any:

        manager = self._require(
            self.git_manager,
            "GitManager",
        )

        method = getattr(
            manager,
            "push",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                "GitManager does not provide push()."
            )

        return method(
            project_path=self.current_project
        )

    # =========================================================
    # DOCUMENTATION
    # =========================================================

    def generate_documentation(
        self,
        target: str | Path | None = None,
    ) -> Any:

        documentation = self._require(
            self.documentation,
            "DocumentationManager",
        )

        target_path = (
            self._resolve_path(target)
            if target is not None
            else self.current_project
        )

        method = getattr(
            documentation,
            "generate",
            None,
        )

        if not callable(method):

            method = getattr(
                documentation,
                "generate_documentation",
                None,
            )

        if not callable(method):
            raise RuntimeError(
                "DocumentationManager does not provide documentation generation."
            )

        return method(
            target_path
        )

    # =========================================================
    # HIGH-LEVEL WORKFLOW
    # =========================================================

    def inspect_and_analyze(
        self,
        target: str | Path | None = None,
    ) -> dict[str, Any]:

        target_path = (
            self._resolve_path(target)
            if target is not None
            else self.current_project
        )

        if target_path is None:
            raise RuntimeError(
                "No target selected."
            )

        result: dict[str, Any] = {
            "target": str(target_path),
        }

        try:
            result["analysis"] = self.analyze(
                target_path
            )
        except Exception as exc:
            logger.exception(
                "Code analysis failed."
            )

            result["analysis_error"] = str(
                exc
            )

        try:
            result["project"] = self.inspect_project()
        except Exception as exc:
            logger.exception(
                "Project inspection failed."
            )

            result["project_error"] = str(
                exc
            )

        return result

    def diagnose(
        self,
        error: str,
        *,
        target: str | Path | None = None,
    ) -> dict[str, Any]:

        result: dict[str, Any] = {
            "error": error,
        }

        try:
            result["error_analysis"] = (
                self.analyze_error(
                    error
                )
            )
        except Exception as exc:

            logger.exception(
                "Error analysis failed."
            )

            result["error_analysis_error"] = str(
                exc
            )

        try:
            result["code_analysis"] = self.analyze(
                target
            )
        except Exception as exc:

            logger.exception(
                "Code analysis failed."
            )

            result["code_analysis_error"] = str(
                exc
            )

        return result

    # =========================================================
    # HELPERS
    # =========================================================

    def _resolve_path(
        self,
        path: str | Path,
    ) -> Path:

        candidate = Path(
            path
        ).expanduser()

        if not candidate.is_absolute():

            if self.current_project is not None:
                candidate = (
                    self.current_project
                    / candidate
                )

        return candidate.resolve()

    @staticmethod
    def _require(
        service: Any,
        name: str,
    ) -> Any:

        if service is None:
            raise RuntimeError(
                f"{name} is not configured."
            )

        return service

    @staticmethod
    def _normalize_result(
        result: Any,
    ) -> dict[str, Any] | Any:

        if isinstance(
            result,
            dict,
        ):
            return result

        return {
            "success": True,
            "result": result,
        }


__all__ = [
    "CodingAgent",
]


