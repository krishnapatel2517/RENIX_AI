"""
RENIX AI - Coding Agent

Handles software-development tasks for RENIX.

Responsibilities:
- Read source files
- Create source files
- Edit source files
- Analyze code
- Search projects
- Run tests
- Run build commands
- Inspect errors
- Manage dependencies
- Work with Git
- Generate documentation

This agent provides the orchestration layer for RENIX's coding
subsystem. Individual coding operations are delegated to the
modules inside the `coding/` package when available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .base_agent import (
    AgentCapability,
    AgentPriority,
    AgentResult,
    AgentTask,
    BaseAgent,
)


class CodingAgent(BaseAgent):
    """
    RENIX software-development agent.

    Provides a unified interface for coding, project management,
    testing, debugging, dependency management and Git operations.
    """

    agent_name = "coding_agent"

    agent_description = (
        "Creates, reads, edits, analyzes, tests and manages "
        "software projects and source code."
    )

    agent_version = "1.0.0"

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: AgentPriority = AgentPriority.NORMAL,
        logger: Optional[logging.Logger] = None,
        event_callback=None,
        confirmation_callback=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            priority=priority,
            logger=logger,
            event_callback=event_callback,
            confirmation_callback=confirmation_callback,
            context=context,
        )

        self.project_manager = None
        self.code_reader = None
        self.code_writer = None
        self.code_editor = None
        self.code_analyzer = None
        self.debugger = None
        self.error_analyzer = None
        self.test_runner = None
        self.build_manager = None
        self.dependency_manager = None
        self.terminal_manager = None
        self.git_manager = None
        self.documentation = None

        self._modules_loaded = False

        self.workspace: Optional[Path] = None

        self.allowed_extensions = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".c",
            ".h",
            ".cpp",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".php",
            ".rb",
            ".swift",
            ".kt",
            ".kts",
            ".dart",
            ".html",
            ".css",
            ".scss",
            ".sass",
            ".vue",
            ".svelte",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".sql",
            ".sh",
            ".bat",
            ".ps1",
            ".md",
            ".txt",
        }

        self.blocked_directories = {
            ".git",
            ".svn",
            ".hg",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "System Volume Information",
        }

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def _register_default_capabilities(self) -> None:
        """Register coding capabilities."""

        super()._register_default_capabilities()

        capabilities = [
            AgentCapability(
                name="read_code",
                description="Read source code from a project.",
            ),
            AgentCapability(
                name="write_code",
                description="Create or overwrite a source file.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="edit_code",
                description="Edit an existing source file.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="analyze_code",
                description="Analyze source code for problems.",
            ),
            AgentCapability(
                name="search_code",
                description="Search a project for code or text.",
            ),
            AgentCapability(
                name="create_project",
                description="Create a new software project.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="run_tests",
                description="Run project tests.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="run_build",
                description="Build a software project.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="debug",
                description="Analyze and debug an error.",
            ),
            AgentCapability(
                name="install_dependency",
                description="Install a project dependency.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="git",
                description="Perform Git operations.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="terminal",
                description="Run a terminal command.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="documentation",
                description="Generate project documentation.",
            ),
        ]

        for capability in capabilities:
            self.register_capability(capability)

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def on_initialize(self) -> bool:
        """
        Initialize the coding subsystem.
        """

        self._load_coding_modules()

        return True

    def _load_coding_modules(self) -> None:
        """Load RENIX coding modules."""

        if self._modules_loaded:
            return

        self._modules_loaded = True

        module_map = {
            "project_manager": (
                "coding.project_manager",
                "ProjectManager",
            ),
            "code_reader": (
                "coding.code_reader",
                "CodeReader",
            ),
            "code_writer": (
                "coding.code_writer",
                "CodeWriter",
            ),
            "code_editor": (
                "coding.code_editor",
                "CodeEditor",
            ),
            "code_analyzer": (
                "coding.code_analyzer",
                "CodeAnalyzer",
            ),
            "debugger": (
                "coding.debugger",
                "Debugger",
            ),
            "error_analyzer": (
                "coding.error_analyzer",
                "ErrorAnalyzer",
            ),
            "test_runner": (
                "coding.test_runner",
                "TestRunner",
            ),
            "build_manager": (
                "coding.build_manager",
                "BuildManager",
            ),
            "dependency_manager": (
                "coding.dependency_manager",
                "DependencyManager",
            ),
            "terminal_manager": (
                "coding.terminal_manager",
                "TerminalManager",
            ),
            "git_manager": (
                "coding.git_manager",
                "GitManager",
            ),
            "documentation": (
                "coding.documentation",
                "Documentation",
            ),
        }

        for attribute, (
            module_name,
            class_name,
        ) in module_map.items():

            try:

                module = __import__(
                    module_name,
                    fromlist=[class_name],
                )

                cls = getattr(
                    module,
                    class_name,
                )

                try:
                    instance = cls()
                except TypeError:
                    instance = cls

                setattr(
                    self,
                    attribute,
                    instance,
                )

            except Exception as exc:

                self.logger.debug(
                    "%s unavailable: %s",
                    class_name,
                    exc,
                )

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    async def execute(
        self,
        task: AgentTask,
    ) -> Any:
        """
        Execute a coding task.
        """

        action = self._resolve_action(
            task
        )

        handlers = {
            "read": self.read_code,
            "read_code": self.read_code,
            "write": self.write_code,
            "write_code": self.write_code,
            "create_file": self.write_code,
            "edit": self.edit_code,
            "edit_code": self.edit_code,
            "analyze": self.analyze_code,
            "analyze_code": self.analyze_code,
            "search": self.search_code,
            "search_code": self.search_code,
            "create_project": self.create_project,
            "run_tests": self.run_tests,
            "test": self.run_tests,
            "build": self.run_build,
            "run_build": self.run_build,
            "debug": self.debug,
            "install": self.install_dependency,
            "install_dependency": self.install_dependency,
            "dependency": self.install_dependency,
            "terminal": self.run_terminal,
            "command": self.run_terminal,
            "git": self.git,
            "documentation": self.generate_documentation,
            "docs": self.generate_documentation,
            "project_info": self.project_info,
        }

        handler = handlers.get(
            action
        )

        if handler is None:

            return self.failure_result(
                task,
                f"Unknown coding action: {action}",
                message=(
                    f"I don't know how to perform coding "
                    f"action '{action}'."
                ),
                started_at=task.created_at,
            )

        try:

            parameters = dict(
                task.parameters
            )

            parameters.pop(
                "action",
                None,
            )

            parameters.pop(
                "capability",
                None,
            )

            result = handler(
                **parameters
            )

            if asyncio.iscoroutine(
                result
            ):
                result = await result

            return result

        except Exception as exc:

            self.logger.exception(
                "Coding action failed: %s",
                action,
            )

            return self.failure_result(
                task,
                str(exc),
                message=(
                    f"Coding action '{action}' failed."
                ),
                started_at=task.created_at,
            )

    # ========================================================================
    # ACTION RESOLUTION
    # ========================================================================

    def _resolve_action(
        self,
        task: AgentTask,
    ) -> str:

        explicit_action = task.parameters.get(
            "action"
        )

        if explicit_action:

            return str(
                explicit_action
            ).strip().lower()

        instruction = (
            task.instruction
            .strip()
            .lower()
        )

        if (
            "read code" in instruction
            or "read file" in instruction
            or instruction.startswith("show code")
        ):
            return "read_code"

        if (
            "create file" in instruction
            or "write code" in instruction
            or "create code" in instruction
        ):
            return "write_code"

        if (
            "edit code" in instruction
            or "modify code" in instruction
            or "change code" in instruction
            or "update code" in instruction
        ):
            return "edit_code"

        if (
            "analyze code" in instruction
            or "analyse code" in instruction
            or "review code" in instruction
        ):
            return "analyze_code"

        if (
            "search code" in instruction
            or "search project" in instruction
            or "find in code" in instruction
        ):
            return "search_code"

        if (
            "create project" in instruction
            or "new project" in instruction
        ):
            return "create_project"

        if (
            "run tests" in instruction
            or "run test" in instruction
            or "test the project" in instruction
        ):
            return "run_tests"

        if (
            "build project" in instruction
            or "build the project" in instruction
        ):
            return "run_build"

        if (
            "debug" in instruction
            or "fix this error" in instruction
            or "fix the error" in instruction
        ):
            return "debug"

        if (
            "install dependency" in instruction
            or "install package" in instruction
            or "pip install" in instruction
            or "npm install" in instruction
        ):
            return "install_dependency"

        if (
            "git " in instruction
            or instruction.startswith("git")
        ):
            return "git"

        if (
            "documentation" in instruction
            or "generate docs" in instruction
            or "generate documentation" in instruction
        ):
            return "documentation"

        if (
            "terminal" in instruction
            or "run command" in instruction
            or instruction.startswith("execute command")
        ):
            return "terminal"

        if (
            "project info" in instruction
            or "project structure" in instruction
        ):
            return "project_info"

        return "unknown"

    # ========================================================================
    # WORKSPACE
    # ========================================================================

    def set_workspace(
        self,
        path: Union[str, Path],
    ) -> Path:

        resolved = Path(
            path
        ).expanduser().resolve()

        if not resolved.exists():

            raise FileNotFoundError(
                f"Workspace does not exist: {resolved}"
            )

        if not resolved.is_dir():

            raise NotADirectoryError(
                f"Workspace is not a directory: {resolved}"
            )

        self.workspace = resolved

        return resolved

    def _resolve_path(
        self,
        path: Union[str, Path],
        *,
        must_exist: bool = False,
    ) -> Path:

        candidate = Path(
            path
        ).expanduser()

        if not candidate.is_absolute():

            if self.workspace is not None:

                candidate = (
                    self.workspace
                    / candidate
                )

            else:

                candidate = (
                    Path.cwd()
                    / candidate
                )

        candidate = candidate.resolve()

        if must_exist and not candidate.exists():

            raise FileNotFoundError(
                f"Path does not exist: {candidate}"
            )

        return candidate

    def _is_safe_project_path(
        self,
        path: Path,
    ) -> bool:

        parts = {
            part.lower()
            for part in path.parts
        }

        blocked = {
            part.lower()
            for part in self.blocked_directories
        }

        return not bool(
            parts.intersection(
                blocked
            )
        )

    # ========================================================================
    # READ CODE
    # ========================================================================

    async def read_code(
        self,
        path: Union[str, Path],
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_chars: int = 200_000,
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._resolve_path(
            path,
            must_exist=True,
        )

        if not target.is_file():

            return {
                "success": False,
                "error": (
                    f"Not a file: {target}"
                ),
            }

        if not self._is_safe_project_path(
            target
        ):

            return {
                "success": False,
                "error": (
                    "Access to this protected directory "
                    "is not allowed."
                ),
            }

        if self.code_reader is not None:

            for method_name in (
                "read",
                "read_file",
                "read_code",
            ):

                method = getattr(
                    self.code_reader,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            str(target)
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "read_code",
                            "path": str(target),
                            "content": self._truncate(
                                result,
                                max_chars,
                            ),
                        }

                    except Exception:
                        pass

        try:

            content = target.read_text(
                encoding="utf-8",
                errors="replace",
            )

            lines = content.splitlines()

            if start_line is not None:

                start_index = max(
                    0,
                    start_line - 1,
                )

            else:

                start_index = 0

            if end_line is not None:

                end_index = min(
                    len(lines),
                    end_line,
                )

            else:

                end_index = len(lines)

            selected = "\n".join(
                lines[
                    start_index:end_index
                ]
            )

            selected = self._truncate(
                selected,
                max_chars,
            )

            return {
                "success": True,
                "action": "read_code",
                "path": str(target),
                "language": self._detect_language(
                    target
                ),
                "start_line": (
                    start_line or 1
                ),
                "end_line": (
                    end_line
                    or len(lines)
                ),
                "content": selected,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # WRITE CODE
    # ========================================================================

    async def write_code(
        self,
        path: Union[str, Path],
        content: str,
        create_parents: bool = True,
        overwrite: bool = True,
        encoding: str = "utf-8",
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._resolve_path(
            path
        )

        if not self._is_safe_project_path(
            target
        ):

            return {
                "success": False,
                "error": (
                    "Writing inside protected directories "
                    "is not allowed."
                ),
            }

        if target.exists() and not overwrite:

            return {
                "success": False,
                "error": (
                    f"File already exists: {target}"
                ),
            }

        if create_parents:

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        try:

            target.write_text(
                content,
                encoding=encoding,
            )

            return {
                "success": True,
                "action": "write_code",
                "path": str(target),
                "bytes": target.stat().st_size,
                "created": True,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # EDIT CODE
    # ========================================================================

    async def edit_code(
        self,
        path: Union[str, Path],
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        replacements: Optional[List[Dict[str, str]]] = None,
        encoding: str = "utf-8",
        replace_all: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._resolve_path(
            path,
            must_exist=True,
        )

        if not target.is_file():

            return {
                "success": False,
                "error": (
                    f"Not a file: {target}"
                ),
            }

        if not self._is_safe_project_path(
            target
        ):

            return {
                "success": False,
                "error": (
                    "Editing protected files is not allowed."
                ),
            }

        try:

            content = target.read_text(
                encoding=encoding,
                errors="replace",
            )

            original_content = content

            if replacements:

                for replacement in replacements:

                    old = replacement.get(
                        "old",
                        "",
                    )

                    new = replacement.get(
                        "new",
                        "",
                    )

                    if not old:
                        continue

                    if replace_all:

                        content = content.replace(
                            old,
                            new,
                        )

                    else:

                        content = content.replace(
                            old,
                            new,
                            1,
                        )

            elif old_text is not None:

                if old_text not in content:

                    return {
                        "success": False,
                        "error": (
                            "The requested old_text "
                            "was not found."
                        ),
                    }

                if replace_all:

                    content = content.replace(
                        old_text,
                        new_text or "",
                    )

                else:

                    content = content.replace(
                        old_text,
                        new_text or "",
                        1,
                    )

            else:

                return {
                    "success": False,
                    "error": (
                        "Provide old_text/new_text or replacements."
                    ),
                }

            if content == original_content:

                return {
                    "success": False,
                    "error": (
                        "No changes were made."
                    ),
                }

            target.write_text(
                content,
                encoding=encoding,
            )

            return {
                "success": True,
                "action": "edit_code",
                "path": str(target),
                "changed": True,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # SEARCH CODE
    # ========================================================================

    async def search_code(
        self,
        query: str,
        path: Union[str, Path] = ".",
        extensions: Optional[Sequence[str]] = None,
        case_sensitive: bool = False,
        max_results: int = 200,
        **_: Any,
    ) -> Dict[str, Any]:

        if not query:

            return {
                "success": False,
                "error": (
                    "Search query cannot be empty."
                ),
            }

        root = self._resolve_path(
            path,
            must_exist=True,
        )

        if root.is_file():

            files = [root]

        else:

            files = []

            for current_root, directories, filenames in os.walk(
                root
            ):

                directories[:] = [
                    directory
                    for directory in directories
                    if directory not in self.blocked_directories
                ]

                for filename in filenames:

                    file_path = (
                        Path(current_root)
                        / filename
                    )

                    if not self._is_safe_project_path(
                        file_path
                    ):
                        continue

                    if extensions:

                        normalized_extensions = {
                            self._normalize_extension(
                                extension
                            )
                            for extension in extensions
                        }

                        if (
                            file_path.suffix.lower()
                            not in normalized_extensions
                        ):
                            continue

                    elif (
                        file_path.suffix.lower()
                        not in self.allowed_extensions
                    ):

                        continue

                    files.append(
                        file_path
                    )

        flags = 0 if case_sensitive else re.IGNORECASE

        pattern = re.compile(
            re.escape(query),
            flags,
        )

        results = []

        for file_path in files:

            if len(results) >= max_results:
                break

            try:

                content = file_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                for line_number, line in enumerate(
                    content.splitlines(),
                    start=1,
                ):

                    if pattern.search(
                        line
                    ):

                        results.append(
                            {
                                "path": str(file_path),
                                "line": line_number,
                                "text": line.strip(),
                            }
                        )

                        if len(results) >= max_results:
                            break

            except (
                OSError,
                UnicodeError,
            ):
                continue

        return {
            "success": True,
            "action": "search_code",
            "query": query,
            "results": results,
            "count": len(results),
        }

    # ========================================================================
    # ANALYZE CODE
    # ========================================================================

    async def analyze_code(
        self,
        path: Union[str, Path],
        language: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._resolve_path(
            path,
            must_exist=True,
        )

        if self.code_analyzer is not None:

            for method_name in (
                "analyze",
                "analyze_code",
                "review",
            ):

                method = getattr(
                    self.code_analyzer,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            str(target)
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "analyze_code",
                            "path": str(target),
                            "analysis": result,
                        }

                    except Exception:
                        pass

        read_result = await self.read_code(
            target
        )

        if not read_result.get(
            "success"
        ):

            return read_result

        content = read_result.get(
            "content",
            "",
        )

        detected_language = (
            language
            or self._detect_language(
                target
            )
        )

        issues = []

        if (
            detected_language == "python"
        ):

            issues.extend(
                self._basic_python_analysis(
                    content
                )
            )

        elif detected_language in {
            "javascript",
            "typescript",
        }:

            issues.extend(
                self._basic_javascript_analysis(
                    content
                )
            )

        line_count = len(
            content.splitlines()
        )

        return {
            "success": True,
            "action": "analyze_code",
            "path": str(target),
            "language": detected_language,
            "line_count": line_count,
            "issues": issues,
            "issue_count": len(issues),
            "summary": (
                "Basic static analysis completed."
            ),
        }

    def _basic_python_analysis(
        self,
        content: str,
    ) -> List[Dict[str, Any]]:

        issues = []

        lines = content.splitlines()

        for index, line in enumerate(
            lines,
            start=1,
        ):

            if len(line) > 120:

                issues.append(
                    {
                        "line": index,
                        "type": "style",
                        "message": (
                            "Line exceeds 120 characters."
                        ),
                    }
                )

            if (
                "import *"
                in line
            ):

                issues.append(
                    {
                        "line": index,
                        "type": "style",
                        "message": (
                            "Wildcard import detected."
                        ),
                    }
                )

            if (
                "except:"
                in line
            ):

                issues.append(
                    {
                        "line": index,
                        "type": "warning",
                        "message": (
                            "Bare except detected."
                        ),
                    }
                )

        return issues

    def _basic_javascript_analysis(
        self,
        content: str,
    ) -> List[Dict[str, Any]]:

        issues = []

        for index, line in enumerate(
            content.splitlines(),
            start=1,
        ):

            if len(line) > 120:

                issues.append(
                    {
                        "line": index,
                        "type": "style",
                        "message": (
                            "Line exceeds 120 characters."
                        ),
                    }
                )

            if (
                "console.log("
                in line
            ):

                issues.append(
                    {
                        "line": index,
                        "type": "debug",
                        "message": (
                            "console.log detected."
                        ),
                    }
                )

        return issues

    # ========================================================================
    # PROJECT CREATION
    # ========================================================================

    async def create_project(
        self,
        path: Union[str, Path],
        project_type: str = "python",
        name: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._resolve_path(
            path
        )

        if target.exists():

            return {
                "success": False,
                "error": (
                    f"Project path already exists: {target}"
                ),
            }

        if self.project_manager is not None:

            for method_name in (
                "create_project",
                "create",
                "initialize",
            ):

                method = getattr(
                    self.project_manager,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            path=str(target),
                            project_type=project_type,
                            name=name,
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "create_project",
                            "path": str(target),
                            "project_type": project_type,
                            "result": result,
                        }

                    except TypeError:

                        try:

                            result = method(
                                str(target),
                                project_type,
                            )

                            if asyncio.iscoroutine(
                                result
                            ):
                                result = await result

                            return {
                                "success": True,
                                "action": "create_project",
                                "path": str(target),
                                "project_type": project_type,
                                "result": result,
                            }

                        except Exception:
                            pass

                    except Exception:
                        pass

        target.mkdir(
            parents=True,
            exist_ok=True,
        )

        created = []

        if project_type.lower() == "python":

            files = {
                "main.py": (
                    'def main():\n'
                    '    print("Hello from RENIX project.")\n'
                    '\n'
                    '\n'
                    'if __name__ == "__main__":\n'
                    '    main()\n'
                ),
                "requirements.txt": "",
                "README.md": (
                    f"# {name or target.name}\n\n"
                    "Python project created by RENIX.\n"
                ),
                ".gitignore": (
                    "__pycache__/\n"
                    "*.pyc\n"
                    ".venv/\n"
                    "venv/\n"
                    ".env\n"
                ),
            }

        elif project_type.lower() in {
            "javascript",
            "node",
            "nodejs",
        }:

            files = {
                "index.js": (
                    'console.log("Hello from RENIX project.");\n'
                ),
                "package.json": json.dumps(
                    {
                        "name": name or target.name,
                        "version": "1.0.0",
                        "private": True,
                        "main": "index.js",
                    },
                    indent=2,
                ),
                "README.md": (
                    f"# {name or target.name}\n\n"
                    "JavaScript project created by RENIX.\n"
                ),
            }

        elif project_type.lower() == "web":

            files = {
                "index.html": (
                    "<!DOCTYPE html>\n"
                    "<html lang=\"en\">\n"
                    "<head>\n"
                    "    <meta charset=\"UTF-8\">\n"
                    "    <meta name=\"viewport\" "
                    "content=\"width=device-width, initial-scale=1.0\">\n"
                    "    <title>RENIX Project</title>\n"
                    "    <link rel=\"stylesheet\" href=\"style.css\">\n"
                    "</head>\n"
                    "<body>\n"
                    "    <h1>Hello from RENIX</h1>\n"
                    "    <script src=\"script.js\"></script>\n"
                    "</body>\n"
                    "</html>\n"
                ),
                "style.css": (
                    "body {\n"
                    "    font-family: Arial, sans-serif;\n"
                    "}\n"
                ),
                "script.js": (
                    'console.log("RENIX project loaded.");\n'
                ),
                "README.md": (
                    f"# {name or target.name}\n"
                ),
            }

        else:

            return {
                "success": False,
                "error": (
                    f"Unsupported project type: {project_type}"
                ),
            }

        for relative_path, content in files.items():

            file_path = (
                target
                / relative_path
            )

            file_path.write_text(
                content,
                encoding="utf-8",
            )

            created.append(
                str(file_path)
            )

        return {
            "success": True,
            "action": "create_project",
            "path": str(target),
            "project_type": project_type,
            "files": created,
        }

    # ========================================================================
    # TEST RUNNER
    # ========================================================================

    async def run_tests(
        self,
        path: Union[str, Path] = ".",
        command: Optional[Union[str, List[str]]] = None,
        timeout: int = 300,
        **_: Any,
    ) -> Dict[str, Any]:

        root = self._resolve_path(
            path,
            must_exist=True,
        )

        if self.test_runner is not None:

            for method_name in (
                "run",
                "run_tests",
                "test",
            ):

                method = getattr(
                    self.test_runner,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            str(root)
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "run_tests",
                            "result": result,
                        }

                    except Exception:
                        pass

        if command is None:

            if (root / "pytest.ini").exists() or (
                root / "pyproject.toml"
            ).exists() or (
                root / "tests"
            ).exists():

                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                ]

            elif (root / "package.json").exists():

                command = [
                    "npm",
                    "test",
                ]

            else:

                return {
                    "success": False,
                    "error": (
                        "Could not determine the test command."
                    ),
                }

        return await self._run_subprocess(
            command,
            cwd=root,
            timeout=timeout,
            action="run_tests",
        )

    # ========================================================================
    # BUILD
    # ========================================================================

    async def run_build(
        self,
        path: Union[str, Path] = ".",
        command: Optional[Union[str, List[str]]] = None,
        timeout: int = 600,
        **_: Any,
    ) -> Dict[str, Any]:

        root = self._resolve_path(
            path,
            must_exist=True,
        )

        if self.build_manager is not None:

            for method_name in (
                "build",
                "run_build",
                "run",
            ):

                method = getattr(
                    self.build_manager,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            str(root)
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "run_build",
                            "result": result,
                        }

                    except Exception:
                        pass

        if command is None:

            if (root / "package.json").exists():

                command = [
                    "npm",
                    "run",
                    "build",
                ]

            elif (root / "pyproject.toml").exists():

                command = [
                    sys.executable,
                    "-m",
                    "build",
                ]

            else:

                return {
                    "success": False,
                    "error": (
                        "Could not determine the build command."
                    ),
                }

        return await self._run_subprocess(
            command,
            cwd=root,
            timeout=timeout,
            action="run_build",
        )

    # ========================================================================
    # DEBUG
    # ========================================================================

    async def debug(
        self,
        error: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        if self.debugger is not None:

            for method_name in (
                "debug",
                "analyze",
                "diagnose",
            ):

                method = getattr(
                    self.debugger,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            error=error,
                            path=str(path)
                            if path
                            else None,
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "debug",
                            "result": result,
                        }

                    except Exception:
                        pass

        analysis = {}

        if self.error_analyzer is not None:

            for method_name in (
                "analyze",
                "analyze_error",
                "diagnose",
            ):

                method = getattr(
                    self.error_analyzer,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            error
                            or ""
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        analysis = result
                        break

                    except Exception:
                        pass

        if not analysis:

            analysis = self._basic_error_analysis(
                error or ""
            )

        return {
            "success": True,
            "action": "debug",
            "error": error,
            "analysis": analysis,
            "path": str(path)
            if path
            else None,
        }

    def _basic_error_analysis(
        self,
        error: str,
    ) -> Dict[str, Any]:

        lower = error.lower()

        suggestions = []

        if "modulenotfounderror" in lower:

            suggestions.append(
                "Check whether the required package is installed."
            )

        if "filenotfounderror" in lower:

            suggestions.append(
                "Check the file path and working directory."
            )

        if "permissionerror" in lower:

            suggestions.append(
                "Check file permissions and whether another "
                "application is locking the file."
            )

        if "syntaxerror" in lower:

            suggestions.append(
                "Inspect the indicated line for invalid syntax."
            )

        if "indentationerror" in lower:

            suggestions.append(
                "Check indentation and mixed tabs/spaces."
            )

        if "nameerror" in lower:

            suggestions.append(
                "Check whether the variable or function name "
                "was defined before use."
            )

        if "typeerror" in lower:

            suggestions.append(
                "Check argument types and function signatures."
            )

        if not suggestions:

            suggestions.append(
                "Inspect the traceback and identify the first "
                "application-level error."
            )

        return {
            "suggestions": suggestions,
        }

    # ========================================================================
    # DEPENDENCIES
    # ========================================================================

    async def install_dependency(
        self,
        package: str,
        manager: str = "pip",
        path: Union[str, Path] = ".",
        timeout: int = 300,
        **_: Any,
    ) -> Dict[str, Any]:

        if not package:

            return {
                "success": False,
                "error": (
                    "Package name cannot be empty."
                ),
            }

        root = self._resolve_path(
            path,
            must_exist=True,
        )

        if self.dependency_manager is not None:

            for method_name in (
                "install",
                "install_dependency",
                "add",
            ):

                method = getattr(
                    self.dependency_manager,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            package=package,
                            manager=manager,
                            path=str(root),
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "install_dependency",
                            "package": package,
                            "result": result,
                        }

                    except Exception:
                        pass

        manager_lower = manager.lower()

        if manager_lower == "pip":

            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                package,
            ]

        elif manager_lower == "pip3":

            command = [
                "pip3",
                "install",
                package,
            ]

        elif manager_lower in {
            "npm",
            "node",
        }:

            command = [
                "npm",
                "install",
                package,
            ]

        elif manager_lower == "yarn":

            command = [
                "yarn",
                "add",
                package,
            ]

        elif manager_lower == "pnpm":

            command = [
                "pnpm",
                "add",
                package,
            ]

        elif manager_lower == "cargo":

            command = [
                "cargo",
                "add",
                package,
            ]

        else:

            return {
                "success": False,
                "error": (
                    f"Unsupported package manager: {manager}"
                ),
            }

        return await self._run_subprocess(
            command,
            cwd=root,
            timeout=timeout,
            action="install_dependency",
        )

    # ========================================================================
    # TERMINAL
    # ========================================================================

    async def run_terminal(
        self,
        command: Union[str, List[str]],
        path: Union[str, Path] = ".",
        timeout: int = 300,
        shell: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:

        root = self._resolve_path(
            path,
            must_exist=True,
        )

        if self.terminal_manager is not None:

            for method_name in (
                "run",
                "execute",
                "run_command",
            ):

                method = getattr(
                    self.terminal_manager,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            command=command,
                            cwd=str(root),
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "terminal",
                            "result": result,
                        }

                    except Exception:
                        pass

        return await self._run_subprocess(
            command,
            cwd=root,
            timeout=timeout,
            action="terminal",
            shell=shell,
        )

    # ========================================================================
    # GIT
    # ========================================================================

    async def git(
        self,
        operation: str,
        path: Union[str, Path] = ".",
        args: Optional[List[str]] = None,
        timeout: int = 120,
        **_: Any,
    ) -> Dict[str, Any]:

        root = self._resolve_path(
            path,
            must_exist=True,
        )

        if self.git_manager is not None:

            for method_name in (
                "execute",
                "run",
                "git",
            ):

                method = getattr(
                    self.git_manager,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            operation=operation,
                            path=str(root),
                            args=args or [],
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "git",
                            "result": result,
                        }

                    except Exception:
                        pass

        safe_operations = {
            "status",
            "log",
            "branch",
            "diff",
            "remote",
            "show",
        }

        operation_clean = (
            operation.strip()
            .lower()
        )

        if operation_clean not in safe_operations:

            return {
                "success": False,
                "error": (
                    f"Git operation '{operation}' requires "
                    "the dedicated Git manager or explicit "
                    "higher-level confirmation."
                ),
            }

        command = [
            "git",
            operation_clean,
        ]

        if args:
            command.extend(
                args
            )

        return await self._run_subprocess(
            command,
            cwd=root,
            timeout=timeout,
            action="git",
        )

    # ========================================================================
    # DOCUMENTATION
    # ========================================================================

    async def generate_documentation(
        self,
        path: Union[str, Path] = ".",
        output: Optional[Union[str, Path]] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        root = self._resolve_path(
            path,
            must_exist=True,
        )

        if self.documentation is not None:

            for method_name in (
                "generate",
                "generate_documentation",
                "build",
            ):

                method = getattr(
                    self.documentation,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            str(root)
                        )

                        if asyncio.iscoroutine(
                            result
                        ):
                            result = await result

                        return {
                            "success": True,
                            "action": "documentation",
                            "result": result,
                        }

                    except Exception:
                        pass

        project_name = root.name

        files = []

        if root.is_file():

            files.append(
                root
            )

        else:

            for current_root, directories, filenames in os.walk(
                root
            ):

                directories[:] = [
                    directory
                    for directory in directories
                    if directory not in self.blocked_directories
                ]

                for filename in filenames:

                    file_path = (
                        Path(current_root)
                        / filename
                    )

                    if (
                        file_path.suffix.lower()
                        in self.allowed_extensions
                    ):

                        files.append(
                            file_path
                        )

        lines = [
            f"# {project_name}",
            "",
            "## Project Documentation",
            "",
            "Generated by RENIX Coding Agent.",
            "",
            "## Files",
            "",
        ]

        for file_path in sorted(
            files
        ):

            try:

                relative = file_path.relative_to(
                    root
                )

            except ValueError:

                relative = file_path.name

            lines.append(
                f"- `{relative}`"
            )

        documentation = "\n".join(
            lines
        ) + "\n"

        if output is not None:

            output_path = self._resolve_path(
                output
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_text(
                documentation,
                encoding="utf-8",
            )

            return {
                "success": True,
                "action": "documentation",
                "output": str(output_path),
                "content": documentation,
            }

        return {
            "success": True,
            "action": "documentation",
            "content": documentation,
            "file_count": len(files),
        }

    # ========================================================================
    # PROJECT INFORMATION
    # ========================================================================

    async def project_info(
        self,
        path: Union[str, Path] = ".",
        max_files: int = 500,
        **_: Any,
    ) -> Dict[str, Any]:

        root = self._resolve_path(
            path,
            must_exist=True,
        )

        if root.is_file():

            return {
                "success": True,
                "action": "project_info",
                "path": str(root),
                "type": "file",
                "extension": root.suffix,
                "size": root.stat().st_size,
            }

        files = []
        directories = []

        for current_root, directory_names, filenames in os.walk(
            root
        ):

            directory_names[:] = [
                directory
                for directory in directory_names
                if directory not in self.blocked_directories
            ]

            current_path = Path(
                current_root
            )

            for directory in directory_names:

                directories.append(
                    str(
                        current_path
                        / directory
                    )
                )

            for filename in filenames:

                files.append(
                    str(
                        current_path
                        / filename
                    )
                )

                if len(files) >= max_files:
                    break

            if len(files) >= max_files:
                break

        return {
            "success": True,
            "action": "project_info",
            "path": str(root),
            "directory_count": len(
                directories
            ),
            "file_count": len(
                files
            ),
            "directories": directories,
            "files": files,
            "truncated": len(files) >= max_files,
        }

    # ========================================================================
    # SUBPROCESS
    # ========================================================================

    async def _run_subprocess(
        self,
        command: Union[str, List[str]],
        cwd: Path,
        timeout: int,
        action: str,
        shell: bool = False,
    ) -> Dict[str, Any]:

        if isinstance(
            command,
            str,
        ):

            if not shell:

                command = self._split_command(
                    command
                )

        if isinstance(
            command,
            list,
        ):

            executable = command[0]

            if not shell and shutil.which(
                executable
            ) is None:

                return {
                    "success": False,
                    "action": action,
                    "error": (
                        f"Command not found: {executable}"
                    ),
                    "command": command,
                }

        try:

            process = await asyncio.create_subprocess_exec(
                *command
                if isinstance(command, list)
                else command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ) if not shell else await asyncio.create_subprocess_shell(
                command
                if isinstance(command, str)
                else " ".join(
                    self._quote_argument(
                        part
                    )
                    for part in command
                ),
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            stdout_text = stdout.decode(
                "utf-8",
                errors="replace",
            )

            stderr_text = stderr.decode(
                "utf-8",
                errors="replace",
            )

            return {
                "success": process.returncode == 0,
                "action": action,
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
            }

        except asyncio.TimeoutError:

            try:
                process.kill()
            except Exception:
                pass

            return {
                "success": False,
                "action": action,
                "error": (
                    f"Command timed out after {timeout} seconds."
                ),
                "command": command,
            }

        except Exception as exc:

            return {
                "success": False,
                "action": action,
                "error": str(exc),
                "command": command,
            }

    # ========================================================================
    # LANGUAGE DETECTION
    # ========================================================================

    @staticmethod
    def _detect_language(
        path: Path,
    ) -> str:

        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
            ".kts": "kotlin",
            ".dart": "dart",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".vue": "vue",
            ".svelte": "svelte",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".sql": "sql",
            ".sh": "shell",
            ".bat": "batch",
            ".ps1": "powershell",
            ".md": "markdown",
            ".txt": "text",
        }

        return mapping.get(
            path.suffix.lower(),
            "unknown",
        )

    @staticmethod
    def _normalize_extension(
        extension: str,
    ) -> str:

        extension = str(
            extension
        ).strip().lower()

        if not extension:
            return extension

        if not extension.startswith(
            "."
        ):

            extension = (
                "."
                + extension
            )

        return extension

    @staticmethod
    def _truncate(
        value: Any,
        max_chars: int,
    ) -> Any:

        if not isinstance(
            value,
            str,
        ):
            return value

        if len(value) <= max_chars:
            return value

        return (
            value[:max_chars]
            + "\n\n[Output truncated by RENIX.]"
        )

    @staticmethod
    def _split_command(
        command: str,
    ) -> List[str]:

        import shlex

        if os.name == "nt":

            try:
                return shlex.split(
                    command,
                    posix=False,
                )
            except Exception:
                return command.split()

        return shlex.split(
            command
        )

    @staticmethod
    def _quote_argument(
        argument: str,
    ) -> str:

        import shlex

        return shlex.quote(
            str(argument)
        )


__all__ = [
    "CodingAgent",
]


