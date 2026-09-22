"""
RENIX Coding System
===================

The coding subsystem provides RENIX with the ability to:

- Understand programming projects
- Read and write source code
- Analyze code
- Edit files safely
- Debug errors
- Run tests
- Build projects
- Manage dependencies
- Work with terminals
- Manage Git repositories
- Generate documentation

The individual modules implement the actual functionality.
This package exposes the main public interfaces.
"""

from .coding_agent import CodingAgent
from .project_manager import ProjectManager
from .code_reader import CodeReader
from .code_writer import CodeWriter
from .code_editor import CodeEditor
from .code_analyzer import CodeAnalyzer
from .debugger import Debugger
from .error_analyzer import ErrorAnalyzer
from .test_runner import TestRunner
from .build_manager import BuildManager
from .dependency_manager import DependencyManager
from .terminal_manager import TerminalManager
from .git_manager import GitManager
from .documentation import DocumentationManager


__all__ = [
    "CodingAgent",
    "ProjectManager",
    "CodeReader",
    "CodeWriter",
    "CodeEditor",
    "CodeAnalyzer",
    "Debugger",
    "ErrorAnalyzer",
    "TestRunner",
    "BuildManager",
    "DependencyManager",
    "TerminalManager",
    "GitManager",
    "DocumentationManager",
]


__version__ = "1.0.0"
__author__ = "RENIX"


