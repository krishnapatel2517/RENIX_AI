"""
RENIX File Management Package
==============================

This package provides RENIX with a unified file-management layer.

Modules include:

- file_manager
- file_search
- semantic_search
- file_classifier
- file_preview
- file_metadata
- duplicate_detector
- file_operations
- folder_manager
- copy_manager
- move_manager
- rename_manager
- delete_manager
- archive_manager
- document_reader
- pdf_reader
- image_reader
- file_permissions
"""

from __future__ import annotations

from typing import Any


__version__ = "1.0.0"
__author__ = "RENIX"
__package_name__ = "RENIX File Management"


# ==============================================================
# SAFE OPTIONAL IMPORTS
# ==============================================================

# The package should remain importable even while individual
# modules are still being developed.

try:
    from .file_manager import FileManager
except ImportError:
    FileManager = None  # type: ignore


try:
    from .file_search import FileSearch
except ImportError:
    FileSearch = None  # type: ignore


try:
    from .semantic_search import SemanticFileSearch
except ImportError:
    SemanticFileSearch = None  # type: ignore


try:
    from .file_classifier import FileClassifier
except ImportError:
    FileClassifier = None  # type: ignore


try:
    from .file_preview import FilePreview
except ImportError:
    FilePreview = None  # type: ignore


try:
    from .file_metadata import FileMetadata
except ImportError:
    FileMetadata = None  # type: ignore


try:
    from .duplicate_detector import DuplicateDetector
except ImportError:
    DuplicateDetector = None  # type: ignore


try:
    from .file_operations import FileOperations
except ImportError:
    FileOperations = None  # type: ignore


try:
    from .folder_manager import FolderManager
except ImportError:
    FolderManager = None  # type: ignore


try:
    from .copy_manager import CopyManager
except ImportError:
    CopyManager = None  # type: ignore


try:
    from .move_manager import MoveManager
except ImportError:
    MoveManager = None  # type: ignore


try:
    from .rename_manager import RenameManager
except ImportError:
    RenameManager = None  # type: ignore


try:
    from .delete_manager import DeleteManager
except ImportError:
    DeleteManager = None  # type: ignore


try:
    from .archive_manager import ArchiveManager
except ImportError:
    ArchiveManager = None  # type: ignore


try:
    from .document_reader import DocumentReader
except ImportError:
    DocumentReader = None  # type: ignore


try:
    from .pdf_reader import PDFReader
except ImportError:
    PDFReader = None  # type: ignore


try:
    from .image_reader import ImageReader
except ImportError:
    ImageReader = None  # type: ignore


try:
    from .file_permissions import FilePermissions
except ImportError:
    FilePermissions = None  # type: ignore


# ==============================================================
# PACKAGE INFORMATION
# ==============================================================


def get_version() -> str:
    """Return the RENIX files package version."""

    return __version__


def get_package_name() -> str:
    """Return the package name."""

    return __package_name__


# ==============================================================
# AVAILABLE COMPONENTS
# ==============================================================


def get_available_components() -> dict[str, bool]:
    """
    Return which file-management components are currently
    available.

    This is useful while RENIX is being built incrementally.
    """

    components: dict[str, Any] = {
        "FileManager": FileManager,
        "FileSearch": FileSearch,
        "SemanticFileSearch": SemanticFileSearch,
        "FileClassifier": FileClassifier,
        "FilePreview": FilePreview,
        "FileMetadata": FileMetadata,
        "DuplicateDetector": DuplicateDetector,
        "FileOperations": FileOperations,
        "FolderManager": FolderManager,
        "CopyManager": CopyManager,
        "MoveManager": MoveManager,
        "RenameManager": RenameManager,
        "DeleteManager": DeleteManager,
        "ArchiveManager": ArchiveManager,
        "DocumentReader": DocumentReader,
        "PDFReader": PDFReader,
        "ImageReader": ImageReader,
        "FilePermissions": FilePermissions,
    }

    return {
        name: component is not None
        for name, component in components.items()
    }


# ==============================================================
# PACKAGE STATUS
# ==============================================================


def get_status() -> dict[str, Any]:
    """
    Return the current status of the RENIX file subsystem.
    """

    available = get_available_components()

    return {
        "package": __package_name__,
        "version": __version__,
        "components": available,
        "available_count": sum(
            available.values()
        ),
        "total_components": len(
            available
        ),
        "ready": all(
            available.values()
        ),
    }


# ==============================================================
# PUBLIC API
# ==============================================================


__all__ = [
    "FileManager",
    "FileSearch",
    "SemanticFileSearch",
    "FileClassifier",
    "FilePreview",
    "FileMetadata",
    "DuplicateDetector",
    "FileOperations",
    "FolderManager",
    "CopyManager",
    "MoveManager",
    "RenameManager",
    "DeleteManager",
    "ArchiveManager",
    "DocumentReader",
    "PDFReader",
    "ImageReader",
    "FilePermissions",
    "get_version",
    "get_package_name",
    "get_available_components",
    "get_status",
]


