"""
RENIX AI - File Agent

Handles file and folder operations through the RENIX files subsystem.

Responsibilities:
- Search files
- Read file metadata
- Create files and folders
- Copy files
- Move files
- Rename files
- Delete files
- Open files
- Create archives
- Detect duplicates
- Preview supported files
- Manage folders

This module is intentionally self-contained and works with the existing
RENIX project structure.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base_agent import (
    AgentCapability,
    AgentPriority,
    AgentResult,
    AgentTask,
    BaseAgent,
)


class FileAgent(BaseAgent):
    """
    RENIX file-management agent.

    Provides a unified interface between the AI layer and the local
    filesystem.
    """

    agent_name = "file_agent"

    agent_description = (
        "Searches, reads, creates, copies, moves, renames, deletes and "
        "organizes files and folders."
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

        self.files_root = Path.cwd() / "files"
        self.data_root = Path.cwd() / "data"

        self.file_manager = None
        self.file_search = None
        self.semantic_search = None
        self.file_classifier = None
        self.file_preview = None
        self.file_metadata = None
        self.duplicate_detector = None
        self.file_operations = None
        self.folder_manager = None
        self.copy_manager = None
        self.move_manager = None
        self.rename_manager = None
        self.delete_manager = None
        self.archive_manager = None
        self.document_reader = None
        self.pdf_reader = None
        self.image_reader = None
        self.file_permissions = None

        self._modules_loaded = False

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def _register_default_capabilities(self) -> None:
        """Register all file-agent capabilities."""

        super()._register_default_capabilities()

        capabilities = [
            AgentCapability(
                name="file_search",
                description="Search for files and folders.",
            ),
            AgentCapability(
                name="file_read",
                description="Read supported text and document files.",
            ),
            AgentCapability(
                name="file_metadata",
                description="Inspect file metadata.",
            ),
            AgentCapability(
                name="file_create",
                description="Create files.",
            ),
            AgentCapability(
                name="folder_create",
                description="Create folders.",
            ),
            AgentCapability(
                name="file_copy",
                description="Copy files and folders.",
            ),
            AgentCapability(
                name="file_move",
                description="Move files and folders.",
            ),
            AgentCapability(
                name="file_rename",
                description="Rename files and folders.",
            ),
            AgentCapability(
                name="file_delete",
                description="Delete files and folders.",
                requires_confirmation=True,
            ),
            AgentCapability(
                name="file_open",
                description="Open files using the operating system.",
            ),
            AgentCapability(
                name="archive",
                description="Create ZIP archives.",
            ),
            AgentCapability(
                name="duplicate_detection",
                description="Find duplicate files.",
            ),
            AgentCapability(
                name="folder_list",
                description="List directory contents.",
            ),
        ]

        for capability in capabilities:
            self.register_capability(capability)

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def on_initialize(self) -> bool:
        """
        Load the existing RENIX file subsystem.

        Missing optional modules do not prevent startup.
        """

        self._load_file_modules()

        return True

    def _load_file_modules(self) -> None:
        """Load available modules from the existing files package."""

        if self._modules_loaded:
            return

        self._modules_loaded = True

        module_map = {
            "file_manager": (
                "files.file_manager",
                "FileManager",
            ),
            "file_search": (
                "files.file_search",
                "FileSearch",
            ),
            "semantic_search": (
                "files.semantic_search",
                "SemanticSearch",
            ),
            "file_classifier": (
                "files.file_classifier",
                "FileClassifier",
            ),
            "file_preview": (
                "files.file_preview",
                "FilePreview",
            ),
            "file_metadata": (
                "files.file_metadata",
                "FileMetadata",
            ),
            "duplicate_detector": (
                "files.duplicate_detector",
                "DuplicateDetector",
            ),
            "file_operations": (
                "files.file_operations",
                "FileOperations",
            ),
            "folder_manager": (
                "files.folder_manager",
                "FolderManager",
            ),
            "copy_manager": (
                "files.copy_manager",
                "CopyManager",
            ),
            "move_manager": (
                "files.move_manager",
                "MoveManager",
            ),
            "rename_manager": (
                "files.rename_manager",
                "RenameManager",
            ),
            "delete_manager": (
                "files.delete_manager",
                "DeleteManager",
            ),
            "archive_manager": (
                "files.archive_manager",
                "ArchiveManager",
            ),
            "document_reader": (
                "files.document_reader",
                "DocumentReader",
            ),
            "pdf_reader": (
                "files.pdf_reader",
                "PDFReader",
            ),
            "image_reader": (
                "files.image_reader",
                "ImageReader",
            ),
            "file_permissions": (
                "files.file_permissions",
                "FilePermissions",
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

                setattr(
                    self,
                    attribute,
                    cls(),
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
        Execute a file-management task.
        """

        action = self._resolve_action(task)

        handlers = {
            "search": self.search_files,
            "read": self.read_file,
            "metadata": self.get_metadata,
            "create_file": self.create_file,
            "create_folder": self.create_folder,
            "list_folder": self.list_folder,
            "copy": self.copy,
            "move": self.move,
            "rename": self.rename,
            "delete": self.delete,
            "open": self.open_file,
            "archive": self.create_archive,
            "duplicates": self.find_duplicates,
            "exists": self.exists,
            "file_size": self.get_file_size,
            "hash": self.get_file_hash,
        }

        handler = handlers.get(action)

        if handler is None:

            return self.failure_result(
                task,
                f"Unknown file action: {action}",
                message=f"I don't know how to perform file action '{action}'.",
                started_at=task.created_at,
            )

        try:

            parameters = dict(task.parameters)

            parameters.pop("action", None)
            parameters.pop("capability", None)

            result = handler(**parameters)

            if asyncio.iscoroutine(result):
                result = await result

            return result

        except Exception as exc:

            self.logger.exception(
                "File action failed: %s",
                action,
            )

            return self.failure_result(
                task,
                str(exc),
                message=f"File action '{action}' failed.",
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
            instruction.startswith("search")
            or "find file" in instruction
            or "find files" in instruction
        ):
            return "search"

        if (
            instruction.startswith("read file")
            or instruction.startswith("read ")
        ):
            return "read"

        if "metadata" in instruction:
            return "metadata"

        if (
            "create file" in instruction
            or "make file" in instruction
        ):
            return "create_file"

        if (
            "create folder" in instruction
            or "make folder" in instruction
            or "create directory" in instruction
        ):
            return "create_folder"

        if (
            instruction.startswith("list folder")
            or "list directory" in instruction
        ):
            return "list_folder"

        if (
            instruction.startswith("copy")
            or "copy file" in instruction
        ):
            return "copy"

        if (
            instruction.startswith("move")
            or "move file" in instruction
        ):
            return "move"

        if (
            instruction.startswith("rename")
            or "rename file" in instruction
        ):
            return "rename"

        if (
            instruction.startswith("delete")
            or "delete file" in instruction
        ):
            return "delete"

        if (
            instruction.startswith("open file")
            or "open document" in instruction
        ):
            return "open"

        if (
            "archive" in instruction
            or "zip" in instruction
        ):
            return "archive"

        if (
            "duplicate" in instruction
            or "duplicates" in instruction
        ):
            return "duplicates"

        if "exist" in instruction:
            return "exists"

        if (
            "file size" in instruction
            or "size of file" in instruction
        ):
            return "file_size"

        if (
            "hash" in instruction
            or "checksum" in instruction
        ):
            return "hash"

        return "unknown"

    # ========================================================================
    # PATH HELPERS
    # ========================================================================

    @staticmethod
    def _path(
        value: str | Path,
    ) -> Path:

        return Path(
            value
        ).expanduser().resolve()

    @staticmethod
    def _ensure_parent(
        path: Path,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================================
    # SEARCH
    # ========================================================================

    def search_files(
        self,
        query: str,
        directory: Optional[str] = None,
        recursive: bool = True,
        file_type: Optional[str] = None,
        limit: int = 100,
        **_: Any,
    ) -> Dict[str, Any]:

        if not query:
            return {
                "success": False,
                "error": "Search query cannot be empty.",
            }

        root = self._path(
            directory
            if directory
            else Path.home()
        )

        if not root.exists():

            return {
                "success": False,
                "error": f"Directory does not exist: {root}",
            }

        # Try RENIX search subsystem first.

        if self.file_search is not None:

            for method_name in (
                "search",
                "find",
                "search_files",
            ):

                method = getattr(
                    self.file_search,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            query=query,
                            directory=str(root),
                            recursive=recursive,
                            limit=limit,
                        )

                        return {
                            "success": True,
                            "action": "search",
                            "query": query,
                            "results": result,
                        }

                    except TypeError:

                        try:

                            result = method(
                                query
                            )

                            return {
                                "success": True,
                                "action": "search",
                                "query": query,
                                "results": result,
                            }

                        except Exception:
                            pass

                    except Exception:
                        pass

        # Fallback filesystem search.

        results: List[Dict[str, Any]] = []

        iterator: Iterable[Path]

        if recursive:
            iterator = root.rglob("*")
        else:
            iterator = root.glob("*")

        query_lower = query.lower()

        for path in iterator:

            if len(results) >= limit:
                break

            try:

                if query_lower not in path.name.lower():
                    continue

                if file_type:

                    normalized_type = (
                        file_type.lower()
                        .lstrip(".")
                    )

                    if (
                        path.is_file()
                        and path.suffix.lower().lstrip(".")
                        != normalized_type
                    ):
                        continue

                results.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "type": (
                            "directory"
                            if path.is_dir()
                            else "file"
                        ),
                        "size": (
                            path.stat().st_size
                            if path.is_file()
                            else None
                        ),
                    }
                )

            except (
                PermissionError,
                OSError,
            ):
                continue

        return {
            "success": True,
            "action": "search",
            "query": query,
            "directory": str(root),
            "results": results,
            "count": len(results),
        }

    # ========================================================================
    # READ
    # ========================================================================

    def read_file(
        self,
        path: str,
        encoding: str = "utf-8",
        max_bytes: int = 5_000_000,
        **_: Any,
    ) -> Dict[str, Any]:

        file_path = self._path(path)

        if not file_path.exists():

            return {
                "success": False,
                "error": f"File does not exist: {file_path}",
            }

        if not file_path.is_file():

            return {
                "success": False,
                "error": f"Not a file: {file_path}",
            }

        if file_path.stat().st_size > max_bytes:

            return {
                "success": False,
                "error": (
                    f"File is larger than the allowed "
                    f"{max_bytes} bytes."
                ),
            }

        # Use document reader if available.

        if self.document_reader is not None:

            for method_name in (
                "read",
                "read_file",
                "extract_text",
            ):

                method = getattr(
                    self.document_reader,
                    method_name,
                    None,
                )

                if callable(method):

                    try:

                        result = method(
                            str(file_path)
                        )

                        return {
                            "success": True,
                            "action": "read",
                            "path": str(file_path),
                            "content": result,
                        }

                    except Exception:
                        pass

        try:

            content = file_path.read_text(
                encoding=encoding,
                errors="replace",
            )

            return {
                "success": True,
                "action": "read",
                "path": str(file_path),
                "content": content,
                "size": len(content),
            }

        except UnicodeDecodeError:

            return {
                "success": False,
                "error": (
                    "File is not a supported text file."
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # METADATA
    # ========================================================================

    def get_metadata(
        self,
        path: str,
        **_: Any,
    ) -> Dict[str, Any]:

        file_path = self._path(path)

        if not file_path.exists():

            return {
                "success": False,
                "error": f"Path does not exist: {file_path}",
            }

        try:

            stat = file_path.stat()

            metadata = {
                "name": file_path.name,
                "path": str(file_path),
                "extension": file_path.suffix,
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir(),
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
            }

            if self.file_metadata is not None:

                for method_name in (
                    "get",
                    "read",
                    "extract",
                ):

                    method = getattr(
                        self.file_metadata,
                        method_name,
                        None,
                    )

                    if callable(method):

                        try:

                            extra = method(
                                str(file_path)
                            )

                            metadata[
                                "extended"
                            ] = extra

                            break

                        except Exception:
                            pass

            return {
                "success": True,
                "action": "metadata",
                "metadata": metadata,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # CREATE FILE
    # ========================================================================

    def create_file(
        self,
        path: str,
        content: str = "",
        overwrite: bool = False,
        encoding: str = "utf-8",
        **_: Any,
    ) -> Dict[str, Any]:

        file_path = self._path(path)

        if file_path.exists() and not overwrite:

            return {
                "success": False,
                "error": (
                    f"File already exists: {file_path}"
                ),
            }

        try:

            self._ensure_parent(
                file_path
            )

            file_path.write_text(
                content,
                encoding=encoding,
            )

            return {
                "success": True,
                "action": "create_file",
                "path": str(file_path),
                "size": file_path.stat().st_size,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # CREATE FOLDER
    # ========================================================================

    def create_folder(
        self,
        path: str,
        exist_ok: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:

        folder_path = self._path(path)

        try:

            folder_path.mkdir(
                parents=True,
                exist_ok=exist_ok,
            )

            return {
                "success": True,
                "action": "create_folder",
                "path": str(folder_path),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # LIST DIRECTORY
    # ========================================================================

    def list_folder(
        self,
        path: str,
        recursive: bool = False,
        include_hidden: bool = True,
        limit: int = 500,
        **_: Any,
    ) -> Dict[str, Any]:

        folder_path = self._path(path)

        if not folder_path.exists():

            return {
                "success": False,
                "error": (
                    f"Directory does not exist: {folder_path}"
                ),
            }

        if not folder_path.is_dir():

            return {
                "success": False,
                "error": (
                    f"Not a directory: {folder_path}"
                ),
            }

        try:

            if recursive:
                entries = folder_path.rglob("*")
            else:
                entries = folder_path.iterdir()

            results: List[Dict[str, Any]] = []

            for item in entries:

                if len(results) >= limit:
                    break

                if (
                    not include_hidden
                    and item.name.startswith(".")
                ):
                    continue

                try:

                    results.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "type": (
                                "directory"
                                if item.is_dir()
                                else "file"
                            ),
                            "size": (
                                item.stat().st_size
                                if item.is_file()
                                else None
                            ),
                        }
                    )

                except (
                    PermissionError,
                    OSError,
                ):
                    continue

            return {
                "success": True,
                "action": "list_folder",
                "path": str(folder_path),
                "items": results,
                "count": len(results),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # COPY
    # ========================================================================

    def copy(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:

        src = self._path(source)
        dst = self._path(destination)

        if not src.exists():

            return {
                "success": False,
                "error": f"Source does not exist: {src}",
            }

        if dst.exists() and not overwrite:

            return {
                "success": False,
                "error": (
                    f"Destination already exists: {dst}"
                ),
            }

        try:

            if self.copy_manager is not None:

                for method_name in (
                    "copy",
                    "copy_file",
                    "copy_item",
                ):

                    method = getattr(
                        self.copy_manager,
                        method_name,
                        None,
                    )

                    if callable(method):

                        try:

                            result = method(
                                str(src),
                                str(dst),
                            )

                            return {
                                "success": True,
                                "action": "copy",
                                "source": str(src),
                                "destination": str(dst),
                                "result": result,
                            }

                        except Exception:
                            pass

            self._ensure_parent(dst)

            if src.is_dir():

                shutil.copytree(
                    src,
                    dst,
                    dirs_exist_ok=overwrite,
                )

            else:

                shutil.copy2(
                    src,
                    dst,
                )

            return {
                "success": True,
                "action": "copy",
                "source": str(src),
                "destination": str(dst),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # MOVE
    # ========================================================================

    def move(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:

        src = self._path(source)
        dst = self._path(destination)

        if not src.exists():

            return {
                "success": False,
                "error": f"Source does not exist: {src}",
            }

        if dst.exists() and not overwrite:

            return {
                "success": False,
                "error": (
                    f"Destination already exists: {dst}"
                ),
            }

        try:

            if self.move_manager is not None:

                for method_name in (
                    "move",
                    "move_file",
                    "move_item",
                ):

                    method = getattr(
                        self.move_manager,
                        method_name,
                        None,
                    )

                    if callable(method):

                        try:

                            result = method(
                                str(src),
                                str(dst),
                            )

                            return {
                                "success": True,
                                "action": "move",
                                "source": str(src),
                                "destination": str(dst),
                                "result": result,
                            }

                        except Exception:
                            pass

            self._ensure_parent(dst)

            if dst.exists() and overwrite:

                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()

            shutil.move(
                str(src),
                str(dst),
            )

            return {
                "success": True,
                "action": "move",
                "source": str(src),
                "destination": str(dst),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # RENAME
    # ========================================================================

    def rename(
        self,
        path: str,
        new_name: str,
        overwrite: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:

        source = self._path(path)

        if not source.exists():

            return {
                "success": False,
                "error": f"Path does not exist: {source}",
            }

        destination = (
            source.parent
            / new_name
        ).resolve()

        if destination.exists() and not overwrite:

            return {
                "success": False,
                "error": (
                    f"Destination already exists: "
                    f"{destination}"
                ),
            }

        try:

            if self.rename_manager is not None:

                for method_name in (
                    "rename",
                    "rename_file",
                    "rename_item",
                ):

                    method = getattr(
                        self.rename_manager,
                        method_name,
                        None,
                    )

                    if callable(method):

                        try:

                            result = method(
                                str(source),
                                new_name,
                            )

                            return {
                                "success": True,
                                "action": "rename",
                                "source": str(source),
                                "destination": str(destination),
                                "result": result,
                            }

                        except Exception:
                            pass

            if (
                destination.exists()
                and overwrite
            ):

                if destination.is_dir():
                    shutil.rmtree(
                        destination
                    )
                else:
                    destination.unlink()

            source.rename(
                destination
            )

            return {
                "success": True,
                "action": "rename",
                "source": str(source),
                "destination": str(destination),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # DELETE
    # ========================================================================

    def delete(
        self,
        path: str,
        permanent: bool = True,
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._path(path)

        if not target.exists():

            return {
                "success": False,
                "error": (
                    f"Path does not exist: {target}"
                ),
            }

        try:

            if self.delete_manager is not None:

                for method_name in (
                    "delete",
                    "remove",
                    "delete_file",
                ):

                    method = getattr(
                        self.delete_manager,
                        method_name,
                        None,
                    )

                    if callable(method):

                        try:

                            result = method(
                                str(target)
                            )

                            return {
                                "success": True,
                                "action": "delete",
                                "path": str(target),
                                "result": result,
                            }

                        except Exception:
                            pass

            if target.is_dir():

                shutil.rmtree(
                    target
                )

            else:

                target.unlink()

            return {
                "success": True,
                "action": "delete",
                "path": str(target),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # OPEN
    # ========================================================================

    def open_file(
        self,
        path: str,
        **_: Any,
    ) -> Dict[str, Any]:

        file_path = self._path(path)

        if not file_path.exists():

            return {
                "success": False,
                "error": (
                    f"File does not exist: {file_path}"
                ),
            }

        try:

            if os.name == "nt":

                os.startfile(
                    str(file_path)
                )

            elif os.uname().sysname == "Darwin":

                subprocess.Popen(
                    [
                        "open",
                        str(file_path),
                    ]
                )

            else:

                subprocess.Popen(
                    [
                        "xdg-open",
                        str(file_path),
                    ]
                )

            return {
                "success": True,
                "action": "open",
                "path": str(file_path),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # ARCHIVE
    # ========================================================================

    def create_archive(
        self,
        source: str,
        destination: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        source_path = self._path(
            source
        )

        if not source_path.exists():

            return {
                "success": False,
                "error": (
                    f"Source does not exist: "
                    f"{source_path}"
                ),
            }

        if destination:

            destination_path = self._path(
                destination
            )

            if destination_path.suffix.lower() != ".zip":
                destination_path = destination_path.with_suffix(
                    ".zip"
                )

        else:

            destination_path = (
                source_path.parent
                / f"{source_path.name}.zip"
            )

        try:

            self._ensure_parent(
                destination_path
            )

            with zipfile.ZipFile(
                destination_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:

                if source_path.is_file():

                    archive.write(
                        source_path,
                        arcname=source_path.name,
                    )

                else:

                    for item in source_path.rglob("*"):

                        if item.is_file():

                            archive.write(
                                item,
                                arcname=str(
                                    item.relative_to(
                                        source_path.parent
                                    )
                                ),
                            )

            return {
                "success": True,
                "action": "archive",
                "source": str(source_path),
                "destination": str(destination_path),
                "size": destination_path.stat().st_size,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # DUPLICATE DETECTION
    # ========================================================================

    def find_duplicates(
        self,
        directory: str,
        recursive: bool = True,
        limit: int = 1000,
        **_: Any,
    ) -> Dict[str, Any]:

        root = self._path(
            directory
        )

        if not root.exists():

            return {
                "success": False,
                "error": (
                    f"Directory does not exist: {root}"
                ),
            }

        try:

            if self.duplicate_detector is not None:

                for method_name in (
                    "find",
                    "find_duplicates",
                    "scan",
                ):

                    method = getattr(
                        self.duplicate_detector,
                        method_name,
                        None,
                    )

                    if callable(method):

                        try:

                            result = method(
                                str(root)
                            )

                            return {
                                "success": True,
                                "action": "duplicates",
                                "directory": str(root),
                                "duplicates": result,
                            }

                        except Exception:
                            pass

            files_by_size: Dict[
                int,
                List[Path],
            ] = {}

            iterator = (
                root.rglob("*")
                if recursive
                else root.glob("*")
            )

            count = 0

            for item in iterator:

                if count >= limit:
                    break

                try:

                    if not item.is_file():
                        continue

                    size = item.stat().st_size

                    files_by_size.setdefault(
                        size,
                        [],
                    ).append(item)

                    count += 1

                except (
                    PermissionError,
                    OSError,
                ):
                    continue

            duplicates: Dict[
                str,
                List[str],
            ] = {}

            for size, candidates in (
                files_by_size.items()
            ):

                if len(candidates) < 2:
                    continue

                hashes: Dict[
                    str,
                    List[str],
                ] = {}

                for item in candidates:

                    try:

                        digest = self._hash_file(
                            item
                        )

                        hashes.setdefault(
                            digest,
                            [],
                        ).append(
                            str(item)
                        )

                    except Exception:
                        continue

                for digest, paths in hashes.items():

                    if len(paths) > 1:

                        duplicates[digest] = paths

            return {
                "success": True,
                "action": "duplicates",
                "directory": str(root),
                "duplicates": duplicates,
                "groups": len(duplicates),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # EXISTENCE
    # ========================================================================

    def exists(
        self,
        path: str,
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._path(
            path
        )

        return {
            "success": True,
            "action": "exists",
            "path": str(target),
            "exists": target.exists(),
            "is_file": target.is_file(),
            "is_directory": target.is_dir(),
        }

    # ========================================================================
    # FILE SIZE
    # ========================================================================

    def get_file_size(
        self,
        path: str,
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._path(
            path
        )

        if not target.exists():

            return {
                "success": False,
                "error": (
                    f"Path does not exist: {target}"
                ),
            }

        try:

            size = target.stat().st_size

            return {
                "success": True,
                "action": "file_size",
                "path": str(target),
                "bytes": size,
                "kilobytes": round(
                    size / 1024,
                    2,
                ),
                "megabytes": round(
                    size / (1024 * 1024),
                    2,
                ),
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    # ========================================================================
    # HASH
    # ========================================================================

    def get_file_hash(
        self,
        path: str,
        algorithm: str = "sha256",
        **_: Any,
    ) -> Dict[str, Any]:

        target = self._path(
            path
        )

        if not target.exists():

            return {
                "success": False,
                "error": (
                    f"File does not exist: {target}"
                ),
            }

        if not target.is_file():

            return {
                "success": False,
                "error": (
                    f"Path is not a file: {target}"
                ),
            }

        try:

            digest = self._hash_file(
                target,
                algorithm=algorithm,
            )

            return {
                "success": True,
                "action": "hash",
                "path": str(target),
                "algorithm": algorithm,
                "hash": digest,
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
            }

    @staticmethod
    def _hash_file(
        path: Path,
        algorithm: str = "sha256",
        chunk_size: int = 1024 * 1024,
    ) -> str:

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
                    chunk_size
                )

                if not chunk:
                    break

                digest.update(
                    chunk
                )

        return digest.hexdigest()


__all__ = [
    "FileAgent",
]


