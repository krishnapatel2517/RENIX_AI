"""
RENIX Holographic UI
File UI
=================

Renderer-independent holographic file browser interface.

Responsibilities:
- Display files and folders
- Track current directory
- Selection / multi-selection
- Search and filtering
- Sort modes
- Breadcrumb navigation
- File preview metadata
- Context actions
- Drag/drop state
- Clipboard-style file actions
- Renderer-independent snapshots

The actual filesystem operations should be handled by:
    RENIX/files/file_manager.py
    RENIX/files/file_operations.py
    RENIX/files/copy_manager.py
    RENIX/files/move_manager.py
    RENIX/files/rename_manager.py
    RENIX/files/delete_manager.py

This module intentionally keeps UI state separate from filesystem logic.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


# ============================================================================
# ENUMS
# ============================================================================


class FileItemType(str, Enum):
    FILE = "file"
    FOLDER = "folder"
    DRIVE = "drive"
    UNKNOWN = "unknown"


class FileSortMode(str, Enum):
    NAME = "name"
    SIZE = "size"
    TYPE = "type"
    MODIFIED = "modified"
    CREATED = "created"


class SortDirection(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class FileViewMode(str, Enum):
    GRID = "grid"
    LIST = "list"
    COMPACT = "compact"


class FileAction(str, Enum):
    OPEN = "open"
    PREVIEW = "preview"
    COPY = "copy"
    CUT = "cut"
    PASTE = "paste"
    MOVE = "move"
    RENAME = "rename"
    DELETE = "delete"
    NEW_FILE = "new_file"
    NEW_FOLDER = "new_folder"
    REFRESH = "refresh"
    PROPERTIES = "properties"


class ClipboardMode(str, Enum):
    NONE = "none"
    COPY = "copy"
    CUT = "cut"


# ============================================================================
# FILE ITEM
# ============================================================================


@dataclass
class FileItem:
    """
    UI representation of a file-system item.

    This object does not perform filesystem mutations.
    """

    id: str

    name: str

    path: str

    item_type: FileItemType = FileItemType.UNKNOWN

    size: int = 0

    modified_at: Optional[float] = None

    created_at: Optional[float] = None

    extension: str = ""

    mime_type: Optional[str] = None

    hidden: bool = False

    selected: bool = False

    thumbnail: Optional[str] = None

    icon: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_folder(self) -> bool:
        return self.item_type == FileItemType.FOLDER

    @property
    def is_file(self) -> bool:
        return self.item_type == FileItemType.FILE


# ============================================================================
# FILE ACTION RESULT
# ============================================================================


@dataclass
class FileActionResult:

    action: FileAction

    success: bool

    message: str = ""

    items: list[FileItem] = field(
        default_factory=list
    )

    data: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# FILE UI
# ============================================================================


class FileUI:

    def __init__(
        self,
        *,
        max_history: int = 100,
        view_mode: FileViewMode = FileViewMode.GRID,
        sort_mode: FileSortMode = FileSortMode.NAME,
        sort_direction: SortDirection = (
            SortDirection.ASCENDING
        ),
        show_hidden: bool = False,
    ) -> None:

        self.max_history = max(
            1,
            int(max_history),
        )

        self.view_mode = view_mode

        self.sort_mode = sort_mode

        self.sort_direction = sort_direction

        self.show_hidden = show_hidden

        self._current_path = str(
            Path.home()
        )

        self._items: dict[
            str,
            FileItem,
        ] = {}

        self._visible_ids: list[str] = []

        self._selected_ids: list[str] = []

        self._history: list[str] = []

        self._history_index = -1

        self._search_query = ""

        self._search_active = False

        self._preview_item_id: Optional[
            str
        ] = None

        self._context_item_id: Optional[
            str
        ] = None

        self._clipboard_items: list[str] = []

        self._clipboard_mode = (
            ClipboardMode.NONE
        )

        self._dragged_ids: list[str] = []

        self._drop_target_id: Optional[
            str
        ] = None

        self._running = False

        self._lock = threading.RLock()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._last_update = time.monotonic()

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:
                return

            self._running = True

            self._last_update = (
                time.monotonic()
            )

        self._emit(
            "started",
            self,
        )

    def stop(self) -> None:

        with self._lock:

            self._running = False

        self._emit(
            "stopped",
            self,
        )

    @property
    def running(self) -> bool:

        with self._lock:
            return self._running

    def update(
        self,
        delta_time: Optional[float] = None,
    ) -> None:

        now = time.monotonic()

        if delta_time is None:
            delta_time = (
                now - self._last_update
            )

        self._last_update = now

        self._emit(
            "updated",
            delta_time,
        )

    # ========================================================================
    # DIRECTORY
    # ========================================================================

    @property
    def current_path(self) -> str:

        with self._lock:
            return self._current_path

    def set_path(
        self,
        path: str,
        *,
        add_history: bool = True,
    ) -> bool:

        if not path:
            return False

        normalized = os.path.abspath(
            os.path.expanduser(path)
        )

        if not os.path.isdir(normalized):
            return False

        with self._lock:

            self._current_path = normalized

            self._clear_selection_locked()

            self._preview_item_id = None

            self._context_item_id = None

            if add_history:

                self._push_history_locked(
                    normalized
                )

        self._emit(
            "path_changed",
            normalized,
        )

        return True

    def navigate_up(self) -> bool:

        current = Path(
            self.current_path
        )

        parent = current.parent

        if str(parent) == str(current):
            return False

        return self.set_path(
            str(parent)
        )

    def navigate_home(self) -> bool:

        return self.set_path(
            str(Path.home())
        )

    def navigate_to(
        self,
        item_id: str,
    ) -> bool:

        item = self.get_item(
            item_id
        )

        if item is None:
            return False

        if not item.is_folder:
            return False

        return self.set_path(
            item.path
        )

    def go_back(self) -> bool:

        with self._lock:

            if self._history_index <= 0:
                return False

            self._history_index -= 1

            target = self._history[
                self._history_index
            ]

            self._current_path = target

        self._emit(
            "path_changed",
            target,
        )

        return True

    def go_forward(self) -> bool:

        with self._lock:

            if (
                self._history_index
                >= len(self._history) - 1
            ):
                return False

            self._history_index += 1

            target = self._history[
                self._history_index
            ]

            self._current_path = target

        self._emit(
            "path_changed",
            target,
        )

        return True

    # ========================================================================
    # BREADCRUMBS
    # ========================================================================

    def get_breadcrumbs(
        self,
    ) -> list[dict[str, str]]:

        path = Path(
            self.current_path
        )

        parts = path.parts

        breadcrumbs: list[
            dict[str, str]
        ] = []

        accumulated = Path(
            parts[0]
        )

        breadcrumbs.append(
            {
                "name": parts[0],
                "path": str(accumulated),
            }
        )

        for part in parts[1:]:

            accumulated = (
                accumulated / part
            )

            breadcrumbs.append(
                {
                    "name": part,
                    "path": str(
                        accumulated
                    ),
                }
            )

        return breadcrumbs

    # ========================================================================
    # ITEMS
    # ========================================================================

    def set_items(
        self,
        items: Iterable[FileItem],
    ) -> None:

        with self._lock:

            self._items = {
                item.id: item
                for item in items
            }

            self._apply_filter_and_sort_locked()

        self._emit(
            "items_changed",
            self.get_visible_items(),
        )

    def add_item(
        self,
        item: FileItem,
    ) -> None:

        with self._lock:

            self._items[item.id] = item

            self._apply_filter_and_sort_locked()

        self._emit(
            "items_changed",
            self.get_visible_items(),
        )

    def remove_item(
        self,
        item_id: str,
    ) -> bool:

        with self._lock:

            if item_id not in self._items:
                return False

            self._items.pop(
                item_id
            )

            if item_id in self._selected_ids:

                self._selected_ids.remove(
                    item_id
                )

            if (
                item_id
                == self._preview_item_id
            ):

                self._preview_item_id = None

            if (
                item_id
                == self._context_item_id
            ):

                self._context_item_id = None

            self._apply_filter_and_sort_locked()

        self._emit(
            "items_changed",
            self.get_visible_items(),
        )

        return True

    def clear_items(self) -> None:

        with self._lock:

            self._items.clear()

            self._visible_ids.clear()

            self._selected_ids.clear()

            self._preview_item_id = None

            self._context_item_id = None

        self._emit(
            "items_changed",
            [],
        )

    def get_item(
        self,
        item_id: str,
    ) -> Optional[FileItem]:

        with self._lock:
            return self._items.get(
                item_id
            )

    def get_visible_items(
        self,
    ) -> list[FileItem]:

        with self._lock:

            return [
                self._items[item_id]
                for item_id
                in self._visible_ids
                if item_id
                in self._items
            ]

    # ========================================================================
    # FILE DISCOVERY
    # ========================================================================

    def scan_directory(
        self,
        path: Optional[str] = None,
    ) -> list[FileItem]:

        directory = os.path.abspath(
            os.path.expanduser(
                path or self.current_path
            )
        )

        if not os.path.isdir(directory):
            return []

        result: list[FileItem] = []

        try:

            entries = list(
                os.scandir(directory)
            )

        except OSError as exc:

            self._emit(
                "scan_error",
                directory,
                str(exc),
            )

            return []

        for entry in entries:

            try:

                hidden = (
                    entry.name.startswith(".")
                )

                if (
                    hidden
                    and not self.show_hidden
                ):
                    continue

                stat = entry.stat(
                    follow_symlinks=False
                )

                if entry.is_dir(
                    follow_symlinks=False
                ):

                    item_type = (
                        FileItemType.FOLDER
                    )

                elif entry.is_file(
                    follow_symlinks=False
                ):

                    item_type = (
                        FileItemType.FILE
                    )

                else:

                    item_type = (
                        FileItemType.UNKNOWN
                    )

                suffix = (
                    Path(entry.name).suffix
                )

                item = FileItem(
                    id=uuid.uuid4().hex,
                    name=entry.name,
                    path=entry.path,
                    item_type=item_type,
                    size=(
                        0
                        if item_type
                        == FileItemType.FOLDER
                        else stat.st_size
                    ),
                    modified_at=(
                        stat.st_mtime
                    ),
                    created_at=(
                        getattr(
                            stat,
                            "st_ctime",
                            None,
                        )
                    ),
                    extension=suffix.lower(),
                    hidden=hidden,
                )

                result.append(item)

            except (
                OSError,
                PermissionError,
            ):
                continue

        self.set_items(result)

        self._emit(
            "directory_scanned",
            directory,
            result,
        )

        return result

    def refresh(self) -> list[FileItem]:

        result = self.scan_directory(
            self.current_path
        )

        self._emit(
            "refreshed",
            result,
        )

        return result

    # ========================================================================
    # SEARCH
    # ========================================================================

    def set_search(
        self,
        query: str,
    ) -> None:

        with self._lock:

            self._search_query = (
                query.strip()
            )

            self._search_active = bool(
                self._search_query
            )

            self._apply_filter_and_sort_locked()

        self._emit(
            "search_changed",
            self._search_query,
        )

    def clear_search(self) -> None:

        self.set_search("")

    @property
    def search_query(self) -> str:

        with self._lock:
            return self._search_query

    @property
    def search_active(self) -> bool:

        with self._lock:
            return self._search_active

    def search(
        self,
        query: str,
    ) -> list[FileItem]:

        self.set_search(query)

        return self.get_visible_items()

    # ========================================================================
    # FILTER + SORT
    # ========================================================================

    def set_sort(
        self,
        mode: FileSortMode,
        direction: Optional[
            SortDirection
        ] = None,
    ) -> None:

        with self._lock:

            self.sort_mode = mode

            if direction is not None:
                self.sort_direction = (
                    direction
                )

            self._apply_filter_and_sort_locked()

        self._emit(
            "sort_changed",
            self.sort_mode,
            self.sort_direction,
        )

    def toggle_sort_direction(self) -> None:

        with self._lock:

            if (
                self.sort_direction
                == SortDirection.ASCENDING
            ):

                self.sort_direction = (
                    SortDirection.DESCENDING
                )

            else:

                self.sort_direction = (
                    SortDirection.ASCENDING
                )

            self._apply_filter_and_sort_locked()

        self._emit(
            "sort_changed",
            self.sort_mode,
            self.sort_direction,
        )

    def set_show_hidden(
        self,
        value: bool,
    ) -> None:

        with self._lock:

            self.show_hidden = bool(value)

            self._apply_filter_and_sort_locked()

        self._emit(
            "hidden_files_changed",
            self.show_hidden,
        )

    def _apply_filter_and_sort_locked(
        self,
    ) -> None:

        items = list(
            self._items.values()
        )

        if not self.show_hidden:

            items = [
                item
                for item in items
                if not item.hidden
            ]

        query = (
            self._search_query.lower()
        )

        if query:

            items = [
                item
                for item in items
                if query
                in item.name.lower()
                or query
                in item.path.lower()
            ]

        reverse = (
            self.sort_direction
            == SortDirection.DESCENDING
        )

        if self.sort_mode == FileSortMode.NAME:

            items.sort(
                key=lambda item: (
                    item.name.lower()
                ),
                reverse=reverse,
            )

        elif self.sort_mode == FileSortMode.SIZE:

            items.sort(
                key=lambda item: item.size,
                reverse=reverse,
            )

        elif self.sort_mode == FileSortMode.TYPE:

            items.sort(
                key=lambda item: (
                    item.extension,
                    item.name.lower(),
                ),
                reverse=reverse,
            )

        elif self.sort_mode == FileSortMode.MODIFIED:

            items.sort(
                key=lambda item: (
                    item.modified_at
                    or 0.0
                ),
                reverse=reverse,
            )

        elif self.sort_mode == FileSortMode.CREATED:

            items.sort(
                key=lambda item: (
                    item.created_at
                    or 0.0
                ),
                reverse=reverse,
            )

        self._visible_ids = [
            item.id
            for item in items
        ]

    # ========================================================================
    # SELECTION
    # ========================================================================

    def select(
        self,
        item_id: str,
        *,
        additive: bool = False,
    ) -> bool:

        item = self.get_item(
            item_id
        )

        if item is None:
            return False

        with self._lock:

            if not additive:

                self._clear_selection_locked()

            if item_id not in self._selected_ids:

                self._selected_ids.append(
                    item_id
                )

            item.selected = True

        self._emit(
            "selection_changed",
            self.get_selected_items(),
        )

        return True

    def deselect(
        self,
        item_id: str,
    ) -> bool:

        item = self.get_item(
            item_id
        )

        if item is None:
            return False

        with self._lock:

            if item_id in self._selected_ids:

                self._selected_ids.remove(
                    item_id
                )

            item.selected = False

        self._emit(
            "selection_changed",
            self.get_selected_items(),
        )

        return True

    def toggle_selection(
        self,
        item_id: str,
    ) -> bool:

        if item_id in self.selected_ids:

            return self.deselect(
                item_id
            )

        return self.select(
            item_id,
            additive=True,
        )

    def select_all(self) -> None:

        with self._lock:

            self._selected_ids.clear()

            for item_id in self._visible_ids:

                item = self._items.get(
                    item_id
                )

                if item is None:
                    continue

                item.selected = True

                self._selected_ids.append(
                    item_id
                )

        self._emit(
            "selection_changed",
            self.get_selected_items(),
        )

    def clear_selection(self) -> None:

        with self._lock:

            self._clear_selection_locked()

        self._emit(
            "selection_changed",
            [],
        )

    def _clear_selection_locked(
        self,
    ) -> None:

        for item_id in self._selected_ids:

            item = self._items.get(
                item_id
            )

            if item:
                item.selected = False

        self._selected_ids.clear()

    @property
    def selected_ids(self) -> list[str]:

        with self._lock:
            return list(
                self._selected_ids
            )

    def get_selected_items(
        self,
    ) -> list[FileItem]:

        with self._lock:

            return [
                self._items[item_id]
                for item_id
                in self._selected_ids
                if item_id
                in self._items
            ]

    # ========================================================================
    # PREVIEW
    # ========================================================================

    def open_preview(
        self,
        item_id: str,
    ) -> bool:

        item = self.get_item(
            item_id
        )

        if item is None:
            return False

        with self._lock:

            self._preview_item_id = (
                item_id
            )

        self._emit(
            "preview_opened",
            item,
        )

        return True

    def close_preview(self) -> None:

        with self._lock:

            item_id = (
                self._preview_item_id
            )

            self._preview_item_id = None

        self._emit(
            "preview_closed",
            item_id,
        )

    def get_preview_item(
        self,
    ) -> Optional[FileItem]:

        with self._lock:

            if (
                self._preview_item_id
                is None
            ):
                return None

            return self._items.get(
                self._preview_item_id
            )

    # ========================================================================
    # CONTEXT MENU
    # ========================================================================

    def open_context_menu(
        self,
        item_id: Optional[str] = None,
    ) -> None:

        if item_id is not None:

            if self.get_item(
                item_id
            ) is None:

                return

        with self._lock:

            self._context_item_id = (
                item_id
            )

        self._emit(
            "context_menu_opened",
            item_id,
            self.get_context_actions(
                item_id
            ),
        )

    def close_context_menu(self) -> None:

        with self._lock:

            self._context_item_id = None

        self._emit(
            "context_menu_closed"
        )

    def get_context_actions(
        self,
        item_id: Optional[str] = None,
    ) -> list[FileAction]:

        item = (
            self.get_item(item_id)
            if item_id
            else None
        )

        if item is None:

            return [
                FileAction.NEW_FOLDER,
                FileAction.NEW_FILE,
                FileAction.PASTE,
                FileAction.REFRESH,
            ]

        actions = [
            FileAction.OPEN,
            FileAction.PREVIEW,
            FileAction.COPY,
            FileAction.CUT,
            FileAction.RENAME,
            FileAction.DELETE,
            FileAction.PROPERTIES,
        ]

        if item.is_folder:
            return actions

        return actions

    # ========================================================================
    # CLIPBOARD
    # ========================================================================

    def copy_selected(self) -> list[str]:

        with self._lock:

            self._clipboard_items = list(
                self._selected_ids
            )

            self._clipboard_mode = (
                ClipboardMode.COPY
            )

        self._emit(
            "clipboard_changed",
            self._clipboard_items,
            self._clipboard_mode,
        )

        return list(
            self._clipboard_items
        )

    def cut_selected(self) -> list[str]:

        with self._lock:

            self._clipboard_items = list(
                self._selected_ids
            )

            self._clipboard_mode = (
                ClipboardMode.CUT
            )

        self._emit(
            "clipboard_changed",
            self._clipboard_items,
            self._clipboard_mode,
        )

        return list(
            self._clipboard_items
        )

    def clear_clipboard(self) -> None:

        with self._lock:

            self._clipboard_items.clear()

            self._clipboard_mode = (
                ClipboardMode.NONE
            )

        self._emit(
            "clipboard_changed",
            [],
            ClipboardMode.NONE,
        )

    def get_clipboard(
        self,
    ) -> tuple[list[str], ClipboardMode]:

        with self._lock:

            return (
                list(
                    self._clipboard_items
                ),
                self._clipboard_mode,
            )

    # ========================================================================
    # DRAG / DROP
    # ========================================================================

    def begin_drag(
        self,
        item_ids: Optional[
            Iterable[str]
        ] = None,
    ) -> list[str]:

        if item_ids is None:

            ids = self.selected_ids

        else:

            ids = list(item_ids)

        valid_ids = [
            item_id
            for item_id in ids
            if self.get_item(
                item_id
            )
            is not None
        ]

        with self._lock:

            self._dragged_ids = valid_ids

            self._drop_target_id = None

        self._emit(
            "drag_started",
            valid_ids,
        )

        return valid_ids

    def set_drop_target(
        self,
        item_id: Optional[str],
    ) -> bool:

        if item_id is not None:

            item = self.get_item(
                item_id
            )

            if (
                item is None
                or not item.is_folder
            ):
                return False

        with self._lock:

            self._drop_target_id = item_id

        self._emit(
            "drop_target_changed",
            item_id,
        )

        return True

    def end_drag(
        self,
    ) -> tuple[list[str], Optional[str]]:

        with self._lock:

            dragged = list(
                self._dragged_ids
            )

            target = (
                self._drop_target_id
            )

            self._dragged_ids.clear()

            self._drop_target_id = None

        self._emit(
            "drag_ended",
            dragged,
            target,
        )

        return dragged, target

    # ========================================================================
    # VIEW
    # ========================================================================

    def set_view_mode(
        self,
        mode: FileViewMode,
    ) -> None:

        with self._lock:

            self.view_mode = mode

        self._emit(
            "view_mode_changed",
            mode,
        )

    # ========================================================================
    # ACTION DISPATCH
    # ========================================================================

    def request_action(
        self,
        action: FileAction,
        item_id: Optional[str] = None,
    ) -> FileActionResult:

        item = (
            self.get_item(item_id)
            if item_id
            else None
        )

        if (
            item_id is not None
            and item is None
        ):

            return FileActionResult(
                action=action,
                success=False,
                message="File item not found.",
            )

        self._emit(
            "action_requested",
            action,
            item,
        )

        if action == FileAction.OPEN:

            if item is None:

                return FileActionResult(
                    action=action,
                    success=False,
                    message="No item selected.",
                )

            if item.is_folder:

                success = self.set_path(
                    item.path
                )

                return FileActionResult(
                    action=action,
                    success=success,
                    message=(
                        "Folder opened."
                        if success
                        else "Unable to open folder."
                    ),
                    items=[item],
                )

            return FileActionResult(
                action=action,
                success=True,
                message="Open requested.",
                items=[item],
            )

        if action == FileAction.PREVIEW:

            if item is None:

                return FileActionResult(
                    action=action,
                    success=False,
                    message="No item selected.",
                )

            success = self.open_preview(
                item.id
            )

            return FileActionResult(
                action=action,
                success=success,
                message=(
                    "Preview opened."
                    if success
                    else "Unable to open preview."
                ),
                items=[item],
            )

        if action == FileAction.COPY:

            if item is not None:

                self.select(item.id)

            ids = self.copy_selected()

            return FileActionResult(
                action=action,
                success=bool(ids),
                message=(
                    "Items copied to clipboard."
                    if ids
                    else "Nothing selected."
                ),
            )

        if action == FileAction.CUT:

            if item is not None:

                self.select(item.id)

            ids = self.cut_selected()

            return FileActionResult(
                action=action,
                success=bool(ids),
                message=(
                    "Items marked for moving."
                    if ids
                    else "Nothing selected."
                ),
            )

        if action == FileAction.PASTE:

            self._emit(
                "paste_requested",
                self.current_path,
            )

            return FileActionResult(
                action=action,
                success=True,
                message="Paste requested.",
            )

        if action == FileAction.RENAME:

            self._emit(
                "rename_requested",
                item,
            )

            return FileActionResult(
                action=action,
                success=True,
                message="Rename requested.",
                items=(
                    [item]
                    if item
                    else []
                ),
            )

        if action == FileAction.DELETE:

            self._emit(
                "delete_requested",
                item,
            )

            return FileActionResult(
                action=action,
                success=True,
                message="Delete requested.",
                items=(
                    [item]
                    if item
                    else []
                ),
            )

        if action == FileAction.PROPERTIES:

            self._emit(
                "properties_requested",
                item,
            )

            return FileActionResult(
                action=action,
                success=True,
                message="Properties requested.",
                items=(
                    [item]
                    if item
                    else []
                ),
            )

        if action == FileAction.NEW_FILE:

            self._emit(
                "new_file_requested",
                self.current_path,
            )

            return FileActionResult(
                action=action,
                success=True,
                message="New file requested.",
            )

        if action == FileAction.NEW_FOLDER:

            self._emit(
                "new_folder_requested",
                self.current_path,
            )

            return FileActionResult(
                action=action,
                success=True,
                message="New folder requested.",
            )

        if action == FileAction.REFRESH:

            items = self.refresh()

            return FileActionResult(
                action=action,
                success=True,
                message="Directory refreshed.",
                items=items,
            )

        return FileActionResult(
            action=action,
            success=False,
            message="Unsupported file action.",
        )

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def get_snapshot(
        self,
    ) -> dict[str, Any]:

        selected = self.get_selected_items()

        preview = self.get_preview_item()

        clipboard_ids, clipboard_mode = (
            self.get_clipboard()
        )

        with self._lock:

            dragged_ids = list(
                self._dragged_ids
            )

            drop_target = (
                self._drop_target_id
            )

        return {
            "running": self.running,
            "current_path": self.current_path,
            "breadcrumbs": self.get_breadcrumbs(),
            "search": {
                "query": self.search_query,
                "active": self.search_active,
            },
            "view": {
                "mode": self.view_mode.value,
                "sort": self.sort_mode.value,
                "direction": (
                    self.sort_direction.value
                ),
                "show_hidden": self.show_hidden,
            },
            "items": [
                self._serialize_item(item)
                for item
                in self.get_visible_items()
            ],
            "selected": [
                self._serialize_item(item)
                for item in selected
            ],
            "preview": (
                self._serialize_item(
                    preview
                )
                if preview
                else None
            ),
            "clipboard": {
                "items": clipboard_ids,
                "mode": clipboard_mode.value,
            },
            "drag": {
                "items": dragged_ids,
                "drop_target": drop_target,
            },
        }

    @staticmethod
    def _serialize_item(
        item: FileItem,
    ) -> dict[str, Any]:

        return {
            "id": item.id,
            "name": item.name,
            "path": item.path,
            "type": item.item_type.value,
            "size": item.size,
            "modified_at": item.modified_at,
            "created_at": item.created_at,
            "extension": item.extension,
            "mime_type": item.mime_type,
            "hidden": item.hidden,
            "selected": item.selected,
            "thumbnail": item.thumbnail,
            "icon": item.icon,
            "metadata": dict(
                item.metadata
            ),
        }

    # ========================================================================
    # HISTORY
    # ========================================================================

    def _push_history_locked(
        self,
        path: str,
    ) -> None:

        if (
            self._history_index >= 0
            and self._history[
                self._history_index
            ] == path
        ):
            return

        if (
            self._history_index
            < len(self._history) - 1
        ):

            self._history = self._history[
                : self._history_index + 1
            ]

        self._history.append(
            path
        )

        self._history_index = (
            len(self._history) - 1
        )

        if len(self._history) > self.max_history:

            remove_count = (
                len(self._history)
                - self.max_history
            )

            self._history = (
                self._history[
                    remove_count:
                ]
            )

            self._history_index = (
                len(self._history) - 1
            )

    # ========================================================================
    # EVENTS
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not event:
            raise ValueError(
                "Event cannot be empty."
            )

        if not callable(callback):
            raise TypeError(
                "Callback must be callable."
            )

        with self._lock:

            self._callbacks.setdefault(
                event,
                [],
            ).append(callback)

    def off(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        with self._lock:

            callbacks = self._callbacks.get(
                event,
                [],
            )

            if callback in callbacks:

                callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks.get(
                    event,
                    [],
                )
            )

        for callback in callbacks:

            try:

                callback(
                    *args,
                    **kwargs,
                )

            except Exception:
                # UI callbacks must never crash RENIX.
                pass

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(self) -> dict[str, Any]:

        with self._lock:

            return {
                "running": self._running,
                "current_path": self._current_path,
                "item_count": len(
                    self._items
                ),
                "visible_count": len(
                    self._visible_ids
                ),
                "selected_count": len(
                    self._selected_ids
                ),
                "search_active": (
                    self._search_active
                ),
                "preview_open": (
                    self._preview_item_id
                    is not None
                ),
                "clipboard_mode": (
                    self._clipboard_mode.value
                ),
                "clipboard_count": len(
                    self._clipboard_items
                ),
                "dragging": bool(
                    self._dragged_ids
                ),
            }


# ============================================================================
# FACTORY
# ============================================================================


def create_file_ui() -> FileUI:

    ui = FileUI(
        max_history=100,
        view_mode=FileViewMode.GRID,
        sort_mode=FileSortMode.NAME,
        sort_direction=(
            SortDirection.ASCENDING
        ),
        show_hidden=False,
    )

    ui.start()

    return ui


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "FileItemType",
    "FileSortMode",
    "SortDirection",
    "FileViewMode",
    "FileAction",
    "ClipboardMode",
    "FileItem",
    "FileActionResult",
    "FileUI",
    "create_file_ui",
]


