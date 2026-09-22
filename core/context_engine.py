"""
RENIX AI - Context Engine
=========================

The Context Engine maintains the current contextual state of RENIX.

Responsibilities:
    - Track the current conversation context
    - Track active applications
    - Track active files
    - Track active tasks
    - Track recent intents
    - Track user/session context
    - Track entities and variables
    - Resolve references such as:
        "this"
        "that"
        "it"
        "the file"
        "the previous one"
        "same folder"
    - Provide context snapshots
    - Provide context history
    - Allow other RENIX modules to update context

This module does NOT execute commands.
It provides contextual information to the rest of RENIX.
"""

from __future__ import annotations

import copy
import logging
import time

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(
    "RENIX.ContextEngine"
)


# ============================================================================
# CONTEXT ENTRY
# ============================================================================

@dataclass
class ContextEntry:
    """
    Represents one contextual value.
    """

    key: str

    value: Any

    source: str = "system"

    timestamp: float = field(
        default_factory=time.time
    )

    confidence: float = 1.0

    temporary: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the context entry into a dictionary.
        """

        return {
            "key": self.key,
            "value": copy.deepcopy(
                self.value
            ),
            "source": self.source,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "temporary": self.temporary,
            "metadata": copy.deepcopy(
                self.metadata
            ),
        }


# ============================================================================
# CONTEXT SNAPSHOT
# ============================================================================

@dataclass
class ContextSnapshot:
    """
    A complete snapshot of RENIX contextual state.
    """

    timestamp: float

    values: dict[str, Any]

    active_application: Optional[str] = None

    active_file: Optional[str] = None

    active_folder: Optional[str] = None

    active_task: Optional[str] = None

    active_window: Optional[str] = None

    last_intent: Optional[str] = None

    conversation_topic: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert snapshot to dictionary.
        """

        return {
            "timestamp": self.timestamp,
            "values": copy.deepcopy(
                self.values
            ),
            "active_application": (
                self.active_application
            ),
            "active_file": (
                self.active_file
            ),
            "active_folder": (
                self.active_folder
            ),
            "active_task": (
                self.active_task
            ),
            "active_window": (
                self.active_window
            ),
            "last_intent": (
                self.last_intent
            ),
            "conversation_topic": (
                self.conversation_topic
            ),
            "metadata": copy.deepcopy(
                self.metadata
            ),
        }


# ============================================================================
# CONTEXT ENGINE
# ============================================================================

class ContextEngine:
    """
    Central contextual state manager for RENIX.
    """

    # ------------------------------------------------------------------------
    # Reference words
    # ------------------------------------------------------------------------

    REFERENCE_WORDS = {
        "this",
        "that",
        "it",
        "these",
        "those",
        "same",
        "previous",
        "last",
        "current",
        "above",
        "below",
        "here",
        "there",
    }

    # ------------------------------------------------------------------------
    # Common contextual keys
    # ------------------------------------------------------------------------

    STANDARD_KEYS = {
        "user",
        "session",
        "conversation",
        "topic",
        "active_application",
        "active_window",
        "active_file",
        "active_folder",
        "active_task",
        "last_intent",
        "last_command",
        "last_response",
        "last_file",
        "last_folder",
        "last_application",
        "last_url",
        "last_search",
        "last_message",
        "selected_item",
        "selected_items",
        "clipboard",
        "current_directory",
        "current_project",
        "current_subject",
        "current_device",
        "current_media",
    }

    def __init__(
        self,
        *,
        max_history: int = 500,
        max_entries: int = 1000,
    ) -> None:

        self.logger = logger

        self.max_history = max(
            1,
            int(max_history),
        )

        self.max_entries = max(
            1,
            int(max_entries),
        )

        # --------------------------------------------------------------------
        # Main context storage
        # --------------------------------------------------------------------

        self._entries: dict[
            str,
            ContextEntry,
        ] = {}

        # --------------------------------------------------------------------
        # Context history
        # --------------------------------------------------------------------

        self._history: deque[
            ContextSnapshot
        ] = deque(
            maxlen=self.max_history
        )

        # --------------------------------------------------------------------
        # Conversation
        # --------------------------------------------------------------------

        self.conversation_history: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.max_history
        )

        # --------------------------------------------------------------------
        # Entity memory
        # --------------------------------------------------------------------

        self.entities: dict[
            str,
            Any,
        ] = {}

        # --------------------------------------------------------------------
        # Variables
        # --------------------------------------------------------------------

        self.variables: dict[
            str,
            Any,
        ] = {}

        # --------------------------------------------------------------------
        # Counters
        # --------------------------------------------------------------------

        self.update_count = 0

        self.lookup_count = 0

        self.reference_resolution_count = 0

        self.snapshot_count = 0

        self.created_at = time.time()

        self.last_updated = (
            self.created_at
        )

        self.initialized = False

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the Context Engine.
        """

        self.initialized = True

        self.logger.info(
            "RENIX Context Engine initialized."
        )

        self.create_snapshot(
            metadata={
                "event": "initialization"
            }
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the Context Engine.
        """

        self.create_snapshot(
            metadata={
                "event": "shutdown"
            }
        )

        self.initialized = False

        self.logger.info(
            "RENIX Context Engine shutdown."
        )

    # ========================================================================
    # NORMALIZE KEY
    # ========================================================================

    @staticmethod
    def normalize_key(
        key: str,
    ) -> str:
        """
        Normalize a context key.
        """

        if key is None:

            return ""

        key = str(key).strip().lower()

        key = key.replace(
            " ",
            "_",
        )

        key = key.replace(
            "-",
            "_",
        )

        return key

    # ========================================================================
    # SET
    # ========================================================================

    def set(
        self,
        key: str,
        value: Any,
        *,
        source: str = "system",
        confidence: float = 1.0,
        temporary: bool = False,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        create_snapshot: bool = True,
    ) -> ContextEntry:
        """
        Set a contextual value.
        """

        normalized_key = (
            self.normalize_key(
                key
            )
        )

        if not normalized_key:

            raise ValueError(
                "Context key cannot be empty."
            )

        confidence = max(
            0.0,
            min(
                1.0,
                float(
                    confidence
                ),
            ),
        )

        entry = ContextEntry(
            key=normalized_key,
            value=copy.deepcopy(
                value
            ),
            source=str(
                source
            ),
            confidence=confidence,
            temporary=bool(
                temporary
            ),
            metadata=copy.deepcopy(
                metadata or {}
            ),
        )

        self._entries[
            normalized_key
        ] = entry

        self.update_count += 1

        self.last_updated = time.time()

        if create_snapshot:

            self.create_snapshot(
                metadata={
                    "event": "set",
                    "key": normalized_key,
                }
            )

        return entry

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        values: dict[str, Any],
        *,
        source: str = "system",
        confidence: float = 1.0,
        temporary: bool = False,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        create_snapshot: bool = True,
    ) -> None:
        """
        Update multiple contextual values.
        """

        if not isinstance(
            values,
            dict,
        ):

            raise TypeError(
                "values must be a dictionary."
            )

        for key, value in (
            values.items()
        ):

            self.set(
                key,
                value,
                source=source,
                confidence=confidence,
                temporary=temporary,
                metadata=metadata,
                create_snapshot=False,
            )

        if create_snapshot:

            self.create_snapshot(
                metadata={
                    "event": "bulk_update",
                    "keys": list(
                        values.keys()
                    ),
                }
            )

    # ========================================================================
    # GET
    # ========================================================================

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a contextual value.
        """

        normalized_key = (
            self.normalize_key(
                key
            )
        )

        self.lookup_count += 1

        entry = self._entries.get(
            normalized_key
        )

        if entry is None:

            return default

        return copy.deepcopy(
            entry.value
        )

    # ========================================================================
    # GET ENTRY
    # ========================================================================

    def get_entry(
        self,
        key: str,
    ) -> Optional[ContextEntry]:
        """
        Get the full ContextEntry.
        """

        normalized_key = (
            self.normalize_key(
                key
            )
        )

        self.lookup_count += 1

        entry = self._entries.get(
            normalized_key
        )

        if entry is None:

            return None

        return copy.deepcopy(
            entry
        )

    # ========================================================================
    # HAS
    # ========================================================================

    def has(
        self,
        key: str,
    ) -> bool:
        """
        Check whether a context key exists.
        """

        normalized_key = (
            self.normalize_key(
                key
            )
        )

        return (
            normalized_key
            in self._entries
        )

    # ========================================================================
    # DELETE
    # ========================================================================

    def delete(
        self,
        key: str,
        *,
        create_snapshot: bool = True,
    ) -> bool:
        """
        Delete a contextual value.
        """

        normalized_key = (
            self.normalize_key(
                key
            )
        )

        existed = (
            normalized_key
            in self._entries
        )

        if existed:

            del self._entries[
                normalized_key
            ]

            self.update_count += 1

            self.last_updated = (
                time.time()
            )

            if create_snapshot:

                self.create_snapshot(
                    metadata={
                        "event": "delete",
                        "key": normalized_key,
                    }
                )

        return existed

    # ========================================================================
    # CLEAR
    # ========================================================================

    def clear(
        self,
        *,
        preserve: Optional[
            set[str] | list[str] | tuple[str, ...]
        ] = None,
    ) -> None:
        """
        Clear context while optionally preserving selected keys.
        """

        preserve_keys = {
            self.normalize_key(
                key
            )
            for key in (
                preserve or []
            )
        }

        keys_to_delete = [
            key
            for key in self._entries
            if key not in preserve_keys
        ]

        for key in keys_to_delete:

            del self._entries[
                key
            ]

        self.entities.clear()

        self.variables.clear()

        self.update_count += 1

        self.last_updated = time.time()

        self.create_snapshot(
            metadata={
                "event": "clear",
                "preserved": list(
                    preserve_keys
                ),
            }
        )

    # ========================================================================
    # SET ACTIVE APPLICATION
    # ========================================================================

    def set_active_application(
        self,
        application: Optional[str],
    ) -> None:
        """
        Set the currently active application.
        """

        self.set(
            "active_application",
            application,
            source="computer",
        )

        if application:

            self.set(
                "last_application",
                application,
                source="computer",
            )

    # ========================================================================
    # GET ACTIVE APPLICATION
    # ========================================================================

    def get_active_application(
        self,
    ) -> Optional[str]:
        """
        Return active application.
        """

        return self.get(
            "active_application"
        )

    # ========================================================================
    # SET ACTIVE WINDOW
    # ========================================================================

    def set_active_window(
        self,
        window: Optional[str],
    ) -> None:
        """
        Set active window.
        """

        self.set(
            "active_window",
            window,
            source="computer",
        )

    # ========================================================================
    # SET ACTIVE FILE
    # ========================================================================

    def set_active_file(
        self,
        file_path: Optional[str],
    ) -> None:
        """
        Set currently selected/active file.
        """

        self.set(
            "active_file",
            file_path,
            source="files",
        )

        if file_path:

            self.set(
                "last_file",
                file_path,
                source="files",
            )

    # ========================================================================
    # GET ACTIVE FILE
    # ========================================================================

    def get_active_file(
        self,
    ) -> Optional[str]:
        """
        Return active file.
        """

        return self.get(
            "active_file"
        )

    # ========================================================================
    # SET ACTIVE FOLDER
    # ========================================================================

    def set_active_folder(
        self,
        folder_path: Optional[str],
    ) -> None:
        """
        Set currently active folder.
        """

        self.set(
            "active_folder",
            folder_path,
            source="files",
        )

        if folder_path:

            self.set(
                "last_folder",
                folder_path,
                source="files",
            )

            self.set(
                "current_directory",
                folder_path,
                source="files",
            )

    # ========================================================================
    # SET ACTIVE TASK
    # ========================================================================

    def set_active_task(
        self,
        task_id: Optional[str],
    ) -> None:
        """
        Set currently active task.
        """

        self.set(
            "active_task",
            task_id,
            source="task_manager",
        )

    # ========================================================================
    # SET PROJECT
    # ========================================================================

    def set_current_project(
        self,
        project: Optional[str],
    ) -> None:
        """
        Set active project.
        """

        self.set(
            "current_project",
            project,
            source="project",
        )

    # ========================================================================
    # SET TOPIC
    # ========================================================================

    def set_conversation_topic(
        self,
        topic: Optional[str],
    ) -> None:
        """
        Set current conversation topic.
        """

        self.set(
            "conversation_topic",
            topic,
            source="conversation",
        )

        self.set(
            "topic",
            topic,
            source="conversation",
        )

    # ========================================================================
    # GET TOPIC
    # ========================================================================

    def get_conversation_topic(
        self,
    ) -> Optional[str]:
        """
        Get current conversation topic.
        """

        return self.get(
            "conversation_topic"
        )

    # ========================================================================
    # RECORD INTENT
    # ========================================================================

    def record_intent(
        self,
        intent: Any,
    ) -> None:
        """
        Record an intent object or intent dictionary.
        """

        if hasattr(
            intent,
            "to_dict",
        ):

            data = intent.to_dict()

        elif isinstance(
            intent,
            dict,
        ):

            data = copy.deepcopy(
                intent
            )

        else:

            data = {
                "name": str(
                    intent
                )
            }

        self.set(
            "last_intent",
            data,
            source="intent_engine",
            create_snapshot=False,
        )

        self.set(
            "last_command",
            data.get(
                "raw_input",
                data.get(
                    "name"
                ),
            ),
            source="intent_engine",
            create_snapshot=False,
        )

        self.last_updated = time.time()

        self.create_snapshot(
            metadata={
                "event": "intent_recorded"
            }
        )

    # ========================================================================
    # RECORD RESPONSE
    # ========================================================================

    def record_response(
        self,
        response: Any,
    ) -> None:
        """
        Record RENIX's latest response.
        """

        self.set(
            "last_response",
            response,
            source="response_engine",
        )

    # ========================================================================
    # CONVERSATION MESSAGE
    # ========================================================================

    def add_conversation_message(
        self,
        role: str,
        content: Any,
        *,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:
        """
        Add a message to conversation context.
        """

        message = {
            "role": str(
                role
            ),
            "content": copy.deepcopy(
                content
            ),
            "timestamp": time.time(),
            "metadata": copy.deepcopy(
                metadata or {}
            ),
        }

        self.conversation_history.append(
            message
        )

        self.set(
            "conversation",
            list(
                self.conversation_history
            ),
            source="conversation",
            create_snapshot=False,
        )

        self.last_updated = time.time()

    # ========================================================================
    # GET CONVERSATION
    # ========================================================================

    def get_conversation(
        self,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Return conversation history.
        """

        messages = list(
            self.conversation_history
        )

        if limit is not None:

            limit = max(
                0,
                int(limit),
            )

            messages = messages[
                -limit:
            ]

        return copy.deepcopy(
            messages
        )

    # ========================================================================
    # ENTITY SET
    # ========================================================================

    def set_entity(
        self,
        name: str,
        value: Any,
        *,
        source: str = "system",
    ) -> None:
        """
        Store a contextual entity.
        """

        normalized = (
            self.normalize_key(
                name
            )
        )

        if not normalized:

            return

        self.entities[
            normalized
        ] = copy.deepcopy(
            value
        )

        self.set(
            f"entity_{normalized}",
            value,
            source=source,
            create_snapshot=False,
        )

        self.last_updated = time.time()

    # ========================================================================
    # ENTITY GET
    # ========================================================================

    def get_entity(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve contextual entity.
        """

        normalized = (
            self.normalize_key(
                name
            )
        )

        if normalized in self.entities:

            return copy.deepcopy(
                self.entities[
                    normalized
                ]
            )

        return self.get(
            f"entity_{normalized}",
            default,
        )

    # ========================================================================
    # VARIABLE SET
    # ========================================================================

    def set_variable(
        self,
        name: str,
        value: Any,
    ) -> None:
        """
        Set a contextual variable.
        """

        normalized = (
            self.normalize_key(
                name
            )
        )

        if not normalized:

            raise ValueError(
                "Variable name cannot be empty."
            )

        self.variables[
            normalized
        ] = copy.deepcopy(
            value
        )

        self.last_updated = time.time()

    # ========================================================================
    # VARIABLE GET
    # ========================================================================

    def get_variable(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """
        Get a contextual variable.
        """

        normalized = (
            self.normalize_key(
                name
            )
        )

        return copy.deepcopy(
            self.variables.get(
                normalized,
                default,
            )
        )

    # ========================================================================
    # RESOLVE REFERENCE
    # ========================================================================

    def resolve_reference(
        self,
        reference: str,
        *,
        expected_type: Optional[str] = None,
    ) -> Any:
        """
        Resolve natural-language references.

        Examples:

            "this file"
            "that file"
            "it"
            "the previous file"
            "same folder"
            "current application"

        """

        self.reference_resolution_count += 1

        normalized = (
            self.normalize_key(
                reference
            )
        )

        if not normalized:

            return None

        # --------------------------------------------------------------------
        # Direct key
        # --------------------------------------------------------------------

        if self.has(
            normalized
        ):

            return self.get(
                normalized
            )

        # --------------------------------------------------------------------
        # Direct entity
        # --------------------------------------------------------------------

        entity = self.get_entity(
            normalized
        )

        if entity is not None:

            return entity

        # --------------------------------------------------------------------
        # File references
        # --------------------------------------------------------------------

        if (
            "file" in normalized
            or normalized in {
                "it",
                "this",
                "that",
                "this_item",
                "that_item",
                "selected",
            }
        ):

            active_file = (
                self.get_active_file()
            )

            if active_file:

                return active_file

            last_file = self.get(
                "last_file"
            )

            if last_file:

                return last_file

            selected = self.get(
                "selected_item"
            )

            if selected:

                return selected

        # --------------------------------------------------------------------
        # Folder references
        # --------------------------------------------------------------------

        if (
            "folder" in normalized
            or "directory" in normalized
        ):

            active_folder = self.get(
                "active_folder"
            )

            if active_folder:

                return active_folder

            current_directory = self.get(
                "current_directory"
            )

            if current_directory:

                return current_directory

            last_folder = self.get(
                "last_folder"
            )

            if last_folder:

                return last_folder

        # --------------------------------------------------------------------
        # Application references
        # --------------------------------------------------------------------

        if (
            "application" in normalized
            or "app" in normalized
            or "program" in normalized
            or "window" in normalized
        ):

            active_app = self.get(
                "active_application"
            )

            if active_app:

                return active_app

            active_window = self.get(
                "active_window"
            )

            if active_window:

                return active_window

            last_app = self.get(
                "last_application"
            )

            if last_app:

                return last_app

        # --------------------------------------------------------------------
        # Task references
        # --------------------------------------------------------------------

        if (
            "task" in normalized
            or "job" in normalized
        ):

            task = self.get(
                "active_task"
            )

            if task:

                return task

        # --------------------------------------------------------------------
        # URL references
        # --------------------------------------------------------------------

        if (
            "url" in normalized
            or "website" in normalized
            or "site" in normalized
        ):

            return self.get(
                "last_url"
            )

        # --------------------------------------------------------------------
        # Search references
        # --------------------------------------------------------------------

        if (
            "search" in normalized
            or "query" in normalized
        ):

            return self.get(
                "last_search"
            )

        # --------------------------------------------------------------------
        # Clipboard
        # --------------------------------------------------------------------

        if (
            "clipboard" in normalized
            or "copied" in normalized
        ):

            return self.get(
                "clipboard"
            )

        # --------------------------------------------------------------------
        # Selected item
        # --------------------------------------------------------------------

        if (
            "selected" in normalized
            or normalized == "item"
        ):

            return self.get(
                "selected_item"
            )

        # --------------------------------------------------------------------
        # Generic "it"
        # --------------------------------------------------------------------

        if normalized == "it":

            candidates = [
                self.get(
                    "selected_item"
                ),
                self.get(
                    "active_file"
                ),
                self.get(
                    "active_application"
                ),
                self.get(
                    "last_command"
                ),
            ]

            for candidate in candidates:

                if candidate is not None:

                    return candidate

        return None

    # ========================================================================
    # RESOLVE TEXT REFERENCES
    # ========================================================================

    def resolve_text_references(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        Detect contextual reference words in a sentence and
        resolve them against current context.
        """

        normalized = (
            str(text)
            .strip()
            .lower()
        )

        results: dict[
            str,
            Any
        ] = {}

        # --------------------------------------------------------------------
        # File references
        # --------------------------------------------------------------------

        if (
            "this file" in normalized
            or "that file" in normalized
            or "the file" in normalized
            or "previous file" in normalized
        ):

            results[
                "file"
            ] = self.resolve_reference(
                "file"
            )

        # --------------------------------------------------------------------
        # Folder references
        # --------------------------------------------------------------------

        if (
            "this folder" in normalized
            or "that folder" in normalized
            or "the folder" in normalized
            or "same folder" in normalized
        ):

            results[
                "folder"
            ] = self.resolve_reference(
                "folder"
            )

        # --------------------------------------------------------------------
        # Application references
        # --------------------------------------------------------------------

        if (
            "this app" in normalized
            or "that app" in normalized
            or "the app" in normalized
            or "current app" in normalized
        ):

            results[
                "application"
            ] = self.resolve_reference(
                "application"
            )

        # --------------------------------------------------------------------
        # Window references
        # --------------------------------------------------------------------

        if (
            "this window" in normalized
            or "that window" in normalized
            or "current window" in normalized
        ):

            results[
                "window"
            ] = self.get(
                "active_window"
            )

        # --------------------------------------------------------------------
        # Task references
        # --------------------------------------------------------------------

        if (
            "this task" in normalized
            or "that task" in normalized
            or "current task" in normalized
        ):

            results[
                "task"
            ] = self.resolve_reference(
                "task"
            )

        # --------------------------------------------------------------------
        # Generic "it"
        # --------------------------------------------------------------------

        if (
            normalized == "it"
            or normalized.startswith(
                "it "
            )
            or " it " in normalized
        ):

            results[
                "it"
            ] = self.resolve_reference(
                "it"
            )

        return results

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def create_snapshot(
        self,
        *,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ContextSnapshot:
        """
        Create and store a context snapshot.
        """

        values = {
            key: copy.deepcopy(
                entry.value
            )
            for key, entry
            in self._entries.items()
        }

        snapshot = ContextSnapshot(
            timestamp=time.time(),
            values=values,
            active_application=self.get(
                "active_application"
            ),
            active_file=self.get(
                "active_file"
            ),
            active_folder=self.get(
                "active_folder"
            ),
            active_task=self.get(
                "active_task"
            ),
            active_window=self.get(
                "active_window"
            ),
            last_intent=self.get(
                "last_intent"
            ),
            conversation_topic=self.get(
                "conversation_topic"
            ),
            metadata=copy.deepcopy(
                metadata or {}
            ),
        )

        self._history.append(
            snapshot
        )

        self.snapshot_count += 1

        return copy.deepcopy(
            snapshot
        )

    # ========================================================================
    # GET SNAPSHOTS
    # ========================================================================

    def get_snapshots(
        self,
        limit: Optional[int] = None,
    ) -> list[ContextSnapshot]:
        """
        Return context snapshots.
        """

        snapshots = list(
            self._history
        )

        if limit is not None:

            limit = max(
                0,
                int(limit),
            )

            snapshots = snapshots[
                -limit:
            ]

        return copy.deepcopy(
            snapshots
        )

    # ========================================================================
    # RESTORE SNAPSHOT
    # ========================================================================

    def restore_snapshot(
        self,
        snapshot: ContextSnapshot,
    ) -> None:
        """
        Restore context from a snapshot.
        """

        if not isinstance(
            snapshot,
            ContextSnapshot,
        ):

            raise TypeError(
                "snapshot must be a ContextSnapshot."
            )

        self._entries.clear()

        for key, value in (
            snapshot.values.items()
        ):

            self.set(
                key,
                value,
                source="snapshot",
                create_snapshot=False,
            )

        self.last_updated = time.time()

        self.create_snapshot(
            metadata={
                "event": "snapshot_restored"
            }
        )

    # ========================================================================
    # GET ALL
    # ========================================================================

    def get_all(
        self,
    ) -> dict[str, Any]:
        """
        Return all contextual values.
        """

        return {
            key: copy.deepcopy(
                entry.value
            )
            for key, entry
            in self._entries.items()
        }

    # ========================================================================
    # GET ENTRIES
    # ========================================================================

    def get_entries(
        self,
    ) -> dict[str, ContextEntry]:
        """
        Return all ContextEntry objects.
        """

        return copy.deepcopy(
            self._entries
        )

    # ========================================================================
    # CONTEXT WINDOW
    # ========================================================================

    def build_context_window(
        self,
        *,
        include_conversation: bool = True,
        conversation_limit: int = 10,
        include_entities: bool = True,
        include_variables: bool = True,
    ) -> dict[str, Any]:
        """
        Build a compact context package for AI reasoning.
        """

        result: dict[
            str,
            Any
        ] = {
            "context": self.get_all(),
            "active_application": self.get(
                "active_application"
            ),
            "active_window": self.get(
                "active_window"
            ),
            "active_file": self.get(
                "active_file"
            ),
            "active_folder": self.get(
                "active_folder"
            ),
            "active_task": self.get(
                "active_task"
            ),
            "topic": self.get(
                "conversation_topic"
            ),
            "last_intent": self.get(
                "last_intent"
            ),
        }

        if include_conversation:

            result[
                "conversation"
            ] = self.get_conversation(
                conversation_limit
            )

        if include_entities:

            result[
                "entities"
            ] = copy.deepcopy(
                self.entities
            )

        if include_variables:

            result[
                "variables"
            ] = copy.deepcopy(
                self.variables
            )

        return result

    # ========================================================================
    # MATCH CONTEXT
    # ========================================================================

    def find_values(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Search contextual values by key or string representation.
        """

        normalized_query = (
            str(query)
            .strip()
            .lower()
        )

        if not normalized_query:

            return {}

        results = {}

        for key, entry in (
            self._entries.items()
        ):

            value_text = str(
                entry.value
            ).lower()

            if (
                normalized_query in key
                or normalized_query
                in value_text
            ):

                results[
                    key
                ] = copy.deepcopy(
                    entry.value
                )

        return results

    # ========================================================================
    # TEMPORARY CONTEXT
    # ========================================================================

    def cleanup_temporary(
        self,
    ) -> int:
        """
        Remove temporary context entries.
        """

        temporary_keys = [
            key
            for key, entry
            in self._entries.items()
            if entry.temporary
        ]

        for key in temporary_keys:

            del self._entries[
                key
            ]

        if temporary_keys:

            self.create_snapshot(
                metadata={
                    "event": (
                        "temporary_cleanup"
                    ),
                    "removed": temporary_keys,
                }
            )

        return len(
            temporary_keys
        )

    # ========================================================================
    # EXPORT
    # ========================================================================

    def export(
        self,
    ) -> dict[str, Any]:
        """
        Export the complete contextual state.
        """

        return {
            "entries": {
                key: entry.to_dict()
                for key, entry
                in self._entries.items()
            },
            "entities": copy.deepcopy(
                self.entities
            ),
            "variables": copy.deepcopy(
                self.variables
            ),
            "conversation": (
                self.get_conversation()
            ),
            "statistics": (
                self.statistics()
            ),
        }

    # ========================================================================
    # IMPORT
    # ========================================================================

    def import_context(
        self,
        data: dict[str, Any],
        *,
        source: str = "import",
    ) -> None:
        """
        Import contextual state.
        """

        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "Context data must be a dictionary."
            )

        entries = data.get(
            "entries",
            {},
        )

        if isinstance(
            entries,
            dict,
        ):

            for key, entry_data in (
                entries.items()
            ):

                if isinstance(
                    entry_data,
                    dict,
                ) and "value" in entry_data:

                    self.set(
                        key,
                        entry_data[
                            "value"
                        ],
                        source=source,
                        confidence=float(
                            entry_data.get(
                                "confidence",
                                1.0,
                            )
                        ),
                        temporary=bool(
                            entry_data.get(
                                "temporary",
                                False,
                            )
                        ),
                        create_snapshot=False,
                    )

                else:

                    self.set(
                        key,
                        entry_data,
                        source=source,
                        create_snapshot=False,
                    )

        entities = data.get(
            "entities",
            {},
        )

        if isinstance(
            entities,
            dict,
        ):

            self.entities.update(
                copy.deepcopy(
                    entities
                )
            )

        variables = data.get(
            "variables",
            {},
        )

        if isinstance(
            variables,
            dict,
        ):

            self.variables.update(
                copy.deepcopy(
                    variables
                )
            )

        self.last_updated = time.time()

        self.create_snapshot(
            metadata={
                "event": "context_imported"
            }
        )

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return Context Engine statistics.
        """

        return {
            "initialized": self.initialized,
            "entries": len(
                self._entries
            ),
            "entities": len(
                self.entities
            ),
            "variables": len(
                self.variables
            ),
            "conversation_messages": len(
                self.conversation_history
            ),
            "snapshots": len(
                self._history
            ),
            "snapshot_count": (
                self.snapshot_count
            ),
            "update_count": (
                self.update_count
            ),
            "lookup_count": (
                self.lookup_count
            ),
            "reference_resolution_count": (
                self.reference_resolution_count
            ),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }


# ============================================================================
# GLOBAL CONTEXT ENGINE
# ============================================================================

context_engine = ContextEngine()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def set_context(
    key: str,
    value: Any,
    *,
    source: str = "system",
    confidence: float = 1.0,
    temporary: bool = False,
) -> ContextEntry:
    """
    Set a value using the global Context Engine.
    """

    return context_engine.set(
        key,
        value,
        source=source,
        confidence=confidence,
        temporary=temporary,
    )


def get_context(
    key: str,
    default: Any = None,
) -> Any:
    """
    Get a value using the global Context Engine.
    """

    return context_engine.get(
        key,
        default,
    )


def resolve_context_reference(
    reference: str,
) -> Any:
    """
    Resolve a natural-language context reference.
    """

    return context_engine.resolve_reference(
        reference
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ContextEntry",
    "ContextSnapshot",
    "ContextEngine",
    "context_engine",
    "set_context",
    "get_context",
    "resolve_context_reference",
]


