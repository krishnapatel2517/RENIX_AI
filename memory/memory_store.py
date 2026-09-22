"""
RENIX Memory Store
==================

Low-level persistent storage layer for the RENIX memory subsystem.

Responsibilities
----------------
- Store Memory objects
- Retrieve Memory objects
- Update Memory objects
- Delete Memory objects
- Iterate over memories
- Search using basic filters
- Persist data to JSON
- Load data from JSON
- Provide thread-safe access
- Keep storage concerns separate from MemoryManager

This module does NOT decide:
- what a memory means
- how important it is
- how memory should be recalled
- how embeddings are generated
- how semantic similarity works

Those responsibilities belong to higher-level RENIX memory components.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence
import json
import logging
import os
import tempfile

from .memory_manager import (
    Memory,
    MemoryImportance,
    MemoryStatus,
    MemoryType,
)


logger = logging.getLogger(__name__)


class MemoryStoreError(Exception):
    """Base exception for memory-store failures."""


class MemoryNotFoundError(MemoryStoreError):
    """Raised when a requested memory does not exist."""


class MemorySerializationError(MemoryStoreError):
    """Raised when a memory cannot be serialized or deserialized."""


class MemoryStore:
    """
    Thread-safe local memory repository.

    The store maintains an in-memory index and can optionally persist
    the complete collection to a JSON file.

    Parameters
    ----------
    storage_path:
        Path to the JSON storage file. If None, the store remains
        in-memory only.

    auto_save:
        Automatically save after mutations.

    create_storage:
        Create the storage directory/file when required.
    """

    VERSION = 1

    def __init__(
        self,
        storage_path: Optional[str | Path] = None,
        *,
        auto_save: bool = True,
        create_storage: bool = True,
    ) -> None:

        self.storage_path: Optional[Path] = (
            Path(storage_path)
            if storage_path
            else None
        )

        self.auto_save = bool(auto_save)

        self.create_storage = bool(
            create_storage
        )

        self._memories: Dict[str, Memory] = {}

        self._lock = RLock()

        self._loaded = False

        self._dirty = False

        if self.storage_path is not None:
            if self.create_storage:
                self._ensure_storage_directory()

            self.load()

        else:
            self._loaded = True

    # ======================================================================
    # BASIC PROPERTIES
    # ======================================================================

    @property
    def loaded(self) -> bool:
        """Return whether the store has completed loading."""

        return self._loaded

    @property
    def dirty(self) -> bool:
        """Return whether unsaved changes exist."""

        with self._lock:
            return self._dirty

    @property
    def count(self) -> int:
        """Return the number of stored memories."""

        with self._lock:
            return len(
                self._memories
            )

    # ======================================================================
    # CREATE
    # ======================================================================

    def add(
        self,
        memory: Memory,
        *,
        overwrite: bool = False,
    ) -> Memory:
        """
        Add a Memory object to the store.

        Parameters
        ----------
        memory:
            Memory instance to store.

        overwrite:
            If False, an existing ID causes an error.
            If True, the existing memory is replaced.
        """

        self._validate_memory(
            memory
        )

        with self._lock:

            if (
                memory.memory_id in self._memories
                and not overwrite
            ):
                raise MemoryStoreError(
                    f"Memory already exists: "
                    f"{memory.memory_id}"
                )

            self._memories[
                memory.memory_id
            ] = memory

            self._dirty = True

            if self.auto_save:
                self.save()

        return memory

    def add_many(
        self,
        memories: Iterable[Memory],
        *,
        overwrite: bool = False,
    ) -> int:
        """
        Add multiple memories.

        Returns the number of memories added.
        """

        memories = list(memories)

        for memory in memories:
            self._validate_memory(
                memory
            )

        added = 0

        with self._lock:

            for memory in memories:

                if (
                    memory.memory_id in self._memories
                    and not overwrite
                ):
                    continue

                self._memories[
                    memory.memory_id
                ] = memory

                added += 1

            if added:
                self._dirty = True

                if self.auto_save:
                    self.save()

        return added

    # ======================================================================
    # READ
    # ======================================================================

    def get(
        self,
        memory_id: str,
    ) -> Optional[Memory]:
        """
        Return a memory by ID.

        Returns None when it does not exist.
        """

        with self._lock:
            return self._memories.get(
                memory_id
            )

    def require(
        self,
        memory_id: str,
    ) -> Memory:
        """
        Return a memory by ID.

        Raises MemoryNotFoundError if absent.
        """

        memory = self.get(
            memory_id
        )

        if memory is None:
            raise MemoryNotFoundError(
                f"Memory not found: {memory_id}"
            )

        return memory

    def exists(
        self,
        memory_id: str,
    ) -> bool:
        """Return whether a memory exists."""

        with self._lock:
            return memory_id in self._memories

    def all(
        self,
        *,
        include_deleted: bool = True,
    ) -> List[Memory]:
        """
        Return all stored memories.
        """

        with self._lock:

            if include_deleted:
                return list(
                    self._memories.values()
                )

            return [
                memory
                for memory in self._memories.values()
                if memory.status
                != MemoryStatus.DELETED
            ]

    def values(
        self,
    ) -> List[Memory]:
        """Return a snapshot of all Memory objects."""

        return self.all()

    def ids(self) -> List[str]:
        """Return all memory IDs."""

        with self._lock:
            return list(
                self._memories.keys()
            )

    def __len__(self) -> int:
        return self.count

    def __contains__(
        self,
        memory_id: str,
    ) -> bool:
        return self.exists(
            memory_id
        )

    def __iter__(self) -> Iterator[Memory]:
        return iter(
            self.all()
        )

    # ======================================================================
    # UPDATE
    # ======================================================================

    def update(
        self,
        memory: Memory,
    ) -> Memory:
        """
        Replace an existing memory.

        Raises MemoryNotFoundError when the ID does not exist.
        """

        self._validate_memory(
            memory
        )

        with self._lock:

            if memory.memory_id not in self._memories:
                raise MemoryNotFoundError(
                    f"Memory not found: "
                    f"{memory.memory_id}"
                )

            self._memories[
                memory.memory_id
            ] = memory

            self._dirty = True

            if self.auto_save:
                self.save()

        return memory

    def update_fields(
        self,
        memory_id: str,
        **fields: Any,
    ) -> Memory:
        """
        Update selected fields of an existing memory.

        Unknown fields are rejected.
        """

        with self._lock:

            memory = self.require(
                memory_id
            )

            allowed = {
                "content",
                "memory_type",
                "importance",
                "status",
                "created_at",
                "updated_at",
                "accessed_at",
                "access_count",
                "user_id",
                "session_id",
                "conversation_id",
                "project_id",
                "task_id",
                "source",
                "tags",
                "metadata",
                "confidence",
                "expires_at",
                "pinned",
                "sensitive",
                "embedding_id",
            }

            invalid = set(
                fields
            ) - allowed

            if invalid:
                raise MemoryStoreError(
                    "Unknown memory fields: "
                    + ", ".join(
                        sorted(invalid)
                    )
                )

            for key, value in fields.items():

                if key == "memory_type":
                    value = (
                        self._normalize_memory_type(
                            value
                        )
                    )

                elif key == "importance":
                    value = (
                        self._normalize_importance(
                            value
                        )
                    )

                elif key == "status":
                    value = (
                        self._normalize_status(
                            value
                        )
                    )

                setattr(
                    memory,
                    key,
                    value,
                )

            self._validate_memory(
                memory
            )

            self._dirty = True

            if self.auto_save:
                self.save()

            return memory

    # ======================================================================
    # DELETE
    # ======================================================================

    def delete(
        self,
        memory_id: str,
        *,
        permanent: bool = False,
    ) -> bool:
        """
        Delete a memory.

        permanent=False:
            Marks the memory as deleted.

        permanent=True:
            Removes the memory completely.
        """

        with self._lock:

            if memory_id not in self._memories:
                return False

            if permanent:

                del self._memories[
                    memory_id
                ]

            else:

                memory = self._memories[
                    memory_id
                ]

                memory.status = (
                    MemoryStatus.DELETED
                )

            self._dirty = True

            if self.auto_save:
                self.save()

            return True

    def delete_many(
        self,
        memory_ids: Sequence[str],
        *,
        permanent: bool = False,
    ) -> int:
        """
        Delete multiple memories.

        Returns the number successfully deleted.
        """

        deleted = 0

        with self._lock:

            for memory_id in memory_ids:

                if memory_id not in self._memories:
                    continue

                if permanent:

                    del self._memories[
                        memory_id
                    ]

                else:

                    self._memories[
                        memory_id
                    ].status = (
                        MemoryStatus.DELETED
                    )

                deleted += 1

            if deleted:

                self._dirty = True

                if self.auto_save:
                    self.save()

        return deleted

    def clear(
        self,
        *,
        permanent: bool = False,
        preserve_pinned: bool = True,
    ) -> int:
        """
        Clear the store.

        By default pinned memories are preserved.
        """

        removed = 0

        with self._lock:

            if permanent:

                ids_to_remove = []

                for memory in self._memories.values():

                    if (
                        preserve_pinned
                        and memory.pinned
                    ):
                        continue

                    ids_to_remove.append(
                        memory.memory_id
                    )

                for memory_id in ids_to_remove:

                    del self._memories[
                        memory_id
                    ]

                    removed += 1

            else:

                for memory in self._memories.values():

                    if (
                        preserve_pinned
                        and memory.pinned
                    ):
                        continue

                    if (
                        memory.status
                        == MemoryStatus.DELETED
                    ):
                        continue

                    memory.status = (
                        MemoryStatus.DELETED
                    )

                    removed += 1

            if removed:

                self._dirty = True

                if self.auto_save:
                    self.save()

        return removed

    # ======================================================================
    # FILTERING
    # ======================================================================

    def filter(
        self,
        *,
        memory_type: Optional[
            MemoryType | str
        ] = None,
        importance: Optional[
            MemoryImportance | str
        ] = None,
        status: Optional[
            MemoryStatus | str
        ] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        pinned: Optional[bool] = None,
        sensitive: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """
        Return memories matching supplied filters.
        """

        normalized_type = (
            self._normalize_memory_type(
                memory_type
            )
            if memory_type is not None
            else None
        )

        normalized_importance = (
            self._normalize_importance(
                importance
            )
            if importance is not None
            else None
        )

        normalized_status = (
            self._normalize_status(
                status
            )
            if status is not None
            else None
        )

        required_tags = {
            str(tag).lower()
            for tag in (tags or [])
        }

        results: List[Memory] = []

        with self._lock:

            for memory in self._memories.values():

                if (
                    normalized_type is not None
                    and memory.memory_type
                    != normalized_type
                ):
                    continue

                if (
                    normalized_importance is not None
                    and memory.importance
                    != normalized_importance
                ):
                    continue

                if (
                    normalized_status is not None
                    and memory.status
                    != normalized_status
                ):
                    continue

                if (
                    user_id is not None
                    and memory.user_id
                    != user_id
                ):
                    continue

                if (
                    session_id is not None
                    and memory.session_id
                    != session_id
                ):
                    continue

                if (
                    conversation_id is not None
                    and memory.conversation_id
                    != conversation_id
                ):
                    continue

                if (
                    project_id is not None
                    and memory.project_id
                    != project_id
                ):
                    continue

                if (
                    task_id is not None
                    and memory.task_id
                    != task_id
                ):
                    continue

                if (
                    pinned is not None
                    and memory.pinned
                    != pinned
                ):
                    continue

                if (
                    sensitive is not None
                    and memory.sensitive
                    != sensitive
                ):
                    continue

                if required_tags:

                    memory_tags = {
                        str(tag).lower()
                        for tag in memory.tags
                    }

                    if not required_tags.issubset(
                        memory_tags
                    ):
                        continue

                results.append(
                    memory
                )

                if (
                    limit is not None
                    and len(results)
                    >= max(
                        1,
                        limit,
                    )
                ):
                    break

        return results

    # ======================================================================
    # TEXT SEARCH
    # ======================================================================

    def text_search(
        self,
        text: str,
        *,
        limit: int = 20,
        include_deleted: bool = False,
    ) -> List[Memory]:
        """
        Perform simple lexical search.

        Higher-level semantic retrieval belongs in retrieval.py.
        """

        if not text:
            return []

        query = self._tokenize(
            text
        )

        if not query:
            return []

        candidates: List[
            tuple[float, Memory]
        ] = []

        with self._lock:

            for memory in self._memories.values():

                if (
                    memory.status
                    == MemoryStatus.DELETED
                    and not include_deleted
                ):
                    continue

                content_tokens = self._tokenize(
                    memory.content
                )

                if not content_tokens:
                    continue

                overlap = (
                    query
                    & content_tokens
                )

                if not overlap:
                    continue

                score = (
                    len(overlap)
                    / max(
                        len(query),
                        1,
                    )
                )

                if text.lower().strip() in (
                    memory.content.lower()
                ):
                    score += 1.0

                candidates.append(
                    (
                        score,
                        memory,
                    )
                )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            memory
            for _, memory
            in candidates[:max(1, limit)]
        ]

    # ======================================================================
    # TYPE HELPERS
    # ======================================================================

    def by_type(
        self,
        memory_type: MemoryType | str,
        *,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Return memories belonging to a type."""

        return self.filter(
            memory_type=memory_type,
            limit=limit,
        )

    def by_project(
        self,
        project_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Return memories belonging to a project."""

        return self.filter(
            project_id=project_id,
            limit=limit,
        )

    def by_task(
        self,
        task_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Return memories belonging to a task."""

        return self.filter(
            task_id=task_id,
            limit=limit,
        )

    def by_session(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Return memories belonging to a session."""

        return self.filter(
            session_id=session_id,
            limit=limit,
        )

    def pinned(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[Memory]:
        """Return pinned memories."""

        return self.filter(
            pinned=True,
            limit=limit,
        )

    # ======================================================================
    # PERSISTENCE
    # ======================================================================

    def save(self) -> None:
        """
        Save the current memory collection.

        Uses an atomic temporary-file replacement to reduce the risk
        of leaving corrupted JSON after an interrupted write.
        """

        if self.storage_path is None:
            self._dirty = False
            return

        with self._lock:

            self._ensure_storage_directory()

            payload = {
                "version": self.VERSION,
                "memories": [
                    self._serialize_memory(
                        memory
                    )
                    for memory
                    in self._memories.values()
                ],
            }

            target = self.storage_path

            temporary_path: Optional[Path] = None

            try:

                fd, temporary_name = (
                    tempfile.mkstemp(
                        prefix=(
                            f".{target.stem}_"
                        ),
                        suffix=".tmp",
                        dir=str(
                            target.parent
                        ),
                    )
                )

                os.close(fd)

                temporary_path = Path(
                    temporary_name
                )

                with temporary_path.open(
                    "w",
                    encoding="utf-8",
                ) as file:

                    json.dump(
                        payload,
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )

                    file.flush()

                    os.fsync(
                        file.fileno()
                    )

                os.replace(
                    temporary_path,
                    target,
                )

                self._dirty = False

            except Exception as exc:

                logger.exception(
                    "Failed to save RENIX memory store."
                )

                if (
                    temporary_path is not None
                    and temporary_path.exists()
                ):
                    try:
                        temporary_path.unlink()
                    except OSError:
                        pass

                raise MemoryStoreError(
                    "Unable to save memory store."
                ) from exc

    def load(self) -> int:
        """
        Load memories from the configured storage file.

        Returns the number of memories loaded.
        """

        if self.storage_path is None:
            self._loaded = True
            return 0

        with self._lock:

            if not self.storage_path.exists():

                self._loaded = True
                self._dirty = False

                return 0

            try:

                with self.storage_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:

                    payload = json.load(
                        file
                    )

                if not isinstance(
                    payload,
                    dict,
                ):
                    raise MemoryStoreError(
                        "Memory store root must be an object."
                    )

                version = payload.get(
                    "version",
                    1,
                )

                if not isinstance(
                    version,
                    int,
                ):
                    version = 1

                memories = payload.get(
                    "memories",
                    [],
                )

                if not isinstance(
                    memories,
                    list,
                ):
                    raise MemoryStoreError(
                        "'memories' must be a list."
                    )

                loaded: Dict[
                    str,
                    Memory,
                ] = {}

                for raw_memory in memories:

                    try:

                        memory = (
                            self._deserialize_memory(
                                raw_memory
                            )
                        )

                        loaded[
                            memory.memory_id
                        ] = memory

                    except Exception as exc:

                        logger.warning(
                            "Skipping invalid memory "
                            "record: %s",
                            exc,
                        )

                self._memories = loaded

                self._loaded = True

                self._dirty = False

                logger.info(
                    "Loaded %d memories from RENIX store "
                    "(version %d).",
                    len(loaded),
                    version,
                )

                return len(
                    loaded
                )

            except MemoryStoreError:
                raise

            except json.JSONDecodeError as exc:

                raise MemoryStoreError(
                    "Memory store contains invalid JSON."
                ) from exc

            except OSError as exc:

                raise MemoryStoreError(
                    "Unable to read memory store."
                ) from exc

    def reload(self) -> int:
        """Discard current in-memory state and reload from disk."""

        with self._lock:

            self._memories.clear()

            return self.load()

    # ======================================================================
    # EXPORT / IMPORT
    # ======================================================================

    def export_data(
        self,
        *,
        include_deleted: bool = True,
    ) -> Dict[str, Any]:
        """
        Export the complete store into a serializable dictionary.
        """

        with self._lock:

            memories = []

            for memory in self._memories.values():

                if (
                    not include_deleted
                    and memory.status
                    == MemoryStatus.DELETED
                ):
                    continue

                memories.append(
                    self._serialize_memory(
                        memory
                    )
                )

            return {
                "version": self.VERSION,
                "memories": memories,
            }

    def import_data(
        self,
        data: Dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> int:
        """
        Import a serialized memory-store dictionary.

        Returns the number of imported records.
        """

        if not isinstance(
            data,
            dict,
        ):
            raise MemoryStoreError(
                "Import data must be a dictionary."
            )

        raw_memories = data.get(
            "memories",
            [],
        )

        if not isinstance(
            raw_memories,
            list,
        ):
            raise MemoryStoreError(
                "Import 'memories' must be a list."
            )

        memories: List[
            Memory
        ] = []

        for raw_memory in raw_memories:

            memories.append(
                self._deserialize_memory(
                    raw_memory
                )
            )

        return self.add_many(
            memories,
            overwrite=overwrite,
        )

    # ======================================================================
    # SNAPSHOT / BACKUP
    # ======================================================================

    def snapshot(self) -> Dict[str, Any]:
        """
        Return an immutable-style data snapshot.

        The returned dictionary contains plain Python values.
        """

        return self.export_data(
            include_deleted=True
        )

    def save_backup(
        self,
        backup_path: str | Path,
    ) -> Path:
        """
        Save a standalone backup of the current store.
        """

        backup = Path(
            backup_path
        )

        backup.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = self.export_data(
            include_deleted=True
        )

        try:

            with backup.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        except OSError as exc:

            raise MemoryStoreError(
                "Unable to create memory backup."
            ) from exc

        return backup

    # ======================================================================
    # VALIDATION
    # ======================================================================

    @staticmethod
    def _validate_memory(
        memory: Memory,
    ) -> None:

        if not isinstance(
            memory,
            Memory,
        ):
            raise TypeError(
                "MemoryStore accepts only Memory objects."
            )

        if not memory.memory_id:
            raise MemorySerializationError(
                "Memory ID cannot be empty."
            )

        if not isinstance(
            memory.content,
            str,
        ):
            raise MemorySerializationError(
                "Memory content must be a string."
            )

        if not isinstance(
            memory.tags,
            list,
        ):
            raise MemorySerializationError(
                "Memory tags must be a list."
            )

        if not isinstance(
            memory.metadata,
            dict,
        ):
            raise MemorySerializationError(
                "Memory metadata must be a dictionary."
            )

    # ======================================================================
    # SERIALIZATION
    # ======================================================================

    @staticmethod
    def _serialize_memory(
        memory: Memory,
    ) -> Dict[str, Any]:

        try:

            data = asdict(
                memory
            )

            data["memory_type"] = (
                memory.memory_type.value
                if isinstance(
                    memory.memory_type,
                    MemoryType,
                )
                else str(
                    memory.memory_type
                )
            )

            data["importance"] = (
                memory.importance.value
                if isinstance(
                    memory.importance,
                    MemoryImportance,
                )
                else str(
                    memory.importance
                )
            )

            data["status"] = (
                memory.status.value
                if isinstance(
                    memory.status,
                    MemoryStatus,
                )
                else str(
                    memory.status
                )
            )

            return data

        except Exception as exc:

            raise MemorySerializationError(
                "Unable to serialize memory."
            ) from exc

    @staticmethod
    def _deserialize_memory(
        data: Dict[str, Any],
    ) -> Memory:

        if not isinstance(
            data,
            dict,
        ):
            raise MemorySerializationError(
                "Memory record must be a dictionary."
            )

        try:

            return Memory(
                memory_id=str(
                    data["memory_id"]
                ),
                content=str(
                    data.get(
                        "content",
                        "",
                    )
                ),
                memory_type=(
                    MemoryStore._normalize_memory_type(
                        data.get(
                            "memory_type",
                            MemoryType.LONG_TERM.value,
                        )
                    )
                ),
                importance=(
                    MemoryStore._normalize_importance(
                        data.get(
                            "importance",
                            MemoryImportance.NORMAL.value,
                        )
                    )
                ),
                status=(
                    MemoryStore._normalize_status(
                        data.get(
                            "status",
                            MemoryStatus.ACTIVE.value,
                        )
                    )
                ),
                created_at=str(
                    data.get(
                        "created_at",
                        "",
                    )
                ),
                updated_at=str(
                    data.get(
                        "updated_at",
                        "",
                    )
                ),
                accessed_at=str(
                    data.get(
                        "accessed_at",
                        "",
                    )
                ),
                access_count=int(
                    data.get(
                        "access_count",
                        0,
                    )
                ),
                user_id=data.get(
                    "user_id"
                ),
                session_id=data.get(
                    "session_id"
                ),
                conversation_id=data.get(
                    "conversation_id"
                ),
                project_id=data.get(
                    "project_id"
                ),
                task_id=data.get(
                    "task_id"
                ),
                source=data.get(
                    "source"
                ),
                tags=list(
                    data.get(
                        "tags",
                        [],
                    )
                ),
                metadata=dict(
                    data.get(
                        "metadata",
                        {},
                    )
                ),
                confidence=float(
                    data.get(
                        "confidence",
                        1.0,
                    )
                ),
                expires_at=data.get(
                    "expires_at"
                ),
                pinned=bool(
                    data.get(
                        "pinned",
                        False,
                    )
                ),
                sensitive=bool(
                    data.get(
                        "sensitive",
                        False,
                    )
                ),
                embedding_id=data.get(
                    "embedding_id"
                ),
            )

        except KeyError as exc:

            raise MemorySerializationError(
                f"Missing required memory field: {exc}"
            ) from exc

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise MemorySerializationError(
                "Invalid memory data."
            ) from exc

    # ======================================================================
    # NORMALIZATION
    # ======================================================================

    @staticmethod
    def _normalize_memory_type(
        value: MemoryType | str,
    ) -> MemoryType:

        if isinstance(
            value,
            MemoryType,
        ):
            return value

        try:
            return MemoryType(
                str(value).lower()
            )

        except ValueError:

            return MemoryType.LONG_TERM

    @staticmethod
    def _normalize_importance(
        value: MemoryImportance | str,
    ) -> MemoryImportance:

        if isinstance(
            value,
            MemoryImportance,
        ):
            return value

        try:
            return MemoryImportance(
                str(value).lower()
            )

        except ValueError:

            return MemoryImportance.NORMAL

    @staticmethod
    def _normalize_status(
        value: MemoryStatus | str,
    ) -> MemoryStatus:

        if isinstance(
            value,
            MemoryStatus,
        ):
            return value

        try:
            return MemoryStatus(
                str(value).lower()
            )

        except ValueError:

            return MemoryStatus.ACTIVE

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:

        cleaned = (
            str(text)
            .lower()
            .replace(
                ",",
                " ",
            )
            .replace(
                ".",
                " ",
            )
            .replace(
                "!",
                " ",
            )
            .replace(
                "?",
                " ",
            )
            .replace(
                ":",
                " ",
            )
            .replace(
                ";",
                " ",
            )
            .replace(
                "/",
                " ",
            )
            .replace(
                "\\",
                " ",
            )
            .replace(
                "-",
                " ",
            )
            .replace(
                "_",
                " ",
            )
        )

        return {
            word
            for word in cleaned.split()
            if len(word) >= 2
        }

    # ======================================================================
    # STORAGE HELPERS
    # ======================================================================

    def _ensure_storage_directory(
        self,
    ) -> None:

        if self.storage_path is None:
            return

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ======================================================================
    # HEALTH
    # ======================================================================

    def health(self) -> Dict[str, Any]:
        """
        Return storage health information.
        """

        with self._lock:

            storage_exists = (
                self.storage_path.exists()
                if self.storage_path
                else False
            )

            return {
                "status": "healthy",
                "loaded": self._loaded,
                "dirty": self._dirty,
                "memory_count": len(
                    self._memories
                ),
                "persistent": (
                    self.storage_path
                    is not None
                ),
                "storage_path": (
                    str(
                        self.storage_path
                    )
                    if self.storage_path
                    else None
                ),
                "storage_exists": storage_exists,
                "auto_save": self.auto_save,
            }


__all__ = [
    "MemoryStore",
    "MemoryStoreError",
    "MemoryNotFoundError",
    "MemorySerializationError",
]


