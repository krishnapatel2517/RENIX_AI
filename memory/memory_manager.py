"""
RENIX Memory Manager
====================

Central coordinator for the RENIX memory subsystem.

Responsibilities
----------------
- Store memories
- Retrieve memories
- Search memories
- Update memories
- Delete memories
- Forget memories
- Categorize memories
- Manage memory importance
- Manage memory timestamps
- Maintain short-term and long-term context
- Coordinate specialized memory stores
- Provide a stable interface for the rest of RENIX
- Enforce basic privacy controls
- Provide memory statistics
- Support automatic cleanup

The Memory Manager is intentionally designed to work even when the
specialized memory modules are not yet initialized. This allows RENIX
to start safely while the rest of the memory subsystem is being built.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence
from datetime import datetime, timezone
import json
import logging
import os
import threading
import uuid


logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class MemoryType(str, Enum):
    """Supported RENIX memory categories."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PROJECT = "project"
    TASK = "task"
    PREFERENCE = "preference"
    CONVERSATION = "conversation"


class MemoryImportance(str, Enum):
    """Importance levels for stored memories."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryStatus(str, Enum):
    """Lifecycle state of a memory."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class Memory:
    """
    Represents one RENIX memory.

    A Memory object is intentionally self-contained so that it can later
    be serialized into JSON, a database, or a vector store.
    """

    memory_id: str

    content: str

    memory_type: MemoryType = MemoryType.LONG_TERM

    importance: MemoryImportance = MemoryImportance.NORMAL

    status: MemoryStatus = MemoryStatus.ACTIVE

    created_at: str = ""

    updated_at: str = ""

    accessed_at: str = ""

    access_count: int = 0

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    conversation_id: Optional[str] = None

    project_id: Optional[str] = None

    task_id: Optional[str] = None

    source: Optional[str] = None

    tags: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    confidence: float = 1.0

    expires_at: Optional[str] = None

    pinned: bool = False

    sensitive: bool = False

    embedding_id: Optional[str] = None


@dataclass
class MemoryQuery:
    """Parameters used when searching or retrieving memories."""

    text: Optional[str] = None

    memory_type: Optional[MemoryType] = None

    importance: Optional[MemoryImportance] = None

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    conversation_id: Optional[str] = None

    project_id: Optional[str] = None

    task_id: Optional[str] = None

    tags: List[str] = field(default_factory=list)

    limit: int = 20

    include_archived: bool = False

    include_deleted: bool = False

    include_sensitive: bool = False


@dataclass
class MemoryResult:
    """Result returned from a memory search."""

    memory: Memory

    score: float = 0.0

    matched_fields: List[str] = field(default_factory=list)


@dataclass
class MemoryStats:
    """Statistics describing the current memory subsystem."""

    total: int = 0

    active: int = 0

    archived: int = 0

    deleted: int = 0

    pinned: int = 0

    sensitive: int = 0

    by_type: Dict[str, int] = field(default_factory=dict)

    by_importance: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# MEMORY MANAGER
# ============================================================================


class MemoryManager:
    """
    Central memory coordinator for RENIX.

    The manager provides a stable API for all RENIX components.

    Other components should communicate with memory through this class
    rather than directly manipulating internal memory collections.
    """

    def __init__(
        self,
        *,
        storage_path: Optional[str] = None,
        auto_persist: bool = True,
        default_memory_type: MemoryType = MemoryType.LONG_TERM,
        max_short_term_memories: int = 50,
        max_memory_results: int = 100,
    ) -> None:

        self.storage_path = storage_path

        self.auto_persist = auto_persist

        self.default_memory_type = default_memory_type

        self.max_short_term_memories = max(
            1,
            int(max_short_term_memories),
        )

        self.max_memory_results = max(
            1,
            int(max_memory_results),
        )

        self._memories: Dict[str, Memory] = {}

        self._lock = threading.RLock()

        self._initialized = False

        self._created_count = 0

        self._updated_count = 0

        self._deleted_count = 0

        self._access_count = 0

        self._initialize()

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize(self) -> None:
        """Initialize the memory manager."""

        with self._lock:

            if self.storage_path:
                directory = os.path.dirname(
                    os.path.abspath(self.storage_path)
                )

                if directory:
                    os.makedirs(
                        directory,
                        exist_ok=True,
                    )

                self._load()

            self._initialized = True

        logger.info("RENIX Memory Manager initialized.")

    @property
    def initialized(self) -> bool:
        """Return whether the memory manager has initialized."""

        return self._initialized

    # ========================================================================
    # STORE
    # ========================================================================

    def store(
        self,
        content: str,
        *,
        memory_type: Optional[MemoryType | str] = None,
        importance: MemoryImportance | str = MemoryImportance.NORMAL,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        expires_at: Optional[str] = None,
        pinned: bool = False,
        sensitive: bool = False,
        memory_id: Optional[str] = None,
    ) -> Memory:
        """
        Store a new memory.

        Returns:
            The created Memory object.
        """

        if not isinstance(content, str):
            content = str(content)

        content = content.strip()

        if not content:
            raise ValueError(
                "Memory content cannot be empty."
            )

        resolved_type = self._normalize_memory_type(
            memory_type or self.default_memory_type
        )

        resolved_importance = self._normalize_importance(
            importance
        )

        confidence = self._clamp(
            confidence,
            0.0,
            1.0,
        )

        now = self._now()

        memory = Memory(
            memory_id=memory_id or self._generate_id(),
            content=content,
            memory_type=resolved_type,
            importance=resolved_importance,
            status=MemoryStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            accessed_at=now,
            access_count=0,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            project_id=project_id,
            task_id=task_id,
            source=source,
            tags=self._normalize_tags(tags),
            metadata=dict(metadata or {}),
            confidence=confidence,
            expires_at=expires_at,
            pinned=pinned,
            sensitive=sensitive,
        )

        with self._lock:
            self._memories[memory.memory_id] = memory

            self._created_count += 1

            self._enforce_short_term_limit()

            if self.auto_persist:
                self._persist()

        logger.debug(
            "Stored RENIX memory: %s",
            memory.memory_id,
        )

        return memory

    # ========================================================================
    # RETRIEVE
    # ========================================================================

    def get(
        self,
        memory_id: str,
        *,
        include_deleted: bool = False,
        include_sensitive: bool = False,
    ) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        """

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return None

            if (
                memory.status == MemoryStatus.DELETED
                and not include_deleted
            ):
                return None

            if (
                memory.sensitive
                and not include_sensitive
            ):
                return None

            self._mark_accessed(memory)

            return memory

    def get_many(
        self,
        memory_ids: Sequence[str],
        *,
        include_deleted: bool = False,
        include_sensitive: bool = False,
    ) -> List[Memory]:
        """
        Retrieve multiple memories.
        """

        result: List[Memory] = []

        for memory_id in memory_ids:

            memory = self.get(
                memory_id,
                include_deleted=include_deleted,
                include_sensitive=include_sensitive,
            )

            if memory is not None:
                result.append(memory)

        return result

    # ========================================================================
    # SEARCH
    # ========================================================================

    def search(
        self,
        query: MemoryQuery | str,
        *,
        limit: Optional[int] = None,
    ) -> List[MemoryResult]:
        """
        Search stored memories.

        This implementation provides a lightweight local search engine.
        A future vector_store/retrieval module can extend this behavior
        without changing the public MemoryManager API.
        """

        if isinstance(query, str):
            query = MemoryQuery(
                text=query
            )

        requested_limit = (
            limit
            if limit is not None
            else query.limit
        )

        requested_limit = max(
            1,
            min(
                requested_limit,
                self.max_memory_results,
            ),
        )

        results: List[MemoryResult] = []

        with self._lock:

            for memory in self._memories.values():

                if not self._matches_filters(
                    memory,
                    query,
                ):
                    continue

                score, fields = self._score_memory(
                    memory,
                    query,
                )

                if query.text and score <= 0:
                    continue

                results.append(
                    MemoryResult(
                        memory=memory,
                        score=score,
                        matched_fields=fields,
                    )
                )

            results.sort(
                key=lambda result: (
                    result.score,
                    self._importance_value(
                        result.memory.importance
                    ),
                    result.memory.access_count,
                    result.memory.updated_at,
                ),
                reverse=True,
            )

            results = results[:requested_limit]

            for result in results:
                self._mark_accessed(
                    result.memory
                )

        return results

    def search_text(
        self,
        text: str,
        *,
        limit: int = 20,
    ) -> List[MemoryResult]:
        """
        Convenience wrapper for text search.
        """

        return self.search(
            MemoryQuery(
                text=text,
                limit=limit,
            )
        )

    # ========================================================================
    # UPDATE
    # ========================================================================

    def update(
        self,
        memory_id: str,
        *,
        content: Optional[str] = None,
        memory_type: Optional[MemoryType | str] = None,
        importance: Optional[MemoryImportance | str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        expires_at: Optional[str] = None,
        pinned: Optional[bool] = None,
        sensitive: Optional[bool] = None,
        source: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Optional[Memory]:
        """
        Update an existing memory.

        Only supplied fields are changed.
        """

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return None

            if memory.status == MemoryStatus.DELETED:
                return None

            if content is not None:
                content = str(content).strip()

                if not content:
                    raise ValueError(
                        "Memory content cannot be empty."
                    )

                memory.content = content

            if memory_type is not None:
                memory.memory_type = (
                    self._normalize_memory_type(
                        memory_type
                    )
                )

            if importance is not None:
                memory.importance = (
                    self._normalize_importance(
                        importance
                    )
                )

            if tags is not None:
                memory.tags = self._normalize_tags(
                    tags
                )

            if metadata is not None:
                memory.metadata = dict(
                    metadata
                )

            if confidence is not None:
                memory.confidence = self._clamp(
                    confidence,
                    0.0,
                    1.0,
                )

            if expires_at is not None:
                memory.expires_at = expires_at

            if pinned is not None:
                memory.pinned = bool(
                    pinned
                )

            if sensitive is not None:
                memory.sensitive = bool(
                    sensitive
                )

            if source is not None:
                memory.source = source

            if project_id is not None:
                memory.project_id = project_id

            if task_id is not None:
                memory.task_id = task_id

            memory.updated_at = self._now()

            self._updated_count += 1

            if self.auto_persist:
                self._persist()

            return memory

    # ========================================================================
    # DELETE / FORGET
    # ========================================================================

    def delete(
        self,
        memory_id: str,
        *,
        permanent: bool = False,
    ) -> bool:
        """
        Delete a memory.

        By default deletion is soft deletion.

        permanent=True physically removes it from the in-memory store.
        """

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return False

            if permanent:
                del self._memories[
                    memory_id
                ]
            else:
                memory.status = MemoryStatus.DELETED
                memory.updated_at = self._now()

            self._deleted_count += 1

            if self.auto_persist:
                self._persist()

            return True

    def forget(
        self,
        memory_id: str,
    ) -> bool:
        """
        Permanently forget a memory.
        """

        return self.delete(
            memory_id,
            permanent=True,
        )

    def forget_by_query(
        self,
        query: MemoryQuery | str,
        *,
        permanent: bool = False,
    ) -> int:
        """
        Delete memories matching a query.

        Returns:
            Number of memories removed/marked deleted.
        """

        results = self.search(
            query
        )

        count = 0

        for result in results:

            if self.delete(
                result.memory.memory_id,
                permanent=permanent,
            ):
                count += 1

        return count

    # ========================================================================
    # ARCHIVE / RESTORE
    # ========================================================================

    def archive(
        self,
        memory_id: str,
    ) -> bool:
        """
        Archive a memory without deleting it.
        """

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return False

            if memory.status == MemoryStatus.DELETED:
                return False

            memory.status = MemoryStatus.ARCHIVED
            memory.updated_at = self._now()

            if self.auto_persist:
                self._persist()

            return True

    def restore(
        self,
        memory_id: str,
    ) -> bool:
        """
        Restore an archived or soft-deleted memory.
        """

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return False

            memory.status = MemoryStatus.ACTIVE
            memory.updated_at = self._now()

            if self.auto_persist:
                self._persist()

            return True

    # ========================================================================
    # PINNING
    # ========================================================================

    def pin(
        self,
        memory_id: str,
    ) -> bool:
        """Pin a memory so cleanup operations preserve it."""

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return False

            memory.pinned = True
            memory.updated_at = self._now()

            if self.auto_persist:
                self._persist()

            return True

    def unpin(
        self,
        memory_id: str,
    ) -> bool:
        """Remove the pinned status from a memory."""

        with self._lock:

            memory = self._memories.get(
                memory_id
            )

            if memory is None:
                return False

            memory.pinned = False
            memory.updated_at = self._now()

            if self.auto_persist:
                self._persist()

            return True

    # ========================================================================
    # CONTEXT
    # ========================================================================

    def get_context(
        self,
        *,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Retrieve useful contextual memories.

        Priority is given to:
            1. Pinned memories
            2. Relevant memories
            3. High importance
            4. Recently accessed memories
        """

        memory_query = MemoryQuery(
            text=query,
            session_id=session_id,
            project_id=project_id,
            task_id=task_id,
            limit=limit,
        )

        results = self.search(
            memory_query,
            limit=limit,
        )

        return [
            result.memory
            for result in results
        ]

    def build_context_text(
        self,
        *,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """
        Build a compact textual context from relevant memories.

        Useful for passing memory context to an AI model.
        """

        memories = self.get_context(
            query=query,
            limit=limit,
        )

        if not memories:
            return ""

        lines: List[str] = []

        for memory in memories:

            lines.append(
                f"- [{memory.memory_type.value}] "
                f"{memory.content}"
            )

        return "\n".join(lines)

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def stats(self) -> MemoryStats:
        """Return current memory statistics."""

        with self._lock:

            stats = MemoryStats()

            stats.total = len(
                self._memories
            )

            for memory in self._memories.values():

                if memory.status == MemoryStatus.ACTIVE:
                    stats.active += 1

                elif memory.status == MemoryStatus.ARCHIVED:
                    stats.archived += 1

                elif memory.status == MemoryStatus.DELETED:
                    stats.deleted += 1

                if memory.pinned:
                    stats.pinned += 1

                if memory.sensitive:
                    stats.sensitive += 1

                memory_type = (
                    memory.memory_type.value
                )

                stats.by_type[memory_type] = (
                    stats.by_type.get(
                        memory_type,
                        0,
                    )
                    + 1
                )

                importance = (
                    memory.importance.value
                )

                stats.by_importance[importance] = (
                    stats.by_importance.get(
                        importance,
                        0,
                    )
                    + 1
                )

            return stats

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def cleanup(
        self,
        *,
        remove_deleted: bool = False,
        remove_expired: bool = True,
    ) -> int:
        """
        Perform basic memory cleanup.

        Pinned memories are never automatically removed.
        """

        removed = 0

        now = self._now()

        with self._lock:

            memory_ids = list(
                self._memories.keys()
            )

            for memory_id in memory_ids:

                memory = self._memories[
                    memory_id
                ]

                if memory.pinned:
                    continue

                if (
                    remove_deleted
                    and memory.status
                    == MemoryStatus.DELETED
                ):
                    del self._memories[
                        memory_id
                    ]

                    removed += 1
                    continue

                if (
                    remove_expired
                    and self._is_expired(
                        memory,
                        now,
                    )
                ):
                    del self._memories[
                        memory_id
                    ]

                    removed += 1

            if removed and self.auto_persist:
                self._persist()

        return removed

    # ========================================================================
    # EXPORT / IMPORT
    # ========================================================================

    def export_memories(
        self,
        *,
        include_deleted: bool = False,
        include_sensitive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Export memories into serializable dictionaries.
        """

        exported: List[Dict[str, Any]] = []

        with self._lock:

            for memory in self._memories.values():

                if (
                    memory.status == MemoryStatus.DELETED
                    and not include_deleted
                ):
                    continue

                if (
                    memory.sensitive
                    and not include_sensitive
                ):
                    continue

                exported.append(
                    self._serialize_memory(
                        memory
                    )
                )

        return exported

    def import_memories(
        self,
        memories: Sequence[Dict[str, Any]],
        *,
        replace_existing: bool = False,
    ) -> int:
        """
        Import serialized memories.

        Returns:
            Number of memories imported.
        """

        imported = 0

        with self._lock:

            for data in memories:

                try:
                    memory = self._deserialize_memory(
                        data
                    )

                    if (
                        memory.memory_id in self._memories
                        and not replace_existing
                    ):
                        continue

                    self._memories[
                        memory.memory_id
                    ] = memory

                    imported += 1

                except Exception as exc:
                    logger.warning(
                        "Failed to import memory: %s",
                        exc,
                    )

            if imported and self.auto_persist:
                self._persist()

        return imported

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def save(self) -> None:
        """
        Persist all current memories to disk.
        """

        with self._lock:
            self._persist()

    def load(self) -> None:
        """
        Reload memories from disk.
        """

        with self._lock:
            self._load()

    def _persist(self) -> None:
        """Internal persistence implementation."""

        if not self.storage_path:
            return

        try:

            directory = os.path.dirname(
                os.path.abspath(
                    self.storage_path
                )
            )

            os.makedirs(
                directory,
                exist_ok=True,
            )

            payload = {
                "version": 1,
                "saved_at": self._now(),
                "memories": self.export_memories(
                    include_deleted=True,
                    include_sensitive=True,
                ),
            }

            temporary_path = (
                f"{self.storage_path}.tmp"
            )

            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    payload,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            os.replace(
                temporary_path,
                self.storage_path,
            )

        except Exception:
            logger.exception(
                "Failed to persist RENIX memories."
            )

    def _load(self) -> None:
        """Internal persistence loader."""

        if not self.storage_path:
            return

        if not os.path.exists(
            self.storage_path
        ):
            return

        try:

            with open(
                self.storage_path,
                "r",
                encoding="utf-8",
            ) as file:

                payload = json.load(
                    file
                )

            memories = payload.get(
                "memories",
                [],
            )

            self._memories.clear()

            for data in memories:

                try:

                    memory = (
                        self._deserialize_memory(
                            data
                        )
                    )

                    self._memories[
                        memory.memory_id
                    ] = memory

                except Exception as exc:
                    logger.warning(
                        "Skipping invalid memory: %s",
                        exc,
                    )

            logger.info(
                "Loaded %d RENIX memories.",
                len(self._memories),
            )

        except Exception:
            logger.exception(
                "Failed to load RENIX memory store."
            )

    # ========================================================================
    # FILTERING / SCORING
    # ========================================================================

    def _matches_filters(
        self,
        memory: Memory,
        query: MemoryQuery,
    ) -> bool:

        if (
            memory.status == MemoryStatus.DELETED
            and not query.include_deleted
        ):
            return False

        if (
            memory.status == MemoryStatus.ARCHIVED
            and not query.include_archived
        ):
            return False

        if (
            memory.sensitive
            and not query.include_sensitive
        ):
            return False

        if (
            query.memory_type is not None
            and memory.memory_type
            != self._normalize_memory_type(
                query.memory_type
            )
        ):
            return False

        if (
            query.importance is not None
            and memory.importance
            != self._normalize_importance(
                query.importance
            )
        ):
            return False

        if (
            query.user_id is not None
            and memory.user_id
            != query.user_id
        ):
            return False

        if (
            query.session_id is not None
            and memory.session_id
            != query.session_id
        ):
            return False

        if (
            query.conversation_id is not None
            and memory.conversation_id
            != query.conversation_id
        ):
            return False

        if (
            query.project_id is not None
            and memory.project_id
            != query.project_id
        ):
            return False

        if (
            query.task_id is not None
            and memory.task_id
            != query.task_id
        ):
            return False

        if query.tags:

            memory_tags = {
                tag.lower()
                for tag in memory.tags
            }

            required_tags = {
                tag.lower()
                for tag in query.tags
            }

            if not required_tags.issubset(
                memory_tags
            ):
                return False

        return True

    def _score_memory(
        self,
        memory: Memory,
        query: MemoryQuery,
    ) -> tuple[float, List[str]]:
        """
        Calculate lightweight lexical relevance.

        This is intentionally simple because semantic/vector retrieval
        belongs to the dedicated retrieval and vector-store modules.
        """

        if not query.text:
            return (
                self._base_memory_score(
                    memory
                ),
                [],
            )

        search_text = query.text.lower().strip()

        content = memory.content.lower()

        metadata_text = " ".join(
            [
                " ".join(memory.tags),
                memory.source or "",
                memory.project_id or "",
                memory.task_id or "",
            ]
        ).lower()

        score = 0.0

        matched_fields: List[str] = []

        if search_text in content:
            score += 0.80
            matched_fields.append(
                "content_exact"
            )

        query_words = self._tokenize(
            search_text
        )

        content_words = self._tokenize(
            content
        )

        if query_words and content_words:

            overlap = (
                query_words
                & content_words
            )

            ratio = (
                len(overlap)
                / len(query_words)
            )

            score += ratio * 0.60

            if overlap:
                matched_fields.append(
                    "content_terms"
                )

        metadata_words = self._tokenize(
            metadata_text
        )

        if query_words & metadata_words:
            score += 0.20
            matched_fields.append(
                "metadata"
            )

        score += (
            memory.confidence
            * 0.10
        )

        score += (
            self._importance_value(
                memory.importance
            )
            * 0.10
        )

        score = self._clamp(
            score,
            0.0,
            1.0,
        )

        return (
            score,
            matched_fields,
        )

    def _base_memory_score(
        self,
        memory: Memory,
    ) -> float:

        score = (
            memory.confidence
            * 0.60
        )

        score += (
            self._importance_value(
                memory.importance
            )
            * 0.25
        )

        if memory.pinned:
            score += 0.15

        return self._clamp(
            score,
            0.0,
            1.0,
        )

    # ========================================================================
    # SHORT-TERM LIMIT
    # ========================================================================

    def _enforce_short_term_limit(
        self,
    ) -> None:
        """
        Keep short-term memory bounded.

        Pinned memories are preserved.
        """

        short_term = [
            memory
            for memory in self._memories.values()
            if (
                memory.memory_type
                == MemoryType.SHORT_TERM
                and memory.status
                == MemoryStatus.ACTIVE
            )
        ]

        if len(short_term) <= self.max_short_term_memories:
            return

        short_term.sort(
            key=lambda memory: (
                memory.pinned,
                self._importance_value(
                    memory.importance
                ),
                memory.updated_at,
            )
        )

        excess = (
            len(short_term)
            - self.max_short_term_memories
        )

        removed = 0

        for memory in short_term:

            if removed >= excess:
                break

            if memory.pinned:
                continue

            self._memories.pop(
                memory.memory_id,
                None,
            )

            removed += 1

    # ========================================================================
    # ACCESS / EXPIRATION
    # ========================================================================

    def _mark_accessed(
        self,
        memory: Memory,
    ) -> None:

        memory.accessed_at = self._now()

        memory.access_count += 1

        self._access_count += 1

    def _is_expired(
        self,
        memory: Memory,
        now: str,
    ) -> bool:

        if not memory.expires_at:
            return False

        try:

            expiry = datetime.fromisoformat(
                memory.expires_at
            )

            current = datetime.fromisoformat(
                now
            )

            return current >= expiry

        except ValueError:
            return False

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    @staticmethod
    def _serialize_memory(
        memory: Memory,
    ) -> Dict[str, Any]:

        data = asdict(
            memory
        )

        data["memory_type"] = (
            memory.memory_type.value
        )

        data["importance"] = (
            memory.importance.value
        )

        data["status"] = (
            memory.status.value
        )

        return data

    @staticmethod
    def _deserialize_memory(
        data: Dict[str, Any],
    ) -> Memory:

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
            memory_type=MemoryManager._normalize_memory_type(
                data.get(
                    "memory_type",
                    MemoryType.LONG_TERM.value,
                )
            ),
            importance=MemoryManager._normalize_importance(
                data.get(
                    "importance",
                    MemoryImportance.NORMAL.value,
                )
            ),
            status=MemoryManager._normalize_status(
                data.get(
                    "status",
                    MemoryStatus.ACTIVE.value,
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

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

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
    def _normalize_tags(
        tags: Optional[Iterable[str]],
    ) -> List[str]:

        if not tags:
            return []

        result: List[str] = []

        for tag in tags:

            normalized = str(
                tag
            ).strip()

            if normalized and normalized not in result:
                result.append(
                    normalized
                )

        return result

    @staticmethod
    def _importance_value(
        importance: MemoryImportance,
    ) -> float:

        values = {
            MemoryImportance.LOW: 0.25,
            MemoryImportance.NORMAL: 0.50,
            MemoryImportance.HIGH: 0.75,
            MemoryImportance.CRITICAL: 1.00,
        }

        return values.get(
            importance,
            0.50,
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:

        if not text:
            return set()

        cleaned = (
            text.lower()
            .replace(",", " ")
            .replace(".", " ")
            .replace("!", " ")
            .replace("?", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("/", " ")
            .replace("\\", " ")
            .replace("_", " ")
            .replace("-", " ")
        )

        return {
            word.strip()
            for word in cleaned.split()
            if len(word.strip()) >= 2
        }

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return minimum

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    @staticmethod
    def _now() -> str:
        """Return a UTC ISO-8601 timestamp."""

        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique memory identifier."""

        return (
            f"memory_{uuid.uuid4().hex}"
        )

    # ========================================================================
    # DEBUG / HEALTH
    # ========================================================================

    def health(self) -> Dict[str, Any]:
        """
        Return a health snapshot for the memory subsystem.
        """

        statistics = self.stats()

        return {
            "status": "healthy"
            if self.initialized
            else "not_initialized",
            "initialized": self.initialized,
            "storage_enabled": bool(
                self.storage_path
            ),
            "auto_persist": self.auto_persist,
            "memory_count": statistics.total,
            "active_memories": statistics.active,
            "archived_memories": statistics.archived,
            "deleted_memories": statistics.deleted,
            "created_count": self._created_count,
            "updated_count": self._updated_count,
            "deleted_count": self._deleted_count,
            "access_count": self._access_count,
        }


# ============================================================================
# SHARED MANAGER
# ============================================================================


_default_memory_manager: Optional[
    MemoryManager
] = None

_default_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """
    Return the shared RENIX MemoryManager instance.

    The manager is initialized lazily.
    """

    global _default_memory_manager

    if _default_memory_manager is None:

        with _default_manager_lock:

            if _default_memory_manager is None:

                _default_memory_manager = (
                    MemoryManager()
                )

    return _default_memory_manager


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def remember(
    content: str,
    **kwargs: Any,
) -> Memory:
    """
    Convenience function for storing a memory.
    """

    return get_memory_manager().store(
        content,
        **kwargs,
    )


def recall(
    memory_id: str,
    **kwargs: Any,
) -> Optional[Memory]:
    """
    Convenience function for retrieving a memory.
    """

    return get_memory_manager().get(
        memory_id,
        **kwargs,
    )


def search_memory(
    query: MemoryQuery | str,
    **kwargs: Any,
) -> List[MemoryResult]:
    """
    Convenience function for searching memories.
    """

    return get_memory_manager().search(
        query,
        **kwargs,
    )


def forget(
    memory_id: str,
) -> bool:
    """
    Convenience function for permanently forgetting a memory.
    """

    return get_memory_manager().forget(
        memory_id
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================


__all__ = [
    "MemoryManager",
    "Memory",
    "MemoryQuery",
    "MemoryResult",
    "MemoryStats",
    "MemoryType",
    "MemoryImportance",
    "MemoryStatus",
    "get_memory_manager",
    "remember",
    "recall",
    "search_memory",
    "forget",
]


