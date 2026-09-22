"""
RENIX Short-Term Memory
=======================

Manages temporary conversational/task context that should remain available
during the current interaction or recent session, without becoming permanent
long-term memory automatically.

Responsibilities
----------------
- Store recent memories
- Maintain recency ordering
- Enforce capacity limits
- Track TTL/expiration
- Promote important memories to persistent storage
- Retrieve recent context
- Remove expired entries
- Provide conversation/task context snapshots

This module intentionally does NOT implement:
- vector embeddings
- semantic similarity
- long-term memory policy
- privacy policy
- LLM reasoning

Those responsibilities belong to other RENIX memory components.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Iterable, List, Optional
import logging
import time
import uuid

from .memory_manager import (
    Memory,
    MemoryImportance,
    MemoryStatus,
    MemoryType,
)
from .memory_store import (
    MemoryNotFoundError,
    MemoryStore,
)


logger = logging.getLogger(__name__)


class ShortTermMemoryError(Exception):
    """Base exception for short-term memory failures."""


@dataclass
class ShortTermEntry:
    """
    Runtime metadata associated with a short-term memory.
    """

    memory_id: str
    created_monotonic: float
    expires_monotonic: Optional[float] = None
    access_count: int = 0
    last_access_monotonic: float = 0.0


class ShortTermMemory:
    """
    RENIX short-term memory manager.

    Short-term memory is optimized for information such as:

    - current conversation
    - recent user instructions
    - active task context
    - recent decisions
    - temporary working context
    - immediate references

    Parameters
    ----------
    store:
        Optional persistent MemoryStore.

    capacity:
        Maximum number of active short-term memories.

    default_ttl:
        Default lifetime in seconds.

    auto_promote:
        Whether high-value memories should be promoted to the persistent
        MemoryStore when evicted.
    """

    DEFAULT_CAPACITY = 100

    DEFAULT_TTL = 60 * 60

    PROMOTION_IMPORTANCE = {
        MemoryImportance.HIGH,
        MemoryImportance.CRITICAL,
    }

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        *,
        capacity: int = DEFAULT_CAPACITY,
        default_ttl: float = DEFAULT_TTL,
        auto_promote: bool = True,
    ) -> None:

        if capacity <= 0:
            raise ValueError(
                "Short-term memory capacity must be greater than zero."
            )

        if default_ttl <= 0:
            raise ValueError(
                "Short-term memory TTL must be greater than zero."
            )

        self.store = store

        self.capacity = int(capacity)

        self.default_ttl = float(
            default_ttl
        )

        self.auto_promote = bool(
            auto_promote
        )

        self._entries: dict[
            str,
            ShortTermEntry,
        ] = {}

        self._lock = RLock()

    # ======================================================================
    # BASIC PROPERTIES
    # ======================================================================

    @property
    def count(self) -> int:
        """Return the number of active short-term memories."""

        with self._lock:
            self._remove_expired_locked()

            return len(
                self._entries
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

    # ======================================================================
    # ADD
    # ======================================================================

    def add(
        self,
        memory: Memory,
        *,
        ttl: Optional[float] = None,
        promote: bool = False,
    ) -> Memory:
        """
        Add a Memory object to short-term memory.

        The memory itself is kept in MemoryStore when a store is attached.
        Short-term memory maintains the active/recency index.
        """

        if not isinstance(
            memory,
            Memory,
        ):
            raise TypeError(
                "Short-term memory accepts Memory objects."
            )

        lifetime = (
            self.default_ttl
            if ttl is None
            else float(ttl)
        )

        if lifetime <= 0:
            raise ValueError(
                "TTL must be greater than zero."
            )

        now = time.monotonic()

        with self._lock:

            self._remove_expired_locked()

            if self.store is not None:

                if self.store.exists(
                    memory.memory_id
                ):
                    self.store.update(
                        memory
                    )
                else:
                    self.store.add(
                        memory
                    )

            self._entries[
                memory.memory_id
            ] = ShortTermEntry(
                memory_id=memory.memory_id,
                created_monotonic=now,
                expires_monotonic=(
                    now + lifetime
                ),
                access_count=0,
                last_access_monotonic=now,
            )

            self._enforce_capacity_locked()

            if promote:
                self._promote_locked(
                    memory
                )

        return memory

    def add_text(
        self,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: MemoryImportance = MemoryImportance.NORMAL,
        ttl: Optional[float] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        confidence: float = 1.0,
        pinned: bool = False,
        sensitive: bool = False,
    ) -> Memory:
        """
        Convenience method for creating short-term memories from text.
        """

        now = self._utc_now()

        memory = Memory(
            memory_id=self._new_id(),
            content=str(content),
            memory_type=memory_type,
            importance=importance,
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
            tags=list(
                tags or []
            ),
            metadata=dict(
                metadata or {}
            ),
            confidence=float(
                confidence
            ),
            expires_at=None,
            pinned=bool(
                pinned
            ),
            sensitive=bool(
                sensitive
            ),
            embedding_id=None,
        )

        return self.add(
            memory,
            ttl=ttl,
        )

    # ======================================================================
    # READ
    # ======================================================================

    def get(
        self,
        memory_id: str,
    ) -> Optional[Memory]:
        """
        Retrieve a short-term memory and update its access metadata.
        """

        with self._lock:

            self._remove_expired_locked()

            entry = self._entries.get(
                memory_id
            )

            if entry is None:
                return None

            memory = self._get_memory_locked(
                memory_id
            )

            if memory is None:
                self._entries.pop(
                    memory_id,
                    None,
                )
                return None

            self._touch_locked(
                memory,
                entry,
            )

            return memory

    def require(
        self,
        memory_id: str,
    ) -> Memory:
        """Retrieve a memory or raise MemoryNotFoundError."""

        memory = self.get(
            memory_id
        )

        if memory is None:
            raise MemoryNotFoundError(
                f"Short-term memory not found: "
                f"{memory_id}"
            )

        return memory

    def exists(
        self,
        memory_id: str,
    ) -> bool:
        """Return whether a memory currently exists in short-term memory."""

        with self._lock:

            self._remove_expired_locked()

            if memory_id not in self._entries:
                return False

            memory = self._get_memory_locked(
                memory_id
            )

            if memory is None:
                self._entries.pop(
                    memory_id,
                    None,
                )
                return False

            return True

    # ======================================================================
    # RECENT CONTEXT
    # ======================================================================

    def recent(
        self,
        *,
        limit: int = 20,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> List[Memory]:
        """
        Return the most recently accessed short-term memories.
        """

        limit = max(
            1,
            int(limit),
        )

        with self._lock:

            self._remove_expired_locked()

            memories: list[
                tuple[
                    float,
                    Memory,
                ]
            ] = []

            for entry in self._entries.values():

                memory = self._get_memory_locked(
                    entry.memory_id
                )

                if memory is None:
                    continue

                if (
                    not include_deleted
                    and memory.status
                    == MemoryStatus.DELETED
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

                memories.append(
                    (
                        entry.last_access_monotonic,
                        memory,
                    )
                )

            memories.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            return [
                memory
                for _, memory
                in memories[:limit]
            ]

    def context(
        self,
        *,
        limit: int = 20,
        separator: str = "\n",
        include_metadata: bool = False,
    ) -> str:
        """
        Build a compact textual context representation suitable for
        feeding into an AI reasoning layer.
        """

        memories = self.recent(
            limit=limit
        )

        lines: list[str] = []

        for memory in reversed(
            memories
        ):

            if include_metadata:

                lines.append(
                    (
                        f"[{memory.memory_type.value}] "
                        f"[{memory.importance.value}] "
                        f"{memory.content}"
                    )
                )

            else:

                lines.append(
                    memory.content
                )

        return separator.join(
            lines
        )

    # ======================================================================
    # SEARCH
    # ======================================================================

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> List[Memory]:
        """
        Search the active short-term memory using the MemoryStore's
        lightweight lexical search when available.
        """

        with self._lock:

            self._remove_expired_locked()

            active_ids = set(
                self._entries.keys()
            )

            if self.store is not None:

                results = self.store.text_search(
                    query,
                    limit=max(
                        limit * 2,
                        limit,
                    ),
                )

                return [
                    memory
                    for memory in results
                    if memory.memory_id
                    in active_ids
                ][:max(1, limit)]

            return self._local_search(
                query,
                limit=limit,
            )

    # ======================================================================
    # TOUCH / ACCESS
    # ======================================================================

    def touch(
        self,
        memory_id: str,
    ) -> Optional[Memory]:
        """
        Mark a memory as recently accessed.
        """

        with self._lock:

            self._remove_expired_locked()

            entry = self._entries.get(
                memory_id
            )

            if entry is None:
                return None

            memory = self._get_memory_locked(
                memory_id
            )

            if memory is None:
                return None

            self._touch_locked(
                memory,
                entry,
            )

            return memory

    # ======================================================================
    # TTL
    # ======================================================================

    def extend(
        self,
        memory_id: str,
        seconds: float,
    ) -> bool:
        """
        Extend the TTL of a short-term memory.
        """

        if seconds <= 0:
            raise ValueError(
                "Extension must be greater than zero."
            )

        with self._lock:

            self._remove_expired_locked()

            entry = self._entries.get(
                memory_id
            )

            if entry is None:
                return False

            now = time.monotonic()

            if (
                entry.expires_monotonic is None
            ):
                entry.expires_monotonic = (
                    now + seconds
                )
            else:
                entry.expires_monotonic += (
                    seconds
                )

            return True

    def remaining_ttl(
        self,
        memory_id: str,
    ) -> Optional[float]:
        """
        Return remaining TTL in seconds.

        Returns None if no expiry exists.
        """

        with self._lock:

            entry = self._entries.get(
                memory_id
            )

            if entry is None:
                return None

            if (
                entry.expires_monotonic
                is None
            ):
                return None

            remaining = (
                entry.expires_monotonic
                - time.monotonic()
            )

            return max(
                0.0,
                remaining,
            )

    def cleanup_expired(self) -> int:
        """
        Remove expired memories.

        Returns the number removed.
        """

        with self._lock:
            return self._remove_expired_locked()

    # ======================================================================
    # PROMOTION
    # ======================================================================

    def promote(
        self,
        memory_id: str,
    ) -> bool:
        """
        Promote a short-term memory into persistent storage.
        """

        with self._lock:

            memory = self._get_memory_locked(
                memory_id
            )

            if memory is None:
                return False

            return self._promote_locked(
                memory
            )

    def promote_all_important(self) -> int:
        """
        Promote all high/critical importance memories.
        """

        promoted = 0

        with self._lock:

            for entry in list(
                self._entries.values()
            ):

                memory = self._get_memory_locked(
                    entry.memory_id
                )

                if memory is None:
                    continue

                if (
                    memory.importance
                    not in self.PROMOTION_IMPORTANCE
                ):
                    continue

                if self._promote_locked(
                    memory
                ):
                    promoted += 1

        return promoted

    # ======================================================================
    # REMOVE
    # ======================================================================

    def remove(
        self,
        memory_id: str,
        *,
        permanent: bool = False,
    ) -> bool:
        """
        Remove a memory from short-term memory.

        permanent controls whether the underlying MemoryStore record is
        also permanently deleted.
        """

        with self._lock:

            existed = (
                memory_id
                in self._entries
            )

            self._entries.pop(
                memory_id,
                None,
            )

            if self.store is not None:

                if permanent:

                    self.store.delete(
                        memory_id,
                        permanent=True,
                    )

                else:

                    memory = self.store.get(
                        memory_id
                    )

                    if memory is not None:
                        memory.status = (
                            MemoryStatus.DELETED
                        )

                        self.store.update(
                            memory
                        )

            return existed

    def clear(
        self,
        *,
        preserve_important: bool = True,
        preserve_pinned: bool = True,
    ) -> int:
        """
        Clear short-term memory.

        Important/pinned memories can be preserved.
        """

        removed = 0

        with self._lock:

            for memory_id in list(
                self._entries.keys()
            ):

                memory = self._get_memory_locked(
                    memory_id
                )

                if memory is None:
                    self._entries.pop(
                        memory_id,
                        None,
                    )
                    continue

                if (
                    preserve_pinned
                    and memory.pinned
                ):
                    continue

                if (
                    preserve_important
                    and memory.importance
                    in self.PROMOTION_IMPORTANCE
                ):
                    continue

                self._entries.pop(
                    memory_id,
                    None,
                )

                removed += 1

        return removed

    # ======================================================================
    # CAPACITY
    # ======================================================================

    def set_capacity(
        self,
        capacity: int,
    ) -> None:
        """
        Change the short-term memory capacity.
        """

        if capacity <= 0:
            raise ValueError(
                "Capacity must be greater than zero."
            )

        with self._lock:

            self.capacity = int(
                capacity
            )

            self._enforce_capacity_locked()

    # ======================================================================
    # SNAPSHOT
    # ======================================================================

    def snapshot(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[dict[str, Any]]:
        """
        Return short-term memory as plain dictionaries.
        """

        memories = self.recent(
            limit=(
                limit
                if limit is not None
                else self.capacity
            )
        )

        return [
            {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "memory_type": (
                    memory.memory_type.value
                ),
                "importance": (
                    memory.importance.value
                ),
                "status": memory.status.value,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
                "accessed_at": memory.accessed_at,
                "access_count": memory.access_count,
                "user_id": memory.user_id,
                "session_id": memory.session_id,
                "conversation_id": memory.conversation_id,
                "project_id": memory.project_id,
                "task_id": memory.task_id,
                "source": memory.source,
                "tags": list(
                    memory.tags
                ),
                "metadata": dict(
                    memory.metadata
                ),
                "confidence": memory.confidence,
                "pinned": memory.pinned,
                "sensitive": memory.sensitive,
                "remaining_ttl": self.remaining_ttl(
                    memory.memory_id
                ),
            }
            for memory in memories
        ]

    # ======================================================================
    # HEALTH
    # ======================================================================

    def health(self) -> dict[str, Any]:
        """Return short-term memory health information."""

        with self._lock:

            self._remove_expired_locked()

            return {
                "status": "healthy",
                "count": len(
                    self._entries
                ),
                "capacity": self.capacity,
                "utilization": (
                    len(self._entries)
                    / self.capacity
                ),
                "default_ttl": self.default_ttl,
                "auto_promote": self.auto_promote,
                "persistent_store": (
                    self.store is not None
                ),
            }

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================

    def _get_memory_locked(
        self,
        memory_id: str,
    ) -> Optional[Memory]:

        if self.store is None:
            return None

        return self.store.get(
            memory_id
        )

    def _touch_locked(
        self,
        memory: Memory,
        entry: ShortTermEntry,
    ) -> None:

        now_monotonic = time.monotonic()

        entry.last_access_monotonic = (
            now_monotonic
        )

        entry.access_count += 1

        memory.access_count += 1

        memory.accessed_at = (
            self._utc_now()
        )

        memory.updated_at = (
            memory.accessed_at
        )

        if self.store is not None:
            self.store.update(
                memory
            )

    def _remove_expired_locked(
        self,
    ) -> int:

        now = time.monotonic()

        expired: list[
            str
        ] = []

        for memory_id, entry in (
            self._entries.items()
        ):

            if (
                entry.expires_monotonic
                is not None
                and now
                >= entry.expires_monotonic
            ):
                expired.append(
                    memory_id
                )

        for memory_id in expired:

            memory = self._get_memory_locked(
                memory_id
            )

            if (
                memory is not None
                and self.auto_promote
                and (
                    memory.pinned
                    or memory.importance
                    in self.PROMOTION_IMPORTANCE
                )
            ):
                self._promote_locked(
                    memory
                )

            self._entries.pop(
                memory_id,
                None,
            )

        return len(
            expired
        )

    def _enforce_capacity_locked(
        self,
    ) -> None:

        while len(
            self._entries
        ) > self.capacity:

            candidate = (
                self._select_eviction_candidate_locked()
            )

            if candidate is None:
                logger.warning(
                    "Short-term memory exceeded capacity "
                    "because all entries are protected."
                )
                break

            memory_id = candidate.memory_id

            memory = self._get_memory_locked(
                memory_id
            )

            if (
                memory is not None
                and self.auto_promote
                and (
                    memory.pinned
                    or memory.importance
                    in self.PROMOTION_IMPORTANCE
                )
            ):
                self._promote_locked(
                    memory
                )

            self._entries.pop(
                memory_id,
                None,
            )

    def _select_eviction_candidate_locked(
        self,
    ) -> Optional[ShortTermEntry]:
        """
        Select the least useful entry for eviction.

        Priority:
        1. Never evict pinned memories.
        2. Prefer lower importance.
        3. Prefer least recently accessed.
        4. Prefer lower access count.
        """

        candidates: list[
            ShortTermEntry
        ] = []

        for entry in self._entries.values():

            memory = self._get_memory_locked(
                entry.memory_id
            )

            if memory is None:
                candidates.append(
                    entry
                )
                continue

            if memory.pinned:
                continue

            candidates.append(
                entry
            )

        if not candidates:
            return None

        importance_rank = {
            MemoryImportance.LOW: 0,
            MemoryImportance.NORMAL: 1,
            MemoryImportance.HIGH: 2,
            MemoryImportance.CRITICAL: 3,
        }

        def eviction_key(
            entry: ShortTermEntry,
        ) -> tuple:

            memory = self._get_memory_locked(
                entry.memory_id
            )

            if memory is None:
                importance = 0

            else:
                importance = (
                    importance_rank.get(
                        memory.importance,
                        1,
                    )
                )

            return (
                importance,
                entry.last_access_monotonic,
                entry.access_count,
            )

        return min(
            candidates,
            key=eviction_key,
        )

    def _promote_locked(
        self,
        memory: Memory,
    ) -> bool:

        if self.store is None:
            return False

        try:

            # Keep the original memory available in the persistent store.
            if self.store.exists(
                memory.memory_id
            ):
                self.store.update(
                    memory
                )
            else:
                self.store.add(
                    memory
                )

            return True

        except Exception as exc:

            logger.warning(
                "Unable to promote short-term memory %s: %s",
                memory.memory_id,
                exc,
            )

            return False

    def _local_search(
        self,
        query: str,
        *,
        limit: int,
    ) -> List[Memory]:

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        candidates: list[
            tuple[
                float,
                Memory,
            ]
        ] = []

        for entry in self._entries.values():

            memory = self._get_memory_locked(
                entry.memory_id
            )

            if memory is None:
                continue

            memory_tokens = self._tokenize(
                memory.content
            )

            overlap = (
                query_tokens
                & memory_tokens
            )

            if not overlap:
                continue

            score = (
                len(overlap)
                / max(
                    len(query_tokens),
                    1,
                )
            )

            if (
                query.lower().strip()
                in memory.content.lower()
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

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:

        cleaned = (
            str(text)
            .lower()
            .replace(",", " ")
            .replace(".", " ")
            .replace("!", " ")
            .replace("?", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("/", " ")
            .replace("\\", " ")
            .replace("-", " ")
            .replace("_", " ")
        )

        return {
            token
            for token in cleaned.split()
            if len(token) >= 2
        }

    @staticmethod
    def _new_id() -> str:
        return (
            "stm_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _utc_now() -> str:
        return (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )


__all__ = [
    "ShortTermMemory",
    "ShortTermMemoryError",
    "ShortTermEntry",
]


