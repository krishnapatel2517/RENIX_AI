"""
RENIX Long-Term Memory
======================

Persistent memory layer for RENIX.

Responsibilities
----------------
- Store important memories beyond the current session.
- Retrieve persistent memories.
- Update memory importance.
- Promote memories from short-term context.
- Preserve user/project/task information.
- Support memory pinning.
- Support archival and forgetting.
- Provide filtered retrieval for RENIX's memory manager.

This module intentionally delegates physical persistence to MemoryStore.
It does not implement embeddings or vector similarity.
Those responsibilities belong to:
    memory/vector_store.py
    memory/embeddings.py
    memory/retrieval.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, List, Optional
import logging
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


class LongTermMemoryError(Exception):
    """Base exception for long-term memory failures."""


@dataclass
class LongTermMemoryStats:
    """Statistics for the long-term memory layer."""

    total: int = 0
    active: int = 0
    archived: int = 0
    deleted: int = 0
    pinned: int = 0
    critical: int = 0
    high: int = 0


class LongTermMemory:
    """
    Persistent RENIX long-term memory.

    Long-term memory is intended for information that should survive
    conversation/session boundaries, for example:

    - stable preferences
    - important user instructions
    - project information
    - recurring tasks
    - important decisions
    - learned facts
    - durable conversation information
    - important behavioral preferences

    Parameters
    ----------
    store:
        Persistent MemoryStore used by RENIX.
    """

    def __init__(
        self,
        store: MemoryStore,
    ) -> None:

        if store is None:
            raise ValueError(
                "LongTermMemory requires a MemoryStore."
            )

        self.store = store

        self._lock = RLock()

    # ======================================================================
    # CREATE
    # ======================================================================

    def add(
        self,
        memory: Memory,
    ) -> Memory:
        """
        Add a memory to long-term storage.
        """

        if not isinstance(
            memory,
            Memory,
        ):
            raise TypeError(
                "Long-term memory accepts Memory objects."
            )

        with self._lock:

            # Long-term memories should not remain classified purely as
            # temporary short-term memories.
            if (
                memory.memory_type
                == MemoryType.SHORT_TERM
            ):
                memory.memory_type = (
                    MemoryType.LONG_TERM
                )

            memory.status = (
                MemoryStatus.ACTIVE
            )

            memory.updated_at = (
                self._utc_now()
            )

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

            return memory

    def add_text(
        self,
        content: str,
        *,
        importance: MemoryImportance = MemoryImportance.NORMAL,
        memory_type: MemoryType = MemoryType.LONG_TERM,
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
        Create and persist a long-term memory from text.
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
            memory
        )

    # ======================================================================
    # READ
    # ======================================================================

    def get(
        self,
        memory_id: str,
        *,
        touch: bool = True,
    ) -> Optional[Memory]:
        """
        Retrieve persistent memory.
        """

        with self._lock:

            memory = self.store.get(
                memory_id
            )

            if memory is None:
                return None

            if (
                memory.status
                == MemoryStatus.DELETED
            ):
                return None

            if touch:
                self._touch(
                    memory
                )

            return memory

    def require(
        self,
        memory_id: str,
    ) -> Memory:
        """
        Retrieve a memory or raise MemoryNotFoundError.
        """

        memory = self.get(
            memory_id
        )

        if memory is None:
            raise MemoryNotFoundError(
                f"Long-term memory not found: "
                f"{memory_id}"
            )

        return memory

    def exists(
        self,
        memory_id: str,
    ) -> bool:

        memory = self.get(
            memory_id,
            touch=False,
        )

        return memory is not None

    # ======================================================================
    # RETRIEVAL
    # ======================================================================

    def recent(
        self,
        *,
        limit: int = 50,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Memory]:
        """
        Return recently updated long-term memories.
        """

        limit = max(
            1,
            int(limit),
        )

        memories = self._all_memories()

        filtered: list[
            Memory
        ] = []

        for memory in memories:

            if (
                memory.status
                != MemoryStatus.ACTIVE
            ):
                continue

            if (
                user_id is not None
                and memory.user_id
                != user_id
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

            filtered.append(
                memory
            )

        filtered.sort(
            key=lambda memory: (
                self._timestamp(
                    memory.updated_at
                )
            ),
            reverse=True,
        )

        return filtered[:limit]

    def important(
        self,
        *,
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """
        Return high-value persistent memories.
        """

        memories = self._all_memories()

        importance_rank = {
            MemoryImportance.LOW: 0,
            MemoryImportance.NORMAL: 1,
            MemoryImportance.HIGH: 2,
            MemoryImportance.CRITICAL: 3,
        }

        result: list[
            Memory
        ] = []

        for memory in memories:

            if (
                memory.status
                != MemoryStatus.ACTIVE
            ):
                continue

            if (
                memory.importance
                not in (
                    MemoryImportance.HIGH,
                    MemoryImportance.CRITICAL,
                )
            ):
                continue

            if (
                user_id is not None
                and memory.user_id
                != user_id
            ):
                continue

            result.append(
                memory
            )

        result.sort(
            key=lambda memory: (
                importance_rank.get(
                    memory.importance,
                    0,
                ),
                self._timestamp(
                    memory.updated_at
                ),
            ),
            reverse=True,
        )

        return result[:max(1, limit)]

    def pinned(
        self,
        *,
        limit: int = 100,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """
        Return memories explicitly pinned by RENIX/user.
        """

        memories = self._all_memories()

        result = [
            memory
            for memory in memories
            if (
                memory.pinned
                and memory.status
                == MemoryStatus.ACTIVE
                and (
                    user_id is None
                    or memory.user_id
                    == user_id
                )
            )
        ]

        result.sort(
            key=lambda memory: (
                self._timestamp(
                    memory.updated_at
                )
            ),
            reverse=True,
        )

        return result[:max(1, limit)]

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> List[Memory]:
        """
        Perform persistent lexical search.

        Semantic/vector retrieval is intentionally handled by the
        dedicated retrieval layer.
        """

        if not query or not query.strip():
            return []

        limit = max(
            1,
            int(limit),
        )

        results = self.store.text_search(
            query,
            limit=limit,
        )

        return [
            memory
            for memory in results
            if memory.status
            == MemoryStatus.ACTIVE
        ]

    # ======================================================================
    # UPDATE
    # ======================================================================

    def update(
        self,
        memory: Memory,
    ) -> Memory:
        """
        Update an existing persistent memory.
        """

        if not isinstance(
            memory,
            Memory,
        ):
            raise TypeError(
                "Expected a Memory object."
            )

        with self._lock:

            if not self.store.exists(
                memory.memory_id
            ):
                raise MemoryNotFoundError(
                    f"Cannot update missing memory: "
                    f"{memory.memory_id}"
                )

            memory.updated_at = (
                self._utc_now()
            )

            self.store.update(
                memory
            )

            return memory

    def update_content(
        self,
        memory_id: str,
        content: str,
    ) -> Optional[Memory]:
        """
        Update the content of an existing memory.
        """

        with self._lock:

            memory = self.get(
                memory_id,
                touch=False,
            )

            if memory is None:
                return None

            memory.content = str(
                content
            )

            return self.update(
                memory
            )

    def set_importance(
        self,
        memory_id: str,
        importance: MemoryImportance,
    ) -> Optional[Memory]:
        """
        Change memory importance.
        """

        with self._lock:

            memory = self.get(
                memory_id,
                touch=False,
            )

            if memory is None:
                return None

            memory.importance = (
                importance
            )

            return self.update(
                memory
            )

    def set_tags(
        self,
        memory_id: str,
        tags: Iterable[str],
    ) -> Optional[Memory]:
        """
        Replace the memory's tags.
        """

        with self._lock:

            memory = self.get(
                memory_id,
                touch=False,
            )

            if memory is None:
                return None

            memory.tags = list(
                dict.fromkeys(
                    str(tag)
                    for tag in tags
                    if str(tag).strip()
                )
            )

            return self.update(
                memory
            )

    # ======================================================================
    # PINNING
    # ======================================================================

    def pin(
        self,
        memory_id: str,
    ) -> bool:
        """
        Permanently prioritize a memory.
        """

        with self._lock:

            memory = self.get(
                memory_id,
                touch=False,
            )

            if memory is None:
                return False

            memory.pinned = True

            self.update(
                memory
            )

            return True

    def unpin(
        self,
        memory_id: str,
    ) -> bool:
        """
        Remove the pinned flag.
        """

        with self._lock:

            memory = self.get(
                memory_id,
                touch=False,
            )

            if memory is None:
                return False

            memory.pinned = False

            self.update(
                memory
            )

            return True

    # ======================================================================
    # ARCHIVING
    # ======================================================================

    def archive(
        self,
        memory_id: str,
    ) -> bool:
        """
        Archive a memory without permanently deleting it.
        """

        with self._lock:

            memory = self.store.get(
                memory_id
            )

            if memory is None:
                return False

            memory.status = (
                MemoryStatus.ARCHIVED
            )

            memory.updated_at = (
                self._utc_now()
            )

            self.store.update(
                memory
            )

            return True

    def restore(
        self,
        memory_id: str,
    ) -> bool:
        """
        Restore an archived memory.
        """

        with self._lock:

            memory = self.store.get(
                memory_id
            )

            if memory is None:
                return False

            memory.status = (
                MemoryStatus.ACTIVE
            )

            memory.updated_at = (
                self._utc_now()
            )

            self.store.update(
                memory
            )

            return True

    # ======================================================================
    # FORGETTING
    # ======================================================================

    def forget(
        self,
        memory_id: str,
        *,
        permanent: bool = False,
    ) -> bool:
        """
        Forget a memory.

        permanent=False:
            Soft-delete the memory.

        permanent=True:
            Permanently remove it from storage.
        """

        with self._lock:

            memory = self.store.get(
                memory_id
            )

            if memory is None:
                return False

            if permanent:

                self.store.delete(
                    memory_id,
                    permanent=True,
                )

                return True

            memory.status = (
                MemoryStatus.DELETED
            )

            memory.updated_at = (
                self._utc_now()
            )

            self.store.update(
                memory
            )

            return True

    # ======================================================================
    # USER / PROJECT MEMORY
    # ======================================================================

    def for_user(
        self,
        user_id: str,
        *,
        limit: int = 100,
    ) -> List[Memory]:
        """
        Return persistent memories belonging to a user.
        """

        return self.recent(
            limit=limit,
            user_id=user_id,
        )

    def for_project(
        self,
        project_id: str,
        *,
        limit: int = 100,
    ) -> List[Memory]:
        """
        Return persistent memories belonging to a project.
        """

        return self.recent(
            limit=limit,
            project_id=project_id,
        )

    def for_task(
        self,
        task_id: str,
        *,
        limit: int = 100,
    ) -> List[Memory]:
        """
        Return persistent memories belonging to a task.
        """

        return self.recent(
            limit=limit,
            task_id=task_id,
        )

    # ======================================================================
    # CONTEXT
    # ======================================================================

    def build_context(
        self,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 30,
    ) -> List[Memory]:
        """
        Build persistent context for RENIX.

        Important memories are placed first, followed by recent memories.
        Duplicate memory IDs are removed.
        """

        result: list[
            Memory
        ] = []

        seen: set[
            str
        ] = set()

        important = self.important(
            limit=limit,
            user_id=user_id,
        )

        for memory in important:

            if project_id is not None:
                if (
                    memory.project_id
                    != project_id
                ):
                    continue

            if task_id is not None:
                if (
                    memory.task_id
                    != task_id
                ):
                    continue

            if memory.memory_id in seen:
                continue

            seen.add(
                memory.memory_id
            )

            result.append(
                memory
            )

        remaining = max(
            0,
            limit - len(result),
        )

        if remaining:

            recent = self.recent(
                limit=remaining * 2,
                user_id=user_id,
                project_id=project_id,
                task_id=task_id,
            )

            for memory in recent:

                if memory.memory_id in seen:
                    continue

                seen.add(
                    memory.memory_id
                )

                result.append(
                    memory
                )

                if len(result) >= limit:
                    break

        return result

    def context_text(
        self,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 30,
    ) -> str:
        """
        Convert persistent context into compact text.
        """

        memories = self.build_context(
            user_id=user_id,
            project_id=project_id,
            task_id=task_id,
            limit=limit,
        )

        return "\n".join(
            f"- {memory.content}"
            for memory in memories
        )

    # ======================================================================
    # STATISTICS
    # ======================================================================

    def statistics(self) -> LongTermMemoryStats:
        """
        Return long-term memory statistics.
        """

        stats = LongTermMemoryStats()

        for memory in self._all_memories():

            stats.total += 1

            if (
                memory.status
                == MemoryStatus.ACTIVE
            ):
                stats.active += 1

            elif (
                memory.status
                == MemoryStatus.ARCHIVED
            ):
                stats.archived += 1

            elif (
                memory.status
                == MemoryStatus.DELETED
            ):
                stats.deleted += 1

            if memory.pinned:
                stats.pinned += 1

            if (
                memory.importance
                == MemoryImportance.CRITICAL
            ):
                stats.critical += 1

            elif (
                memory.importance
                == MemoryImportance.HIGH
            ):
                stats.high += 1

        return stats

    def health(self) -> dict[str, Any]:
        """
        Return health information for RENIX diagnostics.
        """

        stats = self.statistics()

        return {
            "status": "healthy",
            "persistent_store": True,
            "total": stats.total,
            "active": stats.active,
            "archived": stats.archived,
            "deleted": stats.deleted,
            "pinned": stats.pinned,
            "critical": stats.critical,
            "high": stats.high,
        }

    # ======================================================================
    # INTERNAL
    # ======================================================================

    def _touch(
        self,
        memory: Memory,
    ) -> None:

        memory.access_count += 1

        memory.accessed_at = (
            self._utc_now()
        )

        self.store.update(
            memory
        )

    def _all_memories(
        self,
    ) -> List[Memory]:

        """
        Obtain all memories from MemoryStore.

        MemoryStore implementations may expose either all() or list().
        """

        if hasattr(
            self.store,
            "all",
        ):
            result = self.store.all()

        elif hasattr(
            self.store,
            "list",
        ):
            result = self.store.list()

        else:
            raise LongTermMemoryError(
                "MemoryStore does not expose "
                "an all() or list() method."
            )

        return list(
            result
        )

    @staticmethod
    def _new_id() -> str:
        return (
            "ltm_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _utc_now() -> str:
        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    @staticmethod
    def _timestamp(
        value: Any,
    ) -> float:

        if value is None:
            return 0.0

        if isinstance(
            value,
            datetime,
        ):
            return value.timestamp()

        text = str(
            value
        )

        try:

            return datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            ).timestamp()

        except (
            ValueError,
            TypeError,
        ):
            return 0.0


__all__ = [
    "LongTermMemory",
    "LongTermMemoryError",
    "LongTermMemoryStats",
]


