"""
RENIX Semantic Memory
=====================

Stores durable factual/knowledge-oriented memories for RENIX.

Semantic memory answers:
    "What does RENIX know?"

Examples:
    - User prefers a particular workflow.
    - A project uses Python.
    - A particular file belongs to a project.
    - A known fact was established during a previous task.

This module manages semantic records and delegates physical persistence
to MemoryStore.

Vector embeddings and semantic similarity are intentionally NOT implemented
here. Those responsibilities belong to:

    memory/embeddings.py
    memory/vector_store.py
    memory/retrieval.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


class SemanticMemoryError(Exception):
    """Base exception for semantic-memory failures."""


@dataclass
class SemanticFact:
    """
    A structured fact known by RENIX.

    A semantic fact is deliberately separate from raw Memory so that RENIX
    can reason about:

        subject -> predicate -> object

    Example:

        subject:  "RENIX"
        predicate: "name"
        object:    "RENIX"

    Another example:

        subject:  "user"
        predicate: "prefers"
        object:    "night study"
    """

    fact_id: str
    subject: str
    predicate: str
    object: str

    confidence: float = 1.0

    source: Optional[str] = None

    user_id: Optional[str] = None
    project_id: Optional[str] = None
    task_id: Optional[str] = None

    tags: List[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    memory_id: Optional[str] = None

    importance: MemoryImportance = (
        MemoryImportance.NORMAL
    )

    status: MemoryStatus = (
        MemoryStatus.ACTIVE
    )

    created_at: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    updated_at: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )


class SemanticMemory:
    """
    RENIX semantic-memory manager.

    Responsibilities
    ----------------
    - Create factual memories.
    - Store subject/predicate/object relationships.
    - Update facts.
    - Resolve conflicting facts.
    - Search facts.
    - Retrieve facts by subject.
    - Retrieve facts by predicate.
    - Retrieve facts by project/task/user.
    - Link facts to persistent Memory objects.
    - Promote facts into durable memory.
    - Archive/delete obsolete facts.
    """

    def __init__(
        self,
        store: MemoryStore,
    ) -> None:

        if store is None:
            raise ValueError(
                "SemanticMemory requires a MemoryStore."
            )

        self.store = store

        self._facts: dict[
            str,
            SemanticFact,
        ] = {}

        self._lock = RLock()

    # ======================================================================
    # CREATE
    # ======================================================================

    def add(
        self,
        subject: str,
        predicate: str,
        object: str,
        *,
        confidence: float = 1.0,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tags: Optional[
            Iterable[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        importance: MemoryImportance = (
            MemoryImportance.NORMAL
        ),
        memory_id: Optional[str] = None,
    ) -> SemanticFact:
        """
        Create a semantic fact.

        If an equivalent active fact already exists, its confidence,
        metadata and timestamps are updated instead of creating a duplicate.
        """

        subject = str(
            subject
        ).strip()

        predicate = str(
            predicate
        ).strip()

        object = str(
            object
        ).strip()

        if not subject:
            raise ValueError(
                "Semantic fact subject cannot be empty."
            )

        if not predicate:
            raise ValueError(
                "Semantic fact predicate cannot be empty."
            )

        if not object:
            raise ValueError(
                "Semantic fact object cannot be empty."
            )

        confidence = self._clamp_confidence(
            confidence
        )

        with self._lock:

            existing = self.find_exact(
                subject,
                predicate,
                object,
                user_id=user_id,
                project_id=project_id,
                task_id=task_id,
            )

            if existing is not None:

                existing.confidence = max(
                    existing.confidence,
                    confidence,
                )

                if source is not None:
                    existing.source = source

                if memory_id is not None:
                    existing.memory_id = (
                        memory_id
                    )

                if tags:
                    existing.tags = (
                        self._merge_unique(
                            existing.tags,
                            tags,
                        )
                    )

                if metadata:
                    existing.metadata.update(
                        metadata
                    )

                if (
                    self._importance_rank(
                        importance
                    )
                    > self._importance_rank(
                        existing.importance
                    )
                ):
                    existing.importance = (
                        importance
                    )

                existing.updated_at = (
                    self._utc_now()
                )

                return existing

            now = self._utc_now()

            fact = SemanticFact(
                fact_id=self._new_id(),
                subject=subject,
                predicate=predicate,
                object=object,
                confidence=confidence,
                source=source,
                user_id=user_id,
                project_id=project_id,
                task_id=task_id,
                tags=self._clean_list(
                    tags
                ),
                metadata=dict(
                    metadata or {}
                ),
                memory_id=memory_id,
                importance=importance,
                status=MemoryStatus.ACTIVE,
                created_at=now,
                updated_at=now,
            )

            self._facts[
                fact.fact_id
            ] = fact

            return fact

    # ======================================================================
    # CREATE FROM TEXT
    # ======================================================================

    def add_memory(
        self,
        content: str,
        *,
        confidence: float = 1.0,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tags: Optional[
            Iterable[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        importance: MemoryImportance = (
            MemoryImportance.NORMAL
        ),
    ) -> Memory:
        """
        Create a durable Memory object classified as semantic knowledge.

        This method is useful when the semantic statement is already
        expressed as natural language and no structured triple is required.
        """

        now = self._utc_now()

        memory = Memory(
            memory_id=self._new_memory_id(),
            content=str(content),
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            status=MemoryStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            accessed_at=now,
            access_count=0,
            user_id=user_id,
            session_id=None,
            conversation_id=None,
            project_id=project_id,
            task_id=task_id,
            source=source,
            tags=self._clean_list(
                tags
            ),
            metadata=dict(
                metadata or {}
            ),
            confidence=self._clamp_confidence(
                confidence
            ),
            expires_at=None,
            pinned=False,
            sensitive=False,
            embedding_id=None,
        )

        self.store.add(
            memory
        )

        return memory

    # ======================================================================
    # READ
    # ======================================================================

    def get(
        self,
        fact_id: str,
    ) -> Optional[SemanticFact]:
        """
        Retrieve a semantic fact.
        """

        with self._lock:

            return self._facts.get(
                fact_id
            )

    def require(
        self,
        fact_id: str,
    ) -> SemanticFact:
        """
        Retrieve a fact or raise SemanticMemoryError.
        """

        fact = self.get(
            fact_id
        )

        if fact is None:
            raise SemanticMemoryError(
                f"Semantic fact not found: "
                f"{fact_id}"
            )

        return fact

    def exists(
        self,
        fact_id: str,
    ) -> bool:

        with self._lock:

            return (
                fact_id
                in self._facts
            )

    # ======================================================================
    # EXACT MATCH
    # ======================================================================

    def find_exact(
        self,
        subject: str,
        predicate: str,
        object: str,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Optional[SemanticFact]:
        """
        Find an exact semantic triple.
        """

        subject_key = (
            str(subject)
            .strip()
            .casefold()
        )

        predicate_key = (
            str(predicate)
            .strip()
            .casefold()
        )

        object_key = (
            str(object)
            .strip()
            .casefold()
        )

        with self._lock:

            for fact in self._facts.values():

                if (
                    fact.status
                    != MemoryStatus.ACTIVE
                ):
                    continue

                if (
                    fact.subject.casefold()
                    != subject_key
                ):
                    continue

                if (
                    fact.predicate.casefold()
                    != predicate_key
                ):
                    continue

                if (
                    fact.object.casefold()
                    != object_key
                ):
                    continue

                if (
                    user_id is not None
                    and fact.user_id
                    != user_id
                ):
                    continue

                if (
                    project_id is not None
                    and fact.project_id
                    != project_id
                ):
                    continue

                if (
                    task_id is not None
                    and fact.task_id
                    != task_id
                ):
                    continue

                return fact

        return None

    # ======================================================================
    # SUBJECT RETRIEVAL
    # ======================================================================

    def by_subject(
        self,
        subject: str,
        *,
        limit: int = 100,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[SemanticFact]:
        """
        Return all facts associated with a subject.
        """

        key = (
            str(subject)
            .strip()
            .casefold()
        )

        with self._lock:

            result = [
                fact
                for fact
                in self._facts.values()
                if (
                    fact.status
                    == MemoryStatus.ACTIVE
                    and fact.subject.casefold()
                    == key
                    and (
                        user_id is None
                        or fact.user_id
                        == user_id
                    )
                    and (
                        project_id is None
                        or fact.project_id
                        == project_id
                    )
                    and (
                        task_id is None
                        or fact.task_id
                        == task_id
                    )
                )
            ]

        return self._sort_facts(
            result
        )[:max(1, int(limit))]

    # ======================================================================
    # PREDICATE RETRIEVAL
    # ======================================================================

    def by_predicate(
        self,
        predicate: str,
        *,
        limit: int = 100,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[SemanticFact]:
        """
        Return all facts using a predicate.
        """

        key = (
            str(predicate)
            .strip()
            .casefold()
        )

        with self._lock:

            result = [
                fact
                for fact
                in self._facts.values()
                if (
                    fact.status
                    == MemoryStatus.ACTIVE
                    and fact.predicate.casefold()
                    == key
                    and (
                        user_id is None
                        or fact.user_id
                        == user_id
                    )
                    and (
                        project_id is None
                        or fact.project_id
                        == project_id
                    )
                    and (
                        task_id is None
                        or fact.task_id
                        == task_id
                    )
                )
            ]

        return self._sort_facts(
            result
        )[:max(1, int(limit))]

    # ======================================================================
    # OBJECT RETRIEVAL
    # ======================================================================

    def by_object(
        self,
        object: str,
        *,
        limit: int = 100,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[SemanticFact]:
        """
        Return facts matching an object value.
        """

        key = (
            str(object)
            .strip()
            .casefold()
        )

        with self._lock:

            result = [
                fact
                for fact
                in self._facts.values()
                if (
                    fact.status
                    == MemoryStatus.ACTIVE
                    and fact.object.casefold()
                    == key
                    and (
                        user_id is None
                        or fact.user_id
                        == user_id
                    )
                    and (
                        project_id is None
                        or fact.project_id
                        == project_id
                    )
                    and (
                        task_id is None
                        or fact.task_id
                        == task_id
                    )
                )
            ]

        return self._sort_facts(
            result
        )[:max(1, int(limit))]

    # ======================================================================
    # SEARCH
    # ======================================================================

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[SemanticFact]:
        """
        Search semantic facts lexically.

        Scoring prioritizes:
            1. Exact phrase matches.
            2. Subject matches.
            3. Predicate matches.
            4. Object matches.
            5. Tag matches.
            6. Token overlap.
        """

        query = str(
            query
        ).strip()

        if not query:
            return []

        query_lower = query.casefold()

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        scored: list[
            tuple[
                float,
                SemanticFact,
            ]
        ] = []

        with self._lock:

            facts = list(
                self._facts.values()
            )

        for fact in facts:

            if (
                fact.status
                != MemoryStatus.ACTIVE
            ):
                continue

            if (
                user_id is not None
                and fact.user_id
                != user_id
            ):
                continue

            if (
                project_id is not None
                and fact.project_id
                != project_id
            ):
                continue

            if (
                task_id is not None
                and fact.task_id
                != task_id
            ):
                continue

            score = 0.0

            subject = (
                fact.subject.casefold()
            )

            predicate = (
                fact.predicate.casefold()
            )

            object_value = (
                fact.object.casefold()
            )

            tags = " ".join(
                fact.tags
            ).casefold()

            combined = " ".join(
                [
                    subject,
                    predicate,
                    object_value,
                    tags,
                ]
            )

            if (
                query_lower
                in subject
            ):
                score += 5.0

            if (
                query_lower
                in predicate
            ):
                score += 4.0

            if (
                query_lower
                in object_value
            ):
                score += 4.0

            if (
                query_lower
                in combined
            ):
                score += 2.0

            fact_tokens = (
                self._tokenize(
                    combined
                )
            )

            overlap = (
                query_tokens
                & fact_tokens
            )

            score += (
                len(overlap)
                * 1.5
            )

            score += (
                fact.confidence
                * 0.5
            )

            if score > 0:
                scored.append(
                    (
                        score,
                        fact,
                    )
                )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].confidence,
                self._timestamp(
                    item[1].updated_at
                ),
            ),
            reverse=True,
        )

        return [
            fact
            for _, fact
            in scored[
                :max(1, int(limit))
            ]
        ]

    # ======================================================================
    # UPDATE
    # ======================================================================

    def update(
        self,
        fact: SemanticFact,
    ) -> SemanticFact:
        """
        Update an existing semantic fact.
        """

        if not isinstance(
            fact,
            SemanticFact,
        ):
            raise TypeError(
                "Expected SemanticFact."
            )

        with self._lock:

            if (
                fact.fact_id
                not in self._facts
            ):
                raise SemanticMemoryError(
                    f"Fact does not exist: "
                    f"{fact.fact_id}"
                )

            fact.subject = str(
                fact.subject
            ).strip()

            fact.predicate = str(
                fact.predicate
            ).strip()

            fact.object = str(
                fact.object
            ).strip()

            fact.confidence = (
                self._clamp_confidence(
                    fact.confidence
                )
            )

            fact.updated_at = (
                self._utc_now()
            )

            self._facts[
                fact.fact_id
            ] = fact

            return fact

    def set_confidence(
        self,
        fact_id: str,
        confidence: float,
    ) -> Optional[SemanticFact]:
        """
        Update fact confidence.
        """

        with self._lock:

            fact = self.get(
                fact_id
            )

            if fact is None:
                return None

            fact.confidence = (
                self._clamp_confidence(
                    confidence
                )
            )

            fact.updated_at = (
                self._utc_now()
            )

            return fact

    def set_importance(
        self,
        fact_id: str,
        importance: MemoryImportance,
    ) -> Optional[SemanticFact]:
        """
        Change fact importance.
        """

        with self._lock:

            fact = self.get(
                fact_id
            )

            if fact is None:
                return None

            fact.importance = (
                importance
            )

            fact.updated_at = (
                self._utc_now()
            )

            return fact

    # ======================================================================
    # CONFLICT RESOLUTION
    # ======================================================================

    def find_conflicts(
        self,
        subject: str,
        predicate: str,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[SemanticFact]:
        """
        Find competing values for the same subject/predicate pair.

        Example:

            user -> preferred_language -> English
            user -> preferred_language -> Hindi

        Both can be returned as competing facts.
        """

        subject_key = (
            str(subject)
            .strip()
            .casefold()
        )

        predicate_key = (
            str(predicate)
            .strip()
            .casefold()
        )

        with self._lock:

            result = [
                fact
                for fact
                in self._facts.values()
                if (
                    fact.status
                    == MemoryStatus.ACTIVE
                    and fact.subject.casefold()
                    == subject_key
                    and fact.predicate.casefold()
                    == predicate_key
                    and (
                        user_id is None
                        or fact.user_id
                        == user_id
                    )
                    and (
                        project_id is None
                        or fact.project_id
                        == project_id
                    )
                    and (
                        task_id is None
                        or fact.task_id
                        == task_id
                    )
                )
            ]

        return self._sort_facts(
            result
        )

    def resolve_conflict(
        self,
        fact_id: str,
        *,
        keep: bool = True,
        archive_others: bool = True,
    ) -> bool:
        """
        Resolve a fact against competing facts.

        If keep=True, the selected fact remains active.

        If archive_others=True, competing active facts sharing the same
        subject/predicate are archived.
        """

        with self._lock:

            selected = self.get(
                fact_id
            )

            if selected is None:
                return False

            if keep:
                selected.status = (
                    MemoryStatus.ACTIVE
                )

            if archive_others:

                for fact in (
                    self._facts.values()
                ):

                    if (
                        fact.fact_id
                        == selected.fact_id
                    ):
                        continue

                    if (
                        fact.status
                        != MemoryStatus.ACTIVE
                    ):
                        continue

                    if (
                        fact.subject.casefold()
                        != selected.subject.casefold()
                    ):
                        continue

                    if (
                        fact.predicate.casefold()
                        != selected.predicate.casefold()
                    ):
                        continue

                    if (
                        fact.user_id
                        != selected.user_id
                    ):
                        continue

                    if (
                        fact.project_id
                        != selected.project_id
                    ):
                        continue

                    if (
                        fact.task_id
                        != selected.task_id
                    ):
                        continue

                    fact.status = (
                        MemoryStatus.ARCHIVED
                    )

                    fact.updated_at = (
                        self._utc_now()
                    )

            selected.updated_at = (
                self._utc_now()
            )

            return True

    def strongest_fact(
        self,
        subject: str,
        predicate: str,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Optional[SemanticFact]:
        """
        Return the highest-confidence fact for a subject/predicate pair.
        """

        facts = self.find_conflicts(
            subject,
            predicate,
            user_id=user_id,
            project_id=project_id,
            task_id=task_id,
        )

        if not facts:
            return None

        facts.sort(
            key=lambda fact: (
                fact.confidence,
                self._importance_rank(
                    fact.importance
                ),
                self._timestamp(
                    fact.updated_at
                ),
            ),
            reverse=True,
        )

        return facts[0]

    # ======================================================================
    # MEMORY LINKING
    # ======================================================================

    def link_memory(
        self,
        fact_id: str,
        memory: Memory | str,
    ) -> bool:
        """
        Link a semantic fact to a persistent Memory object.
        """

        with self._lock:

            fact = self.get(
                fact_id
            )

            if fact is None:
                return False

            if isinstance(
                memory,
                Memory,
            ):

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

                fact.memory_id = (
                    memory.memory_id
                )

            else:

                memory_id = str(
                    memory
                )

                if not self.store.exists(
                    memory_id
                ):
                    raise MemoryNotFoundError(
                        f"Memory not found: "
                        f"{memory_id}"
                    )

                fact.memory_id = (
                    memory_id
                )

            fact.updated_at = (
                self._utc_now()
            )

            return True

    def get_linked_memory(
        self,
        fact_id: str,
    ) -> Optional[Memory]:
        """
        Retrieve the persistent Memory linked to a fact.
        """

        fact = self.get(
            fact_id
        )

        if (
            fact is None
            or not fact.memory_id
        ):
            return None

        return self.store.get(
            fact.memory_id
        )

    # ======================================================================
    # ARCHIVE / RESTORE / DELETE
    # ======================================================================

    def archive(
        self,
        fact_id: str,
    ) -> bool:
        """
        Archive a semantic fact.
        """

        with self._lock:

            fact = self.get(
                fact_id
            )

            if fact is None:
                return False

            fact.status = (
                MemoryStatus.ARCHIVED
            )

            fact.updated_at = (
                self._utc_now()
            )

            return True

    def restore(
        self,
        fact_id: str,
    ) -> bool:
        """
        Restore an archived semantic fact.
        """

        with self._lock:

            fact = self.get(
                fact_id
            )

            if fact is None:
                return False

            fact.status = (
                MemoryStatus.ACTIVE
            )

            fact.updated_at = (
                self._utc_now()
            )

            return True

    def delete(
        self,
        fact_id: str,
        *,
        permanent: bool = False,
    ) -> bool:
        """
        Delete a semantic fact.

        permanent=False:
            Soft delete.

        permanent=True:
            Remove from in-memory semantic index.
        """

        with self._lock:

            fact = self.get(
                fact_id
            )

            if fact is None:
                return False

            if permanent:

                del self._facts[
                    fact_id
                ]

                return True

            fact.status = (
                MemoryStatus.DELETED
            )

            fact.updated_at = (
                self._utc_now()
            )

            return True

    # ======================================================================
    # USER / PROJECT / TASK
    # ======================================================================

    def for_user(
        self,
        user_id: str,
        *,
        limit: int = 100,
    ) -> List[SemanticFact]:
        """
        Return semantic knowledge associated with a user.
        """

        with self._lock:

            result = [
                fact
                for fact
                in self._facts.values()
                if (
                    fact.status
                    == MemoryStatus.ACTIVE
                    and fact.user_id
                    == user_id
                )
            ]

        return self._sort_facts(
            result
        )[:max(1, int(limit))]

    def for_project(
        self,
        project_id: str,
        *,
        limit: int = 100,
    ) -> List[SemanticFact]:
        """
        Return semantic knowledge associated with a project.
        """

        with self._lock:

            result = [
                fact
                for fact
                in self._facts.values()
                if (
                    fact.status
                    == MemoryStatus.ACTIVE
                    and fact.project_id
                    == project_id
                )
            ]

        return self._sort_facts(
            result
        )[:max(1, int(limit))]

    def for_task(
        self,
        task_id: str,
        *,
        limit: int = 100,
    ) -> List[SemanticFact]:
        """
        Return semantic knowledge associated with a task.
        """

        with self._lock:

            result = [
                fact
                for fact
                in self._facts.values()
                if (
                    fact.status
                    == MemoryStatus.ACTIVE
                    and fact.task_id
                    == task_id
                )
            ]

        return self._sort_facts(
            result
        )[:max(1, int(limit))]

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
    ) -> List[SemanticFact]:
        """
        Build semantic knowledge context for RENIX.
        """

        with self._lock:

            candidates = [
                fact
                for fact
                in self._facts.values()
                if (
                    fact.status
                    == MemoryStatus.ACTIVE
                    and (
                        user_id is None
                        or fact.user_id
                        == user_id
                    )
                    and (
                        project_id is None
                        or fact.project_id
                        == project_id
                    )
                    and (
                        task_id is None
                        or fact.task_id
                        == task_id
                    )
                )
            ]

        candidates.sort(
            key=lambda fact: (
                self._importance_rank(
                    fact.importance
                ),
                fact.confidence,
                self._timestamp(
                    fact.updated_at
                ),
            ),
            reverse=True,
        )

        return candidates[
            :max(1, int(limit))
        ]

    def context_text(
        self,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 30,
    ) -> str:
        """
        Convert semantic facts into compact context text.
        """

        facts = self.build_context(
            user_id=user_id,
            project_id=project_id,
            task_id=task_id,
            limit=limit,
        )

        lines = []

        for fact in facts:

            lines.append(
                f"- {fact.subject} "
                f"{fact.predicate} "
                f"{fact.object} "
                f"(confidence="
                f"{fact.confidence:.2f})"
            )

        return "\n".join(
            lines
        )

    # ======================================================================
    # EXPORT
    # ======================================================================

    def to_dict(
        self,
        fact_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Convert a fact into a JSON-compatible dictionary.
        """

        fact = self.get(
            fact_id
        )

        if fact is None:
            return None

        return {
            "fact_id": fact.fact_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "confidence": fact.confidence,
            "source": fact.source,
            "user_id": fact.user_id,
            "project_id": fact.project_id,
            "task_id": fact.task_id,
            "tags": list(
                fact.tags
            ),
            "metadata": dict(
                fact.metadata
            ),
            "memory_id": fact.memory_id,
            "importance": (
                fact.importance.value
            ),
            "status": fact.status.value,
            "created_at": fact.created_at,
            "updated_at": fact.updated_at,
        }

    def export_all(
        self,
    ) -> List[dict[str, Any]]:
        """
        Export all semantic facts.
        """

        with self._lock:

            facts = list(
                self._facts.values()
            )

        return [
            {
                "fact_id": fact.fact_id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "confidence": fact.confidence,
                "source": fact.source,
                "user_id": fact.user_id,
                "project_id": fact.project_id,
                "task_id": fact.task_id,
                "tags": list(
                    fact.tags
                ),
                "metadata": dict(
                    fact.metadata
                ),
                "memory_id": fact.memory_id,
                "importance": (
                    fact.importance.value
                ),
                "status": fact.status.value,
                "created_at": fact.created_at,
                "updated_at": fact.updated_at,
            }
            for fact in facts
        ]

    # ======================================================================
    # STATISTICS
    # ======================================================================

    def statistics(self) -> dict[str, Any]:
        """
        Return semantic-memory statistics.
        """

        with self._lock:

            facts = list(
                self._facts.values()
            )

        active = sum(
            1
            for fact in facts
            if fact.status
            == MemoryStatus.ACTIVE
        )

        archived = sum(
            1
            for fact in facts
            if fact.status
            == MemoryStatus.ARCHIVED
        )

        deleted = sum(
            1
            for fact in facts
            if fact.status
            == MemoryStatus.DELETED
        )

        linked = sum(
            1
            for fact in facts
            if fact.memory_id
        )

        return {
            "total": len(
                facts
            ),
            "active": active,
            "archived": archived,
            "deleted": deleted,
            "linked_memories": linked,
        }

    def health(self) -> dict[str, Any]:
        """
        Return health information for RENIX diagnostics.
        """

        return {
            "status": "healthy",
            **self.statistics(),
        }

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================

    @staticmethod
    def _new_id() -> str:
        return (
            "fact_"
            + uuid.uuid4().hex
        )

    @staticmethod
    def _new_memory_id() -> str:
        return (
            "semantic_"
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
    def _clamp_confidence(
        confidence: float,
    ) -> float:

        try:
            value = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    @staticmethod
    def _clean_list(
        values: Optional[
            Iterable[str]
        ],
    ) -> List[str]:

        if values is None:
            return []

        result = []

        for value in values:

            value = str(
                value
            ).strip()

            if value:
                result.append(
                    value
                )

        return list(
            dict.fromkeys(
                result
            )
        )

    @staticmethod
    def _merge_unique(
        first: Iterable[str],
        second: Iterable[str],
    ) -> List[str]:

        return list(
            dict.fromkeys(
                [
                    *(
                        str(value)
                        for value in first
                    ),
                    *(
                        str(value)
                        for value in second
                    ),
                ]
            )
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set[str]:

        normalized = (
            str(text)
            .casefold()
        )

        for char in (
            ",",
            ".",
            "!",
            "?",
            ":",
            ";",
            "/",
            "\\",
            "-",
            "_",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "'",
            '"',
        ):
            normalized = (
                normalized.replace(
                    char,
                    " ",
                )
            )

        return {
            token
            for token
            in normalized.split()
            if len(token) >= 2
        }

    @staticmethod
    def _timestamp(
        value: Optional[str],
    ) -> float:

        if not value:
            return 0.0

        try:

            return datetime.fromisoformat(
                str(value).replace(
                    "Z",
                    "+00:00",
                )
            ).timestamp()

        except (
            ValueError,
            TypeError,
        ):
            return 0.0

    @staticmethod
    def _importance_rank(
        importance: MemoryImportance,
    ) -> int:

        ranks = {
            MemoryImportance.LOW: 0,
            MemoryImportance.NORMAL: 1,
            MemoryImportance.HIGH: 2,
            MemoryImportance.CRITICAL: 3,
        }

        return ranks.get(
            importance,
            0,
        )

    @classmethod
    def _sort_facts(
        cls,
        facts: List[SemanticFact],
    ) -> List[SemanticFact]:

        return sorted(
            facts,
            key=lambda fact: (
                cls._importance_rank(
                    fact.importance
                ),
                fact.confidence,
                cls._timestamp(
                    fact.updated_at
                ),
            ),
            reverse=True,
        )


__all__ = [
    "SemanticFact",
    "SemanticMemory",
    "SemanticMemoryError",
]


