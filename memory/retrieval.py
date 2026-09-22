"""
RENIX AI - Memory Retrieval Engine

Responsible for retrieving relevant memories from RENIX memory.

Features:
- Semantic vector retrieval
- Keyword/text matching
- Metadata filtering
- Memory-type filtering
- Task/project/conversation filtering
- Hybrid ranking
- Recency weighting
- Importance weighting
- Deduplication
- Context assembly
- Configurable top-k retrieval
- Graceful operation when embeddings are unavailable

This module works with:
    memory_manager.py
    memory_store.py
    embeddings.py
    vector_store.py
    memory_search.py
    short_term.py
    long_term.py
    episodic.py
    semantic.py
    procedural.py
    project_memory.py
    task_memory.py
    preference_memory.py
    conversation_memory.py
"""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from .embeddings import (
        EmbeddingEngine,
        get_embedding_engine,
        cosine_similarity,
    )
except ImportError:
    from embeddings import (
        EmbeddingEngine,
        get_embedding_engine,
        cosine_similarity,
    )

try:
    from .vector_store import (
        VectorRecord,
        VectorStore,
    )
except ImportError:
    from vector_store import (
        VectorRecord,
        VectorStore,
    )


logger = logging.getLogger(
    "RENIX.memory.retrieval"
)


# ============================================================================
# Utility functions
# ============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(
    value: Any,
) -> Optional[datetime]:
    """Safely parse an ISO timestamp."""

    if value is None:
        return None

    if isinstance(value, datetime):
        result = value

    else:
        try:
            text = str(value).strip()

            if not text:
                return None

            result = datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            )

        except (
            ValueError,
            TypeError,
        ):
            return None

    if result.tzinfo is None:
        result = result.replace(
            tzinfo=timezone.utc
        )

    return result.astimezone(
        timezone.utc
    )


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def _normalize_text(
    text: Any,
) -> str:
    """Normalize text for keyword matching."""

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokenize(
    text: str,
) -> List[str]:
    """Tokenize text for lightweight lexical matching."""

    normalized = _normalize_text(
        text
    )

    return re.findall(
        r"[a-zA-Z0-9_]+",
        normalized,
    )


def _unique(
    values: Iterable[str],
) -> List[str]:
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


# ============================================================================
# Retrieval configuration
# ============================================================================

@dataclass
class RetrievalConfig:
    """
    Configuration for RENIX memory retrieval.

    Scores are combined approximately as:

        semantic * semantic_weight
        + lexical * lexical_weight
        + recency * recency_weight
        + importance * importance_weight
        + metadata * metadata_weight
    """

    default_top_k: int = 8

    semantic_weight: float = 0.55
    lexical_weight: float = 0.20
    recency_weight: float = 0.10
    importance_weight: float = 0.10
    metadata_weight: float = 0.05

    minimum_score: float = 0.05

    recency_half_life_hours: float = 168.0

    deduplicate: bool = True

    include_archived: bool = False

    max_context_memories: int = 12

    semantic_enabled: bool = True
    lexical_enabled: bool = True

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
    ) -> "RetrievalConfig":
        if not data:
            return cls()

        config = cls()

        for key, value in data.items():
            if not hasattr(
                config,
                key,
            ):
                continue

            try:
                setattr(
                    config,
                    key,
                    value,
                )
            except Exception:
                continue

        config.default_top_k = max(
            1,
            int(config.default_top_k),
        )

        config.max_context_memories = max(
            1,
            int(config.max_context_memories),
        )

        return config


# ============================================================================
# Retrieval result
# ============================================================================

@dataclass
class RetrievalResult:
    """Represents one retrieved memory."""

    memory_id: Optional[str]

    vector_id: Optional[str]

    text: str

    memory_type: str

    score: float

    semantic_score: float = 0.0
    lexical_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    metadata_score: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    task_id: Optional[str] = None
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    record: Optional[VectorRecord] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "vector_id": self.vector_id,
            "text": self.text,
            "memory_type": self.memory_type,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
            "recency_score": self.recency_score,
            "importance_score": self.importance_score,
            "metadata_score": self.metadata_score,
            "metadata": self.metadata,
            "task_id": self.task_id,
            "project_id": self.project_id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ============================================================================
# Retrieval Engine
# ============================================================================

class RetrievalEngine:
    """
    Main retrieval engine for RENIX.

    Example:

        retrieval = RetrievalEngine(
            vector_store=vector_store
        )

        results = retrieval.retrieve(
            "What was my previous project?"
        )
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:

        self.config = (
            config
            or RetrievalConfig()
        )

        self.vector_store = (
            vector_store
            or VectorStore()
        )

        self.embedding_engine = (
            embedding_engine
            or get_embedding_engine()
        )

        self._lock = threading.RLock()

    # ------------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        memory_types: Optional[
            Sequence[str]
        ] = None,
        filters: Optional[
            Dict[str, Any]
        ] = None,
        min_score: Optional[float] = None,
        include_archived: Optional[bool] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories relevant to a natural-language query.
        """

        query = str(query).strip()

        if not query:
            return []

        top_k = (
            top_k
            if top_k is not None
            else self.config.default_top_k
        )

        top_k = max(
            1,
            int(top_k),
        )

        if min_score is None:
            min_score = (
                self.config.minimum_score
            )

        if include_archived is None:
            include_archived = (
                self.config.include_archived
            )

        filters = dict(
            filters or {}
        )

        if memory_types:
            filters[
                "memory_type"
            ] = list(memory_types)

        query_embedding: Optional[
            List[float]
        ] = None

        if self.config.semantic_enabled:
            try:
                query_embedding = (
                    self.embedding_engine.embed(
                        query
                    )
                )

            except Exception as exc:
                logger.warning(
                    "Semantic embedding failed: %s",
                    exc,
                )

        candidates = self._collect_candidates(
            query=query,
            query_embedding=query_embedding,
            filters=filters,
            include_archived=include_archived,
        )

        results = self._rank_candidates(
            query=query,
            query_embedding=query_embedding,
            candidates=candidates,
        )

        results = [
            result
            for result in results
            if result.score >= min_score
        ]

        if self.config.deduplicate:
            results = self._deduplicate(
                results
            )

        return results[:top_k]

    # Compatibility aliases.

    search = retrieve
    query = retrieve
    recall = retrieve

    # ------------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------------

    def _collect_candidates(
        self,
        query: str,
        query_embedding: Optional[
            Sequence[float]
        ],
        filters: Dict[str, Any],
        include_archived: bool,
    ) -> List[VectorRecord]:

        # When semantic search is available, use vector store as the
        # primary candidate source.
        if query_embedding is not None:

            try:
                search_results = (
                    self.vector_store.search(
                        query_embedding=query_embedding,
                        top_k=max(
                            self.config.default_top_k
                            * 5,
                            50,
                        ),
                        min_score=-1.0,
                        filters=filters,
                        include_archived=include_archived,
                    )
                )

                return [
                    item["record"]
                    for item in search_results
                    if isinstance(
                        item.get("record"),
                        VectorRecord,
                    )
                ]

            except Exception as exc:
                logger.warning(
                    "Vector search failed: %s",
                    exc,
                )

        # Fallback to direct scanning.
        records = self.vector_store.all(
            include_archived=include_archived
        )

        return [
            record
            for record in records
            if self._matches_filters(
                record,
                filters,
            )
        ]

    # ------------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------------

    def _rank_candidates(
        self,
        query: str,
        query_embedding: Optional[
            Sequence[float]
        ],
        candidates: Sequence[
            VectorRecord
        ],
    ) -> List[RetrievalResult]:

        query_tokens = set(
            _tokenize(query)
        )

        results: List[
            RetrievalResult
        ] = []

        for record in candidates:

            semantic_score = 0.0

            if query_embedding is not None:
                try:
                    semantic_score = _clamp(
                        (
                            cosine_similarity(
                                query_embedding,
                                record.embedding,
                            )
                            + 1.0
                        )
                        / 2.0
                    )

                except Exception:
                    semantic_score = 0.0

            lexical_score = 0.0

            if self.config.lexical_enabled:
                lexical_score = (
                    self._lexical_score(
                        query_tokens,
                        record.text,
                    )
                )

            recency_score = (
                self._recency_score(
                    record
                )
            )

            importance_score = _clamp(
                float(
                    record.importance
                )
            )

            metadata_score = (
                self._metadata_score(
                    query,
                    record,
                )
            )

            score = (
                semantic_score
                * self.config.semantic_weight
                + lexical_score
                * self.config.lexical_weight
                + recency_score
                * self.config.recency_weight
                + importance_score
                * self.config.importance_weight
                + metadata_score
                * self.config.metadata_weight
            )

            result = RetrievalResult(
                memory_id=record.memory_id,
                vector_id=record.vector_id,
                text=record.text,
                memory_type=record.memory_type,
                score=_clamp(score),
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                recency_score=recency_score,
                importance_score=importance_score,
                metadata_score=metadata_score,
                metadata=dict(
                    record.metadata
                ),
                task_id=record.task_id,
                project_id=record.project_id,
                conversation_id=(
                    record.conversation_id
                ),
                session_id=record.session_id,
                created_at=record.created_at,
                updated_at=record.updated_at,
                record=record,
            )

            results.append(result)

        results.sort(
            key=lambda item: (
                item.score,
                item.importance_score,
                item.recency_score,
            ),
            reverse=True,
        )

        return results

    # ------------------------------------------------------------------------
    # Lexical scoring
    # ------------------------------------------------------------------------

    def _lexical_score(
        self,
        query_tokens: set[str],
        text: str,
    ) -> float:

        if not query_tokens:
            return 0.0

        text_tokens = set(
            _tokenize(text)
        )

        if not text_tokens:
            return 0.0

        intersection = (
            query_tokens
            & text_tokens
        )

        overlap = (
            len(intersection)
            / len(query_tokens)
        )

        # Bonus for exact phrase presence.
        query_text = " ".join(
            sorted(query_tokens)
        )

        normalized_text = (
            _normalize_text(text)
        )

        phrase_bonus = (
            0.15
            if query_text
            and query_text in normalized_text
            else 0.0
        )

        return _clamp(
            overlap
            + phrase_bonus
        )

    # ------------------------------------------------------------------------
    # Recency scoring
    # ------------------------------------------------------------------------

    def _recency_score(
        self,
        record: VectorRecord,
    ) -> float:

        timestamp = (
            _parse_datetime(
                record.updated_at
            )
            or _parse_datetime(
                record.created_at
            )
        )

        if timestamp is None:
            return 0.5

        age_hours = max(
            0.0,
            (
                _utc_now()
                - timestamp
            ).total_seconds()
            / 3600.0,
        )

        half_life = max(
            1.0,
            float(
                self.config
                .recency_half_life_hours
            ),
        )

        return _clamp(
            math.pow(
                0.5,
                age_hours / half_life,
            )
        )

    # ------------------------------------------------------------------------
    # Metadata scoring
    # ------------------------------------------------------------------------

    def _metadata_score(
        self,
        query: str,
        record: VectorRecord,
    ) -> float:

        query_normalized = (
            _normalize_text(query)
        )

        score = 0.0

        metadata_values = [
            record.memory_type,
            record.task_id,
            record.project_id,
            record.conversation_id,
            record.session_id,
        ]

        metadata_values.extend(
            str(value)
            for value in record.metadata.values()
        )

        for value in metadata_values:
            if not value:
                continue

            normalized = (
                _normalize_text(value)
            )

            if not normalized:
                continue

            if normalized == query_normalized:
                score = max(
                    score,
                    1.0,
                )

            elif query_normalized in normalized:
                score = max(
                    score,
                    0.8,
                )

        return score

    # ------------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------------

    def _matches_filters(
        self,
        record: VectorRecord,
        filters: Dict[str, Any],
    ) -> bool:

        for key, expected in filters.items():

            if key == "memory_type":
                actual = record.memory_type

            elif key == "memory_id":
                actual = record.memory_id

            elif key == "task_id":
                actual = record.task_id

            elif key == "project_id":
                actual = record.project_id

            elif key == "conversation_id":
                actual = record.conversation_id

            elif key == "session_id":
                actual = record.session_id

            elif key == "archived":
                actual = record.archived

            else:
                actual = record.metadata.get(
                    key
                )

            if isinstance(
                expected,
                (list, tuple, set),
            ):

                if isinstance(
                    actual,
                    (list, tuple, set),
                ):
                    if not any(
                        item in actual
                        for item in expected
                    ):
                        return False

                elif actual not in expected:
                    return False

            elif actual != expected:
                return False

        return True

    # ------------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------------

    def _deduplicate(
        self,
        results: Sequence[
            RetrievalResult
        ],
    ) -> List[RetrievalResult]:

        seen_ids = set()
        seen_texts = set()

        unique_results = []

        for result in results:

            identifier = (
                result.memory_id
                or result.vector_id
            )

            normalized_text = (
                _normalize_text(
                    result.text
                )
            )

            if identifier in seen_ids:
                continue

            if (
                normalized_text
                and normalized_text
                in seen_texts
            ):
                continue

            seen_ids.add(identifier)

            if normalized_text:
                seen_texts.add(
                    normalized_text
                )

            unique_results.append(
                result
            )

        return unique_results

    # ------------------------------------------------------------------------
    # Context generation
    # ------------------------------------------------------------------------

    def build_context(
        self,
        query: str,
        results: Optional[
            Sequence[RetrievalResult]
        ] = None,
        max_memories: Optional[int] = None,
    ) -> str:
        """
        Convert retrieved memories into context suitable for an LLM.
        """

        if results is None:
            results = self.retrieve(
                query
            )

        if max_memories is None:
            max_memories = (
                self.config
                .max_context_memories
            )

        max_memories = max(
            1,
            int(max_memories),
        )

        selected = list(
            results
        )[:max_memories]

        if not selected:
            return ""

        lines = [
            "RELEVANT RENIX MEMORY:"
        ]

        for index, result in enumerate(
            selected,
            start=1,
        ):

            memory_type = (
                result.memory_type
            )

            text = result.text.strip()

            if not text:
                continue

            lines.append(
                f"{index}. "
                f"[{memory_type}] "
                f"{text}"
            )

        return "\n".join(
            lines
        )

    # ------------------------------------------------------------------------
    # Structured context
    # ------------------------------------------------------------------------

    def build_structured_context(
        self,
        results: Sequence[
            RetrievalResult
        ],
    ) -> List[Dict[str, Any]]:
        """
        Return retrieved memories in structured form for the orchestrator.
        """

        return [
            result.to_dict()
            for result in results
        ]

    # ------------------------------------------------------------------------
    # Specialized retrieval
    # ------------------------------------------------------------------------

    def retrieve_for_task(
        self,
        query: str,
        task_id: str,
        top_k: int = 8,
    ) -> List[RetrievalResult]:

        return self.retrieve(
            query=query,
            top_k=top_k,
            filters={
                "task_id": task_id,
            },
        )

    def retrieve_for_project(
        self,
        query: str,
        project_id: str,
        top_k: int = 8,
    ) -> List[RetrievalResult]:

        return self.retrieve(
            query=query,
            top_k=top_k,
            filters={
                "project_id": project_id,
            },
        )

    def retrieve_for_conversation(
        self,
        query: str,
        conversation_id: str,
        top_k: int = 8,
    ) -> List[RetrievalResult]:

        return self.retrieve(
            query=query,
            top_k=top_k,
            filters={
                "conversation_id":
                    conversation_id,
            },
        )

    def retrieve_by_type(
        self,
        query: str,
        memory_type: str,
        top_k: int = 8,
    ) -> List[RetrievalResult]:

        return self.retrieve(
            query=query,
            top_k=top_k,
            memory_types=[
                memory_type
            ],
        )

    # ------------------------------------------------------------------------
    # Memory timeline
    # ------------------------------------------------------------------------

    def recent_memories(
        self,
        limit: int = 10,
        memory_type: Optional[str] = None,
    ) -> List[RetrievalResult]:

        records = self.vector_store.all()

        if memory_type:
            records = [
                record
                for record in records
                if record.memory_type
                == memory_type
            ]

        records.sort(
            key=lambda record: (
                _parse_datetime(
                    record.updated_at
                )
                or datetime.min.replace(
                    tzinfo=timezone.utc
                )
            ),
            reverse=True,
        )

        return [
            self._record_to_result(
                record
            )
            for record in records[
                : max(1, int(limit))
            ]
        ]

    # ------------------------------------------------------------------------
    # Important memories
    # ------------------------------------------------------------------------

    def important_memories(
        self,
        limit: int = 10,
        minimum_importance: float = 0.7,
    ) -> List[RetrievalResult]:

        records = [
            record
            for record in self.vector_store.all()
            if record.importance
            >= minimum_importance
        ]

        records.sort(
            key=lambda record:
            record.importance,
            reverse=True,
        )

        return [
            self._record_to_result(
                record
            )
            for record in records[
                : max(1, int(limit))
            ]
        ]

    # ------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------

    @staticmethod
    def _record_to_result(
        record: VectorRecord,
    ) -> RetrievalResult:

        return RetrievalResult(
            memory_id=record.memory_id,
            vector_id=record.vector_id,
            text=record.text,
            memory_type=record.memory_type,
            score=record.importance,
            semantic_score=0.0,
            lexical_score=0.0,
            recency_score=0.0,
            importance_score=record.importance,
            metadata_score=0.0,
            metadata=dict(
                record.metadata
            ),
            task_id=record.task_id,
            project_id=record.project_id,
            conversation_id=(
                record.conversation_id
            ),
            session_id=record.session_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
            record=record,
        )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:

        vector_stats = (
            self.vector_store.statistics()
        )

        return {
            "vector_store":
                vector_stats,

            "default_top_k":
                self.config.default_top_k,

            "semantic_weight":
                self.config.semantic_weight,

            "lexical_weight":
                self.config.lexical_weight,

            "recency_weight":
                self.config.recency_weight,

            "importance_weight":
                self.config.importance_weight,

            "metadata_weight":
                self.config.metadata_weight,

            "minimum_score":
                self.config.minimum_score,

            "embedding":
                self.embedding_engine.info(),
        }


# ============================================================================
# Module-level singleton
# ============================================================================

_default_retrieval_engine: Optional[
    RetrievalEngine
] = None

_default_lock = threading.Lock()


def get_retrieval_engine() -> RetrievalEngine:
    """
    Return the shared RENIX retrieval engine.
    """

    global _default_retrieval_engine

    if _default_retrieval_engine is None:

        with _default_lock:

            if _default_retrieval_engine is None:
                _default_retrieval_engine = (
                    RetrievalEngine()
                )

    return _default_retrieval_engine


# ============================================================================
# Convenience functions
# ============================================================================

def retrieve(
    query: str,
    top_k: int = 8,
    memory_types: Optional[
        Sequence[str]
    ] = None,
    filters: Optional[
        Dict[str, Any]
    ] = None,
) -> List[RetrievalResult]:

    return get_retrieval_engine().retrieve(
        query=query,
        top_k=top_k,
        memory_types=memory_types,
        filters=filters,
    )


def build_context(
    query: str,
    top_k: int = 8,
) -> str:

    engine = (
        get_retrieval_engine()
    )

    results = engine.retrieve(
        query=query,
        top_k=top_k,
    )

    return engine.build_context(
        query=query,
        results=results,
    )


# ============================================================================
# Compatibility aliases
# ============================================================================

MemoryRetrieval = RetrievalEngine
MemoryRetriever = RetrievalEngine
Retriever = RetrievalEngine


__all__ = [
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievalEngine",
    "MemoryRetrieval",
    "MemoryRetriever",
    "Retriever",
    "get_retrieval_engine",
    "retrieve",
    "build_context",
]


