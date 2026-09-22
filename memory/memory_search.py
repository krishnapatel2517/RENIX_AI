"""
RENIX AI - Memory Search

High-level search interface over RENIX memory.

Responsibilities:
- Natural-language memory search
- Semantic + lexical retrieval
- Search by memory type
- Search by project/task/conversation
- Exact phrase search
- Recent/important memory search
- Context generation for the LLM
- Search result formatting
- Safe fallback when advanced retrieval is unavailable

This module sits above:
    memory_manager.py
    memory_store.py
    embeddings.py
    retrieval.py
    vector_store.py

It should be used by the RENIX orchestrator whenever it needs to
"remember", "find", "recall", or inspect previous information.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
)

try:
    from .retrieval import (
        RetrievalEngine,
        RetrievalResult,
        get_retrieval_engine,
    )
except ImportError:
    from retrieval import (
        RetrievalEngine,
        RetrievalResult,
        get_retrieval_engine,
    )


logger = logging.getLogger(
    "RENIX.memory.memory_search"
)


# ============================================================================
# Search configuration
# ============================================================================

@dataclass
class MemorySearchConfig:
    """Configuration for high-level RENIX memory search."""

    default_limit: int = 8

    max_limit: int = 50

    min_score: float = 0.05

    include_context: bool = True

    context_limit: int = 8

    exact_match_boost: float = 0.20

    keyword_match_boost: float = 0.10

    deduplicate: bool = True

    @classmethod
    def from_dict(
        cls,
        data: Optional[
            Dict[str, Any]
        ],
    ) -> "MemorySearchConfig":

        config = cls()

        if not data:
            return config

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

        config.default_limit = max(
            1,
            int(
                config.default_limit
            ),
        )

        config.max_limit = max(
            config.default_limit,
            int(
                config.max_limit
            ),
        )

        config.context_limit = max(
            1,
            int(
                config.context_limit
            ),
        )

        return config


# ============================================================================
# Search result
# ============================================================================

@dataclass
class MemorySearchResult:
    """User-facing memory search result."""

    memory_id: Optional[str]

    text: str

    memory_type: str

    score: float

    semantic_score: float = 0.0

    lexical_score: float = 0.0

    recency_score: float = 0.0

    importance_score: float = 0.0

    metadata_score: float = 0.0

    exact_match: bool = False

    keyword_matches: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    task_id: Optional[str] = None

    project_id: Optional[str] = None

    conversation_id: Optional[str] = None

    session_id: Optional[str] = None

    created_at: Optional[str] = None

    updated_at: Optional[str] = None

    source: str = "memory"

    raw: Optional[
        RetrievalResult
    ] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "memory_type": self.memory_type,
            "score": self.score,
            "semantic_score":
                self.semantic_score,
            "lexical_score":
                self.lexical_score,
            "recency_score":
                self.recency_score,
            "importance_score":
                self.importance_score,
            "metadata_score":
                self.metadata_score,
            "exact_match":
                self.exact_match,
            "keyword_matches":
                list(
                    self.keyword_matches
                ),
            "metadata":
                dict(self.metadata),
            "task_id":
                self.task_id,
            "project_id":
                self.project_id,
            "conversation_id":
                self.conversation_id,
            "session_id":
                self.session_id,
            "created_at":
                self.created_at,
            "updated_at":
                self.updated_at,
            "source":
                self.source,
        }


# ============================================================================
# Search response
# ============================================================================

@dataclass
class MemorySearchResponse:
    """Complete response returned by the search layer."""

    query: str

    results: List[
        MemorySearchResult
    ] = field(
        default_factory=list
    )

    context: str = ""

    total: int = 0

    search_type: str = "hybrid"

    filters: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "query": self.query,
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "context": self.context,
            "total": self.total,
            "search_type":
                self.search_type,
            "filters":
                dict(self.filters),
        }


# ============================================================================
# Memory Search Engine
# ============================================================================

class MemorySearchEngine:
    """
    High-level memory search interface for RENIX.

    Example:

        search = MemorySearchEngine()

        response = search.search(
            "What did I say about RENIX?"
        )

        print(response.context)
    """

    def __init__(
        self,
        retrieval_engine: Optional[
            RetrievalEngine
        ] = None,
        config: Optional[
            MemorySearchConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or MemorySearchConfig()
        )

        self.retrieval_engine = (
            retrieval_engine
            or get_retrieval_engine()
        )

        self._lock = threading.RLock()

    # ------------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        memory_types: Optional[
            Sequence[str]
        ] = None,
        task_id: Optional[str] = None,
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[
            Dict[str, Any]
        ] = None,
        min_score: Optional[float] = None,
        include_context: Optional[
            bool
        ] = None,
    ) -> MemorySearchResponse:
        """
        Perform hybrid semantic/lexical memory search.
        """

        query = str(query).strip()

        if not query:

            return MemorySearchResponse(
                query="",
                results=[],
                context="",
                total=0,
                search_type="hybrid",
            )

        if limit is None:
            limit = (
                self.config.default_limit
            )

        limit = max(
            1,
            min(
                int(limit),
                self.config.max_limit,
            ),
        )

        if min_score is None:
            min_score = (
                self.config.min_score
            )

        if include_context is None:
            include_context = (
                self.config.include_context
            )

        final_filters = dict(
            filters or {}
        )

        if task_id is not None:
            final_filters[
                "task_id"
            ] = task_id

        if project_id is not None:
            final_filters[
                "project_id"
            ] = project_id

        if conversation_id is not None:
            final_filters[
                "conversation_id"
            ] = conversation_id

        if session_id is not None:
            final_filters[
                "session_id"
            ] = session_id

        retrieval_results = (
            self.retrieval_engine.retrieve(
                query=query,
                top_k=limit * 2,
                memory_types=memory_types,
                filters=final_filters,
                min_score=min_score,
            )
        )

        results = (
            self._convert_results(
                query=query,
                results=retrieval_results,
            )
        )

        results = self._apply_query_boosts(
            query=query,
            results=results,
        )

        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        if self.config.deduplicate:
            results = self._deduplicate(
                results
            )

        results = results[
            :limit
        ]

        context = ""

        if include_context:
            context = (
                self._build_context(
                    query=query,
                    results=results,
                )
            )

        return MemorySearchResponse(
            query=query,
            results=results,
            context=context,
            total=len(results),
            search_type="hybrid",
            filters=final_filters,
        )

    # Compatibility aliases.

    query = search

    recall = search

    find = search

    # ------------------------------------------------------------------------
    # Semantic-only search
    # ------------------------------------------------------------------------

    def semantic_search(
        self,
        query: str,
        limit: int = 8,
        memory_types: Optional[
            Sequence[str]
        ] = None,
    ) -> MemorySearchResponse:

        results = (
            self.retrieval_engine.retrieve(
                query=query,
                top_k=limit,
                memory_types=memory_types,
            )
        )

        converted = [
            self._convert_result(
                result
            )
            for result in results
        ]

        return MemorySearchResponse(
            query=query,
            results=converted,
            context=self._build_context(
                query=query,
                results=converted,
            ),
            total=len(converted),
            search_type="semantic",
        )

    # ------------------------------------------------------------------------
    # Exact text search
    # ------------------------------------------------------------------------

    def exact_search(
        self,
        phrase: str,
        limit: int = 20,
        memory_types: Optional[
            Sequence[str]
        ] = None,
    ) -> MemorySearchResponse:
        """
        Search for an exact phrase.

        This still uses the retrieval layer to obtain candidate records,
        then performs exact matching locally.
        """

        phrase = str(
            phrase
        ).strip()

        if not phrase:
            return MemorySearchResponse(
                query="",
                results=[],
                search_type="exact",
            )

        records = (
            self.retrieval_engine.vector_store.all()
        )

        normalized_phrase = (
            self._normalize(
                phrase
            )
        )

        results = []

        allowed_types = (
            set(memory_types)
            if memory_types
            else None
        )

        for record in records:

            if (
                allowed_types is not None
                and record.memory_type
                not in allowed_types
            ):
                continue

            text = self._normalize(
                record.text
            )

            if (
                normalized_phrase
                not in text
            ):
                continue

            result = (
                self._record_to_search_result(
                    record
                )
            )

            result.exact_match = True

            result.score = min(
                1.0,
                result.score
                + self.config
                .exact_match_boost,
            )

            results.append(
                result
            )

        results.sort(
            key=lambda item:
            item.score,
            reverse=True,
        )

        results = results[
            : max(1, int(limit))
        ]

        return MemorySearchResponse(
            query=phrase,
            results=results,
            context=self._build_context(
                query=phrase,
                results=results,
            ),
            total=len(results),
            search_type="exact",
        )

    # ------------------------------------------------------------------------
    # Keyword search
    # ------------------------------------------------------------------------

    def keyword_search(
        self,
        query: str,
        limit: int = 8,
        memory_types: Optional[
            Sequence[str]
        ] = None,
    ) -> MemorySearchResponse:

        query_tokens = set(
            self._tokenize(query)
        )

        records = (
            self.retrieval_engine.vector_store.all()
        )

        allowed_types = (
            set(memory_types)
            if memory_types
            else None
        )

        results = []

        for record in records:

            if (
                allowed_types is not None
                and record.memory_type
                not in allowed_types
            ):
                continue

            text_tokens = set(
                self._tokenize(
                    record.text
                )
            )

            matches = sorted(
                query_tokens
                & text_tokens
            )

            if not matches:
                continue

            result = (
                self._record_to_search_result(
                    record
                )
            )

            result.keyword_matches = (
                matches
            )

            result.lexical_score = (
                len(matches)
                / max(
                    1,
                    len(query_tokens),
                )
            )

            result.score = min(
                1.0,
                result.score
                + (
                    result.lexical_score
                    * self.config
                    .keyword_match_boost
                ),
            )

            results.append(
                result
            )

        results.sort(
            key=lambda item:
            item.score,
            reverse=True,
        )

        results = results[
            : max(1, int(limit))
        ]

        return MemorySearchResponse(
            query=query,
            results=results,
            context=self._build_context(
                query=query,
                results=results,
            ),
            total=len(results),
            search_type="keyword",
        )

    # ------------------------------------------------------------------------
    # Specialized searches
    # ------------------------------------------------------------------------

    def search_task(
        self,
        query: str,
        task_id: str,
        limit: int = 8,
    ) -> MemorySearchResponse:

        return self.search(
            query=query,
            limit=limit,
            task_id=task_id,
        )

    def search_project(
        self,
        query: str,
        project_id: str,
        limit: int = 8,
    ) -> MemorySearchResponse:

        return self.search(
            query=query,
            limit=limit,
            project_id=project_id,
        )

    def search_conversation(
        self,
        query: str,
        conversation_id: str,
        limit: int = 8,
    ) -> MemorySearchResponse:

        return self.search(
            query=query,
            limit=limit,
            conversation_id=conversation_id,
        )

    def search_session(
        self,
        query: str,
        session_id: str,
        limit: int = 8,
    ) -> MemorySearchResponse:

        return self.search(
            query=query,
            limit=limit,
            session_id=session_id,
        )

    # ------------------------------------------------------------------------
    # Search by memory type
    # ------------------------------------------------------------------------

    def search_type(
        self,
        query: str,
        memory_type: str,
        limit: int = 8,
    ) -> MemorySearchResponse:

        return self.search(
            query=query,
            limit=limit,
            memory_types=[
                memory_type
            ],
        )

    # ------------------------------------------------------------------------
    # Recent memory
    # ------------------------------------------------------------------------

    def recent(
        self,
        limit: int = 10,
        memory_type: Optional[
            str
        ] = None,
    ) -> MemorySearchResponse:

        results = (
            self.retrieval_engine
            .recent_memories(
                limit=limit,
                memory_type=memory_type,
            )
        )

        converted = [
            self._convert_result(
                result
            )
            for result in results
        ]

        return MemorySearchResponse(
            query="",
            results=converted,
            context=self._build_context(
                query="recent memory",
                results=converted,
            ),
            total=len(converted),
            search_type="recent",
        )

    # ------------------------------------------------------------------------
    # Important memory
    # ------------------------------------------------------------------------

    def important(
        self,
        limit: int = 10,
        minimum_importance: float = 0.7,
    ) -> MemorySearchResponse:

        results = (
            self.retrieval_engine
            .important_memories(
                limit=limit,
                minimum_importance=(
                    minimum_importance
                ),
            )
        )

        converted = [
            self._convert_result(
                result
            )
            for result in results
        ]

        return MemorySearchResponse(
            query="important memory",
            results=converted,
            context=self._build_context(
                query="important memory",
                results=converted,
            ),
            total=len(converted),
            search_type="important",
        )

    # ------------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------------

    def build_context(
        self,
        query: str,
        limit: int = 8,
    ) -> str:

        response = self.search(
            query=query,
            limit=limit,
            include_context=True,
        )

        return response.context

    def _build_context(
        self,
        query: str,
        results: Sequence[
            MemorySearchResult
        ],
    ) -> str:

        if not results:
            return ""

        lines = [
            "RENIX MEMORY CONTEXT"
        ]

        lines.append(
            "The following memories may be "
            "relevant to the current request:"
        )

        for index, result in enumerate(
            results[
                : self.config.context_limit
            ],
            start=1,
        ):

            text = result.text.strip()

            if not text:
                continue

            metadata = []

            if result.memory_type:
                metadata.append(
                    result.memory_type
                )

            if result.project_id:
                metadata.append(
                    f"project={result.project_id}"
                )

            if result.task_id:
                metadata.append(
                    f"task={result.task_id}"
                )

            if metadata:
                prefix = (
                    f"[{', '.join(metadata)}]"
                )
            else:
                prefix = "[memory]"

            lines.append(
                f"{index}. "
                f"{prefix} "
                f"{text}"
            )

        return "\n".join(
            lines
        )

    # ------------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------------

    def _apply_query_boosts(
        self,
        query: str,
        results: Sequence[
            MemorySearchResult
        ],
    ) -> List[MemorySearchResult]:

        normalized_query = (
            self._normalize(
                query
            )
        )

        query_tokens = set(
            self._tokenize(query)
        )

        boosted = []

        for result in results:

            normalized_text = (
                self._normalize(
                    result.text
                )
            )

            score = result.score

            if (
                normalized_query
                and normalized_query
                in normalized_text
            ):
                result.exact_match = True

                score += (
                    self.config
                    .exact_match_boost
                )

            text_tokens = set(
                self._tokenize(
                    result.text
                )
            )

            matches = sorted(
                query_tokens
                & text_tokens
            )

            if matches:
                result.keyword_matches = (
                    matches
                )

                score += (
                    len(matches)
                    / max(
                        1,
                        len(query_tokens),
                    )
                ) * self.config.keyword_match_boost

            result.score = min(
                1.0,
                score,
            )

            boosted.append(
                result
            )

        return boosted

    # ------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------

    def _convert_results(
        self,
        query: str,
        results: Sequence[
            RetrievalResult
        ],
    ) -> List[MemorySearchResult]:

        converted = []

        for result in results:

            item = (
                self._convert_result(
                    result
                )
            )

            item.keyword_matches = (
                self._find_keyword_matches(
                    query,
                    result.text,
                )
            )

            converted.append(
                item
            )

        return converted

    @staticmethod
    def _convert_result(
        result: RetrievalResult,
    ) -> MemorySearchResult:

        return MemorySearchResult(
            memory_id=result.memory_id,
            text=result.text,
            memory_type=result.memory_type,
            score=result.score,
            semantic_score=(
                result.semantic_score
            ),
            lexical_score=(
                result.lexical_score
            ),
            recency_score=(
                result.recency_score
            ),
            importance_score=(
                result.importance_score
            ),
            metadata_score=(
                result.metadata_score
            ),
            metadata=dict(
                result.metadata
            ),
            task_id=result.task_id,
            project_id=result.project_id,
            conversation_id=(
                result.conversation_id
            ),
            session_id=result.session_id,
            created_at=result.created_at,
            updated_at=result.updated_at,
            source="memory",
            raw=result,
        )

    @staticmethod
    def _record_to_search_result(
        record: Any,
    ) -> MemorySearchResult:

        importance = float(
            getattr(
                record,
                "importance",
                0.0,
            )
        )

        return MemorySearchResult(
            memory_id=getattr(
                record,
                "memory_id",
                None,
            ),
            text=str(
                getattr(
                    record,
                    "text",
                    "",
                )
            ),
            memory_type=str(
                getattr(
                    record,
                    "memory_type",
                    "unknown",
                )
            ),
            score=importance,
            importance_score=importance,
            metadata=dict(
                getattr(
                    record,
                    "metadata",
                    {},
                )
                or {}
            ),
            task_id=getattr(
                record,
                "task_id",
                None,
            ),
            project_id=getattr(
                record,
                "project_id",
                None,
            ),
            conversation_id=getattr(
                record,
                "conversation_id",
                None,
            ),
            session_id=getattr(
                record,
                "session_id",
                None,
            ),
            created_at=getattr(
                record,
                "created_at",
                None,
            ),
            updated_at=getattr(
                record,
                "updated_at",
                None,
            ),
            source="memory",
        )

    # ------------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------------

    @staticmethod
    def _normalize(
        text: Any,
    ) -> str:

        if text is None:
            return ""

        text = str(
            text
        ).lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _tokenize(
        text: Any,
    ) -> List[str]:

        normalized = (
            MemorySearchEngine
            ._normalize(text)
        )

        return re.findall(
            r"[a-zA-Z0-9_]+",
            normalized,
        )

    def _find_keyword_matches(
        self,
        query: str,
        text: str,
    ) -> List[str]:

        query_tokens = set(
            self._tokenize(query)
        )

        text_tokens = set(
            self._tokenize(text)
        )

        return sorted(
            query_tokens
            & text_tokens
        )

    # ------------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------------

    def _deduplicate(
        self,
        results: Sequence[
            MemorySearchResult
        ],
    ) -> List[MemorySearchResult]:

        seen_ids = set()
        seen_texts = set()

        unique = []

        for result in results:

            memory_id = (
                result.memory_id
            )

            text_key = self._normalize(
                result.text
            )

            if (
                memory_id
                and memory_id
                in seen_ids
            ):
                continue

            if (
                text_key
                and text_key
                in seen_texts
            ):
                continue

            if memory_id:
                seen_ids.add(
                    memory_id
                )

            if text_key:
                seen_texts.add(
                    text_key
                )

            unique.append(
                result
            )

        return unique

    # ------------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------------

    def format_results(
        self,
        response: MemorySearchResponse,
        include_scores: bool = False,
    ) -> str:
        """
        Convert a search response into readable text.
        """

        if not response.results:
            return (
                "I couldn't find any relevant "
                "memory."
            )

        lines = []

        for index, result in enumerate(
            response.results,
            start=1,
        ):

            if include_scores:
                lines.append(
                    f"{index}. "
                    f"[{result.memory_type}] "
                    f"{result.text} "
                    f"(score={result.score:.3f})"
                )
            else:
                lines.append(
                    f"{index}. "
                    f"[{result.memory_type}] "
                    f"{result.text}"
                )

        return "\n".join(
            lines
        )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    def statistics(
        self,
    ) -> Dict[str, Any]:

        return {
            "default_limit":
                self.config.default_limit,
            "max_limit":
                self.config.max_limit,
            "min_score":
                self.config.min_score,
            "context_limit":
                self.config.context_limit,
            "deduplicate":
                self.config.deduplicate,
            "retrieval":
                self.retrieval_engine.statistics(),
        }


# ============================================================================
# Singleton
# ============================================================================

_default_search_engine: Optional[
    MemorySearchEngine
] = None

_default_lock = threading.Lock()


def get_memory_search() -> MemorySearchEngine:
    """Return the shared RENIX memory search engine."""

    global _default_search_engine

    if _default_search_engine is None:

        with _default_lock:

            if _default_search_engine is None:
                _default_search_engine = (
                    MemorySearchEngine()
                )

    return _default_search_engine


# ============================================================================
# Convenience API
# ============================================================================

def search_memory(
    query: str,
    limit: int = 8,
    memory_types: Optional[
        Sequence[str]
    ] = None,
    task_id: Optional[str] = None,
    project_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> MemorySearchResponse:

    return get_memory_search().search(
        query=query,
        limit=limit,
        memory_types=memory_types,
        task_id=task_id,
        project_id=project_id,
        conversation_id=conversation_id,
    )


def semantic_search(
    query: str,
    limit: int = 8,
) -> MemorySearchResponse:

    return get_memory_search().semantic_search(
        query=query,
        limit=limit,
    )


def exact_search(
    phrase: str,
    limit: int = 20,
) -> MemorySearchResponse:

    return get_memory_search().exact_search(
        phrase=phrase,
        limit=limit,
    )


def keyword_search(
    query: str,
    limit: int = 8,
) -> MemorySearchResponse:

    return get_memory_search().keyword_search(
        query=query,
        limit=limit,
    )


def memory_context(
    query: str,
    limit: int = 8,
) -> str:

    return get_memory_search().build_context(
        query=query,
        limit=limit,
    )


# ============================================================================
# Compatibility aliases
# ============================================================================

MemorySearch = MemorySearchEngine
MemorySearcher = MemorySearchEngine
SearchEngine = MemorySearchEngine


__all__ = [
    "MemorySearchConfig",
    "MemorySearchResult",
    "MemorySearchResponse",
    "MemorySearchEngine",
    "MemorySearch",
    "MemorySearcher",
    "SearchEngine",
    "get_memory_search",
    "search_memory",
    "semantic_search",
    "exact_search",
    "keyword_search",
    "memory_context",
]


