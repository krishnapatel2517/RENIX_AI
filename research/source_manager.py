"""
RENIX Source Manager

Handles research sources collected from web search providers,
APIs, local documents, databases, and other source adapters.

The manager is provider-agnostic so RENIX can connect different
search systems without changing the research engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable
from urllib.parse import urlparse
import hashlib


@dataclass
class SourceRecord:
    """Normalized representation of a research source."""

    title: str
    url: str = ""
    content: str = ""
    snippet: str = ""
    source_type: str = "web"
    publisher: str = ""
    author: str = ""
    published_at: str | None = None
    retrieved_at: str | None = None
    credibility: float = 0.0
    relevance: float = 0.0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.title = (
            self.title
            or "Untitled Source"
        ).strip()

        self.url = (
            self.url or ""
        ).strip()

        self.content = (
            self.content or ""
        ).strip()

        self.snippet = (
            self.snippet or ""
        ).strip()

        self.source_type = (
            self.source_type
            or "web"
        ).lower()

        if self.retrieved_at is None:
            self.retrieved_at = (
                datetime.now().isoformat()
            )

        self.credibility = self._clamp(
            self.credibility
        )

        self.relevance = self._clamp(
            self.relevance
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    @property
    def source_id(self) -> str:

        raw = (
            self.url
            or self.title
        ).encode(
            "utf-8",
            errors="ignore",
        )

        return hashlib.sha256(
            raw
        ).hexdigest()[:16]

    @property
    def domain(self) -> str:

        if not self.url:
            return ""

        try:
            return urlparse(
                self.url
            ).netloc.lower()
        except Exception:
            return ""

    @property
    def searchable_text(self) -> str:

        return " ".join(
            [
                self.title,
                self.snippet,
                self.content,
                self.publisher,
                self.author,
                self.domain,
            ]
        ).strip()

    def to_dict(self) -> dict[str, Any]:

        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "snippet": self.snippet,
            "source_type": self.source_type,
            "publisher": self.publisher,
            "author": self.author,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "credibility": self.credibility,
            "relevance": self.relevance,
            "domain": self.domain,
            "metadata": dict(
                self.metadata
            ),
        }


class SourceManager:
    """
    Manages research sources and search providers.

    Providers may expose any of these methods:

        search(query, max_results=...)
        search_sources(query, max_results=...)
        fetch(url)
        get_source(url)

    A provider can also simply return dictionaries containing
    title, url, content, snippet, etc.
    """

    def __init__(
        self,
        *,
        providers: Iterable[Any] | None = None,
        default_credibility: float = 0.5,
    ) -> None:

        self.providers: list[Any] = []

        self.sources: dict[
            str,
            SourceRecord,
        ] = {}

        self.default_credibility = max(
            0.0,
            min(
                1.0,
                float(
                    default_credibility
                ),
            ),
        )

        for provider in providers or []:
            self.register_provider(
                provider
            )

    # ============================================================
    # PROVIDER MANAGEMENT
    # ============================================================

    def register_provider(
        self,
        provider: Any,
    ) -> bool:

        if provider is None:
            return False

        if provider in self.providers:
            return False

        self.providers.append(
            provider
        )

        return True

    def unregister_provider(
        self,
        provider: Any,
    ) -> bool:

        if provider not in self.providers:
            return False

        self.providers.remove(
            provider
        )

        return True

    def clear_providers(self) -> None:

        self.providers.clear()

    def provider_count(self) -> int:

        return len(
            self.providers
        )

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        source_type: str | None = None,
    ) -> list[SourceRecord]:

        query = (
            query or ""
        ).strip()

        if not query:
            return []

        max_results = max(
            1,
            int(max_results),
        )

        collected: list[
            SourceRecord
        ] = []

        for provider in self.providers:

            try:

                raw_results = self._search_provider(
                    provider,
                    query,
                    max_results,
                )

                normalized = (
                    self._normalize_sources(
                        raw_results
                    )
                )

                for source in normalized:

                    if source_type:
                        if (
                            source.source_type
                            != source_type.lower()
                        ):
                            continue

                    source.relevance = (
                        self._calculate_relevance(
                            query,
                            source,
                        )
                    )

                    if source.credibility <= 0:
                        source.credibility = (
                            self.default_credibility
                        )

                    self.add_source(
                        source
                    )

                    collected.append(
                        source
                    )

            except Exception:
                # One broken provider should not stop
                # all other providers.
                continue

            if len(
                collected
            ) >= max_results:
                break

        return self.deduplicate(
            collected
        )[:max_results]

    def _search_provider(
        self,
        provider: Any,
        query: str,
        max_results: int,
    ) -> Any:

        if hasattr(
            provider,
            "search",
        ):

            try:
                return provider.search(
                    query,
                    max_results=max_results,
                )
            except TypeError:
                return provider.search(
                    query
                )

        if hasattr(
            provider,
            "search_sources",
        ):

            try:
                return provider.search_sources(
                    query,
                    max_results=max_results,
                )
            except TypeError:
                return provider.search_sources(
                    query
                )

        if callable(provider):
            return provider(
                query
            )

        return []

    def search_all(
        self,
        queries: Iterable[str],
        *,
        max_results_per_query: int = 5,
    ) -> list[SourceRecord]:

        results: list[
            SourceRecord
        ] = []

        for query in queries:

            results.extend(
                self.search(
                    query,
                    max_results=(
                        max_results_per_query
                    ),
                )
            )

        return self.deduplicate(
            results
        )

    # ============================================================
    # SOURCE STORAGE
    # ============================================================

    def add_source(
        self,
        source: SourceRecord | dict[str, Any],
    ) -> SourceRecord:

        normalized = self._normalize_source(
            source
        )

        existing = self.sources.get(
            normalized.source_id
        )

        if existing is not None:

            self._merge_sources(
                existing,
                normalized,
            )

            return existing

        self.sources[
            normalized.source_id
        ] = normalized

        return normalized

    def add_sources(
        self,
        sources: Iterable[
            SourceRecord | dict[str, Any]
        ],
    ) -> list[SourceRecord]:

        result = []

        for source in sources:

            result.append(
                self.add_source(
                    source
                )
            )

        return result

    def get_source(
        self,
        source_id: str,
    ) -> SourceRecord | None:

        return self.sources.get(
            source_id
        )

    def get_by_url(
        self,
        url: str,
    ) -> SourceRecord | None:

        normalized = (
            url or ""
        ).strip().lower()

        if not normalized:
            return None

        for source in self.sources.values():

            if (
                source.url.lower()
                == normalized
            ):
                return source

        return None

    def all_sources(
        self,
    ) -> list[SourceRecord]:

        return list(
            self.sources.values()
        )

    def remove_source(
        self,
        source_id: str,
    ) -> bool:

        if source_id not in self.sources:
            return False

        del self.sources[
            source_id
        ]

        return True

    def clear_sources(self) -> None:

        self.sources.clear()

    # ============================================================
    # FETCH / UPDATE
    # ============================================================

    def fetch(
        self,
        url: str,
    ) -> SourceRecord | None:

        url = (
            url or ""
        ).strip()

        if not url:
            return None

        existing = self.get_by_url(
            url
        )

        if existing is not None:
            return existing

        for provider in self.providers:

            try:

                result = None

                if hasattr(
                    provider,
                    "fetch",
                ):
                    result = provider.fetch(
                        url
                    )

                elif hasattr(
                    provider,
                    "get_source",
                ):
                    result = provider.get_source(
                        url
                    )

                if result is None:
                    continue

                source = self.add_source(
                    result
                )

                return source

            except Exception:
                continue

        return None

    def update_source(
        self,
        source_id: str,
        **updates: Any,
    ) -> bool:

        source = self.get_source(
            source_id
        )

        if source is None:
            return False

        allowed = {
            "title",
            "url",
            "content",
            "snippet",
            "source_type",
            "publisher",
            "author",
            "published_at",
            "retrieved_at",
            "credibility",
            "relevance",
            "metadata",
        }

        for key, value in updates.items():

            if key not in allowed:
                continue

            if key in {
                "credibility",
                "relevance",
            }:
                value = SourceRecord._clamp(
                    value
                )

            setattr(
                source,
                key,
                value,
            )

        return True

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    def deduplicate(
        self,
        sources: Iterable[
            SourceRecord
        ],
    ) -> list[SourceRecord]:

        unique: dict[
            str,
            SourceRecord,
        ] = {}

        for source in sources:

            source = self._normalize_source(
                source
            )

            key = (
                source.url.strip().lower()
                if source.url
                else source.title.strip().lower()
            )

            if not key:
                continue

            if key not in unique:

                unique[key] = source

            else:

                self._merge_sources(
                    unique[key],
                    source,
                )

        return list(
            unique.values()
        )

    def _merge_sources(
        self,
        target: SourceRecord,
        incoming: SourceRecord,
    ) -> None:

        if (
            not target.content
            and incoming.content
        ):
            target.content = (
                incoming.content
            )

        if (
            not target.snippet
            and incoming.snippet
        ):
            target.snippet = (
                incoming.snippet
            )

        if (
            not target.publisher
            and incoming.publisher
        ):
            target.publisher = (
                incoming.publisher
            )

        if (
            not target.author
            and incoming.author
        ):
            target.author = (
                incoming.author
            )

        if (
            not target.published_at
            and incoming.published_at
        ):
            target.published_at = (
                incoming.published_at
            )

        target.credibility = max(
            target.credibility,
            incoming.credibility,
        )

        target.relevance = max(
            target.relevance,
            incoming.relevance,
        )

        target.metadata.update(
            incoming.metadata
        )

    # ============================================================
    # RANKING HELPERS
    # ============================================================

    def top_sources(
        self,
        *,
        limit: int = 10,
    ) -> list[SourceRecord]:

        limit = max(
            1,
            int(limit),
        )

        return sorted(
            self.sources.values(),
            key=lambda source: (
                source.relevance * 0.6
                + source.credibility * 0.4
            ),
            reverse=True,
        )[:limit]

    def sources_by_domain(
        self,
        domain: str,
    ) -> list[SourceRecord]:

        domain = (
            domain or ""
        ).lower().strip()

        return [
            source
            for source in self.sources.values()
            if source.domain == domain
        ]

    def sources_by_type(
        self,
        source_type: str,
    ) -> list[SourceRecord]:

        source_type = (
            source_type or ""
        ).lower().strip()

        return [
            source
            for source in self.sources.values()
            if source.source_type
            == source_type
        ]

    # ============================================================
    # SEARCH RELEVANCE
    # ============================================================

    def _calculate_relevance(
        self,
        query: str,
        source: SourceRecord,
    ) -> float:

        words = {
            word.lower()
            for word in query.split()
            if len(word) > 2
        }

        if not words:
            return 0.0

        text = (
            source.searchable_text
            .lower()
        )

        if not text:
            return 0.0

        matched = sum(
            word in text
            for word in words
        )

        return min(
            1.0,
            matched / len(words),
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize_source(
        self,
        source: (
            SourceRecord
            | dict[str, Any]
            | Any
        ),
    ) -> SourceRecord:

        if isinstance(
            source,
            SourceRecord,
        ):
            return source

        if isinstance(
            source,
            dict,
        ):

            return SourceRecord(
                title=str(
                    source.get(
                        "title",
                        "Untitled Source",
                    )
                ),
                url=str(
                    source.get(
                        "url",
                        "",
                    )
                ),
                content=str(
                    source.get(
                        "content",
                        "",
                    )
                ),
                snippet=str(
                    source.get(
                        "snippet",
                        "",
                    )
                ),
                source_type=str(
                    source.get(
                        "source_type",
                        source.get(
                            "type",
                            "web",
                        ),
                    )
                ),
                publisher=str(
                    source.get(
                        "publisher",
                        "",
                    )
                ),
                author=str(
                    source.get(
                        "author",
                        "",
                    )
                ),
                published_at=source.get(
                    "published_at"
                ),
                retrieved_at=source.get(
                    "retrieved_at"
                ),
                credibility=float(
                    source.get(
                        "credibility",
                        self.default_credibility,
                    )
                    or 0.0
                ),
                relevance=float(
                    source.get(
                        "relevance",
                        0.0,
                    )
                    or 0.0
                ),
                metadata=dict(
                    source.get(
                        "metadata",
                        {},
                    )
                ),
            )

        return SourceRecord(
            title=str(
                getattr(
                    source,
                    "title",
                    "Untitled Source",
                )
            ),
            url=str(
                getattr(
                    source,
                    "url",
                    "",
                )
            ),
            content=str(
                getattr(
                    source,
                    "content",
                    "",
                )
            ),
            snippet=str(
                getattr(
                    source,
                    "snippet",
                    "",
                )
            ),
            source_type=str(
                getattr(
                    source,
                    "source_type",
                    "web",
                )
            ),
            publisher=str(
                getattr(
                    source,
                    "publisher",
                    "",
                )
            ),
            author=str(
                getattr(
                    source,
                    "author",
                    "",
                )
            ),
            published_at=getattr(
                source,
                "published_at",
                None,
            ),
            retrieved_at=getattr(
                source,
                "retrieved_at",
                None,
            ),
            credibility=float(
                getattr(
                    source,
                    "credibility",
                    self.default_credibility,
                )
                or 0.0
            ),
            relevance=float(
                getattr(
                    source,
                    "relevance",
                    0.0,
                )
                or 0.0
            ),
            metadata=dict(
                getattr(
                    source,
                    "metadata",
                    {},
                )
            ),
        )

    def _normalize_sources(
        self,
        sources: Any,
    ) -> list[SourceRecord]:

        if sources is None:
            return []

        if isinstance(
            sources,
            (
                SourceRecord,
                dict,
            ),
        ):
            sources = [sources]

        normalized = []

        try:
            iterator = iter(
                sources
            )
        except TypeError:
            return []

        for source in iterator:

            try:
                normalized.append(
                    self._normalize_source(
                        source
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        return normalized

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def export_state(
        self,
    ) -> dict[str, Any]:

        return {
            "sources": [
                source.to_dict()
                for source in self.sources.values()
            ]
        }

    def import_state(
        self,
        data: dict[str, Any],
    ) -> None:

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "Source manager state must be a dictionary."
            )

        self.sources.clear()

        raw_sources = data.get(
            "sources",
            [],
        )

        if not isinstance(
            raw_sources,
            list,
        ):
            return

        for raw_source in raw_sources:

            try:

                self.add_source(
                    raw_source
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

    # ============================================================
    # STATISTICS
    # ============================================================

    def statistics(
        self,
    ) -> dict[str, Any]:

        sources = list(
            self.sources.values()
        )

        if not sources:
            return {
                "total": 0,
                "domains": 0,
                "source_types": 0,
                "average_credibility": 0.0,
                "average_relevance": 0.0,
            }

        domains = {
            source.domain
            for source in sources
            if source.domain
        }

        types = {
            source.source_type
            for source in sources
        }

        return {
            "total": len(sources),
            "domains": len(domains),
            "source_types": len(types),
            "average_credibility": round(
                sum(
                    source.credibility
                    for source in sources
                )
                / len(sources),
                3,
            ),
            "average_relevance": round(
                sum(
                    source.relevance
                    for source in sources
                )
                / len(sources),
                3,
            ),
        }


__all__ = [
    "SourceRecord",
    "SourceManager",
]


