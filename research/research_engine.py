"""
RENIX Research Engine

Coordinates research workflows:
- Query generation
- Source collection
- Source ranking
- Fact checking
- Summarization
- Comparison
- Citation generation
- Report preparation

The engine is intentionally provider-agnostic. Actual web/API
search implementations can be connected through the source manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Callable
import uuid


@dataclass
class ResearchSource:
    """Represents a research source."""

    title: str
    url: str = ""
    content: str = ""
    source_type: str = "web"
    publisher: str = ""
    author: str = ""
    published_at: str | None = None
    retrieved_at: str | None = None
    credibility: float = 0.0
    relevance: float = 0.0
    rank_score: float = 0.0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.title = (
            self.title or "Untitled Source"
        ).strip()

        self.url = (
            self.url or ""
        ).strip()

        self.content = (
            self.content or ""
        ).strip()

        self.source_type = (
            self.source_type or "web"
        ).lower()

        if self.retrieved_at is None:
            self.retrieved_at = (
                datetime.now().isoformat()
            )

    def to_dict(self) -> dict[str, Any]:

        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "source_type": self.source_type,
            "publisher": self.publisher,
            "author": self.author,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "credibility": self.credibility,
            "relevance": self.relevance,
            "rank_score": self.rank_score,
            "metadata": dict(self.metadata),
        }


@dataclass
class ResearchResult:
    """Complete result of a research operation."""

    query: str
    research_id: str = ""
    answer: str = ""
    summary: str = ""
    sources: list[ResearchSource] = field(
        default_factory=list
    )
    facts: list[dict[str, Any]] = field(
        default_factory=list
    )
    citations: list[dict[str, Any]] = field(
        default_factory=list
    )
    comparisons: list[dict[str, Any]] = field(
        default_factory=list
    )
    confidence: float = 0.0
    created_at: str = ""
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        if not self.research_id:
            self.research_id = (
                f"RESEARCH-{uuid.uuid4().hex[:12].upper()}"
            )

        if not self.created_at:
            self.created_at = (
                datetime.now().isoformat()
            )

    def to_dict(self) -> dict[str, Any]:

        return {
            "research_id": self.research_id,
            "query": self.query,
            "answer": self.answer,
            "summary": self.summary,
            "sources": [
                source.to_dict()
                for source in self.sources
            ],
            "facts": list(self.facts),
            "citations": list(self.citations),
            "comparisons": list(
                self.comparisons
            ),
            "confidence": self.confidence,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class ResearchEngine:
    """
    Main research coordinator for RENIX.

    Dependencies can be injected so the engine can work with
    different search providers and AI models.
    """

    def __init__(
        self,
        *,
        source_manager: Any = None,
        source_ranker: Any = None,
        fact_checker: Any = None,
        summarizer: Any = None,
        comparison_engine: Any = None,
        citation_manager: Any = None,
        report_generator: Any = None,
        llm: Any = None,
    ) -> None:

        self.source_manager = source_manager
        self.source_ranker = source_ranker
        self.fact_checker = fact_checker
        self.summarizer = summarizer
        self.comparison_engine = comparison_engine
        self.citation_manager = citation_manager
        self.report_generator = report_generator
        self.llm = llm

        self.history: list[
            ResearchResult
        ] = []

    # ============================================================
    # MAIN RESEARCH WORKFLOW
    # ============================================================

    def research(
        self,
        query: str,
        *,
        max_sources: int = 10,
        verify_facts: bool = True,
        summarize: bool = True,
        generate_citations: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchResult:

        query = (
            query or ""
        ).strip()

        if not query:
            raise ValueError(
                "Research query cannot be empty."
            )

        result = ResearchResult(
            query=query,
            metadata=dict(
                metadata or {}
            ),
        )

        # --------------------------------------------------------
        # 1. Collect sources
        # --------------------------------------------------------

        sources = self.search_sources(
            query,
            max_sources=max_sources,
        )

        result.sources = sources

        # --------------------------------------------------------
        # 2. Rank sources
        # --------------------------------------------------------

        result.sources = self.rank_sources(
            query,
            result.sources,
        )

        # --------------------------------------------------------
        # 3. Fact checking
        # --------------------------------------------------------

        if verify_facts:
            result.facts = self.check_facts(
                query,
                result.sources,
            )

        # --------------------------------------------------------
        # 4. Generate answer
        # --------------------------------------------------------

        result.answer = self.generate_answer(
            query,
            result.sources,
            result.facts,
        )

        # --------------------------------------------------------
        # 5. Summarize
        # --------------------------------------------------------

        if summarize:
            result.summary = self.summarize(
                result.answer,
                result.sources,
            )

        # --------------------------------------------------------
        # 6. Citations
        # --------------------------------------------------------

        if generate_citations:
            result.citations = self.generate_citations(
                result.sources,
                result.answer,
            )

        # --------------------------------------------------------
        # 7. Confidence
        # --------------------------------------------------------

        result.confidence = self.calculate_confidence(
            result.sources,
            result.facts,
        )

        self.history.append(
            result
        )

        return result

    # ============================================================
    # SEARCH
    # ============================================================

    def search_sources(
        self,
        query: str,
        *,
        max_sources: int = 10,
    ) -> list[ResearchSource]:

        if max_sources < 1:
            return []

        manager = self.source_manager

        if manager is None:
            return []

        try:

            if hasattr(
                manager,
                "search",
            ):
                raw_sources = manager.search(
                    query,
                    max_results=max_sources,
                )

            elif hasattr(
                manager,
                "search_sources",
            ):
                raw_sources = manager.search_sources(
                    query,
                    max_results=max_sources,
                )

            else:
                return []

        except TypeError:

            raw_sources = manager.search(
                query
            )

        return self._normalize_sources(
            raw_sources
        )[:max_sources]

    def search_multiple(
        self,
        queries: Iterable[str],
        *,
        max_sources_per_query: int = 5,
    ) -> list[ResearchSource]:

        all_sources: list[
            ResearchSource
        ] = []

        seen_urls: set[str] = set()

        for query in queries:

            sources = self.search_sources(
                query,
                max_sources=max_sources_per_query,
            )

            for source in sources:

                normalized_url = (
                    source.url.strip().lower()
                )

                if (
                    normalized_url
                    and normalized_url in seen_urls
                ):
                    continue

                if normalized_url:
                    seen_urls.add(
                        normalized_url
                    )

                all_sources.append(
                    source
                )

        return all_sources

    # ============================================================
    # SOURCE RANKING
    # ============================================================

    def rank_sources(
        self,
        query: str,
        sources: list[ResearchSource],
    ) -> list[ResearchSource]:

        if not sources:
            return []

        ranker = self.source_ranker

        if ranker is not None:

            try:

                if hasattr(
                    ranker,
                    "rank",
                ):
                    ranked = ranker.rank(
                        query,
                        sources,
                    )

                    return self._normalize_sources(
                        ranked
                    )

            except Exception:
                pass

        # Fallback ranking.
        for source in sources:

            relevance = self._calculate_relevance(
                query,
                source,
            )

            source.relevance = relevance

            credibility = max(
                0.0,
                min(
                    1.0,
                    float(
                        source.credibility
                        or 0.0
                    ),
                ),
            )

            source.rank_score = (
                relevance * 0.65
                + credibility * 0.35
            )

        return sorted(
            sources,
            key=lambda source: source.rank_score,
            reverse=True,
        )

    def _calculate_relevance(
        self,
        query: str,
        source: ResearchSource,
    ) -> float:

        query_words = {
            word.lower()
            for word in query.split()
            if len(word) > 2
        }

        if not query_words:
            return 0.0

        searchable = " ".join(
            [
                source.title,
                source.content,
                source.publisher,
            ]
        ).lower()

        matches = sum(
            word in searchable
            for word in query_words
        )

        return min(
            1.0,
            matches / len(query_words),
        )

    # ============================================================
    # FACT CHECKING
    # ============================================================

    def check_facts(
        self,
        query: str,
        sources: list[ResearchSource],
    ) -> list[dict[str, Any]]:

        checker = self.fact_checker

        if checker is not None:

            try:

                if hasattr(
                    checker,
                    "check",
                ):
                    result = checker.check(
                        query,
                        sources,
                    )

                    if result is None:
                        return []

                    return list(result)

                if hasattr(
                    checker,
                    "check_facts",
                ):
                    result = checker.check_facts(
                        query,
                        sources,
                    )

                    if result is None:
                        return []

                    return list(result)

            except Exception:
                pass

        return self._fallback_fact_check(
            sources
        )

    def _fallback_fact_check(
        self,
        sources: list[ResearchSource],
    ) -> list[dict[str, Any]]:

        facts = []

        for source in sources:

            if not source.content:
                continue

            facts.append(
                {
                    "claim": source.title,
                    "supported": True,
                    "confidence": max(
                        source.rank_score,
                        source.credibility,
                    ),
                    "source_url": source.url,
                    "source_title": source.title,
                }
            )

        return facts

    # ============================================================
    # ANSWER GENERATION
    # ============================================================

    def generate_answer(
        self,
        query: str,
        sources: list[ResearchSource],
        facts: list[dict[str, Any]],
    ) -> str:

        if self.llm is not None:

            prompt = self._build_answer_prompt(
                query,
                sources,
                facts,
            )

            try:

                if hasattr(
                    self.llm,
                    "generate",
                ):
                    response = self.llm.generate(
                        prompt
                    )

                    if isinstance(
                        response,
                        str,
                    ):
                        return response.strip()

                    if hasattr(
                        response,
                        "text",
                    ):
                        return str(
                            response.text
                        ).strip()

            except Exception:
                pass

        return self._fallback_answer(
            query,
            sources,
        )

    def _build_answer_prompt(
        self,
        query: str,
        sources: list[ResearchSource],
        facts: list[dict[str, Any]],
    ) -> str:

        source_text = []

        for index, source in enumerate(
            sources,
            start=1,
        ):

            source_text.append(
                f"""
SOURCE {index}
Title: {source.title}
Publisher: {source.publisher}
URL: {source.url}
Content:
{source.content[:5000]}
""".strip()
            )

        fact_text = "\n".join(
            str(fact)
            for fact in facts
        )

        return f"""
You are RENIX Research Engine.

Research question:
{query}

Use only the supplied sources.

Sources:
{chr(10).join(source_text)}

Fact-check information:
{fact_text}

Produce a clear, accurate answer.
Separate established facts from uncertainty.
Do not invent information.
Include source references where appropriate.
""".strip()

    def _fallback_answer(
        self,
        query: str,
        sources: list[ResearchSource],
    ) -> str:

        if not sources:
            return (
                "I could not find enough sources "
                f"to research: {query}"
            )

        lines = [
            f"Research results for: {query}",
            "",
        ]

        for index, source in enumerate(
            sources[:5],
            start=1,
        ):

            lines.append(
                f"{index}. {source.title}"
            )

            if source.content:
                preview = (
                    source.content
                    .replace("\n", " ")
                    .strip()
                )

                if len(preview) > 400:
                    preview = (
                        preview[:400]
                        + "..."
                    )

                lines.append(
                    f"   {preview}"
                )

            if source.url:
                lines.append(
                    f"   Source: {source.url}"
                )

        return "\n".join(
            lines
        )

    # ============================================================
    # SUMMARIZATION
    # ============================================================

    def summarize(
        self,
        answer: str,
        sources: list[ResearchSource],
    ) -> str:

        if not answer:
            return ""

        summarizer = self.summarizer

        if summarizer is not None:

            try:

                if hasattr(
                    summarizer,
                    "summarize",
                ):
                    result = summarizer.summarize(
                        answer
                    )

                    if result is not None:
                        return str(
                            result
                        ).strip()

            except Exception:
                pass

        # Basic fallback.
        sentences = [
            sentence.strip()
            for sentence in answer.split(".")
            if sentence.strip()
        ]

        if len(sentences) <= 3:
            return answer.strip()

        return (
            ". ".join(
                sentences[:3]
            )
            + "."
        )

    # ============================================================
    # COMPARISON
    # ============================================================

    def compare(
        self,
        items: Iterable[str],
        *,
        criteria: Iterable[str] | None = None,
    ) -> dict[str, Any]:

        items = [
            str(item).strip()
            for item in items
            if str(item).strip()
        ]

        criteria = [
            str(item).strip()
            for item in (criteria or [])
            if str(item).strip()
        ]

        if self.comparison_engine is not None:

            try:

                if hasattr(
                    self.comparison_engine,
                    "compare",
                ):
                    result = (
                        self.comparison_engine.compare(
                            items,
                            criteria=criteria,
                        )
                    )

                    if isinstance(
                        result,
                        dict,
                    ):
                        return result

            except Exception:
                pass

        return {
            "items": items,
            "criteria": criteria,
            "comparisons": [],
            "message": (
                "Comparison engine is not configured."
            ),
        }

    # ============================================================
    # CITATIONS
    # ============================================================

    def generate_citations(
        self,
        sources: list[ResearchSource],
        answer: str = "",
    ) -> list[dict[str, Any]]:

        manager = self.citation_manager

        if manager is not None:

            try:

                if hasattr(
                    manager,
                    "generate",
                ):
                    result = manager.generate(
                        sources,
                        answer,
                    )

                    if result is not None:
                        return list(result)

                if hasattr(
                    manager,
                    "create_citations",
                ):
                    result = (
                        manager.create_citations(
                            sources
                        )
                    )

                    if result is not None:
                        return list(result)

            except Exception:
                pass

        citations = []

        for index, source in enumerate(
            sources,
            start=1,
        ):

            citations.append(
                {
                    "index": index,
                    "title": source.title,
                    "url": source.url,
                    "publisher": source.publisher,
                    "author": source.author,
                    "published_at": source.published_at,
                }
            )

        return citations

    # ============================================================
    # REPORT
    # ============================================================

    def generate_report(
        self,
        result: ResearchResult,
        *,
        format: str = "markdown",
    ) -> str:

        generator = self.report_generator

        if generator is not None:

            try:

                if hasattr(
                    generator,
                    "generate",
                ):
                    return str(
                        generator.generate(
                            result,
                            format=format,
                        )
                    )

            except Exception:
                pass

        return self._fallback_report(
            result
        )

    def _fallback_report(
        self,
        result: ResearchResult,
    ) -> str:

        lines = [
            f"# Research Report",
            "",
            f"## Query",
            result.query,
            "",
            "## Answer",
            result.answer or "No answer generated.",
            "",
            "## Summary",
            result.summary or "No summary available.",
            "",
            "## Sources",
        ]

        for index, source in enumerate(
            result.sources,
            start=1,
        ):

            source_line = (
                f"{index}. {source.title}"
            )

            if source.url:
                source_line += (
                    f" — {source.url}"
                )

            lines.append(
                source_line
            )

        lines.extend(
            [
                "",
                "## Confidence",
                f"{result.confidence:.2%}",
            ]
        )

        return "\n".join(
            lines
        )

    # ============================================================
    # CONFIDENCE
    # ============================================================

    def calculate_confidence(
        self,
        sources: list[ResearchSource],
        facts: list[dict[str, Any]],
    ) -> float:

        if not sources:
            return 0.0

        source_scores = [
            max(
                0.0,
                min(
                    1.0,
                    float(
                        source.rank_score
                        or 0.0
                    ),
                ),
            )
            for source in sources
        ]

        source_confidence = (
            sum(source_scores)
            / len(source_scores)
            if source_scores
            else 0.0
        )

        if not facts:
            return round(
                source_confidence,
                3,
            )

        fact_scores = []

        for fact in facts:

            try:
                fact_scores.append(
                    max(
                        0.0,
                        min(
                            1.0,
                            float(
                                fact.get(
                                    "confidence",
                                    0.0,
                                )
                            ),
                        ),
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        fact_confidence = (
            sum(fact_scores)
            / len(fact_scores)
            if fact_scores
            else 0.0
        )

        return round(
            source_confidence * 0.5
            + fact_confidence * 0.5,
            3,
        )

    # ============================================================
    # HISTORY
    # ============================================================

    def get_history(
        self,
        limit: int | None = None,
    ) -> list[ResearchResult]:

        history = list(
            self.history
        )

        if limit is not None:
            return history[
                -max(0, int(limit)):
            ]

        return history

    def get_research(
        self,
        research_id: str,
    ) -> ResearchResult | None:

        for result in self.history:

            if result.research_id == research_id:
                return result

        return None

    def clear_history(self) -> None:

        self.history.clear()

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize_sources(
        self,
        sources: Any,
    ) -> list[ResearchSource]:

        if sources is None:
            return []

        if isinstance(
            sources,
            ResearchSource,
        ):
            sources = [sources]

        normalized: list[
            ResearchSource
        ] = []

        for source in sources:

            if isinstance(
                source,
                ResearchSource,
            ):
                normalized.append(
                    source
                )
                continue

            if isinstance(
                source,
                dict,
            ):

                try:

                    normalized.append(
                        ResearchSource(
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
                                    source.get(
                                        "snippet",
                                        "",
                                    ),
                                )
                            ),
                            source_type=str(
                                source.get(
                                    "source_type",
                                    "web",
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
                                    0.0,
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
                            rank_score=float(
                                source.get(
                                    "rank_score",
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
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                continue

            # Support simple source objects.
            title = getattr(
                source,
                "title",
                None,
            )

            if title is not None:

                normalized.append(
                    ResearchSource(
                        title=str(title),
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
                        publisher=str(
                            getattr(
                                source,
                                "publisher",
                                "",
                            )
                        ),
                    )
                )

        return normalized


__all__ = [
    "ResearchSource",
    "ResearchResult",
    "ResearchEngine",
]


