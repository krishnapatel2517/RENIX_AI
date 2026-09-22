"""
RENIX Research Summarizer

Provides structured summarization for research results.

Responsibilities:
- Summarize individual sources
- Combine multiple sources
- Extract key points
- Preserve important facts and numbers
- Remove repetitive information
- Produce concise, balanced summaries
- Track source references
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import re


@dataclass
class SummaryPoint:
    """A single important point extracted from research."""

    text: str
    importance: float = 0.0
    source: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "importance": self.importance,
        }


@dataclass
class SummaryResult:
    """Structured research summary."""

    title: str
    summary: str
    key_points: list[SummaryPoint] = field(
        default_factory=list
    )
    sources: list[Any] = field(
        default_factory=list
    )
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "key_points": [
                point.to_dict()
                for point in self.key_points
            ],
            "word_count": self.word_count,
        }


class ResearchSummarizer:
    """
    Rule-based research summarizer.

    Designed as a reliable preprocessing layer for RENIX.
    A future LLM summarization provider can be plugged into
    this class without changing the public API.
    """

    STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "at",
        "as",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "they",
        "their",
        "them",
        "there",
        "which",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "may",
        "might",
        "will",
        "also",
        "more",
        "most",
        "some",
        "such",
        "into",
        "about",
        "over",
        "after",
        "before",
        "during",
        "between",
        "through",
        "while",
        "because",
        "very",
        "often",
    }

    IMPORTANT_WORDS = {
        "important",
        "significant",
        "major",
        "critical",
        "key",
        "result",
        "results",
        "evidence",
        "research",
        "study",
        "found",
        "shows",
        "demonstrates",
        "according",
        "conclusion",
        "increase",
        "decrease",
        "growth",
        "decline",
        "risk",
        "benefit",
        "cause",
        "effect",
        "impact",
        "first",
        "largest",
        "highest",
        "lowest",
        "only",
        "approximately",
        "percent",
        "million",
        "billion",
    }

    def __init__(
        self,
        *,
        default_max_sentences: int = 5,
        max_key_points: int = 8,
    ) -> None:

        self.default_max_sentences = max(
            1,
            int(
                default_max_sentences
            ),
        )

        self.max_key_points = max(
            1,
            int(
                max_key_points
            ),
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def summarize(
        self,
        text: str,
        *,
        max_sentences: int | None = None,
        max_words: int | None = None,
        title: str = "",
    ) -> SummaryResult:

        text = self._clean_text(
            text
        )

        if not text:
            return SummaryResult(
                title=title
                or "Research Summary",
                summary="",
                word_count=0,
            )

        sentences = self._split_sentences(
            text
        )

        if not sentences:
            return SummaryResult(
                title=title
                or "Research Summary",
                summary=text,
                word_count=self._word_count(
                    text
                ),
            )

        sentence_scores = (
            self._score_sentences(
                sentences
            )
        )

        selected = self._select_sentences(
            sentence_scores,
            max_sentences=(
                max_sentences
                or self.default_max_sentences
            ),
            max_words=max_words,
        )

        # Preserve original order.
        selected.sort(
            key=lambda item: item[0]
        )

        summary = " ".join(
            sentence
            for _, sentence, _ in selected
        )

        if max_words:
            summary = self._limit_words(
                summary,
                max_words,
            )

        points = [
            SummaryPoint(
                text=sentence,
                importance=score,
            )
            for _, sentence, score
            in selected
        ]

        return SummaryResult(
            title=title
            or "Research Summary",
            summary=summary,
            key_points=points,
            word_count=self._word_count(
                summary
            ),
        )

    def summarize_source(
        self,
        source: Any,
        *,
        max_sentences: int | None = None,
        max_words: int | None = None,
    ) -> SummaryResult:

        title = self._get(
            source,
            "title",
            "Source Summary",
        )

        content = self._source_text(
            source
        )

        result = self.summarize(
            content,
            max_sentences=max_sentences,
            max_words=max_words,
            title=title,
        )

        result.sources = [
            source
        ]

        return result

    def summarize_sources(
        self,
        sources: Iterable[Any],
        *,
        max_sentences: int | None = None,
        max_words: int | None = None,
        title: str = "Research Summary",
    ) -> SummaryResult:

        source_list = list(
            sources or []
        )

        if not source_list:
            return SummaryResult(
                title=title,
                summary="",
            )

        combined = []

        for source in source_list:

            content = self._source_text(
                source
            )

            if not content:
                continue

            combined.append(
                content
            )

        text = "\n".join(
            combined
        )

        result = self.summarize(
            text,
            max_sentences=max_sentences,
            max_words=max_words,
            title=title,
        )

        result.sources = source_list

        return result

    # ============================================================
    # KEY POINT EXTRACTION
    # ============================================================

    def extract_key_points(
        self,
        text: str,
        *,
        limit: int | None = None,
    ) -> list[SummaryPoint]:

        text = self._clean_text(
            text
        )

        if not text:
            return []

        sentences = self._split_sentences(
            text
        )

        scored = self._score_sentences(
            sentences
        )

        scored.sort(
            key=lambda item: item[2],
            reverse=True,
        )

        limit = (
            limit
            or self.max_key_points
        )

        selected = scored[:limit]

        selected.sort(
            key=lambda item: item[0]
        )

        return [
            SummaryPoint(
                text=sentence,
                importance=score,
            )
            for _, sentence, score
            in selected
        ]

    # ============================================================
    # MULTI-SOURCE SYNTHESIS
    # ============================================================

    def synthesize(
        self,
        sources: Iterable[Any],
        *,
        max_words: int = 250,
        title: str = "Research Synthesis",
    ) -> SummaryResult:

        source_list = list(
            sources or []
        )

        if not source_list:
            return SummaryResult(
                title=title,
                summary="",
            )

        all_sentences: list[
            tuple[int, str, float, Any]
        ] = []

        for source_index, source in enumerate(
            source_list
        ):

            text = self._source_text(
                source
            )

            sentences = self._split_sentences(
                text
            )

            scores = self._score_sentences(
                sentences
            )

            for (
                sentence_index,
                sentence,
                score,
            ) in scores:

                adjusted_score = (
                    score
                    + self._source_quality_bonus(
                        source
                    )
                )

                all_sentences.append(
                    (
                        source_index,
                        sentence,
                        adjusted_score,
                        source,
                    )
                )

        if not all_sentences:
            return SummaryResult(
                title=title,
                summary="",
                sources=source_list,
            )

        selected = self._select_diverse_sentences(
            all_sentences,
            max_words=max_words,
        )

        selected.sort(
            key=lambda item: (
                item[0],
                all_sentences.index(item),
            )
        )

        summary = " ".join(
            item[1]
            for item in selected
        )

        summary = self._limit_words(
            summary,
            max_words,
        )

        points = [
            SummaryPoint(
                text=item[1],
                importance=item[2],
                source=item[3],
            )
            for item in selected
        ]

        return SummaryResult(
            title=title,
            summary=summary,
            key_points=points,
            sources=source_list,
            word_count=self._word_count(
                summary
            ),
        )

    # ============================================================
    # SENTENCE SCORING
    # ============================================================

    def _score_sentences(
        self,
        sentences: list[str],
    ) -> list[tuple[int, str, float]]:

        if not sentences:
            return []

        frequency = self._word_frequency(
            sentences
        )

        scored = []

        for index, sentence in enumerate(
            sentences
        ):

            score = self._score_sentence(
                sentence,
                frequency,
                index,
                len(sentences),
            )

            scored.append(
                (
                    index,
                    sentence,
                    score,
                )
            )

        return scored

    def _score_sentence(
        self,
        sentence: str,
        frequency: dict[str, int],
        index: int,
        total_sentences: int,
    ) -> float:

        words = self._content_words(
            sentence
        )

        if not words:
            return 0.0

        score = 0.0

        # --------------------------------------------------------
        # Word importance
        # --------------------------------------------------------

        frequency_score = sum(
            frequency.get(
                word,
                0,
            )
            for word in words
        )

        if words:
            score += min(
                1.0,
                frequency_score
                / (
                    len(words)
                    * 3
                ),
            ) * 0.25

        # --------------------------------------------------------
        # Important terminology
        # --------------------------------------------------------

        important_matches = sum(
            word in self.IMPORTANT_WORDS
            for word in words
        )

        score += min(
            1.0,
            important_matches / 3,
        ) * 0.20

        # --------------------------------------------------------
        # Numbers / measurable facts
        # --------------------------------------------------------

        if re.search(
            r"\d",
            sentence,
        ):
            score += 0.15

        if re.search(
            r"\b\d+(?:\.\d+)?\s*%",
            sentence,
        ):
            score += 0.10

        # --------------------------------------------------------
        # Position
        # --------------------------------------------------------

        if index == 0:
            score += 0.15

        elif index == total_sentences - 1:
            score += 0.08

        elif index < max(
            2,
            total_sentences // 4,
        ):
            score += 0.05

        # --------------------------------------------------------
        # Sentence quality
        # --------------------------------------------------------

        word_count = len(
            sentence.split()
        )

        if 8 <= word_count <= 35:
            score += 0.10

        elif word_count < 5:
            score -= 0.05

        elif word_count > 60:
            score -= 0.05

        # --------------------------------------------------------
        # Explicit conclusion language
        # --------------------------------------------------------

        lowered = sentence.lower()

        conclusion_terms = (
            "therefore",
            "thus",
            "overall",
            "in conclusion",
            "as a result",
            "consequently",
            "this means",
        )

        if any(
            term in lowered
            for term in conclusion_terms
        ):
            score += 0.10

        return self._clamp(
            score
        )

    # ============================================================
    # SENTENCE SELECTION
    # ============================================================

    def _select_sentences(
        self,
        scored: list[tuple[int, str, float]],
        *,
        max_sentences: int,
        max_words: int | None,
    ) -> list[tuple[int, str, float]]:

        ranked = sorted(
            scored,
            key=lambda item: item[2],
            reverse=True,
        )

        selected = []
        current_words = 0

        for item in ranked:

            sentence_words = self._word_count(
                item[1]
            )

            if (
                max_words
                and selected
                and current_words
                + sentence_words
                > max_words
            ):
                continue

            selected.append(
                item
            )

            current_words += (
                sentence_words
            )

            if len(selected) >= max_sentences:
                break

        return selected

    def _select_diverse_sentences(
        self,
        sentences: list[
            tuple[int, str, float, Any]
        ],
        *,
        max_words: int,
    ) -> list[
        tuple[int, str, float, Any]
    ]:

        ranked = sorted(
            sentences,
            key=lambda item: item[2],
            reverse=True,
        )

        selected = []
        used_sources: set[int] = set()
        used_words: set[str] = set()
        current_words = 0

        # First pass prioritizes source diversity.
        for item in ranked:

            source_index, sentence, score, source = (
                item
            )

            sentence_words = (
                self._word_count(
                    sentence
                )
            )

            if (
                current_words
                + sentence_words
                > max_words
            ):
                continue

            if source_index in used_sources:
                continue

            if self._is_redundant(
                sentence,
                used_words,
            ):
                continue

            selected.append(
                item
            )

            used_sources.add(
                source_index
            )

            used_words.update(
                self._content_words(
                    sentence
                )
            )

            current_words += (
                sentence_words
            )

            if current_words >= max_words:
                return selected

        # Second pass fills remaining space.
        for item in ranked:

            if item in selected:
                continue

            sentence_words = (
                self._word_count(
                    item[1]
                )
            )

            if (
                current_words
                + sentence_words
                > max_words
            ):
                continue

            if self._is_redundant(
                item[1],
                used_words,
            ):
                continue

            selected.append(
                item
            )

            used_words.update(
                self._content_words(
                    item[1]
                )
            )

            current_words += (
                sentence_words
            )

            if current_words >= max_words:
                break

        return selected

    # ============================================================
    # REDUNDANCY
    # ============================================================

    def _is_redundant(
        self,
        sentence: str,
        used_words: set[str],
    ) -> bool:

        words = set(
            self._content_words(
                sentence
            )
        )

        if not words or not used_words:
            return False

        overlap = len(
            words
            & used_words
        )

        ratio = (
            overlap
            / len(words)
        )

        return ratio >= 0.80

    # ============================================================
    # SOURCE QUALITY
    # ============================================================

    def _source_quality_bonus(
        self,
        source: Any,
    ) -> float:

        credibility = self._safe_float(
            self._get(
                source,
                "credibility",
                0.5,
            )
        )

        relevance = self._safe_float(
            self._get(
                source,
                "relevance",
                0.5,
            )
        )

        return (
            self._clamp(
                credibility
            )
            * 0.10
            + self._clamp(
                relevance
            )
            * 0.10
        )

    # ============================================================
    # SOURCE TEXT
    # ============================================================

    def _source_text(
        self,
        source: Any,
    ) -> str:

        fields = [
            "content",
            "snippet",
            "description",
            "summary",
        ]

        parts = []

        for field_name in fields:

            value = self._get(
                source,
                field_name,
                "",
            )

            if value:
                parts.append(
                    str(value)
                )

        return " ".join(
            parts
        ).strip()

    # ============================================================
    # WORD PROCESSING
    # ============================================================

    def _word_frequency(
        self,
        sentences: list[str],
    ) -> dict[str, int]:

        frequency: dict[
            str,
            int,
        ] = {}

        for sentence in sentences:

            for word in self._content_words(
                sentence
            ):

                frequency[word] = (
                    frequency.get(
                        word,
                        0,
                    )
                    + 1
                )

        return frequency

    def _content_words(
        self,
        text: str,
    ) -> list[str]:

        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            str(
                text or ""
            ).lower(),
        )

        return [
            word
            for word in words
            if (
                word not in self.STOPWORDS
                and len(word) > 2
            )
        ]

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        text = str(
            text or ""
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _word_count(
        text: str,
    ) -> int:

        return len(
            re.findall(
                r"\b\w+\b",
                str(
                    text or ""
                ),
            )
        )

    @staticmethod
    def _limit_words(
        text: str,
        max_words: int,
    ) -> str:

        if max_words <= 0:
            return ""

        words = text.split()

        if len(words) <= max_words:
            return text

        result = " ".join(
            words[:max_words]
        )

        return result.rstrip(
            " ,;:"
        ) + "…"

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

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

    @staticmethod
    def _get(
        source: Any,
        key: str,
        default: Any = "",
    ) -> Any:

        if isinstance(
            source,
            dict,
        ):
            return source.get(
                key,
                default,
            )

        return getattr(
            source,
            key,
            default,
        )


__all__ = [
    "SummaryPoint",
    "SummaryResult",
    "ResearchSummarizer",
    "Summarizer",
]


Summarizer = ResearchSummarizer
