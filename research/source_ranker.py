"""
RENIX Source Ranker

Ranks research sources according to:
- Relevance to the query
- Source credibility
- Content quality
- Freshness
- Domain reputation
- Information completeness
- Source diversity

The ranker is provider-agnostic and can work with SourceRecord
objects, dictionaries, or compatible source objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlparse
import math
import re


@dataclass
class RankingWeights:
    """Weights used to calculate the final source score."""

    relevance: float = 0.40
    credibility: float = 0.25
    quality: float = 0.15
    freshness: float = 0.10
    completeness: float = 0.10

    def normalize(self) -> None:

        values = [
            max(0.0, float(self.relevance)),
            max(0.0, float(self.credibility)),
            max(0.0, float(self.quality)),
            max(0.0, float(self.freshness)),
            max(0.0, float(self.completeness)),
        ]

        total = sum(values)

        if total <= 0:
            self.relevance = 0.40
            self.credibility = 0.25
            self.quality = 0.15
            self.freshness = 0.10
            self.completeness = 0.10
            return

        (
            self.relevance,
            self.credibility,
            self.quality,
            self.freshness,
            self.completeness,
        ) = [
            value / total
            for value in values
        ]


@dataclass
class SourceScore:
    """Detailed scoring information for one source."""

    source: Any
    relevance: float = 0.0
    credibility: float = 0.0
    quality: float = 0.0
    freshness: float = 0.0
    completeness: float = 0.0
    diversity: float = 0.0
    final_score: float = 0.0
    reasons: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:

        return {
            "relevance": self.relevance,
            "credibility": self.credibility,
            "quality": self.quality,
            "freshness": self.freshness,
            "completeness": self.completeness,
            "diversity": self.diversity,
            "final_score": self.final_score,
            "reasons": list(self.reasons),
        }


class SourceRanker:
    """
    Scores and ranks research sources.

    This class intentionally does not perform web searches itself.
    It evaluates sources supplied by SourceManager or another
    research component.
    """

    DEFAULT_DOMAIN_SCORES = {
        "gov": 0.95,
        "gov.in": 0.95,
        "nic.in": 0.95,
        "edu": 0.90,
        "ac.in": 0.90,
        "org": 0.75,
        "com": 0.60,
        "net": 0.55,
    }

    HIGH_TRUST_DOMAINS = {
        "who.int",
        "un.org",
        "nasa.gov",
        "nih.gov",
        "cdc.gov",
        "nature.com",
        "science.org",
        "ieee.org",
        "acm.org",
        "worldbank.org",
        "imf.org",
        "oecd.org",
        "gov.in",
        "india.gov.in",
    }

    LOW_TRUST_PATTERNS = {
        "clickbait",
        "shocking",
        "you won't believe",
        "miracle",
        "secret they don't want",
        "guaranteed",
    }

    def __init__(
        self,
        *,
        weights: RankingWeights | None = None,
        domain_scores: dict[str, float] | None = None,
        freshness_days: int = 365,
    ) -> None:

        self.weights = (
            weights
            or RankingWeights()
        )

        self.weights.normalize()

        self.domain_scores = dict(
            self.DEFAULT_DOMAIN_SCORES
        )

        if domain_scores:
            self.domain_scores.update(
                domain_scores
            )

        self.freshness_days = max(
            1,
            int(freshness_days),
        )

    # ============================================================
    # MAIN RANKING
    # ============================================================

    def rank(
        self,
        query: str,
        sources: Iterable[Any],
    ) -> list[Any]:

        scored = self.score_sources(
            query,
            sources,
        )

        scored.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        ranked = [
            item.source
            for item in scored
        ]

        for item in scored:
            self._set_rank_score(
                item.source,
                item.final_score,
            )

        return ranked

    def rank_with_scores(
        self,
        query: str,
        sources: Iterable[Any],
    ) -> list[SourceScore]:

        scored = self.score_sources(
            query,
            sources,
        )

        scored.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        for item in scored:
            self._set_rank_score(
                item.source,
                item.final_score,
            )

        return scored

    # ============================================================
    # SCORING
    # ============================================================

    def score_sources(
        self,
        query: str,
        sources: Iterable[Any],
    ) -> list[SourceScore]:

        query = (
            query or ""
        ).strip()

        source_list = list(
            sources or []
        )

        if not source_list:
            return []

        scores = []

        for source in source_list:

            relevance = (
                self.score_relevance(
                    query,
                    source,
                )
            )

            credibility = (
                self.score_credibility(
                    source
                )
            )

            quality = (
                self.score_quality(
                    source
                )
            )

            freshness = (
                self.score_freshness(
                    source
                )
            )

            completeness = (
                self.score_completeness(
                    source
                )
            )

            diversity = (
                self.score_diversity(
                    source,
                    source_list,
                )
            )

            final_score = (
                relevance
                * self.weights.relevance
                + credibility
                * self.weights.credibility
                + quality
                * self.weights.quality
                + freshness
                * self.weights.freshness
                + completeness
                * self.weights.completeness
            )

            reasons = (
                self._build_reasons(
                    relevance,
                    credibility,
                    quality,
                    freshness,
                    completeness,
                    diversity,
                )
            )

            scores.append(
                SourceScore(
                    source=source,
                    relevance=relevance,
                    credibility=credibility,
                    quality=quality,
                    freshness=freshness,
                    completeness=completeness,
                    diversity=diversity,
                    final_score=self._clamp(
                        final_score
                    ),
                    reasons=reasons,
                )
            )

        return scores

    # ============================================================
    # RELEVANCE
    # ============================================================

    def score_relevance(
        self,
        query: str,
        source: Any,
    ) -> float:

        query_terms = self._tokenize(
            query
        )

        if not query_terms:
            return 0.0

        title = self._get(
            source,
            "title",
        )

        snippet = self._get(
            source,
            "snippet",
        )

        content = self._get(
            source,
            "content",
        )

        publisher = self._get(
            source,
            "publisher",
        )

        title_terms = set(
            self._tokenize(
                title
            )
        )

        content_terms = set(
            self._tokenize(
                f"{snippet} {content}"
            )
        )

        publisher_terms = set(
            self._tokenize(
                publisher
            )
        )

        if not content_terms:
            content_terms = set()

        title_matches = len(
            set(query_terms)
            & title_terms
        )

        content_matches = len(
            set(query_terms)
            & content_terms
        )

        publisher_matches = len(
            set(query_terms)
            & publisher_terms
        )

        total_terms = len(
            set(query_terms)
        )

        title_score = (
            title_matches
            / total_terms
            if total_terms
            else 0.0
        )

        content_score = (
            content_matches
            / total_terms
            if total_terms
            else 0.0
        )

        publisher_score = (
            publisher_matches
            / total_terms
            if total_terms
            else 0.0
        )

        # Title matches are more valuable than random
        # occurrences in a long article.
        score = (
            title_score * 0.55
            + content_score * 0.40
            + publisher_score * 0.05
        )

        # Preserve explicit relevance when a source provider
        # already supplied a meaningful score.
        supplied = self._safe_float(
            self._get(
                source,
                "relevance",
                0.0,
            )
        )

        if supplied > 0:
            score = (
                score * 0.75
                + self._clamp(supplied) * 0.25
            )

        return self._clamp(
            score
        )

    # ============================================================
    # CREDIBILITY
    # ============================================================

    def score_credibility(
        self,
        source: Any,
    ) -> float:

        supplied = self._safe_float(
            self._get(
                source,
                "credibility",
                0.0,
            )
        )

        url = self._get(
            source,
            "url",
        )

        domain = self._extract_domain(
            url
        )

        if not domain:
            base_score = 0.40
        else:
            base_score = (
                self._domain_credibility(
                    domain
                )
            )

        publisher = self._get(
            source,
            "publisher",
        )

        if publisher:
            base_score += 0.05

        author = self._get(
            source,
            "author",
        )

        if author:
            base_score += 0.05

        score = self._clamp(
            base_score
        )

        if supplied > 0:
            score = (
                score * 0.55
                + self._clamp(
                    supplied
                )
                * 0.45
            )

        return self._clamp(
            score
        )

    def _domain_credibility(
        self,
        domain: str,
    ) -> float:

        domain = (
            domain or ""
        ).lower()
        domain = domain.removeprefix(
            "www."
        )

        if domain in self.HIGH_TRUST_DOMAINS:
            return 0.98

        for trusted in self.HIGH_TRUST_DOMAINS:

            if domain.endswith(
                "." + trusted
            ):
                return 0.95

        extension = ""

        if "." in domain:
            extension = domain.split(
                "."
            )[-1]

        # Handle multi-part Indian academic/government domains.
        if domain.endswith(
            ".gov.in"
        ):
            return 0.95

        if domain.endswith(
            ".ac.in"
        ):
            return 0.90

        if domain.endswith(
            ".edu"
        ):
            return 0.90

        if domain.endswith(
            ".org"
        ):
            return self.domain_scores.get(
                "org",
                0.75,
            )

        if domain.endswith(
            ".com"
        ):
            return self.domain_scores.get(
                "com",
                0.60,
            )

        if domain.endswith(
            ".net"
        ):
            return self.domain_scores.get(
                "net",
                0.55,
            )

        return self.domain_scores.get(
            extension,
            0.50,
        )

    # ============================================================
    # QUALITY
    # ============================================================

    def score_quality(
        self,
        source: Any,
    ) -> float:

        title = self._get(
            source,
            "title",
        )

        content = self._get(
            source,
            "content",
        )

        snippet = self._get(
            source,
            "snippet",
        )

        publisher = self._get(
            source,
            "publisher",
        )

        author = self._get(
            source,
            "author",
        )

        if not content:
            content = snippet

        score = 0.0

        # --------------------------------------------------------
        # Content length
        # --------------------------------------------------------

        length = len(
            content
        )

        if length >= 5000:
            score += 0.35
        elif length >= 2000:
            score += 0.30
        elif length >= 1000:
            score += 0.25
        elif length >= 500:
            score += 0.18
        elif length >= 200:
            score += 0.12
        elif length > 0:
            score += 0.05

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        if title:
            score += 0.15

        if publisher:
            score += 0.10

        if author:
            score += 0.10

        # --------------------------------------------------------
        # Structure
        # --------------------------------------------------------

        if self._has_structure(
            content
        ):
            score += 0.10

        # --------------------------------------------------------
        # Specificity
        # --------------------------------------------------------

        if self._contains_numbers(
            content
        ):
            score += 0.05

        # --------------------------------------------------------
        # Language quality
        # --------------------------------------------------------

        if (
            content.count(".")
            >= 3
        ):
            score += 0.05

        # --------------------------------------------------------
        # Clickbait penalty
        # --------------------------------------------------------

        combined = (
            f"{title} {content}"
        ).lower()

        for pattern in self.LOW_TRUST_PATTERNS:

            if pattern in combined:
                score -= 0.08

        return self._clamp(
            score
        )

    # ============================================================
    # FRESHNESS
    # ============================================================

    def score_freshness(
        self,
        source: Any,
    ) -> float:

        published_at = self._get(
            source,
            "published_at",
        )

        if not published_at:
            return 0.50

        published = (
            self._parse_datetime(
                published_at
            )
        )

        if published is None:
            return 0.50

        now = datetime.now(
            timezone.utc
        )

        if published.tzinfo is None:
            published = published.replace(
                tzinfo=timezone.utc
            )

        age_days = max(
            0.0,
            (
                now - published
            ).total_seconds()
            / 86400,
        )

        # Exponential decay.
        score = math.exp(
            -age_days
            / self.freshness_days
        )

        return self._clamp(
            score
        )

    # ============================================================
    # COMPLETENESS
    # ============================================================

    def score_completeness(
        self,
        source: Any,
    ) -> float:

        fields = {
            "title": self._get(
                source,
                "title",
            ),
            "url": self._get(
                source,
                "url",
            ),
            "content": self._get(
                source,
                "content",
            ),
            "snippet": self._get(
                source,
                "snippet",
            ),
            "publisher": self._get(
                source,
                "publisher",
            ),
            "author": self._get(
                source,
                "author",
            ),
            "published_at": self._get(
                source,
                "published_at",
            ),
        }

        weights = {
            "title": 0.15,
            "url": 0.15,
            "content": 0.30,
            "snippet": 0.10,
            "publisher": 0.10,
            "author": 0.10,
            "published_at": 0.10,
        }

        score = 0.0

        for field_name, weight in weights.items():

            if fields[field_name]:
                score += weight

        return self._clamp(
            score
        )

    # ============================================================
    # DIVERSITY
    # ============================================================

    def score_diversity(
        self,
        source: Any,
        all_sources: Iterable[Any],
    ) -> float:

        sources = list(
            all_sources or []
        )

        if len(sources) <= 1:
            return 1.0

        current_domain = (
            self._extract_domain(
                self._get(
                    source,
                    "url",
                )
            )
        )

        if not current_domain:
            return 0.5

        other_domains = set()

        for candidate in sources:

            if candidate is source:
                continue

            domain = (
                self._extract_domain(
                    self._get(
                        candidate,
                        "url",
                    )
                )
            )

            if domain:
                other_domains.add(
                    domain
                )

        if not other_domains:
            return 1.0

        if current_domain not in other_domains:
            return 1.0

        same_domain = sum(
            1
            for candidate in sources
            if self._extract_domain(
                self._get(
                    candidate,
                    "url",
                )
            )
            == current_domain
        )

        ratio = (
            same_domain
            / len(sources)
        )

        return self._clamp(
            1.0 - ratio * 0.5
        )

    # ============================================================
    # FILTERING
    # ============================================================

    def filter_sources(
        self,
        scored_sources: Iterable[SourceScore],
        *,
        minimum_score: float = 0.0,
        minimum_credibility: float = 0.0,
        minimum_relevance: float = 0.0,
    ) -> list[SourceScore]:

        minimum_score = self._clamp(
            minimum_score
        )

        minimum_credibility = self._clamp(
            minimum_credibility
        )

        minimum_relevance = self._clamp(
            minimum_relevance
        )

        return [
            item
            for item in scored_sources
            if (
                item.final_score
                >= minimum_score
                and item.credibility
                >= minimum_credibility
                and item.relevance
                >= minimum_relevance
            )
        ]

    # ============================================================
    # TOP SOURCE SELECTION
    # ============================================================

    def select_best(
        self,
        query: str,
        sources: Iterable[Any],
        *,
        limit: int = 5,
        minimum_score: float = 0.0,
        diversify: bool = True,
    ) -> list[Any]:

        scored = self.rank_with_scores(
            query,
            sources,
        )

        scored = [
            item
            for item in scored
            if item.final_score
            >= minimum_score
        ]

        if not diversify:
            return [
                item.source
                for item in scored[:limit]
            ]

        selected = []

        domains_seen: set[str] = set()

        # First pass: maximize domain diversity.
        for item in scored:

            domain = (
                self._extract_domain(
                    self._get(
                        item.source,
                        "url",
                    )
                )
            )

            if (
                domain
                and domain in domains_seen
            ):
                continue

            selected.append(
                item.source
            )

            if domain:
                domains_seen.add(
                    domain
                )

            if len(selected) >= limit:
                return selected

        # Second pass: fill remaining slots.
        for item in scored:

            if item.source in selected:
                continue

            selected.append(
                item.source
            )

            if len(selected) >= limit:
                break

        return selected

    # ============================================================
    # REASONS
    # ============================================================

    def _build_reasons(
        self,
        relevance: float,
        credibility: float,
        quality: float,
        freshness: float,
        completeness: float,
        diversity: float,
    ) -> list[str]:

        reasons = []

        if relevance >= 0.75:
            reasons.append(
                "Highly relevant to the query."
            )
        elif relevance >= 0.45:
            reasons.append(
                "Moderately relevant to the query."
            )
        else:
            reasons.append(
                "Limited query relevance."
            )

        if credibility >= 0.80:
            reasons.append(
                "Strong source credibility."
            )
        elif credibility >= 0.60:
            reasons.append(
                "Moderate source credibility."
            )
        else:
            reasons.append(
                "Credibility should be independently verified."
            )

        if quality >= 0.75:
            reasons.append(
                "High content quality."
            )
        elif quality < 0.40:
            reasons.append(
                "Limited content quality or detail."
            )

        if freshness >= 0.75:
            reasons.append(
                "Relatively fresh information."
            )
        elif freshness < 0.30:
            reasons.append(
                "Information may be outdated."
            )

        if completeness >= 0.75:
            reasons.append(
                "Good metadata and content completeness."
            )

        if diversity >= 0.80:
            reasons.append(
                "Adds useful source diversity."
            )

        return reasons

    # ============================================================
    # UTILITIES
    # ============================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            (text or "").lower(),
        )

    @staticmethod
    def _contains_numbers(
        text: str,
    ) -> bool:

        return bool(
            re.search(
                r"\d",
                text or "",
            )
        )

    @staticmethod
    def _has_structure(
        text: str,
    ) -> bool:

        if not text:
            return False

        return bool(
            re.search(
                r"(^|\n)\s*(#{1,6}\s|[-*]\s|\d+\.\s)",
                text,
            )
        )

    @staticmethod
    def _extract_domain(
        url: str,
    ) -> str:

        if not url:
            return ""

        try:

            domain = urlparse(
                url
            ).netloc.lower()

            return domain.removeprefix(
                "www."
            )

        except Exception:

            return ""

    @staticmethod
    def _parse_datetime(
        value: Any,
    ) -> datetime | None:

        if isinstance(
            value,
            datetime,
        ):
            return value

        if not isinstance(
            value,
            str,
        ):
            return None

        value = value.strip()

        if not value:
            return None

        try:

            parsed = datetime.fromisoformat(
                value.replace(
                    "Z",
                    "+00:00",
                )
            )

            return parsed

        except ValueError:

            formats = [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%B %d, %Y",
                "%b %d, %Y",
            ]

            for fmt in formats:

                try:
                    return datetime.strptime(
                        value,
                        fmt,
                    )
                except ValueError:
                    continue

        return None

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

    @staticmethod
    def _set_rank_score(
        source: Any,
        score: float,
    ) -> None:

        try:

            if isinstance(
                source,
                dict,
            ):
                source[
                    "rank_score"
                ] = score
            else:
                setattr(
                    source,
                    "rank_score",
                    score,
                )

        except Exception:
            pass


__all__ = [
    "RankingWeights",
    "SourceScore",
    "SourceRanker",
]


