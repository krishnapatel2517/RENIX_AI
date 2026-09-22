"""
RENIX Fact Checker

Verifies claims by comparing them against multiple research sources.

Responsibilities:
- Extract claims from research text
- Compare claims against sources
- Identify supporting and contradicting evidence
- Calculate confidence
- Detect conflicting information
- Produce structured verification results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import re


@dataclass
class Evidence:
    """Evidence associated with a claim."""

    source: Any
    supports: bool
    strength: float
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "supports": self.supports,
            "strength": self.strength,
            "explanation": self.explanation,
        }


@dataclass
class FactCheckResult:
    """Result of checking one claim."""

    claim: str
    status: str
    confidence: float
    supporting_evidence: list[Evidence] = field(
        default_factory=list
    )
    contradicting_evidence: list[Evidence] = field(
        default_factory=list
    )
    neutral_evidence: list[Evidence] = field(
        default_factory=list
    )
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status,
            "confidence": self.confidence,
            "supporting_evidence": [
                item.to_dict()
                for item in self.supporting_evidence
            ],
            "contradicting_evidence": [
                item.to_dict()
                for item in self.contradicting_evidence
            ],
            "neutral_evidence": [
                item.to_dict()
                for item in self.neutral_evidence
            ],
            "explanation": self.explanation,
        }


class FactChecker:
    """
    Rule-based fact verification engine.

    This component does not pretend that keyword matching is
    equivalent to human-level fact checking. It provides a
    transparent evidence layer that can later be connected to
    RENIX's LLM reasoning and research systems.
    """

    STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
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
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "their",
        "there",
        "here",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "not",
        "than",
        "then",
        "also",
        "into",
        "about",
        "which",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
    }

    NEGATION_WORDS = {
        "not",
        "never",
        "no",
        "none",
        "neither",
        "without",
        "false",
        "incorrect",
        "wrong",
        "deny",
        "denied",
        "reject",
        "rejected",
    }

    SUPPORT_WORDS = {
        "confirmed",
        "confirms",
        "verified",
        "supports",
        "supported",
        "evidence",
        "shows",
        "demonstrates",
        "according",
        "reported",
        "found",
        "proves",
        "true",
        "correct",
    }

    CONTRADICTION_WORDS = {
        "false",
        "incorrect",
        "wrong",
        "contradicts",
        "contradicted",
        "disproves",
        "denies",
        "denied",
        "debunked",
        "myth",
        "misleading",
        "untrue",
    }

    def __init__(
        self,
        *,
        support_threshold: float = 0.60,
        contradiction_threshold: float = 0.60,
    ) -> None:

        self.support_threshold = self._clamp(
            support_threshold
        )

        self.contradiction_threshold = self._clamp(
            contradiction_threshold
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def check(
        self,
        claim: str,
        sources: Iterable[Any],
    ) -> FactCheckResult:

        claim = self._clean_claim(
            claim
        )

        if not claim:
            return FactCheckResult(
                claim="",
                status="unknown",
                confidence=0.0,
                explanation="No claim was provided.",
            )

        source_list = list(
            sources or []
        )

        if not source_list:
            return FactCheckResult(
                claim=claim,
                status="unverified",
                confidence=0.0,
                explanation=(
                    "No evidence sources were provided."
                ),
            )

        supporting = []
        contradicting = []
        neutral = []

        for source in source_list:

            evidence = self._evaluate_source(
                claim,
                source,
            )

            if evidence.supports:
                supporting.append(
                    evidence
                )

            elif evidence.strength >= (
                self.contradiction_threshold
            ):
                contradicting.append(
                    evidence
                )

            else:
                neutral.append(
                    evidence
                )

        confidence = self._calculate_confidence(
            supporting,
            contradicting,
            source_list,
        )

        status = self._determine_status(
            supporting,
            contradicting,
            confidence,
        )

        explanation = (
            self._build_explanation(
                status,
                confidence,
                supporting,
                contradicting,
                neutral,
            )
        )

        return FactCheckResult(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            neutral_evidence=neutral,
            explanation=explanation,
        )

    def check_many(
        self,
        claims: Iterable[str],
        sources: Iterable[Any],
    ) -> list[FactCheckResult]:

        source_list = list(
            sources or []
        )

        return [
            self.check(
                claim,
                source_list,
            )
            for claim in claims
            if self._clean_claim(
                claim
            )
        ]

    def verify_text(
        self,
        text: str,
        sources: Iterable[Any],
    ) -> list[FactCheckResult]:

        claims = self.extract_claims(
            text
        )

        return self.check_many(
            claims,
            sources,
        )

    # ============================================================
    # CLAIM EXTRACTION
    # ============================================================

    def extract_claims(
        self,
        text: str,
    ) -> list[str]:

        text = (
            text or ""
        ).strip()

        if not text:
            return []

        # Split on sentence boundaries.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        claims = []

        for sentence in sentences:

            sentence = self._clean_claim(
                sentence
            )

            if not sentence:
                continue

            if len(
                self._content_words(
                    sentence
                )
            ) < 3:
                continue

            claims.append(
                sentence
            )

        return self._deduplicate_strings(
            claims
        )

    # ============================================================
    # SOURCE EVALUATION
    # ============================================================

    def _evaluate_source(
        self,
        claim: str,
        source: Any,
    ) -> Evidence:

        source_text = self._source_text(
            source
        )

        if not source_text:

            return Evidence(
                source=source,
                supports=False,
                strength=0.0,
                explanation=(
                    "Source contains no usable text."
                ),
            )

        similarity = (
            self._semantic_overlap(
                claim,
                source_text,
            )
        )

        direct_match = (
            self._direct_claim_match(
                claim,
                source_text,
            )
        )

        support_language = (
            self._support_language_score(
                source_text
            )
        )

        contradiction_language = (
            self._contradiction_language_score(
                source_text
            )
        )

        credibility = self._safe_float(
            self._get(
                source,
                "credibility",
                0.5,
            )
        )

        credibility = self._clamp(
            credibility
        )

        support_score = (
            similarity * 0.40
            + direct_match * 0.30
            + support_language * 0.10
            + credibility * 0.20
        )

        contradiction_score = (
            similarity * 0.35
            + contradiction_language * 0.30
            + credibility * 0.15
            + self._negation_alignment(
                claim,
                source_text,
            ) * 0.20
        )

        if contradiction_score > (
            support_score + 0.10
        ):
            return Evidence(
                source=source,
                supports=False,
                strength=self._clamp(
                    contradiction_score
                ),
                explanation=(
                    "The source contains language "
                    "or evidence that may contradict "
                    "the claim."
                ),
            )

        if support_score >= (
            self.support_threshold
        ):

            return Evidence(
                source=source,
                supports=True,
                strength=self._clamp(
                    support_score
                ),
                explanation=(
                    "The source provides evidence "
                    "consistent with the claim."
                ),
            )

        return Evidence(
            source=source,
            supports=False,
            strength=self._clamp(
                max(
                    similarity,
                    direct_match,
                )
            ),
            explanation=(
                "The source is relevant but does "
                "not provide sufficiently strong "
                "evidence to verify the claim."
            ),
        )

    # ============================================================
    # TEXT SIMILARITY
    # ============================================================

    def _semantic_overlap(
        self,
        claim: str,
        source_text: str,
    ) -> float:

        claim_words = set(
            self._content_words(
                claim
            )
        )

        source_words = set(
            self._content_words(
                source_text
            )
        )

        if not claim_words:
            return 0.0

        overlap = len(
            claim_words
            & source_words
        )

        return self._clamp(
            overlap
            / len(claim_words)
        )

    def _direct_claim_match(
        self,
        claim: str,
        source_text: str,
    ) -> float:

        claim_normalized = (
            self._normalize_text(
                claim
            )
        )

        source_normalized = (
            self._normalize_text(
                source_text
            )
        )

        if not claim_normalized:
            return 0.0

        if (
            claim_normalized
            in source_normalized
        ):
            return 1.0

        claim_words = (
            claim_normalized.split()
        )

        if len(claim_words) < 3:
            return 0.0

        phrase_sizes = [
            5,
            4,
            3,
        ]

        matches = 0
        total = 0

        for size in phrase_sizes:

            if len(
                claim_words
            ) < size:
                continue

            for index in range(
                len(claim_words)
                - size
                + 1
            ):

                phrase = " ".join(
                    claim_words[
                        index:index + size
                    ]
                )

                total += 1

                if phrase in source_normalized:
                    matches += 1

        if total == 0:
            return 0.0

        return self._clamp(
            matches / total
        )

    # ============================================================
    # LANGUAGE ANALYSIS
    # ============================================================

    def _support_language_score(
        self,
        text: str,
    ) -> float:

        words = set(
            self._content_words(
                text
            )
        )

        if not words:
            return 0.0

        matches = len(
            words
            & self.SUPPORT_WORDS
        )

        return self._clamp(
            matches / 3
        )

    def _contradiction_language_score(
        self,
        text: str,
    ) -> float:

        words = set(
            self._content_words(
                text
            )
        )

        if not words:
            return 0.0

        matches = len(
            words
            & self.CONTRADICTION_WORDS
        )

        return self._clamp(
            matches / 3
        )

    def _negation_alignment(
        self,
        claim: str,
        source_text: str,
    ) -> float:

        claim_words = set(
            self._content_words(
                claim
            )
        )

        source_words = set(
            self._content_words(
                source_text
            )
        )

        claim_negated = bool(
            claim_words
            & self.NEGATION_WORDS
        )

        source_negated = bool(
            source_words
            & self.NEGATION_WORDS
        )

        if claim_negated != source_negated:
            return 0.75

        if (
            claim_negated
            and source_negated
        ):
            return 0.15

        return 0.0

    # ============================================================
    # CONFIDENCE
    # ============================================================

    def _calculate_confidence(
        self,
        supporting: list[Evidence],
        contradicting: list[Evidence],
        sources: list[Any],
    ) -> float:

        if not sources:
            return 0.0

        support_strength = sum(
            item.strength
            for item in supporting
        )

        contradiction_strength = sum(
            item.strength
            for item in contradicting
        )

        support_count = len(
            supporting
        )

        contradiction_count = len(
            contradicting
        )

        total_sources = len(
            sources
        )

        support_ratio = (
            support_count
            / total_sources
        )

        contradiction_ratio = (
            contradiction_count
            / total_sources
        )

        average_support = (
            support_strength
            / support_count
            if support_count
            else 0.0
        )

        average_contradiction = (
            contradiction_strength
            / contradiction_count
            if contradiction_count
            else 0.0
        )

        confidence = (
            support_ratio * 0.35
            + average_support * 0.35
            + (
                1.0
                - contradiction_ratio
            ) * 0.15
            + (
                1.0
                - average_contradiction
            ) * 0.15
        )

        return self._clamp(
            confidence
        )

    def _determine_status(
        self,
        supporting: list[Evidence],
        contradicting: list[Evidence],
        confidence: float,
    ) -> str:

        support_strength = sum(
            item.strength
            for item in supporting
        )

        contradiction_strength = sum(
            item.strength
            for item in contradicting
        )

        if (
            supporting
            and contradicting
        ):

            if (
                abs(
                    support_strength
                    - contradiction_strength
                )
                < 0.20
            ):
                return "conflicting"

            if (
                support_strength
                > contradiction_strength
            ):
                return "mostly_supported"

            return "mostly_contradicted"

        if supporting:

            if confidence >= 0.75:
                return "supported"

            return "partially_supported"

        if contradicting:

            if contradiction_strength >= 0.75:
                return "contradicted"

            return "partially_contradicted"

        return "unverified"

    # ============================================================
    # EXPLANATION
    # ============================================================

    def _build_explanation(
        self,
        status: str,
        confidence: float,
        supporting: list[Evidence],
        contradicting: list[Evidence],
        neutral: list[Evidence],
    ) -> str:

        status_messages = {
            "supported": (
                "The available evidence strongly "
                "supports this claim."
            ),
            "partially_supported": (
                "The available evidence supports "
                "parts of this claim, but confidence "
                "is limited."
            ),
            "mostly_supported": (
                "More evidence supports the claim "
                "than contradicts it."
            ),
            "contradicted": (
                "The available evidence strongly "
                "contradicts this claim."
            ),
            "partially_contradicted": (
                "Some evidence contradicts the claim, "
                "but the evidence is not conclusive."
            ),
            "mostly_contradicted": (
                "More evidence contradicts the claim "
                "than supports it."
            ),
            "conflicting": (
                "The sources contain meaningful "
                "conflicting evidence."
            ),
            "unverified": (
                "The available sources do not provide "
                "enough evidence to verify the claim."
            ),
            "unknown": (
                "The claim could not be evaluated."
            ),
        }

        message = status_messages.get(
            status,
            status_messages[
                "unknown"
            ],
        )

        return (
            f"{message} "
            f"Confidence: "
            f"{confidence * 100:.1f}%. "
            f"Supporting sources: "
            f"{len(supporting)}. "
            f"Contradicting sources: "
            f"{len(contradicting)}. "
            f"Neutral sources: "
            f"{len(neutral)}."
        )

    # ============================================================
    # SOURCE TEXT
    # ============================================================

    def _source_text(
        self,
        source: Any,
    ) -> str:

        fields = [
            "title",
            "snippet",
            "content",
            "publisher",
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
    # UTILITIES
    # ============================================================

    @staticmethod
    def _clean_claim(
        claim: str,
    ) -> str:

        claim = str(
            claim or ""
        ).strip()

        claim = re.sub(
            r"\s+",
            " ",
            claim,
        )

        return claim

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:

        text = str(
            text or ""
        ).lower()

        text = re.sub(
            r"[^\w\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

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
    def _deduplicate_strings(
        values: Iterable[str],
    ) -> list[str]:

        seen = set()
        result = []

        for value in values:

            key = value.lower().strip()

            if key in seen:
                continue

            seen.add(key)
            result.append(value)

        return result

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
    "Evidence",
    "FactCheckResult",
    "FactChecker",
]


