"""
RENIX Research Comparison Engine

Compares multiple research sources, products, ideas, claims,
documents, or other structured/unstructured information.

Responsibilities:
- Compare sources side-by-side
- Identify similarities and differences
- Extract comparable attributes
- Detect conflicting claims
- Score options
- Produce structured comparison results
- Generate human-readable comparison summaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import re


@dataclass
class ComparisonAttribute:
    """One attribute being compared."""

    name: str
    values: dict[str, Any] = field(
        default_factory=dict
    )
    winner: str | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": dict(self.values),
            "winner": self.winner,
            "explanation": self.explanation,
        }


@dataclass
class ComparisonResult:
    """Complete comparison result."""

    title: str
    items: list[Any] = field(
        default_factory=list
    )
    attributes: list[ComparisonAttribute] = field(
        default_factory=list
    )
    similarities: list[str] = field(
        default_factory=list
    )
    differences: list[str] = field(
        default_factory=list
    )
    rankings: list[str] = field(
        default_factory=list
    )
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "attributes": [
                attribute.to_dict()
                for attribute in self.attributes
            ],
            "similarities": list(
                self.similarities
            ),
            "differences": list(
                self.differences
            ),
            "rankings": list(
                self.rankings
            ),
            "summary": self.summary,
        }


class ComparisonEngine:
    """
    Generic comparison engine for RENIX.

    Works with dictionaries, dataclasses, objects,
    research sources, and plain text.
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
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "they",
        "their",
        "them",
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
        "will",
        "may",
        "might",
        "than",
        "then",
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
        "while",
        "which",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
    }

    DEFAULT_ATTRIBUTES = [
        "price",
        "cost",
        "rating",
        "score",
        "performance",
        "quality",
        "features",
        "size",
        "weight",
        "speed",
        "battery",
        "capacity",
        "availability",
        "date",
        "release_date",
        "publisher",
        "author",
        "category",
    ]

    def __init__(
        self,
        *,
        attributes: Iterable[str] | None = None,
    ) -> None:

        self.attributes = list(
            attributes
            or self.DEFAULT_ATTRIBUTES
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def compare(
        self,
        items: Iterable[Any],
        *,
        title: str = "Comparison",
        attributes: Iterable[str] | None = None,
    ) -> ComparisonResult:

        item_list = list(
            items or []
        )

        if len(item_list) < 2:

            return ComparisonResult(
                title=title,
                items=item_list,
                summary=(
                    "At least two items are "
                    "required for comparison."
                ),
            )

        labels = self._build_labels(
            item_list
        )

        attribute_names = (
            list(attributes)
            if attributes
            else self.detect_attributes(
                item_list
            )
        )

        comparison_attributes = []

        for attribute_name in attribute_names:

            values = {}

            for index, item in enumerate(
                item_list
            ):

                label = labels[index]

                values[label] = (
                    self.get_attribute(
                        item,
                        attribute_name,
                    )
                )

            winner = self._find_winner(
                values,
                attribute_name,
            )

            explanation = (
                self._attribute_explanation(
                    attribute_name,
                    values,
                    winner,
                )
            )

            comparison_attributes.append(
                ComparisonAttribute(
                    name=attribute_name,
                    values=values,
                    winner=winner,
                    explanation=explanation,
                )
            )

        similarities = (
            self.find_similarities(
                item_list
            )
        )

        differences = (
            self.find_differences(
                item_list
            )
        )

        rankings = (
            self.rank_items(
                item_list,
                comparison_attributes,
            )
        )

        summary = (
            self.generate_summary(
                item_list,
                comparison_attributes,
                rankings,
            )
        )

        return ComparisonResult(
            title=title,
            items=item_list,
            attributes=comparison_attributes,
            similarities=similarities,
            differences=differences,
            rankings=rankings,
            summary=summary,
        )

    def compare_sources(
        self,
        sources: Iterable[Any],
        *,
        title: str = "Source Comparison",
    ) -> ComparisonResult:

        source_list = list(
            sources or []
        )

        return self.compare(
            source_list,
            title=title,
        )

    # ============================================================
    # ATTRIBUTE DETECTION
    # ============================================================

    def detect_attributes(
        self,
        items: Iterable[Any],
    ) -> list[str]:

        item_list = list(
            items or []
        )

        if not item_list:
            return []

        discovered = []

        # First inspect structured objects.
        for item in item_list:

            if isinstance(
                item,
                dict,
            ):

                for key in item.keys():

                    if key not in discovered:
                        discovered.append(
                            str(key)
                        )

            else:

                if hasattr(
                    item,
                    "__dict__",
                ):

                    for key in vars(
                        item
                    ).keys():

                        if (
                            key.startswith("_")
                        ):
                            continue

                        if key not in discovered:
                            discovered.append(
                                key
                            )

        # Prioritize known comparison fields.
        prioritized = []

        for attribute in self.attributes:

            if attribute in discovered:
                prioritized.append(
                    attribute
                )

        # Add remaining fields.
        for attribute in discovered:

            if attribute not in prioritized:
                prioritized.append(
                    attribute
                )

        return prioritized[:30]

    # ============================================================
    # ATTRIBUTE ACCESS
    # ============================================================

    @staticmethod
    def get_attribute(
        item: Any,
        attribute: str,
    ) -> Any:

        if isinstance(
            item,
            dict,
        ):
            return item.get(
                attribute
            )

        return getattr(
            item,
            attribute,
            None,
        )

    # ============================================================
    # SIMILARITIES
    # ============================================================

    def find_similarities(
        self,
        items: Iterable[Any],
    ) -> list[str]:

        item_list = list(
            items or []
        )

        if len(item_list) < 2:
            return []

        text_values = [
            self._item_text(item)
            for item in item_list
        ]

        token_sets = [
            set(
                self._content_words(
                    text
                )
            )
            for text in text_values
        ]

        if not token_sets:
            return []

        common_words = set.intersection(
            *token_sets
        )

        common_words -= self.STOPWORDS

        similarities = []

        if common_words:

            common = sorted(
                common_words,
                key=len,
                reverse=True,
            )[:10]

            similarities.append(
                "All items share these concepts: "
                + ", ".join(common)
                + "."
            )

        # Compare structured attributes.
        common_attributes = (
            self._common_attributes(
                item_list
            )
        )

        if common_attributes:

            similarities.append(
                "All compared items provide "
                "information for: "
                + ", ".join(
                    common_attributes[:10]
                )
                + "."
            )

        return similarities

    # ============================================================
    # DIFFERENCES
    # ============================================================

    def find_differences(
        self,
        items: Iterable[Any],
    ) -> list[str]:

        item_list = list(
            items or []
        )

        if len(item_list) < 2:
            return []

        differences = []

        common_attributes = (
            self._common_attributes(
                item_list
            )
        )

        labels = self._build_labels(
            item_list
        )

        for attribute in common_attributes:

            values = {}

            for index, item in enumerate(
                item_list
            ):

                values[
                    labels[index]
                ] = self.get_attribute(
                    item,
                    attribute,
                )

            normalized = {
                self._normalize_value(
                    value
                )
                for value in values.values()
            }

            if len(normalized) <= 1:
                continue

            differences.append(
                self._format_difference(
                    attribute,
                    values,
                )
            )

        return differences

    # ============================================================
    # RANKING
    # ============================================================

    def rank_items(
        self,
        items: Iterable[Any],
        attributes: Iterable[
            ComparisonAttribute
        ],
    ) -> list[str]:

        item_list = list(
            items or []
        )

        if not item_list:
            return []

        labels = self._build_labels(
            item_list
        )

        scores = {
            label: 0.0
            for label in labels
        }

        for attribute in attributes:

            if not attribute.winner:
                continue

            scores[
                attribute.winner
            ] += 1.0

        ordered = sorted(
            scores.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return [
            f"{index + 1}. {label} "
            f"({score:.0f} attribute wins)"
            for index, (
                label,
                score,
            ) in enumerate(ordered)
        ]

    # ============================================================
    # WINNER DETECTION
    # ============================================================

    def _find_winner(
        self,
        values: dict[str, Any],
        attribute: str,
    ) -> str | None:

        valid = {
            label: value
            for label, value
            in values.items()
            if value is not None
            and value != ""
        }

        if len(valid) < 2:
            return None

        numeric_values = {}

        for label, value in valid.items():

            number = self._extract_number(
                value
            )

            if number is not None:
                numeric_values[
                    label
                ] = number

        if len(
            numeric_values
        ) >= 2:

            lowered = attribute.lower()

            lower_is_better = any(
                term in lowered
                for term in (
                    "price",
                    "cost",
                    "weight",
                    "latency",
                    "delay",
                    "time",
                )
            )

            if lower_is_better:
                return min(
                    numeric_values,
                    key=numeric_values.get,
                )

            return max(
                numeric_values,
                key=numeric_values.get,
            )

        # For text, prefer the most informative value.
        return max(
            valid,
            key=lambda label: len(
                str(
                    valid[label]
                )
            ),
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    def generate_summary(
        self,
        items: list[Any],
        attributes: list[ComparisonAttribute],
        rankings: list[str],
    ) -> str:

        if not items:
            return "No items were provided."

        if not attributes:
            return (
                "The items could not be compared "
                "using structured attributes."
            )

        winners = [
            attribute
            for attribute in attributes
            if attribute.winner
        ]

        if not winners:
            return (
                f"Compared {len(items)} items, "
                "but no clear attribute winners "
                "could be determined."
            )

        winner_counts: dict[
            str,
            int,
        ] = {}

        for attribute in winners:

            winner_counts[
                attribute.winner
            ] = (
                winner_counts.get(
                    attribute.winner,
                    0,
                )
                + 1
            )

        best = max(
            winner_counts,
            key=winner_counts.get,
        )

        count = winner_counts[
            best
        ]

        return (
            f"{best} performed best across "
            f"{count} comparable attribute"
            f"{'s' if count != 1 else ''}. "
            f"The comparison covers "
            f"{len(winners)} attributes with "
            "determinable differences."
        )

    # ============================================================
    # FORMATTING
    # ============================================================

    def _attribute_explanation(
        self,
        attribute: str,
        values: dict[str, Any],
        winner: str | None,
    ) -> str:

        if not winner:
            return (
                f"No clear winner could be "
                f"determined for {attribute}."
            )

        return (
            f"{winner} has the strongest "
            f"comparable value for {attribute}."
        )

    def _format_difference(
        self,
        attribute: str,
        values: dict[str, Any],
    ) -> str:

        parts = []

        for label, value in values.items():

            parts.append(
                f"{label}: {value}"
            )

        return (
            f"{attribute}: "
            + "; ".join(parts)
            + "."
        )

    # ============================================================
    # STRUCTURED DATA
    # ============================================================

    def _common_attributes(
        self,
        items: list[Any],
    ) -> list[str]:

        attribute_sets = []

        for item in items:

            if isinstance(
                item,
                dict,
            ):

                attribute_sets.append(
                    set(
                        item.keys()
                    )
                )

            elif hasattr(
                item,
                "__dict__",
            ):

                attribute_sets.append(
                    {
                        key
                        for key in vars(
                            item
                        ).keys()
                        if not key.startswith(
                            "_"
                        )
                    }
                )

        if not attribute_sets:
            return []

        return list(
            set.intersection(
                *attribute_sets
            )
        )

    # ============================================================
    # TEXT REPRESENTATION
    # ============================================================

    def _item_text(
        self,
        item: Any,
    ) -> str:

        if isinstance(
            item,
            str,
        ):
            return item

        if isinstance(
            item,
            dict,
        ):

            return " ".join(
                str(value)
                for value in item.values()
                if value is not None
            )

        if hasattr(
            item,
            "__dict__",
        ):

            return " ".join(
                str(value)
                for value in vars(
                    item
                ).values()
                if value is not None
            )

        return str(
            item
        )

    # ============================================================
    # LABELS
    # ============================================================

    def _build_labels(
        self,
        items: list[Any],
    ) -> list[str]:

        labels = []

        for index, item in enumerate(
            items
        ):

            label = None

            for field in (
                "name",
                "title",
                "label",
                "id",
            ):

                value = self.get_attribute(
                    item,
                    field,
                )

                if value:
                    label = str(
                        value
                    )
                    break

            if not label:
                label = (
                    f"Item {index + 1}"
                )

            if label in labels:
                label = (
                    f"{label} "
                    f"({index + 1})"
                )

            labels.append(
                label
            )

        return labels

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_value(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(
                value
            ).strip().lower(),
        )

    @staticmethod
    def _extract_number(
        value: Any,
    ) -> float | None:

        if isinstance(
            value,
            bool,
        ):
            return None

        if isinstance(
            value,
            (int, float),
        ):
            return float(
                value
            )

        if not isinstance(
            value,
            str,
        ):
            return None

        # Prefer the first meaningful numeric value.
        match = re.search(
            r"-?\d+(?:\.\d+)?",
            value.replace(
                ",",
                "",
            ),
        )

        if not match:
            return None

        try:
            return float(
                match.group()
            )
        except ValueError:
            return None

    @staticmethod
    def _content_words(
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
                word not in ComparisonEngine.STOPWORDS
                and len(word) > 2
            )
        ]


__all__ = [
    "ComparisonAttribute",
    "ComparisonResult",
    "ComparisonEngine",
]


