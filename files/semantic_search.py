"""
RENIX Semantic File Search
==========================

Semantic-aware search layer for RENIX.

This module provides:
- Text extraction from common text-based files
- Keyword and phrase matching
- Lightweight relevance scoring
- Filename + content ranking
- Extension filtering
- Search result snippets
- Optional embedding backend
- Graceful fallback when an embedding model is unavailable

The module is designed so RENIX can later connect it to:
    memory/embeddings.py
    memory/vector_store.py
    memory/retrieval.py

It does NOT require an external embedding model for basic operation.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# ==============================================================
# CONSTANTS
# ==============================================================

DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024
DEFAULT_SNIPPET_LENGTH = 240

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".ini",
    ".cfg",
    ".conf",
    ".log",
    ".csv",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


# ==============================================================
# DATA MODEL
# ==============================================================


@dataclass
class SemanticSearchResult:
    """A ranked semantic-search result."""

    path: str
    name: str
    score: float
    filename_score: float
    content_score: float
    phrase_score: float
    matched_terms: list[str]
    snippet: str
    extension: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================
# SEMANTIC SEARCH ENGINE
# ==============================================================


class SemanticFileSearch:
    """
    RENIX semantic-aware filesystem search.

    Basic operation does not require an embedding model.

    Optional embedding search can be enabled by providing a
    callable:

        embedding_provider(text) -> list[float]

    The provider can later be connected to RENIX's embedding
    subsystem.
    """

    def __init__(
        self,
        enabled: bool = True,
        case_sensitive: bool = False,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        embedding_provider: Optional[
            Callable[[str], list[float]]
        ] = None,
    ) -> None:

        self.enabled = enabled
        self.case_sensitive = case_sensitive
        self.max_file_size = max_file_size
        self.embedding_provider = (
            embedding_provider
        )

        logger.info(
            "RENIX SemanticFileSearch initialized."
        )

    # ==========================================================
    # STATE
    # ==========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def _check(self) -> None:

        if not self.enabled:
            raise RuntimeError(
                "RENIX SemanticFileSearch is disabled."
            )

    # ==========================================================
    # NORMALIZATION
    # ==========================================================

    def normalize(
        self,
        text: str,
    ) -> str:

        text = str(text)

        if not self.case_sensitive:
            text = text.lower()

        return text

    # ==========================================================
    # TOKENIZATION
    # ==========================================================

    def tokenize(
        self,
        text: str,
    ) -> list[str]:

        text = self.normalize(text)

        tokens = re.findall(
            r"[a-zA-Z0-9_]+",
            text,
        )

        return [
            token
            for token in tokens
            if token not in STOP_WORDS
        ]

    # ==========================================================
    # TEXT EXTRACTION
    # ==========================================================

    def can_read_text_file(
        self,
        path: str | Path,
    ) -> bool:

        file_path = Path(path)

        if not file_path.is_file():
            return False

        if file_path.suffix.lower() in TEXT_EXTENSIONS:
            return True

        return False

    def read_text(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
    ) -> str:

        self._check()

        file_path = Path(path)

        if not file_path.is_file():
            return ""

        try:

            if (
                file_path.stat().st_size
                > self.max_file_size
            ):
                return ""

        except OSError:
            return ""

        if not self.can_read_text_file(
            file_path
        ):
            return ""

        encodings = [
            encoding,
            "utf-8-sig",
            "utf-16",
            "latin-1",
        ]

        for current_encoding in encodings:

            try:

                return file_path.read_text(
                    encoding=current_encoding
                )

            except (
                UnicodeDecodeError,
                LookupError,
            ):
                continue

            except OSError:
                return ""

        return ""

    # ==========================================================
    # KEYWORD SCORE
    # ==========================================================

    def keyword_score(
        self,
        query: str,
        text: str,
    ) -> tuple[float, list[str]]:

        query_tokens = self.tokenize(
            query
        )

        text_tokens = self.tokenize(
            text
        )

        if not query_tokens or not text_tokens:
            return 0.0, []

        text_counter = Counter(
            text_tokens
        )

        matched_terms: list[str] = []

        total_score = 0.0

        for token in set(query_tokens):

            count = text_counter.get(
                token,
                0,
            )

            if count > 0:

                matched_terms.append(
                    token
                )

                # Diminishing returns for repeated
                # occurrences.
                contribution = (
                    1.0
                    + math.log1p(count)
                )

                total_score += contribution

        maximum = len(
            set(query_tokens)
        ) * 2.5

        if maximum <= 0:
            return 0.0, matched_terms

        score = min(
            100.0,
            (total_score / maximum) * 100.0,
        )

        return score, matched_terms

    # ==========================================================
    # PHRASE SCORE
    # ==========================================================

    def phrase_score(
        self,
        query: str,
        text: str,
    ) -> float:

        normalized_query = (
            self.normalize(
                query
            ).strip()
        )

        normalized_text = (
            self.normalize(
                text
            )
        )

        if not normalized_query:
            return 0.0

        if normalized_query in normalized_text:
            return 100.0

        query_tokens = self.tokenize(
            query
        )

        if not query_tokens:
            return 0.0

        # Try ordered partial phrase matching.
        escaped = [
            re.escape(token)
            for token in query_tokens
        ]

        pattern = r".{0,80}".join(
            escaped
        )

        if re.search(
            pattern,
            normalized_text,
        ):
            return 65.0

        return 0.0

    # ==========================================================
    # FILENAME SCORE
    # ==========================================================

    def filename_score(
        self,
        query: str,
        filename: str,
    ) -> float:

        normalized_query = self.normalize(
            query
        ).strip()

        normalized_filename = self.normalize(
            filename
        ).strip()

        if not normalized_query:
            return 0.0

        if (
            normalized_filename
            == normalized_query
        ):
            return 100.0

        stem = Path(
            normalized_filename
        ).stem

        if stem == normalized_query:
            return 100.0

        if stem.startswith(
            normalized_query
        ):
            return 90.0

        if normalized_query in stem:
            return 80.0

        query_tokens = set(
            self.tokenize(query)
        )

        filename_tokens = set(
            self.tokenize(stem)
        )

        if not query_tokens:
            return 0.0

        overlap = (
            query_tokens
            & filename_tokens
        )

        if not overlap:
            return 0.0

        return (
            len(overlap)
            / len(query_tokens)
            * 70.0
        )

    # ==========================================================
    # SNIPPET GENERATION
    # ==========================================================

    def create_snippet(
        self,
        text: str,
        query: str,
        *,
        max_length: int = DEFAULT_SNIPPET_LENGTH,
    ) -> str:

        if not text:
            return ""

        normalized_text = self.normalize(
            text
        )

        normalized_query = self.normalize(
            query
        ).strip()

        position = (
            normalized_text.find(
                normalized_query
            )
            if normalized_query
            else -1
        )

        if position < 0:

            tokens = self.tokenize(
                query
            )

            position = -1

            for token in tokens:

                position = normalized_text.find(
                    token
                )

                if position >= 0:
                    break

        if position < 0:

            snippet = text[:max_length]

            return (
                snippet.strip()
                + ("..." if len(text) > max_length else "")
            )

        half = max_length // 2

        start = max(
            0,
            position - half,
        )

        end = min(
            len(text),
            start + max_length,
        )

        snippet = text[
            start:end
        ].strip()

        if start > 0:
            snippet = "..." + snippet

        if end < len(text):
            snippet += "..."

        return snippet

    # ==========================================================
    # SCORE DOCUMENT
    # ==========================================================

    def score_document(
        self,
        query: str,
        path: str | Path,
        content: Optional[str] = None,
    ) -> Optional[SemanticSearchResult]:

        self._check()

        file_path = Path(path)

        if not file_path.is_file():
            return None

        if content is None:

            content = self.read_text(
                file_path
            )

        filename_score = (
            self.filename_score(
                query,
                file_path.name,
            )
        )

        content_score, matched_terms = (
            self.keyword_score(
                query,
                content,
            )
        )

        phrase_score = (
            self.phrase_score(
                query,
                content,
            )
        )

        # Weighted relevance score.
        final_score = (
            filename_score * 0.40
            + content_score * 0.40
            + phrase_score * 0.20
        )

        if final_score <= 0:
            return None

        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0

        snippet = self.create_snippet(
            content,
            query,
        )

        return SemanticSearchResult(
            path=str(file_path.resolve()),
            name=file_path.name,
            score=round(
                final_score,
                2,
            ),
            filename_score=round(
                filename_score,
                2,
            ),
            content_score=round(
                content_score,
                2,
            ),
            phrase_score=round(
                phrase_score,
                2,
            ),
            matched_terms=sorted(
                matched_terms
            ),
            snippet=snippet,
            extension=file_path.suffix,
            size=size,
        )

    # ==========================================================
    # SEARCH
    # ==========================================================

    def search(
        self,
        query: str,
        root: str | Path = ".",
        *,
        recursive: bool = True,
        extensions: Optional[
            Iterable[str]
        ] = None,
        include_hidden: bool = False,
        limit: int = 50,
        minimum_score: float = 1.0,
    ) -> list[dict[str, Any]]:

        self._check()

        query = str(query).strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        root_path = Path(
            root
        ).expanduser()

        if not root_path.exists():
            raise FileNotFoundError(
                str(root_path)
            )

        if not root_path.is_dir():
            raise NotADirectoryError(
                str(root_path)
            )

        normalized_extensions = None

        if extensions is not None:

            normalized_extensions = set()

            for extension in extensions:

                extension = str(
                    extension
                ).strip().lower()

                if extension and not extension.startswith(
                    "."
                ):
                    extension = "." + extension

                normalized_extensions.add(
                    extension
                )

        if recursive:

            iterator: Iterable[Path] = (
                root_path.rglob("*")
            )

        else:

            iterator = root_path.iterdir()

        results: list[
            SemanticSearchResult
        ] = []

        for path in iterator:

            try:

                if not path.is_file():
                    continue

                if (
                    not include_hidden
                    and path.name.startswith(".")
                ):
                    continue

                if (
                    normalized_extensions is not None
                    and path.suffix.lower()
                    not in normalized_extensions
                ):
                    continue

                if (
                    path.stat().st_size
                    > self.max_file_size
                ):
                    continue

                if not self.can_read_text_file(
                    path
                ):
                    continue

                result = self.score_document(
                    query,
                    path,
                )

                if result is None:
                    continue

                if (
                    result.score
                    < minimum_score
                ):
                    continue

                results.append(
                    result
                )

            except (
                OSError,
                PermissionError,
            ):

                continue

        results.sort(
            key=lambda result: (
                -result.score,
                result.name.lower(),
            )
        )

        return [
            result.to_dict()
            for result in results[:max(1, limit)]
        ]

    # ==========================================================
    # SEARCH A SINGLE FILE
    # ==========================================================

    def search_file(
        self,
        query: str,
        path: str | Path,
    ) -> Optional[dict[str, Any]]:

        result = self.score_document(
            query,
            path,
        )

        if result is None:
            return None

        return result.to_dict()

    # ==========================================================
    # EMBEDDING SUPPORT
    # ==============================================================

    def set_embedding_provider(
        self,
        provider: Optional[
            Callable[[str], list[float]]
        ],
    ) -> None:

        self.embedding_provider = provider

    def has_embedding_provider(self) -> bool:

        return (
            self.embedding_provider
            is not None
        )

    @staticmethod
    def cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:

        if not vector_a or not vector_b:
            return 0.0

        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Embedding vectors must have "
                "the same dimension."
            )

        dot = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b,
            )
        )

        norm_a = math.sqrt(
            sum(
                a * a
                for a in vector_a
            )
        )

        norm_b = math.sqrt(
            sum(
                b * b
                for b in vector_b
            )
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (
            norm_a * norm_b
        )

    def semantic_similarity(
        self,
        query: str,
        text: str,
    ) -> Optional[float]:

        if self.embedding_provider is None:
            return None

        try:

            query_vector = (
                self.embedding_provider(
                    query
                )
            )

            text_vector = (
                self.embedding_provider(
                    text
                )
            )

            similarity = (
                self.cosine_similarity(
                    query_vector,
                    text_vector,
                )
            )

            # Convert [-1, 1] to [0, 100].
            return (
                (similarity + 1.0)
                / 2.0
                * 100.0
            )

        except Exception as exc:

            logger.warning(
                "Embedding search failed: %s",
                exc,
            )

            return None

    # ==========================================================
    # HYBRID SEARCH
    # ==========================================================

    def hybrid_search(
        self,
        query: str,
        root: str | Path = ".",
        *,
        recursive: bool = True,
        limit: int = 50,
        semantic_weight: float = 0.45,
    ) -> list[dict[str, Any]]:

        """
        Combine lexical relevance with embeddings.

        If no embedding provider is configured, this safely
        falls back to normal semantic-aware search.
        """

        self._check()

        if not (
            0.0
            <= semantic_weight
            <= 1.0
        ):

            raise ValueError(
                "semantic_weight must be between "
                "0 and 1."
            )

        if self.embedding_provider is None:

            return self.search(
                query,
                root,
                recursive=recursive,
                limit=limit,
            )

        candidates = self.search(
            query,
            root,
            recursive=recursive,
            limit=max(
                limit * 5,
                50,
            ),
        )

        results = []

        for candidate in candidates:

            path = candidate["path"]

            content = self.read_text(
                path
            )

            semantic_score = (
                self.semantic_similarity(
                    query,
                    content,
                )
            )

            if semantic_score is None:
                semantic_score = 0.0

            lexical_score = float(
                candidate["score"]
            )

            final_score = (
                lexical_score
                * (1.0 - semantic_weight)
                + semantic_score
                * semantic_weight
            )

            candidate = dict(
                candidate
            )

            candidate[
                "lexical_score"
            ] = round(
                lexical_score,
                2,
            )

            candidate[
                "semantic_score"
            ] = round(
                semantic_score,
                2,
            )

            candidate[
                "score"
            ] = round(
                final_score,
                2,
            )

            results.append(
                candidate
            )

        results.sort(
            key=lambda item: -item["score"]
        )

        return results[:max(1, limit)]

    # ==========================================================
    # STATUS
    # ==========================================================

    def get_status(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "case_sensitive": self.case_sensitive,
            "max_file_size": self.max_file_size,
            "embedding_provider": (
                self.has_embedding_provider()
            ),
            "supported_text_extensions": len(
                TEXT_EXTENSIONS
            ),
        }

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX SemanticFileSearch shut down."
        )


# ==============================================================
# SHARED INSTANCE
# ==============================================================


_default_semantic_search: Optional[
    SemanticFileSearch
] = None


def get_semantic_search() -> SemanticFileSearch:
    """
    Return the shared RENIX semantic search instance.
    """

    global _default_semantic_search

    if _default_semantic_search is None:

        _default_semantic_search = (
            SemanticFileSearch()
        )

    return _default_semantic_search


# ==============================================================
# MODULE-LEVEL HELPER
# ==============================================================


def semantic_search(
    query: str,
    root: str | Path = ".",
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:

    return get_semantic_search().search(
        query,
        root,
        limit=limit,
    )


__all__ = [
    "SemanticSearchResult",
    "SemanticFileSearch",
    "get_semantic_search",
    "semantic_search",
]


