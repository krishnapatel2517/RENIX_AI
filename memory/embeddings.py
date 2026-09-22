"""
RENIX AI - Embeddings Engine

Provides a unified embedding interface for RENIX memory.

Responsibilities:
- Convert text into vector embeddings.
- Support configurable local/API embedding providers.
- Normalize vectors.
- Batch embedding generation.
- Caching.
- Similarity utilities.
- Graceful fallback when an external embedding provider is unavailable.

This module is intentionally provider-agnostic so the rest of RENIX can
request embeddings without depending directly on a particular AI service.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


logger = logging.getLogger("RENIX.memory.embeddings")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Configuration for the RENIX embedding engine."""

    provider: str = "auto"

    model: str = "text-embedding-3-small"

    api_key: Optional[str] = None
    api_url: Optional[str] = None

    dimension: int = 1536

    normalize: bool = True

    cache_enabled: bool = True
    cache_size: int = 5000

    timeout: float = 30.0

    batch_size: int = 32

    # Used only when no external embedding provider is available.
    local_dimension: int = 384

    @classmethod
    def from_environment(cls) -> "EmbeddingConfig":
        """Create configuration from environment variables."""

        provider = os.getenv(
            "RENIX_EMBEDDING_PROVIDER",
            "auto",
        )

        model = os.getenv(
            "RENIX_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

        api_key = (
            os.getenv("RENIX_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        api_url = os.getenv(
            "RENIX_EMBEDDING_API_URL"
        )

        try:
            dimension = int(
                os.getenv(
                    "RENIX_EMBEDDING_DIMENSION",
                    "1536",
                )
            )
        except ValueError:
            dimension = 1536

        try:
            cache_size = int(
                os.getenv(
                    "RENIX_EMBEDDING_CACHE_SIZE",
                    "5000",
                )
            )
        except ValueError:
            cache_size = 5000

        try:
            timeout = float(
                os.getenv(
                    "RENIX_EMBEDDING_TIMEOUT",
                    "30",
                )
            )
        except ValueError:
            timeout = 30.0

        try:
            batch_size = int(
                os.getenv(
                    "RENIX_EMBEDDING_BATCH_SIZE",
                    "32",
                )
            )
        except ValueError:
            batch_size = 32

        try:
            local_dimension = int(
                os.getenv(
                    "RENIX_LOCAL_EMBEDDING_DIMENSION",
                    "384",
                )
            )
        except ValueError:
            local_dimension = 384

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            api_url=api_url,
            dimension=dimension,
            cache_enabled=True,
            cache_size=max(1, cache_size),
            timeout=max(1.0, timeout),
            batch_size=max(1, batch_size),
            local_dimension=max(8, local_dimension),
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EmbeddingError(Exception):
    """Base exception for embedding failures."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider fails."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when an embedding has an unexpected dimension."""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def cosine_similarity(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    if not a or not b:
        return 0.0

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(y * y for y in b)
    )

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def euclidean_distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Calculate Euclidean distance between vectors."""

    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    return math.sqrt(
        sum(
            (x - y) ** 2
            for x, y in zip(a, b)
        )
    )


def normalize_vector(
    vector: Iterable[float],
) -> List[float]:
    """Return an L2-normalized vector."""

    values = [
        float(value)
        for value in vector
    ]

    if not values:
        return []

    norm = math.sqrt(
        sum(
            value * value
            for value in values
        )
    )

    if norm == 0.0:
        return values

    return [
        value / norm
        for value in values
    ]


# ---------------------------------------------------------------------------
# Deterministic local fallback
# ---------------------------------------------------------------------------

class LocalHashEmbedder:
    """
    Lightweight deterministic embedding generator.

    This is NOT intended to replace a high-quality neural embedding model.

    Its purpose is to ensure RENIX still has a deterministic vector
    representation when an external embedding model is unavailable.
    """

    def __init__(
        self,
        dimension: int = 384,
    ) -> None:
        self.dimension = max(
            8,
            int(dimension),
        )

    def embed(
        self,
        text: str,
    ) -> List[float]:
        """
        Convert text into a deterministic hashed vector.
        """

        vector = [0.0] * self.dimension

        normalized = (
            str(text)
            .strip()
            .lower()
        )

        if not normalized:
            return vector

        # Character n-grams provide better deterministic behavior than
        # simply hashing the entire string.
        tokens = self._tokens(normalized)

        for token in tokens:
            digest = hashlib.sha256(
                token.encode("utf-8")
            ).digest()

            index = int.from_bytes(
                digest[:4],
                "big",
            ) % self.dimension

            sign = (
                1.0
                if digest[4] % 2 == 0
                else -1.0
            )

            magnitude = (
                0.5
                + (
                    digest[5] / 255.0
                )
            )

            vector[index] += (
                sign * magnitude
            )

        return normalize_vector(vector)

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        return [
            self.embed(text)
            for text in texts
        ]

    @staticmethod
    def _tokens(
        text: str,
    ) -> List[str]:
        words = text.split()

        tokens: List[str] = []

        for word in words:
            tokens.append(word)

            if len(word) >= 3:
                for i in range(
                    len(word) - 2
                ):
                    tokens.append(
                        word[i:i + 3]
                    )

        # Include a few word pairs for context.
        for i in range(
            len(words) - 1
        ):
            tokens.append(
                f"{words[i]}_{words[i + 1]}"
            )

        return tokens


# ---------------------------------------------------------------------------
# API Provider
# ---------------------------------------------------------------------------

class HTTPEmbeddingProvider:
    """
    Generic HTTP embedding provider.

    Supports OpenAI-compatible embedding endpoints.

    Expected response:

        {
            "data": [
                {
                    "embedding": [...]
                }
            ]
        }

    The provider is deliberately implemented without an SDK so RENIX can
    remain lightweight and avoid coupling the memory system to one package.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_url: str,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise EmbeddingProviderError(
                "Embedding API key is missing."
            )

        if not api_url:
            raise EmbeddingProviderError(
                "Embedding API URL is missing."
            )

        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    def embed(
        self,
        text: str,
    ) -> List[float]:
        results = self.embed_batch(
            [text]
        )

        if not results:
            raise EmbeddingProviderError(
                "Embedding provider returned no vector."
            )

        return results[0]

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": list(texts),
        }

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(
                payload
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": (
                    f"Bearer {self.api_key}"
                ),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                body = response.read().decode(
                    "utf-8"
                )

        except urllib.error.HTTPError as exc:
            details = ""

            try:
                details = exc.read().decode(
                    "utf-8"
                )
            except Exception:
                pass

            raise EmbeddingProviderError(
                f"Embedding HTTP error "
                f"{exc.code}: {details}"
            ) from exc

        except urllib.error.URLError as exc:
            raise EmbeddingProviderError(
                f"Embedding network error: "
                f"{exc.reason}"
            ) from exc

        except Exception as exc:
            raise EmbeddingProviderError(
                f"Embedding request failed: {exc}"
            ) from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(
                "Embedding provider returned invalid JSON."
            ) from exc

        return self._parse_response(
            data,
            len(texts),
        )

    @staticmethod
    def _parse_response(
        data: Dict[str, Any],
        expected_count: int,
    ) -> List[List[float]]:
        raw_data = data.get("data")

        if not isinstance(
            raw_data,
            list,
        ):
            raise EmbeddingProviderError(
                "Embedding response does not contain "
                "a valid 'data' list."
            )

        # Some APIs return items in arbitrary order.
        raw_data = sorted(
            raw_data,
            key=lambda item: (
                item.get("index", 0)
                if isinstance(item, dict)
                else 0
            ),
        )

        vectors: List[List[float]] = []

        for item in raw_data:
            if not isinstance(
                item,
                dict,
            ):
                continue

            embedding = item.get(
                "embedding"
            )

            if not isinstance(
                embedding,
                list,
            ):
                continue

            vectors.append(
                [
                    float(value)
                    for value in embedding
                ]
            )

        if len(vectors) != expected_count:
            raise EmbeddingProviderError(
                "Embedding provider returned an "
                "unexpected number of vectors."
            )

        return vectors


# ---------------------------------------------------------------------------
# Embedding Cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Thread-safe in-memory embedding cache."""

    def __init__(
        self,
        max_size: int = 5000,
    ) -> None:
        self.max_size = max(
            1,
            int(max_size),
        )

        self._cache: Dict[
            str,
            List[float]
        ] = {}

        self._lock = threading.RLock()

    def get(
        self,
        key: str,
    ) -> Optional[List[float]]:
        with self._lock:
            value = self._cache.get(key)

            if value is None:
                return None

            return list(value)

    def set(
        self,
        key: str,
        vector: Sequence[float],
    ) -> None:
        with self._lock:
            self._cache[key] = list(vector)

            while len(
                self._cache
            ) > self.max_size:
                oldest_key = next(
                    iter(self._cache)
                )

                del self._cache[
                    oldest_key
                ]

    def delete(
        self,
        key: str,
    ) -> None:
        with self._lock:
            self._cache.pop(
                key,
                None,
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Main embedding service used by RENIX memory.

    Typical usage:

        engine = EmbeddingEngine()

        vector = engine.embed(
            "RENIX should remember this."
        )

        results = engine.similarity(
            vector,
            another_vector,
        )
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
    ) -> None:
        self.config = (
            config
            or EmbeddingConfig.from_environment()
        )

        self._lock = threading.RLock()

        self.cache = EmbeddingCache(
            self.config.cache_size
        )

        self.local_provider = (
            LocalHashEmbedder(
                self.config.local_dimension
            )
        )

        self.remote_provider = (
            self._create_remote_provider()
        )

        self.provider_name = (
            self._select_provider()
        )

    # ------------------------------------------------------------------
    # Provider setup
    # ------------------------------------------------------------------

    def _create_remote_provider(
        self,
    ) -> Optional[HTTPEmbeddingProvider]:
        provider = (
            self.config.provider.lower()
        )

        if provider == "local":
            return None

        api_key = self.config.api_key
        api_url = self.config.api_url

        if not api_url:
            # OpenAI-compatible default.
            if (
                provider == "openai"
                or (
                    provider == "auto"
                    and api_key
                )
            ):
                api_url = (
                    "https://api.openai.com/v1/embeddings"
                )

        if not api_key or not api_url:
            return None

        try:
            return HTTPEmbeddingProvider(
                api_key=api_key,
                model=self.config.model,
                api_url=api_url,
                timeout=self.config.timeout,
            )

        except Exception as exc:
            logger.warning(
                "Could not initialize remote "
                "embedding provider: %s",
                exc,
            )

            return None

    def _select_provider(self) -> str:
        configured = (
            self.config.provider.lower()
        )

        if configured == "local":
            return "local"

        if (
            configured in {
                "openai",
                "api",
                "remote",
            }
            and self.remote_provider
        ):
            return "remote"

        if (
            configured == "auto"
            and self.remote_provider
        ):
            return "remote"

        return "local"

    # ------------------------------------------------------------------
    # Cache keys
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        text: str,
    ) -> str:
        payload = (
            f"{self.provider_name}|"
            f"{self.config.model}|"
            f"{text}"
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        text: str,
        use_cache: Optional[bool] = None,
    ) -> List[float]:
        """
        Generate an embedding for one text input.
        """

        text = str(text)

        if use_cache is None:
            use_cache = (
                self.config.cache_enabled
            )

        cache_key = self._cache_key(
            text
        )

        if use_cache:
            cached = self.cache.get(
                cache_key
            )

            if cached is not None:
                return cached

        with self._lock:
            try:
                if (
                    self.provider_name
                    == "remote"
                    and self.remote_provider
                ):
                    vector = (
                        self.remote_provider.embed(
                            text
                        )
                    )

                else:
                    vector = (
                        self.local_provider.embed(
                            text
                        )
                    )

            except EmbeddingError as exc:
                logger.warning(
                    "Primary embedding provider "
                    "failed: %s. Falling back to "
                    "local embedding.",
                    exc,
                )

                vector = (
                    self.local_provider.embed(
                        text
                    )
                )

            vector = self._validate_vector(
                vector
            )

            if use_cache:
                self.cache.set(
                    cache_key,
                    vector,
                )

            return vector

    def embed_batch(
        self,
        texts: Sequence[str],
        use_cache: Optional[bool] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Cached items are reused and only missing items are sent to the
        provider.
        """

        if not texts:
            return []

        if use_cache is None:
            use_cache = (
                self.config.cache_enabled
            )

        normalized = [
            str(text)
            for text in texts
        ]

        results: List[
            Optional[List[float]]
        ] = [None] * len(
            normalized
        )

        missing_indices: List[int] = []
        missing_texts: List[str] = []

        cache_keys: Dict[
            int,
            str
        ] = {}

        if use_cache:
            for index, text in enumerate(
                normalized
            ):
                key = self._cache_key(
                    text
                )

                cache_keys[index] = key

                cached = self.cache.get(
                    key
                )

                if cached is not None:
                    results[index] = cached

                else:
                    missing_indices.append(
                        index
                    )
                    missing_texts.append(
                        text
                    )

        else:
            missing_indices = list(
                range(
                    len(normalized)
                )
            )

            missing_texts = normalized

        if missing_texts:
            generated: List[
                List[float]
            ] = []

            with self._lock:
                try:
                    if (
                        self.provider_name
                        == "remote"
                        and self.remote_provider
                    ):
                        batch_size = max(
                            1,
                            self.config.batch_size,
                        )

                        for start in range(
                            0,
                            len(missing_texts),
                            batch_size,
                        ):
                            chunk = (
                                missing_texts[
                                    start:
                                    start + batch_size
                                ]
                            )

                            generated.extend(
                                self.remote_provider
                                .embed_batch(
                                    chunk
                                )
                            )

                    else:
                        generated = (
                            self.local_provider
                            .embed_batch(
                                missing_texts
                            )
                        )

                except EmbeddingError as exc:
                    logger.warning(
                        "Batch embedding failed: "
                        "%s. Using local fallback.",
                        exc,
                    )

                    generated = (
                        self.local_provider
                        .embed_batch(
                            missing_texts
                        )
                    )

            if len(generated) != len(
                missing_indices
            ):
                raise EmbeddingProviderError(
                    "Embedding provider returned "
                    "an invalid batch size."
                )

            for index, vector in zip(
                missing_indices,
                generated,
            ):
                validated = (
                    self._validate_vector(
                        vector
                    )
                )

                results[index] = validated

                if use_cache:
                    self.cache.set(
                        cache_keys.get(
                            index,
                            self._cache_key(
                                normalized[index]
                            ),
                        ),
                        validated,
                    )

        return [
            vector
            if vector is not None
            else []
            for vector in results
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_vector(
        self,
        vector: Iterable[float],
    ) -> List[float]:
        values = [
            float(value)
            for value in vector
        ]

        if not values:
            raise EmbeddingDimensionError(
                "Embedding vector is empty."
            )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise EmbeddingDimensionError(
                "Embedding contains non-finite values."
            )

        # Remote providers define their own actual dimension.
        if (
            self.provider_name == "remote"
            and self.config.dimension
            and len(values)
            != self.config.dimension
        ):
            # Do not silently reshape a neural embedding.
            raise EmbeddingDimensionError(
                "Unexpected embedding dimension: "
                f"expected {self.config.dimension}, "
                f"received {len(values)}."
            )

        if self.config.normalize:
            values = normalize_vector(
                values
            )

        return values

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def similarity(
        self,
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:
        """Calculate cosine similarity."""

        return cosine_similarity(
            a,
            b,
        )

    def distance(
        self,
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:
        """Calculate Euclidean distance."""

        return euclidean_distance(
            a,
            b,
        )

    def most_similar(
        self,
        query_embedding: Sequence[float],
        candidates: Sequence[
            Sequence[float]
        ],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate vectors by cosine similarity.
        """

        results: List[
            Dict[str, Any]
        ] = []

        for index, candidate in enumerate(
            candidates
        ):
            score = cosine_similarity(
                query_embedding,
                candidate,
            )

            results.append(
                {
                    "index": index,
                    "score": score,
                    "embedding": candidate,
                }
            )

        results.sort(
            key=lambda item:
            item["score"],
            reverse=True,
        )

        return results[
            : max(1, int(top_k))
        ]

    # ------------------------------------------------------------------
    # Text similarity
    # ------------------------------------------------------------------

    def text_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """Calculate semantic similarity between two texts."""

        vectors = self.embed_batch(
            [
                text_a,
                text_b,
            ]
        )

        return cosine_similarity(
            vectors[0],
            vectors[1],
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""

        self.cache.clear()

    def cache_size(self) -> int:
        """Return number of cached embeddings."""

        return self.cache.size()

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return embedding engine information."""

        return {
            "provider": self.provider_name,
            "configured_provider": (
                self.config.provider
            ),
            "model": self.config.model,
            "dimension": (
                self.config.dimension
                if self.provider_name
                == "remote"
                else self.config.local_dimension
            ),
            "normalization": (
                self.config.normalize
            ),
            "cache_enabled": (
                self.config.cache_enabled
            ),
            "cache_size": self.cache.size(),
            "cache_capacity": (
                self.config.cache_size
            ),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_engine: Optional[
    EmbeddingEngine
] = None

_default_lock = threading.Lock()


def get_embedding_engine() -> EmbeddingEngine:
    """
    Return the shared RENIX embedding engine.

    Using a singleton avoids repeatedly initializing providers and caches.
    """

    global _default_engine

    if _default_engine is None:
        with _default_lock:
            if _default_engine is None:
                _default_engine = (
                    EmbeddingEngine()
                )

    return _default_engine


def embed(
    text: str,
) -> List[float]:
    """Convenience wrapper for RENIX."""
    return get_embedding_engine().embed(
        text
    )


def embed_batch(
    texts: Sequence[str],
) -> List[List[float]]:
    """Convenience batch wrapper."""
    return get_embedding_engine().embed_batch(
        texts
    )


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

Embeddings = EmbeddingEngine
EmbeddingManager = EmbeddingEngine


__all__ = [
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingDimensionError",
    "LocalHashEmbedder",
    "HTTPEmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingEngine",
    "Embeddings",
    "EmbeddingManager",
    "cosine_similarity",
    "euclidean_distance",
    "normalize_vector",
    "get_embedding_engine",
    "embed",
    "embed_batch",
]


