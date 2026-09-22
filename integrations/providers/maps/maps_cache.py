"""
RENIX Maps Cache
================

Thread-safe cache for maps-related data.

Used by:
    - MapsService
    - Geocoding
    - Reverse geocoding
    - Places
    - Routing
    - Distance calculations

Features:
    - TTL expiration
    - Maximum cache size
    - LRU-style eviction
    - Namespace support
    - JSON-safe cache keys
    - Cache statistics
    - Manual invalidation
    - Thread safety

This module stores temporary data only.
Sensitive authentication credentials should never be
placed in this cache.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


# ============================================================
# CACHE ENTRY
# ============================================================


@dataclass
class CacheEntry:
    """Represents one cached maps response."""

    value: Any

    created_at: float

    expires_at: float

    hits: int = 0

    namespace: str = "default"

    @property
    def expired(self) -> bool:
        """Return True when the entry has expired."""

        return time.monotonic() >= self.expires_at

    @property
    def remaining_ttl(self) -> float:
        """Return remaining lifetime in seconds."""

        return max(
            0.0,
            self.expires_at
            - time.monotonic(),
        )


# ============================================================
# CACHE STATISTICS
# ============================================================


@dataclass
class CacheStats:
    """Statistics for the maps cache."""

    hits: int = 0

    misses: int = 0

    sets: int = 0

    deletes: int = 0

    expirations: int = 0

    evictions: int = 0

    clears: int = 0

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0

        return (
            self.hits
            / self.requests
        )


# ============================================================
# MAP CACHE
# ============================================================


class MapCache:
    """
    Thread-safe TTL/LRU cache for RENIX Maps.

    Example:

        cache = MapCache(
            default_ttl=300,
            max_size=500,
        )

        cache.set(
            "geocode",
            {"address": "Mumbai"},
        )

        data = cache.get(
            "geocode",
        )
    """

    def __init__(
        self,
        *,
        default_ttl: int = 300,
        max_size: int = 500,
        enabled: bool = True,
        cleanup_interval: int = 60,
    ) -> None:

        self.default_ttl = max(
            0,
            int(default_ttl),
        )

        self.max_size = max(
            1,
            int(max_size),
        )

        self.enabled = bool(
            enabled
        )

        self.cleanup_interval = max(
            1,
            int(cleanup_interval),
        )

        self._cache: OrderedDict[
            str,
            CacheEntry,
        ] = OrderedDict()

        self._stats = CacheStats()

        self._lock = threading.RLock()

        self._last_cleanup = (
            time.monotonic()
        )

    # ========================================================
    # SET
    # ========================================================

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: int | float | None = None,
        namespace: str = "default",
    ) -> bool:
        """
        Store a value in the cache.

        Returns:
            True if stored successfully.
        """

        if not self.enabled:
            return False

        cache_key = self._make_key(
            namespace,
            key,
        )

        ttl_seconds = (
            self.default_ttl
            if ttl is None
            else max(
                0.0,
                float(ttl),
            )
        )

        now = time.monotonic()

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=(
                now + ttl_seconds
            ),
            namespace=namespace,
        )

        with self._lock:

            self._cleanup_if_needed()

            if cache_key in self._cache:

                self._cache.pop(
                    cache_key,
                    None,
                )

            self._cache[
                cache_key
            ] = entry

            self._cache.move_to_end(
                cache_key
            )

            self._stats.sets += 1

            self._enforce_size_limit()

        return True

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        key: str,
        default: Any = None,
        *,
        namespace: str = "default",
    ) -> Any:
        """
        Retrieve a cached value.

        Expired values are automatically removed.
        """

        if not self.enabled:
            return default

        cache_key = self._make_key(
            namespace,
            key,
        )

        with self._lock:

            entry = self._cache.get(
                cache_key
            )

            if entry is None:

                self._stats.misses += 1

                return default

            if entry.expired:

                self._cache.pop(
                    cache_key,
                    None,
                )

                self._stats.misses += 1
                self._stats.expirations += 1

                return default

            entry.hits += 1

            self._stats.hits += 1

            self._cache.move_to_end(
                cache_key
            )

            return entry.value

    # ========================================================
    # GET ENTRY
    # ========================================================

    def get_entry(
        self,
        key: str,
        *,
        namespace: str = "default",
    ) -> CacheEntry | None:
        """Return the complete cache entry."""

        if not self.enabled:
            return None

        cache_key = self._make_key(
            namespace,
            key,
        )

        with self._lock:

            entry = self._cache.get(
                cache_key
            )

            if entry is None:
                return None

            if entry.expired:

                self._cache.pop(
                    cache_key,
                    None,
                )

                self._stats.expirations += 1

                return None

            self._cache.move_to_end(
                cache_key
            )

            return entry

    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        key: str,
        *,
        namespace: str = "default",
    ) -> bool:
        """Return True if a non-expired value exists."""

        sentinel = object()

        return (
            self.get(
                key,
                sentinel,
                namespace=namespace,
            )
            is not sentinel
        )

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        key: str,
        *,
        namespace: str = "default",
    ) -> bool:
        """Delete one cached value."""

        cache_key = self._make_key(
            namespace,
            key,
        )

        with self._lock:

            if cache_key not in self._cache:
                return False

            self._cache.pop(
                cache_key,
                None,
            )

            self._stats.deletes += 1

            return True

    # ========================================================
    # INVALIDATE NAMESPACE
    # ========================================================

    def invalidate_namespace(
        self,
        namespace: str,
    ) -> int:
        """
        Delete all entries belonging to a namespace.

        Returns:
            Number of removed entries.
        """

        namespace = str(
            namespace
        )

        removed = 0

        with self._lock:

            keys_to_remove = [
                key
                for key, entry
                in self._cache.items()
                if entry.namespace
                == namespace
            ]

            for key in keys_to_remove:

                self._cache.pop(
                    key,
                    None,
                )

                removed += 1

            self._stats.deletes += (
                removed
            )

        return removed

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:
        """Clear the complete cache."""

        with self._lock:

            self._cache.clear()

            self._stats.clears += 1

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of expired entries removed.
        """

        removed = 0

        with self._lock:

            keys_to_remove = [
                key
                for key, entry
                in self._cache.items()
                if entry.expired
            ]

            for key in keys_to_remove:

                self._cache.pop(
                    key,
                    None,
                )

                removed += 1

            self._stats.expirations += (
                removed
            )

            self._last_cleanup = (
                time.monotonic()
            )

        return removed

    # ========================================================
    # SIZE
    # ========================================================

    def size(self) -> int:
        """Return number of cached entries."""

        with self._lock:

            self._cleanup_if_needed()

            return len(
                self._cache
            )

    # ========================================================
    # KEYS
    # ========================================================

    def keys(
        self,
        namespace: str | None = None,
    ) -> list[str]:
        """
        Return cache keys.

        The internal hashed key is intentionally not
        exposed. Returned keys are the logical keys.
        """

        with self._lock:

            self._cleanup_if_needed()

            result = []

            for entry_key, entry in (
                self._cache.items()
            ):

                if (
                    namespace is not None
                    and entry.namespace
                    != namespace
                ):
                    continue

                result.append(
                    entry_key
                )

            return result

    # ========================================================
    # STATS
    # ========================================================

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""

        with self._lock:

            return {
                "enabled": self.enabled,
                "size": len(
                    self._cache
                ),
                "max_size": self.max_size,
                "default_ttl": (
                    self.default_ttl
                ),
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "requests": (
                    self._stats.requests
                ),
                "hit_rate": (
                    self._stats.hit_rate
                ),
                "sets": self._stats.sets,
                "deletes": (
                    self._stats.deletes
                ),
                "expirations": (
                    self._stats.expirations
                ),
                "evictions": (
                    self._stats.evictions
                ),
                "clears": self._stats.clears,
            }

    # ========================================================
    # CACHE KEY
    # ========================================================

    @staticmethod
    def make_request_key(
        operation: str,
        **parameters: Any,
    ) -> str:
        """
        Generate a deterministic key for a Maps request.

        Example:

            MapCache.make_request_key(
                "geocode",
                address="Mumbai",
            )
        """

        payload = {
            "operation": operation,
            "parameters": parameters,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            default=str,
            separators=(
                ",",
                ":",
            ),
        )

        return hashlib.sha256(
            serialized.encode(
                "utf-8"
            )
        ).hexdigest()

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(self) -> None:
        """Enable caching."""

        with self._lock:
            self.enabled = True

    def disable(
        self,
        *,
        clear: bool = False,
    ) -> None:
        """
        Disable caching.

        Args:
            clear:
                Also clear existing cached entries.
        """

        with self._lock:

            self.enabled = False

            if clear:
                self._cache.clear()

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _make_key(
        namespace: str,
        key: str,
    ) -> str:
        """Create an internal namespace-aware key."""

        return (
            f"{namespace}::"
            f"{str(key)}"
        )

    def _cleanup_if_needed(self) -> None:
        """Perform periodic cleanup."""

        now = time.monotonic()

        if (
            now - self._last_cleanup
            >= self.cleanup_interval
        ):

            self.cleanup_expired()

    def _enforce_size_limit(self) -> None:
        """Evict oldest entries until size is valid."""

        while (
            len(self._cache)
            > self.max_size
        ):

            self._cache.popitem(
                last=False
            )

            self._stats.evictions += 1


# ============================================================
# SPECIALIZED MAP CACHE HELPERS
# ============================================================


class MapsCacheKeys:
    """
    Standard namespaces used by RENIX Maps.
    """

    GEOCODING = "geocoding"

    REVERSE_GEOCODING = "reverse_geocoding"

    PLACES = "places"

    ROUTING = "routing"

    DISTANCE = "distance"

    DIRECTIONS = "directions"


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "CacheEntry",
    "CacheStats",
    "MapCache",
    "MapsCacheKeys",
]


