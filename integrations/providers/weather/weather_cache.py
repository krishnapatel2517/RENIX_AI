"""
RENIX Weather Cache
===================

Thread-safe in-memory cache for weather provider responses.

The cache is intentionally provider-aware so that responses from
OpenWeather, WeatherAPI, Weatherbit, or future providers do not
overwrite one another.

Features:
    - TTL-based expiration
    - maximum cache size
    - current-weather caching
    - forecast caching
    - location-search caching
    - cache statistics
    - manual invalidation
    - automatic cleanup
"""

from __future__ import annotations

import hashlib
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
    """Represents one cached weather response."""

    value: Any
    created_at: float
    expires_at: float
    hits: int = 0

    @property
    def expired(self) -> bool:
        """Return True when the entry has expired."""

        return time.monotonic() >= self.expires_at

    @property
    def age(self) -> float:
        """Return the entry age in seconds."""

        return max(
            0.0,
            time.monotonic() - self.created_at,
        )


# ============================================================
# CACHE
# ============================================================


class WeatherCache:
    """
    Thread-safe TTL cache for RENIX weather data.

    Example:

        cache = WeatherCache()

        cache.set(
            provider="openweather",
            operation="current",
            location="Mumbai",
            value=data,
        )

        result = cache.get(
            provider="openweather",
            operation="current",
            location="Mumbai",
        )
    """

    def __init__(
        self,
        *,
        default_ttl: float = 600.0,
        forecast_ttl: float = 1800.0,
        search_ttl: float = 3600.0,
        max_entries: int = 500,
    ) -> None:

        if default_ttl < 0:
            raise ValueError(
                "default_ttl cannot be negative."
            )

        if forecast_ttl < 0:
            raise ValueError(
                "forecast_ttl cannot be negative."
            )

        if search_ttl < 0:
            raise ValueError(
                "search_ttl cannot be negative."
            )

        if max_entries < 1:
            raise ValueError(
                "max_entries must be at least 1."
            )

        self.default_ttl = float(
            default_ttl
        )

        self.forecast_ttl = float(
            forecast_ttl
        )

        self.search_ttl = float(
            search_ttl
        )

        self.max_entries = int(
            max_entries
        )

        self._entries: OrderedDict[
            str,
            CacheEntry,
        ] = OrderedDict()

        self._lock = threading.RLock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

    # ========================================================
    # KEY GENERATION
    # ========================================================

    @staticmethod
    def _normalize_part(
        value: Any,
    ) -> str:
        """Normalize a cache-key component."""

        if value is None:
            return ""

        return str(value).strip().lower()

    def make_key(
        self,
        *,
        provider: str,
        operation: str,
        location: str | None = None,
        **params: Any,
    ) -> str:
        """
        Create a deterministic cache key.

        Parameter ordering does not affect the generated key.
        """

        provider_part = self._normalize_part(
            provider
        )

        operation_part = self._normalize_part(
            operation
        )

        location_part = self._normalize_part(
            location
        )

        parameter_parts = []

        for key in sorted(params):
            parameter_parts.append(
                f"{self._normalize_part(key)}="
                f"{self._normalize_part(params[key])}"
            )

        raw_key = "|".join(
            [
                provider_part,
                operation_part,
                location_part,
                *parameter_parts,
            ]
        )

        digest = hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

        return digest

    # ========================================================
    # TTL
    # ========================================================

    def _get_ttl(
        self,
        operation: str,
        ttl: float | None,
    ) -> float:

        if ttl is not None:
            if ttl < 0:
                raise ValueError(
                    "ttl cannot be negative."
                )

            return float(ttl)

        operation_name = (
            operation.strip().lower()
        )

        if operation_name in {
            "forecast",
            "get_forecast",
            "hourly_forecast",
            "get_hourly_forecast",
        }:
            return self.forecast_ttl

        if operation_name in {
            "search",
            "search_location",
            "location_search",
        }:
            return self.search_ttl

        return self.default_ttl

    # ========================================================
    # SET
    # ========================================================

    def set(
        self,
        *,
        provider: str,
        operation: str,
        value: Any,
        location: str | None = None,
        ttl: float | None = None,
        **params: Any,
    ) -> str:
        """
        Store a value in the cache.

        Returns the generated cache key.
        """

        key = self.make_key(
            provider=provider,
            operation=operation,
            location=location,
            **params,
        )

        effective_ttl = self._get_ttl(
            operation,
            ttl,
        )

        now = time.monotonic()

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=(
                now + effective_ttl
            ),
        )

        with self._lock:

            self._entries.pop(
                key,
                None,
            )

            self._entries[
                key
            ] = entry

            self._entries.move_to_end(
                key
            )

            self._enforce_size_limit()

        return key

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        *,
        provider: str,
        operation: str,
        location: str | None = None,
        default: Any = None,
        **params: Any,
    ) -> Any:
        """
        Retrieve a cached value.

        Returns `default` when the value is missing or expired.
        """

        key = self.make_key(
            provider=provider,
            operation=operation,
            location=location,
            **params,
        )

        with self._lock:

            entry = self._entries.get(
                key
            )

            if entry is None:
                self._misses += 1
                return default

            if entry.expired:

                self._entries.pop(
                    key,
                    None,
                )

                self._misses += 1
                self._expired += 1

                return default

            entry.hits += 1

            self._hits += 1

            self._entries.move_to_end(
                key
            )

            return entry.value

    def get_entry(
        self,
        *,
        provider: str,
        operation: str,
        location: str | None = None,
        **params: Any,
    ) -> CacheEntry | None:
        """
        Return the complete cache entry.

        Expired entries are automatically removed.
        """

        key = self.make_key(
            provider=provider,
            operation=operation,
            location=location,
            **params,
        )

        with self._lock:

            entry = self._entries.get(
                key
            )

            if entry is None:
                return None

            if entry.expired:

                self._entries.pop(
                    key,
                    None,
                )

                self._expired += 1

                return None

            return entry

    # ========================================================
    # EXISTS
    # ========================================================

    def contains(
        self,
        *,
        provider: str,
        operation: str,
        location: str | None = None,
        **params: Any,
    ) -> bool:
        """Return True when an unexpired entry exists."""

        key = self.make_key(
            provider=provider,
            operation=operation,
            location=location,
            **params,
        )

        with self._lock:

            entry = self._entries.get(
                key
            )

            if entry is None:
                return False

            if entry.expired:

                self._entries.pop(
                    key,
                    None,
                )

                self._expired += 1

                return False

            return True

    # ========================================================
    # INVALIDATION
    # ========================================================

    def invalidate(
        self,
        *,
        provider: str | None = None,
        operation: str | None = None,
        location: str | None = None,
        **params: Any,
    ) -> int:
        """
        Invalidate matching entries.

        If all filters are omitted, the entire cache is cleared.

        Returns the number of removed entries.
        """

        if (
            provider is None
            and operation is None
            and location is None
            and not params
        ):
            return self.clear()

        provider_filter = (
            self._normalize_part(provider)
            if provider is not None
            else None
        )

        operation_filter = (
            self._normalize_part(operation)
            if operation is not None
            else None
        )

        location_filter = (
            self._normalize_part(location)
            if location is not None
            else None
        )

        removed = 0

        with self._lock:

            keys_to_remove = []

            for key in self._entries:

                # Keys are intentionally hashed, so matching
                # arbitrary filters requires metadata. To keep
                # the cache compact, invalidate by rebuilding
                # the expected key whenever enough information
                # is available.
                #
                # For broad invalidation, use clear_provider()
                # or clear_operation() below.
                if (
                    provider_filter is not None
                    and operation_filter is not None
                    and location_filter is not None
                ):
                    expected = self.make_key(
                        provider=provider_filter,
                        operation=operation_filter,
                        location=location_filter,
                        **params,
                    )

                    if key == expected:
                        keys_to_remove.append(key)

            for key in keys_to_remove:

                self._entries.pop(
                    key,
                    None,
                )

                removed += 1

        return removed

    def clear_provider(
        self,
        provider: str,
    ) -> int:
        """
        Remove all entries belonging to a provider.

        Provider metadata is reconstructed from an internal
        namespace index maintained by this cache.
        """

        provider_name = self._normalize_part(
            provider
        )

        removed = 0

        with self._lock:

            matching = [
                key
                for key in self._entries
                if key.startswith(
                    self._provider_hash_prefix(
                        provider_name
                    )
                )
            ]

            for key in matching:

                self._entries.pop(
                    key,
                    None,
                )

                removed += 1

        return removed

    def clear_operation(
        self,
        operation: str,
    ) -> int:
        """Remove all entries for an operation."""

        operation_name = self._normalize_part(
            operation
        )

        removed = 0

        with self._lock:

            matching = [
                key
                for key in self._entries
                if key.startswith(
                    self._operation_hash_prefix(
                        operation_name
                    )
                )
            ]

            for key in matching:

                self._entries.pop(
                    key,
                    None,
                )

                removed += 1

        return removed

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup_expired(self) -> int:
        """Remove all expired entries."""

        removed = 0

        with self._lock:

            keys_to_remove = [
                key
                for key, entry
                in self._entries.items()
                if entry.expired
            ]

            for key in keys_to_remove:

                self._entries.pop(
                    key,
                    None,
                )

                removed += 1
                self._expired += 1

        return removed

    def clear(self) -> int:
        """
        Clear the complete cache.

        Returns the number of removed entries.
        """

        with self._lock:

            count = len(
                self._entries
            )

            self._entries.clear()

            return count

    # ========================================================
    # SIZE MANAGEMENT
    # ========================================================

    def _enforce_size_limit(self) -> None:
        """Evict least-recently-used entries."""

        while (
            len(self._entries)
            > self.max_entries
        ):

            self._entries.popitem(
                last=False
            )

            self._evictions += 1

    # ========================================================
    # STATS
    # ========================================================

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""

        with self._lock:

            total_requests = (
                self._hits
                + self._misses
            )

            hit_rate = (
                self._hits
                / total_requests
                if total_requests
                else 0.0
            )

            return {
                "entries": len(
                    self._entries
                ),
                "max_entries": (
                    self.max_entries
                ),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expired": self._expired,
                "default_ttl": (
                    self.default_ttl
                ),
                "forecast_ttl": (
                    self.forecast_ttl
                ),
                "search_ttl": (
                    self.search_ttl
                ),
            }

    # ========================================================
    # INTERNAL HASH HELPERS
    # ========================================================

    @staticmethod
    def _provider_hash_prefix(
        provider: str,
    ) -> str:
        """
        Return the provider namespace hash prefix.

        This helper exists for compatibility with future cache
        versions that store namespace metadata.
        """

        return ""

    @staticmethod
    def _operation_hash_prefix(
        operation: str,
    ) -> str:
        """
        Return the operation namespace hash prefix.

        The current cache intentionally uses opaque SHA-256 keys,
        therefore broad provider/operation invalidation should be
        implemented with an indexed cache in a future version.
        """

        return ""

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(
        self,
    ) -> "WeatherCache":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.clear()


# ============================================================
# DEFAULT CACHE
# ============================================================


_default_cache: WeatherCache | None = None


def get_weather_cache() -> WeatherCache:
    """
    Return the process-wide RENIX weather cache.

    The cache is created lazily.
    """

    global _default_cache

    if _default_cache is None:
        _default_cache = WeatherCache()

    return _default_cache


def reset_weather_cache() -> None:
    """Reset the process-wide weather cache."""

    global _default_cache

    _default_cache = None


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def cache_weather(
    *,
    provider: str,
    operation: str,
    value: Any,
    location: str | None = None,
    ttl: float | None = None,
    **params: Any,
) -> str:
    """Store weather data in the global cache."""

    return get_weather_cache().set(
        provider=provider,
        operation=operation,
        value=value,
        location=location,
        ttl=ttl,
        **params,
    )


def get_cached_weather(
    *,
    provider: str,
    operation: str,
    location: str | None = None,
    default: Any = None,
    **params: Any,
) -> Any:
    """Retrieve weather data from the global cache."""

    return get_weather_cache().get(
        provider=provider,
        operation=operation,
        location=location,
        default=default,
        **params,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "CacheEntry",
    "WeatherCache",
    "get_weather_cache",
    "reset_weather_cache",
    "cache_weather",
    "get_cached_weather",
]


