"""
RENIX Calendar Cache
====================

Thread-safe TTL/LRU cache for calendar data.

Caches:
    - Calendars
    - Events
    - Event searches
    - Free/busy responses

The cache is intentionally provider-independent.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """Single cached calendar value."""

    value: Any
    created_at: float
    expires_at: float
    hits: int = 0

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    @property
    def remaining_ttl(self) -> float:
        return max(
            0.0,
            self.expires_at - time.monotonic(),
        )


@dataclass
class CacheStats:
    """Calendar cache statistics."""

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

        return self.hits / self.requests


class CalendarCache:
    """
    Thread-safe TTL/LRU cache for RENIX Calendar.

    Example:

        cache = CalendarCache()

        key = cache.make_request_key(
            "events",
            calendar_id="primary",
        )

        cache.set(key, events)

        events = cache.get(key)
    """

    def __init__(
        self,
        *,
        default_ttl: int = 300,
        max_size: int = 500,
        enabled: bool = True,
    ) -> None:

        self.default_ttl = max(
            0,
            int(default_ttl),
        )

        self.max_size = max(
            1,
            int(max_size),
        )

        self.enabled = bool(enabled)

        self._cache: OrderedDict[
            str,
            CacheEntry,
        ] = OrderedDict()

        self._stats = CacheStats()

        self._lock = threading.RLock()

    # ========================================================
    # SET
    # ========================================================

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: int | float | None = None,
    ) -> bool:
        """Store a value in the cache."""

        if not self.enabled:
            return False

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
            expires_at=now + ttl_seconds,
        )

        with self._lock:

            self._remove_expired()

            self._cache.pop(
                key,
                None,
            )

            self._cache[key] = entry

            self._cache.move_to_end(
                key
            )

            self._stats.sets += 1

            self._enforce_size()

        return True

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve a cached value."""

        if not self.enabled:
            return default

        with self._lock:

            entry = self._cache.get(
                key
            )

            if entry is None:

                self._stats.misses += 1

                return default

            if entry.expired:

                self._cache.pop(
                    key,
                    None,
                )

                self._stats.misses += 1
                self._stats.expirations += 1

                return default

            entry.hits += 1

            self._stats.hits += 1

            self._cache.move_to_end(
                key
            )

            return entry.value

    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        key: str,
    ) -> bool:
        """Check whether a valid cache entry exists."""

        sentinel = object()

        return (
            self.get(
                key,
                sentinel,
            )
            is not sentinel
        )

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        key: str,
    ) -> bool:
        """Delete one cache entry."""

        with self._lock:

            if key not in self._cache:
                return False

            self._cache.pop(
                key,
                None,
            )

            self._stats.deletes += 1

            return True

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:
        """Clear the entire cache."""

        with self._lock:

            self._cache.clear()

            self._stats.clears += 1

    # ========================================================
    # CLEAR PREFIX
    # ========================================================

    def clear_prefix(
        self,
        prefix: str,
    ) -> int:
        """
        Delete entries whose keys start with prefix.

        Useful when a calendar changes and all related
        event/search cache entries need invalidation.
        """

        removed = 0

        with self._lock:

            keys = [
                key
                for key in self._cache
                if key.startswith(prefix)
            ]

            for key in keys:

                self._cache.pop(
                    key,
                    None,
                )

                removed += 1

            self._stats.deletes += removed

        return removed

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup_expired(self) -> int:
        """Remove all expired entries."""

        with self._lock:
            return self._remove_expired()

    def _remove_expired(self) -> int:

        removed = 0

        expired_keys = [
            key
            for key, entry in self._cache.items()
            if entry.expired
        ]

        for key in expired_keys:

            self._cache.pop(
                key,
                None,
            )

            removed += 1

        self._stats.expirations += removed

        return removed

    # ========================================================
    # SIZE LIMIT
    # ========================================================

    def _enforce_size(self) -> None:
        """Apply LRU size limit."""

        while len(
            self._cache
        ) > self.max_size:

            self._cache.popitem(
                last=False
            )

            self._stats.evictions += 1

    # ========================================================
    # REQUEST KEY
    # ========================================================

    @staticmethod
    def make_request_key(
        operation: str,
        **parameters: Any,
    ) -> str:
        """
        Create a deterministic cache key.

        Example:

            CalendarCache.make_request_key(
                "events",
                calendar_id="primary",
                start="2026-08-23",
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

        digest = hashlib.sha256(
            serialized.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            f"calendar:{operation}:{digest}"
        )

    # ========================================================
    # EVENT INVALIDATION
    # ========================================================

    def invalidate_event(
        self,
        event_id: str,
    ) -> int:
        """
        Invalidate cache entries associated with an event.

        Event IDs can be embedded in keys by callers using
        a predictable prefix.
        """

        return self.clear_prefix(
            f"calendar:event:{event_id}"
        )

    # ========================================================
    # CALENDAR INVALIDATION
    # ========================================================

    def invalidate_calendar(
        self,
        calendar_id: str,
    ) -> int:
        """Invalidate cache entries for a calendar."""

        return self.clear_prefix(
            f"calendar:calendar:{calendar_id}"
        )

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
                "default_ttl": self.default_ttl,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "requests": self._stats.requests,
                "hit_rate": self._stats.hit_rate,
                "sets": self._stats.sets,
                "deletes": self._stats.deletes,
                "expirations": self._stats.expirations,
                "evictions": self._stats.evictions,
                "clears": self._stats.clears,
            }

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
        """Disable caching."""

        with self._lock:

            self.enabled = False

            if clear:
                self._cache.clear()


__all__ = [
    "CacheEntry",
    "CacheStats",
    "CalendarCache",
]


