"""
RENIX Weather Service
=====================

High-level weather service for RENIX.

This layer sits above:

    WeatherProviderRegistry
            |
            +-- OpenWeatherProvider
            +-- WeatherAPIProvider
            +-- WeatherbitProvider

    WeatherCache
            |
            +-- current weather
            +-- forecasts
            +-- location searches

The rest of RENIX should preferably use WeatherService instead
of communicating directly with individual weather providers.

Features:
    - provider selection
    - automatic fallback
    - response caching
    - cache invalidation
    - current weather
    - daily forecast
    - hourly forecast
    - location search
    - provider health
    - service statistics
"""

from __future__ import annotations

import time

from typing import Any

from . import WeatherProvider
from .provider_registry import (
    WeatherProviderRegistry,
    get_weather_provider_registry,
)
from .weather_cache import (
    WeatherCache,
    get_weather_cache,
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_CURRENT_TTL = 600.0
DEFAULT_FORECAST_TTL = 1800.0
DEFAULT_HOURLY_TTL = 900.0
DEFAULT_SEARCH_TTL = 3600.0


# ============================================================
# SERVICE
# ============================================================


class WeatherService:
    """
    Main weather service used by RENIX.

    Example:

        service = WeatherService()

        weather = service.get_current_weather(
            "Mumbai"
        )

        forecast = service.get_forecast(
            "Mumbai",
            days=5,
        )
    """

    def __init__(
        self,
        *,
        registry: WeatherProviderRegistry | None = None,
        cache: WeatherCache | None = None,
        preferred_provider: str | None = None,
        fallback_enabled: bool = True,
        cache_enabled: bool = True,
    ) -> None:

        self.registry = (
            registry
            or get_weather_provider_registry()
        )

        self.cache = (
            cache
            or get_weather_cache()
        )

        self.cache_enabled = bool(
            cache_enabled
        )

        self.fallback_enabled = bool(
            fallback_enabled
        )

        if preferred_provider:
            self.registry.set_preferred_provider(
                preferred_provider
            )

        self._requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._failures = 0

        self._operation_times: dict[
            str,
            list[float],
        ] = {}

    # ========================================================
    # CURRENT WEATHER
    # ========================================================

    def get_current_weather(
        self,
        location: str,
        *,
        provider: str | None = None,
        use_cache: bool | None = None,
        cache_ttl: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get current weather.

        The cache is checked first. If no cached response exists,
        the provider registry executes the request and performs
        fallback when enabled.
        """

        location = self._validate_location(
            location
        )

        cache_enabled = self._should_use_cache(
            use_cache
        )

        provider_name = self._provider_name(
            provider
        )

        if cache_enabled:

            cached = self.cache.get(
                provider=provider_name,
                operation="current",
                location=location,
                **kwargs,
            )

            if cached is not None:
                self._cache_hits += 1
                return cached

            self._cache_misses += 1

        started = time.perf_counter()

        try:

            result = self._execute(
                "get_current_weather",
                location,
                provider=provider,
                **kwargs,
            )

            self._record_operation_time(
                "current",
                started,
            )

            if cache_enabled:

                actual_provider = self._extract_provider(
                    result
                )

                self.cache.set(
                    provider=actual_provider,
                    operation="current",
                    location=location,
                    value=result,
                    ttl=(
                        cache_ttl
                        if cache_ttl is not None
                        else DEFAULT_CURRENT_TTL
                    ),
                    **kwargs,
                )

            return result

        except Exception:
            self._failures += 1
            raise

    # ========================================================
    # DAILY FORECAST
    # ========================================================

    def get_forecast(
        self,
        location: str,
        *,
        days: int = 5,
        provider: str | None = None,
        use_cache: bool | None = None,
        cache_ttl: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Get daily weather forecast.
        """

        location = self._validate_location(
            location
        )

        days = self._validate_positive_integer(
            days,
            "days",
        )

        cache_enabled = self._should_use_cache(
            use_cache
        )

        provider_name = self._provider_name(
            provider
        )

        cache_params = {
            "days": days,
            **kwargs,
        }

        if cache_enabled:

            cached = self.cache.get(
                provider=provider_name,
                operation="forecast",
                location=location,
                **cache_params,
            )

            if cached is not None:
                self._cache_hits += 1
                return cached

            self._cache_misses += 1

        started = time.perf_counter()

        try:

            result = self._execute(
                "get_forecast",
                location,
                provider=provider,
                days=days,
                **kwargs,
            )

            self._record_operation_time(
                "forecast",
                started,
            )

            if not isinstance(
                result,
                list,
            ):
                result = list(result or [])

            if cache_enabled:

                actual_provider = self._extract_provider_from_list(
                    result
                )

                self.cache.set(
                    provider=actual_provider,
                    operation="forecast",
                    location=location,
                    value=result,
                    ttl=(
                        cache_ttl
                        if cache_ttl is not None
                        else DEFAULT_FORECAST_TTL
                    ),
                    **cache_params,
                )

            return result

        except Exception:
            self._failures += 1
            raise

    # ========================================================
    # HOURLY FORECAST
    # ========================================================

    def get_hourly_forecast(
        self,
        location: str,
        *,
        hours: int = 24,
        provider: str | None = None,
        use_cache: bool | None = None,
        cache_ttl: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Get hourly forecast.

        Providers that do not implement hourly forecasts will
        automatically fail over when fallback is enabled.
        """

        location = self._validate_location(
            location
        )

        hours = self._validate_positive_integer(
            hours,
            "hours",
        )

        cache_enabled = self._should_use_cache(
            use_cache
        )

        provider_name = self._provider_name(
            provider
        )

        cache_params = {
            "hours": hours,
            **kwargs,
        }

        if cache_enabled:

            cached = self.cache.get(
                provider=provider_name,
                operation="hourly_forecast",
                location=location,
                **cache_params,
            )

            if cached is not None:
                self._cache_hits += 1
                return cached

            self._cache_misses += 1

        started = time.perf_counter()

        try:

            result = self._execute(
                "get_hourly_forecast",
                location,
                provider=provider,
                hours=hours,
                **kwargs,
            )

            self._record_operation_time(
                "hourly_forecast",
                started,
            )

            if not isinstance(
                result,
                list,
            ):
                result = list(result or [])

            if cache_enabled:

                actual_provider = self._extract_provider_from_list(
                    result
                )

                self.cache.set(
                    provider=actual_provider,
                    operation="hourly_forecast",
                    location=location,
                    value=result,
                    ttl=(
                        cache_ttl
                        if cache_ttl is not None
                        else DEFAULT_HOURLY_TTL
                    ),
                    **cache_params,
                )

            return result

        except Exception:
            self._failures += 1
            raise

    # ========================================================
    # LOCATION SEARCH
    # ========================================================

    def search_location(
        self,
        query: str,
        *,
        provider: str | None = None,
        use_cache: bool | None = None,
        cache_ttl: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for a weather location.
        """

        query = self._validate_location(
            query
        )

        cache_enabled = self._should_use_cache(
            use_cache
        )

        provider_name = self._provider_name(
            provider
        )

        if cache_enabled:

            cached = self.cache.get(
                provider=provider_name,
                operation="search_location",
                location=query,
                **kwargs,
            )

            if cached is not None:
                self._cache_hits += 1
                return cached

            self._cache_misses += 1

        started = time.perf_counter()

        try:

            result = self._execute(
                "search_location",
                query,
                provider=provider,
                **kwargs,
            )

            self._record_operation_time(
                "search_location",
                started,
            )

            if not isinstance(
                result,
                list,
            ):
                result = list(result or [])

            if cache_enabled:

                actual_provider = self._extract_provider_from_list(
                    result
                )

                self.cache.set(
                    provider=actual_provider,
                    operation="search_location",
                    location=query,
                    value=result,
                    ttl=(
                        cache_ttl
                        if cache_ttl is not None
                        else DEFAULT_SEARCH_TTL
                    ),
                    **kwargs,
                )

            return result

        except Exception:
            self._failures += 1
            raise

    # ========================================================
    # GENERIC EXECUTION
    # ========================================================

    def execute(
        self,
        operation: str,
        *args: Any,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an arbitrary provider operation.

        Useful for future provider capabilities.
        """

        if not operation or not operation.strip():
            raise ValueError(
                "Operation cannot be empty."
            )

        return self._execute(
            operation,
            *args,
            provider=provider,
            **kwargs,
        )

    def _execute(
        self,
        operation: str,
        *args: Any,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an operation using the requested provider or
        configured fallback chain.
        """

        self._requests += 1

        if provider:

            provider_instance = (
                self.registry.get_provider(
                    provider
                )
            )

            method = getattr(
                provider_instance,
                operation,
                None,
            )

            if not callable(method):
                raise AttributeError(
                    f"Provider '{provider}' does not "
                    f"implement '{operation}'."
                )

            return method(
                *args,
                **kwargs,
            )

        if self.fallback_enabled:

            return self.registry.execute_with_fallback(
                operation,
                *args,
                **kwargs,
            )

        selected_provider = (
            self.registry.select_provider_name()
        )

        provider_instance = (
            self.registry.get_provider(
                selected_provider
            )
        )

        method = getattr(
            provider_instance,
            operation,
            None,
        )

        if not callable(method):
            raise AttributeError(
                f"Provider '{selected_provider}' "
                f"does not implement '{operation}'."
            )

        return method(
            *args,
            **kwargs,
        )

    # ========================================================
    # PROVIDER MANAGEMENT
    # ========================================================

    def set_provider(
        self,
        provider: str,
    ) -> None:
        """Set the preferred weather provider."""

        self.registry.set_preferred_provider(
            provider
        )

    def get_provider(
        self,
        provider: str | None = None,
    ) -> WeatherProvider:
        """Return a provider instance."""

        return self.registry.get_provider(
            provider
        )

    def list_providers(
        self,
    ) -> list[str]:
        """Return registered weather providers."""

        return self.registry.list_providers()

    def get_provider_health(
        self,
        provider: str | None = None,
    ) -> bool | dict[str, bool]:
        """Return provider health status."""

        return self.registry.health_check(
            provider
        )

    def get_available_providers(
        self,
    ) -> list[str]:
        """Return providers that pass their health checks."""

        return self.registry.get_available_providers()

    # ========================================================
    # CACHE MANAGEMENT
    # ========================================================

    def clear_cache(
        self,
    ) -> int:
        """Clear all weather cache entries."""

        return self.cache.clear()

    def cleanup_cache(
        self,
    ) -> int:
        """Remove expired weather cache entries."""

        return self.cache.cleanup_expired()

    def invalidate_location(
        self,
        location: str,
    ) -> int:
        """
        Invalidate all known cache variants for a location.

        Since cache keys are intentionally opaque, this method
        rebuilds common RENIX weather-operation keys.
        """

        location = self._validate_location(
            location
        )

        removed = 0

        for provider_name in self.registry.list_providers():

            operations = (
                "current",
                "forecast",
                "hourly_forecast",
                "search_location",
            )

            for operation in operations:

                variants: list[
                    dict[str, Any]
                ] = [
                    {},
                    {"days": 1},
                    {"days": 3},
                    {"days": 5},
                    {"days": 7},
                    {"days": 14},
                    {"hours": 12},
                    {"hours": 24},
                    {"hours": 48},
                ]

                for params in variants:

                    key = self.cache.make_key(
                        provider=provider_name,
                        operation=operation,
                        location=location,
                        **params,
                    )

                    with self.cache._lock:
                        if key in self.cache._entries:
                            self.cache._entries.pop(
                                key,
                                None,
                            )
                            removed += 1

        return removed

    # ========================================================
    # STATISTICS
    # ========================================================

    def stats(
        self,
    ) -> dict[str, Any]:
        """Return service and cache statistics."""

        operation_stats: dict[
            str,
            dict[str, float],
        ] = {}

        for operation, values in (
            self._operation_times.items()
        ):

            if not values:
                continue

            average = sum(values) / len(
                values
            )

            operation_stats[
                operation
            ] = {
                "calls": float(
                    len(values)
                ),
                "average_ms": average,
                "last_ms": values[-1],
                "minimum_ms": min(values),
                "maximum_ms": max(values),
            }

        total_requests = (
            self._cache_hits
            + self._cache_misses
        )

        cache_hit_rate = (
            self._cache_hits
            / total_requests
            if total_requests
            else 0.0
        )

        return {
            "requests": self._requests,
            "failures": self._failures,
            "cache": self.cache.stats(),
            "service_cache_hits": self._cache_hits,
            "service_cache_misses": self._cache_misses,
            "service_cache_hit_rate": cache_hit_rate,
            "operations": operation_stats,
            "providers": self.registry.to_dict(),
        }

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def configure(
        self,
        *,
        preferred_provider: str | None = None,
        fallback_enabled: bool | None = None,
        cache_enabled: bool | None = None,
    ) -> None:
        """Update service configuration."""

        if preferred_provider is not None:
            self.set_provider(
                preferred_provider
            )

        if fallback_enabled is not None:
            self.fallback_enabled = bool(
                fallback_enabled
            )

        if cache_enabled is not None:
            self.cache_enabled = bool(
                cache_enabled
            )

    # ========================================================
    # HELPERS
    # ========================================================

    def _should_use_cache(
        self,
        use_cache: bool | None,
    ) -> bool:
        """Resolve cache usage."""

        if use_cache is None:
            return self.cache_enabled

        return bool(use_cache)

    def _provider_name(
        self,
        provider: str | None,
    ) -> str:
        """
        Resolve the provider used for cache lookup.

        When fallback is enabled and no provider was explicitly
        requested, the preferred provider is used for lookup.
        """

        if provider:
            return provider.strip().lower()

        return self.registry.select_provider_name()

    @staticmethod
    def _validate_location(
        location: str,
    ) -> str:
        """Validate a location string."""

        if not isinstance(
            location,
            str,
        ):
            raise TypeError(
                "Location must be a string."
            )

        normalized = location.strip()

        if not normalized:
            raise ValueError(
                "Location cannot be empty."
            )

        return normalized

    @staticmethod
    def _validate_positive_integer(
        value: int,
        name: str,
    ) -> int:
        """Validate a positive integer."""

        try:
            normalized = int(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} must be an integer."
            ) from exc

        if normalized < 1:
            raise ValueError(
                f"{name} must be at least 1."
            )

        return normalized

    @staticmethod
    def _extract_provider(
        result: Any,
    ) -> str:
        """Extract provider name from a normalized response."""

        if isinstance(
            result,
            dict,
        ):
            provider = result.get(
                "provider"
            )

            if provider:
                return str(
                    provider
                ).strip().lower()

        return "unknown"

    @classmethod
    def _extract_provider_from_list(
        cls,
        result: list[Any],
    ) -> str:
        """Extract provider from the first normalized result."""

        for item in result:

            if isinstance(
                item,
                dict,
            ):
                provider = item.get(
                    "provider"
                )

                if provider:
                    return str(
                        provider
                    ).strip().lower()

        return "unknown"

    def _record_operation_time(
        self,
        operation: str,
        started: float,
    ) -> None:
        """Record operation execution duration."""

        duration_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        values = self._operation_times.setdefault(
            operation,
            [],
        )

        values.append(
            duration_ms
        )

        # Keep only recent measurements so statistics do not grow
        # indefinitely during a long-running RENIX session.
        if len(values) > 100:
            del values[:-100]


# ============================================================
# DEFAULT SERVICE
# ============================================================


_default_weather_service: WeatherService | None = None


def get_weather_service() -> WeatherService:
    """
    Return the process-wide RENIX WeatherService.

    The service is created lazily.
    """

    global _default_weather_service

    if _default_weather_service is None:
        _default_weather_service = (
            WeatherService()
        )

    return _default_weather_service


def reset_weather_service() -> None:
    """Reset the global weather service."""

    global _default_weather_service

    _default_weather_service = None


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def get_current_weather(
    location: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get current weather through the global service."""

    return get_weather_service().get_current_weather(
        location,
        **kwargs,
    )


def get_forecast(
    location: str,
    *,
    days: int = 5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Get daily forecast through the global service."""

    return get_weather_service().get_forecast(
        location,
        days=days,
        **kwargs,
    )


def get_hourly_forecast(
    location: str,
    *,
    hours: int = 24,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Get hourly forecast through the global service."""

    return get_weather_service().get_hourly_forecast(
        location,
        hours=hours,
        **kwargs,
    )


def search_weather_location(
    query: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Search weather locations through the global service."""

    return get_weather_service().search_location(
        query,
        **kwargs,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "WeatherService",
    "get_weather_service",
    "reset_weather_service",
    "get_current_weather",
    "get_forecast",
    "get_hourly_forecast",
    "search_weather_location",
]


