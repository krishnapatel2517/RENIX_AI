"""
RENIX Weather Provider Registry
===============================

Central registry for RENIX weather providers.

Supported providers:
    - OpenWeather
    - WeatherAPI
    - Weatherbit

The registry provides:
    - provider registration
    - provider lookup
    - automatic provider creation
    - preferred-provider selection
    - fallback handling
    - health checking
"""

from __future__ import annotations

import os

from typing import Any, Callable

from . import WeatherProvider
from .openweather_provider import OpenWeatherProvider
from .weatherapi_provider import WeatherAPIProvider
from .weatherbit_provider import WeatherbitProvider


# ============================================================
# TYPES
# ============================================================

ProviderFactory = Callable[..., WeatherProvider]


# ============================================================
# DEFAULT PROVIDERS
# ============================================================

DEFAULT_PROVIDER_ORDER = (
    "openweather",
    "weatherapi",
    "weatherbit",
)


# ============================================================
# REGISTRY
# ============================================================


class WeatherProviderRegistry:
    """
    Registry and manager for RENIX weather providers.

    Example:

        registry = WeatherProviderRegistry()

        provider = registry.get_provider(
            "openweather"
        )

        weather = provider.get_current_weather(
            "Mumbai"
        )
    """

    def __init__(
        self,
        *,
        preferred_provider: str | None = None,
        fallback_enabled: bool = True,
        provider_order: list[str] | tuple[str, ...] | None = None,
        auto_register_defaults: bool = True,
    ) -> None:

        self.preferred_provider = (
            preferred_provider
            or os.getenv(
                "RENIX_WEATHER_PROVIDER"
            )
        )

        self.fallback_enabled = bool(
            fallback_enabled
        )

        self.provider_order = list(
            provider_order
            or DEFAULT_PROVIDER_ORDER
        )

        self._factories: dict[
            str,
            ProviderFactory,
        ] = {}

        self._instances: dict[
            str,
            WeatherProvider,
        ] = {}

        if auto_register_defaults:
            self.register_default_providers()

    # ========================================================
    # REGISTRATION
    # ========================================================

    def register(
        self,
        name: str,
        factory: ProviderFactory,
        *,
        replace: bool = False,
    ) -> None:
        """
        Register a provider factory.

        Args:
            name:
                Unique provider name.

            factory:
                Callable that creates the provider.

            replace:
                Whether an existing factory may be replaced.
        """

        normalized_name = self._normalize_name(
            name
        )

        if not callable(factory):
            raise TypeError(
                "Provider factory must be callable."
            )

        if (
            normalized_name in self._factories
            and not replace
        ):
            raise ValueError(
                f"Weather provider "
                f"'{normalized_name}' is already registered."
            )

        self._factories[
            normalized_name
        ] = factory

        # Remove an old instance when replacing a factory.
        if replace:
            self._instances.pop(
                normalized_name,
                None,
            )

        if normalized_name not in self.provider_order:
            self.provider_order.append(
                normalized_name
            )

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Remove a provider from the registry.

        Returns True when a provider existed.
        """

        normalized_name = self._normalize_name(
            name
        )

        existed = (
            normalized_name
            in self._factories
        )

        self._factories.pop(
            normalized_name,
            None,
        )

        self._instances.pop(
            normalized_name,
            None,
        )

        self.provider_order = [
            provider
            for provider in self.provider_order
            if provider != normalized_name
        ]

        return existed

    def register_default_providers(self) -> None:
        """Register RENIX's built-in weather providers."""

        self.register(
            "openweather",
            OpenWeatherProvider,
            replace=True,
        )

        self.register(
            "weatherapi",
            WeatherAPIProvider,
            replace=True,
        )

        self.register(
            "weatherbit",
            WeatherbitProvider,
            replace=True,
        )

    # ========================================================
    # LOOKUP
    # ========================================================

    def is_registered(
        self,
        name: str,
    ) -> bool:
        """Return whether a provider is registered."""

        normalized_name = self._normalize_name(
            name
        )

        return (
            normalized_name
            in self._factories
        )

    def list_providers(
        self,
    ) -> list[str]:
        """Return registered provider names in priority order."""

        registered = set(
            self._factories
        )

        return [
            provider
            for provider in self.provider_order
            if provider in registered
        ]

    def get_factory(
        self,
        name: str,
    ) -> ProviderFactory:
        """Return the factory registered under a name."""

        normalized_name = self._normalize_name(
            name
        )

        try:
            return self._factories[
                normalized_name
            ]
        except KeyError as exc:
            raise KeyError(
                f"Weather provider "
                f"'{normalized_name}' is not registered. "
                f"Available providers: "
                f"{', '.join(self.list_providers()) or 'none'}."
            ) from exc

    # ========================================================
    # INSTANCE MANAGEMENT
    # ========================================================

    def get_provider(
        self,
        name: str | None = None,
        *,
        force_new: bool = False,
        **kwargs: Any,
    ) -> WeatherProvider:
        """
        Get or create a provider instance.

        If `name` is omitted, the preferred provider is used,
        followed by the configured provider order.
        """

        selected_name = (
            self._normalize_name(name)
            if name
            else self.select_provider_name()
        )

        if (
            not force_new
            and not kwargs
            and selected_name in self._instances
        ):
            return self._instances[
                selected_name
            ]

        factory = self.get_factory(
            selected_name
        )

        instance = factory(
            **kwargs
        )

        if not isinstance(
            instance,
            WeatherProvider,
        ):
            raise TypeError(
                f"Provider factory "
                f"'{selected_name}' did not return "
                f"a WeatherProvider instance."
            )

        if not kwargs:
            self._instances[
                selected_name
            ] = instance

        return instance

    def clear_instances(self) -> None:
        """Clear all cached provider instances."""

        self._instances.clear()

    # ========================================================
    # PROVIDER SELECTION
    # ========================================================

    def select_provider_name(
        self,
    ) -> str:
        """
        Select the preferred available provider.

        Selection order:

            1. Explicit preferred provider
            2. Environment-configured provider
            3. Provider priority list
        """

        if self.preferred_provider:
            preferred = self._normalize_name(
                self.preferred_provider
            )

            if self.is_registered(
                preferred
            ):
                return preferred

        for name in self.provider_order:
            if self.is_registered(name):
                return name

        raise RuntimeError(
            "No weather providers are registered."
        )

    def set_preferred_provider(
        self,
        name: str | None,
    ) -> None:
        """Set or clear the preferred provider."""

        if name is None:
            self.preferred_provider = None
            return

        normalized_name = self._normalize_name(
            name
        )

        if not self.is_registered(
            normalized_name
        ):
            raise KeyError(
                f"Cannot select unregistered "
                f"weather provider '{normalized_name}'."
            )

        self.preferred_provider = (
            normalized_name
        )

    def set_provider_order(
        self,
        providers: list[str] | tuple[str, ...],
    ) -> None:
        """
        Set provider priority order.

        Unknown providers are allowed in the list so that
        providers can be registered later.
        """

        normalized: list[str] = []

        for provider in providers:
            name = self._normalize_name(
                provider
            )

            if name not in normalized:
                normalized.append(name)

        self.provider_order = normalized

    # ========================================================
    # FALLBACK
    # ========================================================

    def iter_provider_names(
        self,
        *,
        preferred: str | None = None,
    ):
        """
        Yield providers in fallback order.

        The preferred provider is yielded first.
        """

        seen: set[str] = set()

        first = (
            preferred
            or self.preferred_provider
        )

        if first:
            first_name = self._normalize_name(
                first
            )

            if self.is_registered(
                first_name
            ):
                seen.add(first_name)
                yield first_name

        for name in self.provider_order:
            normalized_name = self._normalize_name(
                name
            )

            if (
                normalized_name in seen
                or not self.is_registered(
                    normalized_name
                )
            ):
                continue

            seen.add(normalized_name)

            yield normalized_name

    def execute_with_fallback(
        self,
        operation: str,
        *args: Any,
        preferred: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute the same provider operation with fallback.

        Example:

            weather = registry.execute_with_fallback(
                "get_current_weather",
                "Mumbai",
            )
        """

        if not operation:
            raise ValueError(
                "Operation cannot be empty."
            )

        errors: list[
            tuple[str, Exception]
        ] = []

        provider_names = self.iter_provider_names(
            preferred=preferred
        )

        for provider_name in provider_names:

            try:
                provider = self.get_provider(
                    provider_name
                )

                method = getattr(
                    provider,
                    operation,
                    None,
                )

                if not callable(method):
                    raise AttributeError(
                        f"Provider '{provider_name}' "
                        f"does not implement "
                        f"'{operation}'."
                    )

                return method(
                    *args,
                    **kwargs,
                )

            except Exception as exc:
                errors.append(
                    (
                        provider_name,
                        exc,
                    )
                )

                if not self.fallback_enabled:
                    raise

        if errors:
            details = "; ".join(
                f"{name}: {error}"
                for name, error in errors
            )

            raise RuntimeError(
                "All weather providers failed. "
                f"Details: {details}"
            ) from errors[-1][1]

        raise RuntimeError(
            "No weather providers are available."
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(
        self,
        name: str | None = None,
    ) -> bool | dict[str, bool]:
        """
        Check provider health.

        With `name`:
            returns one boolean.

        Without `name`:
            returns health status for every registered provider.
        """

        if name:
            provider = self.get_provider(
                name
            )

            try:
                return bool(
                    provider.health_check()
                )
            except Exception:
                return False

        results: dict[
            str,
            bool,
        ] = {}

        for provider_name in self.list_providers():

            try:
                provider = self.get_provider(
                    provider_name
                )

                results[
                    provider_name
                ] = bool(
                    provider.health_check()
                )

            except Exception:
                results[
                    provider_name
                ] = False

        return results

    def get_available_providers(
        self,
    ) -> list[str]:
        """
        Return providers whose health check succeeds.
        """

        health = self.health_check()

        if not isinstance(
            health,
            dict,
        ):
            return []

        return [
            name
            for name, healthy in health.items()
            if healthy
        ]

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def configure(
        self,
        *,
        preferred_provider: str | None = None,
        fallback_enabled: bool | None = None,
        provider_order: list[str] | None = None,
    ) -> None:
        """Update registry configuration."""

        if preferred_provider is not None:
            self.set_preferred_provider(
                preferred_provider
            )

        if fallback_enabled is not None:
            self.fallback_enabled = bool(
                fallback_enabled
            )

        if provider_order is not None:
            self.set_provider_order(
                provider_order
            )

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return registry configuration and state."""

        return {
            "preferred_provider": (
                self.preferred_provider
            ),
            "fallback_enabled": (
                self.fallback_enabled
            ),
            "provider_order": list(
                self.provider_order
            ),
            "registered_providers": (
                self.list_providers()
            ),
            "initialized_providers": list(
                self._instances.keys()
            ),
        }

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """Normalize a provider name."""

        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "Provider name must be a string."
            )

        normalized = name.strip().lower()

        if not normalized:
            raise ValueError(
                "Provider name cannot be empty."
            )

        return normalized


# ============================================================
# DEFAULT REGISTRY
# ============================================================


_default_registry: WeatherProviderRegistry | None = None


def get_weather_provider_registry() -> WeatherProviderRegistry:
    """
    Return the process-wide RENIX weather provider registry.

    The registry is created lazily.
    """

    global _default_registry

    if _default_registry is None:
        _default_registry = (
            WeatherProviderRegistry()
        )

    return _default_registry


def reset_weather_provider_registry() -> None:
    """
    Reset the process-wide registry.

    Primarily useful for testing or application reconfiguration.
    """

    global _default_registry

    _default_registry = None


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def get_weather_provider(
    name: str | None = None,
    **kwargs: Any,
) -> WeatherProvider:
    """
    Convenience wrapper around the global registry.
    """

    return get_weather_provider_registry().get_provider(
        name,
        **kwargs,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "WeatherProviderRegistry",
    "get_weather_provider_registry",
    "reset_weather_provider_registry",
    "get_weather_provider",
]


