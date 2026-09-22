"""
RENIX Maps Provider Registry
============================

Central registry for maps providers used by RENIX.

Responsibilities:
    - Register maps providers
    - Unregister providers
    - Select the active provider
    - Resolve providers by name
    - Check provider availability
    - Support fallback providers
    - Expose provider status
    - Avoid hard-coding provider selection throughout RENIX

Supported providers currently include:
    - Google Maps
    - Mapbox
"""

from __future__ import annotations

import os
import threading
from typing import Any, Iterable

from .maps_provider import MapsProvider


class MapsProviderRegistry:
    """
    Thread-safe registry for RENIX maps providers.

    Example:

        registry = MapsProviderRegistry()

        registry.register(google_provider)
        registry.register(mapbox_provider)

        registry.set_active("google_maps")

        provider = registry.get_active()
    """

    DEFAULT_PROVIDER_ENV = "RENIX_MAPS_PROVIDER"

    def __init__(
        self,
        providers: Iterable[MapsProvider] | None = None,
        *,
        active_provider: str | None = None,
        fallback_providers: Iterable[str] | None = None,
    ) -> None:
        self._providers: dict[str, MapsProvider] = {}
        self._active_provider: str | None = None
        self._fallback_providers: list[str] = []
        self._lock = threading.RLock()

        if providers:
            for provider in providers:
                self.register(provider)

        requested_active = (
            active_provider
            or os.getenv(
                self.DEFAULT_PROVIDER_ENV
            )
        )

        if requested_active:
            normalized = self._normalize_name(
                requested_active
            )

            if normalized in self._providers:
                self._active_provider = normalized

        if fallback_providers:
            for provider_name in fallback_providers:
                self.add_fallback(
                    provider_name
                )

    # ============================================================
    # REGISTRATION
    # ============================================================

    def register(
        self,
        provider: MapsProvider,
        *,
        make_active: bool = False,
        replace: bool = False,
    ) -> MapsProvider:
        """
        Register a maps provider.

        Args:
            provider:
                MapsProvider instance.
            make_active:
                Make this provider the active provider.
            replace:
                Replace an existing provider with the same name.
        """

        if not isinstance(
            provider,
            MapsProvider,
        ):
            raise TypeError(
                "provider must be an instance of MapsProvider."
            )

        name = self._normalize_name(
            provider.name
        )

        if not name:
            raise ValueError(
                "Provider name cannot be empty."
            )

        with self._lock:

            if (
                name in self._providers
                and not replace
            ):
                raise ValueError(
                    f"Maps provider '{name}' is already registered."
                )

            self._providers[
                name
            ] = provider

            if (
                self._active_provider is None
                or make_active
            ):
                self._active_provider = name

            return provider

    def unregister(
        self,
        provider: str,
    ) -> MapsProvider | None:
        """
        Remove a provider from the registry.
        """

        name = self._normalize_name(
            provider
        )

        with self._lock:

            removed = self._providers.pop(
                name,
                None,
            )

            if (
                name
                in self._fallback_providers
            ):
                self._fallback_providers.remove(
                    name
                )

            if (
                self._active_provider
                == name
            ):
                self._active_provider = (
                    self._fallback_providers[0]
                    if self._fallback_providers
                    else next(
                        iter(
                            self._providers
                        ),
                        None,
                    )
                )

            return removed

    def clear(self) -> None:
        """
        Remove all registered providers.
        """

        with self._lock:
            self._providers.clear()
            self._active_provider = None
            self._fallback_providers.clear()

    # ============================================================
    # LOOKUP
    # ============================================================

    def get(
        self,
        provider: str,
    ) -> MapsProvider:
        """
        Retrieve a provider by name.
        """

        name = self._normalize_name(
            provider
        )

        with self._lock:

            if name not in self._providers:
                raise KeyError(
                    f"Maps provider '{provider}' "
                    "is not registered."
                )

            return self._providers[
                name
            ]

    def find(
        self,
        provider: str,
    ) -> MapsProvider | None:
        """
        Retrieve a provider without raising an exception.
        """

        name = self._normalize_name(
            provider
        )

        with self._lock:
            return self._providers.get(
                name
            )

    def has(
        self,
        provider: str,
    ) -> bool:
        """
        Check whether a provider is registered.
        """

        return (
            self.find(provider)
            is not None
        )

    # ============================================================
    # ACTIVE PROVIDER
    # ============================================================

    def set_active(
        self,
        provider: str,
    ) -> MapsProvider:
        """
        Set the active maps provider.
        """

        name = self._normalize_name(
            provider
        )

        with self._lock:

            if name not in self._providers:
                raise KeyError(
                    f"Cannot activate unregistered "
                    f"maps provider '{provider}'."
                )

            self._active_provider = name

            return self._providers[
                name
            ]

    def get_active(
        self,
        *,
        require_enabled: bool = False,
        require_configured: bool = False,
    ) -> MapsProvider:
        """
        Return the active provider.

        Raises:
            RuntimeError:
                If no active provider exists.
        """

        with self._lock:

            if self._active_provider is None:
                raise RuntimeError(
                    "No active maps provider is configured."
                )

            provider = self._providers.get(
                self._active_provider
            )

            if provider is None:
                raise RuntimeError(
                    "The active maps provider is no longer registered."
                )

        self._validate_provider(
            provider,
            require_enabled=require_enabled,
            require_configured=require_configured,
        )

        return provider

    def get_active_name(
        self,
    ) -> str | None:
        """
        Return the active provider name.
        """

        with self._lock:
            return self._active_provider

    # ============================================================
    # FALLBACK PROVIDERS
    # ============================================================

    def add_fallback(
        self,
        provider: str,
        *,
        priority: int | None = None,
    ) -> None:
        """
        Add a provider to the fallback chain.

        Lower priority numbers are tried first.
        """

        name = self._normalize_name(
            provider
        )

        with self._lock:

            if name not in self._providers:
                raise KeyError(
                    f"Cannot add unregistered provider "
                    f"'{provider}' as fallback."
                )

            if name in self._fallback_providers:
                return

            if priority is None:
                self._fallback_providers.append(
                    name
                )
                return

            index = max(
                0,
                min(
                    priority,
                    len(
                        self._fallback_providers
                    ),
                ),
            )

            self._fallback_providers.insert(
                index,
                name,
            )

    def remove_fallback(
        self,
        provider: str,
    ) -> None:
        """
        Remove a provider from the fallback chain.
        """

        name = self._normalize_name(
            provider
        )

        with self._lock:

            if name in self._fallback_providers:
                self._fallback_providers.remove(
                    name
                )

    def get_fallback_names(
        self,
    ) -> list[str]:
        """
        Return fallback provider names in priority order.
        """

        with self._lock:
            return list(
                self._fallback_providers
            )

    def get_fallbacks(
        self,
        *,
        require_enabled: bool = False,
        require_configured: bool = False,
    ) -> list[MapsProvider]:
        """
        Return usable fallback providers.
        """

        with self._lock:
            names = list(
                self._fallback_providers
            )

        providers: list[
            MapsProvider
        ] = []

        for name in names:

            provider = self.find(
                name
            )

            if provider is None:
                continue

            try:

                self._validate_provider(
                    provider,
                    require_enabled=require_enabled,
                    require_configured=require_configured,
                )

                providers.append(
                    provider
                )

            except RuntimeError:
                continue

        return providers

    def get_provider_chain(
        self,
        *,
        include_active: bool = True,
        require_enabled: bool = False,
        require_configured: bool = False,
    ) -> list[MapsProvider]:
        """
        Return the active provider followed by fallbacks.

        Duplicate providers are removed.
        """

        chain: list[
            MapsProvider
        ] = []

        seen: set[str] = set()

        if include_active:

            try:
                active = self.get_active(
                    require_enabled=require_enabled,
                    require_configured=require_configured,
                )

                chain.append(
                    active
                )

                seen.add(
                    self._normalize_name(
                        active.name
                    )
                )

            except RuntimeError:
                pass

        for provider in self.get_fallbacks(
            require_enabled=require_enabled,
            require_configured=require_configured,
        ):

            name = self._normalize_name(
                provider.name
            )

            if name in seen:
                continue

            chain.append(
                provider
            )

            seen.add(
                name
            )

        return chain

    # ============================================================
    # AUTOMATIC PROVIDER SELECTION
    # ============================================================

    def select_best(
        self,
        *,
        capability: str | None = None,
        require_enabled: bool = True,
        require_configured: bool = True,
    ) -> MapsProvider:
        """
        Select the best currently usable provider.

        Preference order:

            1. Active provider
            2. Fallback providers
            3. Any remaining registered provider
        """

        candidates: list[
            MapsProvider
        ] = []

        active = self.find(
            self._active_provider
        ) if self._active_provider else None

        if active is not None:
            candidates.append(
                active
            )

        for provider in self.get_fallbacks():
            if provider not in candidates:
                candidates.append(
                    provider
                )

        for provider in self.get_all():
            if provider not in candidates:
                candidates.append(
                    provider
                )

        for provider in candidates:

            try:

                self._validate_provider(
                    provider,
                    require_enabled=require_enabled,
                    require_configured=require_configured,
                )

            except RuntimeError:
                continue

            if (
                capability
                and not provider.supports(
                    capability
                )
            ):
                continue

            return provider

        capability_text = (
            f" with capability '{capability}'"
            if capability
            else ""
        )

        raise RuntimeError(
            "No usable maps provider found"
            f"{capability_text}."
        )

    # ============================================================
    # PROVIDER HEALTH
    # ============================================================

    def health_status(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Return health/configuration information for every provider.
        """

        with self._lock:
            providers = list(
                self._providers.items()
            )

        result: dict[
            str,
            dict[str, Any],
        ] = {}

        for name, provider in providers:

            try:
                healthy = bool(
                    provider.health_check()
                )
            except Exception as exc:
                healthy = False
                error = str(exc)
            else:
                error = None

            result[name] = {
                "name": name,
                "display_name": (
                    provider.display_name
                ),
                "enabled": (
                    provider.enabled
                ),
                "configured": (
                    provider.is_configured()
                ),
                "healthy": healthy,
                "active": (
                    name
                    == self._active_provider
                ),
                "fallback": (
                    name
                    in self._fallback_providers
                ),
                "capabilities": (
                    provider.capabilities()
                ),
                "error": error,
            }

        return result

    # ============================================================
    # COLLECTION
    # ============================================================

    def get_all(
        self,
    ) -> list[MapsProvider]:
        """
        Return all registered providers.
        """

        with self._lock:
            return list(
                self._providers.values()
            )

    def names(
        self,
    ) -> list[str]:
        """
        Return all registered provider names.
        """

        with self._lock:
            return list(
                self._providers.keys()
            )

    def count(
        self,
    ) -> int:
        """Return the number of registered providers."""

        with self._lock:
            return len(
                self._providers
            )

    # ============================================================
    # IMPORT / EXPORT
    # ============================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Serialize registry metadata.

        API keys and secrets are intentionally excluded.
        """

        with self._lock:

            return {
                "active_provider": (
                    self._active_provider
                ),
                "fallback_providers": list(
                    self._fallback_providers
                ),
                "providers": {
                    name: provider.to_dict()
                    for name, provider
                    in self._providers.items()
                },
            }

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_provider(
        provider: MapsProvider,
        *,
        require_enabled: bool,
        require_configured: bool,
    ) -> None:
        """
        Validate provider usability.
        """

        if require_enabled and not provider.enabled:
            raise RuntimeError(
                f"Maps provider '{provider.name}' "
                "is disabled."
            )

        if (
            require_configured
            and not provider.is_configured()
        ):
            raise RuntimeError(
                f"Maps provider '{provider.name}' "
                "is not configured."
            )

    @staticmethod
    def _normalize_name(
        provider: str | None,
    ) -> str:
        """
        Normalize provider names and common aliases.
        """

        if provider is None:
            return ""

        name = (
            str(provider)
            .strip()
            .lower()
            .replace(
                "-",
                "_",
            )
            .replace(
                " ",
                "_",
            )
        )

        aliases = {
            "google": "google_maps",
            "googlemaps": "google_maps",
            "google_map": "google_maps",
            "maps": "google_maps",
            "map_box": "mapbox",
        }

        return aliases.get(
            name,
            name,
        )

    # ============================================================
    # CONTEXT MANAGER
    # ============================================================

    def __enter__(
        self,
    ) -> "MapsProviderRegistry":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """
        Close providers that expose a close() method.
        """

        for provider in self.get_all():

            close = getattr(
                provider,
                "close",
                None,
            )

            if callable(close):

                try:
                    close()
                except Exception:
                    pass


# ================================================================
# GLOBAL REGISTRY
# ================================================================

_registry: MapsProviderRegistry | None = None
_registry_lock = threading.Lock()


def get_maps_provider_registry(
    *,
    reset: bool = False,
) -> MapsProviderRegistry:
    """
    Return the global RENIX maps provider registry.

    The registry is lazily initialized.
    """

    global _registry

    with _registry_lock:

        if (
            _registry is None
            or reset
        ):
            _registry = (
                MapsProviderRegistry()
            )

        return _registry


def reset_maps_provider_registry() -> None:
    """
    Reset the global maps provider registry.
    """

    global _registry

    with _registry_lock:

        if _registry is not None:

            _registry.clear()

        _registry = None


__all__ = [
    "MapsProviderRegistry",
    "get_maps_provider_registry",
    "reset_maps_provider_registry",
]



