"""
RENIX AI Providers
==================

Provider integrations for external AI services.

This package provides a common namespace for AI providers used
by RENIX's integration layer.

Provider implementations should expose a consistent interface
through the higher-level integrations/api_manager.py and
services.py modules.

Keep provider-specific credentials and secrets outside this
package. They should be supplied through RENIX's configuration
and secrets-management systems.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# PACKAGE METADATA
# ============================================================

__title__ = "RENIX AI Providers"
__description__ = (
    "External AI provider integrations for RENIX."
)
__version__ = "1.0.0"


# ============================================================
# PROVIDER REGISTRY
# ============================================================

_PROVIDER_REGISTRY: dict[str, Any] = {}


def register_provider(
    name: str,
    provider: Any,
    *,
    replace: bool = False,
) -> None:
    """
    Register an AI provider.

    Parameters
    ----------
    name:
        Unique provider name.

    provider:
        Provider implementation or provider class.

    replace:
        Replace an existing provider when True.
    """

    if not isinstance(name, str):
        raise TypeError(
            "Provider name must be a string."
        )

    normalized_name = name.strip().lower()

    if not normalized_name:
        raise ValueError(
            "Provider name cannot be empty."
        )

    if (
        normalized_name in _PROVIDER_REGISTRY
        and not replace
    ):
        raise ValueError(
            f"AI provider already registered: "
            f"{normalized_name}"
        )

    _PROVIDER_REGISTRY[
        normalized_name
    ] = provider


def unregister_provider(
    name: str,
) -> bool:
    """
    Remove an AI provider from the registry.

    Returns True if a provider was removed.
    """

    normalized_name = str(
        name
    ).strip().lower()

    return (
        _PROVIDER_REGISTRY.pop(
            normalized_name,
            None,
        )
        is not None
    )


def get_provider(
    name: str,
) -> Any:
    """
    Retrieve a registered provider.

    Raises
    ------
    KeyError
        If the provider does not exist.
    """

    normalized_name = str(
        name
    ).strip().lower()

    try:
        return _PROVIDER_REGISTRY[
            normalized_name
        ]
    except KeyError as exc:
        raise KeyError(
            f"AI provider not registered: "
            f"{normalized_name}"
        ) from exc


def get_provider_optional(
    name: str,
) -> Any | None:
    """Return a provider or None."""

    normalized_name = str(
        name
    ).strip().lower()

    return _PROVIDER_REGISTRY.get(
        normalized_name
    )


def provider_exists(
    name: str,
) -> bool:
    """Check whether an AI provider is registered."""

    normalized_name = str(
        name
    ).strip().lower()

    return (
        normalized_name
        in _PROVIDER_REGISTRY
    )


def list_providers() -> list[str]:
    """Return all registered provider names."""

    return sorted(
        _PROVIDER_REGISTRY.keys()
    )


def clear_providers() -> None:
    """Clear the provider registry."""

    _PROVIDER_REGISTRY.clear()


# ============================================================
# PROVIDER INFORMATION
# ============================================================


def provider_count() -> int:
    """Return the number of registered providers."""

    return len(
        _PROVIDER_REGISTRY
    )


def provider_status() -> dict[str, Any]:
    """
    Return safe registry information.

    Provider objects themselves are intentionally not exposed.
    """

    return {
        "package": __title__,
        "version": __version__,
        "provider_count": provider_count(),
        "providers": list_providers(),
    }


# ============================================================
# OPTIONAL AUTO-DISCOVERY
# ============================================================


def discover_provider(
    name: str,
) -> Any | None:
    """
    Attempt to lazily import a known provider.

    This keeps __init__.py lightweight and allows individual
    provider modules to be developed independently.

    Unknown providers simply return None.
    """

    normalized_name = str(
        name
    ).strip().lower()

    module_map = {
        "openai": (
            "openai_provider",
            "OpenAIProvider",
        ),
        "gemini": (
            "gemini_provider",
            "GeminiProvider",
        ),
        "google": (
            "gemini_provider",
            "GeminiProvider",
        ),
        "anthropic": (
            "anthropic_provider",
            "AnthropicProvider",
        ),
        "ollama": (
            "ollama_provider",
            "OllamaProvider",
        ),
        "local": (
            "local_provider",
            "LocalAIProvider",
        ),
    }

    target = module_map.get(
        normalized_name
    )

    if target is None:
        return None

    module_name, class_name = target

    try:
        module = __import__(
            f"{__name__}.{module_name}",
            fromlist=[class_name],
        )

        provider = getattr(
            module,
            class_name,
        )

    except (
        ImportError,
        AttributeError,
    ):
        return None

    register_provider(
        normalized_name,
        provider,
        replace=True,
    )

    return provider


def get_or_discover_provider(
    name: str,
) -> Any:
    """
    Get a provider from the registry or attempt discovery.
    """

    existing = get_provider_optional(
        name
    )

    if existing is not None:
        return existing

    discovered = discover_provider(
        name
    )

    if discovered is None:
        raise KeyError(
            f"Unable to discover AI provider: "
            f"{name}"
        )

    return discovered


# ============================================================
# PACKAGE HEALTH
# ============================================================


def health_check() -> dict[str, Any]:
    """
    Return a lightweight package health report.
    """

    providers = {}

    for name, provider in (
        _PROVIDER_REGISTRY.items()
    ):
        providers[name] = {
            "registered": True,
            "type": type(provider).__name__,
        }

    return {
        "healthy": True,
        "package": __title__,
        "version": __version__,
        "providers": providers,
    }


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "register_provider",
    "unregister_provider",
    "get_provider",
    "get_provider_optional",
    "get_or_discover_provider",
    "provider_exists",
    "list_providers",
    "provider_count",
    "provider_status",
    "discover_provider",
    "clear_providers",
    "health_check",
]


