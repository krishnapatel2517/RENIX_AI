"""
RENIX Maps Providers
====================

Maps integration package for RENIX.

This package provides a unified interface for:

    - Geocoding
    - Reverse geocoding
    - Routing
    - Distance and travel time
    - Places search
    - Place details
    - Maps provider management
    - Response caching

Provider implementations are kept separate from the public
service layer so RENIX can switch providers or use fallback
providers without changing the rest of the application.
"""

from __future__ import annotations


# ============================================================
# OPTIONAL IMPORTS
# ============================================================
#
# Imports are intentionally guarded so that the package can
# still be imported while individual provider dependencies are
# unavailable during development or first-time setup.
# ============================================================


try:
    from .maps_provider import (
        MapsProvider,
    )
except ImportError:
    MapsProvider = None  # type: ignore[assignment]


try:
    from .google_maps_provider import (
        GoogleMapsProvider,
    )
except ImportError:
    GoogleMapsProvider = None  # type: ignore[assignment]


try:
    from .mapbox_provider import (
        MapboxProvider,
    )
except ImportError:
    MapboxProvider = None  # type: ignore[assignment]


try:
    from .provider_registry import (
        MapsProviderRegistry,
        get_maps_provider_registry,
        reset_maps_provider_registry,
    )
except ImportError:
    MapsProviderRegistry = None  # type: ignore[assignment]
    get_maps_provider_registry = None  # type: ignore[assignment]
    reset_maps_provider_registry = None  # type: ignore[assignment]


try:
    from .geocoding import (
        GeocodingService,
    )
except ImportError:
    GeocodingService = None  # type: ignore[assignment]


try:
    from .routing import (
        RoutingService,
    )
except ImportError:
    RoutingService = None  # type: ignore[assignment]


try:
    from .places import (
        PlacesService,
    )
except ImportError:
    PlacesService = None  # type: ignore[assignment]


try:
    from .maps_cache import (
        MapsCache,
        get_maps_cache,
        reset_maps_cache,
    )
except ImportError:
    MapsCache = None  # type: ignore[assignment]
    get_maps_cache = None  # type: ignore[assignment]
    reset_maps_cache = None  # type: ignore[assignment]


try:
    from .maps_service import (
        MapsService,
        get_maps_service,
        reset_maps_service,
    )
except ImportError:
    MapsService = None  # type: ignore[assignment]
    get_maps_service = None  # type: ignore[assignment]
    reset_maps_service = None  # type: ignore[assignment]


# ============================================================
# PACKAGE VERSION
# ============================================================

__version__ = "1.0.0"


# ============================================================
# PROVIDER CAPABILITIES
# ============================================================

CAPABILITIES = {
    "geocoding": (
        "Convert an address or place name into coordinates."
    ),
    "reverse_geocoding": (
        "Convert coordinates into a human-readable location."
    ),
    "routing": (
        "Calculate routes, distances, and travel times."
    ),
    "places": (
        "Search for nearby places and points of interest."
    ),
    "place_details": (
        "Retrieve information about a specific place."
    ),
}


# ============================================================
# PACKAGE INFORMATION
# ============================================================


def get_package_info() -> dict[str, object]:
    """
    Return information about the RENIX maps integration.
    """

    return {
        "name": "RENIX Maps Providers",
        "version": __version__,
        "capabilities": list(
            CAPABILITIES.keys()
        ),
        "providers": [
            "google_maps",
            "mapbox",
        ],
    }


# ============================================================
# AVAILABILITY HELPERS
# ============================================================


def is_provider_available(
    provider: str,
) -> bool:
    """
    Check whether a provider class is available.

    This checks whether the provider implementation could be
    imported successfully. It does not perform an API health
    check.
    """

    provider_name = (
        provider.strip().lower()
    )

    if provider_name in {
        "google",
        "google_maps",
        "googlemaps",
    }:
        return GoogleMapsProvider is not None

    if provider_name in {
        "mapbox",
    }:
        return MapboxProvider is not None

    return False


def get_available_providers() -> list[str]:
    """
    Return providers whose implementation is currently
    importable.
    """

    providers: list[str] = []

    if GoogleMapsProvider is not None:
        providers.append(
            "google_maps"
        )

    if MapboxProvider is not None:
        providers.append(
            "mapbox"
        )

    return providers


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # Version / information
    "__version__",
    "CAPABILITIES",
    "get_package_info",

    # Provider interface
    "MapsProvider",

    # Providers
    "GoogleMapsProvider",
    "MapboxProvider",

    # Registry
    "MapsProviderRegistry",
    "get_maps_provider_registry",
    "reset_maps_provider_registry",

    # Services
    "GeocodingService",
    "RoutingService",
    "PlacesService",

    # Cache
    "MapsCache",
    "get_maps_cache",
    "reset_maps_cache",

    # Main service
    "MapsService",
    "get_maps_service",
    "reset_maps_service",

    # Availability
    "is_provider_available",
    "get_available_providers",
]


