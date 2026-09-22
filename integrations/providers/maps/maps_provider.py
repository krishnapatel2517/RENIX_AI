"""
RENIX Maps Provider
===================

Base interface for all RENIX maps providers.

Concrete providers such as Google Maps and Mapbox should inherit
from MapsProvider and implement the supported operations.

The rest of RENIX should depend on this interface rather than
directly depending on a specific maps API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MapsProvider(ABC):
    """
    Abstract base class for RENIX maps providers.

    A provider is responsible for communicating with an external
    maps/location service and returning normalized Python data.

    Implementations should avoid exposing provider-specific
    response formats to the rest of RENIX.
    """

    # ============================================================
    # PROVIDER IDENTITY
    # ============================================================

    name: str = "unknown"
    display_name: str = "Unknown Maps Provider"
    version: str = "1.0.0"

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 10.0,
        enabled: bool = True,
        **config: Any,
    ) -> None:
        """
        Initialize a maps provider.

        Args:
            api_key:
                API credential used by the provider.
            timeout:
                Maximum request time in seconds.
            enabled:
                Whether the provider is enabled.
            config:
                Provider-specific configuration.
        """

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        self.api_key = api_key
        self.timeout = float(timeout)
        self.enabled = bool(enabled)
        self.config = dict(config)

        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0

    # ============================================================
    # REQUIRED OPERATIONS
    # ============================================================

    @abstractmethod
    def geocode(
        self,
        address: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert an address/place name into coordinates.

        Expected normalized structure:

            {
                "address": "...",
                "latitude": 19.0760,
                "longitude": 72.8777,
                "city": "Mumbai",
                "state": "Maharashtra",
                "country": "India",
                "postal_code": "...",
                "provider": "..."
            }
        """

        raise NotImplementedError

    @abstractmethod
    def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert coordinates into a human-readable location.
        """

        raise NotImplementedError

    @abstractmethod
    def route(
        self,
        origin: str | tuple[float, float],
        destination: str | tuple[float, float],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Calculate a route between two locations.

        Expected normalized structure:

            {
                "origin": {...},
                "destination": {...},
                "distance_meters": 1234,
                "duration_seconds": 600,
                "distance_text": "1.2 km",
                "duration_text": "10 mins",
                "polyline": "...",
                "provider": "..."
            }
        """

        raise NotImplementedError

    @abstractmethod
    def search_places(
        self,
        query: str,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        radius: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for places or points of interest.
        """

        raise NotImplementedError

    # ============================================================
    # OPTIONAL OPERATIONS
    # ============================================================

    def get_place_details(
        self,
        place_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get details for a place.

        Providers may override this method if supported.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "get_place_details()."
        )

    def get_distance_matrix(
        self,
        origins: list[
            str | tuple[float, float]
        ],
        destinations: list[
            str | tuple[float, float]
        ],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Calculate distances and travel times between multiple
        origins and destinations.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "get_distance_matrix()."
        )

    def autocomplete(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Return location/place autocomplete suggestions.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "autocomplete()."
        )

    # ============================================================
    # HEALTH / AUTHENTICATION
    # ============================================================

    def is_configured(self) -> bool:
        """
        Return whether the provider has the credentials required
        to make API requests.

        Providers with credential-free APIs may override this.
        """

        return bool(
            self.api_key
        )

    def health_check(self) -> bool:
        """
        Perform a lightweight provider health check.

        Concrete providers should override this with an actual
        API check where appropriate.

        The base implementation only checks configuration.
        """

        return (
            self.enabled
            and self.is_configured()
        )

    def enable(self) -> None:
        """Enable this provider."""

        self.enabled = True

    def disable(self) -> None:
        """Disable this provider."""

        self.enabled = False

    # ============================================================
    # VALIDATION
    # ============================================================

    def ensure_enabled(self) -> None:
        """Raise an error when the provider is disabled."""

        if not self.enabled:
            raise RuntimeError(
                f"Maps provider '{self.name}' is disabled."
            )

    def ensure_configured(self) -> None:
        """Raise an error when provider credentials are missing."""

        if not self.is_configured():
            raise RuntimeError(
                f"Maps provider '{self.name}' is not configured."
            )

    @staticmethod
    def validate_coordinates(
        latitude: float,
        longitude: float,
    ) -> tuple[float, float]:
        """
        Validate and normalize latitude/longitude.

        Returns:
            (latitude, longitude)
        """

        try:
            lat = float(latitude)
            lon = float(longitude)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Latitude and longitude must be numeric."
            ) from exc

        if not -90.0 <= lat <= 90.0:
            raise ValueError(
                "Latitude must be between -90 and 90."
            )

        if not -180.0 <= lon <= 180.0:
            raise ValueError(
                "Longitude must be between -180 and 180."
            )

        return lat, lon

    @staticmethod
    def validate_location(
        location: str,
    ) -> str:
        """Validate a textual location."""

        if not isinstance(
            location,
            str,
        ):
            raise TypeError(
                "Location must be a string."
            )

        location = location.strip()

        if not location:
            raise ValueError(
                "Location cannot be empty."
            )

        return location

    @staticmethod
    def validate_radius(
        radius: int | float,
    ) -> int:
        """Validate a search radius in meters."""

        try:
            normalized = int(radius)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Radius must be numeric."
            ) from exc

        if normalized <= 0:
            raise ValueError(
                "Radius must be greater than zero."
            )

        return normalized

    # ============================================================
    # REQUEST STATISTICS
    # ============================================================

    def record_request(
        self,
        *,
        success: bool,
    ) -> None:
        """
        Record a provider request result.

        Concrete providers can call this around API requests.
        """

        self._request_count += 1

        if success:
            self._success_count += 1
        else:
            self._failure_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Return provider request statistics."""

        success_rate = (
            self._success_count
            / self._request_count
            if self._request_count
            else 0.0
        )

        return {
            "name": self.name,
            "display_name": self.display_name,
            "version": self.version,
            "enabled": self.enabled,
            "configured": self.is_configured(),
            "requests": self._request_count,
            "successes": self._success_count,
            "failures": self._failure_count,
            "success_rate": success_rate,
        }

    def reset_stats(self) -> None:
        """Reset provider request statistics."""

        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0

    # ============================================================
    # CAPABILITIES
    # ============================================================

    def capabilities(self) -> list[str]:
        """
        Return capabilities supported by this provider.

        The base implementation detects methods implemented by
        the subclass.
        """

        capabilities: list[str] = []

        methods = {
            "geocoding": "geocode",
            "reverse_geocoding": "reverse_geocode",
            "routing": "route",
            "places": "search_places",
            "place_details": "get_place_details",
            "distance_matrix": "get_distance_matrix",
            "autocomplete": "autocomplete",
        }

        for capability, method_name in methods.items():

            method = getattr(
                self,
                method_name,
                None,
            )

            if not callable(method):
                continue

            # Methods that are still defined directly on this
            # abstract base class are not considered implemented.
            base_method = getattr(
                MapsProvider,
                method_name,
                None,
            )

            if base_method is method:
                continue

            capabilities.append(
                capability
            )

        return capabilities

    def supports(
        self,
        capability: str,
    ) -> bool:
        """Return whether the provider supports a capability."""

        normalized = (
            capability.strip().lower()
        )

        return normalized in self.capabilities()

    # ============================================================
    # NORMALIZATION HELPERS
    # ============================================================

    def normalize_location_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Add common provider metadata to a location result.
        """

        normalized = dict(
            result or {}
        )

        normalized.setdefault(
            "provider",
            self.name,
        )

        return normalized

    def normalize_route_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Add common provider metadata to a route result."""

        normalized = dict(
            result or {}
        )

        normalized.setdefault(
            "provider",
            self.name,
        )

        return normalized

    def normalize_place_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Add common provider metadata to a place result."""

        normalized = dict(
            result or {}
        )

        normalized.setdefault(
            "provider",
            self.name,
        )

        return normalized

    # ============================================================
    # REPRESENTATION
    # ============================================================

    def to_dict(self) -> dict[str, Any]:
        """Return safe provider metadata."""

        return {
            "name": self.name,
            "display_name": self.display_name,
            "version": self.version,
            "enabled": self.enabled,
            "configured": self.is_configured(),
            "capabilities": self.capabilities(),
            "stats": self.get_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"enabled={self.enabled!r}, "
            f"configured={self.is_configured()!r}"
            f")"
        )


__all__ = [
    "MapsProvider",
]


