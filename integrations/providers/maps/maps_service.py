"""
RENIX Maps Service
==================

High-level Maps service used by RENIX.

This layer provides a single interface for:
    - Geocoding
    - Reverse geocoding
    - Places
    - Routing
    - Distance
    - Directions
    - Nearby search

Provider-specific implementations should remain inside
integrations/providers/maps/provider adapters.

This service intentionally does not depend on a single
maps provider.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from .routing import (
    Coordinate,
    Route,
    RouteStep,
    RoutingResult,
    RoutingService,
)

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class MapsServiceError(RuntimeError):
    """Base exception for the RENIX Maps service."""


class MapsProviderError(MapsServiceError):
    """Raised when a maps provider fails."""


class MapsConfigurationError(MapsServiceError):
    """Raised when the maps service is incorrectly configured."""


# ============================================================
# MAPS SERVICE
# ============================================================


class MapsService:
    """
    Main high-level Maps interface for RENIX.

    Example:

        maps = MapsService(provider_registry)

        route = maps.route(
            (19.0760, 72.8777),
            (19.1136, 72.8697),
        )

        places = maps.search_places(
            "coffee shops",
            latitude=19.0760,
            longitude=72.8777,
        )
    """

    def __init__(
        self,
        provider_registry: Any,
        *,
        default_provider: str | None = None,
        routing_cache_enabled: bool = True,
        routing_cache_ttl: int = 300,
    ) -> None:

        if provider_registry is None:
            raise MapsConfigurationError(
                "provider_registry is required."
            )

        self.registry = provider_registry

        self.default_provider = (
            default_provider
        )

        self.routing = RoutingService(
            provider_registry,
            cache_enabled=(
                routing_cache_enabled
            ),
            cache_ttl=routing_cache_ttl,
        )

    # ========================================================
    # PROVIDER
    # ========================================================

    def provider(
        self,
        name: str | None = None,
    ) -> Any:
        """
        Return a maps provider.

        If no name is supplied, the configured default
        provider or active provider is returned.
        """

        provider_name = (
            name
            or self.default_provider
        )

        if provider_name:

            try:

                return self.registry.get(
                    provider_name
                )

            except Exception as exc:

                raise MapsProviderError(
                    f"Unable to load maps provider "
                    f"'{provider_name}'."
                ) from exc

        getter = getattr(
            self.registry,
            "get_active",
            None,
        )

        if not callable(getter):

            raise MapsProviderError(
                "No active maps provider is available."
            )

        active = getter()

        if active is None:

            raise MapsProviderError(
                "Maps provider registry returned no active provider."
            )

        return active

    # ========================================================
    # GEOCODING
    # ========================================================

    def geocode(
        self,
        address: str,
        *,
        provider: str | None = None,
        limit: int = 5,
        country: str | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Convert a human-readable address into locations.

        Example:

            maps.geocode("Gateway of India, Mumbai")
        """

        if not isinstance(
            address,
            str,
        ) or not address.strip():

            raise ValueError(
                "Address must be a non-empty string."
            )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "geocode",
                "search_address",
                "geocode_address",
            ),
        )

        if method is None:

            raise MapsProviderError(
                "The active maps provider does not "
                "support geocoding."
            )

        payload = {
            "address": address.strip(),
            "limit": max(1, int(limit)),
        }

        if country:
            payload["country"] = country

        if language:
            payload["language"] = language

        payload.update(kwargs)

        try:

            result = method(
                **payload
            )

            return self._normalize_list(
                result
            )

        except Exception as exc:

            logger.exception(
                "Geocoding failed."
            )

            raise MapsProviderError(
                f"Geocoding failed: {exc}"
            ) from exc

    # ========================================================
    # REVERSE GEOCODING
    # ========================================================

    def reverse_geocode(
        self,
        latitude: float | Coordinate,
        longitude: float | None = None,
        *,
        provider: str | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Convert coordinates into a human-readable address.
        """

        coordinate = self._coordinate(
            latitude,
            longitude,
        )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "reverse_geocode",
                "reverse",
                "address_from_coordinates",
            ),
        )

        if method is None:

            raise MapsProviderError(
                "The active maps provider does not "
                "support reverse geocoding."
            )

        payload = {
            "latitude": coordinate.latitude,
            "longitude": coordinate.longitude,
        }

        if language:
            payload["language"] = language

        payload.update(kwargs)

        try:

            return self._normalize_list(
                method(**payload)
            )

        except Exception as exc:

            logger.exception(
                "Reverse geocoding failed."
            )

            raise MapsProviderError(
                f"Reverse geocoding failed: {exc}"
            ) from exc

    # ========================================================
    # PLACES
    # ========================================================

    def search_places(
        self,
        query: str,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        radius: float | None = None,
        provider: str | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Search for places.

        Example:

            maps.search_places(
                "restaurants",
                latitude=19.0760,
                longitude=72.8777,
                radius=5000,
            )
        """

        if not isinstance(
            query,
            str,
        ) or not query.strip():

            raise ValueError(
                "Place query must be non-empty."
            )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "search_places",
                "places_search",
                "search",
                "find_places",
            ),
        )

        if method is None:

            raise MapsProviderError(
                "The active maps provider does not "
                "support place search."
            )

        payload = {
            "query": query.strip(),
            "limit": max(1, int(limit)),
        }

        if (
            latitude is not None
            and longitude is not None
        ):

            coordinate = self._coordinate(
                latitude,
                longitude,
            )

            payload[
                "latitude"
            ] = coordinate.latitude

            payload[
                "longitude"
            ] = coordinate.longitude

        if radius is not None:

            payload[
                "radius"
            ] = float(radius)

        payload.update(kwargs)

        try:

            return self._normalize_list(
                method(**payload)
            )

        except Exception as exc:

            logger.exception(
                "Place search failed."
            )

            raise MapsProviderError(
                f"Place search failed: {exc}"
            ) from exc

    # ========================================================
    # NEARBY PLACES
    # ========================================================

    def nearby_places(
        self,
        latitude: float,
        longitude: float,
        *,
        place_type: str | None = None,
        radius: float = 1000,
        provider: str | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Search for nearby places.
        """

        coordinate = self._coordinate(
            latitude,
            longitude,
        )

        maps_provider = self.provider(
            provider
        )

        method = self._find_method(
            maps_provider,
            (
                "nearby_places",
                "search_nearby",
                "nearby_search",
            ),
        )

        if method is None:

            # Fall back to generic place search.
            query = (
                place_type
                or "places"
            )

            return self.search_places(
                query,
                latitude=coordinate.latitude,
                longitude=coordinate.longitude,
                radius=radius,
                provider=provider,
                limit=limit,
                **kwargs,
            )

        payload = {
            "latitude": coordinate.latitude,
            "longitude": coordinate.longitude,
            "radius": float(radius),
            "limit": max(1, int(limit)),
        }

        if place_type:
            payload[
                "place_type"
            ] = place_type

        payload.update(kwargs)

        try:

            return self._normalize_list(
                method(**payload)
            )

        except Exception as exc:

            raise MapsProviderError(
                f"Nearby place search failed: {exc}"
            ) from exc

    # ========================================================
    # ROUTING
    # ========================================================

    def route(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        provider: str | None = None,
        alternatives: bool = False,
        avoid: Iterable[str] | None = None,
        waypoints: Iterable[
            Coordinate
            | tuple[float, float]
        ]
        | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> RoutingResult:
        """
        Calculate a route.
        """

        return self.routing.route(
            origin,
            destination,
            mode=mode,
            provider=provider,
            alternatives=alternatives,
            avoid=avoid,
            waypoints=waypoints,
            fallback=fallback,
            **kwargs,
        )

    # ========================================================
    # DISTANCE
    # ========================================================

    def distance(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        provider: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Return distance in meters."""

        return self.routing.distance(
            origin,
            destination,
            mode=mode,
            provider=provider,
            **kwargs,
        )

    def distance_km(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        provider: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Return distance in kilometers."""

        return self.routing.distance_km(
            origin,
            destination,
            mode=mode,
            provider=provider,
            **kwargs,
        )

    # ========================================================
    # DIRECTIONS
    # ========================================================

    def directions(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        provider: str | None = None,
        **kwargs: Any,
    ) -> list[RouteStep]:
        """Return turn-by-turn directions."""

        return self.routing.directions(
            origin,
            destination,
            mode=mode,
            provider=provider,
            **kwargs,
        )

    # ========================================================
    # DURATION
    # ========================================================

    def duration(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        provider: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Return route duration in seconds."""

        return self.routing.duration(
            origin,
            destination,
            mode=mode,
            provider=provider,
            **kwargs,
        )

    def duration_minutes(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        provider: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Return route duration in minutes."""

        return self.routing.duration_minutes(
            origin,
            destination,
            mode=mode,
            provider=provider,
            **kwargs,
        )

    # ========================================================
    # SIMPLE NAVIGATION SUMMARY
    # ========================================================

    def navigation_summary(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Return a compact navigation summary suitable for
        RENIX voice responses or UI widgets.
        """

        result = self.route(
            origin,
            destination,
            mode=mode,
            provider=provider,
            **kwargs,
        )

        route = result.best_route

        if route is None:

            raise MapsServiceError(
                "No route available."
            )

        return {
            "provider": result.provider,
            "mode": mode,
            "distance_meters": (
                route.distance_meters
            ),
            "distance_km": (
                route.distance_km()
            ),
            "duration_seconds": (
                route.duration_seconds
            ),
            "duration_minutes": (
                route.duration_minutes()
            ),
            "summary": route.summary,
            "steps": [
                {
                    "instruction": step.instruction,
                    "distance_meters": (
                        step.distance_meters
                    ),
                    "duration_seconds": (
                        step.duration_seconds
                    ),
                    "maneuver": step.maneuver,
                    "road_name": step.road_name,
                }
                for step in route.steps
            ],
        }

    # ========================================================
    # CACHE
    # ========================================================

    def clear_route_cache(self) -> None:
        """Clear routing cache."""

        self.routing.clear_cache()

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _find_method(
        provider: Any,
        names: tuple[str, ...],
    ) -> Any | None:

        for name in names:

            method = getattr(
                provider,
                name,
                None,
            )

            if callable(method):
                return method

        return None

    @staticmethod
    def _normalize_list(
        value: Any,
    ) -> list[Any]:

        if value is None:
            return []

        if isinstance(
            value,
            list,
        ):
            return value

        if isinstance(
            value,
            tuple,
        ):
            return list(value)

        return [value]

    @staticmethod
    def _coordinate(
        latitude: float | Coordinate,
        longitude: float | None = None,
    ) -> Coordinate:

        if isinstance(
            latitude,
            Coordinate,
        ):

            return latitude

        if longitude is None:

            raise ValueError(
                "Longitude is required."
            )

        coordinate = Coordinate(
            latitude=float(
                latitude
            ),
            longitude=float(
                longitude
            ),
        )

        coordinate.validate()

        return coordinate


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "MapsService",
    "MapsServiceError",
    "MapsProviderError",
    "MapsConfigurationError",
]


