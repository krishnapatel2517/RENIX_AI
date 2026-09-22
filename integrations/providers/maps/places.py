"""
RENIX Places Service
====================

Provider-independent Places layer for RENIX.

Responsibilities:
    - Search nearby places
    - Text-based place search
    - Place details
    - Autocomplete
    - Provider fallback
    - Result normalization
    - Basic caching

This module does not directly depend on Google Maps or Mapbox.
It communicates through the registered MapsProvider instances.
"""

from __future__ import annotations

import time
import threading
from typing import Any


class PlacesError(RuntimeError):
    """Base exception for Places-related errors."""


class PlacesNotFoundError(PlacesError):
    """Raised when no places can be found."""


class PlacesService:
    """
    Provider-independent Places service.

    Example:

        places = PlacesService(registry)

        results = places.search(
            "coffee shops",
            latitude=19.0760,
            longitude=72.8777,
        )

        details = places.details(
            "place_id",
            session_token="..."
        )
    """

    def __init__(
        self,
        registry: Any,
        *,
        cache_enabled: bool = True,
        cache_ttl: int = 300,
        max_cache_size: int = 500,
    ) -> None:

        if registry is None:
            raise ValueError(
                "A maps provider registry is required."
            )

        self.registry = registry

        self.cache_enabled = bool(
            cache_enabled
        )

        self.cache_ttl = max(
            0,
            int(cache_ttl),
        )

        self.max_cache_size = max(
            1,
            int(max_cache_size),
        )

        self._search_cache: dict[
            str,
            tuple[
                float,
                list[dict[str, Any]],
            ],
        ] = {}

        self._details_cache: dict[
            str,
            tuple[
                float,
                dict[str, Any],
            ],
        ] = {}

        self._autocomplete_cache: dict[
            str,
            tuple[
                float,
                list[dict[str, Any]],
            ],
        ] = {}

        self._lock = threading.RLock()

    # ============================================================
    # PLACE SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        radius: int | None = None,
        provider: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for places.

        Args:
            query:
                Search term such as "restaurants",
                "coffee shops", "petrol pumps", etc.

            latitude / longitude:
                Optional location around which to search.

            radius:
                Optional search radius in meters.

            provider:
                Optional maps provider name.

            fallback:
                Whether other providers may be tried.

        Returns:
            List of normalized places.
        """

        query = self._validate_query(
            query
        )

        latitude, longitude = (
            self._validate_optional_coordinates(
                latitude,
                longitude,
            )
        )

        cache_key = self._search_cache_key(
            query,
            latitude,
            longitude,
            radius,
            provider,
            kwargs,
        )

        cached = self._get_cache(
            self._search_cache,
            cache_key,
        )

        if cached is not None:
            return cached

        providers = self._get_provider_chain(
            provider=provider,
            fallback=fallback,
        )

        errors: list[str] = []

        for maps_provider in providers:

            try:

                results = (
                    maps_provider.search_places(
                        query,
                        latitude=latitude,
                        longitude=longitude,
                        radius=radius,
                        **kwargs,
                    )
                )

                normalized = [
                    self._normalize_place(
                        result
                    )
                    for result in (
                        results or []
                    )
                ]

                normalized = [
                    result
                    for result in normalized
                    if result is not None
                ]

                if not normalized:
                    raise PlacesNotFoundError(
                        f"No places found for '{query}'."
                    )

                normalized = (
                    self._deduplicate(
                        normalized
                    )
                )

                self._set_cache(
                    self._search_cache,
                    cache_key,
                    normalized,
                )

                return normalized

            except Exception as exc:

                errors.append(
                    f"{maps_provider.name}: {exc}"
                )

        raise PlacesNotFoundError(
            self._build_error(
                "Unable to find places",
                query,
                errors,
            )
        )

    # ============================================================
    # NEARBY SEARCH
    # ============================================================

    def nearby(
        self,
        latitude: float,
        longitude: float,
        *,
        query: str = "",
        radius: int = 1000,
        provider: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for places around coordinates.

        Example:

            places.nearby(
                19.0760,
                72.8777,
                query="restaurants",
                radius=2000,
            )
        """

        latitude, longitude = (
            self._validate_coordinates(
                latitude,
                longitude,
            )
        )

        radius = self._validate_radius(
            radius
        )

        if query.strip():

            return self.search(
                query,
                latitude=latitude,
                longitude=longitude,
                radius=radius,
                provider=provider,
                fallback=fallback,
                **kwargs,
            )

        return self.search(
            "places",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            provider=provider,
            fallback=fallback,
            **kwargs,
        )

    # ============================================================
    # PLACE DETAILS
    # ============================================================

    def details(
        self,
        place_id: str,
        *,
        provider: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Retrieve details for a specific place.

        For providers such as Mapbox Searchbox, a session token
        may be required and should be passed through kwargs.
        """

        place_id = self._validate_place_id(
            place_id
        )

        cache_key = (
            f"{provider or 'auto'}|"
            f"{place_id}"
        )

        cached = self._get_cache(
            self._details_cache,
            cache_key,
        )

        if cached is not None:
            return cached

        providers = self._get_provider_chain(
            provider=provider,
            fallback=fallback,
        )

        errors: list[str] = []

        for maps_provider in providers:

            try:

                result = (
                    maps_provider.get_place_details(
                        place_id,
                        **kwargs,
                    )
                )

                normalized = (
                    self._normalize_place(
                        result
                    )
                )

                if normalized is None:
                    raise PlacesError(
                        "Provider returned an invalid place."
                    )

                self._set_cache(
                    self._details_cache,
                    cache_key,
                    normalized,
                )

                return normalized

            except Exception as exc:

                errors.append(
                    f"{maps_provider.name}: {exc}"
                )

        raise PlacesNotFoundError(
            self._build_error(
                "Unable to retrieve place details",
                place_id,
                errors,
            )
        )

    # ============================================================
    # AUTOCOMPLETE
    # ============================================================

    def autocomplete(
        self,
        query: str,
        *,
        provider: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Return autocomplete suggestions for a place query.
        """

        query = self._validate_query(
            query
        )

        cache_key = (
            f"{provider or 'auto'}|"
            f"{query.lower()}|"
            f"{tuple(sorted(kwargs.items()))}"
        )

        cached = self._get_cache(
            self._autocomplete_cache,
            cache_key,
        )

        if cached is not None:
            return cached

        providers = self._get_provider_chain(
            provider=provider,
            fallback=fallback,
        )

        errors: list[str] = []

        for maps_provider in providers:

            try:

                results = (
                    maps_provider.autocomplete(
                        query,
                        **kwargs,
                    )
                )

                normalized = [
                    self._normalize_suggestion(
                        result
                    )
                    for result in (
                        results or []
                    )
                ]

                normalized = [
                    result
                    for result in normalized
                    if result is not None
                ]

                if not normalized:
                    raise PlacesNotFoundError(
                        f"No suggestions found for '{query}'."
                    )

                self._set_cache(
                    self._autocomplete_cache,
                    cache_key,
                    normalized,
                )

                return normalized

            except Exception as exc:

                errors.append(
                    f"{maps_provider.name}: {exc}"
                )

        raise PlacesNotFoundError(
            self._build_error(
                "Unable to generate place suggestions",
                query,
                errors,
            )
        )

    # ============================================================
    # CATEGORY SEARCH
    # ============================================================

    def category(
        self,
        category: str,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        radius: int = 2000,
        provider: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search places by category.

        Examples:

            category("restaurant")
            category("hospital")
            category("petrol station")
            category("pharmacy")
        """

        category = self._validate_query(
            category
        )

        return self.search(
            category,
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            provider=provider,
            fallback=fallback,
            **kwargs,
        )

    # ============================================================
    # COMMON CONVENIENCE METHODS
    # ============================================================

    def restaurants(
        self,
        latitude: float,
        longitude: float,
        *,
        radius: int = 3000,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Find restaurants nearby."""

        return self.category(
            "restaurants",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def cafes(
        self,
        latitude: float,
        longitude: float,
        *,
        radius: int = 3000,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Find cafes nearby."""

        return self.category(
            "cafes",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def hospitals(
        self,
        latitude: float,
        longitude: float,
        *,
        radius: int = 5000,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Find hospitals nearby."""

        return self.category(
            "hospitals",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def pharmacies(
        self,
        latitude: float,
        longitude: float,
        *,
        radius: int = 3000,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Find pharmacies nearby."""

        return self.category(
            "pharmacies",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def petrol_stations(
        self,
        latitude: float,
        longitude: float,
        *,
        radius: int = 5000,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Find petrol stations nearby."""

        return self.category(
            "petrol stations",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    # ============================================================
    # PROVIDER CHAIN
    # ============================================================

    def _get_provider_chain(
        self,
        *,
        provider: str | None,
        fallback: bool,
    ) -> list[Any]:
        """
        Build the provider execution chain.
        """

        if provider:

            selected = self.registry.get(
                provider
            )

            if not fallback:
                return [selected]

            chain = [
                selected
            ]

            for fallback_provider in (
                self.registry.get_fallbacks()
            ):

                if (
                    fallback_provider
                    not in chain
                ):
                    chain.append(
                        fallback_provider
                    )

            return chain

        if fallback:

            return self.registry.get_provider_chain(
                include_active=True
            )

        return [
            self.registry.get_active()
        ]

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_place(
        place: Any,
    ) -> dict[str, Any] | None:
        """
        Convert provider-specific place data into a
        common RENIX format.
        """

        if not isinstance(
            place,
            dict,
        ):
            return None

        result = dict(
            place
        )

        result.setdefault(
            "place_id",
            result.get(
                "id"
            )
            or result.get(
                "mapbox_id"
            ),
        )

        result.setdefault(
            "name",
            result.get(
                "name_preferred"
            )
            or result.get(
                "text"
            )
            or result.get(
                "address"
            ),
        )

        result.setdefault(
            "formatted_address",
            result.get(
                "address"
            )
            or result.get(
                "full_address"
            ),
        )

        result.setdefault(
            "latitude",
            None,
        )

        result.setdefault(
            "longitude",
            None,
        )

        result.setdefault(
            "provider",
            "unknown",
        )

        result.setdefault(
            "category",
            result.get(
                "categories"
            ),
        )

        result.setdefault(
            "phone",
            None,
        )

        result.setdefault(
            "website",
            None,
        )

        result.setdefault(
            "rating",
            None,
        )

        result.setdefault(
            "opening_hours",
            None,
        )

        result.setdefault(
            "distance_meters",
            None,
        )

        return result

    @staticmethod
    def _normalize_suggestion(
        suggestion: Any,
    ) -> dict[str, Any] | None:
        """
        Normalize autocomplete suggestions.
        """

        if not isinstance(
            suggestion,
            dict,
        ):
            return None

        return {
            "place_id": (
                suggestion.get(
                    "place_id"
                )
                or suggestion.get(
                    "mapbox_id"
                )
                or suggestion.get(
                    "id"
                )
            ),
            "name": (
                suggestion.get(
                    "name"
                )
                or suggestion.get(
                    "name_preferred"
                )
            ),
            "description": (
                suggestion.get(
                    "description"
                )
                or suggestion.get(
                    "full_address"
                )
                or suggestion.get(
                    "place_formatted"
                )
            ),
            "address": suggestion.get(
                "address"
            ),
            "feature_type": suggestion.get(
                "feature_type"
            ),
            "provider": suggestion.get(
                "provider",
                "unknown",
            ),
            "raw": suggestion,
        }

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    @staticmethod
    def _deduplicate(
        places: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Remove duplicate places while preserving order.
        """

        seen_ids: set[str] = set()
        seen_names: set[str] = set()

        results: list[
            dict[str, Any]
        ] = []

        for place in places:

            place_id = place.get(
                "place_id"
            )

            name = place.get(
                "name"
            )

            if place_id:

                key = str(
                    place_id
                ).lower()

                if key in seen_ids:
                    continue

                seen_ids.add(
                    key
                )

            elif name:

                key = str(
                    name
                ).strip().lower()

                if key in seen_names:
                    continue

                seen_names.add(
                    key
                )

            results.append(
                place
            )

        return results

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_query(
        query: str,
    ) -> str:
        """Validate a place search query."""

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "Place query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Place query cannot be empty."
            )

        if len(query) > 500:
            raise ValueError(
                "Place query is too long."
            )

        return query

    @staticmethod
    def _validate_place_id(
        place_id: str,
    ) -> str:
        """Validate a place ID."""

        if not isinstance(
            place_id,
            str,
        ):
            raise TypeError(
                "place_id must be a string."
            )

        place_id = place_id.strip()

        if not place_id:
            raise ValueError(
                "place_id cannot be empty."
            )

        return place_id

    @staticmethod
    def _validate_coordinates(
        latitude: float,
        longitude: float,
    ) -> tuple[float, float]:
        """Validate geographic coordinates."""

        try:

            latitude = float(
                latitude
            )

            longitude = float(
                longitude
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Latitude and longitude must be numeric."
            ) from exc

        if not -90 <= latitude <= 90:
            raise ValueError(
                "Latitude must be between -90 and 90."
            )

        if not -180 <= longitude <= 180:
            raise ValueError(
                "Longitude must be between -180 and 180."
            )

        return latitude, longitude

    @classmethod
    def _validate_optional_coordinates(
        cls,
        latitude: float | None,
        longitude: float | None,
    ) -> tuple[
        float | None,
        float | None,
    ]:
        """Validate optional coordinates."""

        if (
            latitude is None
            and longitude is None
        ):
            return None, None

        if (
            latitude is None
            or longitude is None
        ):
            raise ValueError(
                "Both latitude and longitude "
                "must be supplied together."
            )

        return cls._validate_coordinates(
            latitude,
            longitude,
        )

    @staticmethod
    def _validate_radius(
        radius: int | float,
    ) -> int:
        """Validate search radius in meters."""

        try:
            radius = int(
                radius
            )
        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Radius must be numeric."
            ) from exc

        if radius <= 0:
            raise ValueError(
                "Radius must be greater than zero."
            )

        if radius > 100000:
            raise ValueError(
                "Radius cannot exceed 100000 meters."
            )

        return radius

    # ============================================================
    # CACHE
    # ============================================================

    @staticmethod
    def _get_cache(
        cache: dict[str, Any],
        key: str,
    ) -> Any:
        """
        Return an unexpired cached result.
        """

        item = cache.get(
            key
        )

        if item is None:
            return None

        timestamp, value, ttl = item

        if (
            time.monotonic()
            - timestamp
            > ttl
        ):

            cache.pop(
                key,
                None,
            )

            return None

        if isinstance(
            value,
            list,
        ):
            return [
                dict(item)
                for item in value
            ]

        if isinstance(
            value,
            dict,
        ):
            return dict(
                value
            )

        return value

    def _set_cache(
        self,
        cache: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        """Store a result in cache."""

        if not self.cache_enabled:
            return

        with self._lock:

            while len(cache) >= self.max_cache_size:

                oldest = next(
                    iter(cache),
                    None,
                )

                if oldest is None:
                    break

                cache.pop(
                    oldest,
                    None,
                )

            if isinstance(
                value,
                list,
            ):
                value = [
                    dict(item)
                    for item in value
                ]

            elif isinstance(
                value,
                dict,
            ):
                value = dict(
                    value
                )

            cache[
                key
            ] = (
                time.monotonic(),
                value,
                self.cache_ttl,
            )

    @staticmethod
    def _search_cache_key(
        query: str,
        latitude: float | None,
        longitude: float | None,
        radius: int | None,
        provider: str | None,
        kwargs: dict[str, Any],
    ) -> str:
        """Build a stable cache key."""

        extras = tuple(
            sorted(
                (
                    str(key),
                    str(value),
                )
                for key, value
                in kwargs.items()
            )
        )

        return (
            f"{provider or 'auto'}|"
            f"{query.lower()}|"
            f"{latitude}|"
            f"{longitude}|"
            f"{radius}|"
            f"{extras}"
        )

    def clear_cache(self) -> None:
        """Clear all Places caches."""

        with self._lock:

            self._search_cache.clear()
            self._details_cache.clear()
            self._autocomplete_cache.clear()

    # ============================================================
    # ERRORS
    # ============================================================

    @staticmethod
    def _build_error(
        operation: str,
        target: str,
        errors: list[str],
    ) -> str:
        """Build a readable provider error."""

        message = (
            f"{operation}: '{target}'."
        )

        if errors:

            message += (
                " Provider errors: "
                + " | ".join(errors)
            )

        return message

    # ============================================================
    # CONTEXT MANAGER
    # ============================================================

    def __enter__(
        self,
    ) -> "PlacesService":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.clear_cache()


__all__ = [
    "PlacesService",
    "PlacesError",
    "PlacesNotFoundError",
]



