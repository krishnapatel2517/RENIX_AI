"""
RENIX Geocoding Service
=======================

Provider-independent geocoding layer.

This module keeps RENIX code independent from Google Maps,
Mapbox, or any other individual maps provider.

Responsibilities:
    - Address -> coordinates
    - Coordinates -> address
    - Provider fallback
    - Result normalization
    - Basic validation
    - Simple caching

Example:

    geocoder = GeocodingService(registry)

    location = geocoder.geocode(
        "Gateway of India, Mumbai"
    )

    print(location["latitude"])
    print(location["longitude"])
"""

from __future__ import annotations

import time
import threading
from typing import Any


class GeocodingError(RuntimeError):
    """Base exception for geocoding errors."""


class GeocodingNotFoundError(
    GeocodingError
):
    """Raised when no location can be found."""


class GeocodingService:
    """
    Provider-independent geocoding service.

    The service receives a MapsProvider registry instead of
    depending directly on Google Maps or Mapbox.
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

        self._forward_cache: dict[
            str,
            tuple[
                float,
                dict[str, Any],
            ],
        ] = {}

        self._reverse_cache: dict[
            tuple[float, float],
            tuple[
                float,
                dict[str, Any],
            ],
        ] = {}

        self._lock = threading.RLock()

    # ============================================================
    # FORWARD GEOCODING
    # ============================================================

    def geocode(
        self,
        address: str,
        *,
        provider: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert a textual address into coordinates.

        Args:
            address:
                Address, landmark, city, postal code, etc.

            provider:
                Optional provider name.

            fallback:
                Try fallback providers if the selected provider
                fails.

        Returns:
            Normalized location dictionary.
        """

        address = self._validate_address(
            address
        )

        cache_key = self._forward_cache_key(
            address,
            kwargs,
        )

        cached = self._get_forward_cache(
            cache_key
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

                result = maps_provider.geocode(
                    address,
                    **kwargs,
                )

                normalized = (
                    self._normalize_result(
                        result,
                        address=address,
                    )
                )

                self._validate_result(
                    normalized
                )

                self._set_forward_cache(
                    cache_key,
                    normalized,
                )

                return normalized

            except Exception as exc:

                errors.append(
                    f"{maps_provider.name}: {exc}"
                )

        raise GeocodingNotFoundError(
            self._build_error(
                "Unable to geocode address",
                address,
                errors,
            )
        )

    # ============================================================
    # REVERSE GEOCODING
    # ============================================================

    def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        *,
        provider: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert latitude/longitude into an address.
        """

        latitude, longitude = (
            self._validate_coordinates(
                latitude,
                longitude,
            )
        )

        cache_key = (
            round(latitude, 6),
            round(longitude, 6),
        )

        cached = self._get_reverse_cache(
            cache_key
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
                    maps_provider.reverse_geocode(
                        latitude,
                        longitude,
                        **kwargs,
                    )
                )

                normalized = (
                    self._normalize_result(
                        result,
                        latitude=latitude,
                        longitude=longitude,
                    )
                )

                self._validate_result(
                    normalized
                )

                self._set_reverse_cache(
                    cache_key,
                    normalized,
                )

                return normalized

            except Exception as exc:

                errors.append(
                    f"{maps_provider.name}: {exc}"
                )

        raise GeocodingNotFoundError(
            self._build_error(
                "Unable to reverse geocode coordinates",
                f"{latitude}, {longitude}",
                errors,
            )
        )

    # ============================================================
    # BATCH GEOCODING
    # ============================================================

    def geocode_many(
        self,
        addresses: list[str] | tuple[str, ...],
        *,
        provider: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Geocode multiple addresses.

        Failed addresses are represented by an error object
        instead of stopping the entire operation.
        """

        if not isinstance(
            addresses,
            (list, tuple),
        ):
            raise TypeError(
                "addresses must be a list or tuple."
            )

        results: list[
            dict[str, Any]
        ] = []

        for address in addresses:

            try:

                result = self.geocode(
                    address,
                    provider=provider,
                    fallback=fallback,
                    **kwargs,
                )

                results.append(
                    {
                        "success": True,
                        "query": address,
                        "result": result,
                    }
                )

            except Exception as exc:

                results.append(
                    {
                        "success": False,
                        "query": address,
                        "result": None,
                        "error": str(exc),
                    }
                )

        return results

    # ============================================================
    # PROVIDER HANDLING
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

            chain = [selected]

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
    # RESULT NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_result(
        result: Any,
        *,
        address: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, Any]:
        """
        Normalize provider-specific geocoding results.
        """

        if not isinstance(
            result,
            dict,
        ):
            raise GeocodingError(
                "Maps provider returned an invalid result."
            )

        normalized = dict(
            result
        )

        if address is not None:
            normalized.setdefault(
                "query",
                address,
            )

        if latitude is not None:
            normalized.setdefault(
                "latitude",
                latitude,
            )

        if longitude is not None:
            normalized.setdefault(
                "longitude",
                longitude,
            )

        normalized.setdefault(
            "provider",
            "unknown",
        )

        normalized.setdefault(
            "formatted_address",
            normalized.get(
                "address"
            ),
        )

        return normalized

    @staticmethod
    def _validate_result(
        result: dict[str, Any],
    ) -> None:
        """
        Ensure the normalized result contains coordinates.
        """

        latitude = result.get(
            "latitude"
        )

        longitude = result.get(
            "longitude"
        )

        if latitude is None:
            raise GeocodingError(
                "Geocoding result does not contain latitude."
            )

        if longitude is None:
            raise GeocodingError(
                "Geocoding result does not contain longitude."
            )

        GeocodingService._validate_coordinates(
            latitude,
            longitude,
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_address(
        address: str,
    ) -> str:
        """
        Validate a textual address.
        """

        if not isinstance(
            address,
            str,
        ):
            raise TypeError(
                "Address must be a string."
            )

        address = address.strip()

        if not address:
            raise ValueError(
                "Address cannot be empty."
            )

        if len(address) > 1000:
            raise ValueError(
                "Address is too long."
            )

        return address

    @staticmethod
    def _validate_coordinates(
        latitude: float,
        longitude: float,
    ) -> tuple[float, float]:
        """
        Validate latitude and longitude.
        """

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

    # ============================================================
    # CACHE
    # ============================================================

    def _forward_cache_key(
        self,
        address: str,
        kwargs: dict[str, Any],
    ) -> str:
        """
        Generate a stable forward-geocoding cache key.
        """

        provider = kwargs.get(
            "provider",
            "",
        )

        extras = tuple(
            sorted(
                (
                    str(key),
                    str(value),
                )
                for key, value
                in kwargs.items()
                if key != "provider"
            )
        )

        return (
            f"{address.lower().strip()}"
            f"|{provider}|{extras}"
        )

    def _get_forward_cache(
        self,
        key: str,
    ) -> dict[str, Any] | None:
        """

        Retrieve a valid forward-geocoding cache entry.
        """

        if not self.cache_enabled:
            return None

        with self._lock:

            item = self._forward_cache.get(
                key
            )

            if item is None:
                return None

            timestamp, result = item

            if (
                time.monotonic()
                - timestamp
                > self.cache_ttl
            ):

                self._forward_cache.pop(
                    key,
                    None,
                )

                return None

            return dict(
                result
            )

    def _set_forward_cache(
        self,
        key: str,
        result: dict[str, Any],
    ) -> None:
        """
        Store a forward-geocoding result.
        """

        if not self.cache_enabled:
            return

        with self._lock:

            self._trim_cache(
                self._forward_cache
            )

            self._forward_cache[
                key
            ] = (
                time.monotonic(),
                dict(result),
            )

    def _get_reverse_cache(
        self,
        key: tuple[float, float],
    ) -> dict[str, Any] | None:
        """
        Retrieve a valid reverse-geocoding cache entry.
        """

        if not self.cache_enabled:
            return None

        with self._lock:

            item = self._reverse_cache.get(
                key
            )

            if item is None:
                return None

            timestamp, result = item

            if (
                time.monotonic()
                - timestamp
                > self.cache_ttl
            ):

                self._reverse_cache.pop(
                    key,
                    None,
                )

                return None

            return dict(
                result
            )

    def _set_reverse_cache(
        self,
        key: tuple[float, float],
        result: dict[str, Any],
    ) -> None:
        """
        Store a reverse-geocoding result.
        """

        if not self.cache_enabled:
            return

        with self._lock:

            self._trim_cache(
                self._reverse_cache
            )

            self._reverse_cache[
                key
            ] = (
                time.monotonic(),
                dict(result),
            )

    def clear_cache(self) -> None:
        """
        Clear all cached geocoding results.
        """

        with self._lock:

            self._forward_cache.clear()
            self._reverse_cache.clear()

    def _trim_cache(
        self,
        cache: dict[Any, Any],
    ) -> None:
        """
        Keep cache below configured maximum size.
        """

        while len(cache) >= self.max_cache_size:

            oldest_key = next(
                iter(cache),
                None,
            )

            if oldest_key is None:
                break

            cache.pop(
                oldest_key,
                None,
            )

    # ============================================================
    # ERROR HANDLING
    # ============================================================

    @staticmethod
    def _build_error(
        operation: str,
        target: str,
        errors: list[str],
    ) -> str:
        """
        Build a readable multi-provider error.
        """

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
    ) -> "GeocodingService":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """
        Clear service cache when leaving context.
        """

        self.clear_cache()


__all__ = [
    "GeocodingService",
    "GeocodingError",
    "GeocodingNotFoundError",
]


