"""
RENIX Mapbox Maps Provider
==========================

Mapbox implementation of the RENIX MapsProvider interface.

Supported operations:
    - Geocoding
    - Reverse geocoding
    - Routing
    - Places search
    - Place details
    - Autocomplete

Environment variable:

    MAPBOX_ACCESS_TOKEN

The token may also be supplied directly to the constructor.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from .maps_provider import MapsProvider


# ============================================================
# API ENDPOINTS
# ============================================================

GEOCODING_URL = (
    "https://api.mapbox.com/search/geocode/v6/forward"
)

REVERSE_GEOCODING_URL = (
    "https://api.mapbox.com/search/geocode/v6/reverse"
)

ROUTING_URL = (
    "https://api.mapbox.com/directions/v5/mapbox"
)

SEARCH_URL = (
    "https://api.mapbox.com/search/searchbox/v1/forward"
)

RETRIEVE_URL = (
    "https://api.mapbox.com/search/searchbox/v1/retrieve"
)

SUGGEST_URL = (
    "https://api.mapbox.com/search/searchbox/v1/suggest"
)


# ============================================================
# PROVIDER
# ============================================================


class MapboxProvider(MapsProvider):
    """
    Mapbox provider for RENIX.

    All provider-specific responses are normalized before being
    returned to the rest of the RENIX system.
    """

    name = "mapbox"
    display_name = "Mapbox"
    version = "1.0.0"

    def __init__(
        self,
        *,
        access_token: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
        enabled: bool = True,
        language: str = "en",
        country: str | None = None,
        session: requests.Session | None = None,
        **config: Any,
    ) -> None:

        token = (
            access_token
            or api_key
            or os.getenv("MAPBOX_ACCESS_TOKEN")
            or os.getenv("MAPBOX_TOKEN")
        )

        super().__init__(
            api_key=token,
            timeout=timeout,
            enabled=enabled,
            **config,
        )

        self.access_token = token

        self.language = (
            language.strip()
            if language
            else "en"
        )

        self.country = (
            country.strip()
            if country
            else None
        )

        self.session = (
            session
            or requests.Session()
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def is_configured(self) -> bool:
        """Return whether a Mapbox access token exists."""

        return bool(
            self.access_token
        )

    def health_check(self) -> bool:
        """
        Perform a lightweight Mapbox API check.
        """

        if not self.enabled:
            return False

        if not self.is_configured():
            return False

        try:

            data = self._request(
                GEOCODING_URL,
                {
                    "q": "Mumbai",
                    "limit": 1,
                },
            )

            return bool(
                isinstance(
                    data,
                    dict,
                )
            )

        except Exception:
            return False

    # ========================================================
    # GEOCODING
    # ========================================================

    def geocode(
        self,
        address: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert an address/place name into coordinates.
        """

        address = self.validate_location(
            address
        )

        params = {
            "q": address,
            "limit": kwargs.pop(
                "limit",
                1,
            ),
            **kwargs,
        }

        data = self._request(
            GEOCODING_URL,
            params,
        )

        features = data.get(
            "features",
            [],
        )

        if not features:
            raise LookupError(
                f"No location found for '{address}'."
            )

        return self.normalize_location_result(
            self._normalize_feature(
                features[0]
            )
        )

    # ========================================================
    # REVERSE GEOCODING
    # ========================================================

    def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert coordinates into a human-readable location.
        """

        latitude, longitude = (
            self.validate_coordinates(
                latitude,
                longitude,
            )
        )

        params = {
            "longitude": longitude,
            "latitude": latitude,
            "limit": kwargs.pop(
                "limit",
                1,
            ),
            **kwargs,
        }

        data = self._request(
            REVERSE_GEOCODING_URL,
            params,
        )

        features = data.get(
            "features",
            [],
        )

        if not features:
            raise LookupError(
                "No address found for the supplied coordinates."
            )

        result = self._normalize_feature(
            features[0]
        )

        result[
            "latitude"
        ] = latitude

        result[
            "longitude"
        ] = longitude

        return self.normalize_location_result(
            result
        )

    # ========================================================
    # ROUTING
    # ========================================================

    def route(
        self,
        origin: str | tuple[float, float],
        destination: str | tuple[float, float],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Calculate a route between two coordinates or locations.

        Mapbox Directions works with coordinate pairs, so textual
        locations are geocoded first.
        """

        origin_coordinates = (
            self._resolve_coordinates(
                origin
            )
        )

        destination_coordinates = (
            self._resolve_coordinates(
                destination
            )
        )

        profile = kwargs.pop(
            "profile",
            kwargs.pop(
                "mode",
                "driving",
            ),
        )

        profile = self._normalize_profile(
            profile
        )

        coordinates = (
            f"{origin_coordinates[1]},"
            f"{origin_coordinates[0]};"
            f"{destination_coordinates[1]},"
            f"{destination_coordinates[0]}"
        )

        url = (
            f"{ROUTING_URL}/{profile}/"
            f"{coordinates}"
        )

        params = {
            "overview": kwargs.pop(
                "overview",
                "full",
            ),
            "steps": kwargs.pop(
                "steps",
                "false",
            ),
            "geometries": kwargs.pop(
                "geometries",
                "geojson",
            ),
            **kwargs,
        }

        data = self._request(
            url,
            params,
        )

        routes = data.get(
            "routes",
            [],
        )

        if not routes:
            raise LookupError(
                "No route was found between the supplied locations."
            )

        route = routes[0]

        distance_meters = float(
            route.get(
                "distance",
                0.0,
            )
        )

        duration_seconds = float(
            route.get(
                "duration",
                0.0,
            )
        )

        return self.normalize_route_result(
            {
                "origin": origin,
                "destination": destination,
                "distance_meters": distance_meters,
                "duration_seconds": duration_seconds,
                "distance_text": self._format_distance(
                    distance_meters
                ),
                "duration_text": self._format_duration(
                    duration_seconds
                ),
                "geometry": route.get(
                    "geometry"
                ),
                "legs": route.get(
                    "legs",
                    [],
                ),
                "weight_name": route.get(
                    "weight_name"
                ),
                "weight": route.get(
                    "weight"
                ),
                "provider": self.name,
            }
        )

    # ========================================================
    # PLACES SEARCH
    # ========================================================

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
        Search for places using Mapbox Searchbox.
        """

        query = self.validate_location(
            query
        )

        params: dict[str, Any] = {
            "q": query,
            "limit": kwargs.pop(
                "limit",
                10,
            ),
            **kwargs,
        }

        if (
            latitude is not None
            or longitude is not None
        ):

            if (
                latitude is None
                or longitude is None
            ):
                raise ValueError(
                    "Both latitude and longitude "
                    "must be supplied together."
                )

            latitude, longitude = (
                self.validate_coordinates(
                    latitude,
                    longitude,
                )
            )

            params[
                "proximity"
            ] = (
                f"{longitude},{latitude}"
            )

        if radius is not None:
            # Searchbox does not expose the same radius behavior
            # as Google's Places API. Keep it as a RENIX-level
            # parameter only when supported by the API.
            params[
                "radius"
            ] = self.validate_radius(
                radius
            )

        data = self._request(
            SEARCH_URL,
            params,
        )

        features = data.get(
            "features",
            [],
        )

        return [
            self.normalize_place_result(
                self._normalize_place_feature(
                    feature
                )
            )
            for feature in features
        ]

    # ========================================================
    # PLACE DETAILS
    # ========================================================

    def get_place_details(
        self,
        place_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Retrieve detailed information about a Mapbox place.

        Mapbox Searchbox retrieve requests normally require a
        session token generated during suggestions. If a session
        token is supplied, it is forwarded to the API.
        """

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

        session_token = kwargs.pop(
            "session_token",
            None,
        )

        if not session_token:
            raise ValueError(
                "session_token is required by Mapbox "
                "Searchbox retrieve."
            )

        url = (
            f"{RETRIEVE_URL}/"
            f"{place_id}"
        )

        params = {
            "session_token": session_token,
            **kwargs,
        }

        data = self._request(
            url,
            params,
        )

        features = data.get(
            "features",
            [],
        )

        if not features:
            raise LookupError(
                f"No place found for ID '{place_id}'."
            )

        return self.normalize_place_result(
            self._normalize_place_feature(
                features[0]
            )
        )

    # ========================================================
    # AUTOCOMPLETE
    # ========================================================

    def autocomplete(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Return Mapbox Searchbox suggestions.

        A session token should be supplied by the caller when
        using autocomplete followed by place retrieval.
        """

        query = self.validate_location(
            query
        )

        params = {
            "q": query,
            "limit": kwargs.pop(
                "limit",
                10,
            ),
            **kwargs,
        }

        data = self._request(
            SUGGEST_URL,
            params,
        )

        suggestions = data.get(
            "suggestions",
            [],
        )

        results: list[
            dict[str, Any]
        ] = []

        for suggestion in suggestions:

            results.append(
                {
                    "provider": self.name,
                    "mapbox_id": suggestion.get(
                        "mapbox_id"
                    ),
                    "name": suggestion.get(
                        "name"
                    ),
                    "name_preferred": suggestion.get(
                        "name_preferred"
                    ),
                    "full_address": suggestion.get(
                        "full_address"
                    ),
                    "place_formatted": suggestion.get(
                        "place_formatted"
                    ),
                    "feature_type": suggestion.get(
                        "feature_type"
                    ),
                    "address": suggestion.get(
                        "address"
                    ),
                    "context": suggestion.get(
                        "context"
                    ),
                }
            )

        return results

    # ========================================================
    # HTTP
    # ========================================================

    def _request(
        self,
        url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform an authenticated Mapbox API request.
        """

        self.ensure_enabled()
        self.ensure_configured()

        request_params = dict(
            params
        )

        request_params[
            "access_token"
        ] = self.access_token

        request_params.setdefault(
            "language",
            self.language,
        )

        if self.country:
            request_params.setdefault(
                "country",
                self.country,
            )

        try:

            response = self.session.get(
                url,
                params=request_params,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            if not isinstance(
                data,
                dict,
            ):
                raise RuntimeError(
                    "Mapbox returned an unexpected response."
                )

            return data

        except requests.HTTPError as exc:

            message = (
                "Mapbox API request failed."
            )

            try:
                error_data = (
                    response.json()
                )

                if isinstance(
                    error_data,
                    dict,
                ):

                    message = (
                        error_data.get(
                            "message"
                        )
                        or error_data.get(
                            "error"
                        )
                        or message
                    )

            except Exception:
                pass

            raise RuntimeError(
                message
            ) from exc

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Mapbox network request failed: {exc}"
            ) from exc

        except ValueError as exc:

            raise RuntimeError(
                "Mapbox returned invalid JSON."
            ) from exc

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_feature(
        self,
        feature: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize a Mapbox geocoding feature.
        """

        geometry = feature.get(
            "geometry",
            {},
        )

        coordinates = geometry.get(
            "coordinates",
            [],
        )

        longitude = None
        latitude = None

        if (
            isinstance(
                coordinates,
                (list, tuple),
            )
            and len(coordinates) >= 2
        ):
            longitude = coordinates[0]
            latitude = coordinates[1]

        properties = feature.get(
            "properties",
            {},
        )

        context = feature.get(
            "context",
            [],
        )

        context_data = (
            self._extract_context(
                context
            )
        )

        return {
            "id": feature.get(
                "id"
            ),
            "mapbox_id": properties.get(
                "mapbox_id"
            ),
            "address": (
                properties.get(
                    "full_address"
                )
                or properties.get(
                    "place_formatted"
                )
                or feature.get(
                    "place_name"
                )
            ),
            "formatted_address": (
                properties.get(
                    "full_address"
                )
                or feature.get(
                    "place_name"
                )
            ),
            "name": (
                properties.get(
                    "name"
                )
                or feature.get(
                    "text"
                )
            ),
            "latitude": latitude,
            "longitude": longitude,
            "place_id": feature.get(
                "id"
            ),
            "feature_type": (
                properties.get(
                    "feature_type"
                )
                or feature.get(
                    "place_type",
                    [None],
                )[0]
            ),
            **context_data,
        }

    def _normalize_place_feature(
        self,
        feature: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize a Mapbox Searchbox place feature.
        """

        result = self._normalize_feature(
            feature
        )

        properties = feature.get(
            "properties",
            {},
        )

        result.update(
            {
                "name_preferred": properties.get(
                    "name_preferred"
                ),
                "coordinates": (
                    properties.get(
                        "coordinates"
                    )
                ),
                "category": (
                    properties.get(
                        "category"
                    )
                ),
                "maki": properties.get(
                    "maki"
                ),
                "brand": properties.get(
                    "brand"
                ),
                "website": properties.get(
                    "website"
                ),
                "phone": properties.get(
                    "phone"
                ),
            }
        )

        return result

    @staticmethod
    def _extract_context(
        context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Extract common geographic context from Mapbox results.
        """

        result: dict[str, Any] = {}

        for item in context:

            identifier = str(
                item.get(
                    "id",
                    ""
                )
            )

            text = (
                item.get(
                    "text"
                )
                or item.get(
                    "name"
                )
            )

            if not text:
                continue

            if identifier.startswith(
                "country."
            ):
                result[
                    "country"
                ] = text

            elif identifier.startswith(
                "region."
            ):
                result[
                    "state"
                ] = text

            elif identifier.startswith(
                "district."
            ):
                result[
                    "district"
                ] = text

            elif identifier.startswith(
                "postcode."
            ):
                result[
                    "postal_code"
                ] = text

            elif identifier.startswith(
                "place."
            ):
                result[
                    "city"
                ] = text

            elif identifier.startswith(
                "locality."
            ):
                result[
                    "locality"
                ] = text

            elif identifier.startswith(
                "neighborhood."
            ):
                result[
                    "neighborhood"
                ] = text

        return result

    # ========================================================
    # LOCATION HELPERS
    # ========================================================

    def _resolve_coordinates(
        self,
        location: str | tuple[float, float],
    ) -> tuple[float, float]:
        """
        Convert a RENIX location input into
        (latitude, longitude).
        """

        if isinstance(
            location,
            tuple,
        ):

            if len(location) != 2:
                raise ValueError(
                    "Coordinate tuple must contain "
                    "latitude and longitude."
                )

            return self.validate_coordinates(
                location[0],
                location[1],
            )

        location = self.validate_location(
            location
        )

        result = self.geocode(
            location
        )

        return self.validate_coordinates(
            result["latitude"],
            result["longitude"],
        )

    @staticmethod
    def _normalize_profile(
        profile: str,
    ) -> str:
        """
        Normalize RENIX routing modes to Mapbox profiles.
        """

        profile = (
            str(profile)
            .strip()
            .lower()
        )

        aliases = {
            "car": "driving",
            "drive": "driving",
            "driving-traffic": "driving-traffic",
            "bike": "cycling",
            "bicycle": "cycling",
            "cycle": "cycling",
            "walk": "walking",
            "foot": "walking",
            "pedestrian": "walking",
        }

        normalized = aliases.get(
            profile,
            profile,
        )

        allowed = {
            "driving",
            "driving-traffic",
            "cycling",
            "walking",
        }

        if normalized not in allowed:
            raise ValueError(
                "Unsupported Mapbox routing profile: "
                f"{profile}"
            )

        return normalized

    # ========================================================
    # FORMATTING
    # ========================================================

    @staticmethod
    def _format_distance(
        meters: int | float,
    ) -> str:
        """Format meters into a human-readable distance."""

        meters = float(
            meters
        )

        if meters < 1000:
            return (
                f"{meters:.0f} m"
            )

        return (
            f"{meters / 1000:.1f} km"
        )

    @staticmethod
    def _format_duration(
        seconds: int | float,
    ) -> str:
        """Format seconds into a human-readable duration."""

        total_seconds = int(
            max(
                0,
                seconds,
            )
        )

        minutes, seconds = divmod(
            total_seconds,
            60,
        )

        hours, minutes = divmod(
            minutes,
            60,
        )

        if hours:

            if minutes:
                return (
                    f"{hours} hr "
                    f"{minutes} min"
                )

            return (
                f"{hours} hr"
            )

        if minutes:
            return (
                f"{minutes} min"
            )

        return (
            f"{seconds} sec"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self) -> None:
        """Close the HTTP session."""

        self.session.close()

    def __enter__(
        self,
    ) -> "MapboxProvider":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


__all__ = [
    "MapboxProvider",
]


