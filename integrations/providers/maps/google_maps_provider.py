"""
RENIX Google Maps Provider
==========================

Google Maps implementation of the RENIX MapsProvider interface.

Supported operations:
    - Geocoding
    - Reverse geocoding
    - Directions / routing
    - Places search
    - Place details
    - Distance matrix
    - Places autocomplete

The provider normalizes Google API responses into structures that
can be consumed by the rest of RENIX.

Required environment variable:

    GOOGLE_MAPS_API_KEY

The API key can also be passed directly to the constructor.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from .maps_provider import MapsProvider


# ============================================================
# API ENDPOINTS
# ============================================================

GEOCODE_URL = (
    "https://maps.googleapis.com/maps/api/geocode/json"
)

DIRECTIONS_URL = (
    "https://maps.googleapis.com/maps/api/directions/json"
)

PLACES_TEXT_SEARCH_URL = (
    "https://maps.googleapis.com/maps/api/place/textsearch/json"
)

PLACE_DETAILS_URL = (
    "https://maps.googleapis.com/maps/api/place/details/json"
)

DISTANCE_MATRIX_URL = (
    "https://maps.googleapis.com/maps/api/distancematrix/json"
)

AUTOCOMPLETE_URL = (
    "https://maps.googleapis.com/maps/api/place/autocomplete/json"
)


# ============================================================
# PROVIDER
# ============================================================


class GoogleMapsProvider(MapsProvider):
    """
    Google Maps provider for RENIX.

    This class communicates with Google's Maps Platform APIs and
    converts their responses into RENIX-friendly dictionaries.
    """

    name = "google_maps"
    display_name = "Google Maps"
    version = "1.0.0"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 10.0,
        enabled: bool = True,
        language: str = "en",
        region: str | None = None,
        session: requests.Session | None = None,
        **config: Any,
    ) -> None:

        api_key = (
            api_key
            or os.getenv("GOOGLE_MAPS_API_KEY")
            or os.getenv("GOOGLE_MAPS_KEY")
        )

        super().__init__(
            api_key=api_key,
            timeout=timeout,
            enabled=enabled,
            **config,
        )

        self.language = (
            language.strip()
            if language
            else "en"
        )

        self.region = (
            region.strip()
            if region
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
        """Return whether a Google Maps API key is available."""

        return bool(
            self.api_key
        )

    def health_check(self) -> bool:
        """
        Perform a lightweight API health check.

        A geocoding request is used because it is simple and
        requires no special place-specific parameters.
        """

        if not self.enabled:
            return False

        if not self.is_configured():
            return False

        try:

            response = self._request(
                GEOCODE_URL,
                {
                    "address": "Mumbai, India",
                },
            )

            return response.get(
                "status"
            ) in {
                "OK",
                "ZERO_RESULTS",
            }

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
        Convert an address or place name into coordinates.
        """

        address = self.validate_location(
            address
        )

        data = self._request(
            GEOCODE_URL,
            {
                "address": address,
                **kwargs,
            },
        )

        status = data.get(
            "status"
        )

        if status == "ZERO_RESULTS":
            raise LookupError(
                f"No location found for '{address}'."
            )

        self._ensure_success(
            data,
            "Geocoding",
        )

        results = data.get(
            "results",
            [],
        )

        if not results:
            raise LookupError(
                f"No location found for '{address}'."
            )

        result = results[0]

        return self.normalize_location_result(
            self._normalize_geocode_result(
                result
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
        Convert coordinates into a human-readable address.
        """

        latitude, longitude = (
            self.validate_coordinates(
                latitude,
                longitude,
            )
        )

        data = self._request(
            GEOCODE_URL,
            {
                "latlng": (
                    f"{latitude},{longitude}"
                ),
                **kwargs,
            },
        )

        status = data.get(
            "status"
        )

        if status == "ZERO_RESULTS":
            raise LookupError(
                "No address found for the supplied coordinates."
            )

        self._ensure_success(
            data,
            "Reverse geocoding",
        )

        results = data.get(
            "results",
            [],
        )

        if not results:
            raise LookupError(
                "No address found for the supplied coordinates."
            )

        result = results[0]

        normalized = (
            self._normalize_geocode_result(
                result
            )
        )

        normalized[
            "latitude"
        ] = latitude

        normalized[
            "longitude"
        ] = longitude

        return self.normalize_location_result(
            normalized
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
        Calculate a route between two locations.
        """

        origin_value = (
            self._normalize_location_input(
                origin
            )
        )

        destination_value = (
            self._normalize_location_input(
                destination
            )
        )

        params = {
            "origin": origin_value,
            "destination": destination_value,
            "mode": kwargs.pop(
                "mode",
                "driving",
            ),
            **kwargs,
        }

        data = self._request(
            DIRECTIONS_URL,
            params,
        )

        status = data.get(
            "status"
        )

        if status == "ZERO_RESULTS":
            raise LookupError(
                "No route was found between the supplied locations."
            )

        self._ensure_success(
            data,
            "Routing",
        )

        routes = data.get(
            "routes",
            [],
        )

        if not routes:
            raise LookupError(
                "No route was found."
            )

        route = routes[0]

        legs = route.get(
            "legs",
            [],
        )

        normalized_legs = [
            self._normalize_route_leg(
                leg
            )
            for leg in legs
        ]

        total_distance = sum(
            leg.get(
                "distance_meters",
                0,
            )
            for leg in normalized_legs
        )

        total_duration = sum(
            leg.get(
                "duration_seconds",
                0,
            )
            for leg in normalized_legs
        )

        return self.normalize_route_result(
            {
                "origin": origin,
                "destination": destination,
                "distance_meters": total_distance,
                "duration_seconds": total_duration,
                "distance_text": self._format_distance(
                    total_distance
                ),
                "duration_text": self._format_duration(
                    total_duration
                ),
                "polyline": (
                    route.get(
                        "overview_polyline",
                        {}
                    ).get(
                        "points"
                    )
                ),
                "summary": route.get(
                    "summary"
                ),
                "legs": normalized_legs,
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
        Search for places.

        Supports:
            - text queries
            - optional latitude/longitude
            - optional radius
        """

        query = self.validate_location(
            query
        )

        params: dict[str, Any] = {
            "query": query,
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
                    "must be provided together."
                )

            latitude, longitude = (
                self.validate_coordinates(
                    latitude,
                    longitude,
                )
            )

            params[
                "location"
            ] = f"{latitude},{longitude}"

        if radius is not None:
            params[
                "radius"
            ] = self.validate_radius(
                radius
            )

        data = self._request(
            PLACES_TEXT_SEARCH_URL,
            params,
        )

        if data.get(
            "status"
        ) == "ZERO_RESULTS":
            return []

        self._ensure_success(
            data,
            "Places search",
        )

        results = data.get(
            "results",
            [],
        )

        return [
            self.normalize_place_result(
                self._normalize_place_result(
                    result
                )
            )
            for result in results
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
        Retrieve detailed information about a Google place.
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

        fields = kwargs.pop(
            "fields",
            [
                "place_id",
                "name",
                "formatted_address",
                "geometry",
                "formatted_phone_number",
                "international_phone_number",
                "website",
                "rating",
                "user_ratings_total",
                "opening_hours",
                "types",
                "url",
            ],
        )

        params: dict[str, Any] = {
            "place_id": place_id,
            "fields": (
                ",".join(fields)
                if isinstance(
                    fields,
                    (list, tuple, set),
                )
                else fields
            ),
            **kwargs,
        }

        data = self._request(
            PLACE_DETAILS_URL,
            params,
        )

        if data.get(
            "status"
        ) == "ZERO_RESULTS":
            raise LookupError(
                f"No place found for ID '{place_id}'."
            )

        self._ensure_success(
            data,
            "Place details",
        )

        result = data.get(
            "result"
        )

        if not result:
            raise LookupError(
                f"No place found for ID '{place_id}'."
            )

        return self.normalize_place_result(
            self._normalize_place_result(
                result
            )
        )

    # ========================================================
    # DISTANCE MATRIX
    # ========================================================

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
        Calculate distances and travel times for multiple
        origins and destinations.
        """

        if not origins:
            raise ValueError(
                "At least one origin is required."
            )

        if not destinations:
            raise ValueError(
                "At least one destination is required."
            )

        normalized_origins = [
            self._normalize_location_input(
                origin
            )
            for origin in origins
        ]

        normalized_destinations = [
            self._normalize_location_input(
                destination
            )
            for destination in destinations
        ]

        params = {
            "origins": "|".join(
                normalized_origins
            ),
            "destinations": "|".join(
                normalized_destinations
            ),
            "mode": kwargs.pop(
                "mode",
                "driving",
            ),
            **kwargs,
        }

        data = self._request(
            DISTANCE_MATRIX_URL,
            params,
        )

        if data.get(
            "status"
        ) == "ZERO_RESULTS":
            return {
                "origins": origins,
                "destinations": destinations,
                "rows": [],
            }

        self._ensure_success(
            data,
            "Distance matrix",
        )

        rows = []

        for row in data.get(
            "rows",
            [],
        ):

            elements = []

            for element in row.get(
                "elements",
                [],
            ):

                normalized = {
                    "status": element.get(
                        "status"
                    ),
                }

                distance = element.get(
                    "distance"
                )

                duration = element.get(
                    "duration"
                )

                if distance:
                    normalized[
                        "distance_meters"
                    ] = distance.get(
                        "value"
                    )

                    normalized[
                        "distance_text"
                    ] = distance.get(
                        "text"
                    )

                if duration:
                    normalized[
                        "duration_seconds"
                    ] = duration.get(
                        "value"
                    )

                    normalized[
                        "duration_text"
                    ] = duration.get(
                        "text"
                    )

                elements.append(
                    normalized
                )

            rows.append(
                {
                    "elements": elements,
                }
            )

        return {
            "provider": self.name,
            "origins": origins,
            "destinations": destinations,
            "rows": rows,
        }

    # ========================================================
    # AUTOCOMPLETE
    # ========================================================

    def autocomplete(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Return Google Places autocomplete suggestions.
        """

        query = self.validate_location(
            query
        )

        params = {
            "input": query,
            **kwargs,
        }

        data = self._request(
            AUTOCOMPLETE_URL,
            params,
        )

        if data.get(
            "status"
        ) == "ZERO_RESULTS":
            return []

        self._ensure_success(
            data,
            "Places autocomplete",
        )

        predictions = data.get(
            "predictions",
            [],
        )

        results = []

        for prediction in predictions:

            structured = prediction.get(
                "structured_formatting",
                {},
            )

            results.append(
                {
                    "provider": self.name,
                    "place_id": prediction.get(
                        "place_id"
                    ),
                    "description": prediction.get(
                        "description"
                    ),
                    "main_text": structured.get(
                        "main_text"
                    ),
                    "secondary_text": structured.get(
                        "secondary_text"
                    ),
                    "types": prediction.get(
                        "types",
                        [],
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
        Perform a request against Google Maps.

        The API key, language and region are added automatically.
        """

        self.ensure_enabled()
        self.ensure_configured()

        request_params = dict(
            params
        )

        request_params[
            "key"
        ] = self.api_key

        request_params.setdefault(
            "language",
            self.language,
        )

        if self.region:
            request_params.setdefault(
                "region",
                self.region,
            )

        self.record_request(
            success=False
        )

        try:

            response = self.session.get(
                url,
                params=request_params,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            self._ensure_api_response_valid(
                data
            )

            self._mark_last_request_success()

            return data

        except requests.RequestException:
            raise

        except ValueError as exc:
            raise RuntimeError(
                "Google Maps returned invalid JSON."
            ) from exc

    def _mark_last_request_success(
        self,
    ) -> None:
        """
        Move one recorded request from failure to success.

        The base provider counters intentionally remain simple,
        so this helper adjusts the counters after a successful
        request.
        """

        self._success_count += 1

        if self._failure_count > 0:
            self._failure_count -= 1

    # ========================================================
    # RESPONSE VALIDATION
    # ========================================================

    @staticmethod
    def _ensure_api_response_valid(
        data: dict[str, Any],
    ) -> None:
        """Validate that a Google API response is a dictionary."""

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "Google Maps returned an unexpected response."
            )

    @staticmethod
    def _ensure_success(
        data: dict[str, Any],
        operation: str,
    ) -> None:
        """Raise an informative error for unsuccessful API calls."""

        status = data.get(
            "status"
        )

        if status in {
            None,
            "OK",
            "ZERO_RESULTS",
        }:
            return

        error_message = data.get(
            "error_message"
        )

        message = (
            f"{operation} failed: {status}"
        )

        if error_message:
            message += (
                f" - {error_message}"
            )

        raise RuntimeError(
            message
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_geocode_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize a Google geocoding result."""

        geometry = result.get(
            "geometry",
            {},
        )

        location = geometry.get(
            "location",
            {},
        )

        address_components = (
            result.get(
                "address_components",
                [],
            )
        )

        components = (
            self._extract_address_components(
                address_components
            )
        )

        return {
            "formatted_address": result.get(
                "formatted_address"
            ),
            "address": result.get(
                "formatted_address"
            ),
            "latitude": location.get(
                "lat"
            ),
            "longitude": location.get(
                "lng"
            ),
            "place_id": result.get(
                "place_id"
            ),
            "types": result.get(
                "types",
                [],
            ),
            **components,
        }

    def _normalize_place_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize a Google Places result."""

        geometry = result.get(
            "geometry",
            {},
        )

        location = geometry.get(
            "location",
            {},
        )

        opening_hours = result.get(
            "opening_hours"
        )

        normalized = {
            "place_id": result.get(
                "place_id"
            ),
            "name": result.get(
                "name"
            ),
            "address": result.get(
                "formatted_address"
            ),
            "formatted_address": result.get(
                "formatted_address"
            ),
            "latitude": location.get(
                "lat"
            ),
            "longitude": location.get(
                "lng"
            ),
            "rating": result.get(
                "rating"
            ),
            "user_ratings_total": result.get(
                "user_ratings_total"
            ),
            "types": result.get(
                "types",
                [],
            ),
            "phone": result.get(
                "formatted_phone_number"
            ),
            "international_phone": result.get(
                "international_phone_number"
            ),
            "website": result.get(
                "website"
            ),
            "url": result.get(
                "url"
            ),
        }

        if opening_hours:
            normalized[
                "opening_hours"
            ] = {
                "open_now": opening_hours.get(
                    "open_now"
                ),
                "weekday_text": opening_hours.get(
                    "weekday_text",
                    [],
                ),
            }

        return normalized

    def _normalize_route_leg(
        self,
        leg: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize one Google Directions route leg."""

        distance = leg.get(
            "distance",
            {},
        )

        duration = leg.get(
            "duration",
            {},
        )

        return {
            "start_address": leg.get(
                "start_address"
            ),
            "end_address": leg.get(
                "end_address"
            ),
            "start_location": leg.get(
                "start_location"
            ),
            "end_location": leg.get(
                "end_location"
            ),
            "distance_meters": distance.get(
                "value",
                0,
            ),
            "distance_text": distance.get(
                "text"
            ),
            "duration_seconds": duration.get(
                "value",
                0,
            ),
            "duration_text": duration.get(
                "text"
            ),
            "steps": leg.get(
                "steps",
                [],
            ),
        }

    @staticmethod
    def _extract_address_components(
        components: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract common address fields from Google components."""

        mapping = {
            "street_number": "street_number",
            "route": "route",
            "locality": "city",
            "postal_town": "city",
            "administrative_area_level_2": "district",
            "administrative_area_level_1": "state",
            "country": "country",
            "postal_code": "postal_code",
            "sublocality": "sublocality",
            "sublocality_level_1": "sublocality",
        }

        result: dict[str, Any] = {}

        for component in components:

            types = component.get(
                "types",
                [],
            )

            for component_type in types:

                target = mapping.get(
                    component_type
                )

                if target:
                    result[target] = (
                        component.get(
                            "long_name"
                        )
                    )

        return result

    # ========================================================
    # INPUT HELPERS
    # ========================================================

    @classmethod
    def _normalize_location_input(
        cls,
        location: str | tuple[float, float],
    ) -> str:
        """
        Normalize a location into the format expected by Google.
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

            latitude, longitude = (
                cls.validate_coordinates(
                    location[0],
                    location[1],
                )
            )

            return (
                f"{latitude},{longitude}"
            )

        return cls.validate_location(
            location
        )

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
            return f"{meters:.0f} m"

        return f"{meters / 1000:.1f} km"

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
        """Close the underlying HTTP session."""

        self.session.close()

    def __enter__(
        self,
    ) -> "GoogleMapsProvider":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


__all__ = [
    "GoogleMapsProvider",
]


