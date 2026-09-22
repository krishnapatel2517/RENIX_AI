"""
RENIX OpenWeather Provider
==========================

OpenWeather implementation for RENIX's weather provider
interface.

Environment variables:

    OPENWEATHER_API_KEY
    OPENWEATHER_BASE_URL        optional
    RENIX_WEATHER_TIMEOUT       optional

The provider uses Python's standard-library HTTP client so
RENIX does not need another HTTP dependency just for this
integration.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from typing import Any

from . import WeatherProvider


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_BASE_URL = (
    "https://api.openweathermap.org"
)

DEFAULT_TIMEOUT = 15.0

DEFAULT_UNITS = "metric"


# ============================================================
# OPENWEATHER PROVIDER
# ============================================================


class OpenWeatherProvider(WeatherProvider):
    """
    OpenWeather implementation of RENIX WeatherProvider.

    Example:

        provider = OpenWeatherProvider(
            api_key="YOUR_API_KEY"
        )

        weather = provider.get_current_weather(
            "Mumbai"
        )

        print(weather["temperature"])
    """

    name = "openweather"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        units: str = DEFAULT_UNITS,
        language: str = "en",
        **kwargs: Any,
    ) -> None:

        resolved_key = (
            api_key
            or os.getenv(
                "OPENWEATHER_API_KEY"
            )
        )

        resolved_base_url = (
            base_url
            or os.getenv(
                "OPENWEATHER_BASE_URL"
            )
            or DEFAULT_BASE_URL
        )

        environment_timeout = os.getenv(
            "RENIX_WEATHER_TIMEOUT"
        )

        if timeout is None and environment_timeout:
            try:
                timeout = float(
                    environment_timeout
                )
            except ValueError:
                timeout = DEFAULT_TIMEOUT

        if timeout is None:
            timeout = DEFAULT_TIMEOUT

        super().__init__(
            api_key=resolved_key,
            base_url=resolved_base_url,
            timeout=timeout,
            units=units,
            language=language,
            **kwargs,
        )

        self.base_url = (
            resolved_base_url.rstrip("/")
        )

        self.timeout = float(
            timeout
        )

        self.units = units

        self.language = language

    # ========================================================
    # URL / HTTP
    # ========================================================

    def _build_url(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> str:
        """
        Build a safe OpenWeather API URL.
        """

        if not self.api_key:
            raise ValueError(
                "OPENWEATHER_API_KEY is not configured."
            )

        query = dict(params)

        query["appid"] = self.api_key

        encoded = urllib.parse.urlencode(
            query,
            doseq=True,
        )

        return (
            f"{self.base_url}"
            f"/data/2.5/"
            f"{endpoint.lstrip('/')}"
            f"?{encoded}"
        )

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform an HTTP GET request and decode JSON.
        """

        url = self._build_url(
            endpoint,
            params,
        )

        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": "RENIX/1.0",
            },
        )

        started = time.perf_counter()

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                raw = response.read()

                status_code = getattr(
                    response,
                    "status",
                    200,
                )

            if status_code < 200 or status_code >= 300:
                raise RuntimeError(
                    f"OpenWeather returned HTTP "
                    f"{status_code}."
                )

            try:
                data = json.loads(
                    raw.decode("utf-8")
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise RuntimeError(
                    "OpenWeather returned invalid "
                    "JSON data."
                ) from exc

            if not isinstance(
                data,
                dict,
            ):
                raise RuntimeError(
                    "OpenWeather returned an unexpected "
                    "response format."
                )

            return data

        except urllib.error.HTTPError as exc:
            error_body = ""

            try:
                error_body = (
                    exc.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )
            except Exception:
                pass

            message = self._extract_api_error(
                error_body
            )

            raise RuntimeError(
                f"OpenWeather HTTP {exc.code}: "
                f"{message}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Unable to connect to OpenWeather: "
                f"{exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise TimeoutError(
                "OpenWeather request timed out."
            ) from exc

        finally:
            # Stored for diagnostics without exposing the API
            # key or request URL.
            self._last_request_duration_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

    @staticmethod
    def _extract_api_error(
        body: str,
    ) -> str:
        """
        Extract a useful OpenWeather API error message.
        """

        if not body:
            return "Unknown API error."

        try:
            payload = json.loads(
                body
            )

            if isinstance(
                payload,
                dict,
            ):
                message = payload.get(
                    "message"
                )

                if message:
                    return str(
                        message
                    )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            pass

        return body[:500]

    # ========================================================
    # LOCATION
    # ========================================================

    def _location_params(
        self,
        location: str,
    ) -> dict[str, Any]:
        """
        Convert a location string into API parameters.

        OpenWeather accepts city names, city/country combinations,
        postal codes, and other supported location formats.
        """

        if not location or not location.strip():
            raise ValueError(
                "Location cannot be empty."
            )

        return {
            "q": location.strip(),
            "units": self.units,
            "lang": self.language,
        }

    # ========================================================
    # CURRENT WEATHER
    # ========================================================

    def get_current_weather(
        self,
        location: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get current weather for a location.

        Returns a normalized RENIX weather dictionary.
        """

        params = self._location_params(
            location
        )

        # Allow per-request unit/language overrides.
        if kwargs.get("units"):
            params["units"] = kwargs[
                "units"
            ]

        if kwargs.get("language"):
            params["lang"] = kwargs[
                "language"
            ]

        data = self._request(
            "weather",
            params,
        )

        return self._normalize_current_weather(
            data
        )

    # ========================================================
    # FORECAST
    # ========================================================

    def get_forecast(
        self,
        location: str,
        *,
        days: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Get forecast information.

        OpenWeather's standard forecast endpoint provides
        3-hour forecast intervals. RENIX returns those
        intervals as normalized forecast records.

        `days` is limited to the supported forecast horizon.
        """

        try:
            days = int(days)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "days must be an integer."
            ) from exc

        if days < 1:
            raise ValueError(
                "days must be at least 1."
            )

        # Standard OpenWeather forecast provides up to
        # approximately five days.
        days = min(
            days,
            5,
        )

        params = self._location_params(
            location
        )

        if kwargs.get("units"):
            params["units"] = kwargs[
                "units"
            ]

        if kwargs.get("language"):
            params["lang"] = kwargs[
                "language"
            ]

        data = self._request(
            "forecast",
            params,
        )

        forecast_items = data.get(
            "list",
            [],
        )

        if not isinstance(
            forecast_items,
            list,
        ):
            return []

        result: list[
            dict[str, Any]
        ] = []

        cutoff = (
            days * 24 * 60 * 60
        )

        first_timestamp: int | None = None

        for item in forecast_items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            timestamp = item.get(
                "dt"
            )

            if isinstance(
                timestamp,
                (int, float),
            ):

                if first_timestamp is None:
                    first_timestamp = int(
                        timestamp
                    )

                if (
                    int(timestamp)
                    - first_timestamp
                    > cutoff
                ):
                    continue

            result.append(
                self._normalize_forecast_item(
                    item
                )
            )

        return result

    # ========================================================
    # LOCATION SEARCH
    # ========================================================

    def search_location(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search locations using OpenWeather's Geocoding API.
        """

        if not query or not query.strip():
            raise ValueError(
                "Location query cannot be empty."
            )

        if not self.api_key:
            raise ValueError(
                "OPENWEATHER_API_KEY is not configured."
            )

        params = {
            "q": query.strip(),
            "limit": int(
                kwargs.get(
                    "limit",
                    5,
                )
            ),
        }
        query_string = urllib.parse.urlencode(
            {
                **params,
                "appid": self.api_key,
            }
        )

        url = (
            f"{self.base_url}"
            f"/geo/1.0/direct?"
            f"{query_string}"
        )

        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": "RENIX/1.0",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                payload = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"OpenWeather location search "
                f"failed with HTTP {exc.code}."
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"OpenWeather location search "
                f"connection failed: {exc.reason}"
            ) from exc

        if not isinstance(
            payload,
            list,
        ):
            return []

        locations: list[
            dict[str, Any]
        ] = []

        for item in payload:
            if not isinstance(
                item,
                dict,
            ):
                continue

            locations.append(
                {
                    "name": item.get(
                        "name"
                    ),
                    "latitude": item.get(
                        "lat"
                    ),
                    "longitude": item.get(
                        "lon"
                    ),
                    "country": item.get(
                        "country"
                    ),
                    "state": item.get(
                        "state"
                    ),
                    "provider": self.name,
                }
            )

        return locations

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_current_weather(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert OpenWeather's response into RENIX's
        provider-neutral format.
        """

        main = data.get(
            "main",
            {},
        )

        wind = data.get(
            "wind",
            {},
        )

        clouds = data.get(
            "clouds",
            {},
        )

        sys = data.get(
            "sys",
            {},
        )

        weather = data.get(
            "weather",
            [],
        )

        primary_weather = (
            weather[0]
            if weather
            and isinstance(
                weather[0],
                dict,
            )
            else {}
        )

        coordinates = data.get(
            "coord",
            {},
        )

        return {
            "provider": self.name,
            "location": {
                "name": data.get(
                    "name"
                ),
                "country": sys.get(
                    "country"
                ),
                "latitude": coordinates.get(
                    "lat"
                ),
                "longitude": coordinates.get(
                    "lon"
                ),
            },
            "temperature": main.get(
                "temp"
            ),
            "feels_like": main.get(
                "feels_like"
            ),
            "temperature_min": main.get(
                "temp_min"
            ),
            "temperature_max": main.get(
                "temp_max"
            ),
            "pressure": main.get(
                "pressure"
            ),
            "humidity": main.get(
                "humidity"
            ),
            "visibility": data.get(
                "visibility"
            ),
            "weather": {
                "id": primary_weather.get(
                    "id"
                ),
                "main": primary_weather.get(
                    "main"
                ),
                "description": primary_weather.get(
                    "description"
                ),
                "icon": primary_weather.get(
                    "icon"
                ),
            },
            "wind": {
                "speed": wind.get(
                    "speed"
                ),
                "direction": wind.get(
                    "deg"
                ),
                "gust": wind.get(
                    "gust"
                ),
            },
            "clouds": clouds.get(
                "all"
            ),
            "sunrise": sys.get(
                "sunrise"
            ),
            "sunset": sys.get(
                "sunset"
            ),
            "timestamp": data.get(
                "dt"
            ),
            "timezone_offset": data.get(
                "timezone"
            ),
            "raw": data,
        }

    def _normalize_forecast_item(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize a single forecast interval."""

        main = item.get(
            "main",
            {},
        )

        wind = item.get(
            "wind",
            {},
        )

        weather = item.get(
            "weather",
            [],
        )

        primary_weather = (
            weather[0]
            if weather
            and isinstance(
                weather[0],
                dict,
            )
            else {}
        )

        return {
            "provider": self.name,
            "timestamp": item.get(
                "dt"
            ),
            "datetime": item.get(
                "dt_txt"
            ),
            "temperature": main.get(
                "temp"
            ),
            "feels_like": main.get(
                "feels_like"
            ),
            "temperature_min": main.get(
                "temp_min"
            ),
            "temperature_max": main.get(
                "temp_max"
            ),
            "pressure": main.get(
                "pressure"
            ),
            "humidity": main.get(
                "humidity"
            ),
            "weather": {
                "id": primary_weather.get(
                    "id"
                ),
                "main": primary_weather.get(
                    "main"
                ),
                "description": primary_weather.get(
                    "description"
                ),
                "icon": primary_weather.get(
                    "icon"
                ),
            },
            "wind": {
                "speed": wind.get(
                    "speed"
                ),
                "direction": wind.get(
                    "deg"
                ),
                "gust": wind.get(
                    "gust"
                ),
            },
            "probability_of_precipitation": item.get(
                "pop"
            ),
            "clouds": (
                item.get(
                    "clouds",
                    {},
                ).get(
                    "all"
                )
                if isinstance(
                    item.get(
                        "clouds",
                        {},
                    ),
                    dict,
                )
                else None
            ),
            "rain": item.get(
                "rain"
            ),
            "snow": item.get(
                "snow"
            ),
            "raw": item,
        }

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:
        """
        Check whether OpenWeather credentials and API access
        are functioning.
        """

        if not self.api_key:
            return False

        try:
            self._request(
                "weather",
                {
                    "q": "Mumbai",
                    "units": self.units,
                    "lang": self.language,
                },
            )

            return True

        except Exception:
            return False


# ============================================================
# FACTORY
# ============================================================


def create_openweather_provider(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    units: str = DEFAULT_UNITS,
    language: str = "en",
) -> OpenWeatherProvider:
    """
    Create an OpenWeather provider using explicit values or
    environment variables.
    """

    return OpenWeatherProvider(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        units=units,
        language=language,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "OpenWeatherProvider",
    "create_openweather_provider",
]


