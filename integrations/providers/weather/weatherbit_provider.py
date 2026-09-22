"""
RENIX Weatherbit Provider
=========================

Weatherbit implementation for RENIX's weather provider layer.

Environment variables:

    WEATHERBIT_API_KEY
    WEATHERBIT_BASE_URL       optional
    RENIX_WEATHER_TIMEOUT     optional

The provider exposes a normalized interface so RENIX does not
need to depend on Weatherbit-specific response structures.
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

DEFAULT_BASE_URL = "https://api.weatherbit.io/v2.0"
DEFAULT_TIMEOUT = 15.0


# ============================================================
# PROVIDER
# ============================================================


class WeatherbitProvider(WeatherProvider):
    """
    Weatherbit implementation of WeatherProvider.

    Example:

        provider = WeatherbitProvider()

        current = provider.get_current_weather(
            "Mumbai"
        )

        forecast = provider.get_forecast(
            "Mumbai",
            days=5,
        )
    """

    name = "weatherbit"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        units: str = "M",
        language: str = "en",
        **kwargs: Any,
    ) -> None:

        resolved_key = (
            api_key
            or os.getenv("WEATHERBIT_API_KEY")
        )

        resolved_base_url = (
            base_url
            or os.getenv("WEATHERBIT_BASE_URL")
            or DEFAULT_BASE_URL
        )

        environment_timeout = os.getenv(
            "RENIX_WEATHER_TIMEOUT"
        )

        if timeout is None and environment_timeout:
            try:
                timeout = float(environment_timeout)
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

        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = float(timeout)
        self.units = units
        self.language = language

        self._last_request_duration_ms = 0.0

    # ========================================================
    # URL
    # ========================================================

    def _build_url(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> str:
        """Build a Weatherbit API URL."""

        if not self.api_key:
            raise ValueError(
                "WEATHERBIT_API_KEY is not configured."
            )

        query = dict(params)

        query["key"] = self.api_key
        query.setdefault("units", self.units)
        query.setdefault("lang", self.language)

        encoded = urllib.parse.urlencode(
            query,
            doseq=True,
        )

        return (
            f"{self.base_url}/"
            f"{endpoint.lstrip('/')}"
            f"?{encoded}"
        )

    # ========================================================
    # HTTP
    # ========================================================

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a Weatherbit GET request.

        Weatherbit responses are expected to be JSON objects.
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

            if not 200 <= status_code < 300:
                raise RuntimeError(
                    f"Weatherbit returned HTTP "
                    f"{status_code}."
                )

            try:
                payload = json.loads(
                    raw.decode("utf-8")
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise RuntimeError(
                    "Weatherbit returned invalid JSON."
                ) from exc

            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Weatherbit returned an unexpected "
                    "response format."
                )

            self._raise_for_api_error(payload)

            return payload

        except urllib.error.HTTPError as exc:
            body = ""

            try:
                body = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                pass

            message = self._extract_error_message(body)

            raise RuntimeError(
                f"Weatherbit HTTP {exc.code}: "
                f"{message}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Unable to connect to Weatherbit: "
                f"{exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise TimeoutError(
                "Weatherbit request timed out."
            ) from exc

        finally:
            self._last_request_duration_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    @staticmethod
    def _extract_error_message(
        body: str,
    ) -> str:
        """Extract a useful Weatherbit API error."""

        if not body:
            return "Unknown API error."

        try:
            payload = json.loads(body)

            if isinstance(payload, dict):
                for key in (
                    "error",
                    "message",
                    "error_message",
                ):
                    value = payload.get(key)

                    if value:
                        return str(value)

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            pass

        return body[:500]

    @staticmethod
    def _raise_for_api_error(
        payload: dict[str, Any],
    ) -> None:
        """Detect application-level Weatherbit errors."""

        error = payload.get("error")

        if not error:
            return

        if isinstance(error, dict):
            message = error.get(
                "message",
                error.get(
                    "error",
                    "Unknown Weatherbit error.",
                ),
            )

            raise RuntimeError(
                f"Weatherbit error: {message}"
            )

        raise RuntimeError(
            f"Weatherbit error: {error}"
        )

    # ========================================================
    # LOCATION PARAMETERS
    # ========================================================

    @staticmethod
    def _validate_location(
        location: str,
    ) -> str:
        """Validate and normalize a location string."""

        if not location or not location.strip():
            raise ValueError(
                "Location cannot be empty."
            )

        return location.strip()

    # ========================================================
    # CURRENT WEATHER
    # ========================================================

    def get_current_weather(
        self,
        location: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get current weather for a city/location.
        """

        location = self._validate_location(
            location
        )

        params: dict[str, Any] = {
            "city": location,
        }

        if kwargs.get("units"):
            params["units"] = kwargs["units"]

        if kwargs.get("language"):
            params["lang"] = kwargs["language"]

        data = self._request(
            "current",
            params,
        )

        records = data.get(
            "data",
            [],
        )

        if not isinstance(records, list):
            records = []

        current = (
            records[0]
            if records
            and isinstance(
                records[0],
                dict,
            )
            else {}
        )

        return self._normalize_current(
            current,
            data,
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
        Get daily Weatherbit forecast.

        Weatherbit's daily forecast endpoint can provide a
        multi-day forecast. The request is capped at 16 days
        to avoid accidental oversized requests.
        """

        location = self._validate_location(
            location
        )

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

        days = min(days, 16)

        params: dict[str, Any] = {
            "city": location,
            "days": days,
        }

        if kwargs.get("units"):
            params["units"] = kwargs["units"]

        if kwargs.get("language"):
            params["lang"] = kwargs["language"]

        data = self._request(
            "forecast/daily",
            params,
        )

        records = data.get(
            "data",
            [],
        )

        if not isinstance(records, list):
            return []

        result: list[dict[str, Any]] = []

        for item in records:
            if not isinstance(item, dict):
                continue

            result.append(
                self._normalize_forecast_day(
                    item
                )
            )

        return result

    # ========================================================
    # HOURLY FORECAST
    # ========================================================

    def get_hourly_forecast(
        self,
        location: str,
        *,
        hours: int = 24,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Get hourly forecast information.

        This method is an additional Weatherbit capability and
        does not alter the common WeatherProvider interface.
        """

        location = self._validate_location(
            location
        )

        try:
            hours = int(hours)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "hours must be an integer."
            ) from exc

        if hours < 1:
            raise ValueError(
                "hours must be at least 1."
            )

        hours = min(hours, 120)

        params: dict[str, Any] = {
            "city": location,
            "hours": hours,
        }

        if kwargs.get("units"):
            params["units"] = kwargs["units"]

        if kwargs.get("language"):
            params["lang"] = kwargs["language"]

        data = self._request(
            "forecast/hourly",
            params,
        )

        records = data.get(
            "data",
            [],
        )

        if not isinstance(records, list):
            return []

        result: list[dict[str, Any]] = []

        for item in records:
            if not isinstance(item, dict):
                continue

            result.append(
                self._normalize_hourly(
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
        Search for locations.

        Weatherbit's standard weather endpoints accept city
        names, so this method uses the current-weather endpoint
        to resolve the supplied location and returns normalized
        coordinates when available.
        """

        query = self._validate_location(
            query
        )

        data = self._request(
            "current",
            {
                "city": query,
            },
        )

        records = data.get(
            "data",
            [],
        )

        if not isinstance(records, list):
            return []

        result: list[
            dict[str, Any]
        ] = []

        for item in records:
            if not isinstance(item, dict):
                continue

            result.append(
                {
                    "name": item.get(
                        "city_name"
                    ),
                    "state": item.get(
                        "state_code"
                    ),
                    "country": item.get(
                        "country_code"
                    ),
                    "latitude": item.get(
                        "lat"
                    ),
                    "longitude": item.get(
                        "lon"
                    ),
                    "timezone": item.get(
                        "timezone"
                    ),
                    "provider": self.name,
                }
            )

        return result

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_weather(
        weather: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize Weatherbit condition information."""

        return {
            "code": weather.get(
                "code"
            ),
            "description": weather.get(
                "description"
            ),
            "icon": weather.get(
                "icon"
            ),
        }

    def _normalize_current(
        self,
        item: dict[str, Any],
        raw_response: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize current Weatherbit conditions."""

        return {
            "provider": self.name,
            "location": {
                "name": item.get(
                    "city_name"
                ),
                "state": item.get(
                    "state_code"
                ),
                "country": item.get(
                    "country_code"
                ),
                "latitude": item.get(
                    "lat"
                ),
                "longitude": item.get(
                    "lon"
                ),
                "timezone": item.get(
                    "timezone"
                ),
            },
            "temperature": item.get(
                "temp"
            ),
            "feels_like": item.get(
                "app_temp"
            ),
            "humidity": item.get(
                "rh"
            ),
            "pressure": item.get(
                "pres"
            ),
            "sea_level_pressure": item.get(
                "slp"
            ),
            "visibility": item.get(
                "vis"
            ),
            "uv": item.get(
                "uv"
            ),
            "clouds": item.get(
                "clouds"
            ),
            "precipitation": item.get(
                "precip"
            ),
            "snow": item.get(
                "snow"
            ),
            "wind": {
                "speed": item.get(
                    "wind_spd"
                ),
                "direction": item.get(
                    "wind_dir"
                ),
                "gust": item.get(
                    "gust"
                ),
                "direction_text": item.get(
                    "wind_cdir_full"
                ),
            },
            "weather": self._normalize_weather(
                item.get(
                    "weather",
                    {},
                )
            ),
            "sunrise": item.get(
                "sunrise"
            ),
            "sunset": item.get(
                "sunset"
            ),
            "observation_time": item.get(
                "ob_time"
            ),
            "timestamp": item.get(
                "ts"
            ),
            "raw": raw_response,
        }

    def _normalize_forecast_day(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize one daily forecast record."""

        return {
            "provider": self.name,
            "date": item.get(
                "valid_date"
            ),
            "timestamp": item.get(
                "ts"
            ),
            "temperature": {
                "average": item.get(
                    "temp"
                ),
                "maximum": item.get(
                    "max_temp"
                ),
                "minimum": item.get(
                    "min_temp"
                ),
                "morning": item.get(
                    "app_min_temp"
                ),
                "evening": item.get(
                    "app_max_temp"
                ),
            },
            "feels_like": {
                "average": item.get(
                    "app_temp"
                ),
                "maximum": item.get(
                    "app_max_temp"
                ),
                "minimum": item.get(
                    "app_min_temp"
                ),
            },
            "humidity": item.get(
                "rh"
            ),
            "pressure": item.get(
                "pres"
            ),
            "visibility": item.get(
                "vis"
            ),
            "uv": item.get(
                "uv"
            ),
            "clouds": item.get(
                "clouds"
            ),
            "precipitation": {
                "amount": item.get(
                    "precip"
                ),
                "probability": item.get(
                    "pop"
                ),
                "rain": item.get(
                    "precip"
                ),
            },
            "snow": {
                "amount": item.get(
                    "snow"
                ),
                "probability": item.get(
                    "snow"
                ),
            },
            "wind": {
                "speed": item.get(
                    "wind_spd"
                ),
                "direction": item.get(
                    "wind_dir"
                ),
                "direction_text": item.get(
                    "wind_cdir_full"
                ),
                "gust": item.get(
                    "wind_gust_spd"
                ),
            },
            "weather": self._normalize_weather(
                item.get(
                    "weather",
                    {},
                )
            ),
            "sunrise": item.get(
                "sunrise"
            ),
            "sunset": item.get(
                "sunset"
            ),
            "moonrise": item.get(
                "moonrise_ts"
            ),
            "moonset": item.get(
                "moonset_ts"
            ),
            "raw": item,
        }

    def _normalize_hourly(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize an hourly forecast record."""

        return {
            "provider": self.name,
            "timestamp": item.get(
                "ts"
            ),
            "datetime": item.get(
                "timestamp_local"
            ),
            "temperature": item.get(
                "temp"
            ),
            "feels_like": item.get(
                "app_temp"
            ),
            "humidity": item.get(
                "rh"
            ),
            "pressure": item.get(
                "pres"
            ),
            "visibility": item.get(
                "vis"
            ),
            "uv": item.get(
                "uv"
            ),
            "clouds": item.get(
                "clouds"
            ),
            "precipitation": {
                "amount": item.get(
                    "precip"
                ),
                "probability": item.get(
                    "pop"
                ),
            },
            "wind": {
                "speed": item.get(
                    "wind_spd"
                ),
                "direction": item.get(
                    "wind_dir"
                ),
                "direction_text": item.get(
                    "wind_cdir_full"
                ),
                "gust": item.get(
                    "wind_gust_spd"
                ),
            },
            "weather": self._normalize_weather(
                item.get(
                    "weather",
                    {},
                )
            ),
            "raw": item,
        }

    # ========================================================
    # HEALTH
    # ========================================================

    def health_check(self) -> bool:
        """
        Check Weatherbit API availability.

        A lightweight current-weather request is used to verify
        both credentials and network/API access.
        """

        if not self.api_key:
            return False

        try:
            self._request(
                "current",
                {
                    "city": "Mumbai",
                },
            )

            return True

        except Exception:
            return False


# ============================================================
# FACTORY
# ============================================================


def create_weatherbit_provider(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    units: str = "M",
    language: str = "en",
) -> WeatherbitProvider:
    """
    Create a Weatherbit provider.

    Explicit arguments take priority over environment variables.
    """

    return WeatherbitProvider(
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
    "WeatherbitProvider",
    "create_weatherbit_provider",
]


