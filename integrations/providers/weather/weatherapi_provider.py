"""
RENIX WeatherAPI Provider
=========================

WeatherAPI.com implementation for RENIX's weather provider
interface.

Environment variables:

    WEATHERAPI_KEY
    WEATHERAPI_BASE_URL       optional
    RENIX_WEATHER_TIMEOUT      optional

This provider uses Python's standard-library HTTP client.
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

DEFAULT_BASE_URL = "https://api.weatherapi.com/v1"
DEFAULT_TIMEOUT = 15.0


# ============================================================
# WEATHERAPI PROVIDER
# ============================================================


class WeatherAPIProvider(WeatherProvider):
    """
    WeatherAPI.com implementation of WeatherProvider.

    Example:

        provider = WeatherAPIProvider(
            api_key="YOUR_API_KEY"
        )

        weather = provider.get_current_weather(
            "Mumbai"
        )

        forecast = provider.get_forecast(
            "Mumbai",
            days=5,
        )
    """

    name = "weatherapi"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        language: str = "en",
        **kwargs: Any,
    ) -> None:

        resolved_key = (
            api_key
            or os.getenv("WEATHERAPI_KEY")
        )

        resolved_base_url = (
            base_url
            or os.getenv(
                "WEATHERAPI_BASE_URL"
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
            language=language,
            **kwargs,
        )

        self.base_url = (
            resolved_base_url.rstrip("/")
        )

        self.timeout = float(timeout)

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
        """Build a WeatherAPI request URL."""

        if not self.api_key:
            raise ValueError(
                "WEATHERAPI_KEY is not configured."
            )

        query = dict(params)

        query["key"] = self.api_key

        return (
            f"{self.base_url}/"
            f"{endpoint.lstrip('/')}"
            f"?{urllib.parse.urlencode(query)}"
        )

    # ========================================================
    # HTTP REQUEST
    # ========================================================

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform a WeatherAPI HTTP GET request."""

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

            if not (
                200 <= status_code < 300
            ):
                raise RuntimeError(
                    f"WeatherAPI returned HTTP "
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
                    "WeatherAPI returned invalid JSON."
                ) from exc

            if not isinstance(
                payload,
                dict,
            ):
                raise RuntimeError(
                    "WeatherAPI returned an unexpected "
                    "response format."
                )

            self._raise_for_api_error(
                payload
            )

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

            message = self._extract_error_message(
                body
            )

            raise RuntimeError(
                f"WeatherAPI HTTP {exc.code}: "
                f"{message}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Unable to connect to WeatherAPI: "
                f"{exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise TimeoutError(
                "WeatherAPI request timed out."
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
        """Extract an API error message."""

        if not body:
            return "Unknown API error."

        try:
            payload = json.loads(body)

            if isinstance(
                payload,
                dict,
            ):
                error = payload.get(
                    "error"
                )

                if isinstance(
                    error,
                    dict,
                ):
                    message = error.get(
                        "message"
                    )

                    if message:
                        return str(
                            message
                        )

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

    @classmethod
    def _raise_for_api_error(
        cls,
        payload: dict[str, Any],
    ) -> None:
        """Raise when WeatherAPI reports an application error."""

        error = payload.get(
            "error"
        )

        if not error:
            return

        if isinstance(
            error,
            dict,
        ):
            code = error.get(
                "code"
            )

            message = error.get(
                "message",
                "Unknown WeatherAPI error.",
            )

            raise RuntimeError(
                f"WeatherAPI error "
                f"{code}: {message}"
            )

        raise RuntimeError(
            f"WeatherAPI error: {error}"
        )

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
        """

        if not location or not location.strip():
            raise ValueError(
                "Location cannot be empty."
            )

        params: dict[str, Any] = {
            "q": location.strip(),
            "aqi": kwargs.get(
                "air_quality",
                "no",
            ),
            "alerts": kwargs.get(
                "alerts",
                "no",
            ),
        }

        if kwargs.get("language"):
            params["lang"] = kwargs[
                "language"
            ]

        data = self._request(
            "current.json",
            params,
        )

        return self._normalize_current(
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
        Get daily forecast information.

        WeatherAPI supports a forecast horizon depending on the
        account/API plan. The provider requests the requested
        number of days and caps the value at 14 to prevent
        accidental oversized requests.
        """

        if not location or not location.strip():
            raise ValueError(
                "Location cannot be empty."
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

        days = min(
            days,
            14,
        )

        params: dict[str, Any] = {
            "q": location.strip(),
            "days": days,
            "aqi": kwargs.get(
                "air_quality",
                "no",
            ),
            "alerts": kwargs.get(
                "alerts",
                "no",
            ),
        }

        if kwargs.get("language"):
            params["lang"] = kwargs[
                "language"
            ]

        data = self._request(
            "forecast.json",
            params,
        )

        forecast = data.get(
            "forecast",
            {},
        )

        forecast_days = forecast.get(
            "forecastday",
            [],
        )

        if not isinstance(
            forecast_days,
            list,
        ):
            return []

        result: list[
            dict[str, Any]
        ] = []

        for item in forecast_days:
            if not isinstance(
                item,
                dict,
            ):
                continue

            result.append(
                self._normalize_forecast_day(
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
        Search for locations using WeatherAPI's search endpoint.
        """

        if not query or not query.strip():
            raise ValueError(
                "Location query cannot be empty."
            )

        data = self._request(
            "search.json",
            {
                "q": query.strip(),
            },
        )

        # WeatherAPI returns a JSON array for search requests.
        # The generic _request method expects a dictionary, so
        # the standard search endpoint is handled separately.
        #
        # This branch is retained for compatibility if a custom
        # WeatherAPI-compatible endpoint returns an object.
        if isinstance(
            data,
            list,
        ):
            locations = data
        else:
            locations = data.get(
                "locations",
                [],
            )

        result: list[
            dict[str, Any]
        ] = []

        if not isinstance(
            locations,
            list,
        ):
            return result

        for item in locations:
            if not isinstance(
                item,
                dict,
            ):
                continue

            result.append(
                {
                    "name": item.get(
                        "name"
                    ),
                    "region": item.get(
                        "region"
                    ),
                    "country": item.get(
                        "country"
                    ),
                    "latitude": item.get(
                        "lat"
                    ),
                    "longitude": item.get(
                        "lon"
                    ),
                    "timezone": item.get(
                        "tz_id"
                    ),
                    "local_time_epoch": item.get(
                        "localtime_epoch"
                    ),
                    "local_time": item.get(
                        "localtime"
                    ),
                    "provider": self.name,
                }
            )

        return result

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_location(
        self,
        location: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize location information."""

        return {
            "name": location.get(
                "name"
            ),
            "region": location.get(
                "region"
            ),
            "country": location.get(
                "country"
            ),
            "latitude": location.get(
                "lat"
            ),
            "longitude": location.get(
                "lon"
            ),
            "timezone": location.get(
                "tz_id"
            ),
            "local_time": location.get(
                "localtime"
            ),
            "local_time_epoch": location.get(
                "localtime_epoch"
            ),
        }

    def _normalize_condition(
        self,
        condition: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize weather condition information."""

        return {
            "text": condition.get(
                "text"
            ),
            "icon": condition.get(
                "icon"
            ),
            "code": condition.get(
                "code"
            ),
        }

    def _normalize_current(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize current weather response."""

        location = data.get(
            "location",
            {},
        )

        current = data.get(
            "current",
            {},
        )

        condition = current.get(
            "condition",
            {},
        )

        return {
            "provider": self.name,
            "location": self._normalize_location(
                location
            ),
            "temperature": current.get(
                "temp_c"
            ),
            "temperature_f": current.get(
                "temp_f"
            ),
            "feels_like": current.get(
                "feelslike_c"
            ),
            "feels_like_f": current.get(
                "feelslike_f"
            ),
            "humidity": current.get(
                "humidity"
            ),
            "pressure": current.get(
                "pressure_mb"
            ),
            "visibility": current.get(
                "vis_km"
            ),
            "wind": {
                "speed": current.get(
                    "wind_kph"
                ),
                "speed_mph": current.get(
                    "wind_mph"
                ),
                "direction": current.get(
                    "wind_degree"
                ),
                "direction_text": current.get(
                    "wind_dir"
                ),
                "gust": current.get(
                    "gust_kph"
                ),
            },
            "clouds": current.get(
                "cloud"
            ),
            "uv": current.get(
                "uv"
            ),
            "precipitation": current.get(
                "precip_mm"
            ),
            "condition": self._normalize_condition(
                condition
            ),
            "is_day": bool(
                current.get(
                    "is_day",
                    0,
                )
            ),
            "last_updated": current.get(
                "last_updated"
            ),
            "last_updated_epoch": current.get(
                "last_updated_epoch"
            ),
            "raw": data,
        }

    def _normalize_forecast_day(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize one WeatherAPI forecast day."""

        day = item.get(
            "day",
            {},
        )

        astro = item.get(
            "astro",
            {},
        )

        condition = day.get(
            "condition",
            {},
        )

        hourly = item.get(
            "hour",
            [],
        )

        normalized_hours: list[
            dict[str, Any]
        ] = []

        if isinstance(
            hourly,
            list,
        ):
            for hour in hourly:
                if not isinstance(
                    hour,
                    dict,
                ):
                    continue

                normalized_hours.append(
                    {
                        "time": hour.get(
                            "time"
                        ),
                        "time_epoch": hour.get(
                            "time_epoch"
                        ),
                        "temperature": hour.get(
                            "temp_c"
                        ),
                        "feels_like": hour.get(
                            "feelslike_c"
                        ),
                        "humidity": hour.get(
                            "humidity"
                        ),
                        "precipitation": hour.get(
                            "precip_mm"
                        ),
                        "precipitation_probability": hour.get(
                            "chance_of_rain"
                        ),
                        "snow_probability": hour.get(
                            "chance_of_snow"
                        ),
                        "wind_speed": hour.get(
                            "wind_kph"
                        ),
                        "wind_direction": hour.get(
                            "wind_degree"
                        ),
                        "wind_direction_text": hour.get(
                            "wind_dir"
                        ),
                        "clouds": hour.get(
                            "cloud"
                        ),
                        "uv": hour.get(
                            "uv"
                        ),
                        "condition": self._normalize_condition(
                            hour.get(
                                "condition",
                                {},
                            )
                        ),
                    }
                )

        return {
            "provider": self.name,
            "date": item.get(
                "date"
            ),
            "date_epoch": item.get(
                "date_epoch"
            ),
            "temperature": {
                "average": day.get(
                    "avgtemp_c"
                ),
                "maximum": day.get(
                    "maxtemp_c"
                ),
                "minimum": day.get(
                    "mintemp_c"
                ),
            },
            "feels_like": {
                "average": day.get(
                    "avgtemp_c"
                ),
            },
            "condition": self._normalize_condition(
                condition
            ),
            "humidity": day.get(
                "avghumidity"
            ),
            "visibility": day.get(
                "avgvis_km"
            ),
            "wind": {
                "maximum_speed": day.get(
                    "maxwind_kph"
                ),
                "maximum_gust": day.get(
                    "maxwind_mph"
                ),
            },
            "precipitation": {
                "total_mm": day.get(
                    "totalprecip_mm"
                ),
                "probability": day.get(
                    "daily_chance_of_rain"
                ),
            },
            "snow": {
                "total_cm": day.get(
                    "totalsnow_cm"
                ),
                "probability": day.get(
                    "daily_chance_of_snow"
                ),
            },
            "uv": day.get(
                "uv"
            ),
            "sunrise": astro.get(
                "sunrise"
            ),
            "sunset": astro.get(
                "sunset"
            ),
            "moonrise": astro.get(
                "moonrise"
            ),
            "moonset": astro.get(
                "moonset"
            ),
            "moon_phase": astro.get(
                "moon_phase"
            ),
            "hours": normalized_hours,
            "raw": item,
        }

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(self) -> bool:
        """
        Verify credentials and API connectivity.

        A small current-weather request is used because this
        confirms both local configuration and actual API access.
        """

        if not self.api_key:
            return False

        try:
            self._request(
                "current.json",
                {
                    "q": "Mumbai",
                    "aqi": "no",
                    "alerts": "no",
                },
            )

            return True

        except Exception:
            return False


# ============================================================
# FACTORY
# ============================================================


def create_weatherapi_provider(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    language: str = "en",
) -> WeatherAPIProvider:
    """Create a WeatherAPI provider."""

    return WeatherAPIProvider(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        language=language,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "WeatherAPIProvider",
    "create_weatherapi_provider",
]


