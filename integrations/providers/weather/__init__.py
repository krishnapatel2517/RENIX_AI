"""
RENIX Weather Providers
=======================

Weather provider package for RENIX.

This package will contain integrations with external weather
services while keeping provider-specific API logic separate
from the rest of RENIX.

Example:

    from integrations.providers.weather import (
        WeatherProvider,
    )
"""

from __future__ import annotations

from typing import Any


class WeatherProvider:
    """
    Base interface for weather providers.

    Concrete providers should implement the methods defined
    here. Keeping a small common interface allows RENIX to
    switch weather services without changing the rest of the
    application.
    """

    name: str = "base"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.config = kwargs

    def get_current_weather(
        self,
        location: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Return current weather information.

        Concrete providers must override this method.
        """

        raise NotImplementedError(
            "Weather providers must implement "
            "get_current_weather()."
        )

    def get_forecast(
        self,
        location: str,
        *,
        days: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Return a weather forecast.

        Concrete providers must override this method.
        """

        raise NotImplementedError(
            "Weather providers must implement "
            "get_forecast()."
        )

    def search_location(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for a location.

        Concrete providers may override this method.
        """

        raise NotImplementedError(
            "Weather providers must implement "
            "search_location()."
        )

    def health_check(self) -> bool:
        """
        Check whether the provider is available.

        The base implementation only verifies that the
        provider has an API key.
        """

        return bool(self.api_key)


__all__ = [
    "WeatherProvider",
]


