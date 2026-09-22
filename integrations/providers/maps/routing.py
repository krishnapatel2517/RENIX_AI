"""
RENIX Maps - Routing Provider Layer
====================================

Provider-independent routing service for RENIX.

Responsibilities:
    - Route calculation
    - Distance and duration
    - Turn-by-turn directions
    - Alternative routes
    - Route normalization
    - Provider fallback
    - Route caching

Expected provider interface:

    provider.route(...)
    or
    provider.get_route(...)

The provider-specific implementation should remain inside
the corresponding provider adapter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time

from dataclasses import dataclass, field
from typing import Any, Iterable


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class RoutingError(RuntimeError):
    """Base routing exception."""


class RouteNotFoundError(RoutingError):
    """Raised when no route can be calculated."""


class RoutingProviderError(RoutingError):
    """Raised when a routing provider fails."""


class InvalidRouteRequestError(RoutingError):
    """Raised when route input is invalid."""


# ============================================================
# DATA MODELS
# ============================================================


@dataclass(frozen=True)
class Coordinate:
    """Geographic coordinate."""

    latitude: float
    longitude: float

    def validate(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise InvalidRouteRequestError(
                "Latitude must be between -90 and 90."
            )

        if not -180.0 <= self.longitude <= 180.0:
            raise InvalidRouteRequestError(
                "Longitude must be between -180 and 180."
            )

    def as_tuple(self) -> tuple[float, float]:
        return (
            self.latitude,
            self.longitude,
        )


@dataclass
class RouteStep:
    """One turn-by-turn navigation step."""

    instruction: str = ""

    distance_meters: float = 0.0

    duration_seconds: float = 0.0

    start: Coordinate | None = None

    end: Coordinate | None = None

    maneuver: str | None = None

    road_name: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Route:
    """Normalized RENIX route."""

    route_id: str | None = None

    provider: str = "unknown"

    distance_meters: float = 0.0

    duration_seconds: float = 0.0

    start: Coordinate | None = None

    end: Coordinate | None = None

    geometry: Any = None

    steps: list[RouteStep] = field(
        default_factory=list
    )

    summary: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def distance_km(self) -> float:
        return self.distance_meters / 1000.0

    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0


@dataclass
class RoutingResult:
    """Result containing one or more routes."""

    routes: list[Route]

    provider: str

    origin: Coordinate

    destination: Coordinate

    requested_at: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def best_route(self) -> Route | None:
        return (
            self.routes[0]
            if self.routes
            else None
        )


# ============================================================
# ROUTING SERVICE
# ============================================================


class RoutingService:
    """
    Main provider-independent routing service.

    Example:

        routing = RoutingService(provider_registry)

        result = routing.route(
            origin=(19.0760, 72.8777),
            destination=(19.1136, 72.8697),
            mode="driving",
        )

        print(result.best_route.distance_km())
    """

    def __init__(
        self,
        registry: Any,
        *,
        cache_enabled: bool = True,
        cache_ttl: int = 300,
        max_cache_size: int = 250,
    ) -> None:

        if registry is None:
            raise ValueError(
                "A maps provider registry is required."
            )

        self.registry = registry

        self.cache_enabled = cache_enabled

        self.cache_ttl = max(
            0,
            int(cache_ttl),
        )

        self.max_cache_size = max(
            1,
            int(max_cache_size),
        )

        self._cache: dict[
            str,
            tuple[
                float,
                RoutingResult,
            ],
        ] = {}

        self._lock = threading.RLock()

    # ========================================================
    # PUBLIC ROUTING API
    # ========================================================

    def route(
        self,
        origin: Coordinate | tuple[float, float] | list[float],
        destination: Coordinate | tuple[float, float] | list[float],
        *,
        mode: str = "driving",
        provider: str | None = None,
        alternatives: bool = False,
        avoid: Iterable[str] | None = None,
        waypoints: Iterable[
            Coordinate
            | tuple[float, float]
            | list[float]
        ]
        | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> RoutingResult:
        """
        Calculate a route between two locations.

        Supported common modes:

            driving
            walking
            cycling
            transit

        Provider-specific modes may also be passed through.
        """

        start = self._normalize_coordinate(
            origin
        )

        end = self._normalize_coordinate(
            destination
        )

        start.validate()
        end.validate()

        normalized_waypoints = [
            self._normalize_coordinate(
                point
            )
            for point in (
                waypoints or []
            )
        ]

        for point in normalized_waypoints:
            point.validate()

        mode = self._normalize_mode(
            mode
        )

        avoid_list = [
            str(item).strip().lower()
            for item in (
                avoid or []
            )
            if str(item).strip()
        ]

        cache_key = self._build_cache_key(
            start,
            end,
            mode,
            provider,
            alternatives,
            avoid_list,
            normalized_waypoints,
            kwargs,
        )

        cached = self._get_cache(
            cache_key
        )

        if cached is not None:
            return cached

        providers = self._provider_chain(
            provider=provider,
            fallback=fallback,
        )

        errors: list[str] = []

        for maps_provider in providers:

            try:

                raw_result = self._call_provider(
                    maps_provider,
                    start,
                    end,
                    mode=mode,
                    alternatives=alternatives,
                    avoid=avoid_list,
                    waypoints=normalized_waypoints,
                    **kwargs,
                )

                result = self._normalize_result(
                    raw_result,
                    provider_name=getattr(
                        maps_provider,
                        "name",
                        "unknown",
                    ),
                    origin=start,
                    destination=end,
                )

                if not result.routes:
                    raise RouteNotFoundError(
                        "Provider returned no routes."
                    )

                self._set_cache(
                    cache_key,
                    result,
                )

                return result

            except Exception as exc:

                provider_name = getattr(
                    maps_provider,
                    "name",
                    "unknown",
                )

                errors.append(
                    f"{provider_name}: {exc}"
                )

                logger.warning(
                    "Routing provider failed: %s",
                    provider_name,
                )

        raise RouteNotFoundError(
            self._format_provider_errors(
                start,
                end,
                errors,
            )
        )

    # ========================================================
    # CONVENIENCE METHODS
    # ========================================================

    def driving(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        **kwargs: Any,
    ) -> RoutingResult:
        """Calculate a driving route."""

        return self.route(
            origin,
            destination,
            mode="driving",
            **kwargs,
        )

    def walking(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        **kwargs: Any,
    ) -> RoutingResult:
        """Calculate a walking route."""

        return self.route(
            origin,
            destination,
            mode="walking",
            **kwargs,
        )

    def cycling(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        **kwargs: Any,
    ) -> RoutingResult:
        """Calculate a cycling route."""

        return self.route(
            origin,
            destination,
            mode="cycling",
            **kwargs,
        )

    def transit(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        **kwargs: Any,
    ) -> RoutingResult:
        """Calculate a public-transit route."""

        return self.route(
            origin,
            destination,
            mode="transit",
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
        **kwargs: Any,
    ) -> float:
        """
        Return route distance in meters.
        """

        result = self.route(
            origin,
            destination,
            mode=mode,
            **kwargs,
        )

        if result.best_route is None:
            raise RouteNotFoundError(
                "No route available."
            )

        return result.best_route.distance_meters

    def distance_km(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        **kwargs: Any,
    ) -> float:
        """Return route distance in kilometers."""

        return self.distance(
            origin,
            destination,
            mode=mode,
            **kwargs,
        ) / 1000.0

    # ========================================================
    # DURATION
    # ========================================================

    def duration(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        **kwargs: Any,
    ) -> float:
        """
        Return route duration in seconds.
        """

        result = self.route(
            origin,
            destination,
            mode=mode,
            **kwargs,
        )

        if result.best_route is None:
            raise RouteNotFoundError(
                "No route available."
            )

        return result.best_route.duration_seconds

    def duration_minutes(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        **kwargs: Any,
    ) -> float:
        """Return route duration in minutes."""

        return self.duration(
            origin,
            destination,
            mode=mode,
            **kwargs,
        ) / 60.0

    # ========================================================
    # DIRECTIONS
    # ========================================================

    def directions(
        self,
        origin: Coordinate | tuple[float, float],
        destination: Coordinate | tuple[float, float],
        *,
        mode: str = "driving",
        **kwargs: Any,
    ) -> list[RouteStep]:
        """Return turn-by-turn route instructions."""

        result = self.route(
            origin,
            destination,
            mode=mode,
            **kwargs,
        )

        if result.best_route is None:
            return []

        return result.best_route.steps

    # ========================================================
    # PROVIDER CALL
    # ========================================================

    def _call_provider(
        self,
        provider: Any,
        origin: Coordinate,
        destination: Coordinate,
        *,
        mode: str,
        alternatives: bool,
        avoid: list[str],
        waypoints: list[Coordinate],
        **kwargs: Any,
    ) -> Any:
        """
        Call a provider using its supported routing method.
        """

        payload = {
            "origin": origin.as_tuple(),
            "destination": destination.as_tuple(),
            "mode": mode,
            "alternatives": alternatives,
            "avoid": avoid,
            "waypoints": [
                point.as_tuple()
                for point in waypoints
            ],
            **kwargs,
        }

        route_method = getattr(
            provider,
            "route",
            None,
        )

        if callable(route_method):

            return route_method(
                **payload
            )

        route_method = getattr(
            provider,
            "get_route",
            None,
        )

        if callable(route_method):

            return route_method(
                **payload
            )

        raise RoutingProviderError(
            "Provider does not expose route() "
            "or get_route()."
        )

    # ========================================================
    # PROVIDER CHAIN
    # ========================================================

    def _provider_chain(
        self,
        *,
        provider: str | None,
        fallback: bool,
    ) -> list[Any]:

        if provider:

            selected = self.registry.get(
                provider
            )

            if not fallback:
                return [selected]

            chain = [selected]

            fallbacks = getattr(
                self.registry,
                "get_fallbacks",
                None,
            )

            if callable(fallbacks):

                for item in (
                    fallbacks() or []
                ):

                    if item not in chain:
                        chain.append(item)

            return chain

        if fallback:

            method = getattr(
                self.registry,
                "get_provider_chain",
                None,
            )

            if callable(method):

                return list(
                    method(
                        include_active=True
                    )
                    or []
                )

        active = getattr(
            self.registry,
            "get_active",
            None,
        )

        if not callable(active):
            raise RoutingProviderError(
                "Provider registry has no active provider."
            )

        return [
            active()
        ]

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_result(
        self,
        raw: Any,
        *,
        provider_name: str,
        origin: Coordinate,
        destination: Coordinate,
    ) -> RoutingResult:

        if isinstance(
            raw,
            RoutingResult,
        ):
            return raw

        if isinstance(
            raw,
            dict,
        ):

            raw_routes = raw.get(
                "routes"
            )

            if raw_routes is None:
                raw_routes = [
                    raw
                ]

        elif isinstance(
            raw,
            (list, tuple),
        ):

            raw_routes = raw

        else:

            raise RoutingProviderError(
                "Provider returned unsupported "
                "routing data."
            )

        routes = []

        for index, raw_route in enumerate(
            raw_routes
        ):

            route = self._normalize_route(
                raw_route,
                provider_name=provider_name,
                origin=origin,
                destination=destination,
                index=index,
            )

            if route is not None:
                routes.append(route)

        return RoutingResult(
            routes=routes,
            provider=provider_name,
            origin=origin,
            destination=destination,
        )

    def _normalize_route(
        self,
        raw: Any,
        *,
        provider_name: str,
        origin: Coordinate,
        destination: Coordinate,
        index: int,
    ) -> Route | None:

        if isinstance(
            raw,
            Route,
        ):
            return raw

        if not isinstance(
            raw,
            dict,
        ):
            return None

        distance = self._first_number(
            raw,
            (
                "distance_meters",
                "distance",
                "length",
            ),
        )

        duration = self._first_number(
            raw,
            (
                "duration_seconds",
                "duration",
                "time",
            ),
        )

        raw_steps = (
            raw.get("steps")
            or raw.get("legs")
            or raw.get("instructions")
            or []
        )

        steps = self._normalize_steps(
            raw_steps
        )

        route_id = (
            raw.get("route_id")
            or raw.get("id")
            or f"{provider_name}-{index}"
        )

        return Route(
            route_id=str(
                route_id
            ),
            provider=provider_name,
            distance_meters=distance,
            duration_seconds=duration,
            start=self._extract_coordinate(
                raw.get("start")
            ) or origin,
            end=self._extract_coordinate(
                raw.get("end")
            ) or destination,
            geometry=(
                raw.get("geometry")
                or raw.get("polyline")
                or raw.get("shape")
            ),
            steps=steps,
            summary=str(
                raw.get(
                    "summary",
                    "",
                )
            ),
            metadata=dict(
                raw.get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    def _normalize_steps(
        self,
        raw_steps: Any,
    ) -> list[RouteStep]:

        if not isinstance(
            raw_steps,
            (list, tuple),
        ):
            return []

        result = []

        for raw in raw_steps:

            if isinstance(
                raw,
                RouteStep,
            ):

                result.append(raw)
                continue

            if not isinstance(
                raw,
                dict,
            ):
                continue

            result.append(
                RouteStep(
                    instruction=str(
                        raw.get(
                            "instruction"
                        )
                        or raw.get(
                            "text"
                        )
                        or raw.get(
                            "name"
                        )
                        or ""
                    ),
                    distance_meters=self._first_number(
                        raw,
                        (
                            "distance_meters",
                            "distance",
                        ),
                    ),
                    duration_seconds=self._first_number(
                        raw,
                        (
                            "duration_seconds",
                            "duration",
                        ),
                    ),
                    start=self._extract_coordinate(
                        raw.get(
                            "start"
                        )
                    ),
                    end=self._extract_coordinate(
                        raw.get(
                            "end"
                        )
                    ),
                    maneuver=raw.get(
                        "maneuver"
                    ),
                    road_name=(
                        raw.get(
                            "road_name"
                        )
                        or raw.get(
                            "road"
                        )
                    ),
                    metadata=dict(
                        raw.get(
                            "metadata",
                            {},
                        )
                        or {}
                    ),
                )
            )

        return result

    # ========================================================
    # DATA HELPERS
    # ========================================================

    @staticmethod
    def _normalize_coordinate(
        value: Any,
    ) -> Coordinate:

        if isinstance(
            value,
            Coordinate,
        ):
            return value

        if isinstance(
            value,
            (tuple, list),
        ) and len(value) >= 2:

            try:

                return Coordinate(
                    latitude=float(
                        value[0]
                    ),
                    longitude=float(
                        value[1]
                    ),
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise InvalidRouteRequestError(
                    "Invalid coordinate values."
                ) from exc

        if isinstance(
            value,
            dict,
        ):

            try:

                return Coordinate(
                    latitude=float(
                        value[
                            "latitude"
                        ]
                    ),
                    longitude=float(
                        value[
                            "longitude"
                        ]
                    ),
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:

                raise InvalidRouteRequestError(
                    "Invalid coordinate dictionary."
                ) from exc

        raise InvalidRouteRequestError(
            "Coordinates must be a Coordinate, "
            "(latitude, longitude), or dictionary."
        )

    @staticmethod
    def _extract_coordinate(
        value: Any,
    ) -> Coordinate | None:

        if value is None:
            return None

        try:

            return RoutingService._normalize_coordinate(
                value
            )

        except InvalidRouteRequestError:

            return None

    @staticmethod
    def _normalize_mode(
        mode: str,
    ) -> str:

        if not isinstance(
            mode,
            str,
        ):
            raise InvalidRouteRequestError(
                "Routing mode must be a string."
            )

        mode = mode.strip().lower()

        aliases = {
            "car": "driving",
            "auto": "driving",
            "motor": "driving",
            "foot": "walking",
            "pedestrian": "walking",
            "bike": "cycling",
            "bicycle": "cycling",
            "public": "transit",
            "public_transport": "transit",
        }

        return aliases.get(
            mode,
            mode,
        )

    @staticmethod
    def _first_number(
        data: dict[str, Any],
        keys: tuple[str, ...],
    ) -> float:

        for key in keys:

            value = data.get(
                key
            )

            if value is None:
                continue

            try:
                return float(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        return 0.0

    # ========================================================
    # CACHE
    # ========================================================

    def _build_cache_key(
        self,
        origin: Coordinate,
        destination: Coordinate,
        mode: str,
        provider: str | None,
        alternatives: bool,
        avoid: list[str],
        waypoints: list[Coordinate],
        kwargs: dict[str, Any],
    ) -> str:

        payload = {
            "origin": origin.as_tuple(),
            "destination": destination.as_tuple(),
            "mode": mode,
            "provider": provider,
            "alternatives": alternatives,
            "avoid": avoid,
            "waypoints": [
                item.as_tuple()
                for item in waypoints
            ],
            "options": kwargs,
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()

    def _get_cache(
        self,
        key: str,
    ) -> RoutingResult | None:

        if not self.cache_enabled:
            return None

        with self._lock:

            item = self._cache.get(
                key
            )

            if item is None:
                return None

            timestamp, value = item

            if (
                time.monotonic()
                - timestamp
                > self.cache_ttl
            ):

                self._cache.pop(
                    key,
                    None,
                )

                return None

            return value

    def _set_cache(
        self,
        key: str,
        value: RoutingResult,
    ) -> None:

        if not self.cache_enabled:
            return

        with self._lock:

            while (
                len(self._cache)
                >= self.max_cache_size
            ):

                oldest = next(
                    iter(self._cache),
                    None,
                )

                if oldest is None:
                    break

                self._cache.pop(
                    oldest,
                    None,
                )

            self._cache[
                key
            ] = (
                time.monotonic(),
                value,
            )

    def clear_cache(self) -> None:
        """Clear all cached routes."""

        with self._lock:
            self._cache.clear()

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    @staticmethod
    def _format_provider_errors(
        origin: Coordinate,
        destination: Coordinate,
        errors: list[str],
    ) -> str:

        message = (
            "Unable to calculate a route "
            f"from {origin.latitude:.6f},"
            f"{origin.longitude:.6f} "
            "to "
            f"{destination.latitude:.6f},"
            f"{destination.longitude:.6f}."
        )

        if errors:
            message += (
                " Provider errors: "
                + " | ".join(errors)
            )

        return message


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "Coordinate",
    "RouteStep",
    "Route",
    "RoutingResult",
    "RoutingService",
    "RoutingError",
    "RouteNotFoundError",
    "RoutingProviderError",
    "InvalidRouteRequestError",
]


