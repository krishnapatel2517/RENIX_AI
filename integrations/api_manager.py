"""
RENIX Integrations - API Manager
================================

Central API management layer for RENIX.

Responsibilities:
- Register external API providers
- Manage base URLs and endpoints
- Build authenticated requests
- Handle GET/POST/PUT/PATCH/DELETE requests
- Manage headers and query parameters
- Apply request timeouts
- Retry temporary failures
- Track API health
- Cache optional GET responses
- Provide a common interface to RENIX services

This module intentionally uses only Python's standard library.
Provider-specific SDKs should live inside integrations/providers/.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class APIError(Exception):
    """Base exception for API-related errors."""


class APIConnectionError(APIError):
    """Raised when an API cannot be reached."""


class APIHTTPError(APIError):
    """Raised when an API returns an HTTP error."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        url: str | None = None,
        response_body: Any = None,
        headers: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.status_code = status_code
        self.url = url
        self.response_body = response_body
        self.headers = dict(headers or {})

    def __str__(self) -> str:
        return (
            f"HTTP {self.status_code}: "
            f"{super().__str__()}"
        )


class APITimeoutError(APIConnectionError):
    """Raised when an API request times out."""


class APIRetryExhausted(APIError):
    """Raised when all retry attempts fail."""


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class APIResponse:
    """Normalized response returned by APIManager."""

    request_id: str
    status_code: int
    url: str
    method: str

    data: Any = None
    text: str = ""

    headers: dict[str, str] = field(
        default_factory=dict
    )

    elapsed_seconds: float = 0.0

    success: bool = False

    created_at: str = field(
        default_factory=lambda: (
            datetime.now(timezone.utc)
            .isoformat()
        )
    )

    def json(self) -> Any:
        """Return parsed response data."""

        return self.data

    def ok(self) -> bool:
        """Return whether the response was successful."""

        return self.success

    def to_dict(self) -> dict[str, Any]:
        """Convert response to a serializable dictionary."""

        return {
            "request_id": self.request_id,
            "status_code": self.status_code,
            "url": self.url,
            "method": self.method,
            "data": self.data,
            "text": self.text,
            "headers": dict(self.headers),
            "elapsed_seconds": self.elapsed_seconds,
            "success": self.success,
            "created_at": self.created_at,
        }


@dataclass
class APIProvider:
    """Configuration for an external API provider."""

    name: str
    base_url: str

    default_headers: dict[str, str] = field(
        default_factory=dict
    )

    timeout: float = 30.0

    retries: int = 2

    retry_delay: float = 1.0

    verify_ssl: bool = True

    enabled: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def normalized_base_url(self) -> str:
        """Return a normalized base URL."""

        return self.base_url.rstrip("/") + "/"


@dataclass
class CacheEntry:
    """Single API cache entry."""

    key: str
    response: APIResponse

    created_at: float = field(
        default_factory=time.monotonic
    )

    ttl: float = 60.0

    def expired(self) -> bool:
        """Return whether this cache entry expired."""

        return (
            time.monotonic()
            - self.created_at
            > self.ttl
        )


# ============================================================
# API MANAGER
# ============================================================


class APIManager:
    """
    Centralized API manager for RENIX.

    Example:

        api = APIManager()

        api.register_provider(
            APIProvider(
                name="weather",
                base_url="https://example.com/api",
            )
        )

        response = api.get(
            "weather",
            "/weather",
            params={"city": "Mumbai"},
        )

        print(response.data)
    """

    RETRYABLE_STATUS_CODES = {
        408,
        425,
        429,
        500,
        502,
        503,
        504,
    }

    RETRYABLE_METHODS = {
        "GET",
        "HEAD",
        "OPTIONS",
        "PUT",
        "DELETE",
    }

    def __init__(
        self,
        *,
        default_timeout: float = 30.0,
        default_retries: int = 2,
        default_retry_delay: float = 1.0,
        user_agent: str = "RENIX/1.0",
    ) -> None:
        self.default_timeout = default_timeout
        self.default_retries = default_retries
        self.default_retry_delay = default_retry_delay
        self.user_agent = user_agent

        self._providers: dict[
            str,
            APIProvider,
        ] = {}

        self._cache: dict[
            str,
            CacheEntry,
        ] = {}

        self._request_history: list[
            APIResponse
        ] = []

        self._max_history = 200

        self._lock = threading.RLock()

        self._enabled = True

    # ========================================================
    # PROVIDERS
    # ========================================================

    def register_provider(
        self,
        provider: APIProvider,
    ) -> None:
        """Register or replace an API provider."""

        if not isinstance(
            provider,
            APIProvider,
        ):
            raise TypeError(
                "provider must be an APIProvider."
            )

        if not provider.name.strip():
            raise ValueError(
                "Provider name cannot be empty."
            )

        if not provider.base_url.strip():
            raise ValueError(
                "Provider base_url cannot be empty."
            )

        with self._lock:
            self._providers[
                provider.name
            ] = provider

    def unregister_provider(
        self,
        name: str,
    ) -> bool:
        """Remove a provider."""

        with self._lock:
            return (
                self._providers.pop(
                    name,
                    None,
                )
                is not None
            )

    def get_provider(
        self,
        name: str,
    ) -> APIProvider:
        """Return a provider or raise APIError."""

        with self._lock:
            provider = self._providers.get(
                name
            )

        if provider is None:
            raise APIError(
                f"Unknown API provider: {name}"
            )

        if not provider.enabled:
            raise APIError(
                f"API provider '{name}' is disabled."
            )

        return provider

    def providers(self) -> list[APIProvider]:
        """Return all registered providers."""

        with self._lock:
            return list(
                self._providers.values()
            )

    # ========================================================
    # URL BUILDING
    # ========================================================

    def build_url(
        self,
        provider_name: str,
        endpoint: str = "",
        *,
        params: Mapping[str, Any] | None = None,
    ) -> str:
        """Build a complete provider URL."""

        provider = self.get_provider(
            provider_name
        )

        base = provider.normalized_base_url()

        endpoint = str(endpoint).lstrip("/")

        url = urllib.parse.urljoin(
            base,
            endpoint,
        )

        if params:
            query = urllib.parse.urlencode(
                self._normalize_params(params),
                doseq=True,
            )

            separator = (
                "&"
                if "?" in url
                else "?"
            )

            url = (
                f"{url}{separator}{query}"
            )

        return url

    @staticmethod
    def _normalize_params(
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalize query parameters."""

        result: dict[str, Any] = {}

        for key, value in params.items():
            if value is None:
                continue

            if isinstance(value, bool):
                result[key] = (
                    "true"
                    if value
                    else "false"
                )
            else:
                result[key] = value

        return result

    # ========================================================
    # HEADERS
    # ========================================================

    def build_headers(
        self,
        provider_name: str,
        *,
        headers: Mapping[str, str] | None = None,
        authorization: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, str]:
        """Build request headers."""

        provider = self.get_provider(
            provider_name
        )

        result = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        result.update(
            provider.default_headers
        )

        if headers:
            result.update(
                {
                    str(key): str(value)
                    for key, value in headers.items()
                }
            )

        if authorization:
            result["Authorization"] = (
                authorization
            )

        if content_type:
            result["Content-Type"] = (
                content_type
            )

        return result

    # ========================================================
    # REQUEST
    # ========================================================

    def request(
        self,
        provider_name: str,
        method: str,
        endpoint: str = "",
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_data: Any = None,
        data: bytes | str | None = None,
        authorization: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        cache: bool = False,
        cache_ttl: float = 60.0,
        raise_for_status: bool = True,
    ) -> APIResponse:
        """
        Perform an HTTP request.

        JSON payloads should normally be passed through
        `json_data`.
        """

        if not self._enabled:
            raise APIError(
                "APIManager is disabled."
            )

        method = str(method).upper().strip()

        if not method:
            raise ValueError(
                "HTTP method cannot be empty."
            )

        provider = self.get_provider(
            provider_name
        )

        url = self.build_url(
            provider_name,
            endpoint,
            params=params,
        )

        request_headers = self.build_headers(
            provider_name,
            headers=headers,
            authorization=authorization,
        )

        body: bytes | None = None

        if json_data is not None:
            body = json.dumps(
                json_data,
                ensure_ascii=False,
            ).encode("utf-8")

            request_headers.setdefault(
                "Content-Type",
                "application/json",
            )

        elif data is not None:
            if isinstance(data, str):
                body = data.encode("utf-8")
            else:
                body = data

        request_id = uuid.uuid4().hex

        cache_key = self._cache_key(
            method,
            url,
            body,
        )

        if cache and method == "GET":
            cached = self._get_cached(
                cache_key
            )

            if cached is not None:
                return cached

        max_retries = (
            provider.retries
            if retries is None
            else max(0, int(retries))
        )

        request_timeout = (
            provider.timeout
            if timeout is None
            else float(timeout)
        )

        delay = (
            provider.retry_delay
            if retry_delay is None
            else float(retry_delay)
        )

        last_error: Exception | None = None

        for attempt in range(
            max_retries + 1
        ):
            try:
                response = self._perform_request(
                    request_id=request_id,
                    method=method,
                    url=url,
                    headers=request_headers,
                    body=body,
                    timeout=request_timeout,
                    verify_ssl=provider.verify_ssl,
                )

                self._store_history(
                    response
                )

                if (
                    cache
                    and method == "GET"
                    and response.success
                ):
                    self._store_cache(
                        cache_key,
                        response,
                        cache_ttl,
                    )

                if (
                    raise_for_status
                    and not response.success
                ):
                    raise APIHTTPError(
                        response.status_code,
                        (
                            f"Request failed for "
                            f"{provider_name}"
                        ),
                        url=response.url,
                        response_body=response.data,
                        headers=response.headers,
                    )

                return response

            except APIHTTPError as exc:
                last_error = exc

                if not self._should_retry_http(
                    method,
                    exc.status_code,
                    attempt,
                    max_retries,
                ):
                    raise

            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
            ) as exc:
                last_error = exc

                if attempt >= max_retries:
                    break

            if attempt < max_retries:
                self._sleep_before_retry(
                    delay,
                    attempt,
                )

        raise APIRetryExhausted(
            f"API request failed after "
            f"{max_retries + 1} attempts: "
            f"{url}"
        ) from last_error

    # ========================================================
    # HTTP HELPERS
    # ========================================================

    def get(
        self,
        provider_name: str,
        endpoint: str = "",
        **kwargs: Any,
    ) -> APIResponse:
        """Perform GET request."""

        return self.request(
            provider_name,
            "GET",
            endpoint,
            **kwargs,
        )

    def post(
        self,
        provider_name: str,
        endpoint: str = "",
        **kwargs: Any,
    ) -> APIResponse:
        """Perform POST request."""

        return self.request(
            provider_name,
            "POST",
            endpoint,
            **kwargs,
        )

    def put(
        self,
        provider_name: str,
        endpoint: str = "",
        **kwargs: Any,
    ) -> APIResponse:
        """Perform PUT request."""

        return self.request(
            provider_name,
            "PUT",
            endpoint,
            **kwargs,
        )

    def patch(
        self,
        provider_name: str,
        endpoint: str = "",
        **kwargs: Any,
    ) -> APIResponse:
        """Perform PATCH request."""

        return self.request(
            provider_name,
            "PATCH",
            endpoint,
            **kwargs,
        )

    def delete(
        self,
        provider_name: str,
        endpoint: str = "",
        **kwargs: Any,
    ) -> APIResponse:
        """Perform DELETE request."""

        return self.request(
            provider_name,
            "DELETE",
            endpoint,
            **kwargs,
        )

    def head(
        self,
        provider_name: str,
        endpoint: str = "",
        **kwargs: Any,
    ) -> APIResponse:
        """Perform HEAD request."""

        return self.request(
            provider_name,
            "HEAD",
            endpoint,
            **kwargs,
        )

    # ========================================================
    # INTERNAL HTTP
    # ========================================================

    def _perform_request(
        self,
        *,
        request_id: str,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
        verify_ssl: bool,
    ) -> APIResponse:
        """
        Perform one HTTP request.

        `verify_ssl` is retained in the provider interface for
        future custom SSL contexts. The standard urllib HTTPS
        handler is used here with normal certificate validation.
        """

        started = time.monotonic()

        request = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method=method,
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as raw_response:
                raw_body = raw_response.read()

                elapsed = (
                    time.monotonic()
                    - started
                )

                text = raw_body.decode(
                    "utf-8",
                    errors="replace",
                )

                parsed = self._parse_body(
                    text,
                    raw_response.headers.get(
                        "Content-Type",
                        "",
                    ),
                )

                response = APIResponse(
                    request_id=request_id,
                    status_code=(
                        raw_response.status
                    ),
                    url=url,
                    method=method,
                    data=parsed,
                    text=text,
                    headers={
                        str(key): str(value)
                        for key, value in (
                            raw_response.headers.items()
                        )
                    },
                    elapsed_seconds=elapsed,
                    success=(
                        200
                        <= raw_response.status
                        < 300
                    ),
                )

                return response

        except urllib.error.HTTPError as exc:
            elapsed = (
                time.monotonic()
                - started
            )

            raw_body = exc.read()

            text = raw_body.decode(
                "utf-8",
                errors="replace",
            )

            parsed = self._parse_body(
                text,
                exc.headers.get(
                    "Content-Type",
                    "",
                ),
            )

            response = APIResponse(
                request_id=request_id,
                status_code=exc.code,
                url=url,
                method=method,
                data=parsed,
                text=text,
                headers={
                    str(key): str(value)
                    for key, value in (
                        exc.headers.items()
                    )
                },
                elapsed_seconds=elapsed,
                success=False,
            )

            self._store_history(
                response
            )

            raise APIHTTPError(
                exc.code,
                (
                    exc.reason
                    or "HTTP request failed."
                ),
                url=url,
                response_body=parsed,
                headers=response.headers,
            )

        except TimeoutError as exc:
            raise APITimeoutError(
                f"Request timed out: {url}"
            ) from exc

        except urllib.error.URLError as exc:
            raise APIConnectionError(
                f"Unable to connect to {url}: "
                f"{exc.reason}"
            ) from exc

    # ========================================================
    # PARSING
    # ========================================================

    @staticmethod
    def _parse_body(
        text: str,
        content_type: str = "",
    ) -> Any:
        """Parse JSON responses when appropriate."""

        if not text:
            return None

        normalized_type = (
            content_type.lower()
        )

        if (
            "json" in normalized_type
            or text.lstrip().startswith((
                "{",
                "[",
            ))
        ):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        return text

    # ========================================================
    # RETRY
    # ========================================================

    def _should_retry_http(
        self,
        method: str,
        status_code: int,
        attempt: int,
        max_retries: int,
    ) -> bool:
        """Determine whether an HTTP failure is retryable."""

        if attempt >= max_retries:
            return False

        if method not in self.RETRYABLE_METHODS:
            return False

        return (
            status_code
            in self.RETRYABLE_STATUS_CODES
        )

    @staticmethod
    def _sleep_before_retry(
        delay: float,
        attempt: int,
    ) -> None:
        """Apply exponential retry delay."""

        sleep_time = max(
            0.0,
            delay,
        ) * (2 ** attempt)

        if sleep_time:
            time.sleep(sleep_time)

    # ========================================================
    # CACHE
    # ========================================================

    @staticmethod
    def _cache_key(
        method: str,
        url: str,
        body: bytes | None,
    ) -> str:
        """Generate a deterministic cache key."""

        import hashlib

        digest = hashlib.sha256()

        digest.update(
            method.encode("utf-8")
        )

        digest.update(
            b"\0"
        )

        digest.update(
            url.encode("utf-8")
        )

        if body:
            digest.update(
                b"\0"
            )
            digest.update(body)

        return digest.hexdigest()

    def _get_cached(
        self,
        key: str,
    ) -> APIResponse | None:
        """Return a valid cached response."""

        with self._lock:
            entry = self._cache.get(
                key
            )

            if entry is None:
                return None

            if entry.expired():
                del self._cache[key]
                return None

            return entry.response

    def _store_cache(
        self,
        key: str,
        response: APIResponse,
        ttl: float,
    ) -> None:
        """Store a response in cache."""

        if ttl <= 0:
            return

        with self._lock:
            self._cache[key] = CacheEntry(
                key=key,
                response=response,
                ttl=ttl,
            )

    def clear_cache(self) -> None:
        """Clear all cached API responses."""

        with self._lock:
            self._cache.clear()

    def cleanup_cache(self) -> int:
        """Remove expired cache entries."""

        removed = 0

        with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.expired()
            ]

            for key in expired_keys:
                del self._cache[key]
                removed += 1

        return removed

    # ========================================================
    # HISTORY
    # ========================================================

    def _store_history(
        self,
        response: APIResponse,
    ) -> None:
        """Store response in request history."""

        with self._lock:
            self._request_history.append(
                response
            )

            if (
                len(self._request_history)
                > self._max_history
            ):
                self._request_history.pop(
                    0
                )

    def history(
        self,
        limit: int = 50,
    ) -> list[APIResponse]:
        """Return recent API responses."""

        if limit < 0:
            raise ValueError(
                "limit must be >= 0."
            )

        with self._lock:
            return list(
                self._request_history[
                    -limit:
                ]
            )

    def clear_history(self) -> None:
        """Clear request history."""

        with self._lock:
            self._request_history.clear()

    # ========================================================
    # HEALTH
    # ========================================================

    def check_provider(
        self,
        provider_name: str,
        endpoint: str = "",
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Check whether a provider is reachable."""

        started = time.monotonic()

        try:
            response = self.request(
                provider_name,
                "GET",
                endpoint,
                timeout=timeout,
                retries=0,
                raise_for_status=False,
            )

            return {
                "provider": provider_name,
                "healthy": response.success,
                "status_code": response.status_code,
                "elapsed_seconds": (
                    time.monotonic()
                    - started
                ),
                "error": None,
            }

        except Exception as exc:
            return {
                "provider": provider_name,
                "healthy": False,
                "status_code": None,
                "elapsed_seconds": (
                    time.monotonic()
                    - started
                ),
                "error": str(exc),
            }

    def health_check_all(
        self,
    ) -> dict[str, Any]:
        """Check all registered providers."""

        results: dict[str, Any] = {}

        for provider in self.providers():
            results[
                provider.name
            ] = self.check_provider(
                provider.name
            )

        return results

    # ========================================================
    # CONTROL
    # ========================================================

    def enable(self) -> None:
        """Enable API operations."""

        self._enabled = True

    def disable(self) -> None:
        """Disable API operations."""

        self._enabled = False

    def is_enabled(self) -> bool:
        """Return whether the API manager is enabled."""

        return self._enabled

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict[str, Any]:
        """Return API manager status."""

        with self._lock:
            return {
                "enabled": self._enabled,
                "provider_count": len(
                    self._providers
                ),
                "providers": [
                    {
                        "name": provider.name,
                        "base_url": provider.base_url,
                        "enabled": provider.enabled,
                        "timeout": provider.timeout,
                        "retries": provider.retries,
                    }
                    for provider in self._providers.values()
                ],
                "cache_entries": len(
                    self._cache
                ),
                "history_entries": len(
                    self._request_history
                ),
            }

    def __len__(self) -> int:
        """Return number of registered providers."""

        return len(self._providers)

    def __repr__(self) -> str:
        return (
            "APIManager("
            f"enabled={self._enabled}, "
            f"providers={len(self._providers)})"
        )


__all__ = [
    "APIError",
    "APIConnectionError",
    "APIHTTPError",
    "APITimeoutError",
    "APIRetryExhausted",
    "APIResponse",
    "APIProvider",
    "CacheEntry",
    "APIManager",
]


