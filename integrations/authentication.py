"""
RENIX Integrations - Authentication
===================================

Central authentication and credential-management layer for
external RENIX integrations.

Supported authentication styles:
- API keys
- Bearer tokens
- Basic authentication
- OAuth-style access/refresh tokens
- Custom headers
- Query-parameter credentials

Security notes:
- Secrets are never intentionally included in repr(), status(),
  or normal debug output.
- Tokens can expire and be refreshed through registered callbacks.
- Credentials are kept in memory only by this module.
- Persistent secret storage should be handled by
  security/secrets_manager.py.
"""

from __future__ import annotations

import base64
import hashlib
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================


class AuthenticationError(Exception):
    """Base authentication exception."""


class CredentialNotFoundError(AuthenticationError):
    """Raised when requested credentials do not exist."""


class CredentialExpiredError(AuthenticationError):
    """Raised when credentials have expired."""


class InvalidCredentialError(AuthenticationError):
    """Raised when supplied credentials are invalid."""


class TokenRefreshError(AuthenticationError):
    """Raised when a token refresh operation fails."""


# ============================================================
# ENUMS
# ============================================================


class AuthenticationType(str, Enum):
    """Supported authentication mechanisms."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"
    QUERY_PARAMETER = "query_parameter"


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class Credential:
    """
    Stores authentication information for an integration.

    The actual secret values are kept inside `secret_data`.
    """

    name: str
    auth_type: AuthenticationType

    secret_data: dict[str, Any] = field(
        default_factory=dict
    )

    expires_at: float | None = None

    refresh_token: str | None = None

    refresh_expires_at: float | None = None

    enabled: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    credential_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    def is_expired(
        self,
        *,
        safety_window: float = 0.0,
    ) -> bool:
        """Return whether the credential has expired."""

        if self.expires_at is None:
            return False

        return (
            time.time()
            >= self.expires_at
            - max(0.0, safety_window)
        )

    def has_refresh_token(self) -> bool:
        """Return whether a refresh token exists."""

        return bool(self.refresh_token)

    def refresh_token_expired(self) -> bool:
        """Return whether the refresh token has expired."""

        if self.refresh_expires_at is None:
            return False

        return (
            time.time()
            >= self.refresh_expires_at
        )

    def touch(self) -> None:
        """Update modification timestamp."""

        self.updated_at = time.time()

    def public_dict(self) -> dict[str, Any]:
        """
        Return credential metadata without secret values.
        """

        return {
            "credential_id": self.credential_id,
            "name": self.name,
            "auth_type": self.auth_type.value,
            "enabled": self.enabled,
            "expires_at": self.expires_at,
            "refresh_expires_at": (
                self.refresh_expires_at
            ),
            "has_refresh_token": (
                self.has_refresh_token()
            ),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class AuthenticationResult:
    """Result of an authentication operation."""

    success: bool

    credential_name: str | None = None

    auth_type: AuthenticationType = (
        AuthenticationType.NONE
    )

    headers: dict[str, str] = field(
        default_factory=dict
    )

    query_params: dict[str, str] = field(
        default_factory=dict
    )

    username: str | None = None

    password: str | None = None

    expires_at: float | None = None

    error: str | None = None

    request_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    def to_dict(
        self,
        *,
        include_secrets: bool = False,
    ) -> dict[str, Any]:
        """
        Serialize the result.

        By default, secret headers/query values are masked.
        """

        if include_secrets:
            headers = dict(self.headers)
            query_params = dict(
                self.query_params
            )
            password = self.password

        else:
            headers = {
                key: "***"
                for key in self.headers
            }

            query_params = {
                key: "***"
                for key in self.query_params
            }

            password = (
                "***"
                if self.password
                else None
            )

        return {
            "success": self.success,
            "credential_name": (
                self.credential_name
            ),
            "auth_type": self.auth_type.value,
            "headers": headers,
            "query_params": query_params,
            "username": self.username,
            "password": password,
            "expires_at": self.expires_at,
            "error": self.error,
            "request_id": self.request_id,
        }


RefreshCallback = Callable[
    [Credential],
    Mapping[str, Any],
]


# ============================================================
# AUTHENTICATION MANAGER
# ============================================================


class AuthenticationManager:
    """
    Central authentication manager for RENIX integrations.

    Example:

        auth = AuthenticationManager()

        auth.register_api_key(
            "weather",
            api_key="YOUR_KEY",
            header_name="X-API-Key",
        )

        result = auth.authenticate(
            "weather"
        )

        headers = result.headers
    """

    SENSITIVE_KEYS = {
        "authorization",
        "api_key",
        "apikey",
        "api-key",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "client_secret",
        "key",
    }

    def __init__(
        self,
        *,
        token_safety_window: float = 30.0,
    ) -> None:
        self.token_safety_window = max(
            0.0,
            token_safety_window,
        )

        self._credentials: dict[
            str,
            Credential,
        ] = {}

        self._refresh_callbacks: dict[
            str,
            RefreshCallback,
        ] = {}

        self._lock = threading.RLock()

    # ========================================================
    # REGISTRATION
    # ========================================================

    def register(
        self,
        credential: Credential,
    ) -> str:
        """Register a credential."""

        if not isinstance(
            credential,
            Credential,
        ):
            raise TypeError(
                "credential must be a Credential."
            )

        if not credential.name.strip():
            raise ValueError(
                "Credential name cannot be empty."
            )

        with self._lock:
            self._credentials[
                credential.name
            ] = credential

        return credential.credential_id

    def unregister(
        self,
        name: str,
    ) -> bool:
        """Remove a credential."""

        with self._lock:
            self._refresh_callbacks.pop(
                name,
                None,
            )

            return (
                self._credentials.pop(
                    name,
                    None,
                )
                is not None
            )

    def get(
        self,
        name: str,
    ) -> Credential:
        """Return a credential."""

        with self._lock:
            credential = self._credentials.get(
                name
            )

        if credential is None:
            raise CredentialNotFoundError(
                f"Credential not found: {name}"
            )

        return credential

    def exists(
        self,
        name: str,
    ) -> bool:
        """Return whether a credential exists."""

        with self._lock:
            return name in self._credentials

    # ========================================================
    # API KEY
    # ========================================================

    def register_api_key(
        self,
        name: str,
        *,
        api_key: str,
        header_name: str = "X-API-Key",
        expires_at: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register an API-key credential."""

        if not api_key:
            raise InvalidCredentialError(
                "API key cannot be empty."
            )

        credential = Credential(
            name=name,
            auth_type=(
                AuthenticationType.API_KEY
            ),
            secret_data={
                "api_key": api_key,
                "header_name": header_name,
            },
            expires_at=expires_at,
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(
            credential
        )

    # ========================================================
    # BEARER TOKEN
    # ========================================================

    def register_bearer_token(
        self,
        name: str,
        *,
        token: str,
        expires_at: float | None = None,
        refresh_token: str | None = None,
        refresh_expires_at: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register a bearer/access token."""

        if not token:
            raise InvalidCredentialError(
                "Bearer token cannot be empty."
            )

        credential = Credential(
            name=name,
            auth_type=(
                AuthenticationType.BEARER
            ),
            secret_data={
                "token": token,
            },
            expires_at=expires_at,
            refresh_token=refresh_token,
            refresh_expires_at=(
                refresh_expires_at
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(
            credential
        )

    # ========================================================
    # BASIC AUTH
    # ========================================================

    def register_basic(
        self,
        name: str,
        *,
        username: str,
        password: str,
        expires_at: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register HTTP Basic authentication."""

        if not username:
            raise InvalidCredentialError(
                "Username cannot be empty."
            )

        if not password:
            raise InvalidCredentialError(
                "Password cannot be empty."
            )

        credential = Credential(
            name=name,
            auth_type=(
                AuthenticationType.BASIC
            ),
            secret_data={
                "username": username,
                "password": password,
            },
            expires_at=expires_at,
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(
            credential
        )

    # ========================================================
    # OAUTH2
    # ========================================================

    def register_oauth2(
        self,
        name: str,
        *,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: float | None = None,
        refresh_expires_at: float | None = None,
        token_type: str = "Bearer",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register OAuth-style token credentials."""

        if not access_token:
            raise InvalidCredentialError(
                "Access token cannot be empty."
            )

        credential = Credential(
            name=name,
            auth_type=(
                AuthenticationType.OAUTH2
            ),
            secret_data={
                "access_token": access_token,
                "token_type": token_type,
            },
            expires_at=expires_at,
            refresh_token=refresh_token,
            refresh_expires_at=(
                refresh_expires_at
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(
            credential
        )

    # ========================================================
    # CUSTOM HEADER
    # ========================================================

    def register_custom_header(
        self,
        name: str,
        *,
        headers: Mapping[str, str],
        expires_at: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register custom authentication headers."""

        if not headers:
            raise InvalidCredentialError(
                "At least one header is required."
            )

        credential = Credential(
            name=name,
            auth_type=(
                AuthenticationType.CUSTOM_HEADER
            ),
            secret_data={
                "headers": {
                    str(key): str(value)
                    for key, value in headers.items()
                }
            },
            expires_at=expires_at,
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(
            credential
        )

    # ========================================================
    # QUERY PARAMETER
    # ========================================================

    def register_query_parameter(
        self,
        name: str,
        *,
        parameter_name: str,
        value: str,
        expires_at: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register query-parameter authentication."""

        if not parameter_name:
            raise InvalidCredentialError(
                "Parameter name cannot be empty."
            )

        if not value:
            raise InvalidCredentialError(
                "Parameter value cannot be empty."
            )

        credential = Credential(
            name=name,
            auth_type=(
                AuthenticationType.QUERY_PARAMETER
            ),
            secret_data={
                "parameter_name": parameter_name,
                "value": value,
            },
            expires_at=expires_at,
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(
            credential
        )

    # ========================================================
    # REFRESH CALLBACKS
    # ========================================================

    def register_refresh_callback(
        self,
        credential_name: str,
        callback: RefreshCallback,
    ) -> None:
        """
        Register a callback used to refresh an expired token.

        Callback should return values such as:

            {
                "access_token": "...",
                "expires_at": 1234567890,
                "refresh_token": "..."
            }
        """

        if not callable(callback):
            raise TypeError(
                "callback must be callable."
            )

        self.get(
            credential_name
        )

        with self._lock:
            self._refresh_callbacks[
                credential_name
            ] = callback

    def remove_refresh_callback(
        self,
        credential_name: str,
    ) -> bool:
        """Remove a refresh callback."""

        with self._lock:
            return (
                self._refresh_callbacks.pop(
                    credential_name,
                    None,
                )
                is not None
            )

    # ========================================================
    # AUTHENTICATE
    # ========================================================

    def authenticate(
        self,
        name: str,
        *,
        auto_refresh: bool = True,
    ) -> AuthenticationResult:
        """
        Generate authentication material for a credential.
        """

        credential = self.get(name)

        if not credential.enabled:
            return AuthenticationResult(
                success=False,
                credential_name=name,
                auth_type=credential.auth_type,
                error=(
                    "Credential is disabled."
                ),
            )

        if credential.is_expired(
            safety_window=(
                self.token_safety_window
            ),
        ):
            if (
                auto_refresh
                and credential.has_refresh_token()
            ):
                try:
                    self.refresh(name)
                    credential = self.get(
                        name
                    )

                except AuthenticationError as exc:
                    return AuthenticationResult(
                        success=False,
                        credential_name=name,
                        auth_type=(
                            credential.auth_type
                        ),
                        error=str(exc),
                    )

            else:
                return AuthenticationResult(
                    success=False,
                    credential_name=name,
                    auth_type=credential.auth_type,
                    expires_at=(
                        credential.expires_at
                    ),
                    error=(
                        "Credential has expired."
                    ),
                )

        try:
            return self._build_auth_result(
                credential
            )

        except Exception as exc:
            return AuthenticationResult(
                success=False,
                credential_name=name,
                auth_type=credential.auth_type,
                error=str(exc),
            )

    # ========================================================
    # AUTH BUILDERS
    # ========================================================

    def _build_auth_result(
        self,
        credential: Credential,
    ) -> AuthenticationResult:
        """Build authentication material."""

        auth_type = credential.auth_type
        data = credential.secret_data

        if auth_type == AuthenticationType.NONE:
            return AuthenticationResult(
                success=True,
                credential_name=credential.name,
                auth_type=auth_type,
                expires_at=(
                    credential.expires_at
                ),
            )

        if auth_type == AuthenticationType.API_KEY:
            header_name = str(
                data.get(
                    "header_name",
                    "X-API-Key",
                )
            )

            api_key = str(
                data.get(
                    "api_key",
                    "",
                )
            )

            if not api_key:
                raise InvalidCredentialError(
                    "API key is missing."
                )

            return AuthenticationResult(
                success=True,
                credential_name=credential.name,
                auth_type=auth_type,
                headers={
                    header_name: api_key,
                },
                expires_at=(
                    credential.expires_at
                ),
            )

        if auth_type == AuthenticationType.BEARER:
            token = str(
                data.get(
                    "token",
                    "",
                )
            )

            if not token:
                raise InvalidCredentialError(
                    "Bearer token is missing."
                )

            return AuthenticationResult(
                success=True,
                credential_name=credential.name,
                auth_type=auth_type,
                headers={
                    "Authorization": (
                        f"Bearer {token}"
                    )
                },
                expires_at=(
                    credential.expires_at
                ),
            )

        if auth_type == AuthenticationType.BASIC:
            username = str(
                data.get(
                    "username",
                    "",
                )
            )

            password = str(
                data.get(
                    "password",
                    "",
                )
            )

            if not username or not password:
                raise InvalidCredentialError(
                    "Basic authentication "
                    "credentials are incomplete."
                )

            encoded = base64.b64encode(
                f"{username}:{password}".encode(
                    "utf-8"
                )
            ).decode(
                "ascii"
            )

            return AuthenticationResult(
                success=True,
                credential_name=credential.name,
                auth_type=auth_type,
                headers={
                    "Authorization": (
                        f"Basic {encoded}"
                    )
                },
                username=username,
                password=password,
                expires_at=(
                    credential.expires_at
                ),
            )

        if auth_type == AuthenticationType.OAUTH2:
            access_token = str(
                data.get(
                    "access_token",
                    "",
                )
            )

            token_type = str(
                data.get(
                    "token_type",
                    "Bearer",
                )
            )

            if not access_token:
                raise InvalidCredentialError(
                    "OAuth access token is missing."
                )

            return AuthenticationResult(
                success=True,
                credential_name=credential.name,
                auth_type=auth_type,
                headers={
                    "Authorization": (
                        f"{token_type} "
                        f"{access_token}"
                    )
                },
                expires_at=(
                    credential.expires_at
                ),
            )

        if auth_type == AuthenticationType.CUSTOM_HEADER:
            headers = data.get(
                "headers",
                {},
            )

            if not isinstance(
                headers,
                Mapping,
            ):
                raise InvalidCredentialError(
                    "Custom headers must be a mapping."
                )

            return AuthenticationResult(
                success=True,
                credential_name=credential.name,
                auth_type=auth_type,
                headers={
                    str(key): str(value)
                    for key, value in headers.items()
                },
                expires_at=(
                    credential.expires_at
                ),
            )

        if auth_type == AuthenticationType.QUERY_PARAMETER:
            parameter_name = str(
                data.get(
                    "parameter_name",
                    "",
                )
            )

            value = str(
                data.get(
                    "value",
                    "",
                )
            )

            if not parameter_name or not value:
                raise InvalidCredentialError(
                    "Query authentication "
                    "configuration is incomplete."
                )

            return AuthenticationResult(
                success=True,
                credential_name=credential.name,
                auth_type=auth_type,
                query_params={
                    parameter_name: value,
                },
                expires_at=(
                    credential.expires_at
                ),
            )

        raise InvalidCredentialError(
            f"Unsupported authentication type: "
            f"{auth_type}"
        )

    # ========================================================
    # REFRESH
    # ========================================================

    def refresh(
        self,
        name: str,
    ) -> Credential:
        """
        Refresh a credential using its registered callback.
        """

        credential = self.get(name)

        if not credential.has_refresh_token():
            raise TokenRefreshError(
                f"Credential '{name}' has no refresh token."
            )

        if credential.refresh_token_expired():
            raise TokenRefreshError(
                f"Refresh token for '{name}' has expired."
            )

        with self._lock:
            callback = (
                self._refresh_callbacks.get(
                    name
                )
            )

        if callback is None:
            raise TokenRefreshError(
                f"No refresh callback registered "
                f"for '{name}'."
            )

        try:
            result = callback(
                credential
            )

        except Exception as exc:
            raise TokenRefreshError(
                f"Token refresh failed for "
                f"'{name}': {exc}"
            ) from exc

        if not isinstance(
            result,
            Mapping,
        ):
            raise TokenRefreshError(
                "Refresh callback must return "
                "a mapping."
            )

        self._apply_refresh_result(
            credential,
            result,
        )

        return credential

    def _apply_refresh_result(
        self,
        credential: Credential,
        result: Mapping[str, Any],
    ) -> None:
        """Apply refreshed token information."""

        access_token = result.get(
            "access_token"
        )

        token = result.get(
            "token"
        )

        if access_token:
            credential.secret_data[
                "access_token"
            ] = str(
                access_token
            )

        elif token:
            credential.secret_data[
                "token"
            ] = str(
                token
            )

        new_refresh_token = result.get(
            "refresh_token"
        )

        if new_refresh_token:
            credential.refresh_token = str(
                new_refresh_token
            )

        if "expires_at" in result:
            credential.expires_at = (
                self._normalize_expiry(
                    result["expires_at"]
                )
            )

        if "refresh_expires_at" in result:
            credential.refresh_expires_at = (
                self._normalize_expiry(
                    result["refresh_expires_at"]
                )
            )

        if "token_type" in result:
            credential.secret_data[
                "token_type"
            ] = str(
                result["token_type"]
            )

        credential.touch()

    # ========================================================
    # TOKEN UTILITIES
    # ========================================================

    @staticmethod
    def _normalize_expiry(
        value: Any,
    ) -> float | None:
        """Normalize expiry values to Unix timestamps."""

        if value is None:
            return None

        if isinstance(
            value,
            datetime,
        ):
            if value.tzinfo is None:
                value = value.replace(
                    tzinfo=timezone.utc
                )

            return value.timestamp()

        if isinstance(
            value,
            (int, float),
        ):
            return float(value)

        if isinstance(
            value,
            str,
        ):
            try:
                return float(value)
            except ValueError:
                pass

            try:
                parsed = datetime.fromisoformat(
                    value
                )

                if parsed.tzinfo is None:
                    parsed = parsed.replace(
                        tzinfo=timezone.utc
                    )

                return parsed.timestamp()

            except ValueError as exc:
                raise ValueError(
                    f"Invalid expiry value: {value!r}"
                ) from exc

        raise ValueError(
            f"Unsupported expiry type: "
            f"{type(value).__name__}"
        )

    def token_fingerprint(
        self,
        name: str,
    ) -> str:
        """
        Return a non-reversible fingerprint of the
        current token.

        Useful for detecting token changes without exposing
        the token itself.
        """

        credential = self.get(name)

        values = []

        for key in (
            "token",
            "access_token",
            "api_key",
        ):
            value = credential.secret_data.get(
                key
            )

            if value:
                values.append(
                    str(value)
                )

        if not values:
            return ""

        digest = hashlib.sha256()

        for value in values:
            digest.update(
                value.encode("utf-8")
            )
            digest.update(
                b"\0"
            )

        return digest.hexdigest()

    # ========================================================
    # HEADER / QUERY HELPERS
    # ========================================================

    def get_headers(
        self,
        name: str,
        *,
        auto_refresh: bool = True,
    ) -> dict[str, str]:
        """Return authentication headers."""

        result = self.authenticate(
            name,
            auto_refresh=auto_refresh,
        )

        if not result.success:
            raise AuthenticationError(
                result.error
                or "Authentication failed."
            )

        return dict(
            result.headers
        )

    def get_query_params(
        self,
        name: str,
        *,
        auto_refresh: bool = True,
    ) -> dict[str, str]:
        """Return authentication query parameters."""

        result = self.authenticate(
            name,
            auto_refresh=auto_refresh,
        )

        if not result.success:
            raise AuthenticationError(
                result.error
                or "Authentication failed."
            )

        return dict(
            result.query_params
        )

    def apply(
        self,
        name: str,
        *,
        headers: Mapping[str, str] | None = None,
        query_params: Mapping[str, str] | None = None,
        auto_refresh: bool = True,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
    ]:
        """
        Apply authentication to existing headers and query
        parameters.
        """

        result = self.authenticate(
            name,
            auto_refresh=auto_refresh,
        )

        if not result.success:
            raise AuthenticationError(
                result.error
                or "Authentication failed."
            )

        final_headers = dict(
            headers or {}
        )

        final_query = dict(
            query_params or {}
        )

        final_headers.update(
            result.headers
        )

        final_query.update(
            result.query_params
        )

        return (
            final_headers,
            final_query,
        )

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(
        self,
        name: str,
    ) -> None:
        """Enable a credential."""

        credential = self.get(name)
        credential.enabled = True
        credential.touch()

    def disable(
        self,
        name: str,
    ) -> None:
        """Disable a credential."""

        credential = self.get(name)
        credential.enabled = False
        credential.touch()

    # ========================================================
    # STATUS
    # ========================================================

    def status(
        self,
    ) -> dict[str, Any]:
        """
        Return non-sensitive authentication status.

        Secret values are never returned.
        """

        with self._lock:
            credentials = list(
                self._credentials.values()
            )

        return {
            "credential_count": len(
                credentials
            ),
            "credentials": [
                credential.public_dict()
                for credential in credentials
            ],
            "refresh_callbacks": len(
                self._refresh_callbacks
            ),
            "token_safety_window": (
                self.token_safety_window
            ),
        }

    def list_credentials(
        self,
    ) -> list[dict[str, Any]]:
        """List credential metadata without secrets."""

        with self._lock:
            return [
                credential.public_dict()
                for credential in (
                    self._credentials.values()
                )
            ]

    def clear(
        self,
    ) -> None:
        """
        Remove all credentials and refresh callbacks
        from memory.
        """

        with self._lock:
            self._credentials.clear()
            self._refresh_callbacks.clear()

    def __len__(self) -> int:
        """Return number of registered credentials."""

        return len(self._credentials)

    def __contains__(
        self,
        name: str,
    ) -> bool:
        return self.exists(name)

    def __repr__(self) -> str:
        return (
            "AuthenticationManager("
            f"credentials={len(self._credentials)})"
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def mask_secret(
    value: Any,
    *,
    visible_characters: int = 4,
) -> str:
    """
    Safely mask a secret for logs or UI.

    Example:
        abcdefghijkl -> ********ijkl
    """

    if value is None:
        return ""

    text = str(value)

    if not text:
        return ""

    visible = max(
        0,
        int(visible_characters),
    )

    if visible == 0:
        return "*" * len(text)

    if len(text) <= visible:
        return "*" * len(text)

    return (
        "*" * (len(text) - visible)
        + text[-visible:]
    )


def sanitize_headers(
    headers: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Remove or mask sensitive header values.

    Intended for logging/debugging only.
    """

    sanitized: dict[str, Any] = {}

    for key, value in headers.items():
        normalized = str(key).lower()

        if normalized in {
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "api-key",
            "x-auth-token",
        }:
            sanitized[str(key)] = "***"
        else:
            sanitized[str(key)] = value

    return sanitized


__all__ = [
    "AuthenticationType",
    "Credential",
    "AuthenticationResult",
    "AuthenticationManager",
    "AuthenticationError",
    "CredentialNotFoundError",
    "CredentialExpiredError",
    "InvalidCredentialError",
    "TokenRefreshError",
    "mask_secret",
    "sanitize_headers",
]


