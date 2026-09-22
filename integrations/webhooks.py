"""
RENIX Integrations - Webhooks
=============================

Webhook management for RENIX.

Responsibilities:
- Register webhook endpoints.
- Generate and validate webhook IDs.
- Register event-specific handlers.
- Dispatch incoming webhook events.
- Verify webhook signatures when configured.
- Prevent accidental duplicate event processing.
- Track webhook execution statistics.
- Enable/disable individual webhooks.
- Provide safe webhook status information.

Security notes:
- Never log raw secrets.
- Never expose webhook signing secrets through status().
- Production deployments should use HTTPS.
- Signature verification should be enabled for external
  services whenever the provider supports it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================


class WebhookError(Exception):
    """Base webhook exception."""


class WebhookNotFoundError(WebhookError):
    """Raised when a webhook cannot be found."""


class WebhookAlreadyExistsError(WebhookError):
    """Raised when a webhook already exists."""


class WebhookDisabledError(WebhookError):
    """Raised when a disabled webhook receives an event."""


class WebhookVerificationError(WebhookError):
    """Raised when webhook signature verification fails."""


class WebhookHandlerError(WebhookError):
    """Raised when a webhook handler fails."""


class DuplicateWebhookEventError(WebhookError):
    """Raised when an already processed event is received."""


# ============================================================
# ENUMS
# ============================================================


class WebhookStatus(str, Enum):
    """Webhook lifecycle state."""

    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class SignatureAlgorithm(str, Enum):
    """Supported webhook signature algorithms."""

    SHA256 = "sha256"
    SHA512 = "sha512"


# ============================================================
# TYPE ALIASES
# ============================================================


WebhookHandler = Callable[
    ["WebhookEvent"],
    Any,
]


# ============================================================
# WEBHOOK EVENT
# ============================================================


@dataclass
class WebhookEvent:
    """
    Normalized webhook event.

    The payload is kept as a Python dictionary when possible,
    allowing RENIX handlers to process events consistently.
    """

    event_type: str

    payload: dict[str, Any]

    webhook_id: str

    event_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    source: str | None = None

    timestamp: float = field(
        default_factory=time.time
    )

    headers: dict[str, str] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert event to a serializable dictionary."""

        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "webhook_id": self.webhook_id,
            "source": self.source,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
            "headers": dict(self.headers),
            "metadata": dict(self.metadata),
        }


# ============================================================
# WEBHOOK STATISTICS
# ============================================================


@dataclass
class WebhookStatistics:
    """Runtime statistics for a webhook."""

    total_received: int = 0

    total_processed: int = 0

    total_failed: int = 0

    total_duplicates: int = 0

    total_rejected: int = 0

    last_received: float | None = None

    last_processed: float | None = None

    last_failed: float | None = None

    last_event_type: str | None = None

    last_error: str | None = None

    average_processing_ms: float = 0.0

    def record_received(
        self,
        event_type: str,
    ) -> None:
        """Record an incoming webhook event."""

        self.total_received += 1
        self.last_received = time.time()
        self.last_event_type = event_type

    def record_processed(
        self,
        processing_ms: float,
    ) -> None:
        """Record successful processing."""

        self.total_processed += 1
        self.last_processed = time.time()

        count = self.total_processed

        self.average_processing_ms = (
            (
                self.average_processing_ms
                * (count - 1)
            )
            + processing_ms
        ) / count

        self.last_error = None

    def record_failure(
        self,
        error: str,
    ) -> None:
        """Record failed processing."""

        self.total_failed += 1
        self.last_failed = time.time()
        self.last_error = str(error)

    def record_duplicate(self) -> None:
        """Record duplicate event."""

        self.total_duplicates += 1

    def record_rejected(self) -> None:
        """Record rejected event."""

        self.total_rejected += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistics."""

        return {
            "total_received": self.total_received,
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "total_duplicates": self.total_duplicates,
            "total_rejected": self.total_rejected,
            "last_received": self.last_received,
            "last_processed": self.last_processed,
            "last_failed": self.last_failed,
            "last_event_type": self.last_event_type,
            "last_error": self.last_error,
            "average_processing_ms": (
                self.average_processing_ms
            ),
        }


# ============================================================
# WEBHOOK MODEL
# ============================================================


@dataclass
class Webhook:
    """
    Registered RENIX webhook.

    Example:

        Webhook(
            name="github",
            source="github",
            secret="...",
        )
    """

    name: str

    source: str | None = None

    description: str = ""

    secret: str | None = None

    signature_algorithm: SignatureAlgorithm = (
        SignatureAlgorithm.SHA256
    )

    signature_header: str = "X-Signature-256"

    enabled: bool = True

    status: WebhookStatus = WebhookStatus.ACTIVE

    accepted_events: set[str] = field(
        default_factory=set
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    statistics: WebhookStatistics = field(
        default_factory=WebhookStatistics
    )

    webhook_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    def __post_init__(self) -> None:
        self.name = self.name.strip()

        if self.source:
            self.source = self.source.strip()

        self.accepted_events = {
            str(event).strip()
            for event in self.accepted_events
            if str(event).strip()
        }

    def touch(self) -> None:
        """Update modification timestamp."""

        self.updated_at = time.time()

    def accepts_event(
        self,
        event_type: str,
    ) -> bool:
        """
        Determine whether an event type is accepted.

        An empty accepted_events set means all event types
        are accepted.
        """

        if not self.accepted_events:
            return True

        return event_type in self.accepted_events

    def public_dict(self) -> dict[str, Any]:
        """
        Return webhook information without exposing secrets.
        """

        return {
            "webhook_id": self.webhook_id,
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "signature_algorithm": (
                self.signature_algorithm.value
            ),
            "signature_header": (
                self.signature_header
            ),
            "enabled": self.enabled,
            "status": self.status.value,
            "accepted_events": sorted(
                self.accepted_events
            ),
            "metadata": dict(self.metadata),
            "statistics": self.statistics.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ============================================================
# WEBHOOK MANAGER
# ============================================================


class WebhookManager:
    """
    Central webhook registry and event dispatcher.

    Example:

        manager = WebhookManager()

        manager.register(
            Webhook(
                name="github",
                source="github",
                secret="secret",
            )
        )

        manager.register_handler(
            "github",
            "push",
            handle_push,
        )

        manager.process(
            "github",
            event_type="push",
            payload={"repository": "RENIX"},
        )
    """

    def __init__(
        self,
        *,
        prevent_duplicates: bool = True,
        max_processed_events: int = 10000,
    ) -> None:
        self.prevent_duplicates = (
            prevent_duplicates
        )

        self.max_processed_events = max(
            100,
            int(max_processed_events),
        )

        self._webhooks: dict[
            str,
            Webhook,
        ] = {}

        self._handlers: dict[
            str,
            dict[str, list[WebhookHandler]],
        ] = {}

        self._processed_events: dict[
            str,
            float,
        ] = {}

        self._lock = threading.RLock()

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """Normalize a webhook name."""

        if not isinstance(name, str):
            raise TypeError(
                "Webhook name must be a string."
            )

        value = name.strip()

        if not value:
            raise ValueError(
                "Webhook name cannot be empty."
            )

        return value

    # ========================================================
    # REGISTRATION
    # ========================================================

    def register(
        self,
        webhook: Webhook,
        *,
        replace: bool = False,
    ) -> str:
        """Register a webhook."""

        if not isinstance(
            webhook,
            Webhook,
        ):
            raise TypeError(
                "webhook must be a Webhook instance."
            )

        webhook.name = self._normalize_name(
            webhook.name
        )

        with self._lock:
            if (
                webhook.name in self._webhooks
                and not replace
            ):
                raise WebhookAlreadyExistsError(
                    f"Webhook already exists: "
                    f"{webhook.name}"
                )

            self._webhooks[
                webhook.name
            ] = webhook

            self._handlers.setdefault(
                webhook.name,
                {},
            )

        return webhook.webhook_id

    def register_simple(
        self,
        name: str,
        *,
        source: str | None = None,
        secret: str | None = None,
        signature_algorithm: SignatureAlgorithm = (
            SignatureAlgorithm.SHA256
        ),
        signature_header: str = "X-Signature-256",
        accepted_events: set[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Convenience method for creating a webhook."""

        webhook = Webhook(
            name=name,
            source=source,
            secret=secret,
            signature_algorithm=(
                signature_algorithm
            ),
            signature_header=signature_header,
            accepted_events=set(
                accepted_events or set()
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(webhook)

    # ========================================================
    # LOOKUP
    # ========================================================

    def get(
        self,
        name: str,
    ) -> Webhook:
        """Retrieve a webhook."""

        name = str(name).strip()

        with self._lock:
            webhook = self._webhooks.get(
                name
            )

        if webhook is None:
            raise WebhookNotFoundError(
                f"Webhook not found: {name}"
            )

        return webhook

    def get_optional(
        self,
        name: str,
    ) -> Webhook | None:
        """Return webhook or None."""

        try:
            return self.get(name)
        except WebhookNotFoundError:
            return None

    def exists(
        self,
        name: str,
    ) -> bool:
        """Check whether webhook exists."""

        return self.get_optional(name) is not None

    def unregister(
        self,
        name: str,
    ) -> bool:
        """Remove a webhook."""

        name = str(name).strip()

        with self._lock:
            if name not in self._webhooks:
                return False

            self._webhooks.pop(
                name,
                None,
            )

            self._handlers.pop(
                name,
                None,
            )

            return True

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(
        self,
        name: str,
    ) -> Webhook:
        """Enable a webhook."""

        webhook = self.get(name)

        webhook.enabled = True
        webhook.status = WebhookStatus.ACTIVE
        webhook.touch()

        return webhook

    def disable(
        self,
        name: str,
    ) -> Webhook:
        """Disable a webhook."""

        webhook = self.get(name)

        webhook.enabled = False
        webhook.status = WebhookStatus.DISABLED
        webhook.touch()

        return webhook

    # ========================================================
    # HANDLERS
    # ========================================================

    def register_handler(
        self,
        webhook_name: str,
        event_type: str,
        handler: WebhookHandler,
    ) -> None:
        """
        Register a handler for a specific event type.
        """

        webhook = self.get(webhook_name)

        if not callable(handler):
            raise TypeError(
                "handler must be callable."
            )

        event_type = str(
            event_type
        ).strip()

        if not event_type:
            raise ValueError(
                "event_type cannot be empty."
            )

        with self._lock:
            handlers = self._handlers.setdefault(
                webhook.name,
                {},
            )

            handlers.setdefault(
                event_type,
                [],
            ).append(handler)

        webhook.touch()

    def unregister_handler(
        self,
        webhook_name: str,
        event_type: str,
        handler: WebhookHandler,
    ) -> bool:
        """Remove a specific event handler."""

        webhook = self.get(webhook_name)

        with self._lock:
            handlers = self._handlers.get(
                webhook.name,
                {},
            )

            event_handlers = handlers.get(
                event_type,
                [],
            )

            try:
                event_handlers.remove(
                    handler
                )
            except ValueError:
                return False

            if not event_handlers:
                handlers.pop(
                    event_type,
                    None,
                )

        webhook.touch()

        return True

    def get_handlers(
        self,
        webhook_name: str,
        event_type: str,
    ) -> list[WebhookHandler]:
        """Return handlers for an event."""

        webhook = self.get(webhook_name)

        with self._lock:
            return list(
                self._handlers
                .get(webhook.name, {})
                .get(event_type, [])
            )

    # ========================================================
    # SIGNATURE VERIFICATION
    # ========================================================

    @staticmethod
    def _payload_bytes(
        payload: bytes | str | Mapping[str, Any],
    ) -> bytes:
        """Convert a payload into canonical bytes."""

        if isinstance(payload, bytes):
            return payload

        if isinstance(payload, str):
            return payload.encode("utf-8")

        return json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @staticmethod
    def _extract_signature(
        signature: str,
    ) -> str:
        """
        Normalize common signature formats.

        Supports:
            sha256=<digest>
            sha512=<digest>
            <digest>
        """

        signature = (
            str(signature)
            .strip()
        )

        if "=" in signature:
            prefix, digest = signature.split(
                "=",
                1,
            )

            if prefix.lower() in {
                "sha256",
                "sha512",
                "hmac-sha256",
                "hmac-sha512",
            }:
                return digest.strip()

        return signature

    def generate_signature(
        self,
        webhook_name: str,
        payload: bytes | str | Mapping[str, Any],
    ) -> str:
        """
        Generate a webhook signature using the webhook secret.
        """

        webhook = self.get(webhook_name)

        if not webhook.secret:
            raise WebhookVerificationError(
                "Webhook has no signing secret."
            )

        payload_bytes = self._payload_bytes(
            payload
        )

        if (
            webhook.signature_algorithm
            == SignatureAlgorithm.SHA512
        ):
            digest = hmac.new(
                webhook.secret.encode(
                    "utf-8"
                ),
                payload_bytes,
                hashlib.sha512,
            ).hexdigest()
        else:
            digest = hmac.new(
                webhook.secret.encode(
                    "utf-8"
                ),
                payload_bytes,
                hashlib.sha256,
            ).hexdigest()

        return digest

    def verify_signature(
        self,
        webhook_name: str,
        payload: bytes | str | Mapping[str, Any],
        signature: str,
    ) -> bool:
        """Verify an incoming webhook signature."""

        webhook = self.get(webhook_name)

        if not webhook.secret:
            raise WebhookVerificationError(
                "Webhook has no signing secret."
            )

        expected = self.generate_signature(
            webhook_name,
            payload,
        )

        supplied = self._extract_signature(
            signature
        )

        valid = hmac.compare_digest(
            expected,
            supplied,
        )

        if not valid:
            webhook.statistics.record_rejected()

        return valid

    # ========================================================
    # DUPLICATE EVENT PROTECTION
    # ========================================================

    def _is_duplicate(
        self,
        event_id: str,
    ) -> bool:
        """Check whether an event was processed before."""

        if not self.prevent_duplicates:
            return False

        with self._lock:
            return event_id in self._processed_events

    def _remember_event(
        self,
        event_id: str,
    ) -> None:
        """Remember a successfully processed event."""

        if not self.prevent_duplicates:
            return

        with self._lock:
            self._processed_events[
                event_id
            ] = time.time()

            while (
                len(self._processed_events)
                > self.max_processed_events
            ):
                oldest = min(
                    self._processed_events,
                    key=self._processed_events.get,
                )

                self._processed_events.pop(
                    oldest,
                    None,
                )

    # ========================================================
    # EVENT CREATION
    # ========================================================

    def create_event(
        self,
        webhook_name: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        event_id: str | None = None,
        source: str | None = None,
        headers: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WebhookEvent:
        """Create a normalized RENIX webhook event."""

        webhook = self.get(webhook_name)

        event_type = str(
            event_type
        ).strip()

        if not event_type:
            raise ValueError(
                "event_type cannot be empty."
            )

        if not webhook.accepts_event(
            event_type
        ):
            raise WebhookError(
                f"Webhook '{webhook.name}' "
                f"does not accept event "
                f"'{event_type}'."
            )

        return WebhookEvent(
            event_type=event_type,
            payload=dict(
                payload or {}
            ),
            webhook_id=webhook.webhook_id,
            event_id=(
                event_id
                or uuid.uuid4().hex
            ),
            source=(
                source
                or webhook.source
            ),
            headers={
                str(k): str(v)
                for k, v in (
                    headers or {}
                ).items()
            },
            metadata=dict(
                metadata or {}
            ),
        )

    # ========================================================
    # PROCESS EVENT
    # ========================================================

    def process_event(
        self,
        webhook_name: str,
        event: WebhookEvent,
    ) -> list[Any]:
        """
        Dispatch a normalized webhook event to handlers.
        """

        webhook = self.get(webhook_name)

        webhook.statistics.record_received(
            event.event_type
        )

        if not webhook.enabled:
            webhook.statistics.record_rejected()

            raise WebhookDisabledError(
                f"Webhook is disabled: "
                f"{webhook.name}"
            )

        if not webhook.accepts_event(
            event.event_type
        ):
            webhook.statistics.record_rejected()

            raise WebhookError(
                f"Event '{event.event_type}' "
                f"is not accepted by "
                f"'{webhook.name}'."
            )

        if self._is_duplicate(
            event.event_id
        ):
            webhook.statistics.record_duplicate()

            raise DuplicateWebhookEventError(
                f"Duplicate event: "
                f"{event.event_id}"
            )

        handlers = self.get_handlers(
            webhook.name,
            event.event_type,
        )

        # If no exact handler exists, check wildcard handlers.
        handlers += self.get_handlers(
            webhook.name,
            "*",
        )

        if not handlers:
            # An event can still be considered received
            # even when no handler has been registered.
            webhook.statistics.record_processed(
                0.0
            )

            self._remember_event(
                event.event_id
            )

            return []

        results: list[Any] = []

        started = time.perf_counter()

        try:
            for handler in handlers:
                results.append(
                    handler(event)
                )

        except Exception as exc:
            webhook.status = WebhookStatus.ERROR
            webhook.statistics.record_failure(
                str(exc)
            )
            webhook.touch()

            raise WebhookHandlerError(
                f"Webhook handler failed: "
                f"{exc}"
            ) from exc

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000.0

        webhook.status = WebhookStatus.ACTIVE

        webhook.statistics.record_processed(
            elapsed
        )

        webhook.touch()

        self._remember_event(
            event.event_id
        )

        return results

    # ========================================================
    # SIMPLE PROCESS API
    # ========================================================

    def process(
        self,
        webhook_name: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        event_id: str | None = None,
        source: str | None = None,
        headers: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """Create and immediately process an event."""

        event = self.create_event(
            webhook_name,
            event_type,
            payload,
            event_id=event_id,
            source=source,
            headers=headers,
            metadata=metadata,
        )

        return self.process_event(
            webhook_name,
            event,
        )

    # ========================================================
    # RAW REQUEST PROCESSING
    # ========================================================

    def process_raw(
        self,
        webhook_name: str,
        event_type: str,
        payload: bytes | str | Mapping[str, Any],
        *,
        event_id: str | None = None,
        headers: Mapping[str, str] | None = None,
        source: str | None = None,
        verify_signature: bool = True,
    ) -> list[Any]:
        """
        Process a raw external webhook payload.

        If the webhook has a secret and verification is enabled,
        the appropriate signature header must be present.
        """

        webhook = self.get(webhook_name)

        normalized_headers = {
            str(key): str(value)
            for key, value in (
                headers or {}
            ).items()
        }

        if (
            verify_signature
            and webhook.secret
        ):
            signature = self._find_signature_header(
                normalized_headers,
                webhook.signature_header,
            )

            if not signature:
                webhook.statistics.record_rejected()

                raise WebhookVerificationError(
                    "Webhook signature header "
                    "is missing."
                )

            if not self.verify_signature(
                webhook.name,
                payload,
                signature,
            ):
                raise WebhookVerificationError(
                    "Invalid webhook signature."
                )

        parsed_payload = self._parse_payload(
            payload
        )

        return self.process(
            webhook.name,
            event_type,
            parsed_payload,
            event_id=event_id,
            source=source,
            headers=normalized_headers,
        )

    @staticmethod
    def _find_signature_header(
        headers: Mapping[str, str],
        expected_name: str,
    ) -> str | None:
        """Find a header case-insensitively."""

        expected = expected_name.lower()

        for key, value in headers.items():
            if key.lower() == expected:
                return value

        return None

    @staticmethod
    def _parse_payload(
        payload: bytes | str | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Parse an external webhook payload."""

        if isinstance(
            payload,
            Mapping,
        ):
            return dict(payload)

        if isinstance(
            payload,
            bytes,
        ):
            payload = payload.decode(
                "utf-8"
            )

        if not isinstance(
            payload,
            str,
        ):
            raise TypeError(
                "Webhook payload must be bytes, "
                "string, or mapping."
            )

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {
                "raw": payload
            }

        if isinstance(
            parsed,
            Mapping,
        ):
            return dict(parsed)

        return {
            "data": parsed
        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_statistics(
        self,
        name: str,
    ) -> WebhookStatistics:
        """Return webhook statistics."""

        return self.get(name).statistics

    def list_webhooks(
        self,
        *,
        active_only: bool = False,
    ) -> list[Webhook]:
        """Return registered webhooks."""

        with self._lock:
            webhooks = list(
                self._webhooks.values()
            )

        if active_only:
            webhooks = [
                webhook
                for webhook in webhooks
                if (
                    webhook.enabled
                    and webhook.status
                    == WebhookStatus.ACTIVE
                )
            ]

        return sorted(
            webhooks,
            key=lambda webhook:
                webhook.name.lower(),
        )

    def status(
        self,
    ) -> dict[str, Any]:
        """
        Return safe webhook registry status.

        Secrets are intentionally excluded.
        """

        webhooks = self.list_webhooks()

        return {
            "total": len(webhooks),
            "active": sum(
                webhook.status
                == WebhookStatus.ACTIVE
                and webhook.enabled
                for webhook in webhooks
            ),
            "disabled": sum(
                webhook.status
                == WebhookStatus.DISABLED
                or not webhook.enabled
                for webhook in webhooks
            ),
            "errors": sum(
                webhook.status
                == WebhookStatus.ERROR
                for webhook in webhooks
            ),
            "processed_event_cache_size": len(
                self._processed_events
            ),
            "webhooks": [
                webhook.public_dict()
                for webhook in webhooks
            ],
        }

    # ========================================================
    # EXPORT / IMPORT
    # ========================================================

    def export_config(
        self,
    ) -> list[dict[str, Any]]:
        """
        Export webhook configuration.

        IMPORTANT:
        Signing secrets are deliberately excluded.
        """

        configurations = []

        for webhook in self.list_webhooks():
            configurations.append(
                {
                    "name": webhook.name,
                    "source": webhook.source,
                    "description": webhook.description,
                    "signature_algorithm": (
                        webhook.signature_algorithm.value
                    ),
                    "signature_header": (
                        webhook.signature_header
                    ),
                    "enabled": webhook.enabled,
                    "accepted_events": sorted(
                        webhook.accepted_events
                    ),
                    "metadata": dict(
                        webhook.metadata
                    ),
                }
            )

        return configurations

    def import_config(
        self,
        configurations: list[
            Mapping[str, Any]
        ],
        *,
        replace: bool = False,
    ) -> list[str]:
        """
        Import webhook definitions.

        Secrets must be configured separately through the
        authentication/secrets system.
        """

        imported = []

        for configuration in configurations:
            algorithm_value = configuration.get(
                "signature_algorithm",
                SignatureAlgorithm.SHA256.value,
            )

            try:
                algorithm = SignatureAlgorithm(
                    algorithm_value
                )
            except ValueError:
                algorithm = (
                    SignatureAlgorithm.SHA256
                )

            webhook = Webhook(
                name=str(
                    configuration["name"]
                ),
                source=configuration.get(
                    "source"
                ),
                description=str(
                    configuration.get(
                        "description",
                        "",
                    )
                ),
                secret=None,
                signature_algorithm=algorithm,
                signature_header=str(
                    configuration.get(
                        "signature_header",
                        "X-Signature-256",
                    )
                ),
                enabled=bool(
                    configuration.get(
                        "enabled",
                        True,
                    )
                ),
                accepted_events=set(
                    configuration.get(
                        "accepted_events",
                        [],
                    )
                ),
                metadata=dict(
                    configuration.get(
                        "metadata",
                        {},
                    )
                ),
            )

            self.register(
                webhook,
                replace=replace,
            )

            imported.append(
                webhook.name
            )

        return imported

    # ========================================================
    # SECURITY HELPERS
    # ========================================================

    def rotate_secret(
        self,
        name: str,
        *,
        length: int = 48,
    ) -> str:
        """
        Generate and assign a new webhook secret.

        The generated secret is returned once to the caller.
        Store it securely using RENIX's secrets manager.
        """

        if length < 16:
            raise ValueError(
                "Webhook secret length must be "
                "at least 16 characters."
            )

        webhook = self.get(name)

        new_secret = secrets.token_urlsafe(
            length
        )

        webhook.secret = new_secret
        webhook.touch()

        return new_secret

    def remove_secret(
        self,
        name: str,
    ) -> Webhook:
        """Remove webhook signing secret."""

        webhook = self.get(name)

        webhook.secret = None
        webhook.touch()

        return webhook

    # ========================================================
    # CLEANUP
    # ========================================================

    def clear_processed_events(self) -> None:
        """Clear duplicate-event cache."""

        with self._lock:
            self._processed_events.clear()

    def clear(self) -> None:
        """Remove all webhooks and handlers."""

        with self._lock:
            self._webhooks.clear()
            self._handlers.clear()
            self._processed_events.clear()

    # ========================================================
    # PYTHON PROTOCOLS
    # ========================================================

    def __len__(self) -> int:
        return len(self._webhooks)

    def __contains__(
        self,
        name: str,
    ) -> bool:
        return self.exists(name)

    def __iter__(self):
        return iter(
            self.list_webhooks()
        )

    def __repr__(self) -> str:
        return (
            "WebhookManager("
            f"webhooks={len(self._webhooks)})"
        )


# ============================================================
# DEFAULT FACTORY
# ============================================================


def create_default_webhook_manager() -> WebhookManager:
    """Create a fresh RENIX webhook manager."""

    return WebhookManager()


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "Webhook",
    "WebhookEvent",
    "WebhookStatistics",
    "WebhookManager",
    "WebhookStatus",
    "SignatureAlgorithm",
    "WebhookError",
    "WebhookNotFoundError",
    "WebhookAlreadyExistsError",
    "WebhookDisabledError",
    "WebhookVerificationError",
    "WebhookHandlerError",
    "DuplicateWebhookEventError",
    "create_default_webhook_manager",
]


