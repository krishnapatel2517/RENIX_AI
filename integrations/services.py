"""
RENIX Integrations - Services
=============================

Service registry and lifecycle management for external
RENIX integrations.

Responsibilities:
- Register external services/providers.
- Track service configuration.
- Enable/disable services.
- Track service health/status.
- Store non-secret metadata.
- Resolve service names and aliases.
- Coordinate authentication names.
- Provide a common interface for the rest of RENIX.

Secrets should NOT be stored directly in this module.
Use security/secrets_manager.py and integrations/authentication.py
for credential handling.
"""

from __future__ import annotations

import threading
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================


class ServiceError(Exception):
    """Base exception for service management."""


class ServiceNotFoundError(ServiceError):
    """Raised when a requested service does not exist."""


class ServiceAlreadyExistsError(ServiceError):
    """Raised when a service is already registered."""


class ServiceDisabledError(ServiceError):
    """Raised when an operation targets a disabled service."""


class ServiceConfigurationError(ServiceError):
    """Raised when a service configuration is invalid."""


# ============================================================
# ENUMS
# ============================================================


class ServiceStatus(str, Enum):
    """Current state of an external service."""

    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    ERROR = "error"


class ServiceType(str, Enum):
    """General category of integration."""

    AI = "ai"
    WEATHER = "weather"
    MAPS = "maps"
    CALENDAR = "calendar"
    MEDIA = "media"
    SMART_HOME = "smart_home"
    CLOUD = "cloud"
    COMMUNICATION = "communication"
    STORAGE = "storage"
    SEARCH = "search"
    CUSTOM = "custom"


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class ServiceHealth:
    """Health information for a service."""

    status: ServiceStatus = ServiceStatus.UNKNOWN

    last_check: float | None = None

    last_success: float | None = None

    last_failure: float | None = None

    response_time_ms: float | None = None

    consecutive_failures: int = 0

    total_checks: int = 0

    total_successes: int = 0

    total_failures: int = 0

    last_error: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def record_success(
        self,
        response_time_ms: float | None = None,
    ) -> None:
        """Record a successful health check."""

        now = time.time()

        self.status = ServiceStatus.AVAILABLE
        self.last_check = now
        self.last_success = now
        self.response_time_ms = response_time_ms

        self.consecutive_failures = 0
        self.total_checks += 1
        self.total_successes += 1
        self.last_error = None

    def record_failure(
        self,
        error: str,
        response_time_ms: float | None = None,
    ) -> None:
        """Record a failed health check."""

        now = time.time()

        self.status = ServiceStatus.UNAVAILABLE
        self.last_check = now
        self.last_failure = now
        self.response_time_ms = response_time_ms

        self.consecutive_failures += 1
        self.total_checks += 1
        self.total_failures += 1

        self.last_error = str(error)

    def record_degraded(
        self,
        reason: str | None = None,
    ) -> None:
        """Mark the service as degraded."""

        self.status = ServiceStatus.DEGRADED
        self.last_check = time.time()

        if reason:
            self.last_error = str(reason)

    def to_dict(self) -> dict[str, Any]:
        """Serialize health information."""

        return {
            "status": self.status.value,
            "last_check": self.last_check,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "response_time_ms": self.response_time_ms,
            "consecutive_failures": self.consecutive_failures,
            "total_checks": self.total_checks,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "last_error": self.last_error,
            "metadata": dict(self.metadata),
        }


@dataclass
class Service:
    """
    Represents an external RENIX service.

    Example:

        Service(
            name="openai",
            service_type=ServiceType.AI,
            base_url="https://api.example.com",
        )
    """

    name: str

    service_type: ServiceType = ServiceType.CUSTOM

    display_name: str | None = None

    description: str = ""

    base_url: str | None = None

    version: str | None = None

    enabled: bool = True

    priority: int = 100

    authentication_name: str | None = None

    aliases: list[str] = field(
        default_factory=list
    )

    capabilities: set[str] = field(
        default_factory=set
    )

    configuration: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    health: ServiceHealth = field(
        default_factory=ServiceHealth
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    service_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    def __post_init__(self) -> None:
        self.name = self.name.strip()

        if self.display_name is None:
            self.display_name = self.name

        self.aliases = [
            str(alias).strip()
            for alias in self.aliases
            if str(alias).strip()
        ]

        self.capabilities = {
            str(capability).strip()
            for capability in self.capabilities
            if str(capability).strip()
        }

    def touch(self) -> None:
        """Update modification timestamp."""

        self.updated_at = time.time()

    def supports(
        self,
        capability: str,
    ) -> bool:
        """Check whether the service supports a capability."""

        return capability.lower() in {
            item.lower()
            for item in self.capabilities
        }

    def matches(
        self,
        identifier: str,
    ) -> bool:
        """Check whether an identifier matches this service."""

        identifier = identifier.strip().lower()

        if self.name.lower() == identifier:
            return True

        if (
            self.display_name
            and self.display_name.lower() == identifier
        ):
            return True

        return any(
            alias.lower() == identifier
            for alias in self.aliases
        )

    def public_dict(self) -> dict[str, Any]:
        """
        Return service information safe for normal
        UI/debugging use.

        Secret configuration values should never be placed
        inside Service.configuration.
        """

        return {
            "service_id": self.service_id,
            "name": self.name,
            "service_type": self.service_type.value,
            "display_name": self.display_name,
            "description": self.description,
            "base_url": self.base_url,
            "version": self.version,
            "enabled": self.enabled,
            "priority": self.priority,
            "authentication_name": self.authentication_name,
            "aliases": list(self.aliases),
            "capabilities": sorted(self.capabilities),
            "configuration": dict(self.configuration),
            "metadata": dict(self.metadata),
            "health": self.health.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# Callback used by service health checks.
HealthCheckCallback = Callable[
    [Service],
    Any,
]


# ============================================================
# SERVICE MANAGER
# ============================================================


class ServiceManager:
    """
    Central registry for RENIX integrations.

    Responsibilities include:

    - registration
    - lookup
    - alias resolution
    - enable/disable
    - priority management
    - capability discovery
    - health tracking
    - health checks
    """

    def __init__(
        self,
        *,
        strict_names: bool = True,
    ) -> None:
        self.strict_names = strict_names

        self._services: dict[str, Service] = {}

        self._health_callbacks: dict[
            str,
            HealthCheckCallback,
        ] = {}

        self._lock = threading.RLock()

    # ========================================================
    # NAME VALIDATION
    # ========================================================

    def _validate_name(
        self,
        name: str,
    ) -> str:
        """Validate and normalize service names."""

        if not isinstance(name, str):
            raise ServiceConfigurationError(
                "Service name must be a string."
            )

        normalized = name.strip()

        if not normalized:
            raise ServiceConfigurationError(
                "Service name cannot be empty."
            )

        if self.strict_names:
            allowed = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                "_-."
            )

            if any(
                character not in allowed
                for character in normalized
            ):
                raise ServiceConfigurationError(
                    f"Invalid service name: {name!r}"
                )

        return normalized

    # ========================================================
    # REGISTRATION
    # ========================================================

    def register(
        self,
        service: Service,
        *,
        replace: bool = False,
    ) -> str:
        """Register a service."""

        if not isinstance(
            service,
            Service,
        ):
            raise TypeError(
                "service must be a Service instance."
            )

        service.name = self._validate_name(
            service.name
        )

        with self._lock:
            if (
                service.name in self._services
                and not replace
            ):
                raise ServiceAlreadyExistsError(
                    f"Service already exists: "
                    f"{service.name}"
                )

            self._services[
                service.name
            ] = service

        return service.service_id

    def register_simple(
        self,
        name: str,
        *,
        service_type: ServiceType = ServiceType.CUSTOM,
        base_url: str | None = None,
        aliases: list[str] | None = None,
        capabilities: set[str] | None = None,
        authentication_name: str | None = None,
        configuration: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Convenience method for registering a service."""

        service = Service(
            name=name,
            service_type=service_type,
            base_url=base_url,
            aliases=list(aliases or []),
            capabilities=set(capabilities or set()),
            authentication_name=authentication_name,
            configuration=dict(
                configuration or {}
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(service)

    # ========================================================
    # LOOKUP
    # ========================================================

    def get(
        self,
        name: str,
    ) -> Service:
        """Retrieve a service by name or alias."""

        identifier = str(name).strip()

        with self._lock:
            direct = self._services.get(
                identifier
            )

            if direct is not None:
                return direct

            for service in self._services.values():
                if service.matches(identifier):
                    return service

        raise ServiceNotFoundError(
            f"Service not found: {name}"
        )

    def get_optional(
        self,
        name: str,
    ) -> Service | None:
        """Return a service or None."""

        try:
            return self.get(name)
        except ServiceNotFoundError:
            return None

    def exists(
        self,
        name: str,
    ) -> bool:
        """Check whether a service exists."""

        return self.get_optional(name) is not None

    def unregister(
        self,
        name: str,
    ) -> bool:
        """Remove a service."""

        with self._lock:
            service = self.get_optional(name)

            if service is None:
                return False

            self._services.pop(
                service.name,
                None,
            )

            self._health_callbacks.pop(
                service.name,
                None,
            )

            return True

    # ========================================================
    # ENABLE / DISABLE
    # ========================================================

    def enable(
        self,
        name: str,
    ) -> Service:
        """Enable a service."""

        service = self.get(name)

        service.enabled = True

        if service.health.status == ServiceStatus.DISABLED:
            service.health.status = ServiceStatus.UNKNOWN

        service.touch()

        return service

    def disable(
        self,
        name: str,
    ) -> Service:
        """Disable a service."""

        service = self.get(name)

        service.enabled = False
        service.health.status = ServiceStatus.DISABLED
        service.touch()

        return service

    def is_enabled(
        self,
        name: str,
    ) -> bool:
        """Return whether a service is enabled."""

        return self.get(name).enabled

    def require_enabled(
        self,
        name: str,
    ) -> Service:
        """Return service only if enabled."""

        service = self.get(name)

        if not service.enabled:
            raise ServiceDisabledError(
                f"Service is disabled: {service.name}"
            )

        return service

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def configure(
        self,
        name: str,
        values: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> Service:
        """
        Update service configuration.

        This should contain only non-secret configuration.
        """

        service = self.get(name)

        if replace:
            service.configuration = dict(values)
        else:
            service.configuration.update(
                values
            )

        service.touch()

        return service

    def get_configuration(
        self,
        name: str,
    ) -> dict[str, Any]:
        """Return a copy of service configuration."""

        service = self.get(name)

        return dict(
            service.configuration
        )

    # ========================================================
    # CAPABILITIES
    # ========================================================

    def add_capability(
        self,
        name: str,
        capability: str,
    ) -> Service:
        """Add a capability to a service."""

        service = self.get(name)

        capability = str(
            capability
        ).strip()

        if not capability:
            raise ValueError(
                "Capability cannot be empty."
            )

        service.capabilities.add(
            capability
        )

        service.touch()

        return service

    def remove_capability(
        self,
        name: str,
        capability: str,
    ) -> bool:
        """Remove a capability."""

        service = self.get(name)

        if capability not in service.capabilities:
            return False

        service.capabilities.remove(
            capability
        )

        service.touch()

        return True

    def find_by_capability(
        self,
        capability: str,
        *,
        enabled_only: bool = True,
    ) -> list[Service]:
        """Find services supporting a capability."""

        with self._lock:
            services = list(
                self._services.values()
            )

        matches = []

        for service in services:
            if enabled_only and not service.enabled:
                continue

            if service.supports(capability):
                matches.append(service)

        return sorted(
            matches,
            key=lambda item: item.priority,
        )

    # ========================================================
    # SERVICE TYPES
    # ========================================================

    def find_by_type(
        self,
        service_type: ServiceType,
        *,
        enabled_only: bool = True,
    ) -> list[Service]:
        """Find services by type."""

        with self._lock:
            services = list(
                self._services.values()
            )

        matches = [
            service
            for service in services
            if service.service_type == service_type
            and (
                not enabled_only
                or service.enabled
            )
        ]

        return sorted(
            matches,
            key=lambda item: item.priority,
        )

    # ========================================================
    # PRIORITY
    # ========================================================

    def set_priority(
        self,
        name: str,
        priority: int,
    ) -> Service:
        """Set service priority."""

        service = self.get(name)

        service.priority = int(
            priority
        )

        service.touch()

        return service

    def get_preferred(
        self,
        *,
        service_type: ServiceType | None = None,
        capability: str | None = None,
    ) -> Service | None:
        """
        Return the highest-priority enabled service
        matching the requested criteria.
        """

        if service_type is not None:
            candidates = self.find_by_type(
                service_type
            )
        elif capability is not None:
            candidates = self.find_by_capability(
                capability
            )
        else:
            with self._lock:
                candidates = [
                    service
                    for service
                    in self._services.values()
                    if service.enabled
                ]

            candidates.sort(
                key=lambda item: item.priority
            )

        if not candidates:
            return None

        return candidates[0]

    # ========================================================
    # ALIASES
    # ========================================================

    def add_alias(
        self,
        name: str,
        alias: str,
    ) -> Service:
        """Add an alias to a service."""

        service = self.get(name)

        alias = str(alias).strip()

        if not alias:
            raise ValueError(
                "Alias cannot be empty."
            )

        existing = self.get_optional(alias)

        if (
            existing is not None
            and existing.name != service.name
        ):
            raise ServiceAlreadyExistsError(
                f"Alias already belongs to another "
                f"service: {alias}"
            )

        if alias.lower() != service.name.lower():
            if alias not in service.aliases:
                service.aliases.append(alias)

        service.touch()

        return service

    def remove_alias(
        self,
        name: str,
        alias: str,
    ) -> bool:
        """Remove an alias."""

        service = self.get(name)

        try:
            service.aliases.remove(alias)
        except ValueError:
            return False

        service.touch()

        return True

    # ========================================================
    # AUTHENTICATION LINK
    # ========================================================

    def set_authentication(
        self,
        name: str,
        authentication_name: str | None,
    ) -> Service:
        """
        Link a service to a credential registered in
        integrations/authentication.py.
        """

        service = self.get(name)

        if authentication_name is not None:
            authentication_name = (
                str(authentication_name).strip()
            )

            if not authentication_name:
                authentication_name = None

        service.authentication_name = (
            authentication_name
        )

        service.touch()

        return service

    # ========================================================
    # HEALTH CHECKS
    # ========================================================

    def register_health_check(
        self,
        name: str,
        callback: HealthCheckCallback,
    ) -> None:
        """
        Register a health-check callback.

        Callback receives the Service object and should either:

        - return normally for success
        - raise an exception for failure
        """

        if not callable(callback):
            raise TypeError(
                "Health check callback must be callable."
            )

        service = self.get(name)

        with self._lock:
            self._health_callbacks[
                service.name
            ] = callback

    def remove_health_check(
        self,
        name: str,
    ) -> bool:
        """Remove a health-check callback."""

        service = self.get(name)

        with self._lock:
            return (
                self._health_callbacks.pop(
                    service.name,
                    None,
                )
                is not None
            )

    def check_health(
        self,
        name: str,
    ) -> ServiceHealth:
        """Run a registered health check."""

        service = self.get(name)

        if not service.enabled:
            service.health.status = (
                ServiceStatus.DISABLED
            )
            return service.health

        with self._lock:
            callback = self._health_callbacks.get(
                service.name
            )

        if callback is None:
            service.health.status = (
                ServiceStatus.UNKNOWN
            )
            service.health.last_check = time.time()

            return service.health

        started = time.perf_counter()

        try:
            callback(service)

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            service.health.record_success(
                response_time_ms=elapsed
            )

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            service.health.record_failure(
                str(exc),
                response_time_ms=elapsed,
            )

        service.touch()

        return service.health

    def check_all_health(
        self,
    ) -> dict[str, ServiceHealth]:
        """Run health checks for all registered services."""

        with self._lock:
            names = list(
                self._services.keys()
            )

        results: dict[
            str,
            ServiceHealth,
        ] = {}

        for name in names:
            results[name] = self.check_health(
                name
            )

        return results

    # ========================================================
    # HEALTH STATUS
    # ========================================================

    def set_health_status(
        self,
        name: str,
        status: ServiceStatus,
        *,
        error: str | None = None,
    ) -> Service:
        """Manually set a service health status."""

        service = self.get(name)

        service.health.status = status
        service.health.last_check = time.time()

        if error:
            service.health.last_error = str(
                error
            )

        service.touch()

        return service

    def get_health(
        self,
        name: str,
    ) -> ServiceHealth:
        """Return service health."""

        return self.get(name).health

    # ========================================================
    # LISTING
    # ========================================================

    def list_services(
        self,
        *,
        enabled_only: bool = False,
        service_type: ServiceType | None = None,
    ) -> list[Service]:
        """List registered services."""

        with self._lock:
            services = list(
                self._services.values()
            )

        if enabled_only:
            services = [
                service
                for service in services
                if service.enabled
            ]

        if service_type is not None:
            services = [
                service
                for service in services
                if service.service_type
                == service_type
            ]

        return sorted(
            services,
            key=lambda item: (
                item.priority,
                item.name.lower(),
            ),
        )

    def list_public(
        self,
    ) -> list[dict[str, Any]]:
        """Return public service information."""

        return [
            service.public_dict()
            for service in self.list_services()
        ]

    # ========================================================
    # STATUS
    # ========================================================

    def status(
        self,
    ) -> dict[str, Any]:
        """Return registry status."""

        services = self.list_services()

        counts: dict[str, int] = {}

        for service in services:
            key = service.health.status.value

            counts[key] = (
                counts.get(key, 0)
                + 1
            )

        return {
            "total_services": len(services),
            "enabled_services": sum(
                service.enabled
                for service in services
            ),
            "disabled_services": sum(
                not service.enabled
                for service in services
            ),
            "health": counts,
            "services": [
                {
                    "name": service.name,
                    "type": service.service_type.value,
                    "enabled": service.enabled,
                    "priority": service.priority,
                    "health": (
                        service.health.status.value
                    ),
                }
                for service in services
            ],
        }

    # ========================================================
    # IMPORT / EXPORT
    # ========================================================

    def export_config(
        self,
    ) -> list[dict[str, Any]]:
        """
        Export service configuration.

        Secret values should never be stored here.
        """

        result = []

        for service in self.list_services():
            result.append(
                {
                    "name": service.name,
                    "service_type": (
                        service.service_type.value
                    ),
                    "display_name": service.display_name,
                    "description": service.description,
                    "base_url": service.base_url,
                    "version": service.version,
                    "enabled": service.enabled,
                    "priority": service.priority,
                    "authentication_name": (
                        service.authentication_name
                    ),
                    "aliases": list(
                        service.aliases
                    ),
                    "capabilities": sorted(
                        service.capabilities
                    ),
                    "configuration": dict(
                        service.configuration
                    ),
                    "metadata": dict(
                        service.metadata
                    ),
                }
            )

        return result

    def import_config(
        self,
        configurations: list[
            Mapping[str, Any]
        ],
        *,
        replace: bool = False,
    ) -> list[str]:
        """Import service definitions."""

        registered = []

        for configuration in configurations:
            service_type_value = (
                configuration.get(
                    "service_type",
                    ServiceType.CUSTOM.value,
                )
            )

            try:
                service_type = ServiceType(
                    service_type_value
                )
            except ValueError:
                service_type = ServiceType.CUSTOM

            service = Service(
                name=str(
                    configuration["name"]
                ),
                service_type=service_type,
                display_name=configuration.get(
                    "display_name"
                ),
                description=str(
                    configuration.get(
                        "description",
                        "",
                    )
                ),
                base_url=configuration.get(
                    "base_url"
                ),
                version=configuration.get(
                    "version"
                ),
                enabled=bool(
                    configuration.get(
                        "enabled",
                        True,
                    )
                ),
                priority=int(
                    configuration.get(
                        "priority",
                        100,
                    )
                ),
                authentication_name=(
                    configuration.get(
                        "authentication_name"
                    )
                ),
                aliases=list(
                    configuration.get(
                        "aliases",
                        [],
                    )
                ),
                capabilities=set(
                    configuration.get(
                        "capabilities",
                        [],
                    )
                ),
                configuration=dict(
                    configuration.get(
                        "configuration",
                        {},
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
                service,
                replace=replace,
            )

            registered.append(
                service.name
            )

        return registered

    # ========================================================
    # CLEANUP
    # ========================================================

    def clear(
        self,
    ) -> None:
        """Remove all services and callbacks."""

        with self._lock:
            self._services.clear()
            self._health_callbacks.clear()

    # ========================================================
    # PYTHON PROTOCOLS
    # ========================================================

    def __len__(self) -> int:
        return len(self._services)

    def __contains__(
        self,
        name: str,
    ) -> bool:
        return self.exists(name)

    def __iter__(self):
        return iter(
            self.list_services()
        )

    def __repr__(self) -> str:
        return (
            "ServiceManager("
            f"services={len(self._services)})"
        )


# ============================================================
# DEFAULT SERVICE FACTORY
# ============================================================


def create_default_service_manager() -> ServiceManager:
    """
    Create an empty ServiceManager.

    Providers can later be registered from configuration.
    """

    return ServiceManager()


# ============================================================
# PUBLIC API
# ============================================================


__all__ = [
    "Service",
    "ServiceHealth",
    "ServiceStatus",
    "ServiceType",
    "ServiceManager",
    "ServiceError",
    "ServiceNotFoundError",
    "ServiceAlreadyExistsError",
    "ServiceDisabledError",
    "ServiceConfigurationError",
    "create_default_service_manager",
]



