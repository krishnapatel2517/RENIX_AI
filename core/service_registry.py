"""
RENIX AI
Core Service Registry

Central registry for RENIX services.

Responsibilities:
- Register services
- Unregister services
- Retrieve services
- Check service availability
- Manage service lifecycle
- Track service metadata
- Resolve service dependencies
- Support aliases
- Health/status information
- Prevent duplicate registrations
- Async service initialization/shutdown
"""

from __future__ import annotations

import inspect
import logging
import time

from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(
    "RENIX.ServiceRegistry"
)


# ============================================================================
# SERVICE RECORD
# ============================================================================

@dataclass
class ServiceRecord:
    """
    Stores information about a registered RENIX service.
    """

    name: str

    service: Any

    version: str = "1.0.0"

    description: str = ""

    enabled: bool = True

    initialized: bool = False

    healthy: bool = True

    registered_at: float = field(
        default_factory=time.time
    )

    initialized_at: Optional[float] = None

    shutdown_at: Optional[float] = None

    dependencies: list[str] = field(
        default_factory=list
    )

    aliases: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    call_count: int = 0

    last_called_at: Optional[float] = None

    # ========================================================================
    # DICTIONARY
    # ========================================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the service record to a dictionary.

        The actual service object is intentionally not serialized.
        """

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
            "initialized": self.initialized,
            "healthy": self.healthy,
            "registered_at": self.registered_at,
            "initialized_at": self.initialized_at,
            "shutdown_at": self.shutdown_at,
            "dependencies": list(
                self.dependencies
            ),
            "aliases": list(
                self.aliases
            ),
            "metadata": dict(
                self.metadata
            ),
            "error": self.error,
            "call_count": self.call_count,
            "last_called_at": self.last_called_at,
        }


# ============================================================================
# SERVICE REGISTRY
# ============================================================================

class ServiceRegistry:
    """
    Central service registry used by RENIX.

    Any RENIX subsystem can register a service here and
    other subsystems can retrieve it without directly
    creating duplicate instances.
    """

    def __init__(
        self,
        *,
        allow_replace: bool = False,
    ) -> None:

        self.logger = logger

        self.allow_replace = (
            allow_replace
        )

        self.services: dict[
            str,
            ServiceRecord,
        ] = {}

        self.aliases: dict[
            str,
            str,
        ] = {}

        self.initialized = False

        self.started_at = time.time()

        self.total_registered = 0

        self.total_unregistered = 0

        self.total_resolved = 0

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize all registered services.
        """

        if self.initialized:

            return

        self.initialized = True

        for name in list(
            self.services.keys()
        ):

            try:

                await self.initialize_service(
                    name
                )

            except Exception as exc:

                self.logger.exception(
                    "Failed to initialize service '%s': %s",
                    name,
                    exc,
                )

        self.logger.info(
            "RENIX Service Registry initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shutdown all registered services.
        """

        names = list(
            self.services.keys()
        )

        # Reverse order is useful when services
        # depend on services registered earlier.
        for name in reversed(names):

            try:

                await self.shutdown_service(
                    name
                )

            except Exception as exc:

                self.logger.exception(
                    "Failed to shutdown service '%s': %s",
                    name,
                    exc,
                )

        self.initialized = False

        self.logger.info(
            "RENIX Service Registry shutdown."
        )

    # ========================================================================
    # REGISTER
    # ========================================================================

    def register(
        self,
        name: str,
        service: Any,
        *,
        version: str = "1.0.0",
        description: str = "",
        enabled: bool = True,
        dependencies: Optional[
            list[str]
        ] = None,
        aliases: Optional[
            list[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        replace: Optional[bool] = None,
    ) -> ServiceRecord:
        """
        Register a service.

        Example:

            registry.register(
                "voice",
                voice_service,
            )
        """

        name = self._normalize_name(
            name
        )

        if not name:

            raise ValueError(
                "Service name cannot be empty."
            )

        if service is None:

            raise ValueError(
                f"Cannot register None as service '{name}'."
            )

        if replace is None:

            replace = self.allow_replace

        if (
            name in self.services
            and not replace
        ):

            raise ValueError(
                f"Service already registered: {name}"
            )

        if name in self.aliases:

            raise ValueError(
                f"Service name conflicts with alias: {name}"
            )

        # Remove previous service if replacing.
        if (
            name in self.services
            and replace
        ):

            self.unregister(
                name
            )

        normalized_dependencies = [
            self._normalize_name(
                dependency
            )
            for dependency
            in (
                dependencies
                or []
            )
        ]

        normalized_aliases = [
            self._normalize_name(
                alias
            )
            for alias
            in (
                aliases
                or []
            )
        ]

        for alias in normalized_aliases:

            if not alias:

                continue

            if alias == name:

                raise ValueError(
                    "Service cannot alias itself."
                )

            if alias in self.services:

                raise ValueError(
                    f"Alias conflicts with service: {alias}"
                )

            if alias in self.aliases:

                raise ValueError(
                    f"Alias already registered: {alias}"
                )

        record = ServiceRecord(
            name=name,
            service=service,
            version=version,
            description=description,
            enabled=enabled,
            dependencies=(
                normalized_dependencies
            ),
            aliases=(
                normalized_aliases
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        self.services[
            name
        ] = record

        for alias in normalized_aliases:

            self.aliases[
                alias
            ] = name

        self.total_registered += 1

        self.logger.info(
            "Registered service '%s'.",
            name,
        )

        return record

    # ========================================================================
    # REGISTER INSTANCE
    # ========================================================================

    def register_instance(
        self,
        name: str,
        service: Any,
        **kwargs: Any,
    ) -> ServiceRecord:
        """
        Alias for register().
        """

        return self.register(
            name,
            service,
            **kwargs,
        )

    # ========================================================================
    # UNREGISTER
    # ========================================================================

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Remove a service from the registry.
        """

        canonical_name = (
            self.resolve_name(
                name
            )
        )

        if canonical_name is None:

            return False

        record = self.services.pop(
            canonical_name,
            None,
        )

        if record is None:

            return False

        for alias in list(
            record.aliases
        ):

            self.aliases.pop(
                alias,
                None,
            )

        self.total_unregistered += 1

        self.logger.info(
            "Unregistered service '%s'.",
            canonical_name,
        )

        return True

    # ========================================================================
    # GET
    # ========================================================================

    def get(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a service.

        Returns default when unavailable.
        """

        canonical_name = (
            self.resolve_name(
                name
            )
        )

        if canonical_name is None:

            return default

        record = self.services.get(
            canonical_name
        )

        if record is None:

            return default

        if not record.enabled:

            return default

        if not record.healthy:

            return default

        self.total_resolved += 1

        record.call_count += 1

        record.last_called_at = (
            time.time()
        )

        return record.service

    # ========================================================================
    # REQUIRE
    # ========================================================================

    def require(
        self,
        name: str,
    ) -> Any:
        """
        Retrieve a service or raise an error.
        """

        service = self.get(
            name
        )

        if service is None:

            raise LookupError(
                f"Required RENIX service "
                f"not available: {name}"
            )

        return service

    # ========================================================================
    # RECORD
    # ========================================================================

    def get_record(
        self,
        name: str,
    ) -> Optional[ServiceRecord]:
        """
        Retrieve a service record.
        """

        canonical_name = (
            self.resolve_name(
                name
            )
        )

        if canonical_name is None:

            return None

        return self.services.get(
            canonical_name
        )

    # ========================================================================
    # RESOLVE NAME
    # ========================================================================

    def resolve_name(
        self,
        name: str,
    ) -> Optional[str]:
        """
        Resolve a service name or alias
        into its canonical name.
        """

        normalized = (
            self._normalize_name(
                name
            )
        )

        if normalized in self.services:

            return normalized

        return self.aliases.get(
            normalized
        )

    # ========================================================================
    # EXISTS
    # ========================================================================

    def exists(
        self,
        name: str,
    ) -> bool:
        """
        Check whether a service exists.
        """

        return (
            self.resolve_name(
                name
            )
            is not None
        )

    # ========================================================================
    # ENABLE
    # ========================================================================

    def enable(
        self,
        name: str,
    ) -> bool:
        """
        Enable a service.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        record.enabled = True

        return True

    # ========================================================================
    # DISABLE
    # ========================================================================

    def disable(
        self,
        name: str,
    ) -> bool:
        """
        Disable a service.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        record.enabled = False

        return True

    # ========================================================================
    # HEALTH
    # ========================================================================

    def set_health(
        self,
        name: str,
        healthy: bool,
        *,
        error: Optional[str] = None,
    ) -> bool:
        """
        Update service health status.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        record.healthy = bool(
            healthy
        )

        record.error = error

        return True

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    async def health_check(
        self,
        name: str,
    ) -> bool:
        """
        Perform a health check if the service
        provides one.

        Supported method names:

        - health_check
        - check_health
        - is_healthy
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        service = record.service

        try:

            method = None

            for method_name in (
                "health_check",
                "check_health",
                "is_healthy",
            ):

                candidate = getattr(
                    service,
                    method_name,
                    None,
                )

                if callable(candidate):

                    method = candidate

                    break

            if method is None:

                record.healthy = True

                record.error = None

                return True

            result = method()

            if inspect.isawaitable(
                result
            ):

                result = await result

            healthy = bool(
                result
            )

            record.healthy = healthy

            record.error = (
                None
                if healthy
                else "Health check failed."
            )

            return healthy

        except Exception as exc:

            record.healthy = False

            record.error = str(
                exc
            )

            self.logger.exception(
                "Health check failed for '%s'.",
                name,
            )

            return False

    # ========================================================================
    # HEALTH CHECK ALL
    # ========================================================================

    async def health_check_all(
        self,
    ) -> dict[str, bool]:
        """
        Run health checks for every service.
        """

        results: dict[
            str,
            bool,
        ] = {}

        for name in self.services:

            results[
                name
            ] = await self.health_check(
                name
            )

        return results

    # ========================================================================
    # INITIALIZE SERVICE
    # ========================================================================

    async def initialize_service(
        self,
        name: str,
    ) -> bool:
        """
        Initialize one service.

        Supported lifecycle methods:

        - initialize
        - start
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        if record.initialized:

            return True

        if not record.enabled:

            return False

        # Initialize dependencies first.
        for dependency in (
            record.dependencies
        ):

            dependency_record = (
                self.get_record(
                    dependency
                )
            )

            if dependency_record is None:

                record.healthy = False

                record.error = (
                    "Missing dependency: "
                    f"{dependency}"
                )

                self.logger.error(
                    "Service '%s' has missing dependency '%s'.",
                    name,
                    dependency,
                )

                return False

            success = (
                await self.initialize_service(
                    dependency
                )
            )

            if not success:

                record.healthy = False

                record.error = (
                    "Dependency failed: "
                    f"{dependency}"
                )

                return False

        service = record.service

        try:

            lifecycle_method = None

            for method_name in (
                "initialize",
                "start",
            ):

                candidate = getattr(
                    service,
                    method_name,
                    None,
                )

                if callable(candidate):

                    lifecycle_method = (
                        candidate
                    )

                    break

            if lifecycle_method is not None:

                result = lifecycle_method()

                if inspect.isawaitable(
                    result
                ):

                    await result

            record.initialized = True

            record.initialized_at = (
                time.time()
            )

            record.healthy = True

            record.error = None

            self.logger.info(
                "Initialized service '%s'.",
                name,
            )

            return True

        except Exception as exc:

            record.initialized = False

            record.healthy = False

            record.error = str(
                exc
            )

            self.logger.exception(
                "Failed to initialize service '%s'.",
                name,
            )

            return False

    # ========================================================================
    # SHUTDOWN SERVICE
    # ========================================================================

    async def shutdown_service(
        self,
        name: str,
    ) -> bool:
        """
        Shutdown one service.

        Supported lifecycle methods:

        - shutdown
        - stop
        - close
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        if not record.initialized:

            return True

        service = record.service

        try:

            lifecycle_method = None

            for method_name in (
                "shutdown",
                "stop",
                "close",
            ):

                candidate = getattr(
                    service,
                    method_name,
                    None,
                )

                if callable(candidate):

                    lifecycle_method = (
                        candidate
                    )

                    break

            if lifecycle_method is not None:

                result = lifecycle_method()

                if inspect.isawaitable(
                    result
                ):

                    await result

            record.initialized = False

            record.shutdown_at = (
                time.time()
            )

            self.logger.info(
                "Shutdown service '%s'.",
                name,
            )

            return True

        except Exception as exc:

            record.error = str(
                exc
            )

            self.logger.exception(
                "Failed to shutdown service '%s'.",
                name,
            )

            return False

    # ========================================================================
    # INITIALIZE ALL
    # ========================================================================

    async def initialize_all(
        self,
    ) -> dict[str, bool]:
        """
        Initialize all services.
        """

        results: dict[
            str,
            bool,
        ] = {}

        for name in list(
            self.services.keys()
        ):

            results[
                name
            ] = await self.initialize_service(
                name
            )

        return results

    # ========================================================================
    # SHUTDOWN ALL
    # ========================================================================

    async def shutdown_all(
        self,
    ) -> dict[str, bool]:
        """
        Shutdown all services.
        """

        results: dict[
            str,
            bool,
        ] = {}

        names = list(
            self.services.keys()
        )

        for name in reversed(names):

            results[
                name
            ] = await self.shutdown_service(
                name
            )

        return results

    # ========================================================================
    # ALIAS
    # ========================================================================

    def add_alias(
        self,
        service_name: str,
        alias: str,
    ) -> bool:
        """
        Add an alias to an existing service.
        """

        canonical = (
            self.resolve_name(
                service_name
            )
        )

        if canonical is None:

            return False

        alias = self._normalize_name(
            alias
        )

        if not alias:

            return False

        if alias in self.services:

            raise ValueError(
                f"Alias conflicts with service: {alias}"
            )

        if alias in self.aliases:

            raise ValueError(
                f"Alias already exists: {alias}"
            )

        self.aliases[
            alias
        ] = canonical

        record = self.services[
            canonical
        ]

        if alias not in record.aliases:

            record.aliases.append(
                alias
            )

        return True

    # ========================================================================
    # REMOVE ALIAS
    # ========================================================================

    def remove_alias(
        self,
        alias: str,
    ) -> bool:
        """
        Remove a service alias.
        """

        alias = self._normalize_name(
            alias
        )

        canonical = self.aliases.pop(
            alias,
            None,
        )

        if canonical is None:

            return False

        record = self.services.get(
            canonical
        )

        if record is not None:

            if alias in record.aliases:

                record.aliases.remove(
                    alias
                )

        return True

    # ========================================================================
    # LIST SERVICES
    # ========================================================================

    def list_services(
        self,
        *,
        enabled_only: bool = False,
        healthy_only: bool = False,
    ) -> list[str]:
        """
        Return registered service names.
        """

        result: list[str] = []

        for name, record in (
            self.services.items()
        ):

            if (
                enabled_only
                and not record.enabled
            ):

                continue

            if (
                healthy_only
                and not record.healthy
            ):

                continue

            result.append(
                name
            )

        return result

    # ========================================================================
    # LIST RECORDS
    # ========================================================================

    def list_records(
        self,
    ) -> list[ServiceRecord]:
        """
        Return all service records.
        """

        return list(
            self.services.values()
        )

    # ========================================================================
    # GET DEPENDENCIES
    # ========================================================================

    def get_dependencies(
        self,
        name: str,
    ) -> list[str]:
        """
        Return service dependencies.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return []

        return list(
            record.dependencies
        )

    # ========================================================================
    # GET DEPENDENTS
    # ========================================================================

    def get_dependents(
        self,
        name: str,
    ) -> list[str]:
        """
        Return services depending on a service.
        """

        canonical = (
            self.resolve_name(
                name
            )
        )

        if canonical is None:

            return []

        dependents: list[
            str
        ] = []

        for service_name, record in (
            self.services.items()
        ):

            if canonical in (
                record.dependencies
            ):

                dependents.append(
                    service_name
                )

        return dependents

    # ========================================================================
    # SERVICE STATUS
    # ========================================================================

    def status(
        self,
        name: str,
    ) -> Optional[
        dict[str, Any]
    ]:
        """
        Return the current status of a service.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return None

        return record.to_dict()

    # ========================================================================
    # ALL STATUS
    # ========================================================================

    def all_status(
        self,
    ) -> dict[
        str,
        dict[str, Any],
    ]:
        """
        Return status of all services.
        """

        return {
            name: record.to_dict()
            for name, record
            in self.services.items()
        }

    # ========================================================================
    # CALL SERVICE METHOD
    # ========================================================================

    async def call(
        self,
        service_name: str,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Call a method on a registered service.

        Supports both synchronous and asynchronous methods.
        """

        service = self.require(
            service_name
        )

        method = getattr(
            service,
            method_name,
            None,
        )

        if not callable(method):

            raise AttributeError(
                f"Service '{service_name}' "
                f"does not have callable method "
                f"'{method_name}'."
            )

        result = method(
            *args,
            **kwargs,
        )

        if inspect.isawaitable(
            result
        ):

            result = await result

        return result

    # ========================================================================
    # FIND BY TYPE
    # ========================================================================

    def find_by_type(
        self,
        service_type: type,
    ) -> list[Any]:
        """
        Find all registered services matching a type.
        """

        result: list[Any] = []

        for record in (
            self.services.values()
        ):

            if isinstance(
                record.service,
                service_type,
            ):

                result.append(
                    record.service
                )

        return result

    # ========================================================================
    # FIND BY METADATA
    # ========================================================================

    def find_by_metadata(
        self,
        key: str,
        value: Any,
    ) -> list[Any]:
        """
        Find services with matching metadata.
        """

        result: list[Any] = []

        for record in (
            self.services.values()
        ):

            if record.metadata.get(
                key
            ) == value:

                result.append(
                    record.service
                )

        return result

    # ========================================================================
    # CLEAR
    # ========================================================================

    async def clear(
        self,
    ) -> None:
        """
        Shutdown and remove all services.
        """

        await self.shutdown_all()

        self.services.clear()

        self.aliases.clear()

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return registry statistics.
        """

        healthy = sum(
            1
            for record
            in self.services.values()
            if record.healthy
        )

        initialized = sum(
            1
            for record
            in self.services.values()
            if record.initialized
        )

        enabled = sum(
            1
            for record
            in self.services.values()
            if record.enabled
        )

        return {
            "initialized": (
                self.initialized
            ),
            "service_count": len(
                self.services
            ),
            "alias_count": len(
                self.aliases
            ),
            "healthy_services": healthy,
            "initialized_services": initialized,
            "enabled_services": enabled,
            "total_registered": (
                self.total_registered
            ),
            "total_unregistered": (
                self.total_unregistered
            ),
            "total_resolved": (
                self.total_resolved
            ),
            "started_at": (
                self.started_at
            ),
        }

    # ========================================================================
    # NORMALIZE NAME
    # ========================================================================

    @staticmethod
    def _normalize_name(
        name: Any,
    ) -> str:
        """
        Normalize service names.
        """

        if name is None:

            return ""

        return str(
            name
        ).strip().lower()


# ============================================================================
# GLOBAL REGISTRY
# ============================================================================

service_registry = ServiceRegistry()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def register_service(
    name: str,
    service: Any,
    **kwargs: Any,
) -> ServiceRecord:
    """
    Register a service using the global registry.
    """

    return service_registry.register(
        name,
        service,
        **kwargs,
    )


def unregister_service(
    name: str,
) -> bool:
    """
    Unregister a service.
    """

    return service_registry.unregister(
        name
    )


def get_service(
    name: str,
    default: Any = None,
) -> Any:
    """
    Get a service from the global registry.
    """

    return service_registry.get(
        name,
        default,
    )


def require_service(
    name: str,
) -> Any:
    """
    Get a required service.
    """

    return service_registry.require(
        name
    )


def service_exists(
    name: str,
) -> bool:
    """
    Check whether a service exists.
    """

    return service_registry.exists(
        name
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ServiceRecord",
    "ServiceRegistry",
    "service_registry",
    "register_service",
    "unregister_service",
    "get_service",
    "require_service",
    "service_exists",
]


