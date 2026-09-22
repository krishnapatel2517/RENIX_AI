"""
RENIX AI
Core Capability Manager

Responsible for managing the capabilities available to RENIX.

Responsibilities:
- Register capabilities
- Unregister capabilities
- Enable / disable capabilities
- Capability aliases
- Capability metadata
- Capability providers
- Capability dependencies
- Capability lookup
- Capability execution
- Capability health
- Capability statistics
- Provider prioritization
- Dynamic capability discovery
"""

from __future__ import annotations

import inspect
import logging
import time

from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(
    "RENIX.CapabilityManager"
)


# ============================================================================
# CAPABILITY RECORD
# ============================================================================

@dataclass
class CapabilityRecord:
    """
    Stores information about a RENIX capability.
    """

    name: str

    description: str = ""

    category: str = "general"

    enabled: bool = True

    healthy: bool = True

    providers: list[str] = field(
        default_factory=list
    )

    dependencies: list[str] = field(
        default_factory=list
    )

    aliases: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    updated_at: float = field(
        default_factory=time.time
    )

    execution_count: int = 0

    last_execution: Optional[float] = None

    error: Optional[str] = None

    # ========================================================================
    # TO DICT
    # ========================================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert capability information into a dictionary.
        """

        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "enabled": self.enabled,
            "healthy": self.healthy,
            "providers": list(
                self.providers
            ),
            "dependencies": list(
                self.dependencies
            ),
            "aliases": list(
                self.aliases
            ),
            "metadata": dict(
                self.metadata
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_count": (
                self.execution_count
            ),
            "last_execution": (
                self.last_execution
            ),
            "error": self.error,
        }


# ============================================================================
# CAPABILITY PROVIDER
# ============================================================================

@dataclass
class CapabilityProvider:
    """
    Represents an object that provides a capability.
    """

    name: str

    provider: Any

    priority: int = 100

    enabled: bool = True

    healthy: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    registered_at: float = field(
        default_factory=time.time
    )

    execution_count: int = 0

    last_execution: Optional[float] = None

    error: Optional[str] = None

    # ========================================================================
    # TO DICT
    # ========================================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert provider information into a dictionary.
        """

        return {
            "name": self.name,
            "priority": self.priority,
            "enabled": self.enabled,
            "healthy": self.healthy,
            "metadata": dict(
                self.metadata
            ),
            "registered_at": (
                self.registered_at
            ),
            "execution_count": (
                self.execution_count
            ),
            "last_execution": (
                self.last_execution
            ),
            "error": self.error,
        }


# ============================================================================
# CAPABILITY MANAGER
# ============================================================================

class CapabilityManager:
    """
    Central capability registry for RENIX.

    A capability describes something RENIX can do.

    Examples:

        "open_application"
        "search_web"
        "read_file"
        "write_file"
        "send_message"
        "control_mouse"
        "control_keyboard"
        "speech_to_text"
        "text_to_speech"
        "vision"
        "gesture_control"
        "coding"
        "research"
        "automation"
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

        self.capabilities: dict[
            str,
            CapabilityRecord,
        ] = {}

        self.providers: dict[
            str,
            CapabilityProvider,
        ] = {}

        self.aliases: dict[
            str,
            str,
        ] = {}

        self.initialized = False

        self.started_at = time.time()

        self.total_registered = 0

        self.total_executions = 0

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the capability manager.
        """

        if self.initialized:

            return

        self.initialized = True

        self.logger.info(
            "RENIX Capability Manager initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shutdown the capability manager.
        """

        self.initialized = False

        self.logger.info(
            "RENIX Capability Manager shutdown."
        )

    # ========================================================================
    # REGISTER CAPABILITY
    # ========================================================================

    def register_capability(
        self,
        name: str,
        *,
        description: str = "",
        category: str = "general",
        enabled: bool = True,
        providers: Optional[
            list[str]
        ] = None,
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
    ) -> CapabilityRecord:
        """
        Register a capability.
        """

        name = self._normalize_name(
            name
        )

        if not name:

            raise ValueError(
                "Capability name cannot be empty."
            )

        if replace is None:

            replace = self.allow_replace

        if (
            name in self.capabilities
            and not replace
        ):

            raise ValueError(
                f"Capability already registered: {name}"
            )

        if name in self.aliases:

            raise ValueError(
                f"Capability conflicts with alias: {name}"
            )

        if (
            name in self.capabilities
            and replace
        ):

            self.unregister_capability(
                name
            )

        normalized_providers = [
            self._normalize_name(
                provider
            )
            for provider
            in (
                providers
                or []
            )
            if self._normalize_name(
                provider
            )
        ]

        normalized_dependencies = [
            self._normalize_name(
                dependency
            )
            for dependency
            in (
                dependencies
                or []
            )
            if self._normalize_name(
                dependency
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
            if self._normalize_name(
                alias
            )
        ]

        for alias in normalized_aliases:

            if alias == name:

                raise ValueError(
                    "Capability cannot alias itself."
                )

            if alias in self.capabilities:

                raise ValueError(
                    f"Alias conflicts with capability: {alias}"
                )

            if alias in self.aliases:

                raise ValueError(
                    f"Alias already exists: {alias}"
                )

        record = CapabilityRecord(
            name=name,
            description=description,
            category=category,
            enabled=enabled,
            providers=(
                normalized_providers
            ),
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

        self.capabilities[
            name
        ] = record

        for alias in normalized_aliases:

            self.aliases[
                alias
            ] = name

        self.total_registered += 1

        self.logger.info(
            "Registered capability '%s'.",
            name,
        )

        return record

    # ========================================================================
    # UNREGISTER CAPABILITY
    # ========================================================================

    def unregister_capability(
        self,
        name: str,
    ) -> bool:
        """
        Remove a capability.
        """

        canonical = (
            self.resolve_name(
                name
            )
        )

        if canonical is None:

            return False

        record = self.capabilities.pop(
            canonical,
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

        self.logger.info(
            "Unregistered capability '%s'.",
            canonical,
        )

        return True

    # ========================================================================
    # RESOLVE NAME
    # ========================================================================

    def resolve_name(
        self,
        name: str,
    ) -> Optional[str]:
        """
        Resolve a capability name or alias.
        """

        normalized = (
            self._normalize_name(
                name
            )
        )

        if normalized in (
            self.capabilities
        ):

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
        Check whether a capability exists.
        """

        return (
            self.resolve_name(
                name
            )
            is not None
        )

    # ========================================================================
    # GET CAPABILITY
    # ========================================================================

    def get_capability(
        self,
        name: str,
    ) -> Optional[
        CapabilityRecord
    ]:
        """
        Return a capability record.
        """

        canonical = (
            self.resolve_name(
                name
            )
        )

        if canonical is None:

            return None

        return self.capabilities.get(
            canonical
        )

    # ========================================================================
    # ADD ALIAS
    # ========================================================================

    def add_alias(
        self,
        capability_name: str,
        alias: str,
    ) -> bool:
        """
        Add an alias to a capability.
        """

        canonical = (
            self.resolve_name(
                capability_name
            )
        )

        if canonical is None:

            return False

        alias = self._normalize_name(
            alias
        )

        if not alias:

            return False

        if alias in self.capabilities:

            raise ValueError(
                f"Alias conflicts with capability: {alias}"
            )

        if alias in self.aliases:

            raise ValueError(
                f"Alias already exists: {alias}"
            )

        self.aliases[
            alias
        ] = canonical

        record = self.capabilities[
            canonical
        ]

        if alias not in record.aliases:

            record.aliases.append(
                alias
            )

        record.updated_at = time.time()

        return True

    # ========================================================================
    # REMOVE ALIAS
    # ========================================================================

    def remove_alias(
        self,
        alias: str,
    ) -> bool:
        """
        Remove an alias.
        """

        alias = self._normalize_name(
            alias
        )

        capability_name = (
            self.aliases.pop(
                alias,
                None,
            )
        )

        if capability_name is None:

            return False

        record = self.capabilities.get(
            capability_name
        )

        if record is not None:

            if alias in record.aliases:

                record.aliases.remove(
                    alias
                )

                record.updated_at = (
                    time.time()
                )

        return True

    # ========================================================================
    # ENABLE
    # ========================================================================

    def enable_capability(
        self,
        name: str,
    ) -> bool:
        """
        Enable a capability.
        """

        record = self.get_capability(
            name
        )

        if record is None:

            return False

        record.enabled = True

        record.updated_at = time.time()

        return True

    # ========================================================================
    # DISABLE
    # ========================================================================

    def disable_capability(
        self,
        name: str,
    ) -> bool:
        """
        Disable a capability.
        """

        record = self.get_capability(
            name
        )

        if record is None:

            return False

        record.enabled = False

        record.updated_at = time.time()

        return True

    # ========================================================================
    # REGISTER PROVIDER
    # ========================================================================

    def register_provider(
        self,
        name: str,
        provider: Any,
        *,
        priority: int = 100,
        enabled: bool = True,
        healthy: bool = True,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> CapabilityProvider:
        """
        Register an object that can provide capabilities.
        """

        name = self._normalize_name(
            name
        )

        if not name:

            raise ValueError(
                "Provider name cannot be empty."
            )

        if name in self.providers:

            raise ValueError(
                f"Provider already registered: {name}"
            )

        record = CapabilityProvider(
            name=name,
            provider=provider,
            priority=priority,
            enabled=enabled,
            healthy=healthy,
            metadata=dict(
                metadata or {}
            ),
        )

        self.providers[
            name
        ] = record

        self.logger.info(
            "Registered capability provider '%s'.",
            name,
        )

        return record

    # ========================================================================
    # UNREGISTER PROVIDER
    # ========================================================================

    def unregister_provider(
        self,
        name: str,
    ) -> bool:
        """
        Remove a provider.
        """

        name = self._normalize_name(
            name
        )

        if name not in self.providers:

            return False

        self.providers.pop(
            name
        )

        for capability in (
            self.capabilities.values()
        ):

            if name in capability.providers:

                capability.providers.remove(
                    name
                )

                capability.updated_at = (
                    time.time()
                )

        self.logger.info(
            "Unregistered capability provider '%s'.",
            name,
        )

        return True

    # ========================================================================
    # ADD PROVIDER TO CAPABILITY
    # ========================================================================

    def add_provider(
        self,
        capability_name: str,
        provider_name: str,
    ) -> bool:
        """
        Connect a provider to a capability.
        """

        capability = (
            self.get_capability(
                capability_name
            )
        )

        if capability is None:

            return False

        provider_name = (
            self._normalize_name(
                provider_name
            )
        )

        if provider_name not in (
            self.providers
        ):

            raise ValueError(
                f"Provider not registered: {provider_name}"
            )

        if provider_name not in (
            capability.providers
        ):

            capability.providers.append(
                provider_name
            )

        capability.updated_at = time.time()

        return True

    # ========================================================================
    # REMOVE PROVIDER
    # ========================================================================

    def remove_provider(
        self,
        capability_name: str,
        provider_name: str,
    ) -> bool:
        """
        Remove a provider from a capability.
        """

        capability = (
            self.get_capability(
                capability_name
            )
        )

        if capability is None:

            return False

        provider_name = (
            self._normalize_name(
                provider_name
            )
        )

        if provider_name not in (
            capability.providers
        ):

            return False

        capability.providers.remove(
            provider_name
        )

        capability.updated_at = time.time()

        return True

    # ========================================================================
    # GET PROVIDERS
    # ========================================================================

    def get_providers(
        self,
        capability_name: str,
        *,
        healthy_only: bool = True,
        enabled_only: bool = True,
    ) -> list[
        CapabilityProvider
    ]:
        """
        Get providers for a capability,
        sorted by priority.
        """

        capability = (
            self.get_capability(
                capability_name
            )
        )

        if capability is None:

            return []

        result: list[
            CapabilityProvider
        ] = []

        for provider_name in (
            capability.providers
        ):

            provider = self.providers.get(
                provider_name
            )

            if provider is None:

                continue

            if (
                enabled_only
                and not provider.enabled
            ):

                continue

            if (
                healthy_only
                and not provider.healthy
            ):

                continue

            result.append(
                provider
            )

        result.sort(
            key=lambda item: item.priority
        )

        return result

    # ========================================================================
    # GET BEST PROVIDER
    # ========================================================================

    def get_best_provider(
        self,
        capability_name: str,
    ) -> Optional[
        CapabilityProvider
    ]:
        """
        Return the highest-priority healthy provider.
        """

        providers = self.get_providers(
            capability_name
        )

        if not providers:

            return None

        return providers[0]

    # ========================================================================
    # EXECUTE CAPABILITY
    # ========================================================================

    async def execute(
        self,
        capability_name: str,
        *args: Any,
        method: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a capability through its best available provider.
        """

        capability = (
            self.get_capability(
                capability_name
            )
        )

        if capability is None:

            raise LookupError(
                f"Capability not found: "
                f"{capability_name}"
            )

        if not capability.enabled:

            raise RuntimeError(
                f"Capability is disabled: "
                f"{capability.name}"
            )

        if not capability.healthy:

            raise RuntimeError(
                f"Capability is unhealthy: "
                f"{capability.name}"
            )

        # --------------------------------------------------------------------
        # Check dependencies
        # --------------------------------------------------------------------

        for dependency in (
            capability.dependencies
        ):

            dependency_record = (
                self.get_capability(
                    dependency
                )
            )

            if (
                dependency_record is None
                or not dependency_record.enabled
                or not dependency_record.healthy
            ):

                raise RuntimeError(
                    f"Capability dependency unavailable: "
                    f"{dependency}"
                )

        # --------------------------------------------------------------------
        # Select provider
        # --------------------------------------------------------------------

        selected_provider = None

        if provider is not None:

            provider_name = (
                self._normalize_name(
                    provider
                )
            )

            selected_provider = (
                self.providers.get(
                    provider_name
                )
            )

            if selected_provider is None:

                raise LookupError(
                    f"Capability provider not found: "
                    f"{provider}"
                )

            if (
                provider_name
                not in capability.providers
            ):

                raise RuntimeError(
                    f"Provider '{provider_name}' "
                    f"does not provide capability "
                    f"'{capability.name}'."
                )

        else:

            selected_provider = (
                self.get_best_provider(
                    capability.name
                )
            )

        if selected_provider is None:

            raise RuntimeError(
                f"No healthy provider available "
                f"for capability '{capability.name}'."
            )

        if (
            not selected_provider.enabled
            or not selected_provider.healthy
        ):

            raise RuntimeError(
                f"Provider '{selected_provider.name}' "
                f"is unavailable."
            )

        target = (
            selected_provider.provider
        )

        # --------------------------------------------------------------------
        # Select method
        # --------------------------------------------------------------------

        if method is not None:

            callable_target = getattr(
                target,
                method,
                None,
            )

        else:

            callable_target = None

            # Preferred generic capability methods.
            for candidate_name in (
                capability.name,
                "execute",
                "run",
                "handle",
                "perform",
            ):

                candidate = getattr(
                    target,
                    candidate_name,
                    None,
                )

                if callable(candidate):

                    callable_target = (
                        candidate
                    )

                    break

            # Provider itself can be callable.
            if (
                callable_target is None
                and callable(target)
            ):

                callable_target = target

        if not callable(
            callable_target
        ):

            raise AttributeError(
                f"Provider '{selected_provider.name}' "
                f"cannot execute capability "
                f"'{capability.name}'."
            )

        # --------------------------------------------------------------------
        # Execute
        # --------------------------------------------------------------------

        try:

            result = callable_target(
                *args,
                **kwargs,
            )

            if inspect.isawaitable(
                result
            ):

                result = await result

            capability.execution_count += 1

            capability.last_execution = (
                time.time()
            )

            capability.error = None

            capability.updated_at = (
                time.time()
            )

            selected_provider.execution_count += 1

            selected_provider.last_execution = (
                time.time()
            )

            selected_provider.error = None

            self.total_executions += 1

            return result

        except Exception as exc:

            capability.error = str(
                exc
            )

            selected_provider.error = (
                str(exc)
            )

            self.logger.exception(
                "Capability '%s' failed through provider '%s'.",
                capability.name,
                selected_provider.name,
            )

            raise

    # ========================================================================
    # HEALTH CHECK PROVIDER
    # ========================================================================

    async def health_check_provider(
        self,
        name: str,
    ) -> bool:
        """
        Check provider health.
        """

        provider = self.providers.get(
            self._normalize_name(
                name
            )
        )

        if provider is None:

            return False

        target = provider.provider

        try:

            health_method = None

            for method_name in (
                "health_check",
                "check_health",
                "is_healthy",
            ):

                candidate = getattr(
                    target,
                    method_name,
                    None,
                )

                if callable(candidate):

                    health_method = candidate

                    break

            if health_method is None:

                provider.healthy = True

                provider.error = None

                return True

            result = health_method()

            if inspect.isawaitable(
                result
            ):

                result = await result

            provider.healthy = bool(
                result
            )

            provider.error = (
                None
                if provider.healthy
                else "Health check failed."
            )

            return provider.healthy

        except Exception as exc:

            provider.healthy = False

            provider.error = str(
                exc
            )

            self.logger.exception(
                "Provider health check failed: %s",
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
        Check all providers and capabilities.
        """

        provider_results: dict[
            str,
            bool,
        ] = {}

        for name in self.providers:

            provider_results[
                name
            ] = await self.health_check_provider(
                name
            )

        capability_results: dict[
            str,
            bool,
        ] = {}

        for name, capability in (
            self.capabilities.items()
        ):

            providers = self.get_providers(
                name
            )

            capability.healthy = bool(
                providers
            )

            capability_results[
                name
            ] = capability.healthy

        return {
            "providers": provider_results,
            "capabilities": capability_results,
        }

    # ========================================================================
    # FIND BY CATEGORY
    # ========================================================================

    def find_by_category(
        self,
        category: str,
    ) -> list[
        CapabilityRecord
    ]:
        """
        Find capabilities belonging to a category.
        """

        category = str(
            category
        ).strip().lower()

        return [
            capability
            for capability
            in self.capabilities.values()
            if (
                capability.category.lower()
                == category
            )
        ]

    # ========================================================================
    # FIND BY KEYWORD
    # ========================================================================

    def search(
        self,
        query: str,
    ) -> list[
        CapabilityRecord
    ]:
        """
        Search capabilities by name,
        description, category or aliases.
        """

        query = str(
            query
        ).strip().lower()

        if not query:

            return []

        results: list[
            CapabilityRecord
        ] = []

        for capability in (
            self.capabilities.values()
        ):

            searchable = " ".join(
                [
                    capability.name,
                    capability.description,
                    capability.category,
                    *capability.aliases,
                ]
            ).lower()

            if query in searchable:

                results.append(
                    capability
                )

        return results

    # ========================================================================
    # LIST CAPABILITIES
    # ========================================================================

    def list_capabilities(
        self,
        *,
        enabled_only: bool = False,
        healthy_only: bool = False,
        category: Optional[str] = None,
    ) -> list[str]:
        """
        Return registered capability names.
        """

        result: list[str] = []

        normalized_category = (
            category.strip().lower()
            if category
            else None
        )

        for name, capability in (
            self.capabilities.items()
        ):

            if (
                enabled_only
                and not capability.enabled
            ):

                continue

            if (
                healthy_only
                and not capability.healthy
            ):

                continue

            if (
                normalized_category
                is not None
                and capability.category.lower()
                != normalized_category
            ):

                continue

            result.append(
                name
            )

        return result

    # ========================================================================
    # LIST PROVIDERS
    # ========================================================================

    def list_providers(
        self,
        *,
        enabled_only: bool = False,
        healthy_only: bool = False,
    ) -> list[str]:
        """
        Return provider names.
        """

        result: list[str] = []

        for name, provider in (
            self.providers.items()
        ):

            if (
                enabled_only
                and not provider.enabled
            ):

                continue

            if (
                healthy_only
                and not provider.healthy
            ):

                continue

            result.append(
                name
            )

        return result

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(
        self,
        name: str,
    ) -> Optional[
        dict[str, Any]
    ]:
        """
        Return capability status.
        """

        capability = (
            self.get_capability(
                name
            )
        )

        if capability is None:

            return None

        return capability.to_dict()

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
        Return status of every capability.
        """

        return {
            name: capability.to_dict()
            for name, capability
            in self.capabilities.items()
        }

    # ========================================================================
    # PROVIDER STATUS
    # ========================================================================

    def provider_status(
        self,
        name: str,
    ) -> Optional[
        dict[str, Any]
    ]:
        """
        Return provider status.
        """

        provider = self.providers.get(
            self._normalize_name(
                name
            )
        )

        if provider is None:

            return None

        return provider.to_dict()

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return capability manager statistics.
        """

        enabled_capabilities = sum(
            1
            for capability
            in self.capabilities.values()
            if capability.enabled
        )

        healthy_capabilities = sum(
            1
            for capability
            in self.capabilities.values()
            if capability.healthy
        )

        enabled_providers = sum(
            1
            for provider
            in self.providers.values()
            if provider.enabled
        )

        healthy_providers = sum(
            1
            for provider
            in self.providers.values()
            if provider.healthy
        )

        return {
            "initialized": (
                self.initialized
            ),
            "capability_count": len(
                self.capabilities
            ),
            "provider_count": len(
                self.providers
            ),
            "alias_count": len(
                self.aliases
            ),
            "enabled_capabilities": (
                enabled_capabilities
            ),
            "healthy_capabilities": (
                healthy_capabilities
            ),
            "enabled_providers": (
                enabled_providers
            ),
            "healthy_providers": (
                healthy_providers
            ),
            "total_registered": (
                self.total_registered
            ),
            "total_executions": (
                self.total_executions
            ),
            "started_at": (
                self.started_at
            ),
        }

    # ========================================================================
    # CLEAR
    # ========================================================================

    def clear(
        self,
    ) -> None:
        """
        Remove all capabilities and providers.
        """

        self.capabilities.clear()

        self.providers.clear()

        self.aliases.clear()

        self.logger.info(
            "All RENIX capabilities cleared."
        )

    # ========================================================================
    # NORMALIZE NAME
    # ========================================================================

    @staticmethod
    def _normalize_name(
        name: Any,
    ) -> str:
        """
        Normalize names for consistent lookup.
        """

        if name is None:

            return ""

        return str(
            name
        ).strip().lower()


# ============================================================================
# GLOBAL CAPABILITY MANAGER
# ============================================================================

capability_manager = (
    CapabilityManager()
)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def register_capability(
    name: str,
    **kwargs: Any,
) -> CapabilityRecord:
    """
    Register a capability using the global manager.
    """

    return capability_manager.register_capability(
        name,
        **kwargs,
    )


def register_provider(
    name: str,
    provider: Any,
    **kwargs: Any,
) -> CapabilityProvider:
    """
    Register a provider using the global manager.
    """

    return capability_manager.register_provider(
        name,
        provider,
        **kwargs,
    )


def get_capability(
    name: str,
) -> Optional[
    CapabilityRecord
]:
    """
    Get a capability from the global manager.
    """

    return capability_manager.get_capability(
        name
    )


async def execute_capability(
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Execute a capability using the global manager.
    """

    return await capability_manager.execute(
        name,
        *args,
        **kwargs,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CapabilityRecord",
    "CapabilityProvider",
    "CapabilityManager",
    "capability_manager",
    "register_capability",
    "register_provider",
    "get_capability",
    "execute_capability",
]


