"""
RENIX AI
Core Plugin Manager

Responsible for managing RENIX plugins.

Features:
- Plugin registration
- Plugin discovery
- Plugin loading
- Plugin unloading
- Plugin enabling/disabling
- Plugin dependencies
- Plugin metadata
- Plugin lifecycle
- Plugin health status
- Plugin capabilities
- Plugin lookup
- Plugin aliases
- Plugin execution
- Plugin statistics
"""

from __future__ import annotations

import importlib
import inspect
import logging
import time

from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(
    "RENIX.PluginManager"
)


# ============================================================================
# PLUGIN RECORD
# ============================================================================

@dataclass
class PluginRecord:
    """
    Stores information about a RENIX plugin.
    """

    name: str

    plugin: Any = None

    version: str = "1.0.0"

    description: str = ""

    author: str = ""

    module: Optional[str] = None

    enabled: bool = True

    loaded: bool = False

    initialized: bool = False

    healthy: bool = True

    dependencies: list[str] = field(
        default_factory=list
    )

    capabilities: list[str] = field(
        default_factory=list
    )

    aliases: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    registered_at: float = field(
        default_factory=time.time
    )

    loaded_at: Optional[float] = None

    initialized_at: Optional[float] = None

    unloaded_at: Optional[float] = None

    error: Optional[str] = None

    execution_count: int = 0

    last_execution: Optional[float] = None

    # ========================================================================
    # TO DICT
    # ========================================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert plugin metadata into a dictionary.

        The actual plugin object is intentionally excluded.
        """

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "module": self.module,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "initialized": self.initialized,
            "healthy": self.healthy,
            "dependencies": list(
                self.dependencies
            ),
            "capabilities": list(
                self.capabilities
            ),
            "aliases": list(
                self.aliases
            ),
            "metadata": dict(
                self.metadata
            ),
            "registered_at": self.registered_at,
            "loaded_at": self.loaded_at,
            "initialized_at": self.initialized_at,
            "unloaded_at": self.unloaded_at,
            "error": self.error,
            "execution_count": (
                self.execution_count
            ),
            "last_execution": (
                self.last_execution
            ),
        }


# ============================================================================
# PLUGIN MANAGER
# ============================================================================

class PluginManager:
    """
    Central plugin manager for RENIX.
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

        self.plugins: dict[
            str,
            PluginRecord,
        ] = {}

        self.aliases: dict[
            str,
            str,
        ] = {}

        self.initialized = False

        self.started_at = time.time()

        self.total_registered = 0

        self.total_loaded = 0

        self.total_unloaded = 0

        self.total_executions = 0

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the Plugin Manager.

        Registered plugins are not automatically loaded unless
        they were already registered with an active plugin object.
        """

        if self.initialized:

            return

        self.initialized = True

        for name in list(
            self.plugins.keys()
        ):

            record = self.plugins[
                name
            ]

            if (
                record.plugin is not None
                and record.enabled
                and not record.initialized
            ):

                try:

                    await self.initialize_plugin(
                        name
                    )

                except Exception as exc:

                    self.logger.exception(
                        "Failed to initialize plugin '%s': %s",
                        name,
                        exc,
                    )

        self.logger.info(
            "RENIX Plugin Manager initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shutdown all plugins.
        """

        for name in reversed(
            list(
                self.plugins.keys()
            )
        ):

            try:

                await self.unload_plugin(
                    name
                )

            except Exception as exc:

                self.logger.exception(
                    "Failed to unload plugin '%s': %s",
                    name,
                    exc,
                )

        self.initialized = False

        self.logger.info(
            "RENIX Plugin Manager shutdown."
        )

    # ========================================================================
    # REGISTER
    # ========================================================================

    def register_plugin(
        self,
        name: str,
        plugin: Any = None,
        *,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        module: Optional[str] = None,
        enabled: bool = True,
        dependencies: Optional[
            list[str]
        ] = None,
        capabilities: Optional[
            list[str]
        ] = None,
        aliases: Optional[
            list[str]
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        replace: Optional[bool] = None,
    ) -> PluginRecord:
        """
        Register a plugin.
        """

        name = self._normalize_name(
            name
        )

        if not name:

            raise ValueError(
                "Plugin name cannot be empty."
            )

        if replace is None:

            replace = self.allow_replace

        if (
            name in self.plugins
            and not replace
        ):

            raise ValueError(
                f"Plugin already registered: {name}"
            )

        if name in self.aliases:

            raise ValueError(
                f"Plugin name conflicts with alias: {name}"
            )

        if (
            name in self.plugins
            and replace
        ):

            self.unregister_plugin(
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
            if self._normalize_name(
                dependency
            )
        ]

        normalized_capabilities = [
            str(capability).strip()
            for capability
            in (
                capabilities
                or []
            )
            if str(
                capability
            ).strip()
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
                    "Plugin cannot alias itself."
                )

            if alias in self.plugins:

                raise ValueError(
                    f"Alias conflicts with plugin: {alias}"
                )

            if alias in self.aliases:

                raise ValueError(
                    f"Alias already exists: {alias}"
                )

        record = PluginRecord(
            name=name,
            plugin=plugin,
            version=version,
            description=description,
            author=author,
            module=module,
            enabled=enabled,
            loaded=plugin is not None,
            dependencies=(
                normalized_dependencies
            ),
            capabilities=(
                normalized_capabilities
            ),
            aliases=(
                normalized_aliases
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        if plugin is not None:

            record.loaded_at = time.time()

        self.plugins[
            name
        ] = record

        for alias in normalized_aliases:

            self.aliases[
                alias
            ] = name

        self.total_registered += 1

        self.logger.info(
            "Registered plugin '%s'.",
            name,
        )

        return record

    # ========================================================================
    # REGISTER ALIAS
    # ========================================================================

    def add_alias(
        self,
        plugin_name: str,
        alias: str,
    ) -> bool:
        """
        Add an alias to an existing plugin.
        """

        canonical = (
            self.resolve_name(
                plugin_name
            )
        )

        if canonical is None:

            return False

        alias = self._normalize_name(
            alias
        )

        if not alias:

            return False

        if alias in self.plugins:

            raise ValueError(
                f"Alias conflicts with plugin: {alias}"
            )

        if alias in self.aliases:

            raise ValueError(
                f"Alias already exists: {alias}"
            )

        self.aliases[
            alias
        ] = canonical

        record = self.plugins[
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
        Remove a plugin alias.
        """

        alias = self._normalize_name(
            alias
        )

        plugin_name = self.aliases.pop(
            alias,
            None,
        )

        if plugin_name is None:

            return False

        record = self.plugins.get(
            plugin_name
        )

        if record is not None:

            if alias in record.aliases:

                record.aliases.remove(
                    alias
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
        Resolve a plugin name or alias.
        """

        normalized = (
            self._normalize_name(
                name
            )
        )

        if normalized in self.plugins:

            return normalized

        return self.aliases.get(
            normalized
        )

    # ========================================================================
    # GET
    # ========================================================================

    def get_plugin(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a plugin object.
        """

        canonical = (
            self.resolve_name(
                name
            )
        )

        if canonical is None:

            return default

        record = self.plugins.get(
            canonical
        )

        if record is None:

            return default

        if not record.enabled:

            return default

        if not record.loaded:

            return default

        if not record.healthy:

            return default

        return record.plugin

    # ========================================================================
    # REQUIRE
    # ========================================================================

    def require_plugin(
        self,
        name: str,
    ) -> Any:
        """
        Retrieve a plugin or raise an error.
        """

        plugin = self.get_plugin(
            name
        )

        if plugin is None:

            raise LookupError(
                f"Required RENIX plugin "
                f"is unavailable: {name}"
            )

        return plugin

    # ========================================================================
    # GET RECORD
    # ========================================================================

    def get_record(
        self,
        name: str,
    ) -> Optional[PluginRecord]:
        """
        Retrieve a plugin record.
        """

        canonical = (
            self.resolve_name(
                name
            )
        )

        if canonical is None:

            return None

        return self.plugins.get(
            canonical
        )

    # ========================================================================
    # EXISTS
    # ========================================================================

    def exists(
        self,
        name: str,
    ) -> bool:
        """
        Check whether a plugin exists.
        """

        return (
            self.resolve_name(
                name
            )
            is not None
        )

    # ========================================================================
    # LOAD MODULE
    # ========================================================================

    async def load_plugin(
        self,
        name: str,
        module_name: Optional[str] = None,
        *,
        factory_name: str = "create_plugin",
    ) -> PluginRecord:
        """
        Load a plugin from a Python module.

        Supported module factories:

        1. create_plugin()
        2. Plugin class
        3. plugin object
        """

        canonical = (
            self.resolve_name(
                name
            )
        )

        if canonical is None:

            canonical = (
                self._normalize_name(
                    name
                )
            )

        record = self.plugins.get(
            canonical
        )

        if record is None:

            record = self.register_plugin(
                canonical
            )

        if record.loaded:

            return record

        module_name = (
            module_name
            or record.module
        )

        if not module_name:

            raise ValueError(
                f"No module specified for plugin '{canonical}'."
            )

        try:

            module = importlib.import_module(
                module_name
            )

            plugin = None

            # --------------------------------------------------------------
            # Factory function
            # --------------------------------------------------------------

            factory = getattr(
                module,
                factory_name,
                None,
            )

            if callable(factory):

                plugin = factory()

                if inspect.isawaitable(
                    plugin
                ):

                    plugin = await plugin

            # --------------------------------------------------------------
            # Plugin class
            # --------------------------------------------------------------

            if plugin is None:

                plugin_class = getattr(
                    module,
                    "Plugin",
                    None,
                )

                if (
                    plugin_class is not None
                    and inspect.isclass(
                        plugin_class
                    )
                ):

                    plugin = plugin_class()

            # --------------------------------------------------------------
            # Module itself
            # --------------------------------------------------------------

            if plugin is None:

                plugin = module

            record.plugin = plugin

            record.module = module_name

            record.loaded = True

            record.loaded_at = (
                time.time()
            )

            record.healthy = True

            record.error = None

            self.total_loaded += 1

            self.logger.info(
                "Loaded plugin '%s' from '%s'.",
                canonical,
                module_name,
            )

            return record

        except Exception as exc:

            record.loaded = False

            record.healthy = False

            record.error = str(
                exc
            )

            self.logger.exception(
                "Failed to load plugin '%s'.",
                canonical,
            )

            raise

    # ========================================================================
    # INITIALIZE PLUGIN
    # ========================================================================

    async def initialize_plugin(
        self,
        name: str,
    ) -> bool:
        """
        Initialize one plugin.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        if not record.enabled:

            return False

        if not record.loaded:

            if record.module:

                await self.load_plugin(
                    record.name,
                    record.module,
                )

            else:

                return False

        if record.initialized:

            return True

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

                return False

            success = (
                await self.initialize_plugin(
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

        plugin = record.plugin

        try:

            initializer = None

            for method_name in (
                "initialize",
                "start",
                "on_load",
            ):

                candidate = getattr(
                    plugin,
                    method_name,
                    None,
                )

                if callable(candidate):

                    initializer = candidate

                    break

            if initializer is not None:

                result = initializer()

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
                "Initialized plugin '%s'.",
                record.name,
            )

            return True

        except Exception as exc:

            record.initialized = False

            record.healthy = False

            record.error = str(
                exc
            )

            self.logger.exception(
                "Failed to initialize plugin '%s'.",
                record.name,
            )

            return False

    # ========================================================================
    # UNLOAD
    # ========================================================================

    async def unload_plugin(
        self,
        name: str,
    ) -> bool:
        """
        Unload a plugin.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        plugin = record.plugin

        if plugin is not None:

            try:

                shutdown_method = None

                for method_name in (
                    "shutdown",
                    "stop",
                    "close",
                    "on_unload",
                ):

                    candidate = getattr(
                        plugin,
                        method_name,
                        None,
                    )

                    if callable(candidate):

                        shutdown_method = (
                            candidate
                        )

                        break

                if shutdown_method is not None:

                    result = shutdown_method()

                    if inspect.isawaitable(
                        result
                    ):

                        await result

            except Exception as exc:

                record.error = str(
                    exc
                )

                self.logger.exception(
                    "Plugin '%s' shutdown failed.",
                    record.name,
                )

        record.plugin = None

        record.loaded = False

        record.initialized = False

        record.unloaded_at = (
            time.time()
        )

        self.total_unloaded += 1

        self.logger.info(
            "Unloaded plugin '%s'.",
            record.name,
        )

        return True

    # ========================================================================
    # UNREGISTER
    # ========================================================================

    async def unregister_plugin(
        self,
        name: str,
    ) -> bool:
        """
        Unload and remove a plugin.
        """

        canonical = (
            self.resolve_name(
                name
            )
        )

        if canonical is None:

            return False

        await self.unload_plugin(
            canonical
        )

        record = self.plugins.pop(
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
            "Unregistered plugin '%s'.",
            canonical,
        )

        return True

    # ========================================================================
    # ENABLE
    # ========================================================================

    def enable_plugin(
        self,
        name: str,
    ) -> bool:
        """
        Enable a plugin.
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

    async def disable_plugin(
        self,
        name: str,
    ) -> bool:
        """
        Disable and unload a plugin.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        await self.unload_plugin(
            name
        )

        record.enabled = False

        return True

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    async def health_check(
        self,
        name: str,
    ) -> bool:
        """
        Run a plugin health check.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        if not record.loaded:

            record.healthy = False

            record.error = (
                "Plugin is not loaded."
            )

            return False

        plugin = record.plugin

        try:

            method = None

            for method_name in (
                "health_check",
                "check_health",
                "is_healthy",
            ):

                candidate = getattr(
                    plugin,
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

            record.healthy = bool(
                result
            )

            record.error = (
                None
                if record.healthy
                else "Health check failed."
            )

            return record.healthy

        except Exception as exc:

            record.healthy = False

            record.error = str(
                exc
            )

            self.logger.exception(
                "Health check failed for plugin '%s'.",
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
        Check health of every plugin.
        """

        results: dict[
            str,
            bool,
        ] = {}

        for name in self.plugins:

            results[
                name
            ] = await self.health_check(
                name
            )

        return results

    # ========================================================================
    # EXECUTE
    # ========================================================================

    async def execute(
        self,
        name: str,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a method exposed by a plugin.

        Supports synchronous and asynchronous methods.
        """

        record = self.get_record(
            name
        )

        if record is None:

            raise LookupError(
                f"Plugin not found: {name}"
            )

        if not record.enabled:

            raise RuntimeError(
                f"Plugin is disabled: {name}"
            )

        if not record.loaded:

            if record.module:

                await self.load_plugin(
                    record.name,
                    record.module,
                )

            else:

                raise RuntimeError(
                    f"Plugin is not loaded: {name}"
                )

        if not record.initialized:

            await self.initialize_plugin(
                record.name
            )

        if not record.healthy:

            raise RuntimeError(
                f"Plugin is unhealthy: {name}"
            )

        method = getattr(
            record.plugin,
            method_name,
            None,
        )

        if not callable(method):

            raise AttributeError(
                f"Plugin '{name}' does not have "
                f"method '{method_name}'."
            )

        try:

            result = method(
                *args,
                **kwargs,
            )

            if inspect.isawaitable(
                result
            ):

                result = await result

            record.execution_count += 1

            record.last_execution = (
                time.time()
            )

            self.total_executions += 1

            return result

        except Exception as exc:

            record.error = str(
                exc
            )

            self.logger.exception(
                "Plugin '%s' execution failed.",
                name,
            )

            raise

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def add_capability(
        self,
        name: str,
        capability: str,
    ) -> bool:
        """
        Add a capability to a plugin.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        capability = str(
            capability
        ).strip()

        if not capability:

            return False

        if capability not in (
            record.capabilities
        ):

            record.capabilities.append(
                capability
            )

        return True

    # ========================================================================
    # REMOVE CAPABILITY
    # ========================================================================

    def remove_capability(
        self,
        name: str,
        capability: str,
    ) -> bool:
        """
        Remove a plugin capability.
        """

        record = self.get_record(
            name
        )

        if record is None:

            return False

        if capability not in (
            record.capabilities
        ):

            return False

        record.capabilities.remove(
            capability
        )

        return True

    # ========================================================================
    # FIND BY CAPABILITY
    # ========================================================================

    def find_by_capability(
        self,
        capability: str,
    ) -> list[Any]:
        """
        Find loaded plugins providing a capability.
        """

        capability = str(
            capability
        ).strip()

        result: list[Any] = []

        for record in (
            self.plugins.values()
        ):

            if (
                capability
                in record.capabilities
                and record.enabled
                and record.loaded
                and record.healthy
            ):

                result.append(
                    record.plugin
                )

        return result

    # ========================================================================
    # FIND PLUGIN NAMES BY CAPABILITY
    # ========================================================================

    def find_plugin_names_by_capability(
        self,
        capability: str,
    ) -> list[str]:
        """
        Return names of plugins providing a capability.
        """

        capability = str(
            capability
        ).strip()

        result: list[str] = []

        for name, record in (
            self.plugins.items()
        ):

            if (
                capability
                in record.capabilities
                and record.enabled
                and record.loaded
                and record.healthy
            ):

                result.append(
                    name
                )

        return result

    # ========================================================================
    # DEPENDENCIES
    # ========================================================================

    def get_dependencies(
        self,
        name: str,
    ) -> list[str]:
        """
        Return plugin dependencies.
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
    # DEPENDENTS
    # ========================================================================

    def get_dependents(
        self,
        name: str,
    ) -> list[str]:
        """
        Return plugins depending on this plugin.
        """

        canonical = (
            self.resolve_name(
                name
            )
        )

        if canonical is None:

            return []

        result: list[str] = []

        for plugin_name, record in (
            self.plugins.items()
        ):

            if canonical in (
                record.dependencies
            ):

                result.append(
                    plugin_name
                )

        return result

    # ========================================================================
    # LIST PLUGINS
    # ========================================================================

    def list_plugins(
        self,
        *,
        enabled_only: bool = False,
        loaded_only: bool = False,
        healthy_only: bool = False,
    ) -> list[str]:
        """
        List registered plugin names.
        """

        result: list[str] = []

        for name, record in (
            self.plugins.items()
        ):

            if (
                enabled_only
                and not record.enabled
            ):

                continue

            if (
                loaded_only
                and not record.loaded
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
    ) -> list[PluginRecord]:
        """
        Return all plugin records.
        """

        return list(
            self.plugins.values()
        )

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
        Return status of a plugin.
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
        Return status of every plugin.
        """

        return {
            name: record.to_dict()
            for name, record
            in self.plugins.items()
        }

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return Plugin Manager statistics.
        """

        enabled = sum(
            1
            for record
            in self.plugins.values()
            if record.enabled
        )

        loaded = sum(
            1
            for record
            in self.plugins.values()
            if record.loaded
        )

        initialized = sum(
            1
            for record
            in self.plugins.values()
            if record.initialized
        )

        healthy = sum(
            1
            for record
            in self.plugins.values()
            if record.healthy
        )

        return {
            "initialized": (
                self.initialized
            ),
            "plugin_count": len(
                self.plugins
            ),
            "alias_count": len(
                self.aliases
            ),
            "enabled_plugins": enabled,
            "loaded_plugins": loaded,
            "initialized_plugins": initialized,
            "healthy_plugins": healthy,
            "total_registered": (
                self.total_registered
            ),
            "total_loaded": (
                self.total_loaded
            ),
            "total_unloaded": (
                self.total_unloaded
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

    async def clear(
        self,
    ) -> None:
        """
        Shutdown and remove all plugins.
        """

        for name in reversed(
            list(
                self.plugins.keys()
            )
        ):

            try:

                await self.unregister_plugin(
                    name
                )

            except Exception as exc:

                self.logger.exception(
                    "Failed to remove plugin '%s': %s",
                    name,
                    exc,
                )

        self.plugins.clear()

        self.aliases.clear()

    # ========================================================================
    # NORMALIZE NAME
    # ========================================================================

    @staticmethod
    def _normalize_name(
        name: Any,
    ) -> str:
        """
        Normalize plugin names.
        """

        if name is None:

            return ""

        return str(
            name
        ).strip().lower()


# ============================================================================
# GLOBAL PLUGIN MANAGER
# ============================================================================

plugin_manager = PluginManager()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def register_plugin(
    name: str,
    plugin: Any = None,
    **kwargs: Any,
) -> PluginRecord:
    """
    Register a plugin using the global manager.
    """

    return plugin_manager.register_plugin(
        name,
        plugin,
        **kwargs,
    )


def get_plugin(
    name: str,
    default: Any = None,
) -> Any:
    """
    Get a plugin using the global manager.
    """

    return plugin_manager.get_plugin(
        name,
        default,
    )


def require_plugin(
    name: str,
) -> Any:
    """
    Get a required plugin.
    """

    return plugin_manager.require_plugin(
        name
    )


def plugin_exists(
    name: str,
) -> bool:
    """
    Check whether a plugin exists.
    """

    return plugin_manager.exists(
        name
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PluginRecord",
    "PluginManager",
    "plugin_manager",
    "register_plugin",
    "get_plugin",
    "require_plugin",
    "plugin_exists",
]


