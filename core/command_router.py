"""
RENIX AI - Command Router
=========================

Central command routing system for RENIX.

Responsibilities:
    - Receive normalized commands
    - Identify command type
    - Route commands to registered handlers
    - Support synchronous and asynchronous handlers
    - Support command aliases
    - Support priorities
    - Support middleware
    - Support permissions
    - Support fallback handlers
    - Track command history
    - Track routing statistics
    - Return structured command results

This module does not directly perform computer, browser, file,
voice, vision, or AI operations.

It decides WHERE a command should go.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Union,
)


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(
    "RENIX.CommandRouter"
)


# ============================================================================
# TYPES
# ============================================================================

CommandHandler = Callable[..., Any]

CommandMiddleware = Callable[..., Any]

CommandPermissionChecker = Callable[..., Any]


# ============================================================================
# ENUMS
# ============================================================================

class CommandStatus(str, Enum):
    """
    Status of a routed command.
    """

    RECEIVED = "received"

    ROUTED = "routed"

    EXECUTING = "executing"

    COMPLETED = "completed"

    FAILED = "failed"

    BLOCKED = "blocked"

    UNKNOWN = "unknown"


class CommandPriority(int, Enum):
    """
    Command routing priority.
    """

    LOWEST = 0

    LOW = 1

    NORMAL = 2

    HIGH = 3

    CRITICAL = 4


# ============================================================================
# COMMAND
# ============================================================================

@dataclass
class RENIXCommand:
    """
    Normalized command representation.

    A command can originate from:
        - voice
        - text
        - gesture
        - holographic UI
        - automation
        - another RENIX service
        - external integrations
    """

    command_id: str

    raw_text: str = ""

    command_name: str = ""

    arguments: list[Any] = field(
        default_factory=list
    )

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    source: str = "unknown"

    user_id: Optional[str] = None

    priority: CommandPriority = (
        CommandPriority.NORMAL
    )

    status: CommandStatus = (
        CommandStatus.RECEIVED
    )

    created_at: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    requires_confirmation: bool = False

    confirmed: bool = False

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert command into a serializable dictionary.
        """

        return {
            "command_id": self.command_id,
            "raw_text": self.raw_text,
            "command_name": self.command_name,
            "arguments": list(
                self.arguments
            ),
            "parameters": dict(
                self.parameters
            ),
            "source": self.source,
            "user_id": self.user_id,
            "priority": self.priority.name,
            "priority_value": int(
                self.priority.value
            ),
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": dict(
                self.metadata
            ),
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "confirmed": self.confirmed,
        }


# ============================================================================
# COMMAND ROUTE
# ============================================================================

@dataclass
class CommandRoute:
    """
    Registered route for a command.
    """

    name: str

    handler: CommandHandler

    aliases: list[str] = field(
        default_factory=list
    )

    priority: CommandPriority = (
        CommandPriority.NORMAL
    )

    description: str = ""

    category: str = "general"

    enabled: bool = True

    requires_confirmation: bool = False

    permission: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    call_count: int = 0

    error_count: int = 0

    created_at: float = field(
        default_factory=time.time
    )

    def matches(
        self,
        command_name: str,
    ) -> bool:
        """
        Determine whether this route matches a command name.
        """

        normalized = (
            command_name.strip().lower()
        )

        if (
            self.name.lower()
            == normalized
        ):
            return True

        return any(
            alias.lower()
            == normalized
            for alias in self.aliases
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert route into a serializable dictionary.
        """

        return {
            "name": self.name,
            "aliases": list(
                self.aliases
            ),
            "priority": self.priority.name,
            "priority_value": int(
                self.priority.value
            ),
            "description": self.description,
            "category": self.category,
            "enabled": self.enabled,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "permission": self.permission,
            "metadata": dict(
                self.metadata
            ),
            "call_count": self.call_count,
            "error_count": self.error_count,
            "created_at": self.created_at,
        }


# ============================================================================
# COMMAND RESULT
# ============================================================================

@dataclass
class CommandResult:
    """
    Structured result returned by the command router.
    """

    command_id: str

    status: CommandStatus

    success: bool = False

    result: Any = None

    error: Optional[str] = None

    route: Optional[str] = None

    duration: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "status": self.status.value,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "route": self.route,
            "duration": self.duration,
            "metadata": dict(
                self.metadata
            ),
        }


# ============================================================================
# COMMAND ROUTER
# ============================================================================

class CommandRouter:
    """
    Central command routing engine for RENIX.
    """

    def __init__(
        self,
        *,
        max_history: int = 500,
    ) -> None:

        self.logger = logger

        self.max_history = max(
            1,
            int(max_history),
        )

        # --------------------------------------------------------------------
        # Routes
        # --------------------------------------------------------------------

        self.routes: dict[
            str,
            CommandRoute,
        ] = {}

        self.alias_map: dict[
            str,
            str,
        ] = {}

        # --------------------------------------------------------------------
        # Middleware
        # --------------------------------------------------------------------

        self.middlewares: list[
            CommandMiddleware
        ] = []

        # --------------------------------------------------------------------
        # Permission checkers
        # --------------------------------------------------------------------

        self.permission_checkers: list[
            CommandPermissionChecker
        ] = []

        # --------------------------------------------------------------------
        # Fallback
        # --------------------------------------------------------------------

        self.fallback_handler: Optional[
            CommandHandler
        ] = None

        # --------------------------------------------------------------------
        # History
        # --------------------------------------------------------------------

        self.history: list[
            CommandResult
        ] = []

        # --------------------------------------------------------------------
        # Counters
        # --------------------------------------------------------------------

        self._command_counter = 0

        # --------------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------------

        self.total_received = 0

        self.total_routed = 0

        self.total_completed = 0

        self.total_failed = 0

        self.total_blocked = 0

        self.total_unknown = 0

        self._lock = asyncio.Lock()

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the command router.
        """

        self.logger.info(
            "RENIX Command Router initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the command router.
        """

        async with self._lock:

            self.routes.clear()

            self.alias_map.clear()

            self.middlewares.clear()

            self.permission_checkers.clear()

            self.fallback_handler = None

        self.logger.info(
            "RENIX Command Router shutdown."
        )

    # ========================================================================
    # COMMAND ID
    # ========================================================================

    def _generate_command_id(
        self,
    ) -> str:
        """
        Generate a unique command ID.
        """

        self._command_counter += 1

        return (
            f"renix_command_"
            f"{int(time.time() * 1000)}_"
            f"{self._command_counter}"
        )

    # ========================================================================
    # NORMALIZE NAME
    # ========================================================================

    @staticmethod
    def normalize_command_name(
        name: str,
    ) -> str:
        """
        Normalize a command name.
        """

        if not name:
            return ""

        name = name.strip().lower()

        name = re.sub(
            r"\s+",
            "_",
            name,
        )

        name = re.sub(
            r"[^a-z0-9_.:-]",
            "",
            name,
        )

        return name

    # ========================================================================
    # CREATE COMMAND
    # ========================================================================

    def create_command(
        self,
        *,
        command_name: str,
        raw_text: str = "",
        arguments: Optional[
            list[Any]
        ] = None,
        parameters: Optional[
            dict[str, Any]
        ] = None,
        source: str = "unknown",
        user_id: Optional[str] = None,
        priority: CommandPriority = (
            CommandPriority.NORMAL
        ),
        metadata: Optional[
            dict[str, Any]
        ] = None,
        requires_confirmation: bool = False,
        confirmed: bool = False,
    ) -> RENIXCommand:
        """
        Create a normalized RENIX command.
        """

        normalized_name = (
            self.normalize_command_name(
                command_name
            )
        )

        if not normalized_name:

            raise ValueError(
                "Command name cannot be empty."
            )

        if not isinstance(
            priority,
            CommandPriority,
        ):

            priority = CommandPriority(
                int(priority)
            )

        return RENIXCommand(
            command_id=(
                self._generate_command_id()
            ),
            raw_text=raw_text,
            command_name=normalized_name,
            arguments=(
                arguments or []
            ),
            parameters=(
                parameters or {}
            ),
            source=source,
            user_id=user_id,
            priority=priority,
            metadata=(
                metadata or {}
            ),
            requires_confirmation=(
                requires_confirmation
            ),
            confirmed=confirmed,
        )

    # ========================================================================
    # REGISTER ROUTE
    # ========================================================================

    async def register(
        self,
        name: str,
        handler: CommandHandler,
        *,
        aliases: Optional[
            list[str]
        ] = None,
        priority: CommandPriority = (
            CommandPriority.NORMAL
        ),
        description: str = "",
        category: str = "general",
        requires_confirmation: bool = False,
        permission: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> CommandRoute:
        """
        Register a command route.
        """

        normalized_name = (
            self.normalize_command_name(
                name
            )
        )

        if not normalized_name:

            raise ValueError(
                "Route name cannot be empty."
            )

        if not callable(handler):

            raise TypeError(
                "Command handler must be callable."
            )

        if not isinstance(
            priority,
            CommandPriority,
        ):

            priority = CommandPriority(
                int(priority)
            )

        normalized_aliases: list[str] = []

        for alias in (
            aliases or []
        ):

            normalized_alias = (
                self.normalize_command_name(
                    alias
                )
            )

            if (
                normalized_alias
                and normalized_alias
                != normalized_name
            ):

                normalized_aliases.append(
                    normalized_alias
                )

        route = CommandRoute(
            name=normalized_name,
            handler=handler,
            aliases=normalized_aliases,
            priority=priority,
            description=description,
            category=category,
            requires_confirmation=(
                requires_confirmation
            ),
            permission=permission,
            metadata=(
                metadata or {}
            ),
        )

        async with self._lock:

            self.routes[
                normalized_name
            ] = route

            for alias in normalized_aliases:

                self.alias_map[
                    alias
                ] = normalized_name

        self.logger.debug(
            "Registered command route: %s",
            normalized_name,
        )

        return route

    # ========================================================================
    # REGISTER ALIAS
    # ========================================================================

    async def register_alias(
        self,
        command_name: str,
        alias: str,
    ) -> bool:
        """
        Add an alias to an existing route.
        """

        normalized_name = (
            self.normalize_command_name(
                command_name
            )
        )

        normalized_alias = (
            self.normalize_command_name(
                alias
            )
        )

        route = self.routes.get(
            normalized_name
        )

        if route is None:

            return False

        if not normalized_alias:

            return False

        if (
            normalized_alias
            not in route.aliases
        ):

            route.aliases.append(
                normalized_alias
            )

        self.alias_map[
            normalized_alias
        ] = normalized_name

        return True

    # ========================================================================
    # REMOVE ROUTE
    # ========================================================================

    async def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Remove a command route.
        """

        normalized_name = (
            self.normalize_command_name(
                name
            )
        )

        route = self.routes.pop(
            normalized_name,
            None,
        )

        if route is None:

            return False

        for alias in route.aliases:

            self.alias_map.pop(
                alias,
                None,
            )

        return True

    # ========================================================================
    # ENABLE ROUTE
    # ========================================================================

    def enable(
        self,
        name: str,
    ) -> bool:
        """
        Enable a route.
        """

        route = self.get_route(
            name
        )

        if route is None:

            return False

        route.enabled = True

        return True

    # ========================================================================
    # DISABLE ROUTE
    # ========================================================================

    def disable(
        self,
        name: str,
    ) -> bool:
        """
        Disable a route.
        """

        route = self.get_route(
            name
        )

        if route is None:

            return False

        route.enabled = False

        return True

    # ========================================================================
    # GET ROUTE
    # ========================================================================

    def get_route(
        self,
        name: str,
    ) -> Optional[CommandRoute]:
        """
        Retrieve a route by name or alias.
        """

        normalized_name = (
            self.normalize_command_name(
                name
            )
        )

        route = self.routes.get(
            normalized_name
        )

        if route is not None:

            return route

        canonical_name = (
            self.alias_map.get(
                normalized_name
            )
        )

        if canonical_name is None:

            return None

        return self.routes.get(
            canonical_name
        )

    # ========================================================================
    # ADD MIDDLEWARE
    # ========================================================================

    def add_middleware(
        self,
        middleware: CommandMiddleware,
    ) -> None:
        """
        Add middleware to the routing pipeline.
        """

        if not callable(
            middleware
        ):

            raise TypeError(
                "Middleware must be callable."
            )

        self.middlewares.append(
            middleware
        )

    # ========================================================================
    # REMOVE MIDDLEWARE
    # ========================================================================

    def remove_middleware(
        self,
        middleware: CommandMiddleware,
    ) -> bool:
        """
        Remove middleware.
        """

        try:

            self.middlewares.remove(
                middleware
            )

            return True

        except ValueError:

            return False

    # ========================================================================
    # ADD PERMISSION CHECKER
    # ========================================================================

    def add_permission_checker(
        self,
        checker: CommandPermissionChecker,
    ) -> None:
        """
        Add a permission checker.
        """

        if not callable(
            checker
        ):

            raise TypeError(
                "Permission checker must be callable."
            )

        self.permission_checkers.append(
            checker
        )

    # ========================================================================
    # SET FALLBACK
    # ========================================================================

    def set_fallback(
        self,
        handler: Optional[
            CommandHandler
        ],
    ) -> None:
        """
        Set the fallback handler for unknown commands.
        """

        if (
            handler is not None
            and not callable(handler)
        ):

            raise TypeError(
                "Fallback handler must be callable."
            )

        self.fallback_handler = handler

    # ========================================================================
    # CHECK PERMISSION
    # ========================================================================

    async def _check_permission(
        self,
        command: RENIXCommand,
        route: CommandRoute,
    ) -> bool:
        """
        Check all registered permission systems.
        """

        if route.permission is None:

            return True

        for checker in (
            self.permission_checkers
        ):

            try:

                result = checker(
                    command,
                    route.permission,
                )

                if inspect.isawaitable(
                    result
                ):

                    result = await result

                if not bool(result):

                    return False

            except Exception:

                self.logger.exception(
                    "Permission checker failed."
                )

                return False

        return True

    # ========================================================================
    # RUN MIDDLEWARE
    # ========================================================================

    async def _run_middleware(
        self,
        command: RENIXCommand,
    ) -> Optional[
        RENIXCommand
    ]:
        """
        Pass the command through middleware.

        Middleware may:
            - return the same command
            - return a modified command
            - return None to block the command
        """

        current = command

        for middleware in (
            self.middlewares
        ):

            try:

                result = middleware(
                    current
                )

                if inspect.isawaitable(
                    result
                ):

                    result = await result

                if result is None:

                    return None

                if isinstance(
                    result,
                    RENIXCommand,
                ):

                    current = result

            except Exception:

                self.logger.exception(
                    "Command middleware failed."
                )

                return None

        return current

    # ========================================================================
    # ROUTE COMMAND
    # ========================================================================

    async def route(
        self,
        command: RENIXCommand,
    ) -> CommandResult:
        """
        Route and execute a command.
        """

        start = time.time()

        self.total_received += 1

        command.status = (
            CommandStatus.RECEIVED
        )

        # --------------------------------------------------------------------
        # Middleware
        # --------------------------------------------------------------------

        processed_command = (
            await self._run_middleware(
                command
            )
        )

        if processed_command is None:

            command.status = (
                CommandStatus.BLOCKED
            )

            self.total_blocked += 1

            return self._record_result(
                CommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.BLOCKED,
                    success=False,
                    error=(
                        "Command blocked by middleware."
                    ),
                    duration=(
                        time.time()
                        - start
                    ),
                )
            )

        command = processed_command

        # --------------------------------------------------------------------
        # Find route
        # --------------------------------------------------------------------

        route = self.get_route(
            command.command_name
        )

        if route is None:

            self.total_unknown += 1

            # ---------------------------------------------------------------
            # Fallback
            # ---------------------------------------------------------------

            if self.fallback_handler is not None:

                try:

                    command.status = (
                        CommandStatus.ROUTED
                    )

                    self.total_routed += 1

                    result = (
                        await self._invoke_handler(
                            self.fallback_handler,
                            command,
                        )
                    )

                    command.status = (
                        CommandStatus.COMPLETED
                    )

                    self.total_completed += 1

                    return self._record_result(
                        CommandResult(
                            command_id=(
                                command.command_id
                            ),
                            status=(
                                CommandStatus.COMPLETED
                            ),
                            success=True,
                            result=result,
                            route="fallback",
                            duration=(
                                time.time()
                                - start
                            ),
                        )
                    )

                except Exception as exc:

                    self.total_failed += 1

                    command.status = (
                        CommandStatus.FAILED
                    )

                    return self._record_result(
                        CommandResult(
                            command_id=(
                                command.command_id
                            ),
                            status=(
                                CommandStatus.FAILED
                            ),
                            success=False,
                            error=str(exc),
                            route="fallback",
                            duration=(
                                time.time()
                                - start
                            ),
                        )
                    )

            command.status = (
                CommandStatus.UNKNOWN
            )

            return self._record_result(
                CommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.UNKNOWN,
                    success=False,
                    error=(
                        f"Unknown command: "
                        f"{command.command_name}"
                    ),
                    duration=(
                        time.time()
                        - start
                    ),
                )
            )

        # --------------------------------------------------------------------
        # Check enabled
        # --------------------------------------------------------------------

        if not route.enabled:

            command.status = (
                CommandStatus.BLOCKED
            )

            self.total_blocked += 1

            return self._record_result(
                CommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.BLOCKED,
                    success=False,
                    error=(
                        "Command route is disabled."
                    ),
                    route=route.name,
                    duration=(
                        time.time()
                        - start
                    ),
                )
            )

        # --------------------------------------------------------------------
        # Confirmation
        # --------------------------------------------------------------------

        if (
            route.requires_confirmation
            and not command.confirmed
        ):

            command.status = (
                CommandStatus.BLOCKED
            )

            self.total_blocked += 1

            return self._record_result(
                CommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.BLOCKED,
                    success=False,
                    error=(
                        "Command requires confirmation."
                    ),
                    route=route.name,
                    duration=(
                        time.time()
                        - start
                    ),
                    metadata={
                        "requires_confirmation": True,
                    },
                )
            )

        # --------------------------------------------------------------------
        # Permission
        # --------------------------------------------------------------------

        permitted = (
            await self._check_permission(
                command,
                route,
            )
        )

        if not permitted:

            command.status = (
                CommandStatus.BLOCKED
            )

            self.total_blocked += 1

            return self._record_result(
                CommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.BLOCKED,
                    success=False,
                    error=(
                        "Permission denied."
                    ),
                    route=route.name,
                    duration=(
                        time.time()
                        - start
                    ),
                )
            )

        # --------------------------------------------------------------------
        # Execute
        # --------------------------------------------------------------------

        command.status = (
            CommandStatus.ROUTED
        )

        self.total_routed += 1

        route.call_count += 1

        command.status = (
            CommandStatus.EXECUTING
        )

        try:

            result = (
                await self._invoke_handler(
                    route.handler,
                    command,
                )
            )

            command.status = (
                CommandStatus.COMPLETED
            )

            self.total_completed += 1

            return self._record_result(
                CommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.COMPLETED,
                    success=True,
                    result=result,
                    route=route.name,
                    duration=(
                        time.time()
                        - start
                    ),
                )
            )

        except Exception as exc:

            route.error_count += 1

            self.total_failed += 1

            command.status = (
                CommandStatus.FAILED
            )

            self.logger.exception(
                "Command execution failed: %s",
                command.command_name,
            )

            return self._record_result(
                CommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.FAILED,
                    success=False,
                    error=str(exc),
                    route=route.name,
                    duration=(
                        time.time()
                        - start
                    ),
                )
            )

    # ========================================================================
    # INVOKE HANDLER
    # ========================================================================

    async def _invoke_handler(
        self,
        handler: CommandHandler,
        command: RENIXCommand,
    ) -> Any:
        """
        Invoke synchronous or asynchronous command handlers.

        Handler compatibility:

            handler()

            handler(command)

            handler(*command.arguments)

            handler(command, *command.arguments)

        The preferred RENIX form is:

            async def handler(command):
                ...
        """

        try:

            signature = inspect.signature(
                handler
            )

            parameters = list(
                signature.parameters.values()
            )

        except (
            TypeError,
            ValueError,
        ):

            parameters = []

        # --------------------------------------------------------------------
        # No parameters
        # --------------------------------------------------------------------

        if not parameters:

            result = handler()

        # --------------------------------------------------------------------
        # Handler accepts command
        # --------------------------------------------------------------------

        elif len(parameters) == 1:

            parameter = parameters[0]

            if (
                parameter.kind
                in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            ):

                result = handler(
                    command
                )

            else:

                result = handler(
                    command
                )

        # --------------------------------------------------------------------
        # Handler accepts command + arguments
        # --------------------------------------------------------------------

        else:

            result = handler(
                command,
                *command.arguments,
            )

        if inspect.isawaitable(
            result
        ):

            return await result

        return result

    # ========================================================================
    # ROUTE RAW COMMAND
    # ========================================================================

    async def route_raw(
        self,
        command_name: str,
        *,
        raw_text: str = "",
        arguments: Optional[
            list[Any]
        ] = None,
        parameters: Optional[
            dict[str, Any]
        ] = None,
        source: str = "unknown",
        user_id: Optional[str] = None,
        priority: CommandPriority = (
            CommandPriority.NORMAL
        ),
        metadata: Optional[
            dict[str, Any]
        ] = None,
        confirmed: bool = False,
    ) -> CommandResult:
        """
        Create and immediately route a command.
        """

        command = self.create_command(
            command_name=command_name,
            raw_text=raw_text,
            arguments=arguments,
            parameters=parameters,
            source=source,
            user_id=user_id,
            priority=priority,
            metadata=metadata,
            confirmed=confirmed,
        )

        return await self.route(
            command
        )

    # ========================================================================
    # RECORD RESULT
    # ========================================================================

    def _record_result(
        self,
        result: CommandResult,
    ) -> CommandResult:
        """
        Store a command result in history.
        """

        self.history.append(
            result
        )

        if len(
            self.history
        ) > self.max_history:

            self.history = self.history[
                -self.max_history:
            ]

        return result

    # ========================================================================
    # GET HISTORY
    # ========================================================================

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[CommandResult]:
        """
        Return command history.
        """

        if limit is None:

            return list(
                self.history
            )

        limit = max(
            0,
            int(limit),
        )

        return list(
            self.history[
                -limit:
            ]
        )

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    def clear_history(
        self,
    ) -> None:
        """
        Clear command history.
        """

        self.history.clear()

    # ========================================================================
    # FIND ROUTES
    # ========================================================================

    def find_routes(
        self,
        *,
        category: Optional[str] = None,
        enabled_only: bool = False,
    ) -> list[CommandRoute]:
        """
        Find registered routes.
        """

        routes = list(
            self.routes.values()
        )

        if category is not None:

            category_lower = (
                category.lower()
            )

            routes = [
                route
                for route in routes
                if route.category.lower()
                == category_lower
            ]

        if enabled_only:

            routes = [
                route
                for route in routes
                if route.enabled
            ]

        routes.sort(
            key=lambda route: (
                -int(
                    route.priority.value
                ),
                route.name,
            )
        )

        return routes

    # ========================================================================
    # SEARCH COMMANDS
    # ========================================================================

    def search_commands(
        self,
        query: str,
    ) -> list[CommandRoute]:
        """
        Search registered commands by name,
        alias, category, or description.
        """

        query = (
            query.strip().lower()
        )

        if not query:

            return []

        results: list[
            CommandRoute
        ] = []

        for route in (
            self.routes.values()
        ):

            searchable = " ".join(
                [
                    route.name,
                    *route.aliases,
                    route.category,
                    route.description,
                ]
            ).lower()

            if query in searchable:

                results.append(
                    route
                )

        results.sort(
            key=lambda route: (
                -int(
                    route.priority.value
                ),
                route.name,
            )
        )

        return results

    # ========================================================================
    # ROUTE EXISTS
    # ========================================================================

    def exists(
        self,
        name: str,
    ) -> bool:
        """
        Check whether a route exists.
        """

        return (
            self.get_route(name)
            is not None
        )

    # ========================================================================
    # GET ROUTE COUNT
    # ========================================================================

    def route_count(
        self,
    ) -> int:
        """
        Return number of registered routes.
        """

        return len(
            self.routes
        )

    # ========================================================================
    # GET ENABLED ROUTES
    # ========================================================================

    def enabled_routes(
        self,
    ) -> list[CommandRoute]:
        """
        Return enabled routes.
        """

        return [
            route
            for route in self.routes.values()
            if route.enabled
        ]

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return router statistics.
        """

        return {
            "registered_routes": len(
                self.routes
            ),
            "registered_aliases": len(
                self.alias_map
            ),
            "middleware_count": len(
                self.middlewares
            ),
            "permission_checker_count": len(
                self.permission_checkers
            ),
            "history_size": len(
                self.history
            ),
            "total_received": (
                self.total_received
            ),
            "total_routed": (
                self.total_routed
            ),
            "total_completed": (
                self.total_completed
            ),
            "total_failed": (
                self.total_failed
            ),
            "total_blocked": (
                self.total_blocked
            ),
            "total_unknown": (
                self.total_unknown
            ),
        }

    # ========================================================================
    # EXPORT ROUTES
    # ========================================================================

    def export_routes(
        self,
    ) -> list[dict[str, Any]]:
        """
        Export all registered routes.
        """

        return [
            route.to_dict()
            for route in self.routes.values()
        ]

    # ========================================================================
    # EXPORT HISTORY
    # ========================================================================

    def export_history(
        self,
    ) -> list[dict[str, Any]]:
        """
        Export command history.
        """

        return [
            result.to_dict()
            for result in self.history
        ]


# ============================================================================
# GLOBAL COMMAND ROUTER
# ============================================================================

command_router = CommandRouter()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def register_command(
    name: str,
    handler: CommandHandler,
    *,
    aliases: Optional[
        list[str]
    ] = None,
    priority: CommandPriority = (
        CommandPriority.NORMAL
    ),
    description: str = "",
    category: str = "general",
    requires_confirmation: bool = False,
    permission: Optional[str] = None,
) -> CommandRoute:
    """
    Register a command using the global router.
    """

    return await command_router.register(
        name=name,
        handler=handler,
        aliases=aliases,
        priority=priority,
        description=description,
        category=category,
        requires_confirmation=(
            requires_confirmation
        ),
        permission=permission,
    )


async def route_command(
    command_name: str,
    *,
    raw_text: str = "",
    arguments: Optional[
        list[Any]
    ] = None,
    parameters: Optional[
        dict[str, Any]
    ] = None,
    source: str = "unknown",
    user_id: Optional[str] = None,
    priority: CommandPriority = (
        CommandPriority.NORMAL
    ),
    metadata: Optional[
        dict[str, Any]
    ] = None,
    confirmed: bool = False,
) -> CommandResult:
    """
    Route a command through the global router.
    """

    return await command_router.route_raw(
        command_name=command_name,
        raw_text=raw_text,
        arguments=arguments,
        parameters=parameters,
        source=source,
        user_id=user_id,
        priority=priority,
        metadata=metadata,
        confirmed=confirmed,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CommandHandler",
    "CommandMiddleware",
    "CommandPermissionChecker",
    "CommandStatus",
    "CommandPriority",
    "RENIXCommand",
    "CommandRoute",
    "CommandResult",
    "CommandRouter",
    "command_router",
    "register_command",
    "route_command",
]


