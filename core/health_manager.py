"""
RENIX AI
Core Health Manager

Responsible for monitoring the health and operational state of
RENIX core components, services, providers, agents and modules.

This module is intentionally self-contained so it can be used by
the orchestrator without requiring other RENIX modules to exist.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional


logger = logging.getLogger("RENIX.HealthManager")


# ============================================================================
# ENUMS
# ============================================================================

class HealthStatus(str, Enum):
    """
    Overall health state of a component.
    """

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"


class HealthSeverity(str, Enum):
    """
    Severity of a health problem.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# HEALTH RESULT
# ============================================================================

@dataclass
class HealthResult:
    """
    Result returned by a health check.
    """

    name: str

    status: HealthStatus

    message: str = ""

    severity: HealthSeverity = HealthSeverity.INFO

    latency_ms: float = 0.0

    checked_at: float = field(
        default_factory=time.time
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    # ========================================================================
    # TO DICT
    # ========================================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert the health result into a serializable dictionary.
        """

        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "severity": self.severity.value,
            "latency_ms": self.latency_ms,
            "checked_at": self.checked_at,
            "details": dict(self.details),
            "error": self.error,
        }


# ============================================================================
# HEALTH COMPONENT
# ============================================================================

@dataclass
class HealthComponent:
    """
    Registered health-monitored component.
    """

    name: str

    checker: Any

    critical: bool = False

    enabled: bool = True

    interval: float = 30.0

    timeout: float = 10.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    last_result: Optional[HealthResult] = None

    check_count: int = 0

    failure_count: int = 0

    success_count: int = 0

    consecutive_failures: int = 0

    consecutive_successes: int = 0

    registered_at: float = field(
        default_factory=time.time
    )

    last_checked_at: Optional[float] = None

    # ========================================================================
    # TO DICT
    # ========================================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert component information to a dictionary.
        """

        return {
            "name": self.name,
            "critical": self.critical,
            "enabled": self.enabled,
            "interval": self.interval,
            "timeout": self.timeout,
            "metadata": dict(self.metadata),
            "last_result": (
                self.last_result.to_dict()
                if self.last_result
                else None
            ),
            "check_count": self.check_count,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "consecutive_failures": (
                self.consecutive_failures
            ),
            "consecutive_successes": (
                self.consecutive_successes
            ),
            "registered_at": self.registered_at,
            "last_checked_at": self.last_checked_at,
        }


# ============================================================================
# HEALTH MANAGER
# ============================================================================

class HealthManager:
    """
    Central health monitoring system for RENIX.

    Supports:

    - Component registration
    - Custom health check functions
    - Async health checks
    - Sync health checks
    - Timeouts
    - Critical components
    - Health aggregation
    - Failure tracking
    - Recovery tracking
    - Continuous monitoring
    - Health snapshots
    - System-wide health status
    """

    def __init__(
        self,
        *,
        default_interval: float = 30.0,
        default_timeout: float = 10.0,
    ) -> None:

        self.logger = logger

        self.components: dict[
            str,
            HealthComponent,
        ] = {}

        self.default_interval = max(
            0.1,
            float(default_interval),
        )

        self.default_timeout = max(
            0.1,
            float(default_timeout),
        )

        self.initialized = False

        self.monitoring = False

        self.monitor_task: Optional[
            asyncio.Task
        ] = None

        self.started_at = time.time()

        self.last_system_check: Optional[
            float
        ] = None

        self.total_checks = 0

        self.total_failures = 0

        self._stop_event: Optional[
            asyncio.Event
        ] = None

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the health manager.
        """

        if self.initialized:
            return

        self.initialized = True

        self._stop_event = asyncio.Event()

        self.logger.info(
            "RENIX Health Manager initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shutdown health monitoring.
        """

        await self.stop_monitoring()

        self.initialized = False

        self.logger.info(
            "RENIX Health Manager shutdown."
        )

    # ========================================================================
    # REGISTER
    # ========================================================================

    def register(
        self,
        name: str,
        checker: Any,
        *,
        critical: bool = False,
        enabled: bool = True,
        interval: Optional[float] = None,
        timeout: Optional[float] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
        replace: bool = False,
    ) -> HealthComponent:
        """
        Register a component health checker.

        The checker may be:

        - A callable function
        - An async callable
        - An object with health_check()
        - An object with check_health()
        - An object with is_healthy()
        """

        normalized_name = (
            self._normalize_name(name)
        )

        if not normalized_name:
            raise ValueError(
                "Health component name cannot be empty."
            )

        if (
            normalized_name in self.components
            and not replace
        ):
            raise ValueError(
                f"Health component already registered: "
                f"{normalized_name}"
            )

        component = HealthComponent(
            name=normalized_name,
            checker=checker,
            critical=bool(critical),
            enabled=bool(enabled),
            interval=(
                max(
                    0.1,
                    float(interval),
                )
                if interval is not None
                else self.default_interval
            ),
            timeout=(
                max(
                    0.1,
                    float(timeout),
                )
                if timeout is not None
                else self.default_timeout
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        self.components[
            normalized_name
        ] = component

        self.logger.info(
            "Registered health component '%s'.",
            normalized_name,
        )

        return component

    # ========================================================================
    # UNREGISTER
    # ========================================================================

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Unregister a health component.
        """

        normalized_name = (
            self._normalize_name(name)
        )

        if normalized_name not in self.components:
            return False

        self.components.pop(
            normalized_name
        )

        self.logger.info(
            "Unregistered health component '%s'.",
            normalized_name,
        )

        return True

    # ========================================================================
    # EXISTS
    # ========================================================================

    def exists(
        self,
        name: str,
    ) -> bool:
        """
        Check whether a component exists.
        """

        return (
            self._normalize_name(name)
            in self.components
        )

    # ========================================================================
    # ENABLE
    # ========================================================================

    def enable(
        self,
        name: str,
    ) -> bool:
        """
        Enable health monitoring for a component.
        """

        component = self.components.get(
            self._normalize_name(name)
        )

        if component is None:
            return False

        component.enabled = True

        return True

    # ========================================================================
    # DISABLE
    # ========================================================================

    def disable(
        self,
        name: str,
    ) -> bool:
        """
        Disable health monitoring for a component.
        """

        component = self.components.get(
            self._normalize_name(name)
        )

        if component is None:
            return False

        component.enabled = False

        return True

    # ========================================================================
    # GET
    # ========================================================================

    def get(
        self,
        name: str,
    ) -> Optional[HealthComponent]:
        """
        Return a registered component.
        """

        return self.components.get(
            self._normalize_name(name)
        )

    # ========================================================================
    # CHECK ONE
    # ========================================================================

    async def check(
        self,
        name: str,
    ) -> HealthResult:
        """
        Execute a health check for one component.
        """

        component = self.get(name)

        if component is None:

            return HealthResult(
                name=self._normalize_name(name),
                status=HealthStatus.UNKNOWN,
                message="Component is not registered.",
                severity=HealthSeverity.ERROR,
            )

        if not component.enabled:

            result = HealthResult(
                name=component.name,
                status=HealthStatus.DISABLED,
                message="Health monitoring is disabled.",
                severity=HealthSeverity.INFO,
            )

            component.last_result = result

            return result

        started = time.perf_counter()

        component.check_count += 1

        component.last_checked_at = time.time()

        self.total_checks += 1

        try:

            check_callable = (
                self._resolve_checker(
                    component.checker
                )
            )

            if check_callable is None:

                raise TypeError(
                    f"No health-check method found "
                    f"for '{component.name}'."
                )

            result = check_callable()

            if inspect.isawaitable(
                result
            ):

                result = await asyncio.wait_for(
                    result,
                    timeout=component.timeout,
                )

            else:

                # Run synchronous check in a worker
                # so it does not block the async loop.
                if callable(
                    check_callable
                ):

                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._safe_sync_result,
                            result,
                        ),
                        timeout=component.timeout,
                    )

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            health_result = (
                self._normalize_result(
                    component.name,
                    result,
                    latency_ms,
                )
            )

            self._record_result(
                component,
                health_result,
            )

            return health_result

        except asyncio.TimeoutError:

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            result = HealthResult(
                name=component.name,
                status=HealthStatus.UNHEALTHY,
                message=(
                    "Health check timed out."
                ),
                severity=(
                    HealthSeverity.CRITICAL
                    if component.critical
                    else HealthSeverity.ERROR
                ),
                latency_ms=latency_ms,
                error=(
                    f"Timeout after "
                    f"{component.timeout:.2f}s"
                ),
            )

            self._record_result(
                component,
                result,
            )

            return result

        except Exception as exc:

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            result = HealthResult(
                name=component.name,
                status=HealthStatus.UNHEALTHY,
                message=(
                    "Health check raised an exception."
                ),
                severity=(
                    HealthSeverity.CRITICAL
                    if component.critical
                    else HealthSeverity.ERROR
                ),
                latency_ms=latency_ms,
                error=str(exc),
            )

            self._record_result(
                component,
                result,
            )

            self.logger.exception(
                "Health check failed for '%s'.",
                component.name,
            )

            return result

    # ========================================================================
    # CHECK ALL
    # ========================================================================

    async def check_all(
        self,
    ) -> dict[str, HealthResult]:
        """
        Check every enabled component.
        """

        names = [
            component.name
            for component
            in self.components.values()
            if component.enabled
        ]

        if not names:
            return {}

        results = await asyncio.gather(
            *[
                self.check(name)
                for name in names
            ],
            return_exceptions=False,
        )

        self.last_system_check = time.time()

        return {
            result.name: result
            for result in results
        }

    # ========================================================================
    # CHECK CRITICAL
    # ========================================================================

    async def check_critical(
        self,
    ) -> dict[str, HealthResult]:
        """
        Check only critical components.
        """

        names = [
            component.name
            for component
            in self.components.values()
            if (
                component.enabled
                and component.critical
            )
        ]

        if not names:
            return {}

        results = await asyncio.gather(
            *[
                self.check(name)
                for name in names
            ]
        )

        return {
            result.name: result
            for result in results
        }

    # ========================================================================
    # SYSTEM STATUS
    # ========================================================================

    def system_status(
        self,
    ) -> HealthStatus:
        """
        Calculate the overall RENIX health state.
        """

        enabled = [
            component
            for component
            in self.components.values()
            if component.enabled
        ]

        if not enabled:
            return HealthStatus.UNKNOWN

        results = [
            component.last_result
            for component
            in enabled
            if component.last_result is not None
        ]

        if not results:
            return HealthStatus.UNKNOWN

        critical_unhealthy = any(
            component.critical
            and component.last_result is not None
            and component.last_result.status
            == HealthStatus.UNHEALTHY
            for component
            in enabled
        )

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        if any(
            result.status
            == HealthStatus.UNHEALTHY
            for result in results
        ):
            return HealthStatus.DEGRADED

        if any(
            result.status
            == HealthStatus.DEGRADED
            for result in results
        ):
            return HealthStatus.DEGRADED

        if all(
            result.status
            == HealthStatus.HEALTHY
            for result in results
        ):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    # ========================================================================
    # SYSTEM HEALTH
    # ========================================================================

    def system_health(
        self,
    ) -> dict[str, Any]:
        """
        Return complete system health information.
        """

        component_data = {
            name: component.to_dict()
            for name, component
            in self.components.items()
        }

        healthy = sum(
            1
            for component
            in self.components.values()
            if (
                component.last_result is not None
                and component.last_result.status
                == HealthStatus.HEALTHY
            )
        )

        degraded = sum(
            1
            for component
            in self.components.values()
            if (
                component.last_result is not None
                and component.last_result.status
                == HealthStatus.DEGRADED
            )
        )

        unhealthy = sum(
            1
            for component
            in self.components.values()
            if (
                component.last_result is not None
                and component.last_result.status
                == HealthStatus.UNHEALTHY
            )
        )

        return {
            "status": self.system_status().value,
            "initialized": self.initialized,
            "monitoring": self.monitoring,
            "component_count": len(
                self.components
            ),
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "total_checks": self.total_checks,
            "total_failures": self.total_failures,
            "last_system_check": (
                self.last_system_check
            ),
            "components": component_data,
        }

    # ========================================================================
    # START MONITORING
    # ========================================================================

    async def start_monitoring(
        self,
    ) -> None:
        """
        Start continuous background health monitoring.
        """

        if self.monitoring:
            return

        if not self.initialized:
            await self.initialize()

        if self._stop_event is None:
            self._stop_event = asyncio.Event()

        self._stop_event.clear()

        self.monitoring = True

        self.monitor_task = asyncio.create_task(
            self._monitor_loop()
        )

        self.logger.info(
            "RENIX health monitoring started."
        )

    # ========================================================================
    # STOP MONITORING
    # ========================================================================

    async def stop_monitoring(
        self,
    ) -> None:
        """
        Stop continuous health monitoring.
        """

        if not self.monitoring:
            return

        self.monitoring = False

        if self._stop_event is not None:
            self._stop_event.set()

        task = self.monitor_task

        self.monitor_task = None

        if (
            task is not None
            and task is not asyncio.current_task()
        ):

            try:

                await task

            except asyncio.CancelledError:

                pass

        self.logger.info(
            "RENIX health monitoring stopped."
        )

    # ========================================================================
    # MONITOR LOOP
    # ========================================================================

    async def _monitor_loop(
        self,
    ) -> None:
        """
        Background monitoring loop.
        """

        while self.monitoring:

            try:

                await self.check_all()

            except Exception:

                self.logger.exception(
                    "Unexpected error in health monitor loop."
                )

            interval = (
                self._get_monitor_interval()
            )

            if self._stop_event is None:

                await asyncio.sleep(
                    interval
                )

            else:

                try:

                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=interval,
                    )

                except asyncio.TimeoutError:

                    pass

    # ========================================================================
    # MONITOR INTERVAL
    # ========================================================================

    def _get_monitor_interval(
        self,
    ) -> float:
        """
        Calculate a sensible monitoring interval.
        """

        intervals = [
            component.interval
            for component
            in self.components.values()
            if component.enabled
        ]

        if not intervals:
            return self.default_interval

        return max(
            0.1,
            min(intervals),
        )

    # ========================================================================
    # RECORD RESULT
    # ========================================================================

    def _record_result(
        self,
        component: HealthComponent,
        result: HealthResult,
    ) -> None:
        """
        Update component statistics after a check.
        """

        previous = component.last_result

        component.last_result = result

        if result.status == HealthStatus.HEALTHY:

            component.success_count += 1

            component.consecutive_successes += 1

            component.consecutive_failures = 0

            if (
                previous is not None
                and previous.status
                == HealthStatus.UNHEALTHY
            ):

                self.logger.info(
                    "Component '%s' recovered.",
                    component.name,
                )

        elif result.status in (
            HealthStatus.UNHEALTHY,
            HealthStatus.DEGRADED,
        ):

            component.failure_count += 1

            component.consecutive_failures += 1

            component.consecutive_successes = 0

            self.total_failures += 1

            if (
                previous is not None
                and previous.status
                == HealthStatus.HEALTHY
            ):

                self.logger.warning(
                    "Component '%s' became unhealthy.",
                    component.name,
                )

    # ========================================================================
    # RESOLVE CHECKER
    # ========================================================================

    @staticmethod
    def _resolve_checker(
        checker: Any,
    ) -> Optional[
        Callable[..., Any]
    ]:
        """
        Resolve a health-check callable.
        """

        if callable(checker):

            return checker

        for method_name in (
            "health_check",
            "check_health",
            "is_healthy",
            "health",
            "check",
        ):

            candidate = getattr(
                checker,
                method_name,
                None,
            )

            if callable(candidate):

                return candidate

        return None

    # ========================================================================
    # NORMALIZE RESULT
    # ========================================================================

    @staticmethod
    def _normalize_result(
        name: str,
        result: Any,
        latency_ms: float,
    ) -> HealthResult:
        """
        Convert different checker return values into
        a standard HealthResult.
        """

        if isinstance(
            result,
            HealthResult,
        ):

            result.name = name

            result.latency_ms = latency_ms

            result.checked_at = time.time()

            return result

        if isinstance(
            result,
            dict,
        ):

            raw_status = result.get(
                "status",
                HealthStatus.HEALTHY,
            )

            try:

                status = (
                    raw_status
                    if isinstance(
                        raw_status,
                        HealthStatus,
                    )
                    else HealthStatus(
                        str(
                            raw_status
                        ).lower()
                    )
                )

            except ValueError:

                status = (
                    HealthStatus.HEALTHY
                    if result.get(
                        "healthy",
                        True,
                    )
                    else HealthStatus.UNHEALTHY
                )

            raw_severity = result.get(
                "severity",
                HealthSeverity.INFO,
            )

            try:

                severity = (
                    raw_severity
                    if isinstance(
                        raw_severity,
                        HealthSeverity,
                    )
                    else HealthSeverity(
                        str(
                            raw_severity
                        ).lower()
                    )
                )

            except ValueError:

                severity = HealthSeverity.INFO

            return HealthResult(
                name=name,
                status=status,
                message=str(
                    result.get(
                        "message",
                        "",
                    )
                ),
                severity=severity,
                latency_ms=latency_ms,
                details=dict(
                    result.get(
                        "details",
                        {},
                    )
                ),
                error=result.get(
                    "error"
                ),
            )

        if isinstance(
            result,
            bool,
        ):

            return HealthResult(
                name=name,
                status=(
                    HealthStatus.HEALTHY
                    if result
                    else HealthStatus.UNHEALTHY
                ),
                message=(
                    "Health check passed."
                    if result
                    else "Health check failed."
                ),
                severity=(
                    HealthSeverity.INFO
                    if result
                    else HealthSeverity.ERROR
                ),
                latency_ms=latency_ms,
            )

        if result is None:

            return HealthResult(
                name=name,
                status=HealthStatus.HEALTHY,
                message=(
                    "Health check completed successfully."
                ),
                severity=HealthSeverity.INFO,
                latency_ms=latency_ms,
            )

        return HealthResult(
            name=name,
            status=HealthStatus.HEALTHY,
            message=str(result),
            severity=HealthSeverity.INFO,
            latency_ms=latency_ms,
            details={
                "raw_result": result
            },
        )

    # ========================================================================
    # SAFE SYNC RESULT
    # ========================================================================

    @staticmethod
    def _safe_sync_result(
        result: Any,
    ) -> Any:
        """
        Return an already calculated synchronous result.

        Kept separate so the synchronous result processing
        can safely run through asyncio.to_thread().
        """

        return result

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:
        """
        Return a lightweight health snapshot.
        """

        return {
            "status": self.system_status().value,
            "initialized": self.initialized,
            "monitoring": self.monitoring,
            "components": {
                name: (
                    component.last_result.status.value
                    if component.last_result
                    else HealthStatus.UNKNOWN.value
                )
                for name, component
                in self.components.items()
            },
        }

    # ========================================================================
    # RESET STATISTICS
    # ========================================================================

    def reset_statistics(
        self,
    ) -> None:
        """
        Reset health counters while preserving registrations.
        """

        self.total_checks = 0

        self.total_failures = 0

        for component in (
            self.components.values()
        ):

            component.check_count = 0

            component.failure_count = 0

            component.success_count = 0

            component.consecutive_failures = 0

            component.consecutive_successes = 0

    # ========================================================================
    # LIST COMPONENTS
    # ========================================================================

    def list_components(
        self,
        *,
        enabled_only: bool = False,
        critical_only: bool = False,
    ) -> list[str]:
        """
        List registered health components.
        """

        result: list[str] = []

        for component in (
            self.components.values()
        ):

            if (
                enabled_only
                and not component.enabled
            ):
                continue

            if (
                critical_only
                and not component.critical
            ):
                continue

            result.append(
                component.name
            )

        return result

    # ========================================================================
    # NORMALIZE NAME
    # ========================================================================

    @staticmethod
    def _normalize_name(
        name: Any,
    ) -> str:
        """
        Normalize component names.
        """

        if name is None:
            return ""

        return str(
            name
        ).strip().lower()


# ============================================================================
# GLOBAL HEALTH MANAGER
# ============================================================================

health_manager = HealthManager()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def register_health_check(
    name: str,
    checker: Any,
    **kwargs: Any,
) -> HealthComponent:
    """
    Register a health check using the global manager.
    """

    return health_manager.register(
        name,
        checker,
        **kwargs,
    )


async def check_health(
    name: str,
) -> HealthResult:
    """
    Check one component using the global manager.
    """

    return await health_manager.check(
        name
    )


async def check_all_health(
) -> dict[str, HealthResult]:
    """
    Check all components using the global manager.
    """

    return await health_manager.check_all()


def get_system_health(
) -> dict[str, Any]:
    """
    Get complete system health from the global manager.
    """

    return health_manager.system_health()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "HealthStatus",
    "HealthSeverity",
    "HealthResult",
    "HealthComponent",
    "HealthManager",
    "health_manager",
    "register_health_check",
    "check_health",
    "check_all_health",
    "get_system_health",
]


