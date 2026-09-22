"""
RENIX System Monitor
====================

Central coordinator for all system-monitoring modules.

Responsibilities:
- Collect CPU information
- Collect RAM information
- Collect disk information
- Collect network information
- Collect battery information
- Collect process information
- Collect temperature information
- Collect GPU information when available
- Produce a unified system snapshot
- Track monitoring history
- Detect basic performance problems
"""

from __future__ import annotations

import platform
import socket
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class SystemSnapshot:
    """A complete snapshot of the current system state."""

    timestamp: str

    system: dict[str, Any] = field(
        default_factory=dict
    )

    cpu: dict[str, Any] = field(
        default_factory=dict
    )

    gpu: dict[str, Any] = field(
        default_factory=dict
    )

    memory: dict[str, Any] = field(
        default_factory=dict
    )

    disk: dict[str, Any] = field(
        default_factory=dict
    )

    network: dict[str, Any] = field(
        default_factory=dict
    )

    battery: dict[str, Any] = field(
        default_factory=dict
    )

    processes: dict[str, Any] = field(
        default_factory=dict
    )

    temperatures: dict[str, Any] = field(
        default_factory=dict
    )

    alerts: list[dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SystemMonitor:
    """
    Main monitoring coordinator for RENIX.

    Example:

        monitor = SystemMonitor()

        snapshot = monitor.snapshot()

        print(snapshot.cpu)
        print(snapshot.memory)

    The monitor is designed so individual monitoring modules
    can later be replaced with more advanced implementations.
    """

    def __init__(
        self,
        *,
        history_size: int = 120,
        auto_initialize: bool = True,
    ) -> None:

        self.history_size = max(
            1,
            int(history_size),
        )

        self.history: deque[
            SystemSnapshot
        ] = deque(
            maxlen=self.history_size
        )

        self.running = False

        self.last_snapshot: (
            SystemSnapshot | None
        ) = None

        self.last_update_time: (
            float | None
        ) = None

        self.thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 90.0,
            "disk_percent": 90.0,
            "battery_percent": 15.0,
            "temperature_celsius": 85.0,
        }

        self._components: dict[
            str,
            Any,
        ] = {}

        if auto_initialize:
            self._initialize_components()

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize_components(
        self,
    ) -> None:

        """
        Initialize specialized monitoring components.

        Imports are intentionally lazy so RENIX can still start
        in a reduced mode if one optional monitor is unavailable.
        """

        component_modules = {
            "cpu": (
                ".cpu_monitor",
                "CPUMonitor",
            ),
            "gpu": (
                ".gpu_monitor",
                "GPUMonitor",
            ),
            "memory": (
                ".ram_monitor",
                "RAMMonitor",
            ),
            "disk": (
                ".disk_monitor",
                "DiskMonitor",
            ),
            "network": (
                ".network_monitor",
                "NetworkMonitor",
            ),
            "battery": (
                ".battery_monitor",
                "BatteryMonitor",
            ),
            "processes": (
                ".process_monitor",
                "ProcessMonitor",
            ),
            "temperatures": (
                ".temperature_monitor",
                "TemperatureMonitor",
            ),
        }

        for name, (
            module_name,
            class_name,
        ) in component_modules.items():

            try:

                import importlib

                module = importlib.import_module(
                    module_name,
                    package=__package__,
                )

                monitor_class = getattr(
                    module,
                    class_name,
                )

                self._components[
                    name
                ] = monitor_class()

            except Exception:

                # A missing optional monitor should not
                # prevent the whole RENIX system from starting.
                self._components[
                    name
                ] = None

    # ============================================================
    # PUBLIC SNAPSHOT API
    # ============================================================

    def snapshot(
        self,
        *,
        store: bool = True,
    ) -> SystemSnapshot:

        """
        Collect a complete system snapshot.
        """

        current = SystemSnapshot(
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),
        )

        current.system = (
            self._collect_system_info()
        )

        current.cpu = (
            self._collect_component(
                "cpu",
                "get_stats",
                fallback=self._fallback_cpu,
            )
        )

        current.gpu = (
            self._collect_component(
                "gpu",
                "get_stats",
                fallback=self._fallback_gpu,
            )
        )

        current.memory = (
            self._collect_component(
                "memory",
                "get_stats",
                fallback=self._fallback_memory,
            )
        )

        current.disk = (
            self._collect_component(
                "disk",
                "get_stats",
                fallback=self._fallback_disk,
            )
        )

        current.network = (
            self._collect_component(
                "network",
                "get_stats",
                fallback=self._fallback_network,
            )
        )

        current.battery = (
            self._collect_component(
                "battery",
                "get_stats",
                fallback=self._fallback_battery,
            )
        )

        current.processes = (
            self._collect_component(
                "processes",
                "get_stats",
                fallback=self._fallback_processes,
            )
        )

        current.temperatures = (
            self._collect_component(
                "temperatures",
                "get_stats",
                fallback=self._fallback_temperatures,
            )
        )

        current.alerts = (
            self._generate_alerts(
                current
            )
        )

        self.last_snapshot = current
        self.last_update_time = time.time()

        if store:
            self.history.append(
                current
            )

        return current

    def get_stats(
        self,
    ) -> dict[str, Any]:

        """
        Return the complete monitoring state as a dictionary.
        """

        return self.snapshot().to_dict()

    def update(
        self,
    ) -> SystemSnapshot:

        """
        Alias for snapshot().
        """

        return self.snapshot()

    # ============================================================
    # INDIVIDUAL MONITORS
    # ============================================================

    def cpu(
        self,
    ) -> dict[str, Any]:

        return self._collect_component(
            "cpu",
            "get_stats",
            fallback=self._fallback_cpu,
        )

    def gpu(
        self,
    ) -> dict[str, Any]:

        return self._collect_component(
            "gpu",
            "get_stats",
            fallback=self._fallback_gpu,
        )

    def memory(
        self,
    ) -> dict[str, Any]:

        return self._collect_component(
            "memory",
            "get_stats",
            fallback=self._fallback_memory,
        )

    def disk(
        self,
    ) -> dict[str, Any]:

        return self._collect_component(
            "disk",
            "get_stats",
            fallback=self._fallback_disk,
        )

    def network(
        self,
    ) -> dict[str, Any]:

        return self._collect_component(
            "network",
            "get_stats",
            fallback=self._fallback_network,
        )

    def battery(
        self,
    ) -> dict[str, Any]:

        return self._collect_component(
            "battery",
            "get_stats",
            fallback=self._fallback_battery,
        )

    def processes(
        self,
    ) -> dict[str, Any]:

        return self._collect_component(
            "processes",
            "get_stats",
            fallback=self._fallback_processes,
        )

    def temperatures(
        self,
    ) -> dict[str, Any]:

        return self._collect_component(
            "temperatures",
            "get_stats",
            fallback=self._fallback_temperatures,
        )

    # ============================================================
    # MONITORING LOOP
    # ============================================================

    def start(
        self,
        *,
        interval: float = 2.0,
        callback: Any = None,
    ) -> None:

        """
        Start a blocking monitoring loop.

        callback:
            Optional callable receiving each SystemSnapshot.

        Stop the loop by calling stop() from another thread,
        or by raising KeyboardInterrupt.
        """

        self.running = True

        interval = max(
            0.1,
            float(interval),
        )

        try:

            while self.running:

                current = self.snapshot()

                if callback:

                    try:
                        callback(
                            current
                        )

                    except Exception:
                        # Callback errors must not kill
                        # the monitoring service.
                        pass

                time.sleep(
                    interval
                )

        except KeyboardInterrupt:

            self.stop()

    def stop(
        self,
    ) -> None:

        self.running = False

    # ============================================================
    # HISTORY
    # ============================================================

    def get_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[SystemSnapshot]:

        history = list(
            self.history
        )

        if limit is not None:

            limit = max(
                0,
                int(limit),
            )

            if limit == 0:
                return []

            return history[-limit:]

        return history

    def clear_history(
        self,
    ) -> None:

        self.history.clear()

    # ============================================================
    # THRESHOLDS
    # ============================================================

    def set_threshold(
        self,
        name: str,
        value: float,
    ) -> None:

        if name not in self.thresholds:
            raise KeyError(
                f"Unknown threshold: {name}"
            )

        self.thresholds[
            name
        ] = float(value)

    def get_thresholds(
        self,
    ) -> dict[str, float]:

        return dict(
            self.thresholds
        )

    # ============================================================
    # ALERTS
    # ============================================================

    def get_alerts(
        self,
    ) -> list[dict[str, Any]]:

        snapshot = (
            self.last_snapshot
            or self.snapshot()
        )

        return list(
            snapshot.alerts
        )

    def _generate_alerts(
        self,
        snapshot: SystemSnapshot,
    ) -> list[dict[str, Any]]:

        alerts: list[
            dict[str, Any]
        ] = []

        # --------------------------------------------------------
        # CPU
        # --------------------------------------------------------

        cpu_percent = self._number(
            snapshot.cpu.get(
                "percent"
            )
        )

        if (
            cpu_percent is not None
            and cpu_percent
            >= self.thresholds[
                "cpu_percent"
            ]
        ):

            alerts.append(
                {
                    "type": "cpu",
                    "severity": "warning",
                    "message": (
                        "CPU usage is high."
                    ),
                    "value": cpu_percent,
                }
            )

        # --------------------------------------------------------
        # RAM
        # --------------------------------------------------------

        memory_percent = self._number(
            snapshot.memory.get(
                "percent"
            )
        )

        if (
            memory_percent is not None
            and memory_percent
            >= self.thresholds[
                "memory_percent"
            ]
        ):

            alerts.append(
                {
                    "type": "memory",
                    "severity": "warning",
                    "message": (
                        "Memory usage is high."
                    ),
                    "value": memory_percent,
                }
            )

        # --------------------------------------------------------
        # DISK
        # --------------------------------------------------------

        disk_percent = self._number(
            snapshot.disk.get(
                "percent"
            )
        )

        if (
            disk_percent is not None
            and disk_percent
            >= self.thresholds[
                "disk_percent"
            ]
        ):

            alerts.append(
                {
                    "type": "disk",
                    "severity": "warning",
                    "message": (
                        "Disk usage is high."
                    ),
                    "value": disk_percent,
                }
            )

        # --------------------------------------------------------
        # BATTERY
        # --------------------------------------------------------

        battery_percent = self._number(
            snapshot.battery.get(
                "percent"
            )
        )

        charging = snapshot.battery.get(
            "charging"
        )

        if (
            battery_percent is not None
            and battery_percent
            <= self.thresholds[
                "battery_percent"
            ]
            and charging is not True
        ):

            alerts.append(
                {
                    "type": "battery",
                    "severity": "critical",
                    "message": (
                        "Battery level is low."
                    ),
                    "value": battery_percent,
                }
            )

        # --------------------------------------------------------
        # TEMPERATURE
        # --------------------------------------------------------

        temperature = (
            self._highest_temperature(
                snapshot.temperatures
            )
        )

        if (
            temperature is not None
            and temperature
            >= self.thresholds[
                "temperature_celsius"
            ]
        ):

            alerts.append(
                {
                    "type": "temperature",
                    "severity": "critical",
                    "message": (
                        "System temperature "
                        "is high."
                    ),
                    "value": temperature,
                }
            )

        return alerts

    # ============================================================
    # SYSTEM INFORMATION
    # ============================================================

    def _collect_system_info(
        self,
    ) -> dict[str, Any]:

        info = {
            "platform": platform.system(),
            "platform_release": (
                platform.release()
            ),
            "platform_version": (
                platform.version()
            ),
            "architecture": (
                platform.machine()
            ),
            "processor": (
                platform.processor()
            ),
            "hostname": (
                socket.gethostname()
            ),
            "python_version": (
                platform.python_version()
            ),
        }

        if psutil:

            try:

                info[
                    "boot_time"
                ] = datetime.fromtimestamp(
                    psutil.boot_time()
                ).isoformat(
                    timespec="seconds"
                )

                info[
                    "uptime_seconds"
                ] = max(
                    0.0,
                    time.time()
                    - psutil.boot_time(),
                )

            except Exception:
                pass

        return info

    # ============================================================
    # COMPONENT DISPATCH
    # ============================================================

    def _collect_component(
        self,
        name: str,
        method_name: str,
        *,
        fallback: Any,
    ) -> dict[str, Any]:

        component = self._components.get(
            name
        )

        if component:

            try:

                method = getattr(
                    component,
                    method_name,
                )

                result = method()

                if isinstance(
                    result,
                    dict,
                ):
                    return result

            except Exception:
                pass

        try:

            result = fallback()

            if isinstance(
                result,
                dict,
            ):
                return result

        except Exception:
            pass

        return {
            "available": False
        }

    # ============================================================
    # FALLBACK CPU
    # ============================================================

    @staticmethod
    def _fallback_cpu(
    ) -> dict[str, Any]:

        if not psutil:
            return {
                "available": False
            }

        try:

            frequency = (
                psutil.cpu_freq()
            )

            return {
                "available": True,
                "percent": (
                    psutil.cpu_percent(
                        interval=0.1
                    )
                ),
                "per_core": (
                    psutil.cpu_percent(
                        interval=None,
                        percpu=True,
                    )
                ),
                "cores": (
                    psutil.cpu_count(
                        logical=False
                    )
                ),
                "logical_processors": (
                    psutil.cpu_count(
                        logical=True
                    )
                ),
                "frequency_mhz": (
                    frequency.current
                    if frequency
                    else None
                ),
            }

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # FALLBACK GPU
    # ============================================================

    @staticmethod
    def _fallback_gpu(
    ) -> dict[str, Any]:

        """
        GPU information is optional.

        A future GPUMonitor can provide NVIDIA,
        AMD or Intel-specific telemetry.
        """

        return {
            "available": False,
            "message": (
                "No GPU monitor is currently available."
            ),
        }

    # ============================================================
    # FALLBACK MEMORY
    # ============================================================

    @staticmethod
    def _fallback_memory(
    ) -> dict[str, Any]:

        if not psutil:
            return {
                "available": False
            }

        try:

            memory = (
                psutil.virtual_memory()
            )

            swap = (
                psutil.swap_memory()
            )

            return {
                "available": True,
                "total": memory.total,
                "available_bytes": (
                    memory.available
                ),
                "used": memory.used,
                "free": memory.free,
                "percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_free": swap.free,
                "swap_percent": swap.percent,
            }

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # FALLBACK DISK
    # ============================================================

    @staticmethod
    def _fallback_disk(
    ) -> dict[str, Any]:

        if not psutil:
            return {
                "available": False
            }

        try:

            partitions = []

            for partition in (
                psutil.disk_partitions(
                    all=False
                )
            ):

                try:

                    usage = (
                        psutil.disk_usage(
                            partition.mountpoint
                        )
                    )

                    partitions.append(
                        {
                            "device": (
                                partition.device
                            ),
                            "mountpoint": (
                                partition.mountpoint
                            ),
                            "filesystem": (
                                partition.fstype
                            ),
                            "total": (
                                usage.total
                            ),
                            "used": (
                                usage.used
                            ),
                            "free": (
                                usage.free
                            ),
                            "percent": (
                                usage.percent
                            ),
                        }
                    )

                except Exception:
                    continue

            return {
                "available": True,
                "partitions": partitions,
            }

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # FALLBACK NETWORK
    # ============================================================

    @staticmethod
    def _fallback_network(
    ) -> dict[str, Any]:

        if not psutil:
            return {
                "available": False
            }

        try:

            counters = (
                psutil.net_io_counters()
            )

            connections = []

            try:

                for connection in (
                    psutil.net_connections(
                        kind="inet"
                    )
                ):

                    connections.append(
                        {
                            "family": str(
                                connection.family
                            ),
                            "type": str(
                                connection.type
                            ),
                            "status": (
                                connection.status
                            ),
                            "local_address": (
                                str(
                                    connection.laddr
                                )
                                if connection.laddr
                                else None
                            ),
                            "remote_address": (
                                str(
                                    connection.raddr
                                )
                                if connection.raddr
                                else None
                            ),
                        }
                    )

            except Exception:
                pass

            return {
                "available": True,
                "bytes_sent": (
                    counters.bytes_sent
                ),
                "bytes_received": (
                    counters.bytes_recv
                ),
                "packets_sent": (
                    counters.packets_sent
                ),
                "packets_received": (
                    counters.packets_recv
                ),
                "connections": connections,
            }

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # FALLBACK BATTERY
    # ============================================================

    @staticmethod
    def _fallback_battery(
    ) -> dict[str, Any]:

        if not psutil:
            return {
                "available": False
            }

        try:

            battery = (
                psutil.sensors_battery()
            )

            if battery is None:
                return {
                    "available": False,
                    "message": (
                        "Battery not detected."
                    ),
                }

            return {
                "available": True,
                "percent": battery.percent,
                "charging": battery.power_plugged,
                "seconds_left": (
                    battery.secsleft
                ),
            }

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # FALLBACK PROCESSES
    # ============================================================

    @staticmethod
    def _fallback_processes(
    ) -> dict[str, Any]:

        if not psutil:
            return {
                "available": False
            }

        try:

            processes = []

            for process in psutil.process_iter(
                [
                    "pid",
                    "name",
                    "username",
                    "cpu_percent",
                    "memory_percent",
                    "status",
                ]
            ):

                try:

                    data = process.info

                    processes.append(
                        data
                    )

                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                ):
                    continue

            processes.sort(
                key=lambda item: (
                    item.get(
                        "cpu_percent"
                    )
                    or 0
                ),
                reverse=True,
            )

            return {
                "available": True,
                "count": len(
                    processes
                ),
                "top_processes": (
                    processes[:20]
                ),
            }

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # FALLBACK TEMPERATURE
    # ============================================================

    @staticmethod
    def _fallback_temperatures(
    ) -> dict[str, Any]:

        if not psutil:
            return {
                "available": False
            }

        try:

            if not hasattr(
                psutil,
                "sensors_temperatures",
            ):
                return {
                    "available": False,
                    "message": (
                        "Temperature sensors "
                        "are unavailable."
                    ),
                }

            raw = (
                psutil.sensors_temperatures()
            )

            sensors = {}

            for name, entries in (
                raw.items()
            ):

                sensors[name] = [
                    {
                        "label": (
                            entry.label
                        ),
                        "current": (
                            entry.current
                        ),
                        "high": (
                            entry.high
                        ),
                        "critical": (
                            entry.critical
                        ),
                    }
                    for entry in entries
                ]

            return {
                "available": bool(
                    sensors
                ),
                "sensors": sensors,
            }

        except Exception as exc:

            return {
                "available": False,
                "error": str(exc),
            }

    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================

    @staticmethod
    def _number(
        value: Any,
    ) -> float | None:

        try:

            if value is None:
                return None

            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

    def _highest_temperature(
        self,
        temperatures: dict[str, Any],
    ) -> float | None:

        sensors = temperatures.get(
            "sensors",
            {},
        )

        highest = None

        if not isinstance(
            sensors,
            dict,
        ):
            return None

        for entries in sensors.values():

            if not isinstance(
                entries,
                list,
            ):
                continue

            for entry in entries:

                if not isinstance(
                    entry,
                    dict,
                ):
                    continue

                current = self._number(
                    entry.get(
                        "current"
                    )
                )

                if current is None:
                    continue

                if (
                    highest is None
                    or current > highest
                ):
                    highest = current

        return highest


__all__ = [
    "SystemSnapshot",
    "SystemMonitor",
]


