"""
RENIX Holographic UI
System UI
==================

System-control UI state layer for RENIX.

Responsibilities:
- System status display
- CPU / RAM / GPU / disk / network information
- Battery information
- Volume / brightness state
- System controls
- Process overview
- Performance alerts
- Quick-action controls
- Event-driven UI integration

This module is UI/state oriented.
Actual hardware/system operations should be delegated to:
    system_monitor/
    computer/system_settings.py
    computer/volume.py
    computer/brightness.py
    computer/display.py
    computer/process_manager.py
"""

from __future__ import annotations

import platform
import socket
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ============================================================================
# ENUMS
# ============================================================================


class SystemPanel(str, Enum):
    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    NETWORK = "network"
    STORAGE = "storage"
    BATTERY = "battery"
    PROCESSES = "processes"
    CONTROLS = "controls"


class SystemAction(str, Enum):
    REFRESH = "refresh"
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    SLEEP = "sleep"
    LOCK = "lock"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MUTE = "mute"
    BRIGHTNESS_UP = "brightness_up"
    BRIGHTNESS_DOWN = "brightness_down"
    TOGGLE_WIFI = "toggle_wifi"
    TOGGLE_BLUETOOTH = "toggle_bluetooth"
    OPEN_SETTINGS = "open_settings"
    KILL_PROCESS = "kill_process"


class SystemHealth(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class CPUStatus:
    usage_percent: float = 0.0
    temperature_c: Optional[float] = None
    frequency_mhz: Optional[float] = None
    cores: int = 0


@dataclass
class MemoryStatus:
    used_bytes: int = 0
    total_bytes: int = 0
    available_bytes: int = 0
    usage_percent: float = 0.0


@dataclass
class GPUStatus:
    usage_percent: float = 0.0
    memory_used_bytes: int = 0
    memory_total_bytes: int = 0
    temperature_c: Optional[float] = None


@dataclass
class DiskStatus:
    mount: str = ""
    used_bytes: int = 0
    total_bytes: int = 0
    free_bytes: int = 0
    usage_percent: float = 0.0


@dataclass
class NetworkStatus:
    connected: bool = False
    interface: str = ""
    ip_address: str = ""
    download_rate: float = 0.0
    upload_rate: float = 0.0
    latency_ms: Optional[float] = None


@dataclass
class BatteryStatus:
    available: bool = False
    percent: float = 0.0
    charging: bool = False
    remaining_seconds: Optional[int] = None


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_bytes: int = 0
    status: str = "unknown"


@dataclass
class SystemSnapshot:
    timestamp: float = field(
        default_factory=time.time
    )

    hostname: str = ""

    operating_system: str = ""

    os_version: str = ""

    architecture: str = ""

    cpu: CPUStatus = field(
        default_factory=CPUStatus
    )

    memory: MemoryStatus = field(
        default_factory=MemoryStatus
    )

    gpu: GPUStatus = field(
        default_factory=GPUStatus
    )

    disks: list[DiskStatus] = field(
        default_factory=list
    )

    network: NetworkStatus = field(
        default_factory=NetworkStatus
    )

    battery: BatteryStatus = field(
        default_factory=BatteryStatus
    )

    processes: list[ProcessInfo] = field(
        default_factory=list
    )

    health: SystemHealth = (
        SystemHealth.UNKNOWN
    )


@dataclass
class SystemActionResult:
    action: SystemAction

    success: bool

    message: str = ""

    data: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# SYSTEM UI
# ============================================================================


class SystemUI:

    def __init__(
        self,
        *,
        refresh_interval: float = 1.0,
        max_processes: int = 20,
    ) -> None:

        self.refresh_interval = max(
            0.1,
            float(refresh_interval),
        )

        self.max_processes = max(
            1,
            int(max_processes),
        )

        self._active_panel = (
            SystemPanel.OVERVIEW
        )

        self._snapshot = SystemSnapshot()

        self._running = False

        self._auto_refresh = True

        self._last_refresh = 0.0

        self._refresh_thread: Optional[
            threading.Thread
        ] = None

        self._stop_event = (
            threading.Event()
        )

        self._lock = threading.RLock()

        self._callbacks: dict[
            str,
            list[Callable[..., None]],
        ] = {}

        self._volume = 50.0

        self._muted = False

        self._brightness = 70.0

        self._wifi_enabled = True

        self._bluetooth_enabled = False

        self._alerts: list[str] = []

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self) -> None:

        with self._lock:

            if self._running:
                return

            self._running = True

            self._stop_event.clear()

        self.refresh()

        self._refresh_thread = (
            threading.Thread(
                target=self._refresh_loop,
                name="RENIX-SystemUI",
                daemon=True,
            )
        )

        self._refresh_thread.start()

        self._emit(
            "started",
            self,
        )

    def stop(self) -> None:

        with self._lock:

            if not self._running:
                return

            self._running = False

        self._stop_event.set()

        thread = self._refresh_thread

        if (
            thread is not None
            and thread.is_alive()
            and thread
            is not threading.current_thread()
        ):
            thread.join(
                timeout=2.0
            )

        self._refresh_thread = None

        self._emit(
            "stopped",
            self,
        )

    @property
    def running(self) -> bool:

        with self._lock:
            return self._running

    def _refresh_loop(self) -> None:

        while not self._stop_event.wait(
            self.refresh_interval
        ):

            if not self._running:
                break

            if self._auto_refresh:

                try:
                    self.refresh()

                except Exception as exc:

                    self._emit(
                        "error",
                        exc,
                    )

    # ========================================================================
    # PANELS
    # ========================================================================

    @property
    def active_panel(
        self,
    ) -> SystemPanel:

        with self._lock:
            return self._active_panel

    def set_panel(
        self,
        panel: SystemPanel,
    ) -> None:

        with self._lock:

            self._active_panel = panel

        self._emit(
            "panel_changed",
            panel,
        )

    def next_panel(self) -> SystemPanel:

        panels = list(
            SystemPanel
        )

        with self._lock:

            index = panels.index(
                self._active_panel
            )

            self._active_panel = panels[
                (index + 1) % len(panels)
            ]

            panel = self._active_panel

        self._emit(
            "panel_changed",
            panel,
        )

        return panel

    def previous_panel(
        self,
    ) -> SystemPanel:

        panels = list(
            SystemPanel
        )

        with self._lock:

            index = panels.index(
                self._active_panel
            )

            self._active_panel = panels[
                (index - 1) % len(panels)
            ]

            panel = self._active_panel

        self._emit(
            "panel_changed",
            panel,
        )

        return panel

    # ========================================================================
    # REFRESH
    # ========================================================================

    def refresh(self) -> SystemSnapshot:

        snapshot = SystemSnapshot(
            timestamp=time.time(),
            hostname=socket.gethostname(),
            operating_system=platform.system(),
            os_version=platform.version(),
            architecture=platform.machine(),
        )

        self._collect_cpu(snapshot)

        self._collect_memory(snapshot)

        self._collect_gpu(snapshot)

        self._collect_disks(snapshot)

        self._collect_network(snapshot)

        self._collect_battery(snapshot)

        self._collect_processes(snapshot)

        snapshot.health = (
            self._calculate_health(snapshot)
        )

        with self._lock:

            self._snapshot = snapshot

            self._last_refresh = (
                time.time()
            )

        self._update_alerts(
            snapshot
        )

        self._emit(
            "refreshed",
            snapshot,
        )

        return snapshot

    # ========================================================================
    # CPU
    # ========================================================================

    def _collect_cpu(
        self,
        snapshot: SystemSnapshot,
    ) -> None:

        try:

            import psutil

            snapshot.cpu.usage_percent = (
                float(
                    psutil.cpu_percent(
                        interval=None
                    )
                )
            )

            snapshot.cpu.cores = (
                psutil.cpu_count(
                    logical=True
                )
                or 0
            )

            frequency = (
                psutil.cpu_freq()
            )

            if frequency:

                snapshot.cpu.frequency_mhz = (
                    float(
                        frequency.current
                    )
                )

        except Exception:

            snapshot.cpu.usage_percent = 0.0

            snapshot.cpu.cores = 0

    # ========================================================================
    # MEMORY
    # ========================================================================

    def _collect_memory(
        self,
        snapshot: SystemSnapshot,
    ) -> None:

        try:

            import psutil

            memory = psutil.virtual_memory()

            snapshot.memory.used_bytes = (
                int(memory.used)
            )

            snapshot.memory.total_bytes = (
                int(memory.total)
            )

            snapshot.memory.available_bytes = (
                int(memory.available)
            )

            snapshot.memory.usage_percent = (
                float(memory.percent)
            )

        except Exception:

            pass

    # ========================================================================
    # GPU
    # ========================================================================

    def _collect_gpu(
        self,
        snapshot: SystemSnapshot,
    ) -> None:

        """
        GPU monitoring is intentionally optional.

        A future gpu_monitor.py can populate this
        object directly without changing the UI.
        """

        try:

            import GPUtil

            gpus = GPUtil.getGPUs()

            if not gpus:
                return

            gpu = gpus[0]

            snapshot.gpu.usage_percent = (
                float(gpu.load * 100.0)
            )

            snapshot.gpu.memory_used_bytes = (
                int(gpu.memoryUsed * 1024 * 1024)
            )

            snapshot.gpu.memory_total_bytes = (
                int(gpu.memoryTotal * 1024 * 1024)
            )

            snapshot.gpu.temperature_c = (
                float(gpu.temperature)
            )

        except Exception:

            pass

    # ========================================================================
    # DISK
    # ========================================================================

    def _collect_disks(
        self,
        snapshot: SystemSnapshot,
    ) -> None:

        try:

            import psutil

            partitions = (
                psutil.disk_partitions(
                    all=False
                )
            )

            for partition in partitions:

                try:

                    usage = (
                        psutil.disk_usage(
                            partition.mountpoint
                        )
                    )

                    snapshot.disks.append(
                        DiskStatus(
                            mount=partition.mountpoint,
                            used_bytes=int(
                                usage.used
                            ),
                            total_bytes=int(
                                usage.total
                            ),
                            free_bytes=int(
                                usage.free
                            ),
                            usage_percent=float(
                                usage.percent
                            ),
                        )
                    )

                except Exception:

                    continue

        except Exception:

            pass

    # ========================================================================
    # NETWORK
    # ========================================================================

    def _collect_network(
        self,
        snapshot: SystemSnapshot,
    ) -> None:

        try:

            import psutil

            interfaces = (
                psutil.net_if_addrs()
            )

            for name, addresses in (
                interfaces.items()
            ):

                for address in addresses:

                    if (
                        getattr(
                            address,
                            "family",
                            None,
                        )
                        == socket.AF_INET
                    ):

                        ip = address.address

                        if ip.startswith(
                            "127."
                        ):
                            continue

                        snapshot.network.connected = (
                            True
                        )

                        snapshot.network.interface = (
                            name
                        )

                        snapshot.network.ip_address = (
                            ip
                        )

                        return

        except Exception:

            pass

    # ========================================================================
    # BATTERY
    # ========================================================================

    def _collect_battery(
        self,
        snapshot: SystemSnapshot,
    ) -> None:

        try:

            import psutil

            battery = (
                psutil.sensors_battery()
            )

            if battery is None:
                return

            snapshot.battery.available = True

            snapshot.battery.percent = (
                float(battery.percent)
            )

            snapshot.battery.charging = (
                bool(battery.power_plugged)
            )

            if (
                battery.secsleft
                >= 0
            ):

                snapshot.battery.remaining_seconds = (
                    int(
                        battery.secsleft
                    )
                )

        except Exception:

            pass

    # ========================================================================
    # PROCESSES
    # ========================================================================

    def _collect_processes(
        self,
        snapshot: SystemSnapshot,
    ) -> None:

        try:

            import psutil

            processes: list[
                ProcessInfo
            ] = []

            for process in psutil.process_iter(
                [
                    "pid",
                    "name",
                    "cpu_percent",
                    "memory_info",
                    "status",
                ]
            ):

                try:

                    info = (
                        process.info
                    )

                    memory_info = (
                        info.get(
                            "memory_info"
                        )
                    )

                    memory = (
                        int(
                            memory_info.rss
                        )
                        if memory_info
                        else 0
                    )

                    processes.append(
                        ProcessInfo(
                            pid=int(
                                info.get(
                                    "pid",
                                    0,
                                )
                            ),
                            name=str(
                                info.get(
                                    "name",
                                    "unknown",
                                )
                            ),
                            cpu_percent=float(
                                info.get(
                                    "cpu_percent",
                                    0.0,
                                )
                            ),
                            memory_bytes=memory,
                            status=str(
                                info.get(
                                    "status",
                                    "unknown",
                                )
                            ),
                        )
                    )

                except Exception:

                    continue

            processes.sort(
                key=lambda process: (
                    process.cpu_percent
                ),
                reverse=True,
            )

            snapshot.processes = (
                processes[
                    : self.max_processes
                ]
            )

        except Exception:

            pass

    # ========================================================================
    # HEALTH
    # ========================================================================

    def _calculate_health(
        self,
        snapshot: SystemSnapshot,
    ) -> SystemHealth:

        critical = False

        warning = False

        if (
            snapshot.cpu.usage_percent
            >= 95.0
        ):

            critical = True

        elif (
            snapshot.cpu.usage_percent
            >= 85.0
        ):

            warning = True

        if (
            snapshot.memory.usage_percent
            >= 95.0
        ):

            critical = True

        elif (
            snapshot.memory.usage_percent
            >= 85.0
        ):

            warning = True

        for disk in snapshot.disks:

            if disk.usage_percent >= 98.0:

                critical = True

            elif disk.usage_percent >= 90.0:

                warning = True

        if snapshot.battery.available:

            if (
                snapshot.battery.percent
                <= 5.0
                and not snapshot.battery.charging
            ):

                critical = True

            elif (
                snapshot.battery.percent
                <= 15.0
                and not snapshot.battery.charging
            ):

                warning = True

        if critical:

            return SystemHealth.CRITICAL

        if warning:

            return SystemHealth.WARNING

        if snapshot.cpu.usage_percent < 50:

            return SystemHealth.EXCELLENT

        return SystemHealth.GOOD

    # ========================================================================
    # ALERTS
    # ========================================================================

    def _update_alerts(
        self,
        snapshot: SystemSnapshot,
    ) -> None:

        alerts: list[str] = []

        if snapshot.cpu.usage_percent >= 90:

            alerts.append(
                "CPU usage is very high."
            )

        if snapshot.memory.usage_percent >= 90:

            alerts.append(
                "Memory usage is very high."
            )

        for disk in snapshot.disks:

            if disk.usage_percent >= 90:

                alerts.append(
                    f"Storage almost full: "
                    f"{disk.mount}"
                )

        if (
            snapshot.battery.available
            and snapshot.battery.percent <= 15
            and not snapshot.battery.charging
        ):

            alerts.append(
                "Battery level is low."
            )

        with self._lock:

            self._alerts = alerts

        if alerts:

            self._emit(
                "alerts_changed",
                alerts,
            )

    @property
    def alerts(self) -> list[str]:

        with self._lock:
            return list(
                self._alerts
            )

    def clear_alerts(self) -> None:

        with self._lock:

            self._alerts.clear()

        self._emit(
            "alerts_changed",
            [],
        )

    # ========================================================================
    # VOLUME
    # ========================================================================

    @property
    def volume(self) -> float:

        with self._lock:
            return self._volume

    @property
    def muted(self) -> bool:

        with self._lock:
            return self._muted

    def set_volume(
        self,
        value: float,
    ) -> float:

        value = max(
            0.0,
            min(100.0, float(value)),
        )

        with self._lock:

            self._volume = value

            if value > 0:
                self._muted = False

        self._emit(
            "volume_changed",
            value,
            self._muted,
        )

        return value

    def volume_up(
        self,
        amount: float = 5.0,
    ) -> float:

        return self.set_volume(
            self.volume + amount
        )

    def volume_down(
        self,
        amount: float = 5.0,
    ) -> float:

        return self.set_volume(
            self.volume - amount
        )

    def toggle_mute(self) -> bool:

        with self._lock:

            self._muted = not self._muted

            muted = self._muted

        self._emit(
            "volume_changed",
            self._volume,
            muted,
        )

        return muted

    # ========================================================================
    # BRIGHTNESS
    # ========================================================================

    @property
    def brightness(self) -> float:

        with self._lock:
            return self._brightness

    def set_brightness(
        self,
        value: float,
    ) -> float:

        value = max(
            0.0,
            min(100.0, float(value)),
        )

        with self._lock:

            self._brightness = value

        self._emit(
            "brightness_changed",
            value,
        )

        return value

    def brightness_up(
        self,
        amount: float = 10.0,
    ) -> float:

        return self.set_brightness(
            self.brightness + amount
        )

    def brightness_down(
        self,
        amount: float = 10.0,
    ) -> float:

        return self.set_brightness(
            self.brightness - amount
        )

    # ========================================================================
    # CONNECTIVITY
    # ========================================================================

    @property
    def wifi_enabled(self) -> bool:

        with self._lock:
            return self._wifi_enabled

    def set_wifi_enabled(
        self,
        enabled: bool,
    ) -> bool:

        with self._lock:

            self._wifi_enabled = bool(
                enabled
            )

        self._emit(
            "wifi_changed",
            self._wifi_enabled,
        )

        return self._wifi_enabled

    def toggle_wifi(self) -> bool:

        return self.set_wifi_enabled(
            not self.wifi_enabled
        )

    @property
    def bluetooth_enabled(
        self,
    ) -> bool:

        with self._lock:
            return self._bluetooth_enabled

    def set_bluetooth_enabled(
        self,
        enabled: bool,
    ) -> bool:

        with self._lock:

            self._bluetooth_enabled = bool(
                enabled
            )

        self._emit(
            "bluetooth_changed",
            self._bluetooth_enabled,
        )

        return self._bluetooth_enabled

    def toggle_bluetooth(self) -> bool:

        return self.set_bluetooth_enabled(
            not self.bluetooth_enabled
        )

    # ========================================================================
    # ACTIONS
    # ========================================================================

    def execute_action(
        self,
        action: SystemAction,
        **kwargs: Any,
    ) -> SystemActionResult:

        if action == SystemAction.REFRESH:

            snapshot = self.refresh()

            return SystemActionResult(
                action=action,
                success=True,
                message="System refreshed.",
                data={
                    "health": (
                        snapshot.health.value
                    )
                },
            )

        if action == SystemAction.VOLUME_UP:

            value = self.volume_up(
                kwargs.get(
                    "amount",
                    5.0,
                )
            )

            return SystemActionResult(
                action=action,
                success=True,
                message=(
                    f"Volume set to "
                    f"{value:.0f}%."
                ),
                data={
                    "volume": value
                },
            )

        if action == SystemAction.VOLUME_DOWN:

            value = self.volume_down(
                kwargs.get(
                    "amount",
                    5.0,
                )
            )

            return SystemActionResult(
                action=action,
                success=True,
                message=(
                    f"Volume set to "
                    f"{value:.0f}%."
                ),
                data={
                    "volume": value
                },
            )

        if action == SystemAction.MUTE:

            muted = self.toggle_mute()

            return SystemActionResult(
                action=action,
                success=True,
                message=(
                    "Audio muted."
                    if muted
                    else "Audio unmuted."
                ),
                data={
                    "muted": muted
                },
            )

        if action == SystemAction.BRIGHTNESS_UP:

            value = self.brightness_up(
                kwargs.get(
                    "amount",
                    10.0,
                )
            )

            return SystemActionResult(
                action=action,
                success=True,
                message=(
                    f"Brightness set to "
                    f"{value:.0f}%."
                ),
                data={
                    "brightness": value
                },
            )

        if action == SystemAction.BRIGHTNESS_DOWN:

            value = self.brightness_down(
                kwargs.get(
                    "amount",
                    10.0,
                )
            )

            return SystemActionResult(
                action=action,
                success=True,
                message=(
                    f"Brightness set to "
                    f"{value:.0f}%."
                ),
                data={
                    "brightness": value
                },
            )

        if action == SystemAction.TOGGLE_WIFI:

            value = self.toggle_wifi()

            return SystemActionResult(
                action=action,
                success=True,
                message=(
                    "Wi-Fi enabled."
                    if value
                    else "Wi-Fi disabled."
                ),
                data={
                    "enabled": value
                },
            )

        if action == SystemAction.TOGGLE_BLUETOOTH:

            value = self.toggle_bluetooth()

            return SystemActionResult(
                action=action,
                success=True,
                message=(
                    "Bluetooth enabled."
                    if value
                    else "Bluetooth disabled."
                ),
                data={
                    "enabled": value
                },
            )

        if action == SystemAction.LOCK:

            self._emit(
                "lock_requested"
            )

            return SystemActionResult(
                action=action,
                success=True,
                message="System lock requested.",
            )

        if action == SystemAction.SLEEP:

            self._emit(
                "sleep_requested"
            )

            return SystemActionResult(
                action=action,
                success=True,
                message="System sleep requested.",
            )

        if action == SystemAction.RESTART:

            self._emit(
                "restart_requested"
            )

            return SystemActionResult(
                action=action,
                success=True,
                message="System restart requested.",
            )

        if action == SystemAction.SHUTDOWN:

            self._emit(
                "shutdown_requested"
            )

            return SystemActionResult(
                action=action,
                success=True,
                message="System shutdown requested.",
            )

        if action == SystemAction.OPEN_SETTINGS:

            self._emit(
                "settings_requested"
            )

            return SystemActionResult(
                action=action,
                success=True,
                message="System settings requested.",
            )

        if action == SystemAction.KILL_PROCESS:

            pid = kwargs.get(
                "pid"
            )

            if pid is None:

                return SystemActionResult(
                    action=action,
                    success=False,
                    message="Process PID required.",
                )

            self._emit(
                "kill_process_requested",
                int(pid),
            )

            return SystemActionResult(
                action=action,
                success=True,
                message=(
                    f"Kill request sent "
                    f"for PID {pid}."
                ),
                data={
                    "pid": int(pid)
                },
            )

        return SystemActionResult(
            action=action,
            success=False,
            message="Unsupported system action.",
        )

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    def get_snapshot(
        self,
    ) -> SystemSnapshot:

        with self._lock:

            return self._snapshot

    def get_snapshot_dict(
        self,
    ) -> dict[str, Any]:

        snapshot = self.get_snapshot()

        return {
            "timestamp": snapshot.timestamp,
            "hostname": snapshot.hostname,
            "operating_system": (
                snapshot.operating_system
            ),
            "os_version": (
                snapshot.os_version
            ),
            "architecture": (
                snapshot.architecture
            ),
            "health": snapshot.health.value,
            "cpu": {
                "usage_percent": (
                    snapshot.cpu.usage_percent
                ),
                "temperature_c": (
                    snapshot.cpu.temperature_c
                ),
                "frequency_mhz": (
                    snapshot.cpu.frequency_mhz
                ),
                "cores": snapshot.cpu.cores,
            },
            "memory": {
                "used_bytes": (
                    snapshot.memory.used_bytes
                ),
                "total_bytes": (
                    snapshot.memory.total_bytes
                ),
                "available_bytes": (
                    snapshot.memory.available_bytes
                ),
                "usage_percent": (
                    snapshot.memory.usage_percent
                ),
            },
            "gpu": {
                "usage_percent": (
                    snapshot.gpu.usage_percent
                ),
                "memory_used_bytes": (
                    snapshot.gpu.memory_used_bytes
                ),
                "memory_total_bytes": (
                    snapshot.gpu.memory_total_bytes
                ),
                "temperature_c": (
                    snapshot.gpu.temperature_c
                ),
            },
            "disks": [
                {
                    "mount": disk.mount,
                    "used_bytes": (
                        disk.used_bytes
                    ),
                    "total_bytes": (
                        disk.total_bytes
                    ),
                    "free_bytes": (
                        disk.free_bytes
                    ),
                    "usage_percent": (
                        disk.usage_percent
                    ),
                }
                for disk in snapshot.disks
            ],
            "network": {
                "connected": (
                    snapshot.network.connected
                ),
                "interface": (
                    snapshot.network.interface
                ),
                "ip_address": (
                    snapshot.network.ip_address
                ),
                "download_rate": (
                    snapshot.network.download_rate
                ),
                "upload_rate": (
                    snapshot.network.upload_rate
                ),
                "latency_ms": (
                    snapshot.network.latency_ms
                ),
            },
            "battery": {
                "available": (
                    snapshot.battery.available
                ),
                "percent": (
                    snapshot.battery.percent
                ),
                "charging": (
                    snapshot.battery.charging
                ),
                "remaining_seconds": (
                    snapshot.battery.remaining_seconds
                ),
            },
            "processes": [
                {
                    "pid": process.pid,
                    "name": process.name,
                    "cpu_percent": (
                        process.cpu_percent
                    ),
                    "memory_bytes": (
                        process.memory_bytes
                    ),
                    "status": process.status,
                }
                for process
                in snapshot.processes
            ],
            "controls": {
                "volume": self.volume,
                "muted": self.muted,
                "brightness": self.brightness,
                "wifi_enabled": (
                    self.wifi_enabled
                ),
                "bluetooth_enabled": (
                    self.bluetooth_enabled
                ),
            },
            "alerts": self.alerts,
        }

    # ========================================================================
    # EVENT SYSTEM
    # ========================================================================

    def on(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        if not event:
            raise ValueError(
                "Event name cannot be empty."
            )

        if not callable(callback):
            raise TypeError(
                "Callback must be callable."
            )

        with self._lock:

            self._callbacks.setdefault(
                event,
                [],
            ).append(callback)

    def off(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:

        with self._lock:

            callbacks = self._callbacks.get(
                event,
                [],
            )

            if callback in callbacks:

                callbacks.remove(
                    callback
                )

    def _emit(
        self,
        event: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        with self._lock:

            callbacks = list(
                self._callbacks.get(
                    event,
                    [],
                )
            )

        for callback in callbacks:

            try:

                callback(
                    *args,
                    **kwargs,
                )

            except Exception:

                # UI callbacks should never
                # crash the RENIX runtime.
                pass

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(self) -> dict[str, Any]:

        snapshot = self.get_snapshot()

        return {
            "running": self.running,
            "active_panel": (
                self.active_panel.value
            ),
            "health": (
                snapshot.health.value
            ),
            "last_refresh": (
                self._last_refresh
            ),
            "auto_refresh": (
                self._auto_refresh
            ),
            "volume": self.volume,
            "muted": self.muted,
            "brightness": self.brightness,
            "wifi_enabled": (
                self.wifi_enabled
            ),
            "bluetooth_enabled": (
                self.bluetooth_enabled
            ),
            "alerts": len(
                self.alerts
            ),
        }

    # ========================================================================
    # AUTO REFRESH
    # ========================================================================

    def set_auto_refresh(
        self,
        enabled: bool,
    ) -> None:

        with self._lock:

            self._auto_refresh = bool(
                enabled
            )

        self._emit(
            "auto_refresh_changed",
            self._auto_refresh,
        )

    @property
    def auto_refresh(self) -> bool:

        with self._lock:
            return self._auto_refresh


# ============================================================================
# FACTORY
# ============================================================================


def create_system_ui() -> SystemUI:

    ui = SystemUI(
        refresh_interval=1.0,
        max_processes=20,
    )

    ui.start()

    return ui


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "SystemPanel",
    "SystemAction",
    "SystemHealth",
    "CPUStatus",
    "MemoryStatus",
    "GPUStatus",
    "DiskStatus",
    "NetworkStatus",
    "BatteryStatus",
    "ProcessInfo",
    "SystemSnapshot",
    "SystemActionResult",
    "SystemUI",
    "create_system_ui",
]


